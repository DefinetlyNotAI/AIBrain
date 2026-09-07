"""Standalone inspector for AIBrain's persisted connectome analysis model."""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..connectome.analysis import (
    ADULT_FRAME_FLOOR,
    BABY_FRAME_FLOOR,
    CONSISTENCY_WINDOW,
    FEATURE_SCHEMA,
    ConnectomeAnalyzer,
)
from ..utils.array_api import BACKEND_NAME
from .dashboard import metric_card, metric_grid, progress_card
from .theme import load_colours, stylesheet

LOG = logging.getLogger(__name__)
_TENSOR_NAMES = (
    "encoder_weights",
    "encoder_bias",
    "decoder_weights",
    "decoder_bias",
    "embedding_centroid",
)
_REQUIRED_NAMES = (*_TENSOR_NAMES, "frames_seen", "feature_schema")


class TensorSummary(TypedDict):
    shape: list[int]
    dtype: str
    values: int
    bytes: int
    finite_values: int
    poisoned_values: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    standard_deviation: float | None


class HealthFactor(TypedDict):
    score: int
    detail: str


class HealthSummary(TypedDict):
    score_percent: int
    factors: dict[str, HealthFactor]


class FileSummary(TypedDict):
    size_bytes: int
    modified_at_utc: str
    age: str


class MaturitySummary(TypedDict):
    state: str
    next_state: str | None
    progress_percent: int
    progress_detail: str
    metrics_persisted: bool
    consistency_sustained: bool
    weights_frozen: bool
    readiness: str
    note: str


class LearningSummary(TypedDict):
    lifetime_frames_seen: int
    maturity: MaturitySummary
    quality_note: str


class ArchitectureSummary(TypedDict):
    type: str
    feature_schema: str
    input_features: int
    latent_features: int
    shape: str
    learned_parameters: int
    numerical_backend: str


class AnalysisMetadata(TypedDict):
    status: str
    health: HealthSummary
    health_checks: dict[str, str]
    path: str
    file: FileSummary
    learning: LearningSummary
    architecture: ArchitectureSummary
    tensors: dict[str, TensorSummary]


@dataclass(frozen=True, slots=True)
class AnalysisInspectionResult:
    metadata: AnalysisMetadata
    npz_contents: str


def _age_text(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return "just now"
    for name, size in (("day", 86_400), ("hour", 3_600), ("minute", 60)):
        amount = seconds // size
        if amount:
            suffix = "" if amount == 1 else "s"
            return f"{amount} {name}{suffix} ago"
    return "just now"


def _tensor_summary(value: np.ndarray) -> TensorSummary:
    finite = value[np.isfinite(value)]
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "values": int(value.size),
        "bytes": int(value.nbytes),
        "finite_values": int(finite.size),
        "poisoned_values": int(value.size - finite.size),
        "minimum": float(finite.min()) if finite.size else None,
        "maximum": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "standard_deviation": float(finite.std()) if finite.size else None,
    }


def inspect_npz_model(path: Path) -> AnalysisMetadata:
    """Validate the persisted model and expose its health, age, and learned data."""
    path = Path(path)
    stat = path.stat()
    with np.load(path, allow_pickle=False) as stored:
        missing = [name for name in _REQUIRED_NAMES if name not in stored]
        if missing:
            if "feature_schema" in missing:
                raise ValueError(
                    "The analysis NPZ has no real-time feature schema and may contain "
                    "legacy simulated features. Generate a new response to create "
                    "real-time telemetry analysis memory."
                )
            raise ValueError(
                f"The analysis NPZ is missing required tensors: {', '.join(missing)}"
            )
        tensors = {name: np.asarray(stored[name]) for name in _TENSOR_NAMES}
        raw_frames = np.asarray(stored["frames_seen"])
        feature_schema = str(np.asarray(stored["feature_schema"]).reshape(-1)[0])
        histories = {
            name: (
                np.asarray(stored[name])
                if name in stored
                else np.asarray([], dtype="f4")
            )
            for name in (
                "reconstruction_history",
                "novelty_history",
                "update_magnitude_history",
            )
        }
        raw_state = (
            np.asarray(stored["maturity_state"])
            if "maturity_state" in stored
            else np.asarray("Baby")
        )
        frozen = (
            bool(np.asarray(stored["weights_frozen"]).reshape(-1)[0])
            if "weights_frozen" in stored
            else False
        )

    if raw_frames.size != 1:
        raise ValueError("The analysis NPZ has an invalid frames_seen value")
    frames_seen = int(raw_frames.reshape(-1)[0])
    if feature_schema != FEATURE_SCHEMA:
        raise ValueError(
            f"Unsupported analysis feature schema: {feature_schema or 'missing'}"
        )
    if frames_seen < 0:
        raise ValueError("The analysis NPZ has a negative frames_seen value")
    non_finite = [
        name for name, value in tensors.items() if not np.isfinite(value).all()
    ]
    encoder = tensors["encoder_weights"]
    decoder = tensors["decoder_weights"]
    if encoder.ndim != 2 or decoder.ndim != 2:
        raise ValueError("The encoder and decoder weights must both be two-dimensional")
    input_width, latent_width = encoder.shape
    expected_shapes = {
        "encoder_bias": (latent_width,),
        "decoder_weights": (latent_width, input_width),
        "decoder_bias": (input_width,),
        "embedding_centroid": (latent_width,),
    }
    shape_errors = [
        name
        for name, expected in expected_shapes.items()
        if tensors[name].shape != expected
    ]
    if shape_errors:
        raise ValueError(
            f"The analysis NPZ has incompatible tensor shapes: {', '.join(shape_errors)}"
        )

    modified = datetime.fromtimestamp(stat.st_mtime, UTC)
    state = str(raw_state.reshape(-1)[0])
    if state not in {"Baby", "Teen", "Adult", "Elder"}:
        state = "Baby"
    history_present = all(history.size for history in histories.values())
    history_finite = all(np.isfinite(history).all() for history in histories.values())
    consistency = False
    if (
        history_present
        and history_finite
        and histories["reconstruction_history"].size >= CONSISTENCY_WINDOW
    ):
        recent = histories["reconstruction_history"][-CONSISTENCY_WINDOW:]
        consistency = bool(
            float(recent.std()) <= 0.015 and float(recent.mean()) <= 0.08
        )
    if state == "Teen":
        stage_frames = max(0, frames_seen - BABY_FRAME_FLOOR)
        required = ADULT_FRAME_FLOOR - BABY_FRAME_FLOOR
    else:
        stage_frames = frames_seen
        required = BABY_FRAME_FLOOR if state == "Baby" else ADULT_FRAME_FLOOR
    next_state = {"Baby": "Teen", "Teen": "Adult", "Adult": "Elder", "Elder": None}[
        state
    ]
    maturity: MaturitySummary = {
        "state": state,
        "next_state": next_state,
        "progress_percent": (
            100 if state == "Elder" else min(99, round(stage_frames / required * 100))
        ),
        "progress_detail": (
            "Elder is the terminal safeguarded state."
            if next_state is None
            else f"{stage_frames:,} of {required:,} stage frames; "
            f"{CONSISTENCY_WINDOW} stable reconstruction/update samples are also required."
        ),
        "metrics_persisted": history_present,
        "consistency_sustained": consistency,
        "weights_frozen": frozen,
        "readiness": (
            "ready" if state in {"Adult", "Elder"} and history_present else "caution"
        ),
        "note": (
            "This compatible early real-time file has no rolling histories and is treated as Baby until new evidence is observed."
            if not history_present
            else "Maturity is a persisted learning-health signal, not an accuracy guarantee."
        ),
    }
    finite_tensors = not non_finite
    bounded_tensors = finite_tensors and all(
        not value.size or float(np.abs(value).max()) < 1_000_000
        for value in tensors.values()
    )
    factor_scores = {
        "required_tensors": (15, "All required tensors are present."),
        "valid_frame_counter": (
            10,
            "The lifetime frame counter is a non-negative scalar.",
        ),
        "architecture_compatibility": (
            20,
            "Encoder, decoder, bias, and centroid shapes agree.",
        ),
        "finite_tensor_values": (
            25 if finite_tensors else 0,
            (
                "No NaN or infinity values found."
                if finite_tensors
                else f"Poisoned NaN/infinity values found in: {', '.join(non_finite)}."
            ),
        ),
        "finite_learning_history": (
            (
                15
                if history_present and history_finite
                else 5
                if not history_present
                else 0
            ),
            (
                "Rolling learning histories are finite."
                if history_present and history_finite
                else (
                    "Legacy file has no rolling histories."
                    if not history_present
                    else "Learning history contains NaN/infinity, commonly caused by divide-by-zero contamination."
                )
            ),
        ),
        "bounded_parameter_magnitude": (
            10 if bounded_tensors else 0,
            (
                "Parameter magnitudes are within the corruption guardrail."
                if bounded_tensors
                else "Parameter magnitude is non-finite or implausibly large."
            ),
        ),
        "persistence_format": (5, "Compressed NPZ opened without pickle data."),
    }
    health_score = sum(score for score, _detail in factor_scores.values())
    status = (
        "Healthy"
        if health_score >= 90
        else "Degraded"
        if health_score >= 70
        else "Critical"
    )
    parameter_count = sum(
        int(tensors[name].size)
        for name in (
            "encoder_weights",
            "encoder_bias",
            "decoder_weights",
            "decoder_bias",
        )
    )
    return {
        "status": status,
        "health": {
            "score_percent": health_score,
            "factors": {
                name: {"score": score, "detail": detail_text}
                for name, (score, detail_text) in factor_scores.items()
            },
        },
        "health_checks": {
            name: detail_text for name, (_score, detail_text) in factor_scores.items()
        },
        "path": str(path),
        "file": {
            "size_bytes": stat.st_size,
            "modified_at_utc": modified.isoformat(),
            "age": _age_text(datetime.now(UTC).timestamp() - stat.st_mtime),
        },
        "learning": {
            "lifetime_frames_seen": frames_seen,
            "maturity": maturity,
            "quality_note": maturity["note"],
        },
        "architecture": {
            "type": "online autoencoder",
            "feature_schema": feature_schema,
            "input_features": input_width,
            "latent_features": latent_width,
            "shape": f"{input_width} -> {latent_width} -> {input_width}",
            "learned_parameters": parameter_count,
            "numerical_backend": BACKEND_NAME,
        },
        "tensors": {name: _tensor_summary(value) for name, value in tensors.items()},
    }


def inspect_npz_contents(path: Path) -> str:
    """Render every persisted NPZ entry and value for read-only inspection."""
    path = Path(path)
    sections = [f"NPZ archive: {path}"]
    with np.load(path, allow_pickle=False) as stored:
        names = tuple(stored.files)
        entries = [(name, np.array(stored[name], copy=True)) for name in names]
    sections.append(f"Entries: {len(entries)}")
    for name, value in entries:
        rendered = np.array2string(
            value,
            separator=", ",
            threshold=max(1, value.size),
            max_line_width=132,
            precision=9,
            floatmode="unique",
        )
        sections.extend(
            (
                "",
                f"[{name}]",
                f"dtype: {value.dtype}",
                f"shape: {value.shape}",
                f"values ({value.size}):",
                rendered,
            )
        )
    return "\n".join(sections)


class AnalysisInspectionWorker(QObject):
    """Read the NPZ away from the Qt event loop."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    @Slot()
    def run(self) -> None:
        try:
            self.completed.emit(
                AnalysisInspectionResult(
                    metadata=inspect_npz_model(self._path),
                    npz_contents=inspect_npz_contents(self._path),
                )
            )
        except (OSError, ValueError) as exc:
            LOG.warning("Analysis model inspection failed for %s: %s", self._path, exc)
            self.failed.emit(str(exc))
        except Exception:
            LOG.exception(
                "Unexpected failure while inspecting Analysis+ model at %s", self._path
            )
            self.failed.emit(
                "Unexpected inspection failure. See the Analysis runtime log."
            )


class AnalysisWindow(QMainWindow):
    """Read-only NPZ and exported JSON inspection surface."""

    inspection_finished = Signal(bool)

    def __init__(self, *, auto_refresh: bool = True) -> None:
        super().__init__()
        self.setWindowTitle("AIBrain Analysis Inspector")
        self.resize(1120, 760)
        self.setStyleSheet(stylesheet(load_colours()))
        self._inspection_thread: QThread | None = None
        self._inspection_worker: AnalysisInspectionWorker | None = None
        self._closing = False
        self._last_healthy = False
        self._selected_export: Path | None = None
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("Analysis+ learned model inspector")
        title.setObjectName("title")
        layout.addWidget(title)
        explanation = QLabel(
            "This tool reads only the persisted Analysis+ autoencoder and selected JSON exports. "
            "It never loads a GGUF model or creates the main visualizer."
        )
        explanation.setObjectName("muted")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        actions = QHBoxLayout()
        refresh = QPushButton("Refresh model health")
        refresh.setToolTip(
            "Re-read the persisted Analysis+ model and its rolling metrics"
        )
        refresh.clicked.connect(self.refresh)
        open_export = QPushButton("Inspect exported analysis JSON")
        open_export.setToolTip(
            "Open an explicit JSON or JSON.GZ export for read-only inspection"
        )
        open_export.clicked.connect(self.inspect_export)
        actions.addWidget(refresh)
        actions.addWidget(open_export)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.tabs = QTabWidget()
        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        self.health_row = QWidget()
        health_grid = metric_grid(self.health_row)
        health, self.health_value, self.health_detail = metric_card(
            "MODEL HEALTH", "Reading…"
        )
        self.state_card, self.state_value, self.state_detail = metric_card(
            "MATURITY STATE", "—"
        )
        self.frame_card, self.frame_value, self.frame_detail = metric_card(
            "PERSISTENCE", "—"
        )
        health_grid.addWidget(health, 0, 0)
        health_grid.addWidget(self.state_card, 0, 1)
        health_grid.addWidget(self.frame_card, 0, 2)
        overview_layout.addWidget(self.health_row)
        progress, self.maturity_progress, self.maturity_detail = progress_card(
            "PROGRESS TOWARD NEXT STATE"
        )
        overview_layout.addWidget(progress)
        self.architecture_card, self.architecture_value, self.architecture_detail = (
            metric_card("ARCHITECTURE", "—")
        )
        self.quality_card, self.quality_value, self.quality_detail = metric_card(
            "LEARNING QUALITY", "—"
        )
        self.export_card, self.export_value, self.export_detail = metric_card(
            "EXPORT READINESS", "—"
        )
        self.backend_card, self.backend_value, self.backend_detail = metric_card(
            "ARRAY BACKEND", "—"
        )
        self.evidence_card, self.evidence_value, self.evidence_detail = metric_card(
            "ROLLING EVIDENCE", "—"
        )
        self.storage_card, self.storage_value, self.storage_detail = metric_card(
            "MODEL STORAGE", "—"
        )
        card_tooltips = {
            health: "Overall integrity score for the persisted Analysis+ model",
            self.state_card: "Current learning stage and whether training is active",
            self.frame_card: "Total real-time inference frames retained across sessions",
            progress: "Evidence and frame progress required for the next maturity stage",
            self.architecture_card: "Autoencoder input, latent, and output dimensions",
            self.quality_card: "Stability of recent reconstruction and weight updates",
            self.export_card: "Whether learned findings are ready for normal reporting",
            self.backend_card: "Numerical array backend selected for connectome analysis",
            self.evidence_card: "Rolling samples available for maturity decisions",
            self.storage_card: "Persisted model size and most recent update time",
        }
        for card, tooltip in card_tooltips.items():
            card.setToolTip(tooltip)
        lower = QWidget()
        lower_grid = metric_grid(lower)
        lower_grid.addWidget(self.architecture_card, 0, 0)
        lower_grid.addWidget(self.quality_card, 0, 1)
        lower_grid.addWidget(self.export_card, 0, 2)
        lower_grid.addWidget(self.backend_card, 1, 0)
        lower_grid.addWidget(self.evidence_card, 1, 1)
        lower_grid.addWidget(self.storage_card, 1, 2)
        overview_layout.addWidget(lower)
        overview_layout.addStretch(1)
        self.raw = QPlainTextEdit()
        self.raw.setReadOnly(True)
        self.raw.setPlaceholderText(
            "Select an exported Analysis JSON file to inspect it."
        )
        self.raw.setToolTip(
            "Complete contents of an explicitly selected JSON or JSON.GZ analysis export"
        )
        self.npz = QPlainTextEdit()
        self.npz.setReadOnly(True)
        self.npz.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.npz.setPlaceholderText(
            "Persisted NPZ values appear after model inspection."
        )
        self.npz.setToolTip(
            "Read-only names, shapes, data types, and complete values stored in the Analysis+ NPZ"
        )
        self.tabs.addTab(overview, "Dashboard")
        self.tabs.addTab(self.npz, "NPZ contents")
        self.tabs.addTab(self.raw, "JSON export")
        self.tabs.setTabToolTip(0, "Model-health dashboard")
        self.tabs.setTabToolTip(
            1, "Complete values persisted in the Analysis+ NPZ archive"
        )
        self.tabs.setTabToolTip(
            2, "Complete contents of a selected JSON analysis export"
        )
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(page)
        if auto_refresh:
            QTimer.singleShot(0, self.refresh)

    def refresh(self) -> None:
        if self._closing or self._inspection_thread is not None:
            LOG.debug(
                "Ignoring Analysis+ refresh request (closing=%s, inspection_active=%s)",
                self._closing,
                self._inspection_thread is not None,
            )
            return
        path = ConnectomeAnalyzer.default_model_path()
        if not path.is_file():
            LOG.info("No persisted Analysis+ model found at %s", path)
            self._show_empty(path)
            QTimer.singleShot(0, lambda: self.inspection_finished.emit(False))
            return
        LOG.info("Starting Analysis+ model inspection for %s", path)
        if self._selected_export is None:
            self.raw.setPlainText(
                "Reading persisted Analysis+ model in the background…"
            )
        self.npz.setPlainText("Reading persisted NPZ contents in the background…")
        thread = QThread(self)
        worker = AnalysisInspectionWorker(path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._inspection_ready)
        worker.failed.connect(self._inspection_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._inspection_finished)
        self._inspection_thread = thread
        self._inspection_worker = worker
        thread.start()

    @Slot(object)
    def _inspection_ready(self, result: object) -> None:
        if isinstance(result, AnalysisInspectionResult):
            metadata = result.metadata
            health = metadata.get("health", {})
            learning = metadata.get("learning", {})
            score = (
                health.get("score_percent", "unknown")
                if isinstance(health, dict)
                else "unknown"
            )
            frames = (
                learning.get("lifetime_frames_seen", "unknown")
                if isinstance(learning, dict)
                else "unknown"
            )
            LOG.info(
                "Analysis+ model inspection completed (status=%s, health=%s%%, frames=%s)",
                metadata.get("status", "Unknown"),
                score,
                frames,
            )
            self._show_metadata(metadata)
            self.npz.setPlainText(result.npz_contents)
        else:
            LOG.error(
                "Analysis+ model inspection returned an unexpected result: %r", result
            )
            return
        if self._selected_export is None:
            self.raw.setPlainText(json.dumps(metadata, indent=2))

    @Slot(str)
    def _inspection_failed(self, message: str) -> None:
        LOG.warning("Analysis+ model is unavailable or invalid: %s", message)
        self._last_healthy = False
        self.health_value.setText("Needs repair")
        self.health_detail.setText(message)
        if self._selected_export is None:
            self.raw.setPlainText(f"Status: Invalid Analysis+ model\n\n{message}")
        self.npz.setPlainText(f"Status: NPZ contents unavailable\n\n{message}")

    @Slot()
    def _inspection_finished(self) -> None:
        self._inspection_thread = None
        self._inspection_worker = None
        if self._closing:
            LOG.info(
                "Analysis inspector closed while background inspection was stopping"
            )
            QTimer.singleShot(0, self.close)
            return
        LOG.info("Analysis+ inspection cycle finished (healthy=%s)", self._last_healthy)
        self.inspection_finished.emit(self._last_healthy)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._inspection_thread is not None:
            LOG.info(
                "Deferring Analysis inspector close until background inspection finishes"
            )
            self._closing = True
            self._inspection_thread.quit()
            self.hide()
            event.ignore()
            return
        super().closeEvent(event)

    def inspect_export(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Inspect analysis export", "", "Analysis data (*.json *.json.gz)"
        )
        if not filename:
            LOG.info("Analysis export inspection cancelled")
            return
        try:
            selected = Path(filename)
            LOG.info("Reading selected Analysis export: %s", selected)
            if selected.suffix == ".gz":
                with gzip.open(selected, "rt", encoding="utf-8") as handle:
                    payload = json.load(handle)
            else:
                payload = json.loads(selected.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("Analysis export root must be a JSON object")
            conversation = payload.get("conversation", [])
            summary = {
                "path": str(selected),
                "schema": payload.get("schema"),
                "created_at": payload.get("created_at"),
                "conversation_turns": (
                    len(conversation) if isinstance(conversation, list) else 0
                ),
                "has_nn_findings": isinstance(payload.get("smart_analysis"), dict)
                or "analysis_plus" in payload
                or "neural_network" in payload,
                "recorded_frame_summary": payload.get("recorded_frame_summary", {}),
            }
            LOG.info(
                "Analysis export inspected (schema=%s, conversation_turns=%s, nn_findings=%s)",
                summary["schema"],
                summary["conversation_turns"],
                summary["has_nn_findings"],
            )
            rendered = (
                "Export summary\n"
                "==============\n"
                f"{json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)}\n\n"
                "Complete JSON contents\n"
                "======================\n"
                f"{json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)}"
            )
            self._selected_export = selected
            self.raw.setPlainText(rendered)
            self.raw.verticalScrollBar().setValue(
                self.raw.verticalScrollBar().minimum()
            )
            self.tabs.setCurrentWidget(self.raw)
            self.statusBar().showMessage(f"Loaded analysis export: {selected.name}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            LOG.warning("Could not read selected Analysis export: %s", exc)
            self._selected_export = Path(filename)
            self.raw.setPlainText(
                f"Could not read analysis export\n"
                f"==============================\n"
                f"Path: {filename}\n\n{exc}"
            )
            self.tabs.setCurrentWidget(self.raw)
            self.statusBar().showMessage("Analysis export could not be loaded")

    def _show_empty(self, path: Path) -> None:
        self._last_healthy = False
        self.health_value.setText("No model yet")
        self.health_detail.setText(
            "Generate an Infinite-mode response to begin learning."
        )
        self.state_value.setText("BABY")
        self.state_detail.setText("No persisted metric history")
        self.frame_value.setText("0 frames")
        self.frame_detail.setText(str(path))
        self.maturity_progress.setValue(0)
        self.maturity_detail.setText(
            f"0 of {BABY_FRAME_FLOOR:,} Baby-stage frames; {CONSISTENCY_WINDOW} stable samples are also required."
        )
        self.architecture_value.setText("Not created")
        self.quality_value.setText("Awaiting frames")
        self.export_value.setText("CAUTION")
        self.export_detail.setText("Baby/Teen exports include a readiness caution.")
        self.backend_value.setText(BACKEND_NAME)
        self.backend_detail.setText(
            "Selected through the shared CUDA/CPU array wrapper."
        )
        self.evidence_value.setText("0 samples")
        self.evidence_detail.setText(
            "Reconstruction, novelty, and update histories are empty."
        )
        self.storage_value.setText("Not created")
        self.storage_detail.setText(str(path))
        if self._selected_export is None:
            self.raw.setPlainText(
                f"Status: No Analysis+ model has been learned yet.\n\nExpected location:\n{path}"
            )
        self.npz.setPlainText(
            f"Status: No NPZ contents are available yet.\n\nExpected location:\n{path}"
        )

    def _show_metadata(self, metadata: AnalysisMetadata) -> None:
        learning = metadata.get("learning", {})
        architecture = metadata.get("architecture", {})
        file_data = metadata.get("file", {})
        health = metadata.get("health", {})
        maturity = learning.get("maturity", {}) if isinstance(learning, dict) else {}
        if not isinstance(maturity, dict):
            maturity = {}
        score = int(health.get("score_percent", 0)) if isinstance(health, dict) else 0
        status = str(metadata.get("status", "Unknown"))
        self._last_healthy = status == "Healthy"
        self.health_value.setText(f"{score}%")
        self.health_detail.setText(
            f"{status}: required tensors, shapes, persistence, parameter magnitude, and "
            "NaN/infinity/divide-by-zero contamination are scored."
        )
        state = str(maturity.get("state", "Baby"))
        self.state_value.setText(state.upper())
        self.state_detail.setText(
            "Weights frozen"
            if maturity.get("weights_frozen")
            else "Training weights remain active"
        )
        frames = (
            int(learning.get("lifetime_frames_seen", 0))
            if isinstance(learning, dict)
            else 0
        )
        self.frame_value.setText(f"{frames:,} frames")
        self.frame_detail.setText(
            str(file_data.get("age", "Unknown persistence age"))
            if isinstance(file_data, dict)
            else ""
        )
        progress = int(maturity.get("progress_percent", 0))
        self.maturity_progress.setValue(progress)
        next_state = maturity.get("next_state") or "terminal Elder state"
        self.maturity_detail.setText(
            f"Toward {next_state}: {maturity.get('progress_detail', maturity.get('note', ''))}"
        )
        self.architecture_value.setText(
            str(architecture.get("shape", "Unknown"))
            if isinstance(architecture, dict)
            else "Unknown"
        )
        self.architecture_detail.setText(
            f"{architecture.get('learned_parameters', 0):,} learned parameters"
            if isinstance(architecture, dict)
            else ""
        )
        self.quality_value.setText(
            "Consistent" if maturity.get("consistency_sustained") else "Learning"
        )
        self.quality_detail.setText(
            "Rolling reconstruction and update metrics are persisted."
        )
        readiness = str(maturity.get("readiness", "caution")).upper()
        self.export_value.setText(readiness)
        self.export_detail.setText(
            "Normal smart-analysis report"
            if readiness == "READY"
            else "Export remains valid with explicit readiness caution."
        )
        self.backend_value.setText(
            str(architecture.get("numerical_backend", BACKEND_NAME))
            if isinstance(architecture, dict)
            else BACKEND_NAME
        )
        self.backend_detail.setText(
            "CuPy is preferred when CUDA executes successfully; NumPy is the fallback."
        )
        evidence_count = (
            CONSISTENCY_WINDOW
            if maturity.get("consistency_sustained")
            else min(frames, CONSISTENCY_WINDOW)
        )
        self.evidence_value.setText(f"{evidence_count:,} / {CONSISTENCY_WINDOW:,}")
        self.evidence_detail.setText(
            "Stable rolling reconstruction and update samples required for promotion."
        )
        size_bytes = (
            int(file_data.get("size_bytes", 0)) if isinstance(file_data, dict) else 0
        )
        self.storage_value.setText(f"{size_bytes / 1024:.1f} KiB")
        self.storage_detail.setText(
            str(file_data.get("modified_at_utc", "Unknown update time"))
            if isinstance(file_data, dict)
            else ""
        )
