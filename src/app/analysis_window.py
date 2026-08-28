"""Standalone inspector for AIBrain's persisted connectome analysis model."""
from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit, QPushButton, QTabWidget, \
    QVBoxLayout, QWidget

from ..connectome.analysis import ConnectomeAnalyzer
from .dashboard import metric_card, metric_grid, progress_card, status_badge

_TENSOR_NAMES = ("encoder_weights", "encoder_bias", "decoder_weights", "decoder_bias", "embedding_centroid")
_REQUIRED_NAMES = (*_TENSOR_NAMES, "frames_seen")


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


def _tensor_summary(value: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "values": int(value.size),
        "bytes": int(value.nbytes),
        "minimum": float(value.min()) if value.size else None,
        "maximum": float(value.max()) if value.size else None,
        "mean": float(value.mean()) if value.size else None,
        "standard_deviation": float(value.std()) if value.size else None,
    }


def inspect_npz_model(path: Path) -> dict[str, object]:
    """Validate the persisted model and expose its health, age, and learned data."""
    path = Path(path)
    stat = path.stat()
    with np.load(path, allow_pickle=False) as stored:
        missing = [name for name in _REQUIRED_NAMES if name not in stored]
        if missing:
            raise ValueError(f"The analysis NPZ is missing required tensors: {', '.join(missing)}")
        tensors = {name: np.asarray(stored[name]) for name in _TENSOR_NAMES}
        raw_frames = np.asarray(stored["frames_seen"])
        histories = {
            name: np.asarray(stored[name]) if name in stored else np.asarray([], dtype="f4")
            for name in ("reconstruction_history", "novelty_history", "update_magnitude_history")
        }
        raw_state = np.asarray(stored["maturity_state"]) if "maturity_state" in stored else np.asarray("Baby")
        frozen = bool(np.asarray(stored["weights_frozen"]).reshape(-1)[0]) if "weights_frozen" in stored else False

    if raw_frames.size != 1:
        raise ValueError("The analysis NPZ has an invalid frames_seen value")
    frames_seen = int(raw_frames.reshape(-1)[0])
    if frames_seen < 0:
        raise ValueError("The analysis NPZ has a negative frames_seen value")
    non_finite = [name for name, value in tensors.items() if not np.isfinite(value).all()]
    if non_finite:
        raise ValueError(f"The analysis NPZ contains non-finite values in: {', '.join(non_finite)}")

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
    shape_errors = [name for name, expected in expected_shapes.items() if tensors[name].shape != expected]
    if shape_errors:
        raise ValueError(f"The analysis NPZ has incompatible tensor shapes: {', '.join(shape_errors)}")

    modified = datetime.fromtimestamp(stat.st_mtime, UTC)
    state = str(raw_state.reshape(-1)[0])
    if state not in {"Baby", "Teen", "Adult", "Elder"}:
        state = "Baby"
    history_present = all(history.size for history in histories.values())
    consistency = False
    if history_present and histories["reconstruction_history"].size >= 32:
        recent = histories["reconstruction_history"][-32:]
        consistency = bool(float(recent.std()) <= .015 and float(recent.mean()) <= .08)
    required = 250 if state == "Baby" else 4096
    maturity = {
        "state": state,
        "next_state": {"Baby": "Teen", "Teen": "Adult", "Adult": "Elder", "Elder": None}[state],
        "progress_percent": 100 if state == "Elder" else min(99, round(frames_seen / required * 100)),
        "metrics_persisted": history_present,
        "consistency_sustained": consistency,
        "weights_frozen": frozen,
        "readiness": "ready" if state in {"Adult", "Elder"} and history_present else "caution",
        "note": "Legacy files retain their learned tensors but are conservatively treated as Baby until new rolling metrics are observed."
                if not history_present else "Maturity is a persisted learning-health signal, not an accuracy guarantee.",
    }
    parameter_count = sum(int(value.size) for value in tensors.values() if value.ndim)
    return {
        "status": "Healthy",
        "health_checks": {
            "required_tensors": "passed",
            "finite_numeric_values": "passed",
            "architecture_compatibility": "passed",
            "persistence_format": "compressed NPZ",
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
            "input_features": input_width,
            "latent_features": latent_width,
            "shape": f"{input_width} -> {latent_width} -> {input_width}",
            "learned_parameters": parameter_count,
        },
        "tensors": {name: _tensor_summary(value) for name, value in tensors.items()},
    }


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
            self.completed.emit(inspect_npz_model(self._path))
        except (OSError, ValueError) as exc:
            self.failed.emit(str(exc))


class AnalysisWindow(QMainWindow):
    """Read-only NPZ and exported JSON inspection surface."""

    inspection_finished = Signal(bool)

    def __init__(self, *, auto_refresh: bool = True) -> None:
        super().__init__()
        self.setWindowTitle("AIBrain Analysis Inspector")
        self.resize(1120, 760)
        self._inspection_thread: QThread | None = None
        self._inspection_worker: AnalysisInspectionWorker | None = None
        self._closing = False
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
        refresh.setToolTip("Re-read the persisted Analysis+ model and its rolling metrics")
        refresh.clicked.connect(self.refresh)
        open_export = QPushButton("Inspect exported analysis JSON")
        open_export.setToolTip("Open an explicit JSON or JSON.GZ export for read-only inspection")
        open_export.clicked.connect(self.inspect_export)
        actions.addWidget(refresh)
        actions.addWidget(open_export)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.tabs = QTabWidget()
        self.tabs.setToolTip("Dashboard overview and optional raw export details")
        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        self.health_row = QWidget()
        health_grid = metric_grid(self.health_row)
        health, self.health_value, self.health_detail = metric_card("MODEL HEALTH", "Reading…")
        self.state_card, self.state_value, self.state_detail = metric_card("MATURITY STATE", "—")
        self.frame_card, self.frame_value, self.frame_detail = metric_card("PERSISTENCE", "—")
        health_grid.addWidget(health, 0, 0)
        health_grid.addWidget(self.state_card, 0, 1)
        health_grid.addWidget(self.frame_card, 0, 2)
        overview_layout.addWidget(self.health_row)
        progress, self.maturity_progress, self.maturity_detail = progress_card("PROGRESS TOWARD NEXT STATE")
        overview_layout.addWidget(progress)
        self.architecture_card, self.architecture_value, self.architecture_detail = metric_card("ARCHITECTURE", "—")
        self.quality_card, self.quality_value, self.quality_detail = metric_card("LEARNING QUALITY", "—")
        self.export_card, self.export_value, self.export_detail = metric_card("EXPORT READINESS", "—")
        lower = QWidget()
        lower_grid = metric_grid(lower)
        lower_grid.addWidget(self.architecture_card, 0, 0)
        lower_grid.addWidget(self.quality_card, 0, 1)
        lower_grid.addWidget(self.export_card, 0, 2)
        overview_layout.addWidget(lower)
        overview_layout.addStretch(1)
        self.raw = QPlainTextEdit()
        self.raw.setReadOnly(True)
        self.raw.setPlaceholderText("Raw JSON is available only when needed.")
        self.tabs.addTab(overview, "Dashboard")
        self.tabs.addTab(self.raw, "Raw JSON / details")
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(page)
        if auto_refresh:
            QTimer.singleShot(0, self.refresh)

    def refresh(self) -> None:
        if self._closing or self._inspection_thread is not None:
            return
        path = ConnectomeAnalyzer.default_model_path()
        if not path.is_file():
            self._show_empty(path)
            QTimer.singleShot(0, lambda: self.inspection_finished.emit(False))
            return
        self.raw.setPlainText("Reading persisted Analysis+ model in the background…")
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
    def _inspection_ready(self, metadata: object) -> None:
        if isinstance(metadata, dict):
            self._show_metadata(metadata)
        self.raw.setPlainText(json.dumps(metadata, indent=2))

    @Slot(str)
    def _inspection_failed(self, message: str) -> None:
        self.health_value.setText("Needs repair")
        self.health_detail.setText(message)
        self.raw.setPlainText(f"Status: Invalid Analysis+ model\n\n{message}")

    @Slot()
    def _inspection_finished(self) -> None:
        self._inspection_thread = None
        self._inspection_worker = None
        if self._closing:
            QTimer.singleShot(0, self.close)
            return
        self.inspection_finished.emit(self.health_value.text() == "Healthy")

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._inspection_thread is not None:
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
            return
        try:
            selected = Path(filename)
            if selected.suffix == ".gz":
                with gzip.open(selected, "rt", encoding="utf-8") as handle:
                    payload = json.load(handle)
            else:
                payload = json.loads(selected.read_text(encoding="utf-8"))
            self.raw.appendPlainText("\n\nExport summary:\n" + json.dumps({
                "path": str(selected),
                "schema": payload.get("schema"),
                "created_at": payload.get("created_at"),
                "conversation_turns": len(payload.get("conversation", [])),
                "has_nn_findings": "analysis_plus" in payload or "neural_network" in payload,
                "recorded_frame_summary": payload.get("recorded_frame_summary", {}),
            }, indent=2))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.raw.appendPlainText(f"\n\nCould not read export: {exc}")

    def _show_empty(self, path: Path) -> None:
        self.health_value.setText("No model yet")
        self.health_detail.setText("Generate an Infinite-mode response to begin learning.")
        self.state_value.setText("BABY")
        self.state_detail.setText("No persisted metric history")
        self.frame_value.setText("0 frames")
        self.frame_detail.setText(str(path))
        self.maturity_progress.setValue(0)
        self.maturity_detail.setText("250 persisted frames plus sustained consistency are required for Teen.")
        self.architecture_value.setText("Not created")
        self.quality_value.setText("Awaiting frames")
        self.export_value.setText("CAUTION")
        self.export_detail.setText("Baby/Teen exports include a readiness caution.")
        self.raw.setPlainText(f"Status: No Analysis+ model has been learned yet.\n\nExpected location:\n{path}")

    def _show_metadata(self, metadata: dict[str, object]) -> None:
        learning = metadata.get("learning", {})
        architecture = metadata.get("architecture", {})
        file_data = metadata.get("file", {})
        maturity = learning.get("maturity", {}) if isinstance(learning, dict) else {}
        if not isinstance(maturity, dict):
            maturity = {}
        self.health_value.setText(str(metadata.get("status", "Unknown")))
        self.health_detail.setText("All tensor, finite-value, and architecture checks passed.")
        state = str(maturity.get("state", "Baby"))
        self.state_value.setText(state.upper())
        self.state_detail.setText("Weights frozen" if maturity.get("weights_frozen") else "Training weights remain active")
        frames = int(learning.get("lifetime_frames_seen", 0)) if isinstance(learning, dict) else 0
        self.frame_value.setText(f"{frames:,} frames")
        self.frame_detail.setText(str(file_data.get("age", "Unknown persistence age")) if isinstance(file_data, dict) else "")
        progress = int(maturity.get("progress_percent", 0))
        self.maturity_progress.setValue(progress)
        next_state = maturity.get("next_state") or "terminal Elder state"
        self.maturity_detail.setText(f"Toward {next_state}: {maturity.get('note', '')}")
        self.architecture_value.setText(str(architecture.get("shape", "Unknown")) if isinstance(architecture, dict) else "Unknown")
        self.architecture_detail.setText(f"{architecture.get('learned_parameters', 0):,} learned parameters" if isinstance(architecture, dict) else "")
        self.quality_value.setText("Consistent" if maturity.get("consistency_sustained") else "Learning")
        self.quality_detail.setText("Rolling reconstruction and update metrics are persisted.")
        readiness = str(maturity.get("readiness", "caution")).upper()
        self.export_value.setText(readiness)
        self.export_detail.setText("Normal smart-analysis report" if readiness == "READY" else "Export remains valid with explicit readiness caution.")
