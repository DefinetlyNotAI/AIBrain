"""Standalone inspector for AIBrain's persisted connectome analysis model."""
from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import QFileDialog, QLabel, QMainWindow, QPushButton, QPlainTextEdit, QVBoxLayout, QWidget

from ..connectome.analysis import ConnectomeAnalyzer

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


def _maturity(frames_seen: int) -> tuple[str, int]:
    """Return a transparent persistence-maturity estimate, not a benchmark."""
    if frames_seen == 0:
        return "Untrained", 0
    if frames_seen < 25:
        return "Early learning", 25
    if frames_seen < 250:
        return "Developing", 50
    if frames_seen < 1_000:
        return "Established", 75
    return "Mature", 100


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
    maturity, maturity_score = _maturity(frames_seen)
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
            "maturity_score_out_of_100": maturity_score,
            "quality_note": "Maturity reflects observed-frame coverage and persistence health, not an accuracy benchmark.",
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
        self.resize(970, 700)
        self._inspection_thread: QThread | None = None
        self._inspection_worker: AnalysisInspectionWorker | None = None
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
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlaceholderText("Reading persisted Analysis+ data…")
        layout.addWidget(self.summary, 1)
        refresh = QPushButton("Refresh model health")
        refresh.clicked.connect(self.refresh)
        layout.addWidget(refresh)
        open_export = QPushButton("Inspect exported analysis JSON")
        open_export.clicked.connect(self.inspect_export)
        layout.addWidget(open_export)
        self.setCentralWidget(page)
        if auto_refresh:
            QTimer.singleShot(0, self.refresh)

    def refresh(self) -> None:
        if self._inspection_thread is not None:
            return
        path = ConnectomeAnalyzer.default_model_path()
        if not path.is_file():
            self.summary.setPlainText(
                "Status: No Analysis+ model has been learned yet.\n\n"
                f"Expected location:\n{path}\n\n"
                "Generate an Infinite-mode response to record visual frames and create the model."
            )
            QTimer.singleShot(0, lambda: self.inspection_finished.emit(False))
            return
        self.summary.setPlainText("Reading persisted Analysis+ model in the background…")
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
        self.summary.setPlainText(json.dumps(metadata, indent=2))

    @Slot(str)
    def _inspection_failed(self, message: str) -> None:
        self.summary.setPlainText(f"Status: Invalid Analysis+ model\n\n{message}")

    @Slot()
    def _inspection_finished(self) -> None:
        self._inspection_thread = None
        self._inspection_worker = None
        self.inspection_finished.emit("Status: Healthy" in self.summary.toPlainText())

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
            self.summary.appendPlainText("\n\nExport summary:\n" + json.dumps({
                "path": str(selected),
                "schema": payload.get("schema"),
                "created_at": payload.get("created_at"),
                "conversation_turns": len(payload.get("conversation", [])),
                "has_nn_findings": "analysis_plus" in payload or "neural_network" in payload,
                "recorded_frame_summary": payload.get("recorded_frame_summary", {}),
            }, indent=2))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.summary.appendPlainText(f"\n\nCould not read export: {exc}")
