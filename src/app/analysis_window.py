"""Standalone inspector for AIBrain's persisted connectome analysis model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QFileDialog, QLabel, QMainWindow, QPushButton, QPlainTextEdit, QVBoxLayout, QWidget

from ..connectome.analysis import ConnectomeAnalyzer


def inspect_npz_model(path: Path) -> dict[str, object]:
    """Return validated persisted-model metadata without loading a GGUF or renderer."""
    required = ("encoder_weights", "encoder_bias", "decoder_weights", "decoder_bias", "embedding_centroid", "frames_seen")
    with np.load(path, allow_pickle=False) as stored:
        if any(name not in stored for name in required):
            raise ValueError("The analysis NPZ is missing required tensors")
        tensors = {name: stored[name] for name in required}
    if not all(np.isfinite(value).all() for name, value in tensors.items() if name != "frames_seen"):
        raise ValueError("The analysis NPZ contains non-finite tensor values")
    return {"path": str(path), "frames_seen": int(tensors["frames_seen"]),
            "tensors": {name: list(value.shape) for name, value in tensors.items() if name != "frames_seen"}}


class AnalysisWindow(QMainWindow):
    """Read-only NPZ and exported JSON inspection surface."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AIBrain Analysis Inspector")
        self.resize(900, 650)
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("Persisted analysis model")
        title.setObjectName("title")
        layout.addWidget(title)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        layout.addWidget(self.summary, 1)
        open_export = QPushButton("Inspect exported analysis JSON")
        open_export.clicked.connect(self.inspect_export)
        layout.addWidget(open_export)
        self.setCentralWidget(page)
        self.refresh()

    def refresh(self) -> None:
        path = ConnectomeAnalyzer.default_model_path()
        if not path.is_file():
            self.summary.setPlainText(f"No persisted analysis model exists yet.\n\nExpected location:\n{path}")
            return
        try:
            metadata = inspect_npz_model(path)
        except (OSError, ValueError) as exc:
            self.summary.setPlainText(f"The persisted analysis model is invalid:\n{exc}")
            return
        self.summary.setPlainText(json.dumps(metadata, indent=2))

    def inspect_export(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Inspect analysis export", "", "JSON data (*.json)")
        if not filename:
            return
        try:
            payload = json.loads(Path(filename).read_text(encoding="utf-8"))
            self.summary.appendPlainText("\n\nExport summary:\n" + json.dumps({
                "schema": payload.get("schema"), "created_at": payload.get("created_at"),
                "conversation_turns": len(payload.get("conversation", [])),
            }, indent=2))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.summary.appendPlainText(f"\n\nCould not read export: {exc}")
