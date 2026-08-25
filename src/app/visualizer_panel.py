from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..connectome.activity import ActivityField
from ..connectome.generator import build_connectome
from ..connectome.mapper import ActivityMapper
from ..connectome.renderer import ConnectomeRenderer
from ..models.instrumented_backend import ActivationFrame, ActivitySource


class VisualizerPanel(QWidget):
    def __init__(self, model_key: str = "default") -> None:
        super().__init__()
        self._model_key = model_key
        self._build_graph("Medium")
        self._build_ui()

    def _build_graph(self, quality: str) -> None:
        self.graph = build_connectome(self._model_key, quality)
        self.field = ActivityField(self.graph)
        self.mapper = ActivityMapper(self.field)
        self.renderer = ConnectomeRenderer(self.graph, self.field)
        self.renderer.nodeSelected.connect(self._inspect)

    def _build_ui(self) -> None:
        self.layout = QVBoxLayout(self); self.layout.setContentsMargins(8, 18, 18, 18)
        header = QHBoxLayout(); title = QLabel("Live Connectome"); title.setObjectName("title")
        self.mode = QLabel("SIMULATION") ; self.mode.setObjectName("mode")
        self.quality = QComboBox(); self.quality.addItems(["Low", "Medium", "High"]); self.quality.setCurrentText("Medium"); self.quality.currentTextChanged.connect(self._rebuild)
        self.pause = QPushButton("Pause"); self.pause.clicked.connect(self._toggle_pause)
        reset = QPushButton("Reset view"); reset.clicked.connect(lambda: self.renderer.reset_camera())
        for widget in (title, self.mode, self.quality, self.pause, reset): header.addWidget(widget)
        header.addStretch(1); self.layout.addLayout(header)
        self.layout.addWidget(self.renderer, 1)
        self.overlay = QLabel(); self.overlay.setObjectName("overlay"); self.overlay.setWordWrap(True)
        self.inspector = QLabel("Click a visual neuron to inspect its mapped visual data."); self.inspector.setObjectName("muted")
        self.layout.addWidget(self.overlay); self.layout.addWidget(self.inspector)
        self._refresh_overlay()

    def _rebuild(self, quality: str) -> None:
        old = self.renderer; self.layout.replaceWidget(old, QWidget())
        old.deleteLater(); self._build_graph(quality); self.layout.insertWidget(1, self.renderer, 1); self._refresh_overlay()

    def _toggle_pause(self) -> None:
        self.renderer.paused = not self.renderer.paused; self.pause.setText("Resume" if self.renderer.paused else "Pause")

    def apply_frame(self, frame: ActivationFrame) -> None:
        self.mode.setText(frame.source.value.upper())
        self.mapper.apply(frame); self._refresh_overlay()

    def set_model(self, key: str) -> None:
        self._model_key = key; self._rebuild(self.quality.currentText())

    def _refresh_overlay(self) -> None:
        active_edges = int(sum((self.field.values[self.graph.edges[:, 0]] > .1) | (self.field.values[self.graph.edges[:, 1]] > .1)))
        strongest = int(self.field.values.argmax()) if len(self.field.values) else 0
        region = self.graph.region_names[int(self.graph.regions[strongest])]
        self.overlay.setText(f"Visualization: Simulation (token-driven, not measured activation)  ·  Step {self.field.step}  ·  Active neurons {self.field.active_count:,}  ·  Active pathways {active_edges:,}  ·  Strongest: {region}  ·  Token: {self.field.current_token!r}")

    def _inspect(self, index: int) -> None:
        self.inspector.setText(f"Visual node: {index:05d}  ·  Region: {self.graph.region_names[int(self.graph.regions[index])]}  ·  Current activity: {self.field.values[index]:.3f}  ·  Peak: {self.field.peaks[index]:.3f}  ·  Data source: Simulation")
