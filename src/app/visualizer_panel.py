from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np
from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget

from ..connectome.activity import ActivityField
from ..connectome.analysis import ConnectomeAnalyzer
from ..connectome.export import export_analysis
from ..connectome.generator import build_connectome
from ..connectome.mapper import ActivityMapper
from ..connectome.renderer import ConnectomeRenderer
from ..models.instrumented_backend import ActivationFrame, ActivitySource
from ..utils.gpu import discover_render_adapters, set_windows_gpu_preference


@dataclass(slots=True)
class PlaybackStep:
    frame: ActivationFrame
    values: np.ndarray
    peaks: np.ndarray


class VisualizerPanel(QWidget):
    def __init__(self, model_key: str = "default") -> None:
        super().__init__()
        self._model_key = model_key
        self._cluster_spacing = 1.0
        self._selected_node: int | None = None
        self._playback: list[PlaybackStep] = []
        self._playback_index = -1
        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._advance_playback)
        self._build_graph("Medium")
        self._build_ui()

    def _build_graph(self, quality: str) -> None:
        self.graph = build_connectome(self._model_key, quality, self._cluster_spacing)
        self.field = ActivityField(self.graph)
        self.analyzer = ConnectomeAnalyzer(self.graph)
        self.mapper = ActivityMapper(self.field)
        self.renderer = ConnectomeRenderer(self.graph, self.field)
        self.renderer.nodeSelected.connect(self._inspect)
        self.renderer.backendChanged.connect(self._set_backend)

    def _build_ui(self) -> None:
        self.layout = QVBoxLayout(self); self.layout.setContentsMargins(8, 18, 18, 18)
        header = QHBoxLayout(); title = QLabel("Live Connectome"); title.setObjectName("title")
        self.mode = QLabel("SIMULATION") ; self.mode.setObjectName("mode")
        self.quality = QComboBox(); self.quality.addItems(["Low", "Medium", "High"]); self.quality.setCurrentText("Medium"); self.quality.currentTextChanged.connect(self._rebuild)
        self.render_gpu = QComboBox(); self.render_gpu.setToolTip("AIBrain starts a fresh process after saving the Windows high-performance GPU preference.")
        self._populate_render_adapters()
        self.render_gpu.currentIndexChanged.connect(self._set_render_preference)
        self.spacing = QSlider(Qt.Orientation.Horizontal); self.spacing.setRange(60, 200); self.spacing.setValue(100); self.spacing.setToolTip("Cluster spacing")
        self.spacing.valueChanged.connect(self._set_cluster_spacing)
        self.pause = QPushButton("Pause"); self.pause.clicked.connect(self._toggle_pause)
        self.analysis = QPushButton("Analysis"); self.analysis.clicked.connect(self._show_analysis)
        self.export = QPushButton("Export"); self.export.clicked.connect(self._export_analysis)
        reset = QPushButton("Reset view"); reset.clicked.connect(lambda: self.renderer.reset_camera())
        for widget in (title, self.mode, self.quality, self.render_gpu, QLabel("Spacing"), self.spacing, self.analysis, self.export, self.pause, reset): header.addWidget(widget)
        header.addStretch(1); self.layout.addLayout(header)
        self.layout.addWidget(self.renderer, 1)
        self.overlay = QLabel(); self.overlay.setObjectName("overlay"); self.overlay.setWordWrap(True)
        self.inspector = QLabel("Click a visual neuron to inspect its mapped visual data."); self.inspector.setObjectName("muted")
        self.silence = QPushButton("Silence selected neuron"); self.silence.clicked.connect(self._toggle_selected_node); self.silence.setEnabled(False)
        self.importance = QDoubleSpinBox(); self.importance.setRange(0, 3); self.importance.setSingleStep(.1); self.importance.setValue(1); self.importance.valueChanged.connect(self._change_importance); self.importance.setEnabled(False)
        inspect_controls = QHBoxLayout(); inspect_controls.addWidget(self.silence); inspect_controls.addWidget(QLabel("Importance")); inspect_controls.addWidget(self.importance); inspect_controls.addStretch(1)
        self.layout.addWidget(self.overlay); self.layout.addWidget(self.inspector); self.layout.addLayout(inspect_controls)
        self._refresh_overlay()

    def _rebuild(self, quality: str) -> None:
        self.begin_recording()
        old = self.renderer
        index = self.layout.indexOf(old)
        self.layout.takeAt(index)
        old.setParent(None)
        old.deleteLater()
        self._build_graph(quality)
        self.layout.insertWidget(index, self.renderer, 1)
        self._refresh_overlay()

    def _toggle_pause(self) -> None:
        self.renderer.paused = not self.renderer.paused; self.pause.setText("Resume" if self.renderer.paused else "Pause")

    def apply_frame(self, frame: ActivationFrame, *, record: bool = True) -> None:
        self.mode.setText(frame.source.value.upper())
        self.mapper.apply(frame)
        if record:
            self._playback.append(PlaybackStep(frame, self.field.values.copy(), self.field.peaks.copy()))
            self.analyzer.observe(frame, self.field.values)
        self._refresh_overlay()

    def begin_recording(self) -> None:
        self._playback_timer.stop()
        self._playback.clear()
        self._playback_index = -1
        self.analyzer.records.clear()

    def start_playback(self, speed: float) -> None:
        if not self._playback:
            return
        self._playback_index = -1
        self.renderer.paused = True
        self.pause.setText("Resume")
        self._playback_timer.start(max(35, round(110 / max(.1, speed))))

    def playback_next(self) -> None:
        self._playback_timer.stop()
        self._show_playback_step(min(self._playback_index + 1, len(self._playback) - 1))

    def playback_previous(self) -> None:
        self._playback_timer.stop()
        self._show_playback_step(max(0, self._playback_index - 1))

    def _advance_playback(self) -> None:
        if self._playback_index >= len(self._playback) - 1:
            self._playback_timer.stop()
            return
        self._show_playback_step(self._playback_index + 1)

    def _show_playback_step(self, index: int) -> None:
        if not self._playback:
            return
        step = self._playback[index]
        if step.values.shape != self.field.values.shape:
            self._playback_timer.stop()
            self._playback.clear()
            self._playback_index = -1
            self.inspector.setText("Replay discarded because the connectome graph changed. Generate a new response to record it.")
            return
        self._playback_index = index
        self.field.values[:] = step.values
        self.field.peaks[:] = step.peaks
        self.field.step = step.frame.step
        self.field.current_token = step.frame.token_text
        self.field.last_time = monotonic()
        self.mode.setText(step.frame.source.value.upper())
        self.renderer.update()
        self._refresh_overlay()

    def set_model(self, key: str) -> None:
        self.begin_recording()
        self._model_key = key; self._rebuild(self.quality.currentText())

    def _refresh_overlay(self) -> None:
        active_edges = int(sum((self.field.values[self.graph.edges[:, 0]] > .1) | (self.field.values[self.graph.edges[:, 1]] > .1)))
        strongest = int(self.field.values.argmax()) if len(self.field.values) else 0
        region = self.graph.region_names[int(self.graph.regions[strongest])]
        backend = getattr(self, "_backend", "ModernGL GPU renderer initializing…")
        self.overlay.setText(f"{backend}  ·  Visualization: Simulation (token-driven, not measured activation)  ·  Step {self.field.step}  ·  Active neurons {self.field.active_count:,}  ·  Active pathways {active_edges:,}  ·  Strongest: {region}  ·  Token: {self.field.current_token!r}")

    def _set_backend(self, backend: str) -> None:
        self._backend = backend
        self._refresh_overlay()

    def _inspect(self, index: int) -> None:
        self._selected_node = index; self.silence.setEnabled(True); self.importance.setEnabled(True); self.importance.setValue(float(self.field.importance[index]))
        self.inspector.setText(f"Visual node: {index:05d}  ·  Region: {self.graph.region_names[int(self.graph.regions[index])]}  ·  Current activity: {self.field.values[index]:.3f}  ·  Peak: {self.field.peaks[index]:.3f}  ·  Data source: Simulation")

    def _toggle_selected_node(self) -> None:
        if self._selected_node is None: return
        index = self._selected_node; disabled = not bool(self.field.disabled[index]); self.field.set_disabled(index, disabled)
        self.silence.setText("Restore selected neuron" if disabled else "Silence selected neuron"); self.renderer.update()

    def _change_importance(self, value: float) -> None:
        if self._selected_node is not None: self.field.set_importance(self._selected_node, value)

    def _set_cluster_spacing(self, value: int) -> None:
        self._cluster_spacing = value / 100
        if hasattr(self, "renderer"): self._rebuild(self.quality.currentText())

    def _show_analysis(self) -> None:
        summary = self.analyzer.summary()
        if not summary.get("frames"):
            QMessageBox.information(self, "Connectome analysis", "Generate a response first. Analysis learns from the visual activity stream of the latest response.")
            return
        QMessageBox.information(self, "Connectome analysis (Derived)", f"Frames analysed: {summary['frames']}\nMean novelty: {summary['mean_novelty']:.4f}\nPeak novelty: {summary['peak_novelty']:.4f}\nMost active region: {summary['most_active_region']}\n\nThis adaptive encoder studies the visualization activity stream. It is not measured transformer activation data.")

    def _export_analysis(self) -> None:
        if not self.analyzer.records:
            QMessageBox.information(self, "Export analysis", "Generate a response before exporting analysis.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Export connectome analysis", "connectome-analysis.json", "JSON analysis (*.json);;CSV events (*.csv)")
        if not filename:
            return
        try:
            export_analysis(Path(filename), self.graph, self.analyzer.records, self.analyzer.summary())
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _populate_render_adapters(self) -> None:
        settings = QSettings()
        selected_value = settings.value("render_adapter")
        selected = str(selected_value) if selected_value is not None else "system"
        self.render_gpu.blockSignals(True)
        self.render_gpu.addItem("GPU: System default", "system")
        for adapter in discover_render_adapters():
            self.render_gpu.addItem(f"GPU: {adapter.name}", adapter.identifier)
            if selected_value is None and "nvidia" in adapter.name.lower():
                selected = adapter.identifier
        index = self.render_gpu.findData(selected)
        self.render_gpu.setCurrentIndex(index if index >= 0 else 0)
        self.render_gpu.blockSignals(False)
        if selected != "system":
            set_windows_gpu_preference(True)
            self._backend = "NVIDIA high-performance GPU enforced at launch; verifying OpenGL context…"

    def _set_render_preference(self, _index: int) -> None:
        identifier = str(self.render_gpu.currentData())
        QSettings().setValue("render_adapter", identifier)
        if identifier == "system":
            set_windows_gpu_preference(False)
            self._backend = "ModernGL GPU · Windows system-default adapter"
            self._refresh_overlay()
            return
        applied = set_windows_gpu_preference(True)
        self._backend = "High-performance GPU saved; AIBrain will relaunch on the next start" if applied else "GPU preference could not be saved; configure Windows Graphics Settings"
        self._refresh_overlay()
        QMessageBox.information(
            self, "Rendering adapter preference",
            "AIBrain writes the Windows high-performance setting for both virtual-environment Python hosts and starts the app in a fresh process. "
            "Restart AIBrain for Windows to apply it. If the overlay still reports another adapter, choose that executable "
            "in Windows Settings > System > Display > Graphics. The overlay always reports the actual renderer.",
        )
