from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np
from PySide6.QtCore import QSettings, QTimer, Qt, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, \
    QLabel, QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget, QInputDialog

from ..connectome.activity import ActivityField
from ..connectome.analysis import ConnectomeAnalyzer
from ..connectome.export import export_nn_analysis_plus, export_session_analysis
from ..connectome.generator import build_connectome
from ..connectome.mapper import ActivityMapper
from ..connectome.renderer import ConnectomeRenderer
from ..models.instrumented_backend import ActivationFrame
from ..utils.gpu import discover_render_adapters, set_windows_gpu_preference
from .theme import ColourSettingsDialog, load_colours, save_colours


@dataclass(slots=True)
class PlaybackStep:
    frame: ActivationFrame
    values: np.ndarray
    peaks: np.ndarray


class NeuronInspectorLabel(QLabel):
    """Selectable inspector text with a click target only over the node number."""

    neuronNumberClicked = Signal()
    _prefix = "Selected neuron: "

    def set_neuron_details(self, index: int, details: str) -> None:
        self.setText(f"{self._prefix}{index:05d}{details}")

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        prefix_width = self.fontMetrics().horizontalAdvance(self._prefix)
        number_width = self.fontMetrics().horizontalAdvance("00000")
        if (
                event.button() == Qt.MouseButton.LeftButton and
                prefix_width <= event.position().x() <= prefix_width + number_width
        ):
            self.neuronNumberClicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class VisualizerPanel(QWidget):
    gpuRestartRequested = Signal(str)
    rewindChanged = Signal(bool)
    playbackChanged = Signal(bool)
    analysisMemoryExceeded = Signal(bool)
    coloursChanged = Signal(object)

    def __init__(self, model_key: str = "default") -> None:
        super().__init__()
        self._model_key = model_key
        self._cluster_spacing = 1.0
        self._pending_cluster_spacing = self._cluster_spacing
        self._spacing_timer = QTimer(self)
        self._spacing_timer.setSingleShot(True)
        self._spacing_timer.setInterval(220)
        self._spacing_timer.timeout.connect(self._commit_cluster_spacing)
        self._selected_node: int | None = None
        self._playback: list[PlaybackStep] = []
        self._all_signals: list[PlaybackStep] = []
        self._playback_index = -1
        self._rewind_active = False
        self._replay_active = False
        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._advance_playback)
        self._conversation: list[dict[str, object]] = []
        self._analysis_memory_limit_bytes: int | None = None
        self._analysis_memory_exceeded = False
        self._analysis_retained_bytes = 0
        self._colours = load_colours()
        self._build_graph("Medium")
        self._build_ui()

    def _build_graph(self, quality: str) -> None:
        self.graph = build_connectome(self._model_key, quality, self._cluster_spacing)
        self.field = ActivityField(self.graph)
        self.analyzer = ConnectomeAnalyzer(self.graph)
        self.mapper = ActivityMapper(self.field)
        self.renderer = ConnectomeRenderer(self.graph, self.field)
        self.renderer.setMinimumWidth(0)
        self.renderer.set_background_colour(self._colours["renderer_background"])
        self.renderer.nodeSelected.connect(self._inspect)
        self.renderer.backendChanged.connect(self._set_backend)
        self.renderer.gpuRestartRequested.connect(self.gpuRestartRequested)

    def _build_ui(self) -> None:
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 18, 18, 18)

        header = QGridLayout()
        header.setHorizontalSpacing(8)
        header.setVerticalSpacing(6)

        title = QLabel("Live Connectome")
        title.setObjectName("title")

        self.mode = QLabel("SIMULATION")
        self.mode.setObjectName("mode")

        self.quality = QComboBox()
        self.quality.addItems(["Low", "Medium", "High"])
        self.quality.setCurrentText("Medium")
        self.quality.currentTextChanged.connect(self._rebuild)

        self.render_gpu = QComboBox()
        self.render_gpu.setToolTip(
            "AIBrain starts a fresh process after saving the Windows high-performance GPU preference."
        )
        self._populate_render_adapters()
        self.render_gpu.currentIndexChanged.connect(self._set_render_preference)

        self.spacing = QSlider(Qt.Orientation.Horizontal)
        self.spacing.setRange(60, 200)
        self.spacing.setValue(100)
        self.spacing.setToolTip("Cluster spacing")

        self.spacing_value = QLabel("1.00×")
        self.spacing_value.setObjectName("muted")
        self.spacing.valueChanged.connect(self._preview_cluster_spacing)

        settings = QSettings()

        self.neuron_borders = QCheckBox("Neuron borders")
        self.neuron_borders.setToolTip(
            "Outline each visual neuron for clearer separation"
        )

        neuron_borders_value = settings.value("neuron_borders", True)
        self.neuron_borders.setChecked(
            neuron_borders_value
            if isinstance(neuron_borders_value, bool)
            else True
        )

        self.neuron_borders.toggled.connect(self._set_neuron_borders)

        self.border_width = QDoubleSpinBox()
        self.border_width.setRange(0, 1)
        self.border_width.setSingleStep(.01)
        self.border_width.setDecimals(2)

        border_width_value = settings.value("neuron_border_width", .12)
        self.border_width.setValue(
            float(border_width_value)
            if isinstance(border_width_value, (int, float, str))
            else .12
        )

        self.border_width.setToolTip("Neuron outline width")
        self.border_width.valueChanged.connect(
            lambda _: self._set_neuron_borders(
                self.neuron_borders.isChecked()
            )
        )

        self.analysis = QPushButton("NN Analysis+")
        self.analysis.setToolTip(
            "Analyze every recorded visual frame and export compact neural-network findings"
        )
        self.analysis.clicked.connect(self.run_nn_analysis_plus)
        self.analysis.hide()

        self.view_toggle = QPushButton("3D View")
        self.view_toggle.setCheckable(True)
        self.view_toggle.setToolTip("Toggle the depth-aware 3D map and a flat 2D map")
        self.view_toggle.toggled.connect(self._set_view_mode)
        self.two_d_mode = QComboBox()
        self.two_d_mode.addItem("Full 2D", "full")
        self.two_d_mode.addItem("Sector 2D", "sector")
        self.two_d_mode.currentIndexChanged.connect(self._apply_view_mode)
        self.two_d_mode.hide()
        self.region_selector = QComboBox()
        self.region_selector.currentIndexChanged.connect(self._apply_view_mode)
        self.region_selector.hide()
        self._populate_region_selector()

        self.reset_view = QPushButton("Reset view")
        self.reset_view.setToolTip("Restore the default 2D or 3D camera position and zoom")
        self.reset_view.clicked.connect(lambda: self.renderer.reset_camera())
        self.settings_toggle = QPushButton("View settings")
        self.settings_toggle.setCheckable(True)
        self.settings_toggle.setToolTip("Show or hide rendering and presentation settings")
        self.settings_toggle.toggled.connect(self._set_settings_visible)
        self.colour_settings = QPushButton("Colour settings")
        self.colour_settings.setToolTip("Customize the saved AIBrain interface palette")
        self.colour_settings.clicked.connect(self._edit_colours)

        spacing_label = QLabel("Spacing")
        importance_label = QLabel("Importance")

        header.addWidget(title, 0, 0)
        header.addWidget(self.mode, 0, 1)
        header.addWidget(self.view_toggle, 0, 2)
        header.addWidget(self.reset_view, 0, 3)
        header.addWidget(self.settings_toggle, 0, 4)
        header.addWidget(self.two_d_mode, 1, 1)
        header.addWidget(self.region_selector, 1, 2)
        header.addWidget(self.analysis, 1, 3, 1, 2)
        header.addWidget(self.colour_settings, 1, 0)
        header.setColumnStretch(0, 1)

        self.layout.addLayout(header)
        self.settings_panel = QWidget()
        settings_layout = QFormLayout(self.settings_panel)
        settings_layout.setContentsMargins(0, 2, 0, 6)
        settings_layout.addRow("Simulation Performance", self.quality)
        settings_layout.addRow("Rendering GPU", self.render_gpu)
        spacing_row = QWidget()
        spacing_row_layout = QHBoxLayout(spacing_row)
        spacing_row_layout.setContentsMargins(0, 0, 0, 0)
        spacing_row_layout.addWidget(self.spacing)
        spacing_row_layout.addWidget(self.spacing_value)
        settings_layout.addRow(spacing_label, spacing_row)
        borders_row = QWidget()
        borders_layout = QHBoxLayout(borders_row)
        borders_layout.setContentsMargins(0, 0, 0, 0)
        borders_layout.addWidget(self.neuron_borders)
        borders_layout.addWidget(self.border_width)
        settings_layout.addRow("Neuron Borders (0–1)", borders_row)
        self.settings_panel.setVisible(False)
        self.layout.addWidget(self.settings_panel)
        self.layout.addWidget(self.renderer, 1)
        self.rewind_controls = QWidget()
        rewind_layout = QHBoxLayout(self.rewind_controls)
        rewind_layout.setContentsMargins(0, 4, 0, 4)
        self.previous_rewind = QPushButton("Previous")
        self.replay_rewind = QPushButton("Replay")
        self.next_rewind = QPushButton("Next")
        for button, tip in (
                (self.previous_rewind, "Show the previous recorded token frame"),
                (self.replay_rewind, "Play recorded token frames from the beginning"),
                (self.next_rewind, "Show the next recorded token frame"),
        ):
            button.setToolTip(tip)
            rewind_layout.addWidget(button)
        rewind_layout.addStretch(1)
        self.previous_rewind.clicked.connect(self.playback_previous)
        self.replay_rewind.clicked.connect(lambda: self.start_playback(1.0))
        self.next_rewind.clicked.connect(self.playback_next)
        self.rewind_controls.setVisible(False)
        self.layout.addWidget(self.rewind_controls)

        selectable_text_flags = Qt.TextInteractionFlag(
            Qt.TextInteractionFlag.TextSelectableByMouse.value
            | Qt.TextInteractionFlag.TextSelectableByKeyboard.value
        )

        self.overlay = QLabel()
        self.overlay.setObjectName("overlay")
        self.overlay.setWordWrap(True)
        self.overlay.setTextInteractionFlags(selectable_text_flags)

        self.inspector = NeuronInspectorLabel(
            "Click a visual neuron to inspect its mapped visual data."
        )
        self.inspector.setObjectName("muted")
        self.inspector.setTextInteractionFlags(selectable_text_flags)
        self.inspector.neuronNumberClicked.connect(self._prompt_for_neuron)

        self.silence = QPushButton("Silence selected neuron")
        self.silence.clicked.connect(self._toggle_selected_node)
        self.silence.setEnabled(False)

        self.importance = QDoubleSpinBox()
        self.importance.setRange(0, 3)
        self.importance.setSingleStep(.1)
        self.importance.setValue(1)
        self.importance.valueChanged.connect(self._change_importance)
        self.importance.setEnabled(False)

        inspect_controls = QHBoxLayout()
        inspect_controls.addWidget(self.silence)
        inspect_controls.addWidget(importance_label)
        inspect_controls.addWidget(self.importance)
        inspect_controls.addStretch(1)

        self.layout.addWidget(self.overlay)
        self.layout.addWidget(self.inspector)
        self.layout.addLayout(inspect_controls)

        self._set_neuron_borders(self.neuron_borders.isChecked())
        self._refresh_overlay()

    def _rebuild(self, quality: str) -> None:
        self.analyzer.save_model()
        self.begin_recording()
        old = self.renderer
        index = self.layout.indexOf(old)
        self.layout.takeAt(index)
        old.setParent(None)
        old.deleteLater()
        self._build_graph(quality)
        if hasattr(self, "region_selector"):
            self._populate_region_selector()
            self._apply_view_mode()
        if hasattr(self, "neuron_borders"):
            self.renderer.set_neuron_borders(self.neuron_borders.isChecked(), self.border_width.value())
        self.layout.insertWidget(index, self.renderer, 1)
        self._refresh_overlay()

    def _set_settings_visible(self, visible: bool) -> None:
        self.settings_panel.setVisible(visible)
        self.settings_toggle.setText("Hide view settings" if visible else "View settings")

    def _edit_colours(self) -> None:
        dialog = ColourSettingsDialog(self._colours, self)
        if not dialog.exec():
            return
        self._colours = dialog.colours
        save_colours(self._colours)
        self.renderer.set_background_colour(self._colours["renderer_background"])
        self.coloursChanged.emit(self._colours.copy())

    def _populate_region_selector(self) -> None:
        selected = self.region_selector.currentData()
        self.region_selector.blockSignals(True)
        self.region_selector.clear()
        for index, name in enumerate(self.graph.region_names):
            self.region_selector.addItem(name, index)
        restored = self.region_selector.findData(selected)
        self.region_selector.setCurrentIndex(restored if restored >= 0 else 0)
        self.region_selector.blockSignals(False)

    def _set_view_mode(self, two_dimensional: bool) -> None:
        self.view_toggle.setText("2D View" if two_dimensional else "3D View")
        self.two_d_mode.setVisible(two_dimensional)
        self.region_selector.setVisible(two_dimensional and self.two_d_mode.currentData() == "sector")
        self._apply_view_mode()

    def _apply_view_mode(self, *_: object) -> None:
        two_dimensional = self.view_toggle.isChecked()
        sector = two_dimensional and self.two_d_mode.currentData() == "sector"
        self.region_selector.setVisible(sector)
        selected = self.region_selector.currentData()
        region = int(selected) if sector and selected is not None else None
        self.renderer.set_projection_mode("2d" if two_dimensional else "3d", region)
        self.spacing.setEnabled(not two_dimensional)
        self.spacing.setToolTip(
            "Cluster spacing is unavailable in 2D because the flat projection uses a fixed readable layout."
            if two_dimensional else "Cluster spacing")
        self._refresh_overlay()

    def apply_frame(self, frame: ActivationFrame, *, record: bool = True) -> None:
        self.mode.setText(frame.source.value.upper())
        self.mapper.apply(frame)
        if record:
            signal = PlaybackStep(frame, self.field.values.copy(), self.field.peaks.copy())
            self._playback.append(signal)
            self._all_signals.append(signal)
            self.analyzer.observe(frame, self.field.values)
            self._analysis_retained_bytes += self._signal_bytes(signal)
            self._trim_analysis_memory()
        self._refresh_overlay()

    def begin_recording(self) -> None:
        self._stop_playback()
        self._playback.clear()
        self._all_signals.clear()
        self._playback_index = -1
        self.analyzer.records.clear()
        self._analysis_memory_exceeded = False
        self._analysis_retained_bytes = 0
        self.analysisMemoryExceeded.emit(False)

    def set_analysis_memory_limit(self, megabytes: int | None) -> None:
        self._analysis_memory_limit_bytes = None if megabytes is None else max(16, megabytes) * 1024 * 1024

    @property
    def analysis_memory_exceeded(self) -> bool:
        return self._analysis_memory_exceeded

    @staticmethod
    def _signal_bytes(signal: PlaybackStep) -> int:
        return signal.values.nbytes + signal.peaks.nbytes + len(signal.frame.token_text.encode("utf-8")) + 96

    def _trim_analysis_memory(self) -> None:
        if self._analysis_memory_limit_bytes is None:
            return
        while self._all_signals and self._analysis_retained_bytes > self._analysis_memory_limit_bytes:
            oldest = self._all_signals.pop(0)
            self._analysis_retained_bytes -= self._signal_bytes(oldest)
            if self.analyzer.records:
                self.analyzer.records.pop(0)
            if not self._analysis_memory_exceeded:
                self._analysis_memory_exceeded = True
                self.analysisMemoryExceeded.emit(True)

    def begin_response_recording(self) -> None:
        """Start a new replay while retaining session-wide NN Analysis+ data."""
        self._stop_playback()
        self._playback.clear()
        self._playback_index = -1

    def set_conversation(self, conversation: list[dict[str, object]]) -> None:
        self._conversation = [dict(turn) for turn in conversation]

    def start_playback(self, speed: float) -> None:
        if not self._playback:
            return
        if self._replay_active:
            self._stop_playback()
            return
        self._playback_index = -1
        self._replay_active = True
        self.renderer.paused = True
        self.previous_rewind.setEnabled(False)
        self.next_rewind.setEnabled(False)
        self.replay_rewind.setText("Stop replay")
        self.playbackChanged.emit(True)
        self._playback_timer.start(max(35, round(110 / max(.1, speed))))

    def enter_rewind_mode(self) -> None:
        if not self._playback or self._rewind_active:
            return
        self._rewind_active = True
        self.renderer.paused = True
        self.rewind_controls.setVisible(True)
        self._show_playback_step(0)
        self.rewindChanged.emit(True)

    def exit_rewind_mode(self) -> None:
        if not self._rewind_active:
            return
        self._stop_playback()
        self._rewind_active = False
        self.renderer.paused = False
        self.rewind_controls.setVisible(False)
        self.rewindChanged.emit(False)

    def playback_next(self) -> None:
        self._stop_playback()
        self.renderer.paused = True
        self._show_playback_step(min(self._playback_index + 1, len(self._playback) - 1))

    def playback_previous(self) -> None:
        self._stop_playback()
        self.renderer.paused = True
        self._show_playback_step(max(0, self._playback_index - 1))

    def _advance_playback(self) -> None:
        if self._playback_index >= len(self._playback) - 1:
            self._stop_playback()
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
            self.inspector.setText(
                "Replay discarded because the connectome graph changed. Generate a new response to record it.")
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

    def _stop_playback(self) -> None:
        """Return rewind controls to manual inspection without stale flashes."""
        self._playback_timer.stop()
        if not self._replay_active:
            return
        self._replay_active = False
        self.previous_rewind.setEnabled(True)
        self.next_rewind.setEnabled(True)
        self.replay_rewind.setText("Replay")
        self.field.values.fill(0.0)
        self.field.active_count_cached = 0
        self.renderer.update()
        self.playbackChanged.emit(False)

    def set_model(self, key: str) -> None:
        self.begin_recording()
        self._model_key = key
        self._rebuild(self.quality.currentText())

    def _refresh_overlay(self) -> None:
        active_edges = int(
            sum((self.field.values[self.graph.edges[:, 0]] > .1) | (self.field.values[self.graph.edges[:, 1]] > .1)))
        strongest = int(self.field.values.argmax()) if len(self.field.values) else 0
        region = self.graph.region_names[int(self.graph.regions[strongest])]
        backend = getattr(self, "_backend", "ModernGL GPU renderer initializing…")
        self.overlay.setText(
            f"{backend}"
            f"  ·  Visualization: Simulation (token-driven)"
            f"  ·  View: {'2D sector' if self.renderer.region_filter is not None else self.renderer.view_mode.upper()}"
            f"  ·  Step {self.field.step}"
            f"  ·  Active neurons {self.field.active_count:,}"
            f"  ·  Active pathways {active_edges:,}"
            f"  ·  Strongest: {region}"
            f"  ·  Token: {self.field.current_token!r}")

    def _set_backend(self, backend: str) -> None:
        self._backend = backend
        self._refresh_overlay()

    def _inspect(self, index: int) -> None:
        self._selected_node = index
        self.silence.setEnabled(True)
        self.importance.setEnabled(True)
        self.importance.setValue(float(self.field.importance[index]))
        self.inspector.set_neuron_details(
            index,
            f"  ·  Region: {self.graph.region_names[int(self.graph.regions[index])]}"
            f"  ·  Current activity: {self.field.values[index]:.3f}"
            f"  ·  Peak: {self.field.peaks[index]:.3f}"
            f"  ·  Data source: Simulation"
        )

    def _prompt_for_neuron(self) -> None:
        current = self._selected_node if self._selected_node is not None else 0
        index, accepted = QInputDialog.getInt(self, "Select visual neuron", "Neuron number:", current, 0,
                                              len(self.graph.positions) - 1)
        if accepted:
            self._inspect(index)

    def _toggle_selected_node(self) -> None:
        if self._selected_node is None:
            return
        index = self._selected_node
        disabled = not bool(self.field.disabled[index])
        self.field.set_disabled(index, disabled)
        self.silence.setText("Restore selected neuron" if disabled else "Silence selected neuron")
        self.renderer.update()

    def _change_importance(self, value: float) -> None:
        if self._selected_node is not None:
            self.field.set_importance(self._selected_node, value)

    def _preview_cluster_spacing(self, value: int) -> None:
        self._pending_cluster_spacing = value / 100
        self.spacing_value.setText(f"{self._pending_cluster_spacing:.2f}×")
        self._spacing_timer.start()

    def _commit_cluster_spacing(self) -> None:
        if abs(self._pending_cluster_spacing - self._cluster_spacing) < .001:
            return
        self._cluster_spacing = self._pending_cluster_spacing
        self._rebuild(self.quality.currentText())

    def _set_neuron_borders(self, visible: bool) -> None:
        QSettings().setValue("neuron_borders", visible)
        QSettings().setValue("neuron_border_width", self.border_width.value())
        if hasattr(self, "renderer"):
            self.renderer.set_neuron_borders(visible, self.border_width.value())

    def run_nn_analysis_plus(self) -> None:
        if not self.analyzer.records:
            QMessageBox.information(
                self,
                "NN Analysis+",
                "Generate a response or start an infinite simulation before creating an analysis file."
            )
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Create NN Analysis+ data file",
            "nn-analysis-plus.json",
            "JSON data (*.json);;Compressed JSON data (*.json.gz)")
        if not filename:
            return
        try:
            export_nn_analysis_plus(Path(filename), self.graph, self.analyzer, self._conversation)
        except OSError as exc:
            QMessageBox.critical(self, "NN Analysis+ export failed", str(exc))
            return
        QMessageBox.information(
            self,
            "NN Analysis+ complete",
            f"Created {Path(filename).name}\n\n"
            f"Conversation turns: {len(self._conversation)}\n"
            f"Visual brain-signal frames analyzed: {len(self.analyzer.records)}\n\n"
            f"The JSON contains compact neural-network findings, not a raw frame dump.")

    def run_session_analysis(self) -> None:
        if not self.analyzer.records:
            QMessageBox.information(self, "Analysis", "Generate a normal chat response before exporting session data.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Create session analysis data file", "session-analysis.json",
                                                  "JSON data (*.json);;Compressed JSON data (*.json.gz)")
        if filename:
            export_session_analysis(Path(filename), self.graph, self.analyzer, self._conversation)

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
            self._backend = "NVIDIA high-performance GPU requested before launch; verifying OpenGL context…"

    def _set_render_preference(self, _index: int) -> None:
        identifier = str(self.render_gpu.currentData())
        QSettings().setValue("render_adapter", identifier)
        if identifier == "system":
            set_windows_gpu_preference(False)
            self._backend = "ModernGL GPU · Windows system-default adapter"
            self._refresh_overlay()
            return
        applied = set_windows_gpu_preference(True)
        self._backend = "High-performance GPU saved; AIBrain will relaunch on the next start" if applied else \
            "GPU preference could not be saved; configure Windows Graphics Settings"
        self._refresh_overlay()
        QMessageBox.information(
            self,
            "Rendering adapter preference",
            "AIBrain writes the Windows high-performance setting "
            "for both virtual-environment Python hosts and starts the app in a fresh process.\n"
            "Restart AIBrain for Windows to apply it.\n"
            "If the overlay still reports another adapter, "
            "choose that executable in Windows Settings > System > Display > Graphics.\n"
            "The overlay always reports the actual renderer.",
        )
