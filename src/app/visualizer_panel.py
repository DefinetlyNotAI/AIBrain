from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .theme import ColourSettingsDialog, load_colours, save_colours
from ..connectome.activity import ActivityField
from ..connectome.analysis import AnalysisRecord, ConnectomeAnalyzer
from ..connectome.analysis_cache import AnalysisPageCache
from ..connectome.export import export_nn_analysis_plus, export_session_analysis
from ..connectome.generator import build_connectome
from ..connectome.mapper import ActivityMapper
from ..connectome.renderer import ConnectomeRenderer
from ..models.instrumented_backend import ActivationFrame
from ..models.message_types import TranscriptTurn
from ..utils.gpu import discover_render_adapters, set_windows_gpu_preference


@dataclass(slots=True)
class PlaybackStep:
    frame: ActivationFrame
    values: np.ndarray
    peaks: np.ndarray


def _trim_analysis_memory_data(
        playback: list[PlaybackStep],
        retained_bytes: int,
        memory_limit_bytes: int | None,
        records: list[AnalysisRecord],
        cache: AnalysisPageCache,
) -> tuple[int, bool]:
    """Move old playback and analysis records into the bounded disk cache."""
    if memory_limit_bytes is None:
        return retained_bytes, False
    data_lost = False
    while playback and retained_bytes > memory_limit_bytes:
        oldest = playback.pop(0)
        retained_bytes -= VisualizerPanel._signal_bytes(oldest)
        if records:
            record = records.pop(0)
            overflowed = cache.append(record)
        else:
            overflowed = False
        data_lost = data_lost or overflowed or not cache.enabled
    return retained_bytes, data_lost


class SignalNodeInspectorLabel(QLabel):
    """Selectable inspector text with a click target only over the signal-node number."""

    nodeNumberClicked = Signal()
    _prefix = "Selected signal node: "

    def set_node_details(self, index: int, details: str) -> None:
        self.setText(f"{self._prefix}{index:05d}{details}")

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        prefix_width = self.fontMetrics().horizontalAdvance(self._prefix)
        number_width = self.fontMetrics().horizontalAdvance("00000")
        if (
                event.button() == Qt.MouseButton.LeftButton
                and prefix_width <= event.position().x() <= prefix_width + number_width
        ):
            self.nodeNumberClicked.emit()
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
        self._playback_index = -1
        self._rewind_active = False
        self._replay_active = False
        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._advance_playback)
        self._conversation: list[TranscriptTurn] = []
        self._analysis_memory_limit_bytes: int | None = None
        self._analysis_memory_exceeded = False
        self._analysis_retained_bytes = 0
        self._analysis_cache = AnalysisPageCache()
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
        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(8, 18, 18, 18)

        header = QGridLayout()
        header.setHorizontalSpacing(8)
        header.setVerticalSpacing(6)

        title = QLabel("Real-Time Connectome")
        title.setObjectName("title")

        self.mode = QLabel("REAL-TIME")
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

        self.node_borders = QCheckBox("Signal-node borders")
        self.node_borders.setToolTip(
            "Outline each telemetry display node for clearer separation"
        )

        node_borders_value = settings.value("neuron_borders", True)
        self.node_borders.setChecked(
            node_borders_value if isinstance(node_borders_value, bool) else True
        )

        self.node_borders.toggled.connect(self._set_node_borders)

        self.border_width = QDoubleSpinBox()
        self.border_width.setRange(0, 1)
        self.border_width.setSingleStep(0.01)
        self.border_width.setDecimals(2)

        border_width_value = settings.value("neuron_border_width", 0.12)
        self.border_width.setValue(
            float(border_width_value)
            if isinstance(border_width_value, (int, float, str))
            else 0.12
        )

        self.border_width.setToolTip("Signal-node outline width")
        self.border_width.valueChanged.connect(
            lambda _: self._set_node_borders(self.node_borders.isChecked())
        )

        self.analysis = QPushButton("NN Analysis+")
        self.analysis.setToolTip(
            "Analyze recorded real-time inference telemetry and export compact findings"
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
        self.reset_view.setToolTip(
            "Restore the default 2D or 3D camera position and zoom"
        )
        self.reset_view.clicked.connect(self.renderer.reset_camera)
        self.settings_toggle = QPushButton("View settings")
        self.settings_toggle.setCheckable(True)
        self.settings_toggle.setToolTip(
            "Show or hide rendering and presentation settings"
        )
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

        self.root_layout.addLayout(header)
        settings_body = QWidget()
        settings_layout = QFormLayout(settings_body)
        settings_layout.setContentsMargins(0, 2, 0, 6)
        settings_layout.addRow("Signal-map detail", self.quality)
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
        borders_layout.addWidget(self.node_borders)
        borders_layout.addWidget(self.border_width)
        settings_layout.addRow("Signal-node borders (0–1)", borders_row)
        self.settings_panel = QScrollArea()
        self.settings_panel.setWidgetResizable(True)
        self.settings_panel.setFrameShape(QFrame.Shape.NoFrame)
        self.settings_panel.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.settings_panel.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.settings_panel.setFixedHeight(154)
        self.settings_panel.setWidget(settings_body)
        self.settings_panel.setVisible(False)
        self.root_layout.addWidget(self.settings_panel)
        self.root_layout.addWidget(self.renderer, 1)
        self.rewind_controls = QWidget()
        rewind_layout = QHBoxLayout(self.rewind_controls)
        rewind_layout.setContentsMargins(0, 4, 0, 4)
        self.previous_rewind = QPushButton("Previous")
        self.replay_rewind = QPushButton("Replay")
        self.next_rewind = QPushButton("Next")
        for button, tip in (
                (self.previous_rewind, "Show the previous recorded telemetry frame"),
                (self.replay_rewind, "Play recorded telemetry frames from the beginning"),
                (self.next_rewind, "Show the next recorded telemetry frame"),
        ):
            button.setToolTip(tip)
            rewind_layout.addWidget(button)
        rewind_layout.addStretch(1)
        self.previous_rewind.clicked.connect(self.playback_previous)
        self.replay_rewind.clicked.connect(lambda: self.start_playback(1.0))
        self.next_rewind.clicked.connect(self.playback_next)
        self.rewind_controls.setVisible(False)
        self.root_layout.addWidget(self.rewind_controls)

        selectable_text_flags = Qt.TextInteractionFlag(
            Qt.TextInteractionFlag.TextSelectableByMouse.value
            | Qt.TextInteractionFlag.TextSelectableByKeyboard.value
        )

        self.overlay = QLabel()
        self.overlay.setObjectName("overlay")
        self.overlay.setWordWrap(True)
        self.overlay.setTextInteractionFlags(selectable_text_flags)

        self.inspector = SignalNodeInspectorLabel(
            "Click a signal node to inspect its measured telemetry channel."
        )
        self.inspector.setObjectName("muted")
        self.inspector.setTextInteractionFlags(selectable_text_flags)
        self.inspector.nodeNumberClicked.connect(self._prompt_for_node)

        self.silence = QPushButton("Hide selected signal node")
        self.silence.clicked.connect(self._toggle_selected_node)
        self.silence.setEnabled(False)

        self.importance = QDoubleSpinBox()
        self.importance.setRange(0, 3)
        self.importance.setSingleStep(0.1)
        self.importance.setValue(1)
        self.importance.valueChanged.connect(self._change_importance)
        self.importance.setEnabled(False)

        inspect_controls = QHBoxLayout()
        inspect_controls.addWidget(self.silence)
        inspect_controls.addWidget(importance_label)
        inspect_controls.addWidget(self.importance)
        inspect_controls.addStretch(1)

        self.root_layout.addWidget(self.overlay)
        self.root_layout.addWidget(self.inspector)
        self.root_layout.addLayout(inspect_controls)

        self._set_node_borders(self.node_borders.isChecked())
        self._refresh_overlay()

    def _rebuild(self, quality: str) -> None:
        self.analyzer.save_model()
        self.begin_recording()
        old = self.renderer
        index = self.root_layout.indexOf(old)
        self.root_layout.takeAt(index)
        old.setParent(None)
        old.deleteLater()
        self._build_graph(quality)
        if hasattr(self, "region_selector"):
            self._populate_region_selector()
            self._apply_view_mode()
        if hasattr(self, "node_borders"):
            self.renderer.set_node_borders(
                self.node_borders.isChecked(), self.border_width.value()
            )
        self.root_layout.insertWidget(index, self.renderer, 1)
        self._refresh_overlay()

    def _set_settings_visible(self, visible: bool) -> None:
        self.settings_panel.setVisible(visible)
        if visible:
            scrollbar = self.settings_panel.verticalScrollBar()
            scrollbar.setValue(scrollbar.minimum())
        self.settings_toggle.setText(
            "Hide view settings" if visible else "View settings"
        )

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
        self.region_selector.setCurrentIndex(max(restored, 0))
        self.region_selector.blockSignals(False)

    def _set_view_mode(self, two_dimensional: bool) -> None:
        self.view_toggle.setText("2D View" if two_dimensional else "3D View")
        self.two_d_mode.setVisible(two_dimensional)
        self.region_selector.setVisible(
            two_dimensional and self.two_d_mode.currentData() == "sector"
        )
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
            if two_dimensional
            else "Cluster spacing"
        )
        self._refresh_overlay()

    def apply_frame(self, frame: ActivationFrame, *, record: bool = True) -> None:
        self.mode.setText(frame.source.value.upper())
        self.mapper.apply(frame)
        if record:
            signal = PlaybackStep(
                frame, self.field.values.copy(), self.field.peaks.copy()
            )
            self._playback.append(signal)
            self.analyzer.observe(frame)
            self._analysis_retained_bytes += self._signal_bytes(signal)
            self._trim_analysis_memory()
        self._refresh_overlay()

    def begin_recording(self) -> None:
        self._stop_playback()
        self._playback.clear()
        self._playback_index = -1
        self.analyzer.records.clear()
        self._analysis_cache.cleanup()
        self._analysis_memory_exceeded = False
        self._analysis_retained_bytes = 0
        self.analysisMemoryExceeded.emit(False)

    def set_analysis_memory_limit(self, megabytes: int | None) -> None:
        self._analysis_memory_limit_bytes = (
            None if megabytes is None else max(16, megabytes) * 1024 * 1024
        )

    def set_analysis_cache_limit(self, megabytes: int) -> None:
        had_cached_records = self._analysis_cache.record_count > 0
        self._analysis_cache.set_limit(max(0, megabytes) * 1024 * 1024)
        data_lost = self._analysis_cache.overflowed or (
                megabytes == 0 and had_cached_records
        )
        if data_lost and not self._analysis_memory_exceeded:
            self._analysis_memory_exceeded = True
            self.analysisMemoryExceeded.emit(True)

    @property
    def analysis_memory_exceeded(self) -> bool:
        return self._analysis_memory_exceeded

    @property
    def has_recorded_frames(self) -> bool:
        return bool(self._playback)

    @property
    def has_analysis_records(self) -> bool:
        return bool(self.analyzer.records) or self._analysis_cache.record_count > 0

    def iter_analysis_records(self) -> Iterator[AnalysisRecord]:
        yield from self._analysis_cache.iter_records()
        yield from self.analyzer.records

    def cleanup_analysis_cache(self) -> None:
        self._analysis_cache.cleanup()

    @staticmethod
    def _signal_bytes(signal: PlaybackStep) -> int:
        return (
                signal.values.nbytes
                + signal.peaks.nbytes
                + len(signal.frame.chunk_text.encode("utf-8"))
                + 96
        )

    def _trim_analysis_memory(self) -> None:
        self._analysis_retained_bytes, data_lost = _trim_analysis_memory_data(
            self._playback,
            self._analysis_retained_bytes,
            self._analysis_memory_limit_bytes,
            self.analyzer.records,
            self._analysis_cache,
        )
        if data_lost and not self._analysis_memory_exceeded:
            self._analysis_memory_exceeded = True
            self.analysisMemoryExceeded.emit(True)

    def begin_response_recording(self) -> None:
        """Start a new replay while retaining session-wide NN Analysis+ data."""
        self._stop_playback()
        self._playback.clear()
        self._playback_index = -1
        self._analysis_retained_bytes = 0

    def set_conversation(self, conversation: Sequence[TranscriptTurn]) -> None:
        self._conversation = [turn.copy() for turn in conversation]

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
        self._playback_timer.start(max(35, round(110 / max(0.1, speed))))

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
                "Replay discarded because the connectome graph changed. Generate a new response to record it."
            )
            return
        self._playback_index = index
        self.field.values[:] = step.values
        self.field.peaks[:] = step.peaks
        self.field.step = step.frame.step
        self.field.current_chunk = step.frame.chunk_text
        self.field.current_source = step.frame.source.value
        self.field.telemetry = {
            name: float(value) for name, value in step.frame.metrics.items()
        }
        self.field.channel_values = {
            name: float(np.clip(step.frame.regions.get(name, 0.0), 0.0, 1.0))
            for name in self.graph.region_names
        }
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
            np.count_nonzero(
                (self.field.values[self.graph.edges[:, 0]] > 0.1)
                | (self.field.values[self.graph.edges[:, 1]] > 0.1)
            )
        )
        backend = getattr(self, "_backend", "ModernGL GPU renderer initializing…")
        metrics = self.field.telemetry
        data_text = "awaiting the first generated chunk"
        measurement_text = "  ·  Strongest measurement: awaiting data"
        if metrics:
            strongest_channel = max(
                self.graph.region_names,
                key=lambda name: self.field.channel_values.get(name, 0.0),
            )
            raw_logits_available = bool(metrics.get("raw_logits_available", 0.0))
            data_text = (
                "live raw llama.cpp logits and observed stream timing"
                if raw_logits_available
                else "live stream timing; raw-logit snapshot unavailable"
            )
            raw_logit_text = (
                f"  ·  Raw-logit entropy {metrics.get('raw_logit_entropy_bits', 0.0):.2f} bits"
                f"  ·  Raw top {metrics.get('raw_top_probability', 0.0) * 100:.1f}%"
                if raw_logits_available
                else "  ·  Raw logits unavailable for this chunk"
            )
            measurement_text = (
                f"  ·  Strongest measurement: {strongest_channel}"
                f"{raw_logit_text}"
                f"  ·  Stream latency {metrics.get('stream_latency_ms', 0.0):.1f} ms"
                f"  ·  {metrics.get('retokenized_tokens_per_second', 0.0):.1f} retokenized tok/s"
            )
        self.overlay.setText(
            f"{backend}"
            f"  ·  Data: {data_text}"
            f"  ·  View: {'2D sector' if self.renderer.region_filter is not None else self.renderer.view_mode.upper()}"
            f"  ·  Step {self.field.step}"
            f"  ·  Lit display nodes {self.field.active_count:,}"
            f"  ·  Lit display links {active_edges:,}"
            f"  ·  Output chunk: {self.field.current_chunk!r}"
            f"{measurement_text}"
        )

    def _set_backend(self, backend: str) -> None:
        self._backend = backend
        self._refresh_overlay()

    def _inspect(self, index: int) -> None:
        self._selected_node = index
        self.silence.setEnabled(True)
        self.importance.setEnabled(True)
        self.importance.setValue(float(self.field.importance[index]))
        channel = self.graph.region_names[int(self.graph.regions[index])]
        self.inspector.set_node_details(
            index,
            f"  ·  Channel: {channel}"
            f"  ·  Normalized measurement: {self.field.channel_values.get(channel, 0.0):.3f}"
            f"  ·  Display intensity: {self.field.values[index]:.3f}"
            f"  ·  Display peak: {self.field.peaks[index]:.3f}"
            f"  ·  Data source: {self.field.current_source} llama.cpp telemetry",
        )

    def _prompt_for_node(self) -> None:
        current = self._selected_node if self._selected_node is not None else 0
        index, accepted = QInputDialog.getInt(
            self,
            "Select signal node",
            "Signal-node number:",
            current,
            0,
            len(self.graph.positions) - 1,
        )
        if accepted:
            self._inspect(index)

    def _toggle_selected_node(self) -> None:
        if self._selected_node is None:
            return
        index = self._selected_node
        disabled = not bool(self.field.disabled[index])
        self.field.set_disabled(index, disabled)
        self.silence.setText(
            "Show selected signal node" if disabled else "Hide selected signal node"
        )
        self.renderer.update()

    def _change_importance(self, value: float) -> None:
        if self._selected_node is not None:
            self.field.set_importance(self._selected_node, value)

    def _preview_cluster_spacing(self, value: int) -> None:
        self._pending_cluster_spacing = value / 100
        self.spacing_value.setText(f"{self._pending_cluster_spacing:.2f}×")
        self._spacing_timer.start()

    def _commit_cluster_spacing(self) -> None:
        if abs(self._pending_cluster_spacing - self._cluster_spacing) < 0.001:
            return
        self._cluster_spacing = self._pending_cluster_spacing
        self._rebuild(self.quality.currentText())

    def _set_node_borders(self, visible: bool) -> None:
        QSettings().setValue("neuron_borders", visible)
        QSettings().setValue("neuron_border_width", self.border_width.value())
        if hasattr(self, "renderer"):
            self.renderer.set_node_borders(visible, self.border_width.value())

    def run_nn_analysis_plus(self) -> None:
        if not self.has_analysis_records:
            QMessageBox.information(
                self,
                "NN Analysis+",
                "Generate a response or start an infinite simulation before creating an analysis file.",
            )
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Create NN Analysis+ data file",
            "nn-analysis-plus.json",
            "JSON data (*.json);;Compressed JSON data (*.json.gz)",
        )
        if not filename:
            return
        cache_status = self._analysis_cache.status(len(self.analyzer.records))
        frame_count = int(cache_status["paged_records"]) + int(
            cache_status["resident_records"]
        )
        try:
            export_nn_analysis_plus(
                Path(filename),
                self.graph,
                self.analyzer,
                self._conversation,
                records=self.iter_analysis_records,
                cache_status=cache_status,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "NN Analysis+ export failed", str(exc))
            return
        self._analysis_cache.cleanup()
        QMessageBox.information(
            self,
            "NN Analysis+ complete",
            f"Created {Path(filename).name}\n\n"
            f"Conversation turns: {len(self._conversation)}\n"
            f"Real-time inference frames analyzed: {frame_count}\n\n"
            f"The JSON contains compact telemetry findings, not a raw frame dump.",
        )

    def run_session_analysis(self) -> None:
        if not self.analyzer.records:
            QMessageBox.information(
                self,
                "Analysis",
                "Generate a normal chat response before exporting session data.",
            )
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Create session analysis data file",
            "session-analysis.json",
            "JSON data (*.json);;Compressed JSON data (*.json.gz)",
        )
        if filename:
            export_session_analysis(
                Path(filename), self.graph, self.analyzer, self._conversation
            )

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
        self.render_gpu.setCurrentIndex(max(index, 0))
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
        self._backend = (
            "High-performance GPU saved; AIBrain will relaunch on the next start"
            if applied
            else "GPU preference could not be saved; configure Windows Graphics Settings"
        )
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
