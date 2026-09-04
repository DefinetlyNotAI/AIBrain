from __future__ import annotations

import html
import re

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPlainTextEdit,
                               QPushButton, QScrollArea, QToolButton, QVBoxLayout, QWidget)

from .advanced_settings import AdvancedSettingsDialog
from ..models.infinite_simulation import random_world_opening
from ..models.llama_backend import GenerationConfig
from ..models.model_info import ModelInfo

_SELECTABLE_TEXT_FLAGS = Qt.TextInteractionFlag(
    Qt.TextInteractionFlag.TextSelectableByMouse.value
    | Qt.TextInteractionFlag.TextSelectableByKeyboard.value
)


def _fixed_scroll_content(content: QWidget, height: int) -> QScrollArea:
    """Keep expanded control groups bounded while retaining every control."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setFixedHeight(height)
    scroll.setWidget(content)
    return scroll


def markdown_to_html(markdown: str) -> str:
    """Render the useful Markdown subset safely inside selectable Qt labels."""
    # Keep apostrophes as ordinary text. Qt's rich-text parser can otherwise
    # show the escaped numeric entity literally in streamed conversation text.
    escaped = html.escape(html.unescape(markdown), quote=False)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    escaped = re.sub(r"\[([^]]+)]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', escaped)
    escaped = re.sub(r"(?m)^### (.+)$", r"<h3>\1</h3>", escaped)
    escaped = re.sub(r"(?m)^## (.+)$", r"<h2>\1</h2>", escaped)
    escaped = re.sub(r"(?m)^# (.+)$", r"<h1>\1</h1>", escaped)
    return escaped.replace("\n", "<br>")


class MarkdownLabel(QLabel):
    """A selectable chat label that displays Markdown without losing source text."""

    def __init__(self, text: str = "") -> None:
        self._markdown = ""
        super().__init__()
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(32)
        self._render_timer.timeout.connect(self._flush_markdown)
        self.setText(text)

    def setText(self, text: str) -> None:  # type: ignore[override]
        self._markdown = text
        if hasattr(self, "_render_timer"):
            self._render_timer.stop()
        self._flush_markdown()

    def text(self) -> str:  # type: ignore[override]
        return self._markdown

    def append_markdown(self, text: str) -> None:
        self._markdown += text
        if not self._render_timer.isActive():
            self._render_timer.start()

    def _flush_markdown(self) -> None:
        """Coalesce streamed chunks into one rich-text layout per frame."""
        super().setText(markdown_to_html(self._markdown))


class ChatPanel(QWidget):
    sendRequested = Signal(str)
    stopRequested = Signal()
    regenerateRequested = Signal()
    clearRequested = Signal()
    modelChanged = Signal(object)
    playbackRequested = Signal()
    playbackPreviousRequested = Signal()
    playbackNextRequested = Signal()
    infiniteRequested = Signal(str)
    infiniteContinueRequested = Signal()
    diagnosticsRequested = Signal()
    analysisRequested = Signal()
    chatExportRequested = Signal()
    modeChanged = Signal(bool)
    rewindRequested = Signal()
    rewindExitRequested = Signal()

    def __init__(self, config: GenerationConfig) -> None:
        super().__init__()
        self._analysis_available = False
        self._model_available = False
        self._infinite_mode = False
        self._running = False
        self._rewind_active = False
        self._replay_active = False
        self._regenerate_available = False
        self._rewind_available = False
        self._chat_export_available = False
        self._scroll_timer = QTimer(self)
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.setInterval(32)
        self._scroll_timer.timeout.connect(self._scroll_to_bottom_if_following)
        self._build(config)
        self._playback_controls: QWidget | None = None

    def _build(self, config: GenerationConfig) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        runtime_card = QFrame()
        runtime_card.setObjectName("runtimeCard")
        runtime_layout = QVBoxLayout(runtime_card)
        runtime_layout.setContentsMargins(10, 8, 10, 8)
        title = QLabel("AIBrain")
        title.setObjectName("title")
        runtime_layout.addWidget(title)
        subtitle = QLabel("Local GGUF chat · direct model loading")
        subtitle.setObjectName("muted")
        runtime_layout.addWidget(subtitle)
        self.models = QComboBox()
        self.models.setAccessibleName("Validated local GGUF models")
        self.models.currentIndexChanged.connect(lambda _: self.modelChanged.emit(self.models.currentData()))
        self.diagnostics = QPushButton("Repair and Diagnostics")
        self.diagnostics.setToolTip(
            "Inspect invalid Ollama manifests, repair a selected model, or remove a stale manifest")
        self.diagnostics.clicked.connect(self.diagnosticsRequested)
        model_row = QHBoxLayout()
        model_row.addWidget(self.models, 1)
        model_row.addWidget(self.diagnostics)
        runtime_layout.addLayout(model_row)
        layout.addWidget(runtime_card)

        conversation_heading = QLabel("Conversation")
        conversation_heading.setObjectName("mode")
        layout.addWidget(conversation_heading)
        self.messages_layout = QVBoxLayout()
        self.messages_layout.addStretch(1)
        container = QWidget()
        container.setLayout(self.messages_layout)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(container)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._follow_output = True
        self.scroll.verticalScrollBar().valueChanged.connect(self._refresh_follow_output)
        layout.addWidget(self.scroll, 1)
        self.stats = QLabel("Ready · select an installed GGUF model")
        self.stats.setObjectName("muted")
        self.stats.setWordWrap(True)
        self.stats.setTextInteractionFlags(_SELECTABLE_TEXT_FLAGS)
        layout.addWidget(self.stats)
        compose_heading = QLabel("Compose")
        compose_heading.setObjectName("mode")
        layout.addWidget(compose_heading)
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Message your local model…  (Ctrl+Enter to send)")
        self.input.setFixedHeight(92)
        self.input.textChanged.connect(self._refresh_actions)
        compose_row = QFrame()
        compose_row.setObjectName("composeCard")
        compose_layout = QHBoxLayout(compose_row)
        compose_layout.setContentsMargins(0, 0, 0, 0)
        compose_layout.addWidget(self.input, 1)
        self.send = QToolButton()
        self.send.setObjectName("sendButton")
        self.send.setText("➤")
        self.send.setAccessibleName("Start generation")
        self.send.setToolTip("Send the compose text; changes to Stop while generation is running")
        self.send.clicked.connect(self._start_or_stop)
        compose_layout.addWidget(self.send, 0, Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(compose_row)
        action_card = QFrame()
        action_card.setObjectName("actionCard")
        action_layout = QVBoxLayout(action_card)
        action_layout.setContentsMargins(10, 8, 10, 8)
        self.mode_actions_toggle = QToolButton()
        self.mode_actions_toggle.setText("Hide Mode Actions")
        self.mode_actions_toggle.setCheckable(True)
        self.mode_actions_toggle.setChecked(True)
        self.mode_actions_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.mode_actions_toggle.setArrowType(Qt.ArrowType.DownArrow)
        self.mode_actions_toggle.setToolTip("Show or hide mode and conversation actions")
        self.mode_actions_toggle.toggled.connect(self._set_mode_actions_visible)
        action_layout.addWidget(self.mode_actions_toggle)
        mode_actions_body = QWidget()
        mode_actions_layout = QVBoxLayout(mode_actions_body)
        mode_actions_layout.setContentsMargins(0, 0, 0, 0)
        mode_row = QHBoxLayout()
        self.infinite_mode = QCheckBox("Infinite Mode")
        self.infinite_mode.setToolTip("Switch modes and clear the current conversation and replay")
        self.infinite_mode.toggled.connect(self._set_mode)
        mode_row.addWidget(self.infinite_mode)
        mode_row.addStretch(1)
        mode_actions_layout.addLayout(mode_row)
        buttons = QGridLayout()
        self.regenerate = QPushButton("Regenerate")
        self.rewind = QPushButton("Rewind")
        self.rewind.setToolTip("Enter frame-by-frame connectome replay below the graph")
        self.infinite = QPushButton("Continue")
        self.infinite.setToolTip("Continue the current Infinite Mode scenario after it has stopped")
        self.clear = QPushButton("Clear")
        self.open_analysis = QPushButton("Analysis")
        self.open_analysis.setToolTip("Export NN Analysis+ for recorded frames from the selected model")
        self.open_analysis.setEnabled(False)
        self.export_chat = QPushButton("Export chat")
        self.export_chat.setToolTip(
            "Save the complete conversation as JSON or readable text"
        )
        self.regenerate.clicked.connect(self.regenerateRequested)
        self.rewind.clicked.connect(self._toggle_rewind)
        self.infinite.clicked.connect(self._continue_infinite)
        self.clear.clicked.connect(self.clearRequested)
        self.open_analysis.clicked.connect(self.analysisRequested)
        self.export_chat.clicked.connect(self.chatExportRequested)
        for index, button in enumerate(
            (
                self.regenerate,
                self.rewind,
                self.open_analysis,
                self.export_chat,
                self.clear,
                self.infinite,
            )
        ):
            buttons.addWidget(button, index // 3, index % 3)
        mode_actions_layout.addLayout(buttons)
        self.mode_actions_content = _fixed_scroll_content(mode_actions_body, 124)
        action_layout.addWidget(self.mode_actions_content)
        layout.addWidget(action_card)

        self.advanced_dialog = AdvancedSettingsDialog(config, self)
        self.temperature = self.advanced_dialog.temperature
        self.top_p = self.advanced_dialog.top_p
        self.max_tokens = self.advanced_dialog.max_tokens
        self.context = self.advanced_dialog.context
        self.gpu_layers = self.advanced_dialog.gpu_layers
        self.analysis_memory_mb = self.advanced_dialog.analysis_memory_mb
        self.analysis_cache_mb = self.advanced_dialog.analysis_cache_mb
        self.speed = self.advanced_dialog.speed
        self.speed_value = self.advanced_dialog.speed_value
        self.advanced_settings = QPushButton("Advanced settings")
        self.advanced_settings.setToolTip("Open generation, context, GPU, replay, and Analysis+ settings")
        self.advanced_settings.clicked.connect(self._open_advanced_settings)
        self.regenerate.setToolTip("Generate a new answer for the most recent normal-chat prompt")
        self.clear.setToolTip("Clear the current conversation and recorded replay")
        self.models.setToolTip("Select an available local GGUF model")
        layout.addWidget(self.advanced_settings)
        self._refresh_actions()

    def _send(self) -> None:
        text = self.input.toPlainText().strip()
        if text:
            self.input.clear()
            self.sendRequested.emit(text)

    def _start_or_stop(self) -> None:
        if self._running:
            self.stopRequested.emit()
            return
        if self._infinite_mode:
            self._start_infinite()
        else:
            self._send()

    def _start_infinite(self) -> None:
        seed = self.input.toPlainText().strip() or random_world_opening()
        self.input.clear()
        self.infiniteRequested.emit(seed)

    def _continue_infinite(self) -> None:
        self.infiniteContinueRequested.emit()

    def _toggle_rewind(self) -> None:
        if self._rewind_active:
            self.rewindExitRequested.emit()
        else:
            self.rewindRequested.emit()

    def _open_advanced_settings(self) -> None:
        self.advanced_dialog.exec()

    def _set_mode_actions_visible(self, visible: bool) -> None:
        self.mode_actions_content.setVisible(visible)
        if visible:
            scrollbar = self.mode_actions_content.verticalScrollBar()
            scrollbar.setValue(scrollbar.minimum())
        self.mode_actions_toggle.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)
        self.mode_actions_toggle.setText("Hide Mode Actions" if visible else "Show Mode Actions")

    def _set_mode(self, infinite: bool) -> None:
        if self._infinite_mode == infinite:
            return
        self._infinite_mode = infinite
        self.infinite_mode.blockSignals(True)
        self.infinite_mode.setChecked(infinite)
        self.infinite_mode.blockSignals(False)
        self.input.setPlaceholderText(
            "Describe the first World event, or use Random…"
            if infinite else "Message your local model…  (Ctrl+Enter to send)")
        # Keep every action in the same physical slot across modes. Unavailable
        # actions are disabled instead of disappearing and shifting the UI.
        self.infinite.setEnabled(infinite)
        self.open_analysis.setText("Analysis+" if infinite else "Analysis")
        self.clear_messages()
        self.clear_latest_playback()
        self.modeChanged.emit(infinite)
        self._refresh_actions()

    def config(self) -> GenerationConfig:
        return self.advanced_dialog.config()

    def set_models(self, models: list[ModelInfo]) -> None:
        self.models.blockSignals(True)
        self.models.clear()
        available = [model for model in models if model.available]
        self.models.addItem("No model selected", None)
        for model in available:
            self.models.addItem(model.label, model)
        self.models.blockSignals(False)
        self.models.setEnabled(bool(available))
        self._model_available = False
        self._analysis_available = False
        self.open_analysis.setEnabled(False)
        self._refresh_actions()

    def set_analysis_available(self, available: bool) -> None:
        self._analysis_available = available
        self._refresh_actions()

    def set_regenerate_available(self, available: bool) -> None:
        """Enable regeneration only when a completed normal response exists."""
        self._regenerate_available = available
        self._refresh_actions()

    def set_rewind_available(self, available: bool) -> None:
        """Enable rewind only after the visualizer has recorded telemetry frames."""
        self._rewind_available = available
        self._refresh_actions()

    def set_chat_export_available(self, available: bool) -> None:
        self._chat_export_available = available
        self._refresh_actions()

    def set_validating_models(self, text: str) -> None:
        self.models.blockSignals(True)
        self.models.clear()
        self.models.addItem(text, None)
        self.models.setEnabled(False)
        self.models.blockSignals(False)

    def add_message(self, role: str, text: str) -> QLabel:
        follow_output = self._follow_output
        bubble = MarkdownLabel(text)
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(_SELECTABLE_TEXT_FLAGS)
        bubble.setObjectName(
            "userBubble" if role == "user" else "worldBubble" if role == "world" else "assistantBubble")
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, bubble)
        self._schedule_scroll_to_bottom(follow_output)
        return bubble

    @staticmethod
    def _is_at_bottom(value: int, maximum: int) -> bool:
        """Allow a two-pixel tolerance for scrollbar rounding and style margins."""
        return maximum - value <= 2

    def _refresh_follow_output(self, value: int) -> None:
        scrollbar = self.scroll.verticalScrollBar()
        self._follow_output = self._is_at_bottom(value, scrollbar.maximum())

    def append_message_text(self, bubble: QLabel, text: str) -> None:
        """Append generated text without stealing the user's reading position."""
        follow_output = self._follow_output
        if isinstance(bubble, MarkdownLabel):
            bubble.append_markdown(text)
        else:
            bubble.setText(bubble.text() + text)
        self._schedule_scroll_to_bottom(follow_output)

    def _schedule_scroll_to_bottom(self, follow_output: bool) -> None:
        if not follow_output:
            return
        if not self._scroll_timer.isActive():
            self._scroll_timer.start()

    def _scroll_to_bottom_if_following(self) -> None:
        if self._follow_output:
            scrollbar = self.scroll.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def clear_latest_playback(self) -> None:
        if self._playback_controls is None:
            return
        self.messages_layout.removeWidget(self._playback_controls)
        self._playback_controls.deleteLater()
        self._playback_controls = None

    def show_latest_playback(self) -> None:
        self.clear_latest_playback()
        controls = QWidget()
        layout = QHBoxLayout(controls)
        layout.setContentsMargins(4, 1, 4, 5)
        previous = QPushButton("◀ Frame")
        replay = QPushButton("▶ Replay signals")
        next_token = QPushButton("Frame ▶")
        previous.clicked.connect(self.playbackPreviousRequested)
        replay.clicked.connect(self.playbackRequested)
        next_token.clicked.connect(self.playbackNextRequested)
        layout.addWidget(previous)
        layout.addWidget(replay)
        layout.addWidget(next_token)
        layout.addStretch(1)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, controls)
        self._playback_controls = controls

    def clear_messages(self) -> None:
        self._playback_controls = None

        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            if item is None:
                continue

            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def generating(self, running: bool) -> None:
        self._running = running
        self._refresh_actions()

    def set_analysis_mode(self, infinite: bool) -> None:
        self.open_analysis.setText("Analysis+" if infinite else "Analysis")
        self.open_analysis.setToolTip(
            "Export compact neural findings for an Infinite simulation" if infinite
            else "Export normal chat session data without neural-network findings"
        )

    def set_model_available(self, available: bool) -> None:
        self._model_available = available
        self._refresh_actions()

    def set_rewind_mode(self, active: bool) -> None:
        """The graph owns rewind navigation; chat remains read-only until exit."""
        self._rewind_active = active
        self.rewind.setText("Exit rewind" if active else "Rewind")
        self.rewind.setToolTip("Return to normal chat actions" if active else "Enter frame-by-frame connectome replay")
        if active:
            for control in (self.send, self.regenerate, self.open_analysis, self.export_chat, self.clear, self.infinite,
                            self.models, self.infinite_mode, self.advanced_settings):
                control.setEnabled(False)
            self.rewind.setEnabled(not self._replay_active)
            return
        self._refresh_actions()

    def set_analysis_memory_exceeded(self, exceeded: bool) -> None:
        label = "⚠ Analysis+" if exceeded and self._infinite_mode else "Analysis+" if self._infinite_mode else "Analysis"
        self.open_analysis.setText(label)
        self.open_analysis.setToolTip(
            "Analysis+ cache is full or disabled. Oldest records were discarded; export soon."
            if exceeded else "Export compact neural findings for an Infinite simulation" if self._infinite_mode
            else "Export normal chat session data without neural-network findings"
        )

    def set_replay_mode(self, active: bool) -> None:
        """Do not allow navigation or exit while timed replay owns the graph."""
        self._replay_active = active
        if self._rewind_active:
            self.rewind.setEnabled(not active)

    def _refresh_actions(self) -> None:
        ready = self._model_available
        has_compose_text = bool(self.input.toPlainText().strip())
        if self._running:
            self.send.setText("■")
            self.send.setAccessibleName("Stop generation")
            self.send.setToolTip("Stop the running generation")
        elif self._infinite_mode and not has_compose_text:
            self.send.setText("🎲")
            self.send.setAccessibleName("Start with a random World prompt")
            self.send.setToolTip("Choose a random pre-made World event and start with the Participant response")
        else:
            self.send.setText("➤")
            self.send.setAccessibleName("Start generation")
            self.send.setToolTip(
                "Send this World event and start with the Participant response"
                if self._infinite_mode else "Send the compose text; changes to Stop while generation is running"
            )
        can_start = self._infinite_mode or has_compose_text
        self.send.setEnabled(self._running or (ready and can_start and not self._rewind_active))
        self.regenerate.setEnabled(
            ready and self._regenerate_available and not self._running and not self._infinite_mode)
        self.rewind.setEnabled(
            (self._rewind_active or (ready and self._rewind_available and not self._running))
            and not self._replay_active
        )
        self.infinite.setEnabled(ready and not self._running and self._infinite_mode)
        self.clear.setEnabled(not self._running)
        self.models.setEnabled(not self._running and self.models.count() > 1)
        self.open_analysis.setEnabled(ready and not self._running and self._analysis_available)
        self.export_chat.setEnabled(
            not self._running and self._chat_export_available
        )
        self.infinite_mode.setEnabled(not self._running and not self._rewind_active)
        self.advanced_settings.setEnabled(not self._running and not self._rewind_active)
