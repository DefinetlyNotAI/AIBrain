from __future__ import annotations

import html
import re

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel,
                               QPlainTextEdit, QPushButton, QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget)

from ..models.llama_backend import GenerationConfig
from ..models.model_info import ModelInfo

_SELECTABLE_TEXT_FLAGS = Qt.TextInteractionFlag(
    Qt.TextInteractionFlag.TextSelectableByMouse.value
    | Qt.TextInteractionFlag.TextSelectableByKeyboard.value
)


def markdown_to_html(markdown: str) -> str:
    """Render the useful Markdown subset safely inside selectable Qt labels."""
    escaped = html.escape(html.unescape(markdown))
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
        self.setText(text)

    def setText(self, text: str) -> None:  # type: ignore[override]
        self._markdown = text
        super().setText(markdown_to_html(text))

    def text(self) -> str:  # type: ignore[override]
        return self._markdown

    def append_markdown(self, text: str) -> None:
        self.setText(self._markdown + text)


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
    diagnosticsRequested = Signal()
    analysisRequested = Signal()

    def __init__(self, config: GenerationConfig) -> None:
        super().__init__()
        self._analysis_available = False
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
        layout.addWidget(self.input)
        action_card = QFrame()
        action_card.setObjectName("actionCard")
        action_layout = QVBoxLayout(action_card)
        action_layout.setContentsMargins(10, 8, 10, 8)
        normal_label = QLabel("CHAT ACTIONS")
        normal_label.setObjectName("muted")
        action_layout.addWidget(normal_label)
        buttons = QHBoxLayout()
        self.send = QPushButton("Send")
        self.stop = QPushButton("Stop")
        self.regenerate = QPushButton("Regenerate")
        self.infinite = QPushButton("∞ Infinite mode")
        self.infinite.setToolTip(
            "Start an open-ended world/participant roleplay simulation using two local model instances")
        self.clear = QPushButton("Clear")
        self.open_analysis = QPushButton("Analysis")
        self.open_analysis.setToolTip("Export NN Analysis+ for recorded frames from the selected model")
        self.open_analysis.setEnabled(False)
        self.stop.setEnabled(False)
        # noinspection DuplicatedCode
        self.send.clicked.connect(self._send)
        self.stop.clicked.connect(self.stopRequested)
        self.regenerate.clicked.connect(self.regenerateRequested)
        self.infinite.clicked.connect(self._start_infinite)
        self.clear.clicked.connect(self.clearRequested)
        self.open_analysis.clicked.connect(self.analysisRequested)
        for button in (self.send, self.stop, self.regenerate, self.open_analysis, self.clear):
            buttons.addWidget(button)
        action_layout.addLayout(buttons)
        infinite_row = QHBoxLayout()
        infinite_label = QLabel("INFINITE-MODE ACTION")
        infinite_label.setObjectName("muted")
        infinite_row.addWidget(infinite_label)
        infinite_row.addWidget(self.infinite)
        infinite_row.addStretch(1)
        action_layout.addLayout(infinite_row)
        layout.addWidget(action_card)

        advanced_group = QGroupBox("Advanced generation controls")
        advanced = QFormLayout(advanced_group)
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0, 2)
        self.temperature.setSingleStep(.05)
        self.temperature.setValue(config.temperature)
        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(.05, 1)
        self.top_p.setSingleStep(.05)
        self.top_p.setValue(config.top_p)
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(1, 8192)
        self.max_tokens.setValue(config.max_tokens)
        self.context = QSpinBox()
        self.context.setRange(512, 32768)
        self.context.setSingleStep(512)
        self.context.setValue(config.context_length)
        self.gpu_layers = QSpinBox()
        self.gpu_layers.setRange(-1, 200)
        self.gpu_layers.setValue(config.gpu_layers)
        self.speed = QSlider()
        self.speed.setOrientation(Qt.Orientation.Horizontal)
        self.speed.setRange(1, 10)
        self.speed.setValue(round(config.speed * 10))
        self.speed_value = QLabel()
        self.speed.valueChanged.connect(self._update_speed_label)
        self._update_speed_label(self.speed.value())
        speed_row = QWidget()
        speed_layout = QHBoxLayout(speed_row)
        speed_layout.setContentsMargins(0, 0, 0, 0)
        speed_layout.addWidget(self.speed)
        speed_layout.addWidget(self.speed_value)
        advanced.addRow("Temperature", self.temperature)
        advanced.addRow("Top-p", self.top_p)
        advanced.addRow("Max tokens", self.max_tokens)
        advanced.addRow("Context", self.context)
        advanced.addRow("GPU layers (-1 auto)", self.gpu_layers)
        advanced.addRow("Generation speed", speed_row)
        layout.addWidget(advanced_group)

    def _send(self) -> None:
        text = self.input.toPlainText().strip()
        if text:
            self.input.clear()
            self.sendRequested.emit(text)

    def _start_infinite(self) -> None:
        seed = self.input.toPlainText().strip() or "Begin an ordinary day in a new embodied world."
        self.input.clear()
        self.infiniteRequested.emit(seed)

    def config(self) -> GenerationConfig:
        return GenerationConfig(self.temperature.value(), self.top_p.value(), self.max_tokens.value(),
                                self.context.value(), self.gpu_layers.value(), self.speed.value() / 10)

    def _update_speed_label(self, value: int) -> None:
        self.speed_value.setText(f"{value / 10:.1f}×")

    def set_models(self, models: list[ModelInfo]) -> None:
        self.models.blockSignals(True)
        self.models.clear()
        available = [model for model in models if model.available]
        self.models.addItem("No model selected", None)
        for model in available:
            self.models.addItem(model.label, model)
        self.models.blockSignals(False)
        self.models.setEnabled(bool(available))
        self._analysis_available = False
        self.open_analysis.setEnabled(False)

    def set_analysis_available(self, available: bool) -> None:
        self._analysis_available = available
        self.open_analysis.setEnabled(available and not self.stop.isEnabled())

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
        QTimer.singleShot(0, self._scroll_to_bottom_if_following)

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
        previous = QPushButton("◀ Token")
        replay = QPushButton("▶ Replay neurons")
        next_token = QPushButton("Token ▶")
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
        self.send.setEnabled(not running)
        self.stop.setEnabled(running)
        self.regenerate.setEnabled(not running)
        self.infinite.setEnabled(not running)
        self.models.setEnabled(not running)
        self.open_analysis.setEnabled(not running and self._analysis_available)

    def set_analysis_mode(self, infinite: bool) -> None:
        self.open_analysis.setText("Analysis+" if infinite else "Analysis")
        self.open_analysis.setToolTip(
            "Export compact neural findings for an Infinite simulation" if infinite
            else "Export normal chat session data without neural-network findings"
        )
