from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QHBoxLayout, QLabel,
                               QPlainTextEdit, QPushButton, QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget)

from ..models.llama_backend import GenerationConfig
from ..models.model_info import ModelInfo


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

    def __init__(self, config: GenerationConfig) -> None:
        super().__init__()
        self._build(config)
        self._playback_controls: QWidget | None = None

    def _build(self, config: GenerationConfig) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        title = QLabel("AIBrain")
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel("Local GGUF chat · direct model loading")
        subtitle.setObjectName("muted")
        layout.addWidget(subtitle)
        self.models = QComboBox()
        self.models.currentIndexChanged.connect(lambda _: self.modelChanged.emit(self.models.currentData()))
        layout.addWidget(self.models)
        self.messages_layout = QVBoxLayout()
        self.messages_layout.addStretch(1)
        container = QWidget()
        container.setLayout(self.messages_layout)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(container)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(self.scroll, 1)
        self.stats = QLabel("Ready · select an installed GGUF model")
        self.stats.setObjectName("muted")
        self.stats.setWordWrap(True)
        self.stats.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        layout.addWidget(self.stats)
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Message your local model…  (Ctrl+Enter to send)")
        self.input.setFixedHeight(92)
        layout.addWidget(self.input)
        buttons = QHBoxLayout()
        self.send = QPushButton("Send")
        self.stop = QPushButton("Stop")
        self.regenerate = QPushButton("Regenerate")
        self.infinite = QPushButton("∞ Inf")
        self.infinite.setToolTip("Start an open-ended world/participant roleplay simulation using two local model instances")
        self.clear = QPushButton("Clear")
        self.stop.setEnabled(False)
        self.send.clicked.connect(self._send)
        self.stop.clicked.connect(self.stopRequested)
        self.regenerate.clicked.connect(self.regenerateRequested)
        self.infinite.clicked.connect(self._start_infinite)
        self.clear.clicked.connect(self.clearRequested)
        for button in (self.send, self.stop, self.regenerate, self.infinite, self.clear): buttons.addWidget(button)
        layout.addLayout(buttons)
        advanced = QFormLayout()
        self.temperature = QDoubleSpinBox(); self.temperature.setRange(0, 2); self.temperature.setSingleStep(.05); self.temperature.setValue(config.temperature)
        self.top_p = QDoubleSpinBox(); self.top_p.setRange(.05, 1); self.top_p.setSingleStep(.05); self.top_p.setValue(config.top_p)
        self.max_tokens = QSpinBox(); self.max_tokens.setRange(1, 8192); self.max_tokens.setValue(config.max_tokens)
        self.context = QSpinBox(); self.context.setRange(512, 32768); self.context.setSingleStep(512); self.context.setValue(config.context_length)
        self.gpu_layers = QSpinBox(); self.gpu_layers.setRange(-1, 200); self.gpu_layers.setValue(config.gpu_layers)
        self.speed = QSlider(); self.speed.setOrientation(Qt.Orientation.Horizontal); self.speed.setRange(1, 10); self.speed.setValue(round(config.speed * 10))
        self.speed_value = QLabel(); self.speed.valueChanged.connect(self._update_speed_label); self._update_speed_label(self.speed.value())
        speed_row = QWidget(); speed_layout = QHBoxLayout(speed_row); speed_layout.setContentsMargins(0, 0, 0, 0); speed_layout.addWidget(self.speed); speed_layout.addWidget(self.speed_value)
        advanced.addRow("Temperature", self.temperature); advanced.addRow("Top-p", self.top_p); advanced.addRow("Max tokens", self.max_tokens)
        advanced.addRow("Context", self.context); advanced.addRow("GPU layers (-1 auto)", self.gpu_layers); advanced.addRow("Generation speed", speed_row)
        layout.addLayout(advanced)

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
        return GenerationConfig(self.temperature.value(), self.top_p.value(), self.max_tokens.value(), self.context.value(), self.gpu_layers.value(), self.speed.value() / 10)

    def _update_speed_label(self, value: int) -> None:
        self.speed_value.setText(f"{value / 10:.1f}×")

    def set_models(self, models: list[ModelInfo]) -> None:
        self.models.blockSignals(True)
        self.models.clear()
        if not models: self.models.addItem("No Ollama GGUF models discovered", None)
        for model in (model for model in models if model.available):
            self.models.addItem(model.label, model)
        self.models.blockSignals(False)
        self.models.setEnabled(bool(models))

    def set_validating_models(self, text: str) -> None:
        self.models.blockSignals(True)
        self.models.clear()
        self.models.addItem(text, None)
        self.models.setEnabled(False)
        self.models.blockSignals(False)

    def add_message(self, role: str, text: str) -> QLabel:
        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        bubble.setObjectName("userBubble" if role == "user" else "worldBubble" if role == "world" else "assistantBubble")
        self.messages_layout.insertWidget(self.messages_layout.count()-1, bubble)
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())
        return bubble

    def clear_latest_playback(self) -> None:
        if self._playback_controls is None:
            return
        self.messages_layout.removeWidget(self._playback_controls)
        self._playback_controls.deleteLater()
        self._playback_controls = None

    def show_latest_playback(self) -> None:
        self.clear_latest_playback()
        controls = QWidget()
        layout = QHBoxLayout(controls); layout.setContentsMargins(4, 1, 4, 5)
        previous = QPushButton("◀ Token")
        replay = QPushButton("▶ Replay neurons")
        next_token = QPushButton("Token ▶")
        previous.clicked.connect(self.playbackPreviousRequested)
        replay.clicked.connect(self.playbackRequested)
        next_token.clicked.connect(self.playbackNextRequested)
        layout.addWidget(previous); layout.addWidget(replay); layout.addWidget(next_token); layout.addStretch(1)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, controls)
        self._playback_controls = controls

    def clear_messages(self) -> None:
        self._playback_controls = None
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    def generating(self, running: bool) -> None:
        self.send.setEnabled(not running); self.stop.setEnabled(running); self.regenerate.setEnabled(not running); self.infinite.setEnabled(not running); self.models.setEnabled(not running)
