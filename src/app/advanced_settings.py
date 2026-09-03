"""Dedicated advanced generation and Analysis+ settings dialog."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..models.llama_backend import GenerationConfig


class AdvancedSettingsDialog(QDialog):
    """Keep optional generation and Analysis+ limits in a separate screen."""

    def __init__(self, config: GenerationConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AIBrain advanced settings")
        self.setModal(True)
        layout = QVBoxLayout(self)
        explanation = QLabel("Changes apply when the next generation starts.")
        explanation.setObjectName("muted")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        body = QWidget()
        form = QFormLayout(body)
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
        self.analysis_memory_mb = QSpinBox()
        self.analysis_memory_mb.setRange(16, 4096)
        self.analysis_memory_mb.setSingleStep(16)
        self.analysis_memory_mb.setValue(100)
        self.analysis_memory_mb.setToolTip(
            "Maximum RAM retained for Infinite-mode rewind frames; "
            "older Analysis+ records move to the temporary cache"
        )
        self.analysis_cache_mb = QSpinBox()
        self.analysis_cache_mb.setRange(0, 8192)
        self.analysis_cache_mb.setSingleStep(128)
        self.analysis_cache_mb.setValue(1024)
        self.analysis_cache_mb.setSpecialValueText("Disabled")
        self.analysis_cache_mb.setToolTip(
            "Maximum temporary Analysis+ cache in .cache/temp; 0 disables paging"
        )
        self.speed = QSlider(Qt.Orientation.Horizontal)
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
        form.addRow("Temperature", self.temperature)
        form.addRow("Top-p", self.top_p)
        form.addRow("Max tokens", self.max_tokens)
        form.addRow("Context", self.context)
        form.addRow("GPU layers (-1 auto)", self.gpu_layers)
        form.addRow("Analysis+ RAM (MB)", self.analysis_memory_mb)
        form.addRow("Analysis+ cache (MB)", self.analysis_cache_mb)
        form.addRow("Generation speed", speed_row)

        self.temperature.setToolTip("Sampling randomness for Normal Chat")
        self.top_p.setToolTip("Nucleus sampling limit for Normal Chat")
        self.max_tokens.setToolTip("Maximum generated tokens per Normal Chat response")
        self.context.setToolTip("Local model context window size")
        self.gpu_layers.setToolTip("Number of layers requested on the GPU; -1 is automatic")
        self.speed.setToolTip("Replay presentation speed")

        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.settings_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.settings_scroll.setWidget(body)
        layout.addWidget(self.settings_scroll, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(520, 460)

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """Fit the settings screen to the active display and show its top."""
        super().showEvent(event)
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(
                min(520, max(340, available.width() - 40)),
                min(460, max(280, available.height() - 40)),
            )
            self.move(available.center() - self.rect().center())
        scrollbar = self.settings_scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.minimum())

    def config(self) -> GenerationConfig:
        return GenerationConfig(
            self.temperature.value(),
            self.top_p.value(),
            self.max_tokens.value(),
            self.context.value(),
            self.gpu_layers.value(),
            self.speed.value() / 10,
        )

    def _update_speed_label(self, value: int) -> None:
        self.speed_value.setText(f"{value / 10:.1f}×")
