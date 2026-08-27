from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget


class LoadingWindow(QWidget):
    """A compact startup surface shown while local GGUFs are checked."""

    cancelled = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._completed = False
        self.setWindowTitle("AIBrain - Preparing local intelligence")
        self.setFixedSize(560, 270)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setStyleSheet(
            "QWidget { background: #08131c; color: #e4f4f8; font: 10pt 'Segoe UI'; }"
            "QLabel#title { font-size: 24px; font-weight: 700; color: #f4fbff; }"
            "QLabel#eyebrow { color: #62d6f4; font-weight: 700; letter-spacing: 1px; }"
            "QLabel#detail { color: #91aab5; }"
            "QProgressBar { border: 1px solid #275060; border-radius: 7px; height: 12px; text-align: center; background: #10242e; }"
            "QProgressBar::chunk { border-radius: 6px; background: #35bfdc; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(38, 34, 38, 34)
        layout.setSpacing(10)

        eyebrow = QLabel("AIBRAIN  /  STARTUP")
        eyebrow.setObjectName("eyebrow")
        layout.addWidget(eyebrow)

        title = QLabel("Preparing your local workspace")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Checking installed GGUF models before opening the connectome.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("detail")
        layout.addWidget(subtitle)
        layout.addSpacing(8)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

        self.detail = QLabel("Starting local services")
        self.detail.setObjectName("detail")
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        layout.addStretch(1)

    def set_progress(self, current: int, total: int, detail: str) -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(current)
        self.detail.setText(detail)

    def finish(self) -> None:
        """Close after a successful handoff without treating it as cancellation."""
        self._completed = True
        self.close()

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if not self._completed:
            self.cancelled.emit()
        super().closeEvent(event)
