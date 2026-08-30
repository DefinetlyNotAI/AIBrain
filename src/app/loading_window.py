from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import (
    QGuiApplication,
    QOffscreenSurface,
    QOpenGLContext,
    QSurfaceFormat,
)
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

LOG = logging.getLogger(__name__)


class GpuProbe(QObject):
    """Probe the Qt OpenGL adapter before creating the main application UI."""

    completed = Signal(str, str)
    failed = Signal(str)

    @Slot()
    def run(self) -> None:
        LOG.info("Starting OpenGL adapter probe for startup loader")
        surface = QOffscreenSurface()
        surface.setFormat(QSurfaceFormat.defaultFormat())
        surface.create()
        context = QOpenGLContext()
        context.setFormat(surface.format())

        if not context.create() or not context.makeCurrent(surface):
            LOG.warning("OpenGL startup probe could not create a current context")
            self.failed.emit("Could not create an OpenGL startup context")
            return

        try:
            import moderngl

            gl_context = moderngl.create_context(require=330)
            renderer = str(gl_context.info.get("GL_RENDERER", "Unknown renderer"))
            vendor = str(gl_context.info.get("GL_VENDOR", "Unknown vendor"))
            LOG.info("Startup loader detected OpenGL adapter: vendor=%s renderer=%s", vendor, renderer)
            self.completed.emit(vendor, renderer)
            gl_context.release()
        except Exception as exc:
            LOG.exception("Startup loader could not inspect the OpenGL adapter")
            self.failed.emit(f"Could not inspect the OpenGL adapter: {exc}")
        finally:
            context.doneCurrent()


class LoadingWindow(QWidget):
    """A compact startup surface shown while local GGUFs are checked."""

    cancelled = Signal()

    def __init__(
            self,
            *,
            eyebrow_text: str = "AIBRAIN  /  STARTUP",
            title_text: str = "Preparing your local workspace",
            subtitle_text: str = "Checking installed GGUF models before opening the connectome.",
            detail_text: str = "Starting local services",
    ) -> None:
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

        eyebrow = QLabel(eyebrow_text)
        eyebrow.setObjectName("eyebrow")
        layout.addWidget(eyebrow)

        title = QLabel(title_text)
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel(subtitle_text)
        subtitle.setWordWrap(True)
        subtitle.setObjectName("detail")
        layout.addWidget(subtitle)
        layout.addSpacing(8)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

        self.detail = QLabel(detail_text)
        self.detail.setObjectName("detail")
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        layout.addStretch(1)

    def show_centered(self) -> None:
        """Show the compact loader in the centre of the primary work area."""
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.move(available.center() - self.frameGeometry().center())
        LOG.info("Showing compact startup loader (size=%sx%s)", self.width(), self.height())
        self.show()

    def set_progress(self, current: int, total: int, detail: str) -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(current)
        self.detail.setText(detail)
        LOG.debug("Startup loader progress: %s/%s %s", current, total, detail)

    def finish(self) -> None:
        """Close after a successful handoff without treating it as cancellation."""
        self._completed = True
        LOG.info("Startup loader handoff completed")
        self.close()

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if not self._completed:
            LOG.info("Startup loader was cancelled")
            self.cancelled.emit()
        super().closeEvent(event)
