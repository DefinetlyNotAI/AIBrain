"""AIBrain desktop entry point."""
from __future__ import annotations

import signal
import sys


def require_virtual_environment() -> None:
    """Prevent accidental system-wide package use or installation."""
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        print(
            "AIBrain must run inside a Python virtual environment.\n"
            "Create and install: py -3.11 install.py\n"
            "Activate it:          .\\.venv\\Scripts\\Activate.ps1\n"
            "Run AIBrain:          python main.py",
            file=sys.stderr,
        )
        raise SystemExit(1)


def main() -> int:
    require_virtual_environment()
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtWidgets import QApplication
    from src.app.main_window import MainWindow
    from src.utils.logging import configure_logging

    surface = QSurfaceFormat()
    surface.setVersion(3, 3)
    surface.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface.setDepthBufferSize(24)
    surface.setSamples(4)
    QSurfaceFormat.setDefaultFormat(surface)
    configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("AIBrain")
    app.setOrganizationName("AIBrain")
    window = MainWindow()
    window.show()
    signal.signal(signal.SIGINT, lambda _signal, _frame: app.quit())
    interrupt_timer = QTimer(app)
    interrupt_timer.timeout.connect(lambda: None)
    interrupt_timer.start(200)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
