"""AIBrain desktop entry point."""
from __future__ import annotations

import os
import signal
import sys



if os.name == "nt":
    os.system("")


class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


def color(text: str, *styles: str) -> str:
    return "".join(styles) + text + Color.RESET


def require_virtual_environment() -> None:
    """Prevent accidental system-wide package use or installation."""
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        print(file=sys.stderr)

        print(
            "  "
            + color("✗", Color.RED, Color.BOLD)
            + " "
            + color(
                "AIBrain must run inside a Python virtual environment.",
                Color.RED,
                Color.BOLD,
            ),
            file=sys.stderr,
        )

        print(file=sys.stderr)

        print(
            "  "
            + color("1.", Color.CYAN, Color.BOLD)
            + " "
            + color("Create and install:", Color.GRAY)
            + "   "
            + color("py installer.py", Color.WHITE, Color.BOLD),
            file=sys.stderr,
        )

        print(
            "  "
            + color("2.", Color.CYAN, Color.BOLD)
            + " "
            + color("Activate it:", Color.GRAY)
            + "          "
            + color(
                r".\.venv\Scripts\Activate.ps1",
                Color.WHITE,
                Color.BOLD,
            ),
            file=sys.stderr,
        )

        print(
            "  "
            + color("3.", Color.CYAN, Color.BOLD)
            + " "
            + color("Run AIBrain:", Color.GRAY)
            + "          "
            + color("python main.py", Color.WHITE, Color.BOLD),
            file=sys.stderr,
        )

        print(file=sys.stderr)

        raise SystemExit(1)


def main() -> int:
    require_virtual_environment()
    os.environ.setdefault("QT_OPENGL", "desktop")
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QFont, QSurfaceFormat
    from PySide6.QtWidgets import QApplication
    from src.app.main_window import MainWindow
    from src.utils.logging import configure_logging
    from src.utils.gpu import relaunch_for_gpu_preference, set_windows_gpu_preference, should_prefer_high_performance_gpu

    prefer_high_performance = should_prefer_high_performance_gpu()
    if prefer_high_performance and set_windows_gpu_preference(True) and relaunch_for_gpu_preference():
        return 0

    surface = QSurfaceFormat()
    surface.setVersion(3, 3)
    surface.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface.setDepthBufferSize(24)
    surface.setSamples(0)
    QSurfaceFormat.setDefaultFormat(surface)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL, True)
    configure_logging()
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
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
