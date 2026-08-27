"""AIBrain desktop entry point."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.console_ui import error, instruction_list
from src.utils.gpu import GPU_RELAUNCH_EXIT_CODE

_GPU_SUPERVISOR_ENV = "AIBRAIN_GPU_SUPERVISOR"
_GPU_RELAUNCH_ATTEMPT_ENV = "AIBRAIN_GPU_RELAUNCH_ATTEMPT"
_MAX_GPU_RELAUNCH_ATTEMPTS = 1


def require_virtual_environment() -> None:
    """Prevent accidental system-wide package use or installation."""
    # Nuitka embeds this entry point in its own isolated Python runtime. It is
    # deliberately not a development virtual environment, but is still the
    # managed self-contained distribution we support.
    if "__compiled__" in globals():
        return
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        error("AIBrain must run inside a Python virtual environment.")
        instruction_list(
            [
                ("1.", "Create and install:", "py cli\\installer.py"),
                ("2.", "Activate it:", r".\.venv\Scripts\Activate.ps1"),
                ("3.", "Run AIBrain:", "python cli\\main.py"),
            ],
            stream=sys.stderr,
        )

        raise SystemExit(1)


def _child_command() -> list[str]:
    """Build the exact command for a supervised Python or Nuitka child."""
    if "__compiled__" in globals():
        return [sys.executable, *sys.argv[1:]]
    return [sys.executable, *sys.argv]


def _supervise_gpu_launch() -> int:
    """Keep this console process alive while a GPU-configured child runs."""
    environment = os.environ.copy()
    environment[_GPU_SUPERVISOR_ENV] = "1"

    for attempt in range(_MAX_GPU_RELAUNCH_ATTEMPTS + 1):
        child_environment = environment.copy()
        child_environment[_GPU_RELAUNCH_ATTEMPT_ENV] = str(attempt)
        child = subprocess.Popen(
            _child_command(),
            env=child_environment,
            close_fds=False,
        )
        try:
            exit_code = child.wait()
        except KeyboardInterrupt:
            child.terminate()
            child.wait()
            return 130

        if exit_code != GPU_RELAUNCH_EXIT_CODE:
            return exit_code

    return GPU_RELAUNCH_EXIT_CODE


def main() -> int:
    require_virtual_environment()
    os.environ.setdefault("QT_OPENGL", "desktop")
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QFont, QSurfaceFormat
    from PySide6.QtWidgets import QApplication
    from src.app.main_window import MainWindow
    from src.utils.logging import configure_logging
    from src.utils.gpu import set_windows_gpu_preference, should_prefer_high_performance_gpu

    prefer_high_performance = should_prefer_high_performance_gpu()
    if prefer_high_performance:
        set_windows_gpu_preference(True)
        if sys.platform == "win32" and not os.environ.get(_GPU_SUPERVISOR_ENV):
            return _supervise_gpu_launch()

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
