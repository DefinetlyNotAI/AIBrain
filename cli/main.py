"""AIBrain desktop entry point."""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.console_ui import clear_screen, error, header, instruction_list
from src.utils.gpu import GPU_RELAUNCH_EXIT_CODE
from src.utils.runtime import require_managed_runtime

LOG = logging.getLogger(__name__)

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
        header("AIBrain", "Desktop connectome launcher")
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
            try:
                child.terminate()
            except OSError:
                pass
            else:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            return 130

        if exit_code != GPU_RELAUNCH_EXIT_CODE:
            return exit_code

    return GPU_RELAUNCH_EXIT_CODE


def main() -> int:
    if not require_managed_runtime(ROOT, "main"):
        return 1
    clear_screen()
    os.environ.setdefault("QT_OPENGL", "desktop")
    from PySide6.QtCore import QCoreApplication, QObject, Qt, QThread, QTimer, Slot
    from PySide6.QtGui import QFont, QSurfaceFormat
    from PySide6.QtWidgets import QApplication
    from src.app.loading_window import GpuProbe, LoadingWindow
    from src.app.main_window import MainWindow
    from src.utils.logging import configure_logging
    from src.models.model_validator import StartupWorker
    from src.utils.gpu import (
        can_request_gpu_relaunch,
        set_windows_gpu_preference,
        should_prefer_high_performance_gpu,
    )

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
    configure_logging("main")
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setApplicationName("AIBrain")
    app.setOrganizationName("AIBrain")
    interrupted = False
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def quit_for_keyboard_interrupt(_signal: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        app.quit()

    signal.signal(signal.SIGINT, quit_for_keyboard_interrupt)
    interrupt_timer = QTimer(app)
    interrupt_timer.timeout.connect(lambda: None)
    interrupt_timer.start(200)

    loading = LoadingWindow()
    startup_thread = QThread(app)
    startup_worker = StartupWorker()
    startup_worker.moveToThread(startup_thread)
    gpu_probe = GpuProbe(app)

    class StartupCoordinator(QObject):
        """Receive worker completion signals on the QApplication thread."""

        def __init__(self) -> None:
            super().__init__(app)
            self._models: list[object] | None = None
            self._gpu_checked = False
            self._gpu_relaunch_requested = False

        @Slot(object)
        def models_ready(self, models: object) -> None:
            self._models = models if isinstance(models, list) else []
            self._show_main_when_ready()

        @Slot(str, str)
        def gpu_ready(self, vendor: str, renderer: str) -> None:
            is_nvidia = "nvidia" in f"{vendor} {renderer}".lower()
            loading.set_progress(0, 1, f"OpenGL adapter: {vendor} - {renderer}")
            if prefer_high_performance and not is_nvidia and can_request_gpu_relaunch():
                LOG.warning(
                    "Loader detected a non-NVIDIA OpenGL adapter: vendor=%s renderer=%s; restarting before main UI",
                    vendor,
                    renderer,
                )
                # The worker may have already queued its completion callback.
                # Mark this path terminal before asking Qt to leave its event
                # loop so no MainWindow can flash between loader processes.
                self._gpu_relaunch_requested = True
                startup_worker.cancel()
                startup_thread.quit()
                loading.finish()
                QCoreApplication.exit(GPU_RELAUNCH_EXIT_CODE)
                return
            self._gpu_checked = True
            self._show_main_when_ready()

        @Slot(str)
        def gpu_probe_failed(self, message: str) -> None:
            LOG.warning("%s", message)
            loading.set_progress(0, 1, message)
            self._gpu_checked = True
            self._show_main_when_ready()

        def _show_main_when_ready(self) -> None:
            if (
                    self._gpu_relaunch_requested
                    or not loading.isVisible()
                    or self._models is None
                    or not self._gpu_checked
            ):
                return
            window = MainWindow(self._models)
            app.main_window = window  # type: ignore[attr-defined]
            # Maximize as an ordinary resizable desktop window; never enter
            # borderless/fullscreen mode, so system controls remain available.
            window.showMaximized()
            loading.finish()
            startup_thread.quit()

        @Slot(str)
        def show_startup_error(self, message: str) -> None:
            loading.set_progress(1, 1, message)
            self.models_ready([])

    startup_coordinator = StartupCoordinator()
    app.startup_coordinator = startup_coordinator  # type: ignore[attr-defined]

    startup_thread.started.connect(startup_worker.run)
    startup_worker.progress.connect(loading.set_progress)
    startup_worker.finished.connect(startup_coordinator.models_ready)
    startup_worker.failed.connect(startup_coordinator.show_startup_error)
    gpu_probe.completed.connect(startup_coordinator.gpu_ready)
    gpu_probe.failed.connect(startup_coordinator.gpu_probe_failed)
    startup_thread.finished.connect(startup_worker.deleteLater)
    loading.cancelled.connect(startup_worker.cancel)
    loading.cancelled.connect(startup_thread.quit)
    loading.cancelled.connect(app.quit)
    loading.showMaximized()
    QTimer.singleShot(0, startup_thread.start)
    QTimer.singleShot(0, gpu_probe.run)
    try:
        exit_code = app.exec()
        return 130 if interrupted else exit_code
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        startup_worker.cancel()
        if startup_thread.isRunning():
            startup_thread.quit()
            startup_thread.wait()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        error("AIBrain cancelled by keyboard interrupt.")
        raise SystemExit(130)
