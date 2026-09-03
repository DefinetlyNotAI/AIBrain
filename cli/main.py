"""AIBrain desktop entry point."""
from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.console_ui import (
    clear_screen,
    error,
    header,
    instruction_list,
    report_gui_closed,
    report_keyboard_interrupt,
    section,
    status,
)
from src.utils.gpu import GPU_RELAUNCH_EXIT_CODE, configure_opengl_surface, prepare_gpu_launch
from src.utils.logging import configure_cli_logging, report_exception
from src.utils.runtime import require_managed_runtime

LOG = logging.getLogger(__name__)

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


def main() -> int:
    runtime_log, _ = configure_cli_logging("main")
    if not require_managed_runtime(ROOT, "main"):
        return 1
    gpu_exit = prepare_gpu_launch(compiled="__compiled__" in globals())
    if gpu_exit is not None:
        return gpu_exit
    clear_screen()
    header("AIBrain", "Desktop connectome launcher")
    section("Desktop startup", 1)
    status("LOG", f"CLI output: {runtime_log}")
    status("START", "Preparing Qt, local model validation, and the OpenGL adapter check")
    print()
    from PySide6.QtCore import QCoreApplication, QObject, QThread, QTimer, Slot
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication
    from src.app.loading_window import GpuProbe, LoadingWindow
    from src.app.main_window import MainWindow
    from src.models.model_validator import StartupWorker
    from src.utils.gpu import (
        can_request_gpu_relaunch,
        should_prefer_high_performance_gpu,
    )

    prefer_high_performance = should_prefer_high_performance_gpu()
    configure_opengl_surface()
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setApplicationName("AIBrain")
    app.setOrganizationName("AIBrain")
    interrupted = False
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

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
            self._models_finished = False
            self._gpu_checked = False
            self._gpu_relaunch_requested = False
            self._stopping = False
            self._main_window_started = False
            self._shutdown_exit_code = 0

        @Slot(object)
        def models_ready(self, models: object) -> None:
            self._models = models if isinstance(models, list) else []

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
                self._stop_startup(exit_code=GPU_RELAUNCH_EXIT_CODE)
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
                    self._stopping
                    or self._gpu_relaunch_requested
                    or not loading.isVisible()
                    or self._models is None
                    or not self._models_finished
                    or not self._gpu_checked
                    or self._main_window_started
            ):
                return
            self._main_window_started = True
            try:
                window = MainWindow(self._models)
            except Exception:
                LOG.exception("Could not construct the AIBrain main window")
                self._stop_startup(exit_code=1)
                return
            app.main_window = window  # type: ignore[attr-defined]
            # Maximize as an ordinary resizable desktop window; never enter
            # borderless/fullscreen mode, so system controls remain available.
            window.showMaximized()
            loading.finish()

        def _stop_startup(self, *, exit_code: int = 0) -> None:
            """Cancel startup and leave Qt only after its worker thread stops."""
            if self._stopping:
                return
            self._stopping = True
            self._shutdown_exit_code = exit_code
            startup_worker.cancel()
            loading.finish()
            if startup_thread.isRunning():
                startup_thread.quit()
            else:
                QTimer.singleShot(0, self.startup_thread_finished)

        @Slot()
        def cancel_startup(self) -> None:
            """Handle loader cancellation without destroying an active QThread."""
            LOG.info("Cancelling desktop startup and waiting for validation to stop")
            self._stop_startup()

        @Slot()
        def startup_thread_finished(self) -> None:
            if self._stopping:
                QCoreApplication.exit(self._shutdown_exit_code)
                return
            self._models_finished = True
            self._show_main_when_ready()

        @Slot(str)
        def show_startup_error(self, message: str) -> None:
            loading.set_progress(1, 1, message)
            self.models_ready([])

    startup_coordinator = StartupCoordinator()
    app.startup_coordinator = startup_coordinator  # type: ignore[attr-defined]

    startup_thread.started.connect(startup_worker.run)
    startup_worker.progress.connect(loading.set_progress)
    startup_worker.finished.connect(startup_coordinator.models_ready)
    startup_worker.finished.connect(startup_thread.quit)
    startup_worker.failed.connect(startup_coordinator.show_startup_error)
    startup_worker.failed.connect(startup_thread.quit)
    gpu_probe.completed.connect(startup_coordinator.gpu_ready)
    gpu_probe.failed.connect(startup_coordinator.gpu_probe_failed)
    startup_thread.finished.connect(startup_worker.deleteLater)
    startup_thread.finished.connect(startup_coordinator.startup_thread_finished)
    loading.cancelled.connect(startup_coordinator.cancel_startup)

    def quit_for_keyboard_interrupt(_signal: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        QTimer.singleShot(0, startup_coordinator.cancel_startup)

    signal.signal(signal.SIGINT, quit_for_keyboard_interrupt)

    def start_startup_worker() -> None:
        if not startup_coordinator._stopping:
            startup_thread.start()

    loading.show_centered()
    QTimer.singleShot(0, start_startup_worker)
    QTimer.singleShot(0, gpu_probe.run)
    try:
        exit_code = app.exec()
        final_exit_code = 130 if interrupted else exit_code
        if interrupted:
            report_keyboard_interrupt("AIBrain")
        elif final_exit_code == 0:
            report_gui_closed("AIBrain desktop")
        return final_exit_code
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
        report_keyboard_interrupt("AIBrain")
        raise SystemExit(130)
    except Exception as exc:
        report_exception("AIBrain desktop launcher failed", exc)
        raise SystemExit(1)
