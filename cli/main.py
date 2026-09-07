"""AIBrain desktop entry point."""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

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
from src.utils.gpu import (
    configure_opengl_surface,
    prepare_gpu_launch,
)
from src.utils.logging import (
    REPORTABLE_EXCEPTIONS,
    configure_cli_logging,
    report_exception,
)
from src.utils.runtime import require_managed_runtime


if TYPE_CHECKING:
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication

    from src.app.startup_coordinator import StartupCoordinator
    from src.models.model_validator import StartupWorker


def require_virtual_environment() -> None:
    """Prevent accidental system-wide package use or installation."""
    if "__compiled__" in globals():
        return

    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return

    header("AIBrain", "Desktop connectome launcher")
    error("AIBrain must run inside a Python virtual environment.")

    instruction_list(
        [
            ("1.", "Create and install:", r"py cli\installer.py"),
            ("2.", "Activate it:", r".\.venv\Scripts\Activate.ps1"),
            ("3.", "Run AIBrain:", r"python cli\main.py"),
        ],
        stream=sys.stderr,
    )

    raise SystemExit(1)


def _show_startup_banner(runtime_log: Path) -> None:
    """Render desktop startup information."""
    clear_screen()

    header("AIBrain", "Desktop connectome launcher")
    section("Desktop startup", 1)

    status("LOG", f"CLI output: {runtime_log}")
    status(
        "START",
        "Preparing Qt, local model validation, and the OpenGL adapter check",
    )

    print()


def _create_application() -> QApplication:
    """Create and configure the Qt application."""
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    app.setFont(QFont("Segoe UI", 10))
    app.setApplicationName("AIBrain")
    app.setOrganizationName("AIBrain")

    return app


def _run_event_loop(
    app: QApplication,
    coordinator: StartupCoordinator,
    startup_worker: StartupWorker,
    startup_thread: QThread,
) -> int:
    """Run Qt while providing orderly SIGINT and worker shutdown handling."""
    from PySide6.QtCore import QTimer

    interrupted = False
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    interrupt_timer = QTimer(app)
    interrupt_timer.timeout.connect(lambda: None)
    interrupt_timer.start(200)

    def handle_keyboard_interrupt(
        _signal_number: int,
        _frame: object,
    ) -> None:
        nonlocal interrupted

        if interrupted:
            return

        interrupted = True

        window = coordinator.main_window

        if window is not None:
            QTimer.singleShot(0, window.close)
            return

        QTimer.singleShot(0, coordinator.cancel_startup)

    signal.signal(signal.SIGINT, handle_keyboard_interrupt)

    try:
        exit_code = app.exec()

        if interrupted:
            report_keyboard_interrupt("AIBrain")
            return 130

        if exit_code == 0:
            report_gui_closed("AIBrain desktop")

        return exit_code
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)

        startup_worker.cancel()

        if startup_thread.isRunning():
            startup_thread.quit()
            startup_thread.wait()


def main() -> int:
    """Launch the AIBrain desktop application."""
    runtime_log, _ = configure_cli_logging("main")

    if not require_managed_runtime(ROOT, "main"):
        return 1

    gpu_exit = prepare_gpu_launch(
        compiled="__compiled__" in globals(),
    )

    if gpu_exit is not None:
        return gpu_exit

    _show_startup_banner(runtime_log)

    from PySide6.QtCore import QThread, QTimer

    from src.app.loading_window import GpuProbe, LoadingWindow
    from src.app.startup_coordinator import StartupCoordinator
    from src.models.model_validator import StartupWorker
    from src.utils.gpu import should_prefer_high_performance_gpu

    configure_opengl_surface()

    app = _create_application()

    loading = LoadingWindow()

    startup_thread = QThread(app)
    startup_worker = StartupWorker()
    startup_worker.moveToThread(startup_thread)

    gpu_probe = GpuProbe(app)

    coordinator = StartupCoordinator(
        app,
        loading,
        startup_thread,
        startup_worker,
        prefer_high_performance_gpu=should_prefer_high_performance_gpu(),
    )

    coordinator.connect_signals(gpu_probe)

    loading.show_centered()

    QTimer.singleShot(0, coordinator.start)
    QTimer.singleShot(0, gpu_probe.run)

    return _run_event_loop(
        app,
        coordinator,
        startup_worker,
        startup_thread,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        report_keyboard_interrupt("AIBrain")
        raise SystemExit(130) from None
    except REPORTABLE_EXCEPTIONS as exc:
        report_exception("AIBrain desktop launcher failed", exc)
        raise SystemExit(1) from exc
