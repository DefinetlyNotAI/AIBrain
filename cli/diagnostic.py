"""Standalone Repair and Diagnostics application entry point."""

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
    header,
    report_gui_closed,
    report_keyboard_interrupt,
    section,
    status,
)
from src.utils.logging import configure_cli_logging, report_exception
from src.utils.gpu import configure_opengl_surface, prepare_gpu_launch
from src.utils.runtime import require_managed_runtime

LOG = logging.getLogger(__name__)

PRESERVE_CONSOLE_FLAG = "--preserve-console"


def _consume_preserve_console_flag() -> bool:
    """Remove the desktop-launch marker before Qt parses command-line options."""
    preserve_console = PRESERVE_CONSOLE_FLAG in sys.argv[1:]
    if preserve_console:
        sys.argv[:] = [argument for argument in sys.argv if argument != PRESERVE_CONSOLE_FLAG]
    return preserve_console

def main() -> int:
    runtime_log, _ = configure_cli_logging("diagnostic")
    LOG.info("Starting AIBrain Diagnostics CLI (runtime log: %s)", runtime_log)
    if not require_managed_runtime(ROOT, "diagnostic"):
        LOG.error("Diagnostics CLI startup stopped because runtime preflight failed")
        return 1
    gpu_exit = prepare_gpu_launch(compiled="__compiled__" in globals())
    if gpu_exit is not None:
        return gpu_exit
    if not _consume_preserve_console_flag():
        clear_screen()
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtWidgets import QApplication
    from src.app.diagnostics_window import DiagnosticsWindow
    from src.app.loading_window import LoadingWindow

    header("AIBrain", "Repair and diagnostics launcher")
    section("Desktop startup", 1)
    status("LOG", f"CLI output: {runtime_log}")
    status("START", "Checking local model health before opening diagnostics")
    print()
    LOG.info("Diagnostics runtime preflight passed; creating the desktop window")
    configure_opengl_surface()
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("AIBrain Repair and Diagnostics")
    interrupted = False
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def quit_for_keyboard_interrupt(_signal: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        app.quit()

    signal.signal(signal.SIGINT, quit_for_keyboard_interrupt)
    loading = LoadingWindow(
        eyebrow_text="AIBRAIN  /  DIAGNOSTICS",
        title_text="Checking local model health",
        subtitle_text="Validating local GGUF manifests and backend compatibility before opening diagnostics.",
        detail_text="Inspecting local Ollama models",
    )
    window = DiagnosticsWindow(auto_refresh=False)
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    loading.cancelled.connect(window.close)
    window.destroyed.connect(app.quit)

    def open_diagnostics(_healthy: bool) -> None:
        LOG.info(
            "Initial diagnostics inspection completed (healthy=%s); showing window",
            _healthy,
        )
        loading.finish()
        QTimer.singleShot(0, window.showMaximized)

    window.inspection_finished.connect(open_diagnostics)
    window.inspection_progress.connect(loading.set_progress)
    loading.show_centered()
    QTimer.singleShot(0, window.refresh)
    try:
        exit_code = app.exec()
        final_exit_code = 130 if interrupted else exit_code
        LOG.info(
            "Diagnostics window exited (exit_code=%s, interrupted=%s)",
            final_exit_code,
            interrupted,
        )
        if interrupted:
            report_keyboard_interrupt("AIBrain Diagnostics")
        elif final_exit_code == 0:
            report_gui_closed("AIBrain Diagnostics")
        return final_exit_code
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        report_keyboard_interrupt("AIBrain Diagnostics")
        raise SystemExit(130)
    except Exception as exc:
        report_exception("AIBrain Diagnostics launcher failed", exc)
        raise SystemExit(1)
