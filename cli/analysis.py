"""Standalone AIBrain connectome analysis application entry point."""

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
from src.utils.gpu import configure_opengl_surface, prepare_gpu_launch
from src.utils.logging import (
    REPORTABLE_EXCEPTIONS,
    configure_cli_logging,
    report_exception,
)
from src.utils.runtime import require_managed_runtime

LOG = logging.getLogger(__name__)


def main() -> int:
    runtime_log, _ = configure_cli_logging("analysis")
    LOG.info("Starting AIBrain Analysis CLI (runtime log: %s)", runtime_log)
    if not require_managed_runtime(ROOT, "analysis"):
        LOG.error("Analysis CLI startup stopped because runtime preflight failed")
        return 1
    gpu_exit = prepare_gpu_launch(compiled="__compiled__" in globals())
    if gpu_exit is not None:
        return gpu_exit
    clear_screen()
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication

    from src.app.analysis_window import AnalysisWindow
    from src.app.loading_window import LoadingWindow

    header("AIBrain", "Persisted analysis-model inspector")
    section("Desktop startup", 1)
    status("LOG", f"CLI output: {runtime_log}")
    status("START", "Opening the Analysis+ model inspector")
    print()
    LOG.info("Analysis runtime preflight passed; creating the desktop inspector")
    configure_opengl_surface()
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("AIBrain Analysis")
    interrupted = False
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def quit_for_keyboard_interrupt(_signal: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        app.quit()

    signal.signal(signal.SIGINT, quit_for_keyboard_interrupt)
    loading = LoadingWindow(
        eyebrow_text="AIBRAIN  /  ANALYSIS+",
        title_text="Inspecting learned analysis data",
        subtitle_text="Checking the persisted Analysis+ autoencoder before opening the inspector.",
        detail_text="Preparing the model-health report",
    )
    window = AnalysisWindow(auto_refresh=False)
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    loading.cancelled.connect(window.close)
    window.destroyed.connect(app.quit)

    def open_inspector(_healthy: bool) -> None:
        LOG.info(
            "Initial Analysis+ inspection completed (healthy=%s); showing inspector",
            _healthy,
        )
        loading.finish()
        window.showMaximized()

    window.inspection_finished.connect(open_inspector)
    loading.show_centered()
    QTimer.singleShot(0, window.refresh)
    try:
        exit_code = app.exec()
        final_exit_code = 130 if interrupted else exit_code
        LOG.info(
            "Analysis inspector exited (exit_code=%s, interrupted=%s)",
            final_exit_code,
            interrupted,
        )
        if interrupted:
            report_keyboard_interrupt("AIBrain Analysis")
        elif final_exit_code == 0:
            report_gui_closed("AIBrain Analysis")
        return final_exit_code
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        report_keyboard_interrupt("AIBrain Analysis")
        raise SystemExit(130) from None
    except REPORTABLE_EXCEPTIONS as exc:
        report_exception("AIBrain Analysis launcher failed", exc)
        raise SystemExit(1) from exc
