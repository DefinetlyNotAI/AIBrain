"""Standalone AIBrain connectome analysis application entry point."""
from __future__ import annotations

import sys
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.console_ui import clear_screen, error, header
from src.utils.logging import configure_logging
from src.utils.runtime import require_managed_runtime


def main() -> int:
    if not require_managed_runtime(ROOT, "analysis"):
        return 1
    clear_screen()
    from PySide6.QtWidgets import QApplication
    from src.app.analysis_window import AnalysisWindow
    from src.app.main_window import STYLESHEET

    configure_logging("analysis")
    header("AIBrain", "Persisted analysis-model inspector")
    app = QApplication(sys.argv)
    app.setApplicationName("AIBrain Analysis")
    interrupted = False
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def quit_for_keyboard_interrupt(_signal: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        app.quit()

    signal.signal(signal.SIGINT, quit_for_keyboard_interrupt)
    window = AnalysisWindow()
    window.setStyleSheet(STYLESHEET)
    window.show()
    try:
        exit_code = app.exec()
        return 130 if interrupted else exit_code
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        error("Analysis cancelled by keyboard interrupt.")
        raise SystemExit(130)
