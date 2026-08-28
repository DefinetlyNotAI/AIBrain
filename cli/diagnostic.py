"""Standalone Repair and Diagnostics application entry point."""
from __future__ import annotations

import sys
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.console_ui import clear_screen, error
from src.utils.logging import configure_logging


def main() -> int:
    clear_screen()
    from PySide6.QtWidgets import QApplication
    from src.app.diagnostics_window import DiagnosticsWindow

    configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("AIBrain Repair and Diagnostics")
    interrupted = False
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def quit_for_keyboard_interrupt(_signal: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        app.quit()

    signal.signal(signal.SIGINT, quit_for_keyboard_interrupt)
    window = DiagnosticsWindow()
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
        error("Diagnostics cancelled by keyboard interrupt.")
        raise SystemExit(130)
