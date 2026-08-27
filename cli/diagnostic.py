"""Standalone Repair and Diagnostics application entry point."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.console_ui import clear_screen
from src.utils.logging import configure_logging


def main() -> int:
    clear_screen()
    from PySide6.QtWidgets import QApplication
    from src.app.diagnostics_window import DiagnosticsWindow

    configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("AIBrain Repair and Diagnostics")
    window = DiagnosticsWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
