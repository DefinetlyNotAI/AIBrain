"""AIBrain desktop entry point."""
from __future__ import annotations

import sys


def require_virtual_environment() -> None:
    """Prevent accidental system-wide package use or installation."""
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        print(
            "AIBrain must run inside a Python virtual environment.\n"
            "Create one:  python -m venv .venv\n"
            "Activate it: .\\.venv\\Scripts\\Activate.ps1\n"
            "Install deps:  py -3.11 install.py",
            file=sys.stderr,
        )
        raise SystemExit(1)


def main() -> int:
    require_virtual_environment()
    from PySide6.QtWidgets import QApplication
    from src.app.main_window import MainWindow
    from src.utils.logging import configure_logging

    configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("AIBrain")
    app.setOrganizationName("AIBrain")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
