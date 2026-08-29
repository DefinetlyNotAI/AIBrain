"""Standalone Repair and Diagnostics application entry point."""
from __future__ import annotations

import sys
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.console_ui import clear_screen, error, header, section, status
from src.utils.logging import configure_cli_logging
from src.utils.runtime import require_managed_runtime


def main() -> int:
    runtime_log, _ = configure_cli_logging("diagnostic")
    if not require_managed_runtime(ROOT, "diagnostic"):
        return 1
    clear_screen()
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtWidgets import QApplication
    from src.app.diagnostics_window import DiagnosticsWindow
    from src.app.loading_window import LoadingWindow

    header("AIBrain", "Repair and diagnostics launcher")
    section("Desktop startup", 1)
    status("LOG", f"CLI output: {runtime_log}")
    status("START", "Checking local model health before opening diagnostics")
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
        loading.finish()
        window.showMaximized()

    window.inspection_finished.connect(open_diagnostics)
    loading.showMaximized()
    QTimer.singleShot(0, window.refresh)
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
