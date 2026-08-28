"""Standalone AIBrain connectome analysis application entry point."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.console_ui import Color, clear_screen, header, info, panel, section, status
from src.utils.logging import configure_logging


def relative_age(path: Path) -> str:
    """Return a compact age for a local manifest without relying on locale."""
    seconds = max(0, int(datetime.now(timezone.utc).timestamp() - path.stat().st_mtime))
    if seconds < 60:
        return "less than one minute"
    for unit, length in (("day", 86_400), ("hour", 3_600), ("minute", 60)):
        amount, remainder = divmod(seconds, length)
        if amount:
            suffix = "" if amount == 1 else "s"
            return f"{amount} {unit}{suffix}" + (
                f" {remainder // 60} minutes" if unit == "day" and remainder >= 60 else "")
    return "unknown"


def main() -> int:
    clear_screen()
    from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
    from src.app.diagnostics_window import DiagnosticsWindow
    from src.app.main_window import STYLESHEET
    from src.app.visualizer_panel import VisualizerPanel
    from src.models.diagnostics import OllamaDiagnostics

    configure_logging()
    header("AIBrain", "Connectome analysis launcher")
    app = QApplication(sys.argv)
    app.setApplicationName("AIBrain Analysis")
    section("Inspect local analysis models", 1)
    diagnostics = OllamaDiagnostics().inspect(verify_backend=True)
    available = [item for item in diagnostics if item.available]
    if diagnostics:
        rows: list[tuple[str, str]] = []
        for item in diagnostics:
            age = relative_age(item.manifest_path)
            size = f"{item.blob_path.stat().st_size / 1024 ** 3:.2f} GB" if item.blob_path and item.blob_path.exists() else "Unavailable"
            health = "Healthy: llama.cpp loaded it" if item.available else f"Needs repair: {item.detail}"
            rows.extend(
                [
                    (f"{item.reference} health", health),
                    (f"{item.reference} age", age),
                    (f"{item.reference} features", f"GGUF blob, {size}"),
                ]
            )
        panel("LOCAL MODEL HEALTH", rows, footer="Health includes a direct llama.cpp compatibility load.",
              tone=Color.GREEN if available else Color.YELLOW)
    else:
        status("INFO", "No local Ollama manifests were found", Color.YELLOW)
    info(f"Validated local GGUF models available for analysis: {len(available)}")
    window = QMainWindow()
    window.setWindowTitle("AIBrain Analysis")
    window.setStyleSheet(STYLESHEET)
    if available:
        visualizer = VisualizerPanel(available[0].reference)
        visualizer.analysis.setEnabled(True)
        window.setCentralWidget(visualizer)
    else:
        page = QWidget()
        layout = QVBoxLayout(page)
        message = QLabel("Analysis is unavailable until a validated local GGUF model is installed.")
        message.setWordWrap(True)
        message.setObjectName("muted")
        repair = QPushButton("Open Repair and Diagnostics")
        repair.clicked.connect(lambda: DiagnosticsWindow(window).exec())
        layout.addWidget(message)
        layout.addWidget(repair)
        layout.addStretch(1)
        window.setCentralWidget(page)
    window.resize(1200, 800)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
