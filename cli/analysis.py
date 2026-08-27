"""Standalone AIBrain connectome analysis application entry point."""
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
    from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
    from src.app.diagnostics_window import DiagnosticsWindow
    from src.app.main_window import STYLESHEET
    from src.app.visualizer_panel import VisualizerPanel
    from src.models.ollama_discovery import OllamaDiscovery

    configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("AIBrain Analysis")
    models = OllamaDiscovery().discover()
    window = QMainWindow()
    window.setWindowTitle("AIBrain Analysis")
    window.setStyleSheet(STYLESHEET)
    if models:
        visualizer = VisualizerPanel(f"{models[0].name}:{models[0].tag}")
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
