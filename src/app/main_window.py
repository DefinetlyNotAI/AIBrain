from __future__ import annotations

import logging
from time import monotonic

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QMainWindow, QMessageBox, QSplitter

from .chat_panel import ChatPanel
from .settings import load_generation_settings, save_generation_settings
from .visualizer_panel import VisualizerPanel
from ..models.generation_worker import GenerationWorker
from ..models.llama_backend import LlamaBackend
from ..models.model_info import ModelInfo
from ..models.ollama_discovery import OllamaDiscovery
from ..models.model_validator import ModelValidator

LOG = logging.getLogger(__name__)


STYLESHEET = """
QMainWindow { background: #09131c; color: #dceaf1; }
QWidget { font: 10pt 'Segoe UI'; color: #dceaf1; }
QLabel#title { font-size: 22px; font-weight: 650; color: #f2f8fb; }
QLabel#muted { color: #88a0ae; }
QLabel#mode { background: #123849; color: #82e7ff; border-radius: 9px; padding: 4px 8px; font-weight: 700; }
QLabel#overlay { background: #0c1e2a; border: 1px solid #193746; border-radius: 8px; padding: 8px; color: #9ac4d4; }
QLabel#userBubble, QLabel#assistantBubble { border-radius: 10px; padding: 10px; margin: 3px 0; }
QLabel#userBubble { background: #143e50; color: #e4f8ff; }
QLabel#assistantBubble { background: #10212d; border: 1px solid #1a3645; }
QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit { background: #0d202b; border: 1px solid #264553; border-radius: 6px; padding: 6px; color: #e1eef3; }
QPushButton { background: #15536b; border: none; border-radius: 6px; padding: 7px 12px; color: white; font-weight: 600; }
QPushButton:hover { background: #1b6d8b; } QPushButton:disabled { background: #26353b; color: #82939a; }
QScrollArea { background: transparent; } QSplitter::handle { background: #1a3440; width: 1px; }
"""


class MainWindow(QMainWindow):
    startGeneration = Signal(object, object, object)
    unloadModel = Signal()
    validateModels = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AIBrain — Local LLM Connectome")
        self.resize(1500, 900)
        self.setStyleSheet(STYLESHEET)
        self.config = load_generation_settings()
        self.history: list[dict[str, str]] = []
        self.current_model: ModelInfo | None = None
        self._assistant_bubble = None
        self._awaiting_first_token = False
        self._started = 0.0
        self._setup_worker()
        self.chat = ChatPanel(self.config)
        self.visualizer = VisualizerPanel()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.chat); splitter.addWidget(self.visualizer); splitter.setSizes([560, 940])
        self.setCentralWidget(splitter)
        self.chat.sendRequested.connect(self.send)
        self.chat.stopRequested.connect(self.stop)
        self.chat.regenerateRequested.connect(self.regenerate)
        self.chat.clearRequested.connect(self.clear)
        self.chat.modelChanged.connect(self.select_model)
        self.chat.playbackRequested.connect(lambda: self.visualizer.start_playback(self.config.speed))
        self.chat.playbackPreviousRequested.connect(self.visualizer.playback_previous)
        self.chat.playbackNextRequested.connect(self.visualizer.playback_next)
        self._escape_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._escape_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._escape_shortcut.activated.connect(self._handle_escape)
        self._discover()

    def _setup_worker(self) -> None:
        self.backend = LlamaBackend()
        self.worker_thread = QThread(self)
        self.worker = GenerationWorker(self.backend)
        self.worker.moveToThread(self.worker_thread)
        self.startGeneration.connect(self.worker.generate)
        self.unloadModel.connect(self.worker.unload)
        self.worker.token.connect(self._on_token)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker_thread.start()

    def _discover(self) -> None:
        models = OllamaDiscovery().discover()
        if not models:
            self.chat.stats.setText("No usable GGUF blob found in %USERPROFILE%\\.ollama\\models. Install a GGUF Ollama model, then reopen AIBrain.")
            return
        self.chat.set_validating_models("Validating installed GGUF models…")
        self.chat.stats.setText(f"Validating {len(models)} installed GGUF models with the local backend…")
        self.validation_thread = QThread(self)
        self.validator = ModelValidator()
        self.validator.moveToThread(self.validation_thread)
        self.validateModels.connect(self.validator.validate)
        self.validation_thread.started.connect(lambda: self.validateModels.emit(models))
        self.validator.progress.connect(self.chat.stats.setText)
        self.validator.finished.connect(self._validation_finished)
        self.validator.finished.connect(self.validation_thread.quit)
        self.validation_thread.finished.connect(self.validator.deleteLater)
        self.validation_thread.start()

    def _validation_finished(self, models: list[ModelInfo]) -> None:
        self.chat.set_models(models)
        if models:
            self.select_model(models[0])
            self.chat.stats.setText(f"Validated {len(models)} GGUF model(s); select one to begin.")
        else:
            self.chat.stats.setText("No installed GGUF model could be loaded by this llama.cpp backend.")

    def select_model(self, model: ModelInfo | None) -> None:
        if model == self.current_model:
            return
        self.unloadModel.emit()
        self.current_model = model
        if model is not None:
            self.visualizer.set_model(f"{model.name}:{model.tag}")
            self.history.clear()
            self.chat.clear_messages()
            self.chat.stats.setText("Selected " + model.label + ("" if model.available else f" — {model.error}") + " · conversation and connectome refreshed")

    def _handle_escape(self) -> None:
        if self.chat.stop.isEnabled():
            self.stop()
        else:
            self.chat.stats.setText("Escape interrupts an active generation.")

    def send(self, prompt: str) -> None:
        if not self.current_model or not self.current_model.available or not self.current_model.blob_path:
            QMessageBox.warning(self, "Model unavailable", "Choose an available GGUF model discovered from Ollama first.")
            return
        self.config = self.chat.config(); save_generation_settings(self.config)
        self.history.append({"role": "user", "content": prompt})
        self.chat.clear_latest_playback()
        self.visualizer.begin_recording()
        self.chat.add_message("user", prompt)
        self._assistant_bubble = self.chat.add_message("assistant", "Thinking…")
        self._awaiting_first_token = True
        self._started = monotonic(); self.chat.generating(True)
        self.startGeneration.emit(self.history.copy(), self.config, self.current_model.blob_path)

    def stop(self) -> None:
        # Direct flag write is safe and necessary while worker is iterating blocking native code.
        self.worker.cancel()
        self.chat.stats.setText("Stopping after the current generated token…")

    def regenerate(self) -> None:
        if self.history and self.history[-1]["role"] == "assistant": self.history.pop()
        if self.history and self.history[-1]["role"] == "user":
            prompt = self.history.pop()["content"]
            self.send(prompt)

    def clear(self) -> None:
        self.history.clear(); self.chat.clear_messages(); self.chat.stats.setText("Conversation cleared.")

    def _on_token(self, text: str, frame: object) -> None:
        if self._assistant_bubble is not None:
            if self._awaiting_first_token:
                self._assistant_bubble.setText("")
                self._awaiting_first_token = False
            self._assistant_bubble.setText(self._assistant_bubble.text() + text)
            self.chat.scroll.verticalScrollBar().setValue(self.chat.scroll.verticalScrollBar().maximum())
        self.visualizer.apply_frame(frame)  # type: ignore[arg-type]

    def _finished(self, stats: dict[str, object]) -> None:
        text = self._assistant_bubble.text() if self._assistant_bubble is not None else ""
        if text and text != "Thinking…":
            self.history.append({"role": "assistant", "content": text})
            self.chat.show_latest_playback()
        elif self._assistant_bubble is not None:
            self._assistant_bubble.setText("No response generated.")
        seconds = float(stats["seconds"]); tokens = int(stats["generated_tokens"])
        rate = tokens / seconds if seconds else 0
        suffix = " (stopped)" if stats.get("cancelled") else ""
        self.chat.stats.setText(f"Prompt {int(stats['prompt_tokens'])} tokens · generated {tokens} tokens in {seconds:.2f}s · {rate:.1f} tokens/s{suffix}")
        self.chat.generating(False); self._assistant_bubble = None; self._awaiting_first_token = False

    def _failed(self, error: str) -> None:
        LOG.error("%s", error)
        self.chat.stats.setText(error); self.chat.generating(False); self._assistant_bubble = None; self._awaiting_first_token = False
        QMessageBox.critical(self, "Generation error", error)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if hasattr(self, "validator"):
            self.validator.cancel()
        if hasattr(self, "validation_thread") and self.validation_thread.isRunning():
            self.validation_thread.quit()
            self.validation_thread.wait(3000)
        if self.worker_thread.isRunning():
            self.worker.cancel(); self.worker_thread.quit(); self.worker_thread.wait(3000)
        super().closeEvent(event)
