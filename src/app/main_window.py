from __future__ import annotations

import logging
from time import monotonic
from typing import TypedDict

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QSplitter

from .chat_panel import ChatPanel
from .diagnostics_window import DiagnosticsWindow
from .settings import load_generation_settings, save_generation_settings
from .visualizer_panel import VisualizerPanel
from ..models.generation_worker import GenerationWorker
from ..models.infinite_simulation import InfiniteSimulationWorker
from ..models.llama_backend import LlamaBackend
from ..models.model_info import ModelInfo

LOG = logging.getLogger(__name__)

# noinspection LongLine
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
QLabel#worldBubble { background: #28203b; border: 1px solid #5c4b86; color: #e7ddff; border-radius: 10px; padding: 10px; margin: 3px 0; }
QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit { background: #0d202b; border: 1px solid #264553; border-radius: 6px; padding: 6px; color: #e1eef3; }
QPushButton { background: #15536b; border: none; border-radius: 6px; padding: 7px 12px; color: white; font-weight: 600; }
QPushButton:hover { background: #1b6d8b; } QPushButton:disabled { background: #26353b; color: #82939a; }
QScrollArea { background: transparent; } QSplitter::handle { background: #1a3440; width: 1px; }
"""


class SimulationStats(TypedDict):
    seconds: float
    turns: int
    participant_tokens: int
    cancelled: bool


class GenerationStats(TypedDict):
    seconds: float
    generated_tokens: int
    prompt_tokens: int
    cancelled: bool


class MainWindow(QMainWindow):
    startGeneration = Signal(object, object, object)
    unloadModel = Signal()
    startInfiniteSimulation = Signal(object, object, object)

    def __init__(self, models: list[ModelInfo]) -> None:
        super().__init__()
        self.setWindowTitle("AIBrain — Local LLM Connectome")
        self._display_fitted = False
        self.setStyleSheet(STYLESHEET)
        self.config = load_generation_settings()
        self.history: list[dict[str, str]] = []
        self.current_model: ModelInfo | None = None
        self._assistant_bubble: QLabel | None = None
        self._awaiting_first_token = False
        self._started = 0.0
        self._simulation_bubbles: dict[tuple[str, int], QLabel] = {}
        self.simulation_transcript: list[dict[str, object]] = []
        self._setup_worker()
        self.chat = ChatPanel(self.config)
        self.visualizer = VisualizerPanel()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.chat)
        splitter.addWidget(self.visualizer)
        splitter.setSizes([560, 940])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)
        self.chat.sendRequested.connect(self.send)
        self.chat.stopRequested.connect(self.stop)
        self.chat.regenerateRequested.connect(self.regenerate)
        self.chat.clearRequested.connect(self.clear)
        self.chat.infiniteRequested.connect(self.start_infinite_simulation)
        self.chat.diagnosticsRequested.connect(self.open_diagnostics)
        self.chat.modelChanged.connect(self.select_model)
        self.chat.playbackRequested.connect(lambda: self.visualizer.start_playback(self.config.speed))
        self.chat.playbackPreviousRequested.connect(self.visualizer.playback_previous)
        self.chat.playbackNextRequested.connect(self.visualizer.playback_next)
        self._escape_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._escape_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._escape_shortcut.activated.connect(self._handle_escape)
        self._set_models(models)

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().showEvent(event)
        if not self._display_fitted:
            self._display_fitted = True
            QTimer.singleShot(0, self._fit_to_display)

    def _fit_to_display(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        available = screen.availableGeometry()
        self.setMaximumSize(available.size())
        self.resize(min(1500, available.width()), min(900, available.height()))

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
        self.simulation_thread = QThread(self)
        self.simulation_worker = InfiniteSimulationWorker()
        self.simulation_worker.moveToThread(self.simulation_thread)
        self.startInfiniteSimulation.connect(self.simulation_worker.run)
        # noinspection DuplicatedCode
        self.simulation_worker.turnStarted.connect(self._simulation_turn_started)
        self.simulation_worker.token.connect(self._simulation_token)
        self.simulation_worker.turnFinished.connect(self._simulation_turn_finished)
        self.simulation_worker.finished.connect(self._simulation_finished)
        self.simulation_worker.failed.connect(self._simulation_failed)
        self.simulation_thread.start()

    def _set_models(self, models: list[ModelInfo]) -> None:
        self.chat.set_models(models)
        if models:
            self.select_model(models[0])
            self.chat.stats.setText(f"Validated {len(models)} GGUF model(s); select one to begin.")
        else:
            self.chat.stats.setText(
                "No usable GGUF blob found in %USERPROFILE%\\.ollama\\models. "
                "Install a GGUF Ollama model, then reopen AIBrain."
            )

    def select_model(self, model: ModelInfo | None) -> None:
        if model == self.current_model:
            return
        self.unloadModel.emit()
        self.simulation_worker.unload()
        self.current_model = model
        if model is not None:
            self.visualizer.set_model(f"{model.name}:{model.tag}")
            self.history.clear()
            self.visualizer.set_conversation(self.history)
            self.chat.clear_messages()
            self.chat.stats.setText(
                "Selected "
                + model.label
                + ("" if model.available else f" — {model.error or 'Unknown error'}")
                + " · conversation and connectome refreshed"
            )

    def _handle_escape(self) -> None:
        if self.chat.stop.isEnabled():
            self.stop()
        else:
            self.chat.stats.setText("Escape interrupts an active generation.")

    def open_diagnostics(self) -> None:
        """Open model recovery without interrupting a currently loaded model."""
        DiagnosticsWindow(self).exec()

    def send(self, prompt: str) -> None:
        if not self.current_model or not self.current_model.available or not self.current_model.blob_path:
            QMessageBox.warning(self, "Model unavailable",
                                "Choose an available GGUF model discovered from Ollama first.")
            return
        self.config = self.chat.config()
        save_generation_settings(self.config)
        self.history.append({"role": "user", "content": prompt})
        self.visualizer.set_conversation(self.history)
        self.chat.clear_latest_playback()
        self.visualizer.begin_response_recording()
        self.chat.add_message("user", prompt)
        self._assistant_bubble = self.chat.add_message("assistant", "Thinking…")
        self._awaiting_first_token = True
        self._started = monotonic()
        self.chat.generating(True)
        self.startGeneration.emit(self.history.copy(), self.config, self.current_model.blob_path)

    def stop(self) -> None:
        # Direct flag write is safe and necessary while worker is iterating blocking native code.
        self.worker.cancel()
        self.simulation_worker.cancel()
        self.chat.stats.setText("Stopping after the current generated token…")

    def start_infinite_simulation(self, seed: str) -> None:
        if not self.current_model or not self.current_model.available or not self.current_model.blob_path:
            QMessageBox.warning(self, "Model unavailable",
                                "Choose an available GGUF model discovered from Ollama first.")
            return
        self.config = self.chat.config()
        save_generation_settings(self.config)
        self.unloadModel.emit()
        self.visualizer.begin_recording()
        self.simulation_transcript = [{"role": "user", "content": seed, "turn": 0}]
        self.visualizer.set_conversation(self.simulation_transcript)
        self.chat.clear_latest_playback()
        self.chat.add_message("user", seed)
        self._simulation_bubbles.clear()
        self.chat.generating(True)
        self.chat.stats.setText(
            "∞ Simulation loading two local model instances; the right pane is the participant brain.")
        self.startInfiniteSimulation.emit(seed, self.config, self.current_model.blob_path)

    def _simulation_turn_started(self, role: str, turn: int) -> None:
        label = "World" if role == "world" else "Participant"
        bubble = self.chat.add_message(role, f"{label} {turn}: ")
        self._simulation_bubbles[(role, turn)] = bubble

    def _simulation_token(self, role: str, turn: int, text: str, frame: object) -> None:
        bubble = self._simulation_bubbles.get((role, turn))
        if bubble is not None:
            self.chat.append_message_text(bubble, text)
        if role == "participant" and frame is not None:
            self.visualizer.apply_frame(frame)  # type: ignore[arg-type]

    def _simulation_turn_finished(self, role: str, text: str, turn: int) -> None:
        if text:
            self.simulation_transcript.append({"role": role, "content": text, "turn": turn})
            self.visualizer.set_conversation(self.simulation_transcript)

    def _simulation_finished(self, stats: SimulationStats) -> None:
        seconds = stats["seconds"]
        turns = stats["turns"]
        tokens = stats["participant_tokens"]
        suffix = " (stopped)" if stats["cancelled"] else ""
        self.chat.stats.setText(
            f"∞ Simulation: {turns} world turn(s), {tokens} participant tokens in {seconds:.1f}s{suffix}"
        )
        self.chat.generating(False)

    def _simulation_failed(self, error: str) -> None:
        LOG.error("%s", error)
        self.chat.stats.setText(error)
        self.chat.generating(False)
        QMessageBox.critical(self, "Infinite simulation error", error)

    def regenerate(self) -> None:
        if self.history and self.history[-1]["role"] == "assistant":
            self.history.pop()
        if self.history and self.history[-1]["role"] == "user":
            prompt = self.history.pop()["content"]
            self.send(prompt)

    def clear(self) -> None:
        self.history.clear()
        self.simulation_transcript.clear()
        self.visualizer.begin_recording()
        self.visualizer.set_conversation([])
        self.chat.clear_messages()
        self.chat.stats.setText("Conversation cleared.")

    def _on_token(self, text: str, frame: object) -> None:
        if self._assistant_bubble is not None:
            if self._awaiting_first_token:
                self._assistant_bubble.setText("")
                self._awaiting_first_token = False
            self.chat.append_message_text(self._assistant_bubble, text)
        self.visualizer.apply_frame(frame)  # type: ignore[arg-type]

    def _finished(self, stats: GenerationStats) -> None:
        text = self._assistant_bubble.text() if self._assistant_bubble is not None else ""
        if text and text != "Thinking…":
            self.history.append({"role": "assistant", "content": text})
            self.visualizer.set_conversation(self.history)
            self.chat.show_latest_playback()
        elif self._assistant_bubble is not None:
            self._assistant_bubble.setText("No response generated.")

        seconds = stats["seconds"]
        tokens = stats["generated_tokens"]
        rate = tokens / seconds if seconds else 0.0
        suffix = " (stopped)" if stats["cancelled"] else ""

        self.chat.stats.setText(
            f"Prompt {stats['prompt_tokens']} tokens · "
            f"generated {tokens} tokens in {seconds:.2f}s · "
            f"{rate:.1f} tokens/s{suffix}"
        )

        self.chat.generating(False)
        self._assistant_bubble = None
        self._awaiting_first_token = False

    def _failed(self, error: str) -> None:
        LOG.error("%s", error)
        self.chat.stats.setText(error)
        self.chat.generating(False)
        self._assistant_bubble = None
        self._awaiting_first_token = False
        QMessageBox.critical(self, "Generation error", error)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.visualizer.analyzer.save_model()
        if self.worker_thread.isRunning():
            self.worker.cancel()
            self.worker_thread.quit()
            self.worker_thread.wait(3000)
        if self.simulation_thread.isRunning():
            self.simulation_worker.cancel()
            self.simulation_thread.quit()
            self.simulation_thread.wait(3000)
        super().closeEvent(event)
