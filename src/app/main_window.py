from __future__ import annotations

import logging
import sys
from pathlib import Path
from time import monotonic
from typing import TypedDict

from PySide6.QtCore import (
    QCoreApplication,
    QProcess,
    QSettings,
    QThread,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QSizePolicy, QSplitter

from .chat_panel import ChatPanel
from .settings import load_generation_settings, save_generation_settings
from .theme import load_colours, stylesheet
from .visualizer_panel import VisualizerPanel
from ..models.generation_worker import GenerationWorker
from ..models.infinite_simulation import InfiniteSimulationWorker
from ..models.llama_backend import LlamaBackend
from ..models.model_info import ModelInfo
from ..utils.gpu import GPU_RELAUNCH_EXIT_CODE

LOG = logging.getLogger(__name__)


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
    startInfiniteSimulation = Signal(object, object, object, object)

    def __init__(self, models: list[ModelInfo]) -> None:
        super().__init__()
        self.setWindowTitle("AIBrain — Local LLM Connectome")
        self._display_fitted = False
        self.colours = load_colours()
        self.setStyleSheet(stylesheet(self.colours))
        self.config = load_generation_settings()
        self.history: list[dict[str, str]] = []
        self.current_model: ModelInfo | None = None
        self._assistant_bubble: QLabel | None = None
        self._awaiting_first_token = False
        self._started = 0.0
        self._simulation_bubbles: dict[tuple[str, int], QLabel] = {}
        self.simulation_transcript: list[dict[str, object]] = []
        self._analysis_is_infinite = False
        self._gpu_relaunching = False
        self._setup_worker()
        self.chat = ChatPanel(self.config)
        self.visualizer = VisualizerPanel()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        splitter.addWidget(self.chat)
        splitter.addWidget(self.visualizer)
        self.chat.setMinimumWidth(0)
        self.visualizer.setMinimumWidth(0)
        self.chat.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.visualizer.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        saved_sizes = QSettings().value("main_splitter_sizes")
        if isinstance(saved_sizes, list) and len(saved_sizes) == 2:
            splitter.setSizes([int(size) for size in saved_sizes])
        else:
            splitter.setSizes([560, 940])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        self._splitter = splitter
        self.setCentralWidget(splitter)
        self.chat.sendRequested.connect(self.send)
        self.chat.stopRequested.connect(self.stop)
        self.chat.regenerateRequested.connect(self.regenerate)
        self.chat.clearRequested.connect(self.clear)
        self.chat.infiniteRequested.connect(self.start_infinite_simulation)
        self.chat.infiniteContinueRequested.connect(self.continue_infinite_simulation)
        self.chat.diagnosticsRequested.connect(self.open_diagnostics)
        self.chat.analysisRequested.connect(self.open_analysis)
        self.chat.modeChanged.connect(self._change_mode)
        self.chat.rewindRequested.connect(self.visualizer.enter_rewind_mode)
        self.chat.rewindExitRequested.connect(self.visualizer.exit_rewind_mode)
        self.visualizer.rewindChanged.connect(self.chat.set_rewind_mode)
        self.visualizer.playbackChanged.connect(self.chat.set_replay_mode)
        self.visualizer.analysisMemoryExceeded.connect(
            self.chat.set_analysis_memory_exceeded
        )
        self.chat.modelChanged.connect(self.select_model)
        self.chat.playbackRequested.connect(
            lambda: self.visualizer.start_playback(self.config.speed)
        )
        self.chat.playbackPreviousRequested.connect(self.visualizer.playback_previous)
        self.chat.playbackNextRequested.connect(self.visualizer.playback_next)
        self.visualizer.gpuRestartRequested.connect(self._restart_for_gpu_mismatch)
        self.visualizer.coloursChanged.connect(self._apply_colours)
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
        # Do not constrain the top-level maximum size: Qt must be free to fit
        # normal window chrome inside the actual available display geometry.
        self.resize(min(1500, available.width()), min(900, available.height()))

    def _apply_colours(self, colours: dict[str, str]) -> None:
        self.colours = colours.copy()
        self.setStyleSheet(stylesheet(self.colours))

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
        available = [model for model in models if model.available]
        if available:
            self.chat.stats.setText("Select a local GGUF model to begin.")
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
        self.chat.set_model_available(
            bool(model and model.available and model.blob_path)
        )
        self.chat.set_analysis_available(False)
        self.chat.set_regenerate_available(False)
        if model is not None:
            self.visualizer.set_model(f"{model.name}:{model.tag}")
            self.history.clear()
            self.visualizer.set_conversation(self.history)
            self.chat.clear_messages()
            self.chat.models.setToolTip(
                f"{model.label}\n{model.blob_path}"
                + (f"\n{model.error}" if model.error else "")
            )
            self.chat.stats.setText("Conversation and connectome refreshed.")

    def _handle_escape(self) -> None:
        if self.chat.send.text() == "Stop":
            self.stop()
        else:
            self.chat.stats.setText("Escape interrupts an active generation.")

    def _change_mode(self, infinite: bool) -> None:
        """Switching modes is intentionally a clean conversation/replay boundary."""
        self.history.clear()
        self.simulation_transcript.clear()
        self._analysis_is_infinite = infinite
        self.visualizer.begin_recording()
        self.visualizer.set_conversation([])
        self.chat.set_analysis_mode(infinite)
        self.chat.set_analysis_available(False)
        self.chat.set_regenerate_available(False)
        self.chat.stats.setText(
            "Infinite Mode ready." if infinite else "Normal Chat ready."
        )

    def open_diagnostics(self) -> None:
        """Launch the one standalone Diagnostics app without interrupting inference."""
        root = Path(__file__).resolve().parents[2]
        packaged = Path(sys.executable).with_name("diagnostic.exe")
        if "__compiled__" in globals() and packaged.is_file():
            started = QProcess.startDetached(str(packaged), [])
        else:
            started = QProcess.startDetached(
                sys.executable, [str(root / "cli" / "diagnostic.py")], str(root)
            )
        if not started:
            QMessageBox.critical(
                self,
                "Diagnostics launch failed",
                "Could not start the Diagnostics application.",
            )

    def open_analysis(self) -> None:
        if self.current_model is None:
            return
        if self._analysis_is_infinite:
            self.visualizer.run_nn_analysis_plus()
        else:
            self.visualizer.run_session_analysis()

    def _restart_for_gpu_mismatch(self, message: str) -> None:
        """Close the main UI cleanly so the supervisor returns to the loader."""
        if self._gpu_relaunching:
            return
        self._gpu_relaunching = True
        LOG.warning("%s; closing the current window before supervised restart", message)
        self.chat.stats.setText(
            "GPU mismatch detected. Returning to the startup loader…"
        )
        QCoreApplication.exit(GPU_RELAUNCH_EXIT_CODE)
        self.close()

    def _show_repairable_error(self, title: str, message: str) -> None:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle(title)
        dialog.setText(message)
        diagnostics = dialog.addButton(
            "Open Repair and Diagnostics", QMessageBox.ButtonRole.ActionRole
        )
        dialog.addButton(QMessageBox.StandardButton.Ok)
        dialog.exec()
        if dialog.clickedButton() is diagnostics:
            self.open_diagnostics()

    def send(self, prompt: str) -> None:
        if (
            not self.current_model
            or not self.current_model.available
            or not self.current_model.blob_path
        ):
            QMessageBox.warning(
                self,
                "Model unavailable",
                "Choose an available GGUF model discovered from Ollama first.",
            )
            return
        self.config = self.chat.config()
        self.visualizer.set_analysis_memory_limit(None)
        self._analysis_is_infinite = False
        self.chat.set_analysis_mode(False)
        self.chat.set_analysis_available(False)
        self.chat.set_regenerate_available(False)
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
        self.startGeneration.emit(
            self.history.copy(), self.config, self.current_model.blob_path
        )

    def stop(self) -> None:
        # Direct flag write is safe and necessary while worker is iterating blocking native code.
        self.worker.cancel()
        self.simulation_worker.cancel()
        self.chat.stats.setText("Stopping after the current generated token…")

    def start_infinite_simulation(self, seed: str, continuation: bool = False) -> None:
        if (
            not self.current_model
            or not self.current_model.available
            or not self.current_model.blob_path
        ):
            QMessageBox.warning(
                self,
                "Model unavailable",
                "Choose an available GGUF model discovered from Ollama first.",
            )
            return
        self._analysis_is_infinite = True
        self.chat.set_analysis_mode(True)
        self.config = self.chat.config()
        self.visualizer.set_analysis_memory_limit(self.chat.analysis_memory_mb.value())
        save_generation_settings(self.config)
        self.unloadModel.emit()
        if not continuation:
            self.visualizer.begin_recording()
            self.chat.set_analysis_memory_exceeded(False)
            self.chat.set_regenerate_available(False)
            self.simulation_transcript = [{"role": "world", "content": seed, "turn": 0}]
            self.visualizer.set_conversation(self.simulation_transcript)
            self.chat.clear_messages()
            self.chat.clear_latest_playback()
            self.chat.add_message("world", seed)
            self._simulation_bubbles.clear()
        self.chat.generating(True)
        self.chat.stats.setText(
            "∞ Simulation Thinking… loading the world and participant roles; the right pane records their visual signals."
        )
        transcript = self.simulation_transcript.copy() if continuation else None
        self.startInfiniteSimulation.emit(
            seed, self.config, self.current_model.blob_path, transcript
        )

    def continue_infinite_simulation(self) -> None:
        if not self.simulation_transcript:
            self.chat.stats.setText(
                "Start an Infinite Mode scenario before continuing it."
            )
            return
        seed = str(
            self.simulation_transcript[0].get(
                "content", "Continue the current scenario."
            )
        )
        self.start_infinite_simulation(seed, continuation=True)

    def _simulation_turn_started(self, role: str, turn: int) -> None:
        label = "World" if role == "world" else "Participant"
        bubble = self.chat.add_message(role, f"{label} {turn}: ")
        self._simulation_bubbles[(role, turn)] = bubble

    def _simulation_token(self, role: str, turn: int, text: str, frame: object) -> None:
        bubble = self._simulation_bubbles.get((role, turn))
        if bubble is not None:
            self.chat.append_message_text(bubble, text)
        if frame is not None:
            self.visualizer.apply_frame(frame)  # type: ignore[arg-type]

    def _simulation_turn_finished(self, role: str, text: str, turn: int) -> None:
        if text:
            self.simulation_transcript.append(
                {"role": role, "content": text, "turn": turn}
            )
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
        self.chat.set_analysis_available(bool(self.visualizer.analyzer.records))
        self.chat.set_analysis_memory_exceeded(self.visualizer.analysis_memory_exceeded)

    def _simulation_failed(self, error: str) -> None:
        LOG.error("%s", error)
        self.chat.stats.setText(error)
        self.chat.generating(False)
        self._show_repairable_error("Infinite simulation error", error)

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
        self.visualizer.exit_rewind_mode()
        self.chat.clear_messages()
        self.chat.clear_latest_playback()
        self.chat.set_analysis_available(False)
        self.chat.set_regenerate_available(False)
        self.chat.stats.setText("Conversation cleared.")

    def _on_token(self, text: str, frame: object) -> None:
        if self._assistant_bubble is not None:
            if self._awaiting_first_token:
                self._assistant_bubble.setText("")
                self._awaiting_first_token = False
            self.chat.append_message_text(self._assistant_bubble, text)
        self.visualizer.apply_frame(frame)  # type: ignore[arg-type]

    def _finished(self, stats: GenerationStats) -> None:
        text = (
            self._assistant_bubble.text() if self._assistant_bubble is not None else ""
        )
        if text and text != "Thinking…":
            self.history.append({"role": "assistant", "content": text})
            self.visualizer.set_conversation(self.history)
            # Rewind navigation is intentionally colocated with the graph,
            # rather than interrupting the chat transcript.
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
        self.chat.set_analysis_available(
            bool(text) and text != "Thinking…" and not stats["cancelled"]
        )
        self.chat.set_regenerate_available(
            bool(text) and text != "Thinking…" and not stats["cancelled"]
        )
        self._assistant_bubble = None
        self._awaiting_first_token = False

    def _failed(self, error: str) -> None:
        LOG.error("%s", error)
        self.chat.stats.setText(error)
        self.chat.generating(False)
        self.chat.set_analysis_available(False)
        self._assistant_bubble = None
        self._awaiting_first_token = False
        self._show_repairable_error("Generation error", error)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        QSettings().setValue("main_splitter_sizes", self._splitter.sizes())
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
