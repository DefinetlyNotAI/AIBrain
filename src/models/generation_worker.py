from __future__ import annotations

from pathlib import Path
from threading import Event
from time import monotonic

from PySide6.QtCore import QObject, Signal, Slot

from .instrumented_backend import ActivationFrame, ActivitySource
from .llama_backend import GenerationConfig, LlamaBackend


class GenerationWorker(QObject):
    token = Signal(str, object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, backend: LlamaBackend) -> None:
        super().__init__()
        self.backend = backend
        self._cancelled = Event()

    @Slot(object, object, object)
    def generate(self, messages: list[dict[str, str]], config: GenerationConfig, model_path: Path) -> None:
        self._cancelled.clear()
        start = monotonic()
        generated_tokens = 0
        previous_token_at = monotonic()
        normal_interval = .04
        try:
            self.backend.load(model_path, config)
            prompt_tokens = sum(len(self.backend.tokenize(message["content"])) for message in messages)
            for text in self.backend.stream_chat(messages, config):
                if self._cancelled.is_set():
                    break
                now = monotonic()
                normal_interval = normal_interval * .8 + (now - previous_token_at) * .2
                token_ids = self.backend.tokenize(text)
                generated_tokens += max(1, len(token_ids))
                frame = ActivationFrame(token_ids[-1] if token_ids else None, text, generated_tokens,
                                        ActivitySource.SIMULATION)
                self.token.emit(text, frame)
                if config.speed < 1.0:
                    self._cancelled.wait(max(0.0, normal_interval * (1.0 / config.speed - 1.0)))
                previous_token_at = monotonic()
            elapsed = monotonic() - start
            self.finished.emit({"prompt_tokens": prompt_tokens, "generated_tokens": generated_tokens,
                                "seconds": elapsed, "cancelled": self._cancelled.is_set()})
        except Exception as exc:
            self.failed.emit(f"Generation failed: {exc}")

    @Slot()
    def cancel(self) -> None:
        self._cancelled.set()

    @Slot()
    def unload(self) -> None:
        self.backend.unload()
