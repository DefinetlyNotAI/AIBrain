from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from threading import Event
from time import monotonic

from PySide6.QtCore import QObject, Signal, Slot

from .instrumented_backend import realtime_activation_frame, retokenized_throughput
from .llama_backend import GenerationConfig, LlamaBackend, reached_sentence_end, sentence_grace_config

LOG = logging.getLogger(__name__)


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
            response_chunks: list[str] = []
            recent_output: deque[str] = deque(maxlen=32)
            for chunk in self.backend.stream_chat(messages, sentence_grace_config(config)):
                if self._cancelled.is_set():
                    break
                now = monotonic()
                latency = now - previous_token_at
                normal_interval = normal_interval * .8 + latency * .2
                text = chunk.text
                retokenized_ids = chunk.retokenized_ids
                retokenized_count = len(retokenized_ids)
                generated_tokens += max(1, retokenized_count)
                response_chunks.append(text)
                recent_output_occurrences = recent_output.count(text)
                frame = realtime_activation_frame(
                    chunk,
                    generated_tokens,
                    max_tokens=config.max_tokens,
                    context_limit=config.context_length,
                    stream_latency_seconds=latency,
                    retokenized_tokens_per_second=retokenized_throughput(
                        retokenized_count, latency
                    ),
                    recent_output_occurrences=recent_output_occurrences,
                )
                recent_output.append(text)
                self.token.emit(text, frame)
                if reached_sentence_end("".join(response_chunks), generated_tokens, config.max_tokens):
                    break
                if config.speed < 1.0:
                    self._cancelled.wait(max(0.0, normal_interval * (1.0 / config.speed - 1.0)))
                previous_token_at = monotonic()
            elapsed = monotonic() - start
            self.finished.emit({"prompt_tokens": prompt_tokens, "generated_tokens": generated_tokens,
                                "seconds": elapsed, "cancelled": self._cancelled.is_set()})
        except Exception as exc:
            LOG.exception("Local model generation failed")
            self.failed.emit(f"Generation failed: {exc}")

    @Slot()
    def cancel(self) -> None:
        self._cancelled.set()

    @Slot()
    def unload(self) -> None:
        self.backend.unload()
