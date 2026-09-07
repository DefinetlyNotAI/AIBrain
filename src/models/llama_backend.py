from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from .instrumented_backend import GenerationChunk, LogitMetrics
from .llama_runtime import load_llama_cpp
from .message_types import ChatMessage

if TYPE_CHECKING:
    from llama_cpp import Llama
    from llama_cpp.llama_types import ChatCompletionRequestMessage

LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    context_length: int = 4096
    gpu_layers: int = -1
    speed: float = 1.0


class GenerationBackend(Protocol):
    """Backend contract used by chat and infinite-simulation workers."""

    def load(self, path: Path, config: GenerationConfig, /) -> None: ...

    def stream_chat(
            self, messages: Sequence[ChatMessage], config: GenerationConfig, /
    ) -> Iterator[GenerationChunk]: ...

    def tokenize(self, text: str, /) -> list[int]: ...

    def unload(self) -> None: ...


_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*\s*$")


class _LogitTelemetryProbe:
    """Read raw model logits before sampler transforms without changing them."""

    def __init__(self, context_tokens: Callable[[], int]) -> None:
        self._context_tokens = context_tokens
        self._latest: LogitMetrics | None = None

    def __call__(self, _input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float32)
        vocabulary_size = int(values.size)

        if not vocabulary_size or not bool(np.isfinite(values).all()):
            self._latest = None
            return scores

        maximum = float(values.max())
        weights = np.exp(values - maximum)
        total = float(weights.sum(dtype=np.float64))

        if not np.isfinite(total) or total <= 0:
            self._latest = None
            return scores

        probabilities = weights / total
        entropy_nats = float(
            np.log(total)
            - np.sum(
                probabilities * (values - maximum),
                dtype=np.float64,
            )
        )

        maximum_entropy = float(np.log(vocabulary_size))
        entropy_nats = min(maximum_entropy, max(0.0, entropy_nats))

        candidate_count = min(5, vocabulary_size)
        top = np.partition(
            probabilities,
            vocabulary_size - candidate_count,
        )[-candidate_count:]

        top.sort()

        top_probability = float(top[-1])
        second_probability = (
            float(top[-2])
            if candidate_count > 1
            else 0.0
        )

        self._latest = LogitMetrics(
            vocabulary_size=vocabulary_size,
            context_tokens=max(0, int(self._context_tokens())),
            raw_entropy_bits=entropy_nats / np.log(2.0),
            raw_normalized_entropy=min(
                1.0,
                max(
                    0.0,
                    entropy_nats / max(maximum_entropy, 1e-6),
                ),
            ),
            raw_top_probability=top_probability,
            raw_top_five_mass=min(
                1.0,
                max(
                    0.0,
                    float(top.sum(dtype=np.float64)),
                ),
            ),
            raw_confidence_margin=min(
                1.0,
                max(
                    0.0,
                    top_probability - second_probability,
                ),
            ),
        )

        return scores

    def take(self) -> LogitMetrics | None:
        latest = self._latest
        self._latest = None
        return latest


def sentence_grace_config(config: GenerationConfig) -> GenerationConfig:
    """Permit a short overrun so a requested response can finish its sentence."""
    return replace(
        config,
        max_tokens=min(config.max_tokens + 64, 8192),
    )


def reached_sentence_end(
        text: str,
        token_count: int,
        requested_max_tokens: int,
) -> bool:
    return (
            token_count >= requested_max_tokens
            and bool(_SENTENCE_END_RE.search(text))
    )


class LlamaBackend:
    """Run GGUF chat generation through the managed llama.cpp runtime."""

    def __init__(self) -> None:
        self._llm: Llama | None = None
        self.loaded_path: Path | None = None

    def load(self, path: Path, config: GenerationConfig) -> None:
        """Load a GGUF model, falling back to CPU layers if GPU loading fails."""
        if self.loaded_path == path and self._llm is not None:
            return

        self.unload()

        try:
            llama_cpp = load_llama_cpp()
            llama_class = llama_cpp.Llama
        except ImportError as exc:
            raise RuntimeError(
                r"llama-cpp-python is not installed. Run: python cli\installer.py"
            ) from exc

        LOG.info(
            "Loading GGUF directly: %s (GPU layers: %s)",
            path,
            config.gpu_layers,
        )

        try:
            self._llm = llama_class(
                model_path=str(path),
                n_ctx=config.context_length,
                n_gpu_layers=config.gpu_layers,
                verbose=False,
            )
        except Exception as exc:
            if config.gpu_layers == 0:
                raise

            LOG.warning(
                "Model load with %s GPU layer(s) failed (%s); "
                "retrying the same GGUF on CPU layers",
                config.gpu_layers,
                exc,
                exc_info=True,
            )

            self._llm = llama_class(
                model_path=str(path),
                n_ctx=config.context_length,
                n_gpu_layers=0,
                verbose=False,
            )

        self.loaded_path = path

    @staticmethod
    def _chat_request_message(role: str, content: str) -> ChatCompletionRequestMessage:
        return cast(
            "ChatCompletionRequestMessage",
            cast(
                object,
                {
                    "role": role,
                    "content": content,
                },
            ),
        )

    def stream_chat(
            self,
            messages: Sequence[ChatMessage],
            config: GenerationConfig,
    ) -> Iterator[GenerationChunk]:
        """Stream chat tokens and their associated raw-logit telemetry."""
        if self._llm is None:
            raise RuntimeError("No model is loaded")

        llm = self._llm
        telemetry = _LogitTelemetryProbe(lambda: llm.n_tokens)

        request_messages: list[ChatCompletionRequestMessage] = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                request_message = self._chat_request_message("system", content)
            elif role == "user":
                request_message = self._chat_request_message("user", content)
            else:
                request_message = self._chat_request_message("assistant", content)

            request_messages.append(request_message)

        llama_cpp = load_llama_cpp()

        response = llm.create_chat_completion(
            messages=request_messages,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stream=True,
            logits_processor=llama_cpp.LogitsProcessorList([telemetry]),
        )

        if isinstance(response, dict):
            raise TypeError(
                "llama.cpp returned a non-streaming chat response"
            )

        for chunk in response:
            if not isinstance(chunk, dict):
                continue

            choices = chunk.get("choices")

            if not isinstance(choices, list) or not choices:
                continue

            choice = choices[0]

            if not isinstance(choice, dict):
                continue

            delta = choice.get("delta")

            if not isinstance(delta, dict):
                continue

            token = delta.get("content")

            if isinstance(token, str) and token:
                yield GenerationChunk(
                    token,
                    tuple(self.tokenize(token)),
                    telemetry.take(),
                )

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text with the currently loaded model."""
        if self._llm is None:
            return []

        return self._llm.tokenize(
            text.encode("utf-8"),
            add_bos=False,
        )

    def unload(self) -> None:
        """Release the currently loaded llama.cpp model."""
        self._llm = None
        self.loaded_path = None
