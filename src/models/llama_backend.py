from __future__ import annotations

import logging
import re
from dataclasses import replace
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .llama_runtime import load_llama_cpp

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


_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*\s*$")


def sentence_grace_config(config: GenerationConfig) -> GenerationConfig:
    """Permit a short overrun so a requested response can finish its sentence."""
    return replace(config, max_tokens=min(config.max_tokens + 64, 8192))


def reached_sentence_end(text: str, token_count: int, requested_max_tokens: int) -> bool:
    return token_count >= requested_max_tokens and bool(_SENTENCE_END_RE.search(text))


class LlamaBackend:
    def __init__(self) -> None:
        self._llm: Llama | None = None
        self.loaded_path: Path | None = None

    def load(self, path: Path, config: GenerationConfig) -> None:
        if self.loaded_path == path and self._llm is not None:
            return

        self.unload()

        try:
            llama_cpp = load_llama_cpp()
            Llama = llama_cpp.Llama
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
            self._llm = Llama(
                model_path=str(path),
                n_ctx=config.context_length,
                n_gpu_layers=config.gpu_layers,
                verbose=False,
            )
        except Exception as exc:
            if config.gpu_layers == 0:
                raise
            LOG.warning(
                "Model load with %s GPU layer(s) failed (%s); retrying the same GGUF on CPU layers",
                config.gpu_layers,
                exc,
                exc_info=True,
            )
            self._llm = Llama(
                model_path=str(path),
                n_ctx=config.context_length,
                n_gpu_layers=0,
                verbose=False,
            )

        self.loaded_path = path

    def stream_chat(
            self,
            messages: list[ChatCompletionRequestMessage],
            config: GenerationConfig,
    ) -> Iterator[str]:
        if self._llm is None:
            raise RuntimeError("No model is loaded")

        response = self._llm.create_chat_completion(
            messages=messages,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stream=True,
        )

        for chunk in response:
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
                yield token

    def tokenize(self, text: str) -> list[int]:
        if self._llm is None:
            return []

        return self._llm.tokenize(
            text.encode("utf-8"),
            add_bos=False,
        )

    def unload(self) -> None:
        self._llm = None
        self.loaded_path = None
