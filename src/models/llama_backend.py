from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    context_length: int = 4096
    gpu_layers: int = -1


class LlamaBackend:
    def __init__(self) -> None:
        self._llm: object | None = None
        self.loaded_path: Path | None = None

    def load(self, path: Path, config: GenerationConfig) -> None:
        if self.loaded_path == path and self._llm is not None:
            return
        self.unload()
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python is not installed. Run: py -3.11 install.py") from exc
        LOG.info("Loading GGUF directly: %s (GPU layers: %s)", path, config.gpu_layers)
        self._llm = Llama(model_path=str(path), n_ctx=config.context_length, n_gpu_layers=config.gpu_layers,
                          verbose=False)
        self.loaded_path = path

    def stream_chat(self, messages: list[dict[str, str]], config: GenerationConfig) -> Iterator[str]:
        if self._llm is None:
            raise RuntimeError("No model is loaded")
        response = self._llm.create_chat_completion(messages=messages, temperature=config.temperature,
                                                    top_p=config.top_p, max_tokens=config.max_tokens, stream=True)
        for chunk in response:
            choice = chunk.get("choices", [{}])[0]
            token = choice.get("delta", {}).get("content", "")
            if token:
                yield token

    def tokenize(self, text: str) -> list[int]:
        if self._llm is None:
            return []
        return self._llm.tokenize(text.encode("utf-8"), add_bos=False)

    def unload(self) -> None:
        self._llm = None
        self.loaded_path = None
