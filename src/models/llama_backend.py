from __future__ import annotations

import ctypes
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llama_cpp import Llama
    from llama_cpp.llama_types import ChatCompletionRequestMessage

LOG = logging.getLogger(__name__)

_NATIVE_LOG_CALLBACK: Any = None


def _handle_native_log(_level: int, text: bytes | None, _user_data: object) -> None:
    """Log llama.cpp errors without allowing exceptions across the C boundary."""
    if not text:
        return

    try:
        message = text.decode("utf-8", errors="replace").strip()
        lower = message.lower()
        if message and message.strip(".") and (
                "error" in lower
                or "failed" in lower
                or "unknown model architecture" in lower
        ):
            LOG.error("llama.cpp: %s", message)
    except Exception:
        # A C callback must never propagate a Python exception.
        return


@dataclass(slots=True)
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    context_length: int = 4096
    gpu_layers: int = -1
    speed: float = 1.0


class LlamaBackend:
    def __init__(self) -> None:
        self._llm: Llama | None = None
        self.loaded_path: Path | None = None

    def load(self, path: Path, config: GenerationConfig) -> None:
        if self.loaded_path == path and self._llm is not None:
            return

        self.unload()

        try:
            from llama_cpp import Llama, llama_log_callback, llama_log_set
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Run: py -3.11 install.py"
            ) from exc

        LOG.info(
            "Loading GGUF directly: %s (GPU layers: %s)",
            path,
            config.gpu_layers,
        )

        @llama_log_callback
        def native_log(_level: int, text: bytes | None, _user_data: object) -> None:
            """Accept llama.cpp's level, message, and user-data callback ABI."""
            _handle_native_log(_level, text, _user_data)

        global _NATIVE_LOG_CALLBACK
        _NATIVE_LOG_CALLBACK = native_log

        llama_log_set(
            _NATIVE_LOG_CALLBACK,
            ctypes.c_void_p(),
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
