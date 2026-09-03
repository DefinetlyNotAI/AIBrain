from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.models.llama_backend import GenerationConfig, LlamaBackend
from src.models.llama_runtime import _handle_native_log


class LlamaBackendLoggingTests(unittest.TestCase):
    def test_native_log_handler_accepts_the_three_argument_llama_abi(self) -> None:
        with self.assertLogs("src.models.llama_runtime", level="ERROR") as logs:
            _handle_native_log(4, b"model load failed", None)

        self.assertEqual(logs.output, ["ERROR:src.models.llama_runtime:llama.cpp: model load failed"])

    def test_native_log_handler_ignores_empty_and_non_error_messages(self) -> None:
        with self.assertNoLogs("src.models.llama_runtime", level="ERROR"):
            _handle_native_log(1, None, None)
            _handle_native_log(1, b"...", None)
            _handle_native_log(1, b"model metadata loaded", None)

    def test_load_retries_cpu_when_automatic_gpu_offload_fails(self) -> None:
        calls: list[int] = []

        class FakeLlama:
            def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
                calls.append(kwargs["n_gpu_layers"])
                if kwargs["n_gpu_layers"] != 0:
                    raise RuntimeError("GPU offload unavailable")

        fake_module = SimpleNamespace(
            Llama=FakeLlama,
            llama_log_callback=lambda callback: callback,
            llama_log_set=lambda _callback, _context: None,
        )
        with (
            patch.dict("sys.modules", {"llama_cpp": fake_module}),
            patch("src.models.llama_runtime._LOG_CALLBACK", None),
        ):
            backend = LlamaBackend()
            backend.load(Path("model.gguf"), GenerationConfig(gpu_layers=-1))

        self.assertEqual(calls, [-1, 0])
        self.assertEqual(backend.loaded_path, Path("model.gguf"))
