from __future__ import annotations

import unittest

from src.models.llama_backend import _handle_native_log


class LlamaBackendLoggingTests(unittest.TestCase):
    def test_native_log_handler_accepts_the_three_argument_llama_abi(self) -> None:
        with self.assertLogs("src.models.llama_backend", level="ERROR") as logs:
            _handle_native_log(2, b"model load failed", None)

        self.assertEqual(logs.output, ["ERROR:src.models.llama_backend:llama.cpp: model load failed"])

    def test_native_log_handler_ignores_empty_and_non_error_messages(self) -> None:
        with self.assertNoLogs("src.models.llama_backend", level="ERROR"):
            _handle_native_log(1, None, None)
            _handle_native_log(1, b"...", None)
            _handle_native_log(1, b"model metadata loaded", None)
