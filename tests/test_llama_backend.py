from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from llama_cpp import Llama

from src.models.instrumented_backend import (
    ActivitySource,
    GenerationChunk,
    LogitMetrics,
    realtime_activation_frame,
)
from src.models.llama_backend import (
    GenerationConfig,
    LlamaBackend,
    _LogitTelemetryProbe,
)
from src.models.llama_runtime import _handle_native_log


class LlamaBackendLoggingTests(unittest.TestCase):
    def test_logit_probe_measures_distribution_without_changing_scores(self) -> None:
        probe = _LogitTelemetryProbe(lambda: 42)
        scores = np.zeros(5, dtype=np.float32)

        returned = probe(np.array([1, 2], dtype=np.int32), scores)
        metrics = probe.take()

        self.assertIs(returned, scores)
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertEqual(metrics.vocabulary_size, 5)
        self.assertEqual(metrics.context_tokens, 42)
        self.assertAlmostEqual(metrics.raw_entropy_bits, np.log2(5), places=6)
        self.assertAlmostEqual(metrics.raw_normalized_entropy, 1.0, places=6)
        self.assertAlmostEqual(metrics.raw_top_probability, 0.2, places=6)
        self.assertAlmostEqual(metrics.raw_top_five_mass, 1.0, places=6)
        self.assertAlmostEqual(metrics.raw_confidence_margin, 0.0, places=6)

    def test_streamed_chunk_carries_live_logit_telemetry(self) -> None:
        fake_llama = Mock(spec=Llama)
        fake_llama.n_tokens = 19

        def tokenize(_text: bytes, *, add_bos: bool) -> list[int]:
            self.assertFalse(add_bos)
            return [27]

        def create_chat_completion(**kwargs):
            processor = kwargs["logits_processor"][0]
            scores = np.array([0.0, 1.0, 2.0], dtype=np.float32)
            processor(np.array([4, 5], dtype=np.int32), scores)
            return iter([{"choices": [{"delta": {"content": "hello"}}]}])

        fake_llama.tokenize.side_effect = tokenize
        fake_llama.create_chat_completion.side_effect = create_chat_completion

        backend = LlamaBackend()
        backend._llm = fake_llama

        chunks = list(backend.stream_chat([], GenerationConfig()))

        self.assertEqual(len(chunks), 1)
        self.assertIsInstance(chunks[0], GenerationChunk)
        self.assertEqual(chunks[0].text, "hello")
        self.assertEqual(chunks[0].retokenized_ids, (27,))
        self.assertIsNotNone(chunks[0].logits)
        assert chunks[0].logits is not None
        self.assertEqual(chunks[0].logits.context_tokens, 19)
        self.assertGreater(chunks[0].logits.raw_top_probability, 0.6)

    def test_logit_probe_rejects_non_finite_measurements(self) -> None:
        probe = _LogitTelemetryProbe(lambda: 42)
        scores = np.array([0.0, np.nan, 1.0], dtype=np.float32)

        returned = probe(np.array([1, 2], dtype=np.int32), scores)

        self.assertIs(returned, scores)
        self.assertIsNone(probe.take())

    def test_realtime_frame_keeps_session_step_separate_from_reply_progress(
        self,
    ) -> None:
        chunk = GenerationChunk(
            "token",
            (8,),
            LogitMetrics(32_000, 1_024, 7.5, 0.6, 0.25, 0.55, 0.1),
        )

        frame = realtime_activation_frame(
            chunk,
            501,
            output_tokens=1,
            max_tokens=100,
            context_limit=4_096,
            stream_latency_seconds=0.02,
            retokenized_tokens_per_second=50.0,
            recent_output_occurrences=0,
        )

        self.assertEqual(frame.source, ActivitySource.REAL_TIME)
        self.assertEqual(frame.step, 501)
        self.assertAlmostEqual(frame.regions["Requested output progress"], 0.01)
        self.assertAlmostEqual(frame.regions["Context utilisation"], 0.25)
        self.assertEqual(frame.metrics["output_tokens"], 1.0)
        self.assertEqual(frame.metrics["raw_logits_available"], 1.0)

    def test_realtime_frame_marks_a_missing_logit_snapshot(self) -> None:
        frame = realtime_activation_frame(
            GenerationChunk("visible", (4,), None),
            1,
            max_tokens=100,
            context_limit=4_096,
            stream_latency_seconds=0.02,
            retokenized_tokens_per_second=50.0,
            recent_output_occurrences=0,
        )

        self.assertEqual(frame.metrics["raw_logits_available"], 0.0)
        self.assertEqual(frame.regions["Raw-logit entropy"], 0.0)
        self.assertGreater(frame.regions["Stream latency"], 0.0)

    def test_native_log_handler_accepts_the_three_argument_llama_abi(self) -> None:
        with self.assertLogs("src.models.llama_runtime", level="ERROR") as logs:
            _handle_native_log(4, b"model load failed", None)

        self.assertEqual(
            logs.output, ["ERROR:src.models.llama_runtime:llama.cpp: model load failed"]
        )

    def test_native_log_handler_ignores_empty_and_non_error_messages(self) -> None:
        with self.assertNoLogs("src.models.llama_runtime", level="ERROR"):
            _handle_native_log(1, None, None)
            _handle_native_log(1, b"...", None)
            _handle_native_log(1, b"model metadata loaded", None)

    def test_load_retries_cpu_when_automatic_gpu_offload_fails(self) -> None:
        calls: list[int] = []

        class FakeLlama:
            def __init__(self, **kwargs) -> None:
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
