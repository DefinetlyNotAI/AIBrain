from __future__ import annotations

import hashlib
import struct
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

from src.models import model_validator
from src.models.model_info import ModelInfo
from src.models.model_validator import ModelValidator


class ModelValidatorTests(unittest.TestCase):
    def test_known_vision_models_are_rejected_before_llama_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gemma_blob = root / "gemma.gguf"
            qwen_blob = root / "qwen.gguf"
            gemma_blob.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            qwen_blob.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            models = [
                ModelInfo(
                    "gemma3",
                    "4b",
                    gemma_blob,
                    family="gemma3",
                    parameter_size="4B",
                    size_bytes=gemma_blob.stat().st_size,
                ),
                ModelInfo(
                    "qwen2.5vl",
                    "3b",
                    qwen_blob,
                    family="qwen2.5vl",
                    parameter_size="3B",
                    size_bytes=qwen_blob.stat().st_size,
                ),
            ]
            with (
                patch.object(model_validator, "CACHE_DIRECTORY", root / ".cache"),
                patch("src.models.model_validator.LlamaBackend") as backend_type,
            ):
                validated = ModelValidator.validate(
                    models, Event(), lambda *_: None, verify_backend=True
                )

        self.assertTrue(all(not model.available for model in validated))
        self.assertTrue(all(model.error is not None for model in validated))
        self.assertTrue(
            all(
                "vision-capable Ollama model" in (model.error or "")
                for model in validated
            )
        )
        backend_type.return_value.load.assert_not_called()

    def test_text_only_gemma3_variant_still_reaches_llama_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "model.gguf"
            blob.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            model = ModelInfo(
                "gemma3",
                "1b",
                blob,
                family="gemma3",
                parameter_size="1B",
                size_bytes=blob.stat().st_size,
            )
            with (
                patch.object(model_validator, "CACHE_DIRECTORY", root / ".cache"),
                patch("src.models.model_validator.LlamaBackend") as backend_type,
            ):
                validated = ModelValidator.validate(
                    [model], Event(), lambda *_: None, verify_backend=True
                )

        self.assertTrue(validated[0].available)
        backend_type.return_value.load.assert_called_once()

    def test_structural_validation_accepts_a_valid_gguf_without_loading_llama(
            self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            blob = Path(directory) / "model.gguf"
            blob.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            model = ModelInfo("demo", "latest", blob, size_bytes=blob.stat().st_size)
            progress: list[tuple[int, int, str]] = []

            validated = ModelValidator.validate(
                [model],
                Event(),
                lambda current, total, detail: progress.append(
                    (current, total, detail)
                ),
                verify_backend=False,
            )

        self.assertTrue(validated[0].available)
        self.assertIsNone(validated[0].error)
        self.assertEqual(progress, [(1, 1, "Checking demo:latest")])

    def test_validation_reuses_the_result_for_duplicate_blob_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            blob = Path(directory) / "model.gguf"
            blob.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            models = [
                ModelInfo("demo", "latest", blob, size_bytes=blob.stat().st_size),
                ModelInfo("demo", "copy", blob, size_bytes=blob.stat().st_size),
            ]

            validated = ModelValidator.validate(
                models,
                Event(),
                lambda *_: None,
                verify_backend=False,
            )

        self.assertTrue(all(model.available for model in validated))

    def test_digest_mismatch_is_reported_as_invalid_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            blob = Path(directory) / "model.gguf"
            blob.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            wrong_digest = "sha256:" + hashlib.sha256(b"other").hexdigest()
            model = ModelInfo(
                "demo",
                "latest",
                blob,
                size_bytes=blob.stat().st_size,
                digest=wrong_digest,
            )

            progress: list[str] = []
            validated = ModelValidator.validate(
                [model],
                Event(),
                lambda _current, _total, message: progress.append(message),
                verify_backend=False,
            )

        self.assertFalse(validated[0].available)
        self.assertIsNotNone(validated[0].error)
        assert validated[0].error is not None
        self.assertIn("Invalid HASH", validated[0].error)
        self.assertIn("Hashing demo:latest (start)", progress)
        self.assertIn("Hashing demo:latest (complete)", progress)

    def test_cache_reuses_full_validation_only_until_the_blob_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "model.gguf"
            blob.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            model = ModelInfo("demo", "latest", blob, size_bytes=blob.stat().st_size)
            with (
                patch.object(model_validator, "CACHE_DIRECTORY", root / ".cache"),
                patch(
                    "src.models.model_validator.OllamaDiscovery._validate_gguf",
                    return_value=None,
                ) as validate,
            ):
                ModelValidator.validate(
                    [model], Event(), lambda *_: None, verify_backend=False
                )
                ModelValidator.validate(
                    [model], Event(), lambda *_: None, verify_backend=False
                )
                self.assertEqual(validate.call_count, 1)

                blob.write_bytes(blob.read_bytes() + b"changed")
                ModelValidator.validate(
                    [model], Event(), lambda *_: None, verify_backend=False
                )

            self.assertEqual(validate.call_count, 2)

    def test_structural_cache_never_skips_a_requested_backend_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "model.gguf"
            blob.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            model = ModelInfo("demo", "latest", blob, size_bytes=blob.stat().st_size)
            with (
                patch.object(model_validator, "CACHE_DIRECTORY", root / ".cache"),
                patch(
                    "src.models.model_validator.OllamaDiscovery._validate_gguf",
                    return_value=None,
                ),
                patch("src.models.model_validator.LlamaBackend") as backend_type,
            ):
                backend_type.return_value.load.return_value = None
                ModelValidator.validate(
                    [model], Event(), lambda *_: None, verify_backend=False
                )
                ModelValidator.validate(
                    [model], Event(), lambda *_: None, verify_backend=True
                )

            backend_type.return_value.load.assert_called_once()

    def test_backend_validation_preserves_the_full_exception_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "model.gguf"
            blob.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            model = ModelInfo("demo", "latest", blob, size_bytes=blob.stat().st_size)
            with (
                patch.object(
                    model_validator, "CACHE_DIRECTORY", root / ".cache" / "validation"
                ),
                patch(
                    "src.models.model_validator.OllamaDiscovery._validate_gguf",
                    return_value=None,
                ),
                patch("src.models.model_validator.LlamaBackend") as backend_type,
            ):
                backend_type.return_value.load.side_effect = RuntimeError(
                    "unsupported model architecture"
                )
                validated = ModelValidator.validate(
                    [model], Event(), lambda *_: None, verify_backend=True
                )

        self.assertIsNotNone(validated[0].error)
        assert validated[0].error is not None
        self.assertIn("llama.cpp compatibility check failed", validated[0].error)
        self.assertIn("Traceback", validated[0].error)
        self.assertIn("unsupported model architecture", validated[0].error)

    def test_validation_cache_lives_in_the_repairable_validation_directory(
            self,
    ) -> None:
        self.assertEqual(model_validator.CACHE_DIRECTORY.name, "validation")
        self.assertEqual(model_validator.CACHE_DIRECTORY.parent.name, ".cache")
