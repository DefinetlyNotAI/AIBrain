from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

from src.models.model_info import ModelInfo
from src.models import model_validator
from src.models.model_validator import ModelValidator


class ModelValidatorTests(unittest.TestCase):
    def test_structural_validation_accepts_a_valid_gguf_without_loading_llama(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            blob = Path(directory) / "model.gguf"
            blob.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            model = ModelInfo("demo", "latest", blob, size_bytes=blob.stat().st_size)
            progress: list[tuple[int, int, str]] = []

            validated = ModelValidator.validate(
                [model],
                Event(),
                lambda current, total, detail: progress.append((current, total, detail)),
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

    def test_cache_reuses_full_validation_only_until_the_blob_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "model.gguf"
            blob.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            model = ModelInfo("demo", "latest", blob, size_bytes=blob.stat().st_size)
            with (
                patch.object(model_validator, "CACHE_DIRECTORY", root / ".cache"),
                patch("src.models.model_validator.OllamaDiscovery._validate_gguf", return_value=None) as validate,
            ):
                ModelValidator.validate([model], Event(), lambda *_: None, verify_backend=False)
                ModelValidator.validate([model], Event(), lambda *_: None, verify_backend=False)
                self.assertEqual(validate.call_count, 1)

                blob.write_bytes(blob.read_bytes() + b"changed")
                ModelValidator.validate([model], Event(), lambda *_: None, verify_backend=False)

            self.assertEqual(validate.call_count, 2)

    def test_structural_cache_never_skips_a_requested_backend_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "model.gguf"
            blob.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            model = ModelInfo("demo", "latest", blob, size_bytes=blob.stat().st_size)
            with (
                patch.object(model_validator, "CACHE_DIRECTORY", root / ".cache"),
                patch("src.models.model_validator.OllamaDiscovery._validate_gguf", return_value=None),
                patch("src.models.llama_backend.LlamaBackend") as backend_type,
            ):
                backend_type.return_value.load.return_value = None
                ModelValidator.validate([model], Event(), lambda *_: None, verify_backend=False)
                ModelValidator.validate([model], Event(), lambda *_: None, verify_backend=True)

            backend_type.return_value.load.assert_called_once()
