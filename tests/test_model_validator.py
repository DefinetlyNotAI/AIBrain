from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from threading import Event

from src.models.model_info import ModelInfo
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
