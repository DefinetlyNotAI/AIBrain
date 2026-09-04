from __future__ import annotations

import builtins
import importlib
import unittest
from unittest.mock import patch

from src.connectome.analysis import _normal
from src.utils.array_api import BACKEND_NAME, array_api, to_numpy


class ArrayApiTests(unittest.TestCase):
    def test_unified_backend_performs_array_work(self) -> None:
        values = array_api.asarray([1.0, 2.0, 3.0], dtype=array_api.float32)

        self.assertEqual(float(values.mean()), 2.0)
        self.assertEqual(to_numpy(values).tolist(), [1.0, 2.0, 3.0])
        self.assertIn(BACKEND_NAME, {"NumPy / CPU", "CuPy / CUDA"})

    def test_missing_optional_cupy_backend_falls_back_without_logging(self) -> None:
        module = importlib.import_module("src.utils.array_api")
        original_import = builtins.__import__

        def import_without_cupy(name: str, *args: object, **kwargs: object) -> object:
            if name == "cupy":
                raise ModuleNotFoundError("No module named 'cupy'")
            return original_import(name, *args, **kwargs)

        with self.assertNoLogs(module.__name__, level="DEBUG"), patch(
            "builtins.__import__", side_effect=import_without_cupy
        ):
            backend, backend_name, fallback_reason = module._select_backend()

        self.assertIs(backend, module._numpy)
        self.assertEqual(backend_name, "NumPy / CPU")
        self.assertEqual(fallback_reason, "No module named 'cupy'")

    def test_normal_sampler_uses_the_generator_api_shared_by_cupy(self) -> None:
        class Generator:
            def __init__(self) -> None:
                self.requested_size: tuple[int, ...] | None = None

            def standard_normal(self, size: tuple[int, ...]) -> object:
                self.requested_size = size
                return array_api.ones(size, dtype=array_api.float32)

            def normal(self, *_: object) -> object:
                raise AssertionError("NumPy-only normal() must not be called")

        generator = Generator()
        sample = _normal(generator, 0.5, 0.25, (2, 3))

        self.assertEqual(generator.requested_size, (2, 3))
        self.assertEqual(to_numpy(sample).tolist(), [[0.75] * 3] * 2)


if __name__ == "__main__":
    unittest.main()
