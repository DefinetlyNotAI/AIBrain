from __future__ import annotations

import unittest

from src.utils.array_api import BACKEND_NAME, array_api, to_numpy


class ArrayApiTests(unittest.TestCase):
    def test_unified_backend_performs_array_work(self) -> None:
        values = array_api.asarray([1.0, 2.0, 3.0], dtype=array_api.float32)

        self.assertEqual(float(values.mean()), 2.0)
        self.assertEqual(to_numpy(values).tolist(), [1.0, 2.0, 3.0])
        self.assertIn(BACKEND_NAME, {"NumPy / CPU", "CuPy / CUDA"})


if __name__ == "__main__":
    unittest.main()
