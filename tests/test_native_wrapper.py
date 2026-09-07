from __future__ import annotations

import unittest

import numpy as np

from src.native.wrapper.connectome_kernels import _DLL_PATH, native


class NativeConnectomeTests(unittest.TestCase):
    def test_wrapper_uses_the_standardized_dll_name(self) -> None:
        self.assertEqual(_DLL_PATH.name, "aibrain.connectome.dll")

    def test_region_activity_matches_expected_sums(self) -> None:
        values = np.array([0.4, 0.7, 0.2, 0.9], dtype=np.float32)
        regions = np.array([0, 1, 1, 2], dtype=np.int16)

        sums, active = native.regions(values, regions, 3, threshold=0.3)

        np.testing.assert_allclose(sums, np.array([0.4, 0.9, 0.9], dtype=np.float32))
        self.assertEqual(active, 3)


if __name__ == "__main__":
    unittest.main()
