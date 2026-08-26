from __future__ import annotations

import unittest

import numpy as np

from src.native.wrapper.connectome import native


class NativeConnectomeTests(unittest.TestCase):
    def test_region_activity_matches_expected_sums(self) -> None:
        values = np.array([.4, .7, .2, .9], dtype=np.float32)
        regions = np.array([0, 1, 1, 2], dtype=np.int16)

        sums, active = native.regions(values, regions, 3, threshold=.3)

        np.testing.assert_allclose(sums, np.array([.4, .9, .9], dtype=np.float32))
        self.assertEqual(active, 3)


if __name__ == "__main__":
    unittest.main()
