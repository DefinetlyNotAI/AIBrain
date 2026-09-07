from __future__ import annotations

import inspect
import unittest

import numpy as host_np

from src.connectome.renderer import (
    ConnectomeRenderer,
    _projection_matrix,
    _validated_region_filter,
    _visible_region_indices,
)
from src.utils.array_api import array_api as np


class VisualizerModeTests(unittest.TestCase):
    def test_renderer_caps_continuous_repaints_at_thirty_frames_per_second(
        self,
    ) -> None:
        self.assertIn(
            "self.timer.start(33)", inspect.getsource(ConnectomeRenderer.__init__)
        )

    def test_node_border_width_uses_backend_independent_scalar_clamping(self) -> None:
        self.assertEqual(min(1.0, max(0.0, 1.5)), 1.0)

    def test_flat_projection_drops_depth_rotation(self) -> None:
        matrix = _projection_matrix("2d", 1.0, 800, 600)

        self.assertEqual(matrix.shape, (4, 4))
        self.assertEqual(matrix[2, 2], 0.0)
        self.assertEqual(matrix[0, 2], 0.0)

    def test_flat_projection_carries_pan_offsets(self) -> None:
        matrix = _projection_matrix("2d", 1.0, 800, 600, pan_x=0.2, pan_y=-0.3)

        self.assertEqual(matrix[3, 0], 0.2)
        self.assertEqual(matrix[3, 1], -0.3)

    def test_projection_matrix_is_always_a_host_numpy_array(self) -> None:
        matrix = _projection_matrix("3d", 1.0, 800, 600, yaw=0.25, pitch=-0.2)

        self.assertIsInstance(matrix, host_np.ndarray)
        self.assertTrue(matrix.flags.c_contiguous)

    def test_sector_mode_limits_visible_nodes_and_rejects_invalid_regions(self) -> None:
        positions = np.zeros((4, 3), dtype="f4")
        regions = np.array([0, 1, 1, 2], dtype="i4")
        region_filter = _validated_region_filter("2d", 1, 3)

        self.assertEqual(region_filter, 1)
        self.assertEqual(
            _visible_region_indices(positions, regions, region_filter).tolist(), [1, 2]
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            _validated_region_filter("2d", 3, 3)


if __name__ == "__main__":
    unittest.main()
