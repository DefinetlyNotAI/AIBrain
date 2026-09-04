from __future__ import annotations

import inspect
from types import SimpleNamespace
import unittest

import numpy as host_np

from src.connectome.renderer import ConnectomeRenderer
from src.utils.array_api import array_api as np


class VisualizerModeTests(unittest.TestCase):
    def test_renderer_caps_continuous_repaints_at_thirty_frames_per_second(self) -> None:
        self.assertIn("self.timer.start(33)", inspect.getsource(ConnectomeRenderer.__init__))

    def test_node_border_width_uses_backend_independent_scalar_clamping(self) -> None:
        renderer = SimpleNamespace(update=lambda: None)

        ConnectomeRenderer.set_node_borders(renderer, True, 1.5)  # type: ignore[arg-type]

        self.assertTrue(renderer.show_node_borders)
        self.assertEqual(renderer.node_border_width, 1.0)

    def test_flat_projection_drops_depth_rotation(self) -> None:
        renderer = SimpleNamespace(view_mode="2d", zoom=1.0, width=lambda: 800, height=lambda: 600)

        matrix = ConnectomeRenderer._mvp(renderer)  # type: ignore[arg-type]

        self.assertEqual(matrix.shape, (4, 4))
        self.assertEqual(matrix[2, 2], 0.0)
        self.assertEqual(matrix[0, 2], 0.0)

    def test_flat_projection_carries_pan_offsets(self) -> None:
        renderer = SimpleNamespace(view_mode="2d", zoom=1.0, pan_x=.2, pan_y=-.3, width=lambda: 800, height=lambda: 600)

        matrix = ConnectomeRenderer._mvp(renderer)  # type: ignore[arg-type]

        self.assertEqual(matrix[3, 0], .2)
        self.assertEqual(matrix[3, 1], -.3)

    def test_projection_matrix_is_always_a_host_numpy_array(self) -> None:
        renderer = SimpleNamespace(
            view_mode="3d",
            zoom=1.0,
            yaw=.25,
            pitch=-.2,
            width=lambda: 800,
            height=lambda: 600,
        )

        matrix = ConnectomeRenderer._mvp(renderer)  # type: ignore[arg-type]

        self.assertIsInstance(matrix, host_np.ndarray)
        self.assertTrue(matrix.flags.c_contiguous)

    def test_sector_mode_limits_visible_nodes_and_rejects_invalid_regions(self) -> None:
        renderer = SimpleNamespace(
            graph=SimpleNamespace(
                positions=np.zeros((4, 3), dtype="f4"),
                regions=np.array([0, 1, 1, 2], dtype="i4"),
                region_names=("A", "B", "C"),
            ),
            view_mode="3d",
            region_filter=None,
            update=lambda: None,
        )

        ConnectomeRenderer.set_projection_mode(renderer, "2d", 1)  # type: ignore[arg-type]

        self.assertEqual(renderer.view_mode, "2d")
        self.assertEqual(renderer.region_filter, 1)
        self.assertEqual(ConnectomeRenderer._visible_indices(renderer).tolist(), [1, 2])  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "outside"):
            ConnectomeRenderer.set_projection_mode(renderer, "2d", 3)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
