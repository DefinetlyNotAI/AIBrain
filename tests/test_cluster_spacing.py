"""Cluster-spacing graph contract."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.connectome.activity import ACTIVITY_DECAY_SECONDS, ActivityField
from src.connectome.generator import build_connectome
from src.models.instrumented_backend import ActivationFrame, ActivitySource


class ClusterSpacingTests(unittest.TestCase):
    def test_generator_supports_random_backends_without_choice(self) -> None:
        graph = build_connectome("startup-regression", "Medium")

        self.assertEqual(graph.positions.shape, (11000, 3))
        self.assertEqual(graph.regions.shape, (11000,))
        self.assertGreater(len(graph.edges), 0)

    def test_activity_mapping_uses_measured_channels_without_token_randomization(
        self,
    ) -> None:
        graph = build_connectome("activity-regression", "Low")
        field = ActivityField(graph)
        measurements = {
            name: (index + 1) / len(graph.region_names)
            for index, name in enumerate(graph.region_names)
        }

        field.update(
            ActivationFrame(
                "token",
                1,
                ActivitySource.REAL_TIME,
                regions=measurements,
            )
        )

        self.assertGreater(np.count_nonzero(field.values), 0)
        for region, name in enumerate(graph.region_names):
            region_values = field.values[graph.regions == region]
            self.assertTrue(np.allclose(region_values, measurements[name]))

        first = field.values.copy()
        field.update(
            ActivationFrame(
                "different text",
                500,
                ActivitySource.REAL_TIME,
                regions=measurements,
            )
        )
        self.assertTrue(np.array_equal(first, field.values))

    def test_activity_fades_to_about_one_third_in_a_tenth_of_a_second(self) -> None:
        graph = build_connectome("decay-regression", "Low")
        field = ActivityField(graph)
        field.values.fill(1.0)
        field.last_time = 10.0

        with patch(
            "src.connectome.activity.monotonic",
            return_value=10.0 + ACTIVITY_DECAY_SECONDS,
        ):
            field.decay()

        self.assertAlmostEqual(float(field.values[0]), float(np.exp(-1)), places=5)

    def test_display_controls_do_not_rewrite_measured_channel_values(self) -> None:
        graph = build_connectome("measurement-display-separation", "Low")
        field = ActivityField(graph)
        node = 0
        channel = graph.region_names[int(graph.regions[node])]
        measurements = dict.fromkeys(graph.region_names, 0.25)
        frame = ActivationFrame(
            "token",
            1,
            ActivitySource.REAL_TIME,
            regions=measurements,
        )

        field.set_importance(node, 2.0)
        field.update(frame)
        self.assertEqual(field.channel_values[channel], 0.25)
        self.assertEqual(field.values[node], 0.5)

        field.set_disabled(node, True)
        field.update(frame)
        self.assertEqual(field.channel_values[channel], 0.25)
        self.assertEqual(field.values[node], 0.0)

    def test_spacing_changes_the_deterministic_cluster_layout(self) -> None:
        compact = build_connectome("spacing-check", "Low", 0.6)
        spread = build_connectome("spacing-check", "Low", 1.8)

        self.assertTrue(np.array_equal(compact.regions, spread.regions))
        self.assertFalse(np.allclose(compact.positions, spread.positions))
        self.assertGreater(
            float(np.linalg.norm(spread.positions.mean(axis=0))),
            float(np.linalg.norm(compact.positions.mean(axis=0))),
        )
