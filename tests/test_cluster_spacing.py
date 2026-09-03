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

    def test_activity_mapping_uses_the_host_generator_and_activates_pathways(self) -> None:
        graph = build_connectome("activity-regression", "Low")
        field = ActivityField(graph)

        field.update(ActivationFrame(7, "token", 1, ActivitySource.SIMULATION))

        self.assertGreater(np.count_nonzero(field.values), 0)
        active_pathways = np.count_nonzero(
            (field.values[graph.edges[:, 0]] > 0.1)
            | (field.values[graph.edges[:, 1]] > 0.1)
        )
        self.assertGreater(active_pathways, 0)

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

    def test_spacing_changes_the_deterministic_cluster_layout(self) -> None:
        compact = build_connectome("spacing-check", "Low", .6)
        spread = build_connectome("spacing-check", "Low", 1.8)

        self.assertTrue(np.array_equal(compact.regions, spread.regions))
        self.assertFalse(np.allclose(compact.positions, spread.positions))
        self.assertGreater(float(np.linalg.norm(spread.positions.mean(axis=0))),
                           float(np.linalg.norm(compact.positions.mean(axis=0))))
