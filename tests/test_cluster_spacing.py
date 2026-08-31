"""Cluster-spacing graph contract."""
from __future__ import annotations

import unittest

import numpy as np

from src.connectome.generator import build_connectome


class ClusterSpacingTests(unittest.TestCase):
    def test_generator_supports_random_backends_without_choice(self) -> None:
        graph = build_connectome("startup-regression", "Medium")

        self.assertEqual(graph.positions.shape, (11000, 3))
        self.assertEqual(graph.regions.shape, (11000,))
        self.assertGreater(len(graph.edges), 0)

    def test_spacing_changes_the_deterministic_cluster_layout(self) -> None:
        compact = build_connectome("spacing-check", "Low", .6)
        spread = build_connectome("spacing-check", "Low", 1.8)

        self.assertTrue(np.array_equal(compact.regions, spread.regions))
        self.assertFalse(np.allclose(compact.positions, spread.positions))
        self.assertGreater(float(np.linalg.norm(spread.positions.mean(axis=0))),
                           float(np.linalg.norm(compact.positions.mean(axis=0))))
