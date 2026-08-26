"""Stable visual-region colour contract."""
from __future__ import annotations

import unittest

from src.connectome.generator import CLUSTER_COLOR_MAP, REGIONS


class ClusterPaletteTests(unittest.TestCase):
    def test_every_named_region_has_a_unique_stable_colour(self) -> None:
        self.assertEqual(tuple(CLUSTER_COLOR_MAP), REGIONS)
        self.assertEqual(len(set(CLUSTER_COLOR_MAP.values())), len(REGIONS))
        self.assertEqual(CLUSTER_COLOR_MAP["Input / tokens"], "#4CC9F0")
        self.assertEqual(CLUSTER_COLOR_MAP["Output / logits"], "#90BE6D")
