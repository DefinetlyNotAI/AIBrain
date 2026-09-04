"""Stable visual-region colour contract."""
from __future__ import annotations

import unittest

from src.connectome.generator import (
    CLUSTER_COLOR_MAP,
    LIGHT_BACKGROUND_CLUSTER_COLOR_MAP,
    REGIONS,
    cluster_colour_map_for_background,
)


class ClusterPaletteTests(unittest.TestCase):
    def test_every_named_region_has_a_unique_stable_colour(self) -> None:
        self.assertEqual(tuple(CLUSTER_COLOR_MAP), REGIONS)
        self.assertEqual(len(set(CLUSTER_COLOR_MAP.values())), len(REGIONS))
        self.assertEqual(CLUSTER_COLOR_MAP["Context utilisation"], "#75E6FF")
        self.assertEqual(CLUSTER_COLOR_MAP["Recent output rarity"], "#C5F28A")

    def test_light_renderer_background_uses_a_contrasting_palette(self) -> None:
        self.assertEqual(cluster_colour_map_for_background("#F8F8F8"), LIGHT_BACKGROUND_CLUSTER_COLOR_MAP)
        self.assertEqual(cluster_colour_map_for_background("#071018"), CLUSTER_COLOR_MAP)
