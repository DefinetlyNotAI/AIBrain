from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.app.theme import DEFAULT_COLOURS, load_colours, save_colours, stylesheet


class ThemeTests(unittest.TestCase):
    def test_custom_colours_round_trip_to_the_requested_cache_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "aibrain.color.json"
            with patch("src.app.theme.COLOUR_FILE", target):
                colours = DEFAULT_COLOURS | {"background": "#112233", "renderer_background": "#445566"}
                save_colours(colours)

                self.assertEqual(load_colours()["background"], "#112233")
                self.assertEqual(load_colours()["renderer_background"], "#445566")

    def test_stylesheet_uses_the_semantic_custom_colours(self) -> None:
        colours = DEFAULT_COLOURS | {"accent": "#123456"}

        self.assertIn("#123456", stylesheet(colours))


if __name__ == "__main__":
    unittest.main()
