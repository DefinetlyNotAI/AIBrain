from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.app.theme import (
    DEFAULT_COLOURS,
    ColourSettingsDialog,
    load_colours,
    save_colours,
    stylesheet,
)


class ThemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

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

    def test_colour_settings_use_two_columns_in_a_bounded_scroll_area(self) -> None:
        dialog = ColourSettingsDialog(DEFAULT_COLOURS)
        try:
            positions = {
                (dialog._palette_grid.getItemPosition(index)[0],
                 dialog._palette_grid.getItemPosition(index)[1])
                for index in range(dialog._palette_grid.count())
            }

            self.assertEqual({column for _row, column in positions}, {0, 1})
            self.assertTrue(dialog._palette_scroll.widgetResizable())
            self.assertLessEqual(dialog.width(), 720)
            self.assertLessEqual(dialog.height(), 500)
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
