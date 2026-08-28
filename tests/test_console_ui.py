from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from src.utils.console_ui import Color, panel


class ConsoleUiTests(unittest.TestCase):
    def test_panel_uses_canonical_box_drawing_characters(self) -> None:
        output = StringIO()

        with redirect_stdout(output):
            panel(
                "Verification",
                [("Status", "Ready")],
                footer="Continue",
                tone=Color.GREEN,
            )

        rendered = output.getvalue()
        self.assertIn(chr(0x256d), rendered)
        self.assertIn(chr(0x2500), rendered)
        self.assertIn(chr(0x2570), rendered)
        self.assertNotIn("\ufffd", rendered)

    def test_cli_sources_contain_no_common_mojibake_markers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source_files = [*sorted((root / "cli").glob("*.py")), root / "src" / "utils" / "console_ui.py"]
        markers = (chr(0x00e2), chr(0x00c3), chr(0x00c2), chr(0xfffd))

        for source_file in source_files:
            with self.subTest(source_file=source_file):
                content = source_file.read_text(encoding="utf-8")
                self.assertFalse(any(marker in content for marker in markers))

    def test_every_cli_entry_point_uses_the_native_clear_screen_api(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for source_file in sorted((root / "cli").glob("*.py")):
            with self.subTest(source_file=source_file):
                content = source_file.read_text(encoding="utf-8")
                self.assertIn("clear_screen", content)
                self.assertIn("clear_screen()", content)
