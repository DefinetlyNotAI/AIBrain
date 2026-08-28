from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from src.utils.console_ui import Color, panel


class ConsoleUiTests(unittest.TestCase):
    def test_panel_is_ascii_safe_for_every_windows_console_code_page(self) -> None:
        output = StringIO()

        with redirect_stdout(output):
            panel(
                "Verification",
                [("Status", "Ready")],
                footer="Continue",
                tone=Color.GREEN,
            )

        rendered = output.getvalue()
        self.assertIn("+", rendered)
        self.assertIn("-", rendered)
        self.assertIn("|", rendered)
        self.assertTrue(rendered.isascii())

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
        for source_file in sorted(
            path for path in (root / "cli").glob("*.py") if path.name != "__init__.py"
        ):
            with self.subTest(source_file=source_file):
                content = source_file.read_text(encoding="utf-8")
                self.assertIn("clear_screen", content)
                self.assertIn("clear_screen()", content)

    def test_cli_sources_do_not_change_console_code_pages_or_stream_encodings(self) -> None:
        root = Path(__file__).resolve().parents[1]
        forbidden = ("SetConsoleOutputCP", "SetConsoleCP", ".reconfigure(")
        source_files = [*sorted((root / "cli").glob("*.py")), root / "src" / "utils" / "console_ui.py"]

        for source_file in source_files:
            with self.subTest(source_file=source_file):
                content = source_file.read_text(encoding="utf-8")
                self.assertFalse(any(marker in content for marker in forbidden))
