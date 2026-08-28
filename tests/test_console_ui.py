from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.utils import console_ui
from src.utils.console_ui import Color, panel


class ConsoleUiTests(unittest.TestCase):
    def test_panel_uses_clean_box_drawing_without_source_mojibake(self) -> None:
        output = StringIO()

        with redirect_stdout(output):
            panel(
                "Verification",
                [("Status", "Ready")],
                footer="Continue",
                tone=Color.GREEN,
            )

        rendered = output.getvalue()
        self.assertIn(chr(0x256D), rendered)
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
        for source_file in sorted(
            path for path in (root / "cli").glob("*.py") if path.name != "__init__.py"
        ):
            with self.subTest(source_file=source_file):
                content = source_file.read_text(encoding="utf-8")
                self.assertIn("clear_screen", content)
                self.assertIn("clear_screen()", content)

    def test_only_the_shared_console_ui_configures_windows_output_encoding(self) -> None:
        root = Path(__file__).resolve().parents[1]
        forbidden = ("SetConsoleOutputCP", "SetConsoleCP", ".reconfigure(")

        for source_file in sorted((root / "cli").glob("*.py")):
            with self.subTest(source_file=source_file):
                content = source_file.read_text(encoding="utf-8")
                self.assertFalse(any(marker in content for marker in forbidden))

        shared_ui = (root / "src" / "utils" / "console_ui.py").read_text(encoding="utf-8")
        self.assertIn("SetConsoleOutputCP(65001)", shared_ui)
        self.assertIn("stream.reconfigure(encoding=\"utf-8\"", shared_ui)

    def test_native_clear_passes_a_wchar_value_not_a_string_pointer(self) -> None:
        class Call:
            def __init__(self, result=1) -> None:  # type: ignore[no-untyped-def]
                self.result = result
                self.calls: list[tuple[object, ...]] = []

            def __call__(self, *args):  # type: ignore[no-untyped-def]
                self.calls.append(args)
                if self is screen_info:
                    info = args[1]._obj
                    info.size.x = 80
                    info.size.y = 25
                    info.window.left = 0
                    info.window.right = 79
                return self.result

        get_handle = Call(1)
        screen_info = Call(1)
        fill_character = Call(1)
        kernel32 = SimpleNamespace(
            GetStdHandle=get_handle,
            GetConsoleScreenBufferInfo=screen_info,
            FillConsoleOutputCharacterW=fill_character,
            FillConsoleOutputAttribute=Call(1),
            SetConsoleCursorPosition=Call(1),
        )
        with patch.object(console_ui.os, "name", "nt"), patch.object(console_ui.sys.stdout, "isatty", return_value=True), patch("ctypes.windll", SimpleNamespace(kernel32=kernel32)):
            console_ui.clear_screen()
            self.assertEqual(console_ui.terminal_width(), 80)

        self.assertEqual(fill_character.calls[0][1], " ")
