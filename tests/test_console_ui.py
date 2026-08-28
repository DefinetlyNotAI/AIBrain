from __future__ import annotations

import os
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.utils import console_ui
from src.utils.console_ui import Color, panel


class ConsoleUiTests(unittest.TestCase):
    def test_panel_uses_the_selected_safe_border_glyphs(self) -> None:
        output = StringIO()

        with redirect_stdout(output):
            panel(
                "Verification",
                [("Status", "Ready")],
                footer="Continue",
                tone=Color.GREEN,
            )

        rendered = output.getvalue()
        self.assertIn(console_ui.BOX_TOP_LEFT, rendered)
        self.assertIn(console_ui.BOX_HORIZONTAL, rendered)
        self.assertIn(console_ui.BOX_BOTTOM_LEFT, rendered)
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

    def test_cli_sources_do_not_mutate_host_console_encodings(self) -> None:
        root = Path(__file__).resolve().parents[1]
        forbidden = ("SetConsoleOutputCP", "SetConsoleCP", ".reconfigure(")

        source_files = [*sorted((root / "cli").glob("*.py")), root / "src" / "utils" / "console_ui.py"]
        for source_file in source_files:
            with self.subTest(source_file=source_file):
                content = source_file.read_text(encoding="utf-8")
                self.assertFalse(any(marker in content for marker in forbidden))

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
            get_std_handle=get_handle,
            get_console_screen_buffer_info=screen_info,
            fill_console_output_character=fill_character,
            fill_console_output_attribute=Call(1),
            set_console_cursor_position=Call(1),
        )
        with patch.object(console_ui.os, "name", "nt"), patch.object(console_ui.sys.stdout, "isatty", return_value=True), patch.object(console_ui, "_kernel32_bindings", return_value=kernel32):
            console_ui.clear_screen()
            self.assertEqual(console_ui.terminal_width(), 76)

        self.assertEqual(fill_character.calls[0][1], " ")

    def test_console_ui_avoids_dynamic_ctypes_dll_attributes(self) -> None:
        source = Path(console_ui.__file__).read_text(encoding="utf-8")

        self.assertIn("ctypes.WINFUNCTYPE", source)
        self.assertNotIn("ctypes.windll", source)

    def test_terminal_width_reserves_four_columns_at_the_right_edge(self) -> None:
        with patch.object(console_ui.os, "name", "posix"), patch.object(console_ui.shutil, "get_terminal_size", return_value=os.terminal_size((100, 24))):
            self.assertEqual(console_ui.terminal_width(), 96)

    def test_command_preview_wraps_with_aligned_continuations(self) -> None:
        output = StringIO()
        command = [
            r".\.venv\Scripts\python.exe",
            "-m",
            "nuitka",
            "--standalone",
            "--enable-plugin=pyside6",
            "--windows-console-mode=attach",
        ]

        with patch.object(console_ui, "terminal_width", return_value=58), redirect_stdout(output):
            console_ui.command_preview(command)

        lines = [console_ui.strip_ansi(line) for line in output.getvalue().splitlines() if line]
        self.assertGreater(len(lines), 1)
        self.assertTrue(lines[0].startswith("  " + console_ui.PROMPT + " .\\.venv"))
        self.assertTrue(all(line.startswith("     ") for line in lines[1:]))
        self.assertTrue(all(len(line) <= 58 for line in lines))
        self.assertFalse(any(line.endswith("...") for line in lines))

    def test_executable_path_shortening_requires_an_exact_path_match(self) -> None:
        local_executable = Path(sys.executable).resolve()
        with patch.object(console_ui.shutil, "which", return_value=str(local_executable)):
            self.assertEqual(
                console_ui.shorten_command_argument(str(local_executable)),
                r".\.venv\Scripts\python.exe",
            )

        executable = Path(sys._base_executable).resolve()
        with patch.object(console_ui.shutil, "which", return_value=str(executable)):
            self.assertEqual(console_ui.shorten_command_argument(str(executable)), executable.name)

        with patch.object(console_ui.shutil, "which", return_value=None):
            self.assertEqual(console_ui.shorten_command_argument(str(executable)), str(executable))
