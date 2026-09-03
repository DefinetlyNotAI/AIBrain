from __future__ import annotations

import os
import re
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.utils import console_ui
from src.utils.console_ui import Color, panel


class TerminalOutput(StringIO):
    """Apply the box's VT cursor commands instead of counting raw redraws."""

    def __init__(self) -> None:
        super().__init__()
        self.rows = [""]
        self.row = 0
        self.column = 0
        self.frames: list[list[str]] = []

    def isatty(self) -> bool:
        return True

    def write(self, text: str) -> int:
        for token in re.findall(r"\x1b\[[0-9;]*[A-Za-z]|[^\x1b]", text):
            if token == "\x1b[1A":
                self.row = max(0, self.row - 1)
            elif token == "\x1b[2K":
                self.rows[self.row] = ""
            elif token.startswith("\x1b[") and token.endswith("m"):
                continue
            elif token == "\r":
                self.column = 0
            elif token == "\n":
                self.row += 1
                self.column = 0
                if self.row == len(self.rows):
                    self.rows.append("")
            else:
                line = self.rows[self.row].ljust(self.column)
                self.rows[self.row] = line[:self.column] + token + line[self.column + 1:]
                self.column += 1
        return super().write(text)

    def flush(self) -> None:
        self.frames.append([line.rstrip() for line in self.rows if line.strip()])
        super().flush()


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

    def test_error_preserves_each_traceback_line_with_an_aligned_gutter(self) -> None:
        output = StringIO()
        with redirect_stderr(output):
            console_ui.error(
                "Model validation failed\n"
                "Traceback (most recent call last):\n"
                "  File \"model.py\", line 7, in validate\n"
                "RuntimeError: incompatible backend"
            )

        rendered = console_ui.strip_ansi(output.getvalue()).splitlines()
        self.assertEqual(rendered[0], f"  {console_ui.CROSS} Model validation failed")
        self.assertEqual(rendered[1], " " * 4 + "Traceback (most recent call last):")
        self.assertEqual(rendered[2], " " * 6 + 'File "model.py", line 7, in validate')
        self.assertEqual(rendered[3], " " * 4 + "RuntimeError: incompatible backend")

    def test_cli_sources_contain_no_common_mojibake_markers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source_files = [
            *sorted((root / "cli").glob("*.py")),
            root / "src" / "utils" / "console_ui.py",
        ]
        markers = (
            chr(0x00E2),
            chr(0x00C3),
            chr(0x00C2),
            chr(0xFFFD),
        )

        for source_file in source_files:
            with self.subTest(source_file=source_file):
                content = source_file.read_text(encoding="utf-8")
                self.assertFalse(any(marker in content for marker in markers))

    def test_every_cli_entry_point_uses_the_native_clear_screen_api(self) -> None:
        root = Path(__file__).resolve().parents[1]

        for source_file in sorted(
            path
            for path in (root / "cli").glob("*.py")
            if path.name != "__init__.py"
        ):
            with self.subTest(source_file=source_file):
                content = source_file.read_text(encoding="utf-8")
                self.assertIn("clear_screen", content)
                self.assertIn("clear_screen()", content)

    def test_every_cli_entry_point_handles_keyboard_interrupt(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for source_file in sorted(
                path for path in (root / "cli").glob("*.py") if path.name != "__init__.py"
        ):
            with self.subTest(source_file=source_file):
                content = source_file.read_text(encoding="utf-8-sig")
                self.assertIn("except KeyboardInterrupt", content)

    def test_cli_sources_do_not_mutate_host_console_encodings(self) -> None:
        root = Path(__file__).resolve().parents[1]
        forbidden = (
            "SetConsoleOutputCP",
            "SetConsoleCP",
            ".reconfigure(",
        )

        source_files = [
            *sorted((root / "cli").glob("*.py")),
            root / "src" / "utils" / "console_ui.py",
        ]

        for source_file in source_files:
            with self.subTest(source_file=source_file):
                content = source_file.read_text(encoding="utf-8")
                self.assertFalse(any(marker in content for marker in forbidden))

    def test_native_clear_passes_a_wchar_value_not_a_string_pointer(self) -> None:
        class Call:
            def __init__(
                self,
                result: int = 1,
                writes_count: bool = False,
            ) -> None:
                self.result = result
                self.writes_count = writes_count
                self.calls: list[tuple[object, ...]] = []

            def __call__(self, *args: object) -> int:
                self.calls.append(args)

                if self is screen_info:
                    info = args[1]._obj  # type: ignore[attr-defined]
                    info.size.x = 80
                    info.size.y = 25
                    info.window.left = 0
                    info.window.right = 79

                if self.writes_count:
                    args[4]._obj.value = args[2]  # type: ignore[attr-defined]

                return self.result

        get_handle = Call(1)
        screen_info = Call(1)
        fill_character = Call(1, writes_count=True)

        kernel32 = SimpleNamespace(
            get_std_handle=get_handle,
            get_console_screen_buffer_info=screen_info,
            fill_console_output_character=fill_character,
            fill_console_output_attribute=Call(1, writes_count=True),
            set_console_cursor_position=Call(1),
        )

        with patch.object(
            console_ui,
            "_kernel32_bindings",
            return_value=kernel32,
        ):
            result = console_ui._clear_native_console()

        self.assertTrue(result)
        self.assertEqual(fill_character.calls[0][1], " ")
        self.assertEqual(fill_character.calls[0][2], 2_000)
        self.assertEqual(fill_character.calls[0][3].x, 0)  # type: ignore[attr-defined]
        self.assertEqual(fill_character.calls[0][3].y, 0)  # type: ignore[attr-defined]

    def test_clear_screen_uses_vt_sequence_when_available(self) -> None:
        stdout = Mock()
        stdout.isatty.return_value = True

        with (
            patch.object(console_ui.os, "name", "nt"),
            patch.object(console_ui.sys, "stdout", stdout),
            patch.object(
                console_ui,
                "_enable_virtual_terminal",
                return_value=True,
            ),
            patch.object(console_ui, "_clear_native_console") as native_clear,
        ):
            console_ui.clear_screen()

        stdout.write.assert_called_once_with("\x1b[2J\x1b[3J\x1b[H")
        stdout.flush.assert_called_once_with()
        native_clear.assert_not_called()

    def test_clear_screen_falls_back_to_native_when_vt_is_unavailable(self) -> None:
        stdout = Mock()
        stdout.isatty.return_value = True

        with (
            patch.object(console_ui.os, "name", "nt"),
            patch.object(console_ui.sys, "stdout", stdout),
            patch.object(
                console_ui,
                "_enable_virtual_terminal",
                return_value=False,
            ),
            patch.object(
                console_ui,
                "_clear_native_console",
                return_value=True,
            ) as native_clear,
        ):
            console_ui.clear_screen()

        stdout.write.assert_not_called()
        native_clear.assert_called_once_with()

    def test_native_clear_uses_attached_console_when_stdout_is_redirected(
        self,
    ) -> None:
        class ScreenInfo:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(
                self,
                _handle: object,
                info_pointer: object,
            ) -> int:
                self.calls += 1

                if self.calls == 1:
                    return 0

                info = info_pointer._obj  # type: ignore[attr-defined]
                info.size.x = 120
                info.size.y = 900
                info.attributes = 7
                return 1

        def fill(
            _handle: object,
            _value: object,
            count: int,
            _origin: object,
            written: object,
        ) -> int:
            written._obj.value = count  # type: ignore[attr-defined]
            return 1

        stdout = Mock()
        stdout.isatty.return_value = False

        create_file = Mock(return_value=99)
        close_handle = Mock(return_value=1)

        kernel32 = SimpleNamespace(
            get_std_handle=Mock(return_value=1),
            get_console_screen_buffer_info=ScreenInfo(),
            create_file=create_file,
            close_handle=close_handle,
            fill_console_output_character=Mock(side_effect=fill),
            fill_console_output_attribute=Mock(side_effect=fill),
            set_console_cursor_position=Mock(return_value=1),
        )

        with (
            patch.object(console_ui.os, "name", "nt"),
            patch.object(console_ui.sys, "stdout", stdout),
            patch.object(
                console_ui,
                "_kernel32_bindings",
                return_value=kernel32,
            ),
        ):
            console_ui.clear_screen()

        self.assertEqual(create_file.call_args.args[0], "CONOUT$")
        self.assertEqual(
            kernel32.fill_console_output_character.call_args.args[2],
            108_000,
        )
        close_handle.assert_called_once_with(99)

    def test_console_ui_avoids_dynamic_ctypes_dll_attributes(self) -> None:
        source = Path(console_ui.__file__).read_text(encoding="utf-8")

        self.assertIn("ctypes.WINFUNCTYPE", source)
        self.assertNotIn("ctypes.windll", source)

    def test_terminal_width_reserves_four_columns_at_the_right_edge(self) -> None:
        with (
            patch.object(console_ui.os, "name", "posix"),
            patch.object(
                console_ui.shutil,
                "get_terminal_size",
                return_value=os.terminal_size((100, 24)),
            ),
        ):
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

        with (
            patch.object(console_ui, "terminal_width", return_value=58),
            redirect_stdout(output),
        ):
            console_ui.command_preview(command)

        lines = [
            console_ui.strip_ansi(line)
            for line in output.getvalue().splitlines()
            if line
        ]

        self.assertGreater(len(lines), 1)
        self.assertTrue(
            lines[0].startswith(
                "  " + console_ui.PROMPT + " .\\.venv"
            )
        )
        self.assertTrue(all(line.startswith("    ") for line in lines[1:]))
        self.assertTrue(all(len(line) <= 58 for line in lines))
        self.assertFalse(any(line.endswith("...") for line in lines))

    def test_command_preview_quotes_versioned_packages_for_powershell(self) -> None:
        rendered = console_ui.display_command(
            [
                r".\.venv\Scripts\python.exe",
                "-m",
                "pip",
                "install",
                "PySide6>=6.7,<7",
                "cupy-cuda13x[ctk]>=14,<15",
            ]
        )

        self.assertIn("'PySide6>=6.7,<7'", rendered)
        self.assertIn("'cupy-cuda13x[ctk]>=14,<15'", rendered)

    def test_live_command_footer_moves_with_output_and_stays_visible_between_frames(self) -> None:
        output = TerminalOutput()
        with (
            patch.object(console_ui, "terminal_width", return_value=42),
            redirect_stdout(output),
        ):
            box = console_ui.CommandOutputBox(live=True)
            box.open()
            self.assertEqual(output.getvalue(), "")
            self.assertEqual(output.frames, [])
            box.write("First completed line\nSecond completed line")
            stable = output.frames[-1][:-1]
            box.write_partial("Downloading a very long model filename at 25 percent")
            self.assertGreater(len(output.frames[-1]), len(stable) + 2)
            box.write_partial("Downloading model: 50%")
            self.assertEqual(len(output.frames[-1]), len(stable) + 2)
            self.assertNotIn("filename", "\n".join(output.frames[-1]))
            box.write("Download complete")
            box.write("Next completed line")
            self.assertEqual(output.frames[-1][:len(stable)], stable)
            self.assertIn("Download complete", output.frames[-1][-3])
            self.assertIn("Next completed line", output.frames[-1][-2])
            self.assertNotIn("50%", "\n".join(output.frames[-1]))

            footer = (
                box.prefix + console_ui.BOX_BOTTOM_LEFT
                + console_ui.BOX_HORIZONTAL * box.inner
                + console_ui.BOX_BOTTOM_RIGHT
            )
            for frame in output.frames:
                self.assertEqual(frame[-1], footer)
                # ASCII terminals use the same characters for both borders.
                self.assertFalse(any(line == footer for line in frame[1:-1]))
            before_close = output.getvalue()
            box.close()
            box.close()
            self.assertEqual(output.getvalue(), before_close)

    def test_empty_command_boxes_leave_no_frame_in_live_or_captured_output(self) -> None:
        for live in (True, False):
            with self.subTest(live=live), redirect_stdout(StringIO()) as output:
                with console_ui.CommandOutputBox(live=live) as box:
                    box.write("")
                    box.write(" \n\t")
                    box.write_partial(" \n")
                    self.assertEqual(output.getvalue(), "")
                self.assertEqual(output.getvalue(), "")

    def test_captured_progress_is_periodic_and_stage_changes_are_immediate(self) -> None:
        output = StringIO()
        with redirect_stdout(output), console_ui.CommandOutputBox(live=False) as box:
            for now, message in ((10.0, "Analyzing alpha"), (11.0, "Analyzing beta"), (13.0, "Analyzing gamma")):
                with patch.object(console_ui.time, "monotonic", return_value=now):
                    box.write_progress(message)
            self.assertIn("Analyzing alpha", output.getvalue())
            self.assertNotIn("Analyzing beta", output.getvalue())
            self.assertIn("Analyzing gamma", output.getvalue())
            box.write("Starting the next pass")
            with patch.object(console_ui.time, "monotonic", return_value=13.1):
                box.write_progress("Analyzing delta")
            self.assertIn("Analyzing delta", output.getvalue())

    def test_first_partial_output_draws_one_complete_frame(self) -> None:
        output = TerminalOutput()
        with redirect_stdout(output):
            with console_ui.CommandOutputBox(live=True) as box:
                box.write_partial("Waiting for compiler output")
                self.assertEqual(len(output.frames), 1)
                self.assertEqual(len(output.frames[0]), 3)
                self.assertIn("Waiting for compiler output", output.frames[0][1])

    def test_status_paths_are_relative_and_long_messages_wrap_at_the_current_width(self) -> None:
        project_log = console_ui.ROOT / "logs" / "aibrain.build_dist.log"
        with redirect_stdout(StringIO()) as output:
            console_ui.info(f"Log file: {project_log}")
        self.assertEqual(
            console_ui.strip_ansi(output.getvalue()).strip(),
            f"{console_ui.BULLET} Log file: .\\logs\\aibrain.build_dist.log",
        )

        for width in (38, 64):
            with self.subTest(width=width), patch.object(console_ui, "terminal_width", return_value=width):
                for emit in (console_ui.info, console_ui.success, console_ui.warning, console_ui.error):
                    output = StringIO()
                    with redirect_stdout(output), redirect_stderr(output):
                        emit("Staging a long runtime message with enough words to wrap onto multiple rows\n  indented detail")
                    lines = console_ui.strip_ansi(output.getvalue()).splitlines()
                    self.assertTrue(all(len(line) <= width for line in lines))
                    prefix_width = lines[0].index("Staging")
                    self.assertTrue(all(line.startswith(" " * prefix_width) for line in lines[1:]))
                    self.assertEqual(lines[-1], " " * (prefix_width + 2) + "indented detail")

    def test_long_command_preview_counts_flags_and_keeps_the_full_command_in_the_log(self) -> None:
        command = [str(console_ui.ROOT / ".venv" / "Scripts" / "python.exe"), "-u", "-m", "nuitka"]
        options = [f"--include-module=module_{index}" for index in range(49)]
        command += [*options, "--output-dir", "a folder", "--", "--positional-script.py"]
        with (
            patch.object(console_ui, "terminal_width", return_value=64),
            redirect_stdout(StringIO()) as output,
            self.assertLogs("aibrain.command", level="INFO") as captured,
        ):
            console_ui.command_preview(command)
        lines = [line for line in console_ui.strip_ansi(output.getvalue()).splitlines() if line]
        rendered = " ".join(line.strip() for line in lines)
        self.assertIn(r".\.venv\Scripts\python.exe -u -m nuitka", rendered)
        self.assertIn("(50 flags attached - Full command in log file)", rendered)
        self.assertNotIn("--include-module", rendered)
        self.assertNotIn("--positional-script.py", rendered)
        self.assertTrue(all(len(line) <= 64 for line in lines))
        self.assertTrue(all(line.startswith("    ") for line in lines[1:]))
        self.assertIn(options[-1], captured.output[0])
        self.assertIn("--positional-script.py", captured.output[0])
        self.assertIn(str(console_ui.ROOT), captured.output[0])

    def test_path_shortening_preserves_neighboring_directories_and_handles_flag_values(self) -> None:
        neighbor = str(console_ui.ROOT) + "-backup\\file.py"
        self.assertEqual(console_ui.shorten_command_argument(neighbor), neighbor)
        self.assertEqual(console_ui.shorten_output_paths(neighbor), neighbor)
        rendered = console_ui.display_command(["tool", f"--output-dir={console_ui.ROOT / 'dist'}"])
        self.assertIn(r"--output-dir=.\dist", rendered)

    def test_redirected_command_output_has_no_cursor_redraws_or_duplicate_progress(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            with console_ui.CommandOutputBox() as box:
                box.write_partial("Downloading: 25%")
                box.write_partial("Downloading: 50%")
                box.write("Download complete")
        rendered = console_ui.strip_ansi(output.getvalue())
        self.assertEqual(len(rendered.splitlines()), 3)
        self.assertIn("Download complete", rendered)
        self.assertNotIn("Downloading:", rendered)
        self.assertNotIn("\x1b[1A", output.getvalue())

    def test_executable_path_shortening_requires_an_exact_path_match(
        self,
    ) -> None:
        local_executable = (
            Path(console_ui.ROOT)
            / ".venv"
            / "Scripts"
            / "python.exe"
        )

        with patch.object(
            console_ui.shutil,
            "which",
            return_value=str(local_executable),
        ):
            self.assertEqual(
                console_ui.shorten_command_argument(str(local_executable)),
                r".\.venv\Scripts\python.exe",
            )

        executable = Path(sys._base_executable).resolve()

        with patch.object(
            console_ui.shutil,
            "which",
            return_value=str(executable),
        ):
            self.assertEqual(
                console_ui.shorten_command_argument(str(executable)),
                executable.name,
            )

        with patch.object(
            console_ui.shutil,
            "which",
            return_value=None,
        ):
            self.assertEqual(
                console_ui.shorten_command_argument(str(executable)),
                str(executable),
            )

    def test_relative_executable_path_remains_explicit_when_it_is_on_path(self) -> None:
        local_executable = (
            Path(console_ui.ROOT)
            / ".venv"
            / "Scripts"
            / "python.exe"
        )

        with patch.object(
            console_ui.shutil,
            "which",
            return_value=str(local_executable),
        ):
            self.assertEqual(
                console_ui.shorten_command_argument(r".\.venv\Scripts\python.exe"),
                r".\.venv\Scripts\python.exe",
            )

    def test_choice_accepts_a_lowercase_full_label_and_renders_it_white(self) -> None:
        output = StringIO()
        stdin = StringIO()
        stdin.isatty = lambda: True  # type: ignore[method-assign]

        with (
            patch.object(console_ui.sys, "stdin", stdin),
            patch("builtins.input", return_value="repair"),
            redirect_stdout(output),
        ):
            selected = console_ui.ask_choice(
                "Choose action", {"i": "Install", "r": "Repair"}, default="i"
            )

        self.assertEqual(selected, "r")
        self.assertIn(Color.WHITE, output.getvalue())
        self.assertIn("Repair", console_ui.strip_ansi(output.getvalue()))

    def test_empty_boolean_uses_a_colored_default_on_the_prompt_line(self) -> None:
        output = StringIO()
        stdin = StringIO()
        stdin.isatty = lambda: True  # type: ignore[method-assign]

        with (
            patch.object(console_ui.sys, "stdin", stdin),
            patch("builtins.input", return_value=""),
            redirect_stdout(output),
        ):
            selected = console_ui.ask_boolean("Enable acceleration", default=True)

        self.assertTrue(selected)
        self.assertIn(Color.GREEN, output.getvalue())
        self.assertIn("Yes", console_ui.strip_ansi(output.getvalue()))

    def test_exit_reports_distinguish_a_gui_close_from_keyboard_interrupt(self) -> None:
        stdout = StringIO()
        stderr = StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            console_ui.report_gui_closed("AIBrain Analysis")
            console_ui.report_keyboard_interrupt("AIBrain Analysis")

        self.assertIn(console_ui.CHECK, console_ui.strip_ansi(stdout.getvalue()))
        self.assertIn("User closed AIBrain Analysis.", stdout.getvalue())
        self.assertTrue(stdout.getvalue().endswith("\n\n"))
        self.assertIn(console_ui.CROSS, console_ui.strip_ansi(stderr.getvalue()))
        self.assertIn("User ended AIBrain Analysis with KeyboardInterrupt.", stderr.getvalue())

    def test_prompt_answer_redraws_the_original_terminal_line(self) -> None:
        class InteractiveOutput(StringIO):
            def isatty(self) -> bool:
                return True

        output = InteractiveOutput()
        with (
            patch.object(console_ui.sys, "stdout", output),
            patch.object(console_ui.os, "name", "posix"),
        ):
            console_ui._render_prompt_answer("  Continue [Y]: ", "Yes", Color.GREEN)

        self.assertTrue(output.getvalue().startswith("\x1b[1A\r\x1b[2K"))
        self.assertIn("Continue [Y]: ", console_ui.strip_ansi(output.getvalue()))
        self.assertIn("Yes", console_ui.strip_ansi(output.getvalue()))

    def test_boolean_accepts_common_true_and_false_aliases(self) -> None:
        stdin = StringIO()
        stdin.isatty = lambda: True  # type: ignore[method-assign]

        cases = (
            ("true", True),
            ("yes", True),
            ("y", True),
            ("1", True),
            ("false", False),
            ("no", False),
            ("n", False),
            ("0", False),
        )
        for answer, expected in cases:
            with (
                self.subTest(answer=answer),
                patch.object(console_ui.sys, "stdin", stdin),
                patch("builtins.input", return_value=answer),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(
                    console_ui.ask_boolean("Continue", default=False), expected
                )
