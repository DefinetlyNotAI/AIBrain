from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from cli import build_native


class _OutputStream:
    def __init__(self, lines: list[str], events: list[str]) -> None:
        self._lines = lines
        self._events = events

    def __iter__(self):  # type: ignore[no-untyped-def]
        for line in self._lines:
            self._events.append(f"read:{line.rstrip()}")
            yield line


class _Process:
    def __init__(self, lines: list[str], return_code: int, events: list[str]) -> None:
        self.stdout = _OutputStream(lines, events)
        self.returncode = return_code
        self._events = events

    def wait(self) -> int:
        self._events.append("wait")
        return self.returncode

    def poll(self) -> int:
        return self.returncode


class _OutputBox:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def write(self, output: str) -> None:
        self._events.append(f"write:{output.rstrip()}")

    @property
    def is_live(self) -> bool:
        return True

    def write_progress(self, output: str) -> None:
        self._events.append(f"progress:{output}")


class NativeBuildCommandTests(unittest.TestCase):
    def test_command_output_is_rendered_before_process_completion(self) -> None:
        events: list[str] = []
        process = _Process(["compiling source\n", "linking library\n"], 0, events)

        with (
            patch.object(build_native.subprocess, "Popen", return_value=process),
            patch.object(build_native, "CommandOutputBox", return_value=_OutputBox(events)),
            patch.object(build_native, "command_preview"),
            patch.object(build_native, "log_completed_command"),
        ):
            result = build_native.run_command(["compiler.exe", "source.c"])

        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            [event for event in events if event.startswith("write:")],
            ["write:compiling source", "write:linking library"],
        )
        self.assertLess(events.index("write:compiling source"), events.index("wait"))
        self.assertLess(events.index("write:linking library"), events.index("wait"))

    def test_silent_command_does_not_invent_exit_code_output(self) -> None:
        events: list[str] = []
        process = _Process([], 0, events)

        with (
            patch.object(build_native.subprocess, "Popen", return_value=process),
            patch.object(build_native, "CommandOutputBox", return_value=_OutputBox(events)),
            patch.object(build_native, "command_preview"),
            patch.object(build_native, "log_completed_command") as log_command,
        ):
            result = build_native.run_command(["git", "diff", "--quiet"])

        self.assertEqual(result.stdout, "")
        self.assertEqual(events, ["wait"])
        log_command.assert_called_once_with(
            ["git", "diff", "--quiet"], "", return_code=0
        )

    def test_checked_failure_retains_streamed_output(self) -> None:
        events: list[str] = []
        process = _Process(["compile failed\n"], 2, events)

        with (
            patch.object(build_native.subprocess, "Popen", return_value=process),
            patch.object(build_native, "CommandOutputBox", return_value=_OutputBox(events)),
            patch.object(build_native, "command_preview"),
            patch.object(build_native, "log_completed_command"),
        ):
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                build_native.run_command(["compiler.exe"], check=True)

        self.assertEqual(raised.exception.returncode, 2)
        self.assertEqual(raised.exception.output, "compile failed")

    def test_compiler_commands_enable_verbose_toolchain_output(self) -> None:
        gnu = build_native.command_for(
            build_native.Compiler(build_native.Path("gcc.exe"), "gnu"), False
        )
        msvc = build_native.command_for(
            build_native.Compiler(build_native.Path("cl.exe"), "msvc"), False
        )

        self.assertIn("-v", gnu)
        self.assertIn("/Bt+", msvc)
        self.assertEqual(msvc[-2:], ["/link", "/VERBOSE"])


if __name__ == "__main__":
    unittest.main()
