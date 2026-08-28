from __future__ import annotations

import subprocess
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from cli import build_dist
from src.utils.console_ui import BOX_TOP_LEFT


class BuildDistributionTests(unittest.TestCase):
    @patch("cli.build_dist.time.sleep")
    @patch("cli.build_dist.subprocess.Popen")
    def test_build_command_streams_output_inside_shared_framed_panel(self, popen_mock,
                                                                     sleep_mock) -> None:  # type: ignore[no-untyped-def]
        class StreamingProcess:
            def __init__(self, output_file) -> None:  # type: ignore[no-untyped-def]
                self.output_file = output_file
                self.poll_count = 0

            def poll(self) -> int | None:
                self.poll_count += 1
                if self.poll_count == 1:
                    self.output_file.write("compiled\n")
                    self.output_file.flush()
                    return None
                self.output_file.write("warning\n")
                self.output_file.flush()
                return 0

            def wait(self) -> int:
                return 0

        popen_mock.side_effect = lambda *_args, **kwargs: StreamingProcess(kwargs["stdout"])
        output = StringIO()

        with redirect_stdout(output):
            build_dist.run(["tool"])

        rendered = output.getvalue()
        self.assertIn("compiled", rendered)
        self.assertIn("warning", rendered)
        self.assertIn(BOX_TOP_LEFT, rendered)
        sleep_mock.assert_called_once_with(0.05)

    @patch("cli.build_dist.time.sleep")
    @patch("cli.build_dist.subprocess.Popen")
    def test_failed_build_command_raises_after_rendering_output(self, popen_mock,
                                                                _sleep_mock) -> None:  # type: ignore[no-untyped-def]
        class FailedProcess:
            def __init__(self, output_file) -> None:  # type: ignore[no-untyped-def]
                output_file.write("broken\n")
                output_file.flush()

            def poll(self) -> int:
                return 4

            def wait(self) -> int:
                return 4

        popen_mock.side_effect = lambda *_args, **kwargs: FailedProcess(kwargs["stdout"])

        with self.assertRaises(subprocess.CalledProcessError) as raised:
            build_dist.run(["tool"])

        self.assertEqual(raised.exception.returncode, 4)

    @patch("cli.build_dist.subprocess.run")
    @patch("cli.build_dist.subprocess.Popen")
    def test_interrupted_build_terminates_the_child_tree_before_propagating(self, popen_mock, taskkill_mock) -> None:  # type: ignore[no-untyped-def]
        class InterruptedProcess:
            def __init__(self) -> None:
                self.pid = 123
                self.terminated = False
                self.wait_timeouts: list[float | None] = []

            def poll(self) -> int | None:
                if not self.terminated:
                    raise KeyboardInterrupt
                return 130

            def terminate(self) -> None:
                self.terminated = True

            def wait(self, timeout: float | None = None) -> int:
                self.wait_timeouts.append(timeout)
                return 130

        process = InterruptedProcess()
        popen_mock.return_value = process

        with redirect_stdout(StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                build_dist.run(["tool"])

        taskkill_mock.assert_called_once()
        self.assertEqual(process.wait_timeouts, [5])

    def test_each_packaged_application_has_its_named_entry_point_and_icon(self) -> None:
        targets = {target.executable: target for target in build_dist.APPLICATIONS}

        self.assertEqual(set(targets), {"ai_brain.exe", "diagnostic.exe", "analysis.exe"})
        self.assertEqual(targets["ai_brain.exe"].console_mode, "attach")
        self.assertEqual(targets["analysis.exe"].console_mode, "attach")
        for target in targets.values():
            command = build_dist.nuitka_command(target, Path("build"), [Path("vcomp140.dll")])
            self.assertIn(f"--windows-icon-from-ico={target.icon}", command)
            self.assertIn(f"--output-filename={target.executable}", command)
            self.assertNotIn("--include-package=src", command)


if __name__ == "__main__":
    unittest.main()
