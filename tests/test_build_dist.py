from __future__ import annotations

import subprocess
import sys
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from unittest.mock import patch

from cli import build_dist
from src.utils.console_ui import BOX_TOP_LEFT


class BuildDistributionTests(unittest.TestCase):
    def setUp(self) -> None:
        """Use the subprocess runner for tests that supply Popen doubles."""
        conpty_support = patch(
            "cli.build_dist._supports_conpty",
            return_value=False,
        )
        conpty_support.start()
        self.addCleanup(conpty_support.stop)

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

    def test_tail_renders_unterminated_progress_without_waiting_for_a_newline(self) -> None:
        class OutputBox:
            def __init__(self) -> None:
                self.completed: list[str] = []
                self.partial: list[str] = []

            def write(self, text: str) -> None:
                self.completed.append(text)

            def write_partial(self, text: str) -> None:
                self.partial.append(text)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nuitka-output.txt"
            path.write_text("Downloading: 25%", encoding="utf-8")
            output_box = OutputBox()
            offset, pending = build_dist._render_new_build_output(path, 0, "", output_box)  # type: ignore[arg-type]
            self.assertEqual(pending, "Downloading: 25%")
            self.assertEqual(output_box.partial, ["Downloading: 25%"])

            path.write_text("Downloading: 25%\rDownloading: 50%\rDone\n", encoding="utf-8")
            _, pending = build_dist._render_new_build_output(path, offset, pending, output_box)  # type: ignore[arg-type]

        self.assertEqual(pending, "")
        self.assertEqual(output_box.partial, ["Downloading: 25%", "Downloading: 25%", "Downloading: 50%"])
        self.assertEqual(output_box.completed, ["Done"])

    @patch("cli.build_dist.time.monotonic", return_value=115.0)
    def test_silent_build_heartbeat_reports_live_work(self, _monotonic) -> None:  # type: ignore[no-untyped-def]
        class OutputBox:
            def __init__(self) -> None:
                self.completed: list[str] = []

            def write(self, text: str) -> None:
                self.completed.append(text)

        output_box = OutputBox()
        next_heartbeat = build_dist._report_build_heartbeat(output_box, 100.0)  # type: ignore[arg-type]

        self.assertEqual(next_heartbeat, 115.0)
        self.assertEqual(len(output_box.completed), 1)
        self.assertIn("Still working: Nuitka is running", output_box.completed[0])
        self.assertIn("15s", output_box.completed[0])

    @patch("cli.build_dist._show_build_stall_dialog", return_value="stop")
    @patch("cli.build_dist.time.monotonic", return_value=300.0)
    def test_stall_monitor_requests_a_stop_after_five_minutes_of_silence(
        self, _monotonic, dialog
    ) -> None:  # type: ignore[no-untyped-def]
        action, state = build_dist._monitor_build_stall(
            0.0,
            build_dist.BuildStallState(build_dist.BUILD_STALL_SECONDS),
        )

        self.assertEqual(action, "stop")
        self.assertEqual(state.next_check, build_dist.BUILD_STALL_SECONDS)
        dialog.assert_called_once_with(300.0, follow_up=False)

    @patch("cli.build_dist._show_build_stall_dialog", return_value="wait")
    @patch("cli.build_dist.time.monotonic", return_value=300.0)
    def test_stall_monitor_rechecks_after_the_requested_five_minutes(
        self, _monotonic, dialog
    ) -> None:  # type: ignore[no-untyped-def]
        action, state = build_dist._monitor_build_stall(
            0.0,
            build_dist.BuildStallState(build_dist.BUILD_STALL_SECONDS),
        )

        self.assertIsNone(action)
        self.assertEqual(state.next_check, 600.0)
        self.assertTrue(state.awaiting_requested_recheck)
        dialog.assert_called_once_with(300.0, follow_up=False)

    def test_run_renders_unbuffered_progress_before_the_child_completes(self) -> None:
        class OutputBox:
            partial_times: list[float] = []
            completed_times: list[float] = []

            def __enter__(self) -> OutputBox:
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def write(self, _text: str) -> None:
                self.completed_times.append(time.monotonic())

            def write_partial(self, _text: str) -> None:
                self.partial_times.append(time.monotonic())

        command = [
            sys.executable,
            "-u",
            "-c",
            "import sys, time; sys.stdout.write('Nuitka: 25%\\r'); time.sleep(0.3); print('Nuitka: done')",
        ]

        with patch("cli.build_dist.CommandOutputBox", OutputBox):
            build_dist.run(command)

        self.assertEqual(len(OutputBox.partial_times), 1)
        self.assertEqual(len(OutputBox.completed_times), 1)
        self.assertLess(OutputBox.partial_times[0], OutputBox.completed_times[0])

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
        numpy_runtime = (Path("numpy-runtime"), Path("numpy-dlls"))

        self.assertEqual(set(targets), {"ai_brain.exe", "diagnostic.exe", "analysis.exe"})
        self.assertTrue(all(target.console_mode == "attach" for target in targets.values()))
        for target in targets.values():
            command = build_dist.nuitka_command(target, Path("build"), [Path("vcomp140.dll")], numpy_runtime)
            self.assertIn(f"--windows-icon-from-ico={target.icon}", command)
            self.assertIn(f"--output-filename={target.executable}", command)
            self.assertIn("--show-progress", command)
            self.assertIn("--show-scons", command)
            excluded = {
                item.removeprefix("--nofollow-import-to=")
                for item in command
                if item.startswith("--nofollow-import-to=")
            }
            self.assertEqual(excluded, set(build_dist.EXCLUDED_PACKAGING_IMPORTS))
            self.assertEqual(excluded, {"glcontext", "numpy"})
            self.assertIn("--include-raw-dir=numpy-runtime=numpy", command)
            self.assertIn("--include-raw-dir=numpy-dlls=numpy.libs", command)
            self.assertIn("--include-module=textwrap", command)
            self.assertNotIn("--include-module=numpy", command)
            self.assertNotIn("--include-package=src", command)
            self.assertEqual(command[1:3], ["-u", "-m"])

    def test_target_sections_follow_the_build_session_section(self) -> None:
        source = Path(build_dist.__file__).read_text(encoding="utf-8")

        self.assertIn('section("Build session", 1)', source)
        self.assertIn("targets,\n                start=2,", source)

    def test_numpy_runtime_staging_keeps_only_required_submodules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_package = root / "source" / "numpy"
            source_dlls = root / "source" / "numpy.libs"
            source_package.mkdir(parents=True)
            source_dlls.mkdir(parents=True)
            (source_package / "__init__.py").write_text("", encoding="utf-8")
            (source_package / "conftest.py").write_text("", encoding="utf-8")
            (source_dlls / "numpy.dll").write_bytes(b"dll")
            for name in build_dist.NUMPY_RUNTIME_SUBDIRECTORIES:
                module = source_package / name
                module.mkdir()
                (module / "__init__.py").write_text("", encoding="utf-8")
            (source_package / "fft").mkdir()
            (source_package / "fft" / "unused.py").write_text("", encoding="utf-8")

            with patch("cli.build_dist.NUMPY_PACKAGE_ROOT", source_package), \
                    patch("cli.build_dist.NUMPY_DLL_ROOT", source_dlls):
                package, dlls = build_dist.stage_numpy_runtime(root / "stage")

            self.assertTrue((package / "__init__.py").is_file())
            self.assertFalse((package / "conftest.py").exists())
            self.assertTrue((package / "random" / "__init__.py").is_file())
            self.assertFalse((package / "fft").exists())
            self.assertEqual((dlls / "numpy.dll").read_bytes(), b"dll")


if __name__ == "__main__":
    unittest.main()
