from __future__ import annotations

import subprocess
import os
import sys
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from unittest.mock import patch

from cli import build_dist
from src.utils.console_ui import BOX_HORIZONTAL, BOX_TOP_LEFT


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
        self.assertEqual(output_box.partial, ["Downloading: 25%", "Downloading: 50%"])
        self.assertEqual(output_box.completed, ["Done"])

    def test_renderer_coalesces_live_frames_to_the_newest_value(self) -> None:
        class OutputBox:
            def __init__(self) -> None:
                self.completed: list[str] = []
                self.partial: list[str] = []

            def write(self, text: str) -> None:
                self.completed.append(text)

            def write_partial(self, text: str) -> None:
                self.partial.append(text)

        output_box = OutputBox()
        pending = build_dist._render_output_text(
            "PASS 1: 10%\rPASS 1: 20%\rPASS 1: 30%\r",
            "",
            output_box,  # type: ignore[arg-type]
        )

        self.assertEqual(pending, "")
        self.assertEqual(output_box.completed, [])
        self.assertEqual(output_box.partial, ["PASS 1: 30%"])

    def test_renderer_logs_nuitka_option_echo_but_keeps_only_warnings_in_console(self) -> None:
        class OutputBox:
            def __init__(self) -> None:
                self.completed: list[str] = []
                self.partial: list[str] = []

            def write(self, text: str) -> None:
                self.completed.append(text)

            def write_partial(self, text: str) -> None:
                self.partial.append(text)

        output_box = OutputBox()
        with patch("cli.build_dist._record_build_output") as record:
            pending = build_dist._render_output_text(
                "Nuitka-Options: --standalone\nNuitka-Options:WARNING: check flags\nNuitka: compiling\n",
                "",
                output_box,  # type: ignore[arg-type]
            )

        self.assertEqual(pending, "")
        self.assertEqual(output_box.completed, ["Nuitka-Options:WARNING: check flags", "Nuitka: compiling"])
        self.assertEqual(output_box.partial, [])
        self.assertEqual(
            [call.args[0] for call in record.call_args_list],
            ["Nuitka-Options: --standalone", "Nuitka-Options:WARNING: check flags", "Nuitka: compiling"],
        )

    def test_native_progress_bars_remain_intact_while_diagnostics_are_preserved(self) -> None:
        from tests.test_console_ui import TerminalOutput

        output = TerminalOutput()
        lines = [
            "Nuitka-Progress: PASS 1:",
            "Nuitka-Progress: Not finished with the module due to following change kinds: var_usage",
            "Nuitka-Progress: Finished with the module.",
            "Nuitka-Inclusion: Demoting module 'alpha' to bytecode from 'alpha.py'.",
            "Nuitka-Memory: Total memory usage: 100 MB",
        ]
        with redirect_stdout(output), patch.object(build_dist, "_record_build_output") as record:
            with build_dist.CommandOutputBox(live=True) as box:
                build_dist._render_output_text("\n".join(lines) + "\n", "", box)
                first = "PASS 1 ------ 25.0% | 1/4 modules | alpha"
                latest = "PASS 1 ------------------ 75.0% | 3/4 modules | beta"
                build_dist._render_output_text(first + "\r", "", box)
                self.assertIn(first, "\n".join(output.frames[-1]))
                build_dist._render_output_text(latest + "\r", "", box)
                screen = "\n".join(output.frames[-1])
                self.assertIn(latest, screen)
                self.assertNotIn(first, screen)
                self.assertIn("PASS 1", screen)
                self.assertNotIn("var_usage", output.getvalue())
                self.assertNotIn("bytecode", output.getvalue())
                self.assertNotIn("100 MB", output.getvalue())
                for message in (
                    "Nuitka-Memory:WARNING: high memory usage",
                    "Nuitka-Inclusion:ERROR: dependency missing",
                    "Nuitka: Generating source code for C backend compiler.",
                    "Nuitka-Scons: Backend C linking with 20 files.",
                ):
                    build_dist._write_completed_build_line(box, message)
                    self.assertIn(message, output.getvalue())
        self.assertEqual([call.args[0] for call in record.call_args_list[:len(lines)]], lines)

    def test_native_bar_environment_is_scoped_to_nuitka_and_matches_box_width(self) -> None:
        with (
            patch.dict(os.environ, {"TTY_COMPATIBLE": "0", "TTY_INTERACTIVE": "0", "COLUMNS": "140"}),
            patch.object(build_dist, "terminal_width", return_value=80),
        ):
            environment = build_dist._build_process_environment([sys.executable, "-u", "-m", "nuitka"])
            self.assertEqual(environment["TTY_COMPATIBLE"], "1")
            self.assertEqual(environment["TTY_INTERACTIVE"], "1")
            self.assertEqual(environment["COLUMNS"], "74")
            self.assertEqual(environment["PYTHONIOENCODING"], "utf-8")
            self.assertEqual(os.environ["TTY_COMPATIBLE"], "0")
            self.assertEqual(os.environ["COLUMNS"], "140")
            pip_environment = build_dist._build_process_environment(
                [sys.executable, "-m", "pip", "install", "nuitka"]
            )
            self.assertEqual(pip_environment["PIP_PROGRESS_BAR"], "raw")
            self.assertEqual(pip_environment["PYTHONUNBUFFERED"], "1")
            self.assertNotIn("PIP_PROGRESS_BAR", os.environ)

    def test_pip_raw_updates_render_as_a_moving_byte_progress_bar(self) -> None:
        class OutputBox:
            def __init__(self) -> None:
                self.completed: list[str] = []
                self.progress: list[str] = []

            def write(self, text: str) -> None:
                self.completed.append(text)

            def write_progress(self, text: str) -> None:
                self.progress.append(text)

            def write_partial(self, _text: str) -> None:
                return None

        box = OutputBox()
        with patch.object(build_dist, "_record_build_output") as record:
            pending = build_dist._render_output_text(
                "Collecting demo\nProgress 0 of 1048576\n"
                "Progress 524288 of 1048576\nProgress 1048576 of 1048576\n"
                "Downloaded demo\n",
                "",
                box,  # type: ignore[arg-type]
            )
        self.assertEqual(pending, "")
        self.assertEqual(box.completed, ["Collecting demo", "Downloaded demo"])
        self.assertEqual(len(box.progress), 3)
        self.assertIn("0%", box.progress[0])
        self.assertIn("50%", box.progress[1])
        self.assertIn("100%", box.progress[2])
        self.assertIn("512.0 KB / 1.0 MB", box.progress[1])
        self.assertEqual(
            [call.args[0] for call in record.call_args_list],
            [
                "Collecting demo", "Progress 0 of 1048576",
                "Progress 524288 of 1048576", "Progress 1048576 of 1048576",
                "Downloaded demo",
            ],
        )

    def test_pip_unknown_total_still_reports_received_bytes(self) -> None:
        self.assertEqual(
            build_dist._format_pip_progress("Progress 2048 of 0"),
            "pip download [2.0 KB received]",
        )

    def test_captured_native_bars_are_visible_without_printing_every_redraw(self) -> None:
        with redirect_stdout(StringIO()) as output, build_dist.CommandOutputBox(live=False) as box:
            for now, percentage in ((10.0, 25), (11.0, 50), (13.0, 75)):
                frame = f"PASS 1 ------ {percentage}.0% | 1/4 modules | example"
                with patch.object(build_dist.time, "monotonic", return_value=now):
                    build_dist._render_output_text(frame + "\r", "", box)
        self.assertIn("25.0%", output.getvalue())
        self.assertNotIn("50.0%", output.getvalue())
        self.assertIn("75.0%", output.getvalue())

    @patch("cli.build_dist.time.monotonic", return_value=115.0)
    def test_silent_build_heartbeat_reports_live_work(self, _monotonic) -> None:  # type: ignore[no-untyped-def]
        class OutputBox:
            is_live = True

            def __init__(self) -> None:
                self.partial: list[str] = []

            def write_partial(self, text: str) -> None:
                self.partial.append(text)

        output_box = OutputBox()
        next_heartbeat = build_dist._report_build_heartbeat(
            output_box, 100.0, 80.0, activity="Nuitka", started_at=50.0, process_id=123
        )  # type: ignore[arg-type]

        self.assertEqual(next_heartbeat, 115.0)
        self.assertEqual(len(output_box.partial), 1)
        self.assertIn("Still working: Nuitka is running", output_box.partial[0])
        self.assertIn("35s", output_box.partial[0])
        self.assertIn("Elapsed: 65s", output_box.partial[0])
        self.assertIn("PID: 123", output_box.partial[0])

    def test_silent_status_redraws_each_second_but_captured_output_is_throttled(self) -> None:
        for live, interval in ((True, 1.0), (False, 15.0)):
            with self.subTest(live=live), redirect_stdout(StringIO()) as output:
                with build_dist.CommandOutputBox(live=live) as box:
                    with patch.object(build_dist.time, "monotonic", return_value=100.5):
                        self.assertEqual(build_dist._report_build_heartbeat(
                            box, 100.0, 100.0, activity="Nuitka"
                        ), 100.0)
                    self.assertNotIn("Still working", output.getvalue())
                    with patch.object(build_dist.time, "monotonic", return_value=100.0 + interval):
                        self.assertEqual(build_dist._report_build_heartbeat(
                            box, 100.0, 100.0, activity="Nuitka"
                        ), 100.0 + interval)
                    self.assertIn("Still working", output.getvalue())
                    with patch.object(build_dist.time, "monotonic", return_value=100.0 + 2 * interval):
                        build_dist._report_build_heartbeat(
                            box, 100.0 + interval, 100.0, activity="Nuitka"
                        )
                    self.assertEqual("\x1b[1A" in output.getvalue(), live)

    @patch("cli.build_dist.time.monotonic", return_value=300.0)
    def test_stall_monitor_reports_liveness_without_blocking_after_five_minutes(
        self, _monotonic
    ) -> None:  # type: ignore[no-untyped-def]
        message, state = build_dist._monitor_build_stall(
            0.0,
            build_dist.BuildStallState(build_dist.BUILD_STALL_SECONDS),
            activity="Nuitka",
        )

        self.assertIsNotNone(message)
        self.assertIn("process is still alive", message)
        self.assertIn("Ctrl+C", message)
        self.assertEqual(state.next_check, 600.0)

    @patch("cli.build_dist.time.monotonic", return_value=450.0)
    def test_stall_monitor_does_not_repeat_before_the_next_check(
        self, _monotonic
    ) -> None:  # type: ignore[no-untyped-def]
        message, state = build_dist._monitor_build_stall(
            0.0,
            build_dist.BuildStallState(600.0),
            activity="Nuitka",
        )

        self.assertIsNone(message)
        self.assertEqual(state.next_check, 600.0)

    def test_activity_names_the_executed_command_instead_of_a_package_argument(self) -> None:
        cases = (
            ([sys.executable, "-m", "pip", "install", "nuitka"], "pip install"),
            ([sys.executable, "-m", "pip", "check"], "pip check"),
            ([sys.executable, "-m", "ensurepip", "--upgrade"], "ensurepip"),
            ([sys.executable, "-u", "-m", "nuitka", "cli/main.py"], "Nuitka"),
            ([sys.executable, "-m", "nuitka", "cli/main.py"], "Nuitka"),
            ([sys.executable, "-X", "utf8", "-m", "pip", "install"], "pip install"),
            ([sys.executable, "-c", "print('nuitka')"], "Python command"),
            ([sys.executable, str(Path("cli") / "build_native.py")], "build_native.py"),
            (["ollama", "pull", "demo"], "ollama"),
            (["nuitka.exe", "cli/main.py"], "Nuitka"),
            (["nuitka.exe"], "Nuitka"),
        )
        for command, expected in cases:
            with self.subTest(command=command):
                self.assertEqual(build_dist._command_activity(command), expected)

    def test_silent_pip_subprocess_gets_dynamic_heartbeat_and_stall_messages(self) -> None:
        class OutputBox:
            is_live = True
            partial: list[str] = []
            completed: list[str] = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def write(self, text):
                self.completed.append(text)

            def write_partial(self, text):
                self.partial.append(text)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # A local pip stand-in exercises a real silent subprocess without
            # installing packages or relying on network access.
            (root / "pip.py").write_text(
                "import time; time.sleep(0.3); print('Installation finished')",
                encoding="utf-8",
            )
            with (
                patch.object(build_dist, "ROOT", root),
                patch.object(build_dist, "CommandOutputBox", OutputBox),
                patch.object(build_dist, "BUILD_HEARTBEAT_SECONDS", 0.05),
                patch.object(build_dist, "BUILD_STALL_SECONDS", 0.1),
            ):
                build_dist.run([sys.executable, "-u", "-m", "pip", "install", "nuitka"])

        self.assertTrue(OutputBox.partial)
        self.assertTrue(all("pip install is running" in line for line in OutputBox.partial))
        self.assertTrue(any("No new output from pip install" in line for line in OutputBox.completed))
        self.assertIn("Installation finished", OutputBox.completed)
        messages = " ".join(OutputBox.partial + OutputBox.completed)
        self.assertNotIn("Nuitka", messages)
        self.assertNotIn("linking", messages)

    def test_run_renders_unbuffered_progress_before_the_child_completes(self) -> None:
        class OutputBox:
            is_live = True
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

    def test_verbose_failure_streams_all_lines_but_retains_only_a_bounded_error_tail(self) -> None:
        class OutputBox:
            is_live = True
            completed: list[str] = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def write(self, text):
                self.completed.append(text)

            def write_partial(self, text):
                pass

        command = [
            sys.executable, "-u", "-c",
            "import sys; print('first compilation unit'); "
            "[print('compiled unit %04d: ' % i + 'x' * 80) for i in range(1000)]; "
            "print('link failed', file=sys.stderr); sys.exit(7)",
        ]
        with patch.object(build_dist, "CommandOutputBox", OutputBox):
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                build_dist.run(command)

        self.assertEqual(raised.exception.returncode, 7)
        self.assertEqual(len(OutputBox.completed), 1002)
        self.assertEqual(OutputBox.completed[0], "first compilation unit")
        self.assertEqual(OutputBox.completed[-1], "link failed")
        self.assertLessEqual(len(raised.exception.output), 65536)
        self.assertNotIn("first compilation unit", raised.exception.output)
        self.assertTrue(raised.exception.output.endswith("link failed"))

    @patch("cli.build_dist.log_completed_command")
    @patch("cli.build_dist.subprocess.run")
    @patch("cli.build_dist.subprocess.Popen")
    def test_interrupted_build_terminates_the_child_tree_without_replaying_output(
        self, popen_mock, taskkill_mock, completed_command
    ) -> None:  # type: ignore[no-untyped-def]
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
        completed_command.assert_called_once_with(
            ["tool"], "", return_code=None, interrupted=True
        )

    def test_command_execution_receives_every_flag_and_records_them_at_completion(self) -> None:
        command = [
            str(build_dist.VENV_PYTHON),
            "-u",
            "-m",
            "nuitka",
            "--standalone",
            "--include-module=typing",
            "--output-filename=diagnostic.exe",
            "--output-dir=build",
            str(build_dist.ROOT / "cli" / "diagnostic.py"),
        ]

        with (
            patch.object(build_dist, "command_preview") as preview,
            patch.object(build_dist, "_run_with_file_tailer") as runner,
        ):
            build_dist.run(command)
        preview.assert_called_once_with(command)
        runner.assert_called_once_with(command)

        quiet_command = [sys.executable, "-c", "pass", *[f"--option{index}=value" for index in range(12)]]
        with redirect_stdout(StringIO()) as output, self.assertLogs("aibrain.command", level="INFO") as captured:
            build_dist.run(quiet_command)
        self.assertIn("12 flags attached", output.getvalue())
        self.assertNotIn(BOX_TOP_LEFT + BOX_HORIZONTAL * 10, output.getvalue())
        self.assertNotIn("--option", output.getvalue())
        self.assertEqual(len(captured.records), 2)
        self.assertIn("Command started:", captured.records[0].getMessage())
        self.assertIn("Command completed", captured.records[1].getMessage())
        for record in captured.records:
            self.assertIn("--option11=value", record.getMessage())

    def test_each_packaged_application_has_its_named_entry_point_and_icon(self) -> None:
        targets = {target.executable: target for target in build_dist.APPLICATIONS}
        numpy_runtime = (Path("numpy-runtime"), Path("numpy-dlls"))

        self.assertEqual(set(targets), {"ai_brain.exe", "diagnostic.exe", "analysis.exe"})
        self.assertTrue(all(target.console_mode == "attach" for target in targets.values()))
        for target in targets.values():
            command = build_dist.nuitka_command(target, Path("build"), [Path("vcomp140.dll")], numpy_runtime)
            self.assertIn(f"--windows-icon-from-ico={target.icon}", command)
            self.assertIn(f"--output-filename={target.executable}", command)
            self.assertIn("--progress-bar=rich", command)
            for option in ("--verbose", "--show-progress", "--show-scons", "--show-modules", "--show-memory"):
                self.assertNotIn(option, command)
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

        self.assertEqual(targets["ai_brain.exe"].directory, "main")

    def test_focused_main_build_retains_the_ai_brain_cli_alias(self) -> None:
        expected = (build_dist.APPLICATIONS[0],)

        self.assertEqual(build_dist._selected_application_targets("main"), expected)
        self.assertEqual(build_dist._selected_application_targets("ai_brain"), expected)
        self.assertEqual(
            build_dist._selected_application_targets(None),
            build_dist.APPLICATIONS,
        )

    def test_merged_distribution_contains_all_apps_and_keeps_individual_builds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory)

            for target in build_dist.APPLICATIONS:
                application = release / target.directory
                (application / "dll").mkdir(parents=True)
                (application / target.executable).write_bytes(b"MZ" + b"x" * 100_000)
                (application / "dll" / build_dist.NATIVE_LIBRARY.name).write_bytes(b"native")
                (application / "msvcp140.dll").write_bytes(b"runtime")
                (application / "vcomp140.dll").write_bytes(b"runtime")
                (application / "shared-runtime.dll").write_bytes(b"shared")
                (application / f"{target.directory}.data").write_text(
                    target.executable,
                    encoding="utf-8",
                )

            with patch.object(build_dist, "set_windows_executable_gpu_preference") as gpu:
                merged = build_dist.merge_application_distributions(
                    release,
                    build_dist.APPLICATIONS,
                )

            self.assertEqual(merged, release / "AIBrain")
            self.assertEqual(
                {path.name for path in merged.glob("*.exe")},
                {"ai_brain.exe", "diagnostic.exe", "analysis.exe"},
            )
            for target in build_dist.APPLICATIONS:
                self.assertTrue((release / target.directory / target.executable).is_file())
                self.assertTrue((merged / f"{target.directory}.data").is_file())
            gpu.assert_called_once_with(merged / "ai_brain.exe")

    def test_merged_distribution_rejects_different_shared_runtime_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory)
            first, second = build_dist.APPLICATIONS[:2]
            for target, contents in ((first, b"first"), (second, b"second")):
                application = release / target.directory
                application.mkdir()
                (application / "shared-runtime.dll").write_bytes(contents)

            with self.assertRaisesRegex(RuntimeError, "copies of shared-runtime.dll differ"):
                build_dist.merge_application_distributions(
                    release,
                    (first, second),
                )

            self.assertFalse((release / "AIBrain").exists())
            self.assertFalse((release / "_AIBrain.merge").exists())

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
