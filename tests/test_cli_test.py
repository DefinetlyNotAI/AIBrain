from __future__ import annotations

import subprocess
from pathlib import Path
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from cli import test


class TestLauncherTests(unittest.TestCase):
    @patch("cli.test.subprocess.run")
    def test_run_suite_renders_command_output_and_status(self, run_mock) -> None:  # type: ignore[no-untyped-def]
        run_mock.return_value = subprocess.CompletedProcess(["python"], 0, "ok", "")
        output = StringIO()

        with redirect_stdout(output):
            result = test.run_suite("Example", ["tests.example"])

        self.assertEqual(result, 0)
        self.assertIn("ok", output.getvalue())
        self.assertIn("PASS", output.getvalue())

    def test_menu_contains_distinct_verification_groups(self) -> None:
        self.assertEqual(
            set(test.SUITES),
            {"1", "2", "3", "4", "5"},
        )

        modules = [module for _title, suite in test.SUITES.values() for module in suite]

        self.assertEqual(
            len(modules),
            len(set(modules)),
        )

    @patch("cli.test.in_managed_virtual_environment", return_value=False)
    @patch("cli.test.subprocess.run")
    @patch("pathlib.Path.is_file", return_value=True)
    def test_global_python_relaunches_the_managed_test_runtime(
        self, _is_file, run_mock, _managed
    ) -> None:  # type: ignore[no-untyped-def]
        run_mock.return_value = subprocess.CompletedProcess([], 0)

        self.assertEqual(test.main(), 0)

        command = run_mock.call_args.args[0]
        self.assertEqual(Path(command[0]).name, "python.exe")
        self.assertEqual(Path(command[1]).name, "test.py")

    @patch("cli.test.subprocess.run")
    def test_empty_suite_runs_discovery_for_every_test_file(self, run_mock) -> None:  # type: ignore[no-untyped-def]
        run_mock.return_value = subprocess.CompletedProcess(["python"], 0, "ok", "")

        with redirect_stdout(StringIO()):
            self.assertEqual(
                test.run_suite(test.ALL_TESTS, []),
                0,
            )

        self.assertEqual(
            run_mock.call_args.args[0][-4:],
            ["discover", "-s", "tests", "-b"],
        )

    @patch("builtins.input", side_effect=EOFError)
    def test_menu_exits_cleanly_when_interactive_input_is_unavailable(self, _input_mock) -> None:  # type: ignore[no-untyped-def]
        with redirect_stdout(StringIO()):
            self.assertIsNone(test.choose_suite())

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_menu_propagates_keyboard_interrupt_to_the_cli_exit_handler(self, _input_mock) -> None:  # type: ignore[no-untyped-def]
        with redirect_stdout(StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                test.choose_suite()


if __name__ == "__main__":
    unittest.main()
