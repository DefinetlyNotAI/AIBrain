from __future__ import annotations

import subprocess
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
        self.assertEqual(set(test.SUITES), {"1", "2", "3"})
        modules = [module for _title, suite in test.SUITES.values() for module in suite]
        self.assertEqual(len(modules), len(set(modules)))

    @patch("cli.test.subprocess.run")
    def test_empty_suite_runs_discovery_for_every_test_file(self, run_mock) -> None:  # type: ignore[no-untyped-def]
        run_mock.return_value = subprocess.CompletedProcess(["python"], 0, "ok", "")

        self.assertEqual(test.run_suite(test.ALL_TESTS, []), 0)
        self.assertEqual(run_mock.call_args.args[0][-3:], ["discover", "-s", "tests"])


if __name__ == "__main__":
    unittest.main()
