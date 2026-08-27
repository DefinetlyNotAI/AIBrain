from __future__ import annotations

import subprocess
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from cli import build_dist


class BuildDistributionTests(unittest.TestCase):
    @patch("cli.build_dist.subprocess.run")
    def test_build_command_output_uses_shared_framed_panel(self, run_mock) -> None:  # type: ignore[no-untyped-def]
        run_mock.return_value = subprocess.CompletedProcess(["tool"], 0, "compiled", "warning")
        output = StringIO()

        with redirect_stdout(output):
            build_dist.run(["tool"])

        rendered = output.getvalue()
        self.assertIn("compiled", rendered)
        self.assertIn("warning", rendered)
        self.assertIn(chr(0x256D), rendered)

    @patch("cli.build_dist.subprocess.run")
    def test_failed_build_command_raises_after_rendering_output(self, run_mock) -> None:  # type: ignore[no-untyped-def]
        run_mock.return_value = subprocess.CompletedProcess(["tool"], 4, "", "broken")

        with self.assertRaises(subprocess.CalledProcessError) as raised:
            build_dist.run(["tool"])

        self.assertEqual(raised.exception.returncode, 4)


if __name__ == "__main__":
    unittest.main()
