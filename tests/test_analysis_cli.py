from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from cli.analysis import relative_age


class AnalysisCliTests(unittest.TestCase):
    def test_relative_age_reports_recent_and_older_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest"
            manifest.touch()
            self.assertEqual(relative_age(manifest), "less than one minute")

            old = manifest.stat().st_mtime - 2 * 86_400
            os.utime(manifest, (old, old))
            self.assertTrue(relative_age(manifest).startswith("2 days"))


if __name__ == "__main__":
    unittest.main()
