from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.app.main_window as main_window_module
from src.app.main_window import MainWindow


class PackagedUtilityTests(unittest.TestCase):
    def test_launches_sibling_utility_from_standalone_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory)
            main_executable = release / "ai_brain" / "ai_brain.exe"
            diagnostic = release / "diagnostic" / "diagnostic.exe"
            main_executable.parent.mkdir()
            diagnostic.parent.mkdir()
            main_executable.touch()
            diagnostic.touch()

            with patch.dict(main_window_module.__dict__, {"__compiled__": True}):
                with patch("src.app.main_window.sys.executable", str(main_executable)):
                    with patch("src.app.main_window.subprocess.Popen") as popen:
                        self.assertTrue(MainWindow._launch_packaged_utility("diagnostic", "diagnostic.exe"))

            popen.assert_called_once_with([str(diagnostic)], close_fds=False)

    def test_development_mode_uses_the_in_process_fallback(self) -> None:
        with patch.dict(main_window_module.__dict__, {}, clear=False):
            main_window_module.__dict__.pop("__compiled__", None)
            self.assertFalse(MainWindow._launch_packaged_utility("diagnostic", "diagnostic.exe"))


if __name__ == "__main__":
    unittest.main()
