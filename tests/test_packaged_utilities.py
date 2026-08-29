from __future__ import annotations

import inspect
import unittest
from src.app.main_window import MainWindow


class PackagedUtilityTests(unittest.TestCase):
    def test_main_launches_the_dedicated_diagnostics_application(self) -> None:
        source = inspect.getsource(MainWindow.open_diagnostics)

        self.assertIn('with_name("diagnostic.exe")', source)
        self.assertIn("QProcess.startDetached", source)
        self.assertNotIn("DiagnosticsWindow(", source)


if __name__ == "__main__":
    unittest.main()
