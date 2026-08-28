from __future__ import annotations

import inspect
import unittest
from src.app.main_window import MainWindow


class PackagedUtilityTests(unittest.TestCase):
    def test_main_opens_diagnostics_in_process_not_as_a_sibling_executable(self) -> None:
        source = inspect.getsource(MainWindow.open_diagnostics)

        self.assertIn("DiagnosticsWindow(self, auto_refresh=False)", source)
        self.assertIn("LoadingWindow(", source)
        self.assertNotIn("Popen", source)


if __name__ == "__main__":
    unittest.main()
