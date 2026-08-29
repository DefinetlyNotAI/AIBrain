from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.app.main_window import MainWindow


class MainLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_splitter_accepts_a_twenty_five_to_seventy_five_ratio(self) -> None:
        window = MainWindow([])
        try:
            window.show()
            self.app.processEvents()
            window.resize(1280, 720)
            window._splitter.setSizes([320, 960])
            self.app.processEvents()

            left, right = window._splitter.sizes()
            self.assertGreaterEqual(left, 300)
            self.assertGreaterEqual(right, 900)
            self.assertLess(window.minimumSizeHint().width(), 400)
        finally:
            window.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
