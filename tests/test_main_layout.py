from __future__ import annotations

import os
import inspect
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QScrollArea

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

    def test_visualizer_settings_expand_inside_a_fixed_scroll_area(self) -> None:
        window = MainWindow([])
        try:
            panel = window.visualizer
            panel.settings_toggle.click()

            self.assertIsInstance(panel.settings_panel, QScrollArea)
            self.assertFalse(panel.settings_panel.isHidden())
            self.assertEqual(panel.settings_panel.minimumHeight(), 154)
            self.assertEqual(panel.settings_panel.maximumHeight(), 154)
        finally:
            window.close()
            self.app.processEvents()

    def test_window_close_defers_without_blocking_on_worker_threads(self) -> None:
        source = inspect.getsource(MainWindow.closeEvent)

        self.assertIn("event.ignore()", source)
        self.assertIn("_shutdown_timer.start()", source)
        self.assertNotIn(".wait(", source)

    def test_cancelled_response_with_output_enables_post_stop_actions(self) -> None:
        window = MainWindow([])
        try:
            window.chat.set_model_available(True)
            window.history = [{"role": "user", "content": "Prompt"}]
            window._assistant_bubble = window.chat.add_message(
                "assistant", "Partial response"
            )
            window.visualizer._playback.append(object())  # type: ignore[arg-type]

            window._finished(
                {
                    "seconds": 1.0,
                    "generated_tokens": 3,
                    "prompt_tokens": 1,
                    "cancelled": True,
                }
            )

            self.assertTrue(window.chat.regenerate.isEnabled())
            self.assertTrue(window.chat.rewind.isEnabled())
            self.assertTrue(window.chat.open_analysis.isEnabled())
        finally:
            window.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
