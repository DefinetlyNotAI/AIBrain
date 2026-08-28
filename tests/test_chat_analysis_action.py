from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.app.chat_panel import ChatPanel
from src.models.llama_backend import GenerationConfig


class ChatAnalysisActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_analysis_action_requires_a_completed_exportable_session(self) -> None:
        panel = ChatPanel(GenerationConfig())

        self.assertFalse(panel.open_analysis.isEnabled())
        self.assertFalse(panel.send.isEnabled())
        panel.set_model_available(True)
        panel.set_analysis_available(True)
        self.assertTrue(panel.open_analysis.isEnabled())
        panel.generating(True)
        self.assertFalse(panel.open_analysis.isEnabled())
        panel.generating(False)
        self.assertTrue(panel.open_analysis.isEnabled())
        panel.set_analysis_available(False)
        self.assertFalse(panel.open_analysis.isEnabled())
        panel.deleteLater()

    def test_mode_switch_clears_messages_and_changes_the_available_actions(self) -> None:
        panel = ChatPanel(GenerationConfig())
        panel.set_model_available(True)
        panel.add_message("user", "A prior conversation")
        panel.infinite_mode.click()

        self.assertFalse(panel.infinite.isHidden())
        self.assertTrue(panel.regenerate.isHidden())
        self.assertEqual(panel.messages_layout.count(), 1)
        self.assertEqual(panel.open_analysis.text(), "Analysis+")
        panel.deleteLater()


if __name__ == "__main__":
    unittest.main()
