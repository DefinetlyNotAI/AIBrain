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
        panel.set_analysis_available(True)
        self.assertTrue(panel.open_analysis.isEnabled())
        panel.generating(True)
        self.assertFalse(panel.open_analysis.isEnabled())
        panel.generating(False)
        self.assertTrue(panel.open_analysis.isEnabled())
        panel.set_analysis_available(False)
        self.assertFalse(panel.open_analysis.isEnabled())
        panel.deleteLater()


if __name__ == "__main__":
    unittest.main()
