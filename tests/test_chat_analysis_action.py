from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QScrollArea

from src.app.chat_panel import ChatPanel, MarkdownLabel
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
        self.assertFalse(panel.regenerate.isEnabled())
        self.assertEqual(panel.messages_layout.count(), 1)
        self.assertEqual(panel.open_analysis.text(), "Analysis+")
        panel.deleteLater()

    def test_mode_actions_are_collapsible_without_moving_controls(self) -> None:
        panel = ChatPanel(GenerationConfig())

        self.assertIsInstance(panel.mode_actions_content, QScrollArea)
        self.assertEqual(panel.mode_actions_content.minimumHeight(), 124)
        self.assertEqual(panel.mode_actions_content.maximumHeight(), 124)
        self.assertFalse(panel.mode_actions_content.isHidden())
        panel.mode_actions_toggle.click()
        self.assertTrue(panel.mode_actions_content.isHidden())
        self.assertEqual(panel.mode_actions_toggle.text(), "Show Mode Actions")
        panel.deleteLater()

    def test_advanced_controls_expand_inside_a_fixed_scroll_area(self) -> None:
        panel = ChatPanel(GenerationConfig())

        panel.advanced_toggle.click()

        self.assertIsInstance(panel.advanced_content, QScrollArea)
        self.assertFalse(panel.advanced_content.isHidden())
        self.assertEqual(panel.advanced_content.minimumHeight(), 188)
        self.assertEqual(panel.advanced_content.maximumHeight(), 188)
        self.assertIsNotNone(panel.advanced_content.widget())
        panel.deleteLater()

    def test_regenerate_requires_a_completed_normal_response(self) -> None:
        panel = ChatPanel(GenerationConfig())
        panel.set_model_available(True)

        self.assertFalse(panel.regenerate.isEnabled())
        panel.set_regenerate_available(True)
        self.assertTrue(panel.regenerate.isEnabled())
        panel.set_regenerate_available(False)
        self.assertFalse(panel.regenerate.isEnabled())
        panel.deleteLater()

    def test_stopped_generation_restores_all_recorded_session_actions(self) -> None:
        panel = ChatPanel(GenerationConfig())
        panel.set_model_available(True)
        panel.set_analysis_available(True)
        panel.set_regenerate_available(True)
        panel.set_rewind_available(True)

        panel.generating(True)
        self.assertFalse(panel.regenerate.isEnabled())
        self.assertFalse(panel.rewind.isEnabled())
        self.assertFalse(panel.open_analysis.isEnabled())

        panel.generating(False)
        self.assertTrue(panel.regenerate.isEnabled())
        self.assertTrue(panel.rewind.isEnabled())
        self.assertTrue(panel.open_analysis.isEnabled())
        panel.deleteLater()

    def test_infinite_mode_allows_recorded_rewind_and_analysis_after_stop(self) -> None:
        panel = ChatPanel(GenerationConfig())
        panel.set_model_available(True)
        panel.infinite_mode.click()
        panel.set_analysis_available(True)
        panel.set_rewind_available(True)

        self.assertTrue(panel.rewind.isEnabled())
        self.assertTrue(panel.open_analysis.isEnabled())
        self.assertFalse(panel.regenerate.isEnabled())
        panel.deleteLater()

    def test_streamed_markdown_is_batched_before_relayout(self) -> None:
        label = MarkdownLabel("Start")

        label.append_markdown(" one")
        label.append_markdown(" two")

        self.assertTrue(label._render_timer.isActive())
        self.assertEqual(label.text(), "Start one two")
        label._flush_markdown()
        self.assertIn("Start one two", QLabel.text(label))
        label.deleteLater()

    def test_rewind_mode_locks_chat_actions_until_the_graph_exits(self) -> None:
        panel = ChatPanel(GenerationConfig())
        panel.set_model_available(True)
        panel.set_rewind_mode(True)
        self.assertFalse(panel.send.isEnabled())
        self.assertFalse(panel.clear.isEnabled())
        panel.set_rewind_mode(False)
        self.assertFalse(panel.send.isEnabled())
        panel.input.setPlainText("Resume normal chat")
        self.assertTrue(panel.send.isEnabled())
        panel.deleteLater()


if __name__ == "__main__":
    unittest.main()
