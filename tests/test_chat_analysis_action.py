from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QScrollArea

from src.app.chat_panel import ChatPanel, MarkdownLabel
from src.models.infinite_simulation import WORLD_OPENINGS
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

    def test_empty_infinite_compose_uses_random_world_event(self) -> None:
        panel = ChatPanel(GenerationConfig())
        requested: list[str] = []
        panel.infiniteRequested.connect(requested.append)
        panel.set_model_available(True)
        panel.infinite_mode.click()

        self.assertTrue(panel.send.isEnabled())
        self.assertEqual(panel.send.text(), "🎲")
        self.assertIn("random World prompt", panel.send.accessibleName())
        with patch("src.app.chat_panel.random_world_opening", return_value=WORLD_OPENINGS[3]):
            panel.send.click()

        self.assertEqual(requested, [WORLD_OPENINGS[3]])
        panel.deleteLater()

    def test_custom_infinite_compose_sends_world_event_directly(self) -> None:
        panel = ChatPanel(GenerationConfig())
        requested: list[str] = []
        panel.infiniteRequested.connect(requested.append)
        panel.set_model_available(True)
        panel.infinite_mode.click()
        panel.input.setPlainText("The observatory dome opens by itself.")

        self.assertEqual(panel.send.text(), "➤")
        panel.send.click()

        self.assertEqual(requested, ["The observatory dome opens by itself."])
        self.assertEqual(panel.input.toPlainText(), "")
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

    def test_advanced_settings_button_opens_a_scrollable_dialog(self) -> None:
        panel = ChatPanel(GenerationConfig())

        with patch.object(panel.advanced_dialog, "exec", return_value=0) as execute:
            panel.advanced_settings.click()

        execute.assert_called_once_with()
        self.assertIsInstance(panel.advanced_dialog.settings_scroll, QScrollArea)
        self.assertIsNotNone(panel.advanced_dialog.settings_scroll.widget())
        self.assertEqual(panel.analysis_cache_mb.minimum(), 0)
        self.assertEqual(panel.analysis_cache_mb.value(), 1024)
        self.assertEqual(panel.analysis_cache_mb.specialValueText(), "Disabled")
        panel.temperature.setValue(1.1)
        self.assertEqual(panel.config().temperature, 1.1)
        panel.deleteLater()

    def test_reopened_mode_actions_return_to_the_first_control(self) -> None:
        panel = ChatPanel(GenerationConfig())
        scrollbar = panel.mode_actions_content.verticalScrollBar()
        scrollbar.setRange(0, 100)
        scrollbar.setValue(100)

        panel.mode_actions_toggle.click()
        panel.mode_actions_toggle.click()

        self.assertEqual(scrollbar.value(), scrollbar.minimum())
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

    def test_chat_export_tracks_transcript_availability(self) -> None:
        panel = ChatPanel(GenerationConfig())

        self.assertFalse(panel.export_chat.isEnabled())
        panel.set_chat_export_available(True)
        self.assertTrue(panel.export_chat.isEnabled())
        panel.generating(True)
        self.assertFalse(panel.export_chat.isEnabled())
        panel.generating(False)
        self.assertTrue(panel.export_chat.isEnabled())
        panel.deleteLater()

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
