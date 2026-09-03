from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.app.chat_export import write_chat_export


class ChatExportTests(unittest.TestCase):
    def test_json_export_preserves_complete_conversation_metadata(self) -> None:
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.json"
            write_chat_export(
                path, conversation, mode="normal", model="model:latest"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], "aibrain.chat.v1")
        self.assertEqual(payload["mode"], "normal")
        self.assertEqual(payload["model"], "model:latest")
        self.assertEqual(payload["conversation"], conversation)

    def test_text_export_labels_infinite_roles_and_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.txt"
            write_chat_export(
                path,
                [{"role": "participant", "content": "I enter.", "turn": 2}],
                mode="infinite",
                model="model:latest",
            )
            text = path.read_text(encoding="utf-8")

        self.assertIn("Mode: infinite", text)
        self.assertIn("PARTICIPANT 2", text)
        self.assertIn("I enter.", text)


if __name__ == "__main__":
    unittest.main()
