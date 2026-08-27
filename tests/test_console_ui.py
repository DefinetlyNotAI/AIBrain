from __future__ import annotations

import unittest

from src.utils.console_ui import _console_text


class ConsoleUiEncodingTests(unittest.TestCase):
    def test_unicode_ui_literals_are_not_mojibake(self) -> None:
        text = "╭─╮ │ ✓ ✗ ● › ╰─╯"
        self.assertNotIn("â", text)
        self.assertNotIn("Ã", text)
        rendered = _console_text(text)
        self.assertNotIn("â", rendered)
        self.assertNotIn("Ã", rendered)
