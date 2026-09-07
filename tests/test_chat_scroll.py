"""Follow-output policy does not override a user's scroll position."""

from __future__ import annotations

import unittest

from src.app.chat_panel import ChatPanel


class ChatScrollTests(unittest.TestCase):
    def test_following_is_limited_to_the_bottom_with_a_small_tolerance(self) -> None:
        self.assertTrue(ChatPanel._is_at_bottom(100, 100))
        self.assertTrue(ChatPanel._is_at_bottom(98, 100))
        self.assertFalse(ChatPanel._is_at_bottom(97, 100))
