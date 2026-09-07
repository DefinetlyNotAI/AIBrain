from __future__ import annotations

import unittest

from src.app.chat_panel import markdown_to_html


class MarkdownRenderingTests(unittest.TestCase):
    def test_safe_markdown_renders_and_escapes_html(self) -> None:
        rendered = markdown_to_html(
            "# Title\n**bold** and `code` [link](https://example.com) <tag>"
        )
        self.assertIn("<h1>Title</h1>", rendered)
        self.assertIn("<b>bold</b>", rendered)
        self.assertIn("<code>code</code>", rendered)
        self.assertIn('href="https://example.com"', rendered)
        self.assertIn("&lt;tag&gt;", rendered)

    def test_encoded_apostrophe_is_rendered_as_plain_text(self) -> None:
        rendered = markdown_to_html("The world says &#x27;hello&#x27;.")

        self.assertEqual(rendered, "The world says 'hello'.")
