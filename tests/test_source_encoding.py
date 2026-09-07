from __future__ import annotations

import unittest
from pathlib import Path


class SourceEncodingTests(unittest.TestCase):
    def test_project_python_sources_contain_no_common_utf8_mojibake_markers(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        markers = (chr(0x00E2), chr(0x00C3), chr(0x00C2), chr(0xFFFD))
        source_files = [
            *root.joinpath("cli").glob("*.py"),
            *root.joinpath("src").rglob("*.py"),
            *root.joinpath("tests").glob("*.py"),
        ]

        for source_file in source_files:
            with self.subTest(source_file=source_file):
                content = source_file.read_text(encoding="utf-8")
                self.assertFalse(any(marker in content for marker in markers))


if __name__ == "__main__":
    unittest.main()
