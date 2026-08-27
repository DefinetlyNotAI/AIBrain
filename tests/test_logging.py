from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from src.utils.logging import AlignedFormatter, BoundedFileHandler, configure_logging


class LoggingTests(unittest.TestCase):
    @staticmethod
    def _close_root_handlers() -> None:
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()

    def tearDown(self) -> None:
        self._close_root_handlers()

    def test_formatter_preserves_multiline_message_with_aligned_gutter(self) -> None:
        record = logging.LogRecord("aibrain.test", logging.WARNING, "", 0, "first\nsecond", (), None)
        rendered = AlignedFormatter(colour=False).format(record)
        first, second = rendered.splitlines()
        self.assertTrue(first.endswith("first"))
        self.assertEqual(second, " " * first.index("first") + "second")

    def test_configure_logging_replaces_previous_logs_and_keeps_crash_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_directory = Path(directory)
            (log_directory / "aibrain.log").write_text("old", encoding="utf-8")
            runtime_log, crash_log = configure_logging(log_directory)
            logger = logging.getLogger("aibrain.test")
            logger.warning("first line\nsecond line")
            logger.error("recorded failure")
            for handler in logging.getLogger().handlers:
                handler.flush()

            runtime_text = runtime_log.read_text(encoding="utf-8")
            self.assertNotIn("old", runtime_text)
            self.assertIn("first line\n", runtime_text)
            self.assertIn("recorded failure", crash_log.read_text(encoding="utf-8"))
            self._close_root_handlers()

    def test_bounded_handler_discards_oldest_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bounded.log"
            handler = BoundedFileHandler(path, max_bytes=60)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger = logging.getLogger("aibrain.bound-test")
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
            try:
                logger.info("oldest-aaaaaaaaaaaaaaaaaaaaaaaa")
                logger.info("newest-bbbbbbbbbbbbbbbbbbbbbbbb")
            finally:
                logger.removeHandler(handler)
                handler.close()
                logger.propagate = True

            text = path.read_text(encoding="utf-8")
            self.assertNotIn("oldest", text)
            self.assertIn("newest", text)


if __name__ == "__main__":
    unittest.main()
