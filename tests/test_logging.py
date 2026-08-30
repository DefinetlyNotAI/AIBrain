from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.logging import (
    AlignedFormatter,
    BoundedFileHandler,
    _uncaught_exception,
    configure_cli_logging,
    configure_logging,
    _start_fresh_log,
    restore_cli_output,
)


class LoggingTests(unittest.TestCase):
    @staticmethod
    def _close_root_handlers() -> None:
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()

    def tearDown(self) -> None:
        restore_cli_output()
        self._close_root_handlers()

    def test_formatter_preserves_multiline_message_with_aligned_gutter(self) -> None:
        record = logging.LogRecord("aibrain.test", logging.WARNING, "", 0, "first\nsecond", (), None)
        rendered = AlignedFormatter(colour=False).format(record)
        first, second = rendered.splitlines()
        self.assertTrue(first.endswith("first"))
        self.assertEqual(second, " " * first.index("first") + "second")

    def test_feature_logs_are_scoped_and_crash_logs_are_lazy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_directory = Path(directory)
            (log_directory / "aibrain.main.log").write_text("old", encoding="utf-8")
            runtime_log, crash_log = configure_logging("diagnostic", log_directory)
            logger = logging.getLogger("aibrain.test")
            logger.warning("first line\nsecond line")
            logger.error("recorded failure")
            for handler in logging.getLogger().handlers:
                handler.flush()

            runtime_text = runtime_log.read_text(encoding="utf-8")
            self.assertEqual(runtime_log.name, "aibrain.diagnostic.log")
            self.assertEqual(crash_log.name, "crash.diagnostic.log")
            self.assertIn("first line\n", runtime_text)
            self.assertIn("recorded failure", runtime_text)
            self.assertFalse(crash_log.exists())
            try:
                raise RuntimeError("boom")
            except RuntimeError as exc:
                _uncaught_exception(type(exc), exc, exc.__traceback__)
            self.assertIn("RuntimeError: boom", crash_log.read_text(encoding="utf-8"))
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

    def test_cli_logging_captures_stdout_and_stderr_without_ansi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_log, _ = configure_cli_logging("test", Path(directory))
            try:
                print("standard CLI output")
                print("\x1b[31mCLI error output\x1b[0m", file=sys.stderr)
                sys.stdout.flush()
                sys.stderr.flush()
                captured = runtime_log.read_text(encoding="utf-8")
            finally:
                restore_cli_output()
                self._close_root_handlers()

        self.assertIn("standard CLI output", captured)
        self.assertIn("CLI error output", captured)
        self.assertNotIn("\x1b[31m", captured)

    def test_cli_logging_records_application_logger_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_log, _ = configure_cli_logging("analysis", Path(directory))
            try:
                logging.getLogger("aibrain.analysis").info(
                    "inspection complete: Healthy"
                )
                for handler in logging.getLogger().handlers:
                    handler.flush()
                captured = runtime_log.read_text(encoding="utf-8")
            finally:
                restore_cli_output()
                self._close_root_handlers()

        self.assertIn("inspection complete: Healthy", captured)

    def test_locked_feature_log_uses_an_isolated_timestamped_run_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aibrain.main.log"
            with patch.object(Path, "unlink", side_effect=PermissionError("locked")):
                selected = _start_fresh_log(path)

        self.assertNotEqual(selected, path)
        self.assertTrue(selected.name.startswith("aibrain.main."))
        self.assertEqual(selected.suffix, ".log")

    def test_reconfiguring_logging_closes_existing_handlers_before_log_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_directory = Path(directory)
            first_runtime, _ = configure_logging("main", log_directory)
            logging.getLogger("aibrain.test").info("first process")
            try:
                second_runtime, _ = configure_logging("main", log_directory)
            finally:
                self._close_root_handlers()

        self.assertEqual(first_runtime, second_runtime)

    def test_every_cli_entry_point_configures_console_capture(self) -> None:
        root = Path(__file__).resolve().parents[1]
        scripts = (
            "analysis.py",
            "build_dist.py",
            "build_native.py",
            "diagnostic.py",
            "installer.py",
            "main.py",
            "test.py",
        )

        for script in scripts:
            source = (root / "cli" / script).read_text(encoding="utf-8-sig")
            self.assertIn("configure_cli_logging", source, script)

    def test_diagnostics_cli_records_startup_and_shutdown_lifecycle(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "cli" / "diagnostic.py").read_text(encoding="utf-8-sig")

        self.assertIn("Starting AIBrain Diagnostics CLI", source)
        self.assertIn("Diagnostics runtime preflight passed", source)
        self.assertIn("Diagnostics window exited", source)


if __name__ == "__main__":
    unittest.main()
