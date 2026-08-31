from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from src.utils.logging import (
    AlignedFormatter,
    BoundedFileHandler,
    ConsoleFormatter,
    FILE_LOG_LINE_WIDTH,
    MAX_LOG_BYTES,
    _uncaught_exception,
    configure_cli_logging,
    configure_logging,
    log_completed_command,
    report_exception,
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

    def test_formatter_uses_fixed_columns_and_word_aware_continuations(self) -> None:
        record = logging.LogRecord(
            "aibrain.test",
            logging.WARNING,
            "",
            0,
            "first line with enough words to require a continuation " * 4,
            (),
            None,
        )
        rendered = AlignedFormatter(colour=False).format(record)
        first, *continuations = rendered.splitlines()
        timestamp, severity, source, message = first.split(" | ", maxsplit=3)

        self.assertEqual(len(timestamp), 19)
        self.assertEqual(severity, "WARNING ")
        self.assertEqual(len(source), 28)
        self.assertTrue(message.startswith("first line"))
        self.assertTrue(continuations)
        self.assertTrue(all(len(line) <= FILE_LOG_LINE_WIDTH for line in rendered.splitlines()))
        self.assertTrue(all(line.startswith(" " * 19 + " | " + " " * 8 + " | ") for line in continuations))

    def test_console_formatter_uses_ui_style_messages_without_log_columns(self) -> None:
        record = logging.LogRecord(
            "aibrain.model_validator",
            logging.WARNING,
            "",
            0,
            "first\nsecond",
            (),
            None,
        )

        rendered = ConsoleFormatter(colour=False).format(record)
        first, second = rendered.splitlines()

        self.assertEqual(first, "  ! first")
        self.assertEqual(second, "    second")
        self.assertNotIn("aibrain.model_validator", rendered)
        self.assertNotRegex(rendered, r"\d{2}:\d{2}:\d{2}")
        self.assertNotIn("WARNING", rendered)

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

    def test_handled_exception_preserves_traceback_in_runtime_and_crash_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_log, crash_log = configure_logging("test", Path(directory))
            try:
                raise RuntimeError("complete traceback")
            except RuntimeError as exc:
                report_exception("Test entry point failed", exc)
            for handler in logging.getLogger().handlers:
                handler.flush()
            runtime_text = runtime_log.read_text(encoding="utf-8")
            crash_text = crash_log.read_text(encoding="utf-8")
            self._close_root_handlers()

        self.assertIn("Test entry point failed", runtime_text)
        self.assertIn("Traceback (most recent call last)", runtime_text)
        self.assertIn("RuntimeError: complete traceback", runtime_text)
        self.assertIn("Traceback (most recent call last)", crash_text)

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

    def test_runtime_logs_are_limited_to_twenty_megabytes(self) -> None:
        self.assertEqual(MAX_LOG_BYTES, 20 * 1024 * 1024)

    def test_cli_logging_excludes_decorative_stdout_and_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_log, _ = configure_cli_logging("test", Path(directory))
            try:
                print("standard CLI output")
                print("\x1b[31mCLI error output\x1b[0m", file=sys.stderr)
                captured = runtime_log.read_text(encoding="utf-8")
            finally:
                restore_cli_output()
                self._close_root_handlers()

        self.assertNotIn("standard CLI output", captured)
        self.assertNotIn("CLI error output", captured)

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
        first_line = captured.splitlines()[0]
        self.assertRegex(first_line, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| INFO")

    def test_completed_command_output_is_logged_once_without_console_decoration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            console = StringIO()
            with redirect_stderr(console):
                runtime_log, _ = configure_logging("test", Path(directory))
                log_completed_command(
                    ["python", "-m", "unittest"],
                    "ok\nfinished",
                    return_code=0,
                )
                for handler in logging.getLogger().handlers:
                    handler.flush()
                captured = runtime_log.read_text(encoding="utf-8")
                self._close_root_handlers()

        self.assertIn("Command completed with exit code 0", captured)
        self.assertIn("ok", captured)
        self.assertNotIn("+---", captured)
        self.assertEqual(console.getvalue(), "")

    def test_new_run_removes_all_stale_application_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_directory = Path(directory)
            stale = log_directory / "aibrain.main.earlier-run.log"
            other_feature = log_directory / "crash.build_dist.log"
            stale.write_text("old", encoding="utf-8")
            other_feature.write_text("old crash", encoding="utf-8")
            runtime_log, _ = configure_logging("main", log_directory)
            self.assertFalse(stale.exists())
            self.assertFalse(other_feature.exists())
            self._close_root_handlers()

        self.assertEqual(runtime_log.name, "aibrain.main.log")

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
            self.assertIn("report_exception", source, script)

    def test_diagnostics_cli_records_startup_and_shutdown_lifecycle(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "cli" / "diagnostic.py").read_text(encoding="utf-8-sig")

        self.assertIn("Starting AIBrain Diagnostics CLI", source)
        self.assertIn("Diagnostics runtime preflight passed", source)
        self.assertIn("Diagnostics window exited", source)


if __name__ == "__main__":
    unittest.main()
