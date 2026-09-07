"""Project-local, readable runtime and crash logging."""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import textwrap
import threading
import traceback as traceback_module
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import ClassVar

from .console_ui import BULLET, CROSS, Color, color, console_message_lines

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    EOFError,
    ImportError,
    LookupError,
    MemoryError,
    OSError,
    ReferenceError,
    RuntimeError,
    TypeError,
    ValueError,
)
MAX_LOG_BYTES = 20 * 1024 * 1024
MAX_CRASH_LOG_BYTES = MAX_LOG_BYTES
FILE_LOG_LINE_WIDTH = 140
_TIME_WIDTH = 19
_SEVERITY_WIDTH = 8
_SOURCE_WIDTH = 28

_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|][^\x07]*(?:\x07|\x1b\\))")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def restore_cli_output() -> None:
    """Compatibility hook for callers that previously restored console tees."""


class AlignedFormatter(logging.Formatter):
    """Write fixed-column, word-wrapped runtime records for developer logs."""

    _COLOURS: ClassVar[Mapping[int, str]] = {
        logging.DEBUG: "\x1b[38;5;245m",
        logging.INFO: "\x1b[38;5;45m",
        logging.WARNING: "\x1b[38;5;220m",
        logging.ERROR: "\x1b[38;5;203m",
        logging.CRITICAL: "\x1b[1;38;5;196m",
    }
    _RESET = "\x1b[0m"

    def __init__(self, *, colour: bool) -> None:
        super().__init__()
        self.colour = colour

    def format(self, record: logging.LogRecord) -> str:
        timestamp = (
            datetime.fromtimestamp(record.created)
            .astimezone()
            .strftime("%Y-%m-%d %H:%M:%S")
        )
        source = record.name.removeprefix("src.").removeprefix("aibrain.")
        if len(source) > _SOURCE_WIDTH:
            source = source[: _SOURCE_WIDTH - 3] + "..."
        prefix = (
            f"{timestamp:<{_TIME_WIDTH}} | {record.levelname:<{_SEVERITY_WIDTH}} | "
            f"{source:<{_SOURCE_WIDTH}} | "
        )
        continuation = (
            f"{'':<{_TIME_WIDTH}} | {'':<{_SEVERITY_WIDTH}} | {'':<{_SOURCE_WIDTH}} | "
        )
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        elif record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"

        available = max(FILE_LOG_LINE_WIDTH - len(prefix), 1)
        lines = message.splitlines() or [""]
        wrapped = [
            segment
            for line in lines
            for segment in (
                    textwrap.wrap(
                        line,
                        width=available,
                        break_long_words=True,
                        break_on_hyphens=False,
                    )
                    or [""]
            )
        ]
        rendered = "\n".join(
            [
                f"{prefix}{wrapped[0]}",
                *[f"{continuation}{line}" for line in wrapped[1:]],
            ]
        )
        if self.colour:
            return f"{self._COLOURS.get(record.levelno, '')}{rendered}{self._RESET}"
        return rendered


class ConsoleFormatter(logging.Formatter):
    """Render runtime records as compact messages that match the CLI UI."""

    _PRESENTATION: ClassVar[Mapping[int, tuple[str, str]]] = {
        logging.DEBUG: ("·", Color.GRAY),
        logging.INFO: (BULLET, Color.CYAN),
        logging.WARNING: ("!", Color.YELLOW),
        logging.ERROR: (CROSS, Color.RED),
        logging.CRITICAL: (CROSS, Color.RED),
    }

    def __init__(self, *, colour: bool) -> None:
        super().__init__()
        self.colour = colour

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        elif record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"

        marker, tone = self._PRESENTATION.get(record.levelno, (BULLET, Color.CYAN))
        prefix = f"  {marker} "
        rendered = "\n".join(console_message_lines(prefix, message))
        return color(rendered, tone, Color.BOLD) if self.colour else rendered


class ConsoleRecordFilter(logging.Filter):
    """Keep file-only command transcripts out of the interactive console."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name != "aibrain.command"


class BoundedFileHandler(logging.FileHandler):
    """Keep the newest log data and discard old complete lines above the cap."""

    def __init__(
            self, filename: Path, *, max_bytes: int = MAX_LOG_BYTES, delay: bool = False
    ) -> None:
        super().__init__(filename, mode="a", encoding="utf-8", delay=delay)
        self.max_bytes = max_bytes

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self._trim()

    def _trim(self) -> None:
        try:
            self.flush()
            path = Path(self.baseFilename)
            if path.stat().st_size <= self.max_bytes:
                return
            with path.open("rb") as source:
                source.seek(-self.max_bytes, 2)
                source.readline()
                retained = source.read()
            with path.open("wb") as destination:
                destination.write(retained)
        except OSError:
            # Logging cannot safely report a logging-storage failure without
            # risking recursive writes to this same handler.
            return


class CrashFileHandler(BoundedFileHandler):
    """Create a large crash log only for an uncaught exception record."""

    def emit(self, record: logging.LogRecord) -> None:
        if not record.exc_info or record.name != "aibrain.crash":
            return
        super().emit(record)


def _uncaught_exception(
        exc_type: type[BaseException], value: BaseException, traceback: TracebackType | None
) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        return
    report_exception("Unhandled application exception", value, traceback)


def format_exception(exception: BaseException) -> str:
    """Return an exception with every traceback line preserved for display."""
    return "".join(
        traceback_module.format_exception(
            type(exception), exception, exception.__traceback__
        )
    ).rstrip()


def log_completed_command(
        command_line: list[str],
        output: str,
        *,
        return_code: int | None,
        interrupted: bool = False,
) -> None:
    """Persist command output once its process has completed or been stopped."""
    state = "interrupted" if interrupted else "completed"
    exit_detail = "" if return_code is None else f" with exit code {return_code}"
    message = f"Command {state}{exit_detail}: {subprocess.list2cmdline(command_line)}"
    clean_output = _strip_ansi(output).strip()
    if clean_output:
        message = f"{message}\n{clean_output}"
    logging.getLogger("aibrain.command").info("%s", message)


def report_exception(
        context: str,
        exception: BaseException,
        traceback: TracebackType | None = None,
) -> None:
    """Record a handled fatal exception once, including its complete traceback.

    CLI entry points call this at their outer boundary.  When logging has not
    started yet (for example, a failure while configuring it), fall back to
    stderr so no traceback is swallowed.
    """
    exception_traceback = (
        traceback if traceback is not None else exception.__traceback__
    )
    if logging.getLogger().handlers:
        logging.getLogger("aibrain.crash").critical(
            context,
            exc_info=(type(exception), exception, exception_traceback),
        )
        return

    sys.stderr.write(f"{context}\n")
    traceback_module.print_exception(
        type(exception), exception, exception_traceback, file=sys.stderr
    )


def _thread_exception(args: threading.ExceptHookArgs) -> None:
    value = args.exc_value
    if value is not None:
        _uncaught_exception(
            args.exc_type,
            value,
            args.exc_traceback,
        )


def _start_fresh_log(path: Path) -> Path:
    """Truncate the stable feature log so every run replaces the previous one."""
    if path.exists():
        path.write_text("", encoding="utf-8")
    return path


def _clear_previous_run_logs(directory: Path, feature: str) -> None:
    """Remove only stale logs belonging to the application being started."""
    names = tuple(f"{prefix}.{feature}" for prefix in ("aibrain", "crash"))
    for path in directory.iterdir():
        if not path.is_file() or path.suffix != ".log":
            continue
        stem = path.name.removesuffix(path.suffix)
        if not any(stem == name or stem.startswith(f"{name}.") for name in names):
            continue
        try:
            path.unlink()
        except PermissionError:
            # The stable path is truncated separately. This branch primarily
            # covers legacy timestamped files still held by an older process.
            continue


def configure_logging(
        feature: str | Path = "main", log_directory: Path | None = None
) -> tuple[Path, Path]:
    """Start fresh normal and crash logs for this application run.

    Both files are bounded while the program runs so a long session preserves
    the newest details instead of consuming disk space indefinitely.
    """
    if isinstance(feature, Path):
        log_directory = feature
        feature = "main"
    directory = log_directory or PROJECT_ROOT / "logs"
    directory.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    _clear_previous_run_logs(directory, feature)
    runtime_log = _start_fresh_log(directory / f"aibrain.{feature}.log")
    crash_log = _start_fresh_log(directory / f"crash.{feature}.log")
    root.setLevel(logging.INFO)

    formatter = ConsoleFormatter(colour=sys.stderr.isatty())
    stream = logging.StreamHandler()
    stream.addFilter(ConsoleRecordFilter())
    stream.setFormatter(formatter)
    root.addHandler(stream)

    runtime_handler = BoundedFileHandler(runtime_log)
    runtime_handler.setFormatter(AlignedFormatter(colour=False))
    root.addHandler(runtime_handler)

    crash_handler = CrashFileHandler(
        crash_log, max_bytes=MAX_CRASH_LOG_BYTES, delay=True
    )
    crash_handler.setLevel(logging.CRITICAL)
    crash_handler.setFormatter(AlignedFormatter(colour=False))
    root.addHandler(crash_handler)

    sys.excepthook = _uncaught_exception
    threading.excepthook = _thread_exception
    return runtime_log, crash_log


def configure_cli_logging(
        feature: str, log_directory: Path | None = None
) -> tuple[Path, Path]:
    """Configure file logging without copying decorative console presentation."""
    restore_cli_output()
    return configure_logging(feature, log_directory)
