"""Project-local, readable runtime and crash logging."""
from __future__ import annotations

import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from types import TracebackType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_LOG_BYTES = 5 * 1024 * 1024
MAX_CRASH_LOG_BYTES = 20 * 1024 * 1024


class AlignedFormatter(logging.Formatter):
    """Format complete multi-line records with an aligned continuation gutter."""

    _COLOURS = {
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
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        prefix = f"{timestamp} | {record.levelname:<8} | {record.name:<28} | "
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        elif record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"

        lines = message.splitlines() or [""]
        rendered = "\n".join([f"{prefix}{lines[0]}", *[(" " * len(prefix)) + line for line in lines[1:]]])
        if self.colour:
            return f"{self._COLOURS.get(record.levelno, '')}{rendered}{self._RESET}"
        return rendered


class BoundedFileHandler(logging.FileHandler):
    """Keep the newest log data and discard old complete lines above the cap."""

    def __init__(self, filename: Path, *, max_bytes: int = MAX_LOG_BYTES, delay: bool = False) -> None:
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
    logging.getLogger("aibrain.crash").critical(
        "Unhandled application exception", exc_info=(exc_type, value, traceback)
    )


def _thread_exception(args: threading.ExceptHookArgs) -> None:
    _uncaught_exception(args.exc_type, args.exc_value, args.exc_traceback)


def configure_logging(feature: str | Path = "main", log_directory: Path | None = None) -> tuple[Path, Path]:
    """Start fresh normal and crash logs for this application run.

    Both files are bounded while the program runs so a long session preserves
    the newest details instead of consuming disk space indefinitely.
    """
    if isinstance(feature, Path):
        log_directory = feature
        feature = "main"
    directory = log_directory or PROJECT_ROOT / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    runtime_log = directory / f"aibrain.{feature}.log"
    crash_log = directory / f"crash.{feature}.log"
    runtime_log.unlink(missing_ok=True)
    crash_log.unlink(missing_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    formatter = AlignedFormatter(colour=sys.stderr.isatty())
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)

    runtime_handler = BoundedFileHandler(runtime_log)
    runtime_handler.setFormatter(AlignedFormatter(colour=False))
    root.addHandler(runtime_handler)

    crash_handler = CrashFileHandler(crash_log, max_bytes=MAX_CRASH_LOG_BYTES, delay=True)
    crash_handler.setLevel(logging.CRITICAL)
    crash_handler.setFormatter(AlignedFormatter(colour=False))
    root.addHandler(crash_handler)

    sys.excepthook = _uncaught_exception
    threading.excepthook = _thread_exception
    return runtime_log, crash_log
