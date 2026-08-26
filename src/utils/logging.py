from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path


class OneLineFormatter(logging.Formatter):
    """Compact structured logs that never wrap or inject terminal control text."""

    _COLOURS = {logging.DEBUG: "\x1b[38;5;245m", logging.INFO: "\x1b[38;5;45m", logging.WARNING: "\x1b[38;5;220m",
                logging.ERROR: "\x1b[38;5;203m", logging.CRITICAL: "\x1b[1;38;5;196m"}
    _RESET = "\x1b[0m"

    def __init__(self, *, colour: bool, terminal_width: bool) -> None:
        super().__init__()
        self.colour = colour
        self.terminal_width = terminal_width

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        message = " ".join(record.getMessage().split())
        line = f"{timestamp} | {record.levelname:<8} | {record.name:<24.24} | {message}"
        if self.terminal_width:
            width = max(40, shutil.get_terminal_size(fallback=(120, 24)).columns)
            if len(line) > width:
                line = line[: max(1, width - 1)] + "…"
        if self.colour:
            line = f"{self._COLOURS.get(record.levelno, '')}{line}{self._RESET}"
        return line


def configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    stream = logging.StreamHandler()
    stream.setFormatter(OneLineFormatter(colour=sys.stderr.isatty(), terminal_width=True))
    root.addHandler(stream)
    try:
        log_directory = Path.home() / ".aibrain" / "logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_directory / "aibrain.log", encoding="utf-8")
        file_handler.setFormatter(OneLineFormatter(colour=False, terminal_width=False))
        root.addHandler(file_handler)
    except OSError as exc:
        root.warning("File logging disabled: %s", exc)
