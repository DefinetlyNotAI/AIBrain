from __future__ import annotations

import logging
from pathlib import Path


def configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)
    try:
        logs = Path.home() / ".aibrain" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logs / "aibrain.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        logging.getLogger(__name__).warning("Could not create file logging", exc_info=True)
