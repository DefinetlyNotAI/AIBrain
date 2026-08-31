from __future__ import annotations

import json
import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, Signal, Slot

from .model_info import ModelInfo
from ..utils.logging import format_exception
from .ollama_discovery import OllamaDiscovery

LOG = logging.getLogger(__name__)

CACHE_DIRECTORY = Path(__file__).resolve().parents[2] / ".cache" / "validation"
_VALIDATION_FORMAT = 4


def _validate_digest(
    model: ModelInfo,
    cancelled: Event,
    report: Callable[[int, int, str], None],
    index: int,
    total: int,
) -> str | None:
    """Stream an Ollama SHA-256 check without blocking GUI progress updates."""
    if (
        model.blob_path is None
        or not model.digest
        or not model.digest.startswith("sha256:")
    ):
        return None
    digest = hashlib.sha256()
    size = max(1, model.blob_path.stat().st_size)
    read = 0
    label = f"Hashing {model.name}:{model.tag}"
    report(index, total, f"{label} (start)")
    with model.blob_path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            if cancelled.is_set():
                return "Validation cancelled"
            digest.update(chunk)
            read += len(chunk)
            report(
                index, total, f"{label} ({read / size:.0%})"
            )
    actual = digest.hexdigest()
    expected = model.digest.partition(":")[2].lower()
    report(index, total, f"{label} (complete)")
    return (
        None
        if actual == expected
        else f"Invalid HASH (expected {expected}, calculated {actual})"
    )


def _cache_path(model: ModelInfo) -> Path:
    name = re.sub(r"[^a-z0-9_.-]+", "_", f"{model.name}_{model.tag}".lower()).strip("_")
    return CACHE_DIRECTORY / f"aibrain.{name or 'model'}.cache"


def _signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "modified_ns": stat.st_mtime_ns}


def _cache_profile(*, verify_backend: bool) -> str:
    """Separate cheap header-only checks from full llama.cpp compatibility checks."""
    return "llama-cpp" if verify_backend else "structural"


def _cached_result(model: ModelInfo, *, verify_backend: bool) -> str | None | object:
    if model.blob_path is None:
        return _CACHE_MISS
    try:
        payload = json.loads(_cache_path(model).read_text(encoding="utf-8"))
        if (
            payload.get("format") == _VALIDATION_FORMAT
            and payload.get("profile") == _cache_profile(verify_backend=verify_backend)
            and payload.get("blob_path") == str(model.blob_path.resolve())
            and payload.get("signature") == _signature(model.blob_path)
            and (payload.get("error") is None or isinstance(payload.get("error"), str))
        ):
            return payload.get("error")
    except (OSError, ValueError, TypeError):
        pass
    return _CACHE_MISS


_CACHE_MISS = object()


def _store_result(model: ModelInfo, error: str | None, *, verify_backend: bool) -> None:
    if model.blob_path is None:
        return
    try:
        CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        _cache_path(model).write_text(
            json.dumps(
                {
                    "format": _VALIDATION_FORMAT,
                    "profile": _cache_profile(verify_backend=verify_backend),
                    "blob_path": str(model.blob_path.resolve()),
                    "signature": _signature(model.blob_path),
                    "error": error,
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        return


class ModelValidator:
    """Perform quick, non-destructive GGUF validation without loading models."""

    @staticmethod
    def validate(
        candidates: list[ModelInfo],
        cancelled: Event,
        report: Callable[[int, int, str], None],
        *,
        verify_backend: bool = True,
    ) -> list[ModelInfo]:
        """Confirm that each unique GGUF has a valid header and loads in llama.cpp."""
        validated: list[ModelInfo] = []
        checked_paths: dict[Path, str | None] = {}
        backend = None
        if verify_backend:
            from .llama_backend import GenerationConfig, LlamaBackend

            backend = LlamaBackend()
        total = len(candidates)

        for index, model in enumerate(candidates, start=1):
            if cancelled.is_set():
                break

            report(index, total, f"Checking {model.name}:{model.tag}")
            if model.blob_path is None:
                validated.append(
                    replace(model, available=False, error="Model layer blob is missing")
                )
                continue

            error = checked_paths.get(model.blob_path)
            if model.blob_path not in checked_paths:
                cached = _cached_result(model, verify_backend=verify_backend)
                if cached is _CACHE_MISS:
                    error = OllamaDiscovery._validate_gguf(
                        model.blob_path, model.size_bytes
                    )
                    if error is None:
                        error = _validate_digest(model, cancelled, report, index, total)
                    if error is None and backend is not None:
                        report(
                            index,
                            total,
                            f"Loading {model.name}:{model.tag} with llama.cpp",
                        )
                        try:
                            backend.load(
                                model.blob_path,
                                GenerationConfig(context_length=512, gpu_layers=0),
                            )
                        except Exception as exc:
                            LOG.exception(
                                "llama.cpp compatibility check failed for %s:%s",
                                model.name,
                                model.tag,
                            )
                            error = (
                                f"llama.cpp compatibility check failed: {exc}\n"
                                + format_exception(exc)
                            )
                        finally:
                            backend.unload()
                else:
                    error = cached
                checked_paths[model.blob_path] = error
                _store_result(model, error, verify_backend=verify_backend)
            validated.append(replace(model, available=error is None, error=error))

        return validated


class StartupWorker(QObject):
    """Discover and structurally validate GGUFs before the main UI is built."""

    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = Event()

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit(0, 0, "Scanning local Ollama model manifests")
            candidates = OllamaDiscovery().discover()
            if self._cancelled.is_set():
                self.finished.emit([])
                return

            if not candidates:
                self.progress.emit(1, 1, "No local GGUF models found")
                self.finished.emit([])
                return

            models = ModelValidator.validate(
                candidates, self._cancelled, self.progress.emit
            )
            self.finished.emit(models)
        except Exception as exc:
            LOG.exception("Startup model validation failed")
            self.failed.emit(f"Startup validation failed: {exc}")

    @Slot()
    def cancel(self) -> None:
        self._cancelled.set()
