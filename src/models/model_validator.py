from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, Signal, Slot

from .model_info import ModelInfo
from .ollama_discovery import OllamaDiscovery


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
                validated.append(replace(model, available=False, error="Model layer blob is missing"))
                continue

            error = checked_paths.get(model.blob_path)
            if model.blob_path not in checked_paths:
                error = OllamaDiscovery._validate_gguf(model.blob_path, model.size_bytes)
                if error is None and backend is not None:
                    report(index, total, f"Loading {model.name}:{model.tag} with llama.cpp")
                    try:
                        backend.load(
                            model.blob_path,
                            GenerationConfig(context_length=512, gpu_layers=0),
                        )
                    except Exception as exc:
                        error = f"llama.cpp compatibility check failed: {exc}"
                    finally:
                        backend.unload()
                checked_paths[model.blob_path] = error
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

            models = ModelValidator.validate(candidates, self._cancelled, self.progress.emit)
            self.finished.emit(models)
        except Exception as exc:
            self.failed.emit(f"Startup validation failed: {exc}")

    @Slot()
    def cancel(self) -> None:
        self._cancelled.set()
