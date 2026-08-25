from __future__ import annotations

from threading import Event

from PySide6.QtCore import QObject, Signal, Slot

from .llama_backend import GenerationConfig, LlamaBackend
from .model_info import ModelInfo


class ModelValidator(QObject):
    """Verifies every discovered GGUF against the installed backend off the UI thread."""

    progress = Signal(str)
    finished = Signal(object)

    @Slot(object)
    def validate(self, candidates: list[ModelInfo]) -> None:
        backend = LlamaBackend()
        usable: list[ModelInfo] = []
        for model in candidates:
            if getattr(self, "_cancelled", Event()).is_set():
                break
            if not model.blob_path:
                continue
            self.progress.emit(f"Checking {model.name}:{model.tag}…")
            try:
                # CPU and a small context make this a compatibility check, not a
                # GPU-memory benchmark. We still load the real GGUF parser/model.
                backend.load(model.blob_path, GenerationConfig(context_length=512, gpu_layers=0))
                usable.append(model)
            except Exception as exc:
                self.progress.emit(f"Skipping {model.name}:{model.tag} — {exc}")
            finally:
                backend.unload()
        self.finished.emit(usable)

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = Event()

    @Slot()
    def cancel(self) -> None:
        self._cancelled.set()
