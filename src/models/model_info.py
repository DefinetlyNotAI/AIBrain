from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelInfo:
    name: str
    tag: str
    blob_path: Path | None
    family: str = "Unknown"
    parameter_size: str = "Unknown"
    quantization: str = "Unknown"
    size_bytes: int = 0
    available: bool = False
    error: str | None = None

    @property
    def label(self) -> str:
        size = f"{self.size_bytes / 1024 ** 3:.1f} GB" if self.size_bytes else "unavailable"
        return f"{self.name}:{self.tag} — {self.family} · {self.quantization} · {size}"
