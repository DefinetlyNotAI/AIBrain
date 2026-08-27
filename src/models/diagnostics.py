"""Non-destructive diagnostics and explicit repair actions for Ollama models."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .model_info import ModelInfo
from .ollama_discovery import OllamaDiscovery


@dataclass(frozen=True, slots=True)
class ModelDiagnostic:
    """One manifest and its model-blob validation outcome."""

    reference: str
    manifest_path: Path
    blob_path: Path | None
    available: bool
    detail: str

    @property
    def can_remove_manifest(self) -> bool:
        return not self.available and self.manifest_path.is_file()


class OllamaDiagnostics:
    """Inspect locally installed manifests without changing any Ollama data."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.home() / ".ollama" / "models"
        self._discovery = OllamaDiscovery(self.root)

    def inspect(self) -> list[ModelDiagnostic]:
        manifest_root = self.root / "manifests"
        if not manifest_root.is_dir():
            return []

        diagnostics: list[ModelDiagnostic] = []
        for manifest in sorted(path for path in manifest_root.rglob("*") if path.is_file()):
            reference = self._reference(manifest, manifest_root)
            try:
                model = self._discovery._parse_manifest(manifest, manifest_root)
                diagnostics.append(self._from_model(reference, manifest, model))
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                diagnostics.append(
                    ModelDiagnostic(reference, manifest, None, False, f"Invalid manifest: {exc}")
                )
        return diagnostics

    @staticmethod
    def remove_stale_manifest(diagnostic: ModelDiagnostic) -> None:
        """Remove only the selected broken manifest, never its shared blob data."""
        if not diagnostic.can_remove_manifest:
            raise ValueError("Only an existing invalid manifest can be removed")
        diagnostic.manifest_path.unlink()

    @staticmethod
    def _from_model(reference: str, manifest: Path, model: ModelInfo) -> ModelDiagnostic:
        if model.available:
            detail = "GGUF header and model blob are healthy"
        else:
            detail = model.error or "Model is unavailable"
        return ModelDiagnostic(reference, manifest, model.blob_path, model.available, detail)

    @staticmethod
    def _reference(manifest: Path, manifest_root: Path) -> str:
        parts = list(manifest.relative_to(manifest_root).parts)
        if parts and parts[0].startswith("registry."):
            parts.pop(0)
        if parts and parts[0] == "library":
            parts.pop(0)
        if len(parts) < 2:
            return manifest.name
        return f"{'/'.join(parts[:-1])}:{parts[-1]}"
