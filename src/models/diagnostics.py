"""Non-destructive diagnostics and explicit repair actions for Ollama models."""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from .model_info import ModelInfo
from .model_validator import ModelValidator
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
    def issue_code(self) -> str:
        """Return a stable, human-readable problem category."""
        if self.available:
            return "Healthy"
        lowered = self.detail.lower()
        if "manifest" in lowered or "json" in lowered:
            return "Invalid manifest"
        if "size mismatch" in lowered:
            return "Incomplete download"
        if "hash" in lowered or "digest" in lowered:
            return "Invalid hash"
        if "missing" in lowered or "not a regular file" in lowered:
            return "Missing model data"
        if "vision-capable" in lowered or "text-only gguf" in lowered:
            return "Unsupported model type"
        if "llama.cpp" in lowered or "backend" in lowered:
            return "Backend incompatibility"
        if "gguf" in lowered or "header" in lowered:
            return "Invalid GGUF"
        if "permission" in lowered or "access" in lowered:
            return "File access blocked"
        return "Broken install"

    @property
    def can_remove_manifest(self) -> bool:
        return (
            not self.available
            and self.issue_code
            not in {"Backend incompatibility", "Unsupported model type"}
            and self.manifest_path.is_file()
        )

    @property
    def can_repair(self) -> bool:
        """Return whether an automated data or runtime repair can help."""
        return not self.available and self.issue_code != "Unsupported model type"

    @property
    def reason(self) -> str:
        """A mandatory, user-readable cause for failed diagnostics."""
        return "Healthy model" if self.available else self.issue_code


class OllamaDiagnostics:
    """Inspect locally installed manifests without changing any Ollama data."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.home() / ".ollama" / "models"
        self._discovery = OllamaDiscovery(self.root)

    def inspect(
        self,
        *,
        verify_backend: bool = False,
        cancelled: Event | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> list[ModelDiagnostic]:
        manifest_root = self.root / "manifests"
        if not manifest_root.is_dir():
            return []

        parsed: list[tuple[str, Path, ModelInfo]] = []
        diagnostics: list[ModelDiagnostic] = []
        for manifest in sorted(
            path for path in manifest_root.rglob("*") if path.is_file()
        ):
            reference = self._reference(manifest, manifest_root)
            try:
                model = self._discovery._parse_manifest(manifest, manifest_root)
                parsed.append((reference, manifest, model))
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                diagnostics.append(
                    ModelDiagnostic(
                        reference, manifest, None, False, f"Invalid manifest: {exc}"
                    )
                )
        models = [model for _reference, _manifest, model in parsed]
        if verify_backend and models:
            checked = ModelValidator.validate(
                models,
                cancelled or Event(),
                progress or (lambda _current, _total, _detail: None),
            )
        else:
            checked = models
        diagnostics.extend(
            self._from_model(reference, manifest, model)
            for (reference, manifest, _original), model in zip(parsed, checked)
        )
        return sorted(diagnostics, key=lambda item: item.reference)

    def subsystem_health(self) -> dict[str, tuple[str, str]]:
        """Cheap, non-mutating health cards for the repair dashboard."""
        project_root = Path(__file__).resolve().parents[2]
        cache = project_root / ".cache"
        model_state = "Ready" if (self.root / "manifests").is_dir() else "Needs setup"
        return {
            "Models": (
                model_state,
                "Ollama manifests and GGUF blobs are inspected separately below.",
            ),
            "Python environment": (
                "Ready",
                "The diagnostics process started in the managed runtime.",
            ),
            "pip / libraries": (
                "Ready",
                "Installed packages are verified when a backend model check runs.",
            ),
            "Native DLLs": (
                "Ready" if any(project_root.rglob("*.dll")) else "Info",
                "Native acceleration is optional; missing DLLs use the supported Python fallback.",
            ),
            ".cache": (
                "Ready" if cache.exists() else "Info",
                "Refresh never deletes cache entries or model blobs.",
            ),
        }

    @staticmethod
    def remove_stale_manifest(diagnostic: ModelDiagnostic) -> None:
        """Remove only the selected broken manifest, never its shared blob data."""
        if not diagnostic.can_remove_manifest:
            raise ValueError("Only an existing invalid manifest can be removed")
        diagnostic.manifest_path.unlink()

    @staticmethod
    def quarantine_for_redownload(diagnostic: ModelDiagnostic) -> Path | None:
        """Move the exact corrupt artifact aside so ``ollama pull`` must replace it."""
        if diagnostic.available or diagnostic.issue_code in {
            "Backend incompatibility",
            "Unsupported model type",
        }:
            return None
        source = (
            diagnostic.blob_path
            if diagnostic.blob_path and diagnostic.blob_path.is_file()
            else diagnostic.manifest_path
        )
        if not source.is_file():
            return None
        local_app_data = Path(
            os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        )
        quarantine = local_app_data / "AIBrain" / "repair-quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        target = quarantine / f"{stamp}-{source.name}"
        source.replace(target)
        return target

    @staticmethod
    def invalidate_validation_cache(project_root: Path) -> Path:
        """Rebuild only disposable validation data, never models or native DLLs."""
        root = project_root.resolve()
        target = root / ".cache" / "validation"
        if not target.resolve().is_relative_to(root):
            raise ValueError("Validation cache target is outside the project")
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _from_model(
        reference: str, manifest: Path, model: ModelInfo
    ) -> ModelDiagnostic:
        if model.available:
            detail = "GGUF header and model blob are healthy"
        else:
            detail = model.error or "Model is unavailable"
        return ModelDiagnostic(
            reference, manifest, model.blob_path, model.available, detail
        )

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
