from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from .model_info import ModelInfo

LOG = logging.getLogger(__name__)


class OllamaDiscovery:
    """Reads Ollama's content-addressed store without starting Ollama."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.home() / ".ollama" / "models"

    def discover(self) -> list[ModelInfo]:
        manifest_root = self.root / "manifests"
        if not manifest_root.exists():
            return []
        models: list[ModelInfo] = []
        for manifest in manifest_root.rglob("*"):
            if not manifest.is_file():
                continue
            try:
                models.append(self._parse_manifest(manifest, manifest_root))
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                LOG.warning("Skipping invalid Ollama manifest %s: %s", manifest, exc)
        return sorted(models, key=lambda item: (item.name, item.tag))

    def _parse_manifest(self, path: Path, manifest_root: Path) -> ModelInfo:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        relative = path.relative_to(manifest_root).parts
        name_parts = list(relative[:-1])
        if name_parts and name_parts[0].startswith("registry."):
            name_parts.pop(0)
        if name_parts and name_parts[0] == "library":
            name_parts.pop(0)
        name = "/".join(name_parts) or data.get("name", "unknown")
        tag = relative[-1] if relative else "latest"
        config = data.get("config") or {}
        layers = data.get("layers") or []
        blob: Path | None = None
        for layer in layers:
            digest = str(layer.get("digest", ""))
            media_type = str(layer.get("mediaType", ""))
            candidate = self.root / "blobs" / digest.replace(":", "-")
            if candidate.exists() and ("model" in media_type or blob is None):
                blob = candidate
                if self._is_gguf(candidate):
                    break
        size = blob.stat().st_size if blob else 0
        # Docker-style Ollama manifests usually contain no descriptive metadata.
        # Preserve available data, otherwise use a clearly inferred family label.
        inferred_family = name.rsplit("/", 1)[-1]
        family = str(config.get("family") or config.get("model_family") or data.get("model") or inferred_family)
        details = config.get("details") or {}
        return ModelInfo(
            name=name, tag=tag, blob_path=blob, family=family,
            parameter_size=str(details.get("parameter_size") or config.get("parameter_size") or self._size_from_tag(tag)),
            quantization=str(details.get("quantization_level") or config.get("quantization") or "GGUF (manifest does not specify quantization)"),
            size_bytes=size, available=bool(blob and self._is_gguf(blob)),
            error=None if blob and self._is_gguf(blob) else "No usable GGUF blob found",
        )

    @staticmethod
    def _size_from_tag(tag: str) -> str:
        for part in tag.lower().replace("-", " ").split():
            if part.endswith("b") and part[:-1].replace(".", "").isdigit():
                return part.upper()
        return "Unknown"

    @staticmethod
    def _is_gguf(path: Path) -> bool:
        try:
            return path.open("rb").read(4) == b"GGUF"
        except OSError:
            return False
