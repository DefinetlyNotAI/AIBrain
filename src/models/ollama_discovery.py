from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Any

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
                info = self._parse_manifest(manifest, manifest_root)
                if info.available:
                    models.append(info)
                else:
                    LOG.warning(
                        "Skipping unavailable GGUF %s: %s", info.name, info.error
                    )
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
        expected_size = 0
        expected_digest: str | None = None
        validation_error = "Model layer blob is missing"
        for layer in layers:
            digest = str(layer.get("digest", ""))
            media_type = str(layer.get("mediaType", ""))
            candidate = self.root / "blobs" / digest.replace(":", "-")
            if candidate.exists() and ("model" in media_type or blob is None):
                blob = candidate
                expected_size = int(layer.get("size", 0) or 0)
                expected_digest = digest or None
                validation_error = self._validate_gguf(candidate, expected_size)
                if validation_error is None:
                    break
        size = blob.stat().st_size if blob else 0
        # Docker-style Ollama manifests usually contain no descriptive metadata.
        # Preserve available data, otherwise use a clearly inferred family label.
        inferred_family = name.rsplit("/", 1)[-1]
        family = str(
            config.get("family")
            or config.get("model_family")
            or data.get("model")
            or inferred_family
        )
        details = config.get("details") or {}
        return ModelInfo(
            name=name,
            tag=tag,
            blob_path=blob,
            family=family,
            parameter_size=str(
                details.get("parameter_size")
                or config.get("parameter_size")
                or self._size_from_tag(tag)
            ),
            quantization=str(
                details.get("quantization_level")
                or config.get("quantization")
                or "GGUF (manifest does not specify quantization)"
            ),
            size_bytes=size,
            digest=expected_digest,
            available=validation_error is None,
            error=validation_error,
        )

    @staticmethod
    def _size_from_tag(tag: str) -> str:
        for part in tag.lower().replace("-", " ").split():
            if part.endswith("b") and part[:-1].replace(".", "").isdigit():
                return part.upper()
        return "Unknown"

    @staticmethod
    def _validate_gguf(path: Path, expected_size: int = 0) -> str | None:
        try:
            if not path.is_file():
                return "Model blob is not a regular file"
            size = path.stat().st_size
            if expected_size and size != expected_size:
                return (
                    f"Blob size mismatch (expected {expected_size:,}, found {size:,})"
                )
            with path.open("rb") as handle:
                header = handle.read(24)
            if len(header) != 24 or header[:4] != b"GGUF":
                return "Blob does not have a GGUF header"
            version, tensor_count, metadata_count = struct.unpack("<IQQ", header[4:])
            if version not in (2, 3):
                return f"Unsupported GGUF version {version}"
            if tensor_count < 1 or metadata_count < 1:
                return "GGUF header has no tensors or metadata"
            return None
        except (OSError, struct.error) as exc:
            return f"Cannot read GGUF header: {exc}"
