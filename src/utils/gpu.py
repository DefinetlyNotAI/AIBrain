from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RenderAdapter:
    identifier: str
    name: str
    driver: str = "Unknown driver"


def discover_render_adapters() -> list[RenderAdapter]:
    """Discover Windows display adapters; Qt/OpenGL makes the final device choice."""
    command = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
        raw = json.loads(result.stdout)
        entries = raw if isinstance(raw, list) else [raw]
        adapters = [RenderAdapter(str(index), str(item.get("Name") or "Unknown adapter"), str(item.get("DriverVersion") or "Unknown driver")) for index, item in enumerate(entries)]
        if adapters:
            return adapters
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, TypeError):
        pass
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=5)
        return [RenderAdapter(str(index), *[part.strip() for part in line.split(",", maxsplit=1)]) for index, line in enumerate(result.stdout.splitlines()) if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []
