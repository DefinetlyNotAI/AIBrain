from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RenderAdapter:
    identifier: str
    name: str
    driver: str = "Unknown driver"


LOG = logging.getLogger(__name__)


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


def set_windows_gpu_preference(high_performance: bool) -> bool:
    """Request Windows' high-performance adapter for this Python executable.

    The choice is applied by Windows when the process is created, so it takes
    effect on the next AIBrain launch. Qt still reports the actual GL adapter.
    """
    if sys.platform != "win32":
        return False
    try:
        import winreg

        key_path = r"Software\Microsoft\DirectX\UserGpuPreferences"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            executables = (sys.executable, str(Path(sys.executable).with_name("pythonw.exe")))
            for executable in executables:
                if high_performance:
                    winreg.SetValueEx(key, executable, 0, winreg.REG_SZ, "GpuPreference=2;")
                    LOG.info("Requested Windows high-performance GPU for %s", executable)
                else:
                    try:
                        winreg.DeleteValue(key, executable)
                    except FileNotFoundError:
                        pass
        return True
    except OSError as exc:
        LOG.warning("Could not set Windows GPU preference: %s", exc)
        return False
