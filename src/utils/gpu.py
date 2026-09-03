from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RenderAdapter:
    identifier: str
    name: str
    driver: str = "Unknown driver"


LOG = logging.getLogger(__name__)
GPU_RELAUNCH_EXIT_CODE = 75
_GPU_RELAUNCH_ATTEMPT_ENV = "AIBRAIN_GPU_RELAUNCH_ATTEMPT"
_MAX_GPU_RELAUNCH_ATTEMPTS = 1
_GPU_SUPERVISOR_ENV = "AIBRAIN_GPU_SUPERVISOR"


def supervise_gpu_launch(command: list[str]) -> int:
    """Keep the console attached to a fresh GPU-configured application process."""
    from .console_ui import report_keyboard_interrupt

    environment = os.environ.copy()
    environment[_GPU_SUPERVISOR_ENV] = "1"
    for attempt in range(_MAX_GPU_RELAUNCH_ATTEMPTS + 1):
        child_environment = environment.copy()
        child_environment[_GPU_RELAUNCH_ATTEMPT_ENV] = str(attempt)
        child = subprocess.Popen(command, env=child_environment, close_fds=False)
        try:
            exit_code = child.wait()
        except KeyboardInterrupt:
            try:
                child.terminate()
            except OSError:
                pass
            else:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            report_keyboard_interrupt("AIBrain")
            return 130
        if exit_code != GPU_RELAUNCH_EXIT_CODE:
            return exit_code
    return GPU_RELAUNCH_EXIT_CODE


def prepare_gpu_launch(*, compiled: bool = False) -> int | None:
    """Apply the saved adapter preference before any Qt application is created.

    A numeric result belongs to the supervised child and must be returned by the
    launcher. None means this process can continue creating its Qt application.
    """
    os.environ.setdefault("QT_OPENGL", "desktop")
    if sys.platform != "win32" or not should_prefer_high_performance_gpu():
        return None
    set_windows_gpu_preference(True)
    if not os.environ.get(_GPU_SUPERVISOR_ENV):
        arguments = sys.argv[1:] if compiled else sys.argv
        return supervise_gpu_launch([sys.executable, *arguments])
    return None


def configure_opengl_surface() -> None:
    """Use the same desktop OpenGL context in every desktop launcher."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtWidgets import QApplication

    surface = QSurfaceFormat()
    surface.setVersion(3, 3)
    surface.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface.setDepthBufferSize(24)
    surface.setSamples(0)
    QSurfaceFormat.setDefaultFormat(surface)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL, True)


def _gpu_host_executables() -> set[str]:
    """Include the real Python process image behind Windows venv redirectors."""
    executable = Path(sys.executable).resolve()
    hosts = {executable}
    if executable.name.lower() in {"python.exe", "pythonw.exe"}:
        base = getattr(sys, "_base_executable", None)
        if base:
            hosts.add(Path(base).resolve())
        hosts.update(host.with_name("pythonw.exe") for host in list(hosts))
    launched_program = Path(sys.argv[0])
    if launched_program.suffix.lower() == ".exe":
        hosts.add(launched_program.resolve())
    return {str(host) for host in hosts}


def gpu_relaunch_attempt() -> int:
    """Return the validated number of GPU-process retries already performed."""
    try:
        return max(0, int(os.environ.get(_GPU_RELAUNCH_ATTEMPT_ENV, "0")))
    except ValueError:
        return 0


def can_request_gpu_relaunch() -> bool:
    """Allow one renderer-triggered restart after a wrong adapter is detected."""
    return sys.platform == "win32" and gpu_relaunch_attempt() < _MAX_GPU_RELAUNCH_ATTEMPTS


def discover_render_adapters() -> list[RenderAdapter]:
    """Discover Windows display adapters; Qt/OpenGL makes the final device choice."""
    command = ["powershell", "-NoProfile", "-Command",
               "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
        raw = json.loads(result.stdout)
        entries = raw if isinstance(raw, list) else [raw]
        adapters = [RenderAdapter(str(index), str(item.get("Name") or "Unknown adapter"),
                                  str(item.get("DriverVersion") or "Unknown driver")) for index, item in
                    enumerate(entries)]
        if adapters:
            return adapters
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, TypeError):
        pass
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True, timeout=5)
        return [RenderAdapter(str(index), *[part.strip() for part in line.split(",", maxsplit=1)]) for index, line in
                enumerate(result.stdout.splitlines()) if line.strip()]
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
            # The venv's python.exe redirects to the base Python process image;
            # Windows chooses the graphics adapter for that image at creation.
            for executable in _gpu_host_executables():
                if high_performance:
                    desired = "GpuPreference=2;"
                    try:
                        persisted, _ = winreg.QueryValueEx(key, executable)
                    except FileNotFoundError:
                        persisted = None
                    if persisted == desired:
                        continue
                    winreg.SetValueEx(
                        key, executable,
                        0, winreg.REG_SZ, desired
                    )
                    persisted, _ = winreg.QueryValueEx(key, executable)
                    if persisted != desired:
                        raise OSError(f"Windows did not persist the high-performance preference for {executable}")
                    LOG.info("Windows high-performance GPU preference persisted for %s", executable)
                else:
                    try:
                        winreg.DeleteValue(key, executable)
                    except FileNotFoundError:
                        pass
        return True
    except OSError as exc:
        LOG.warning("Could not set Windows GPU preference: %s", exc)
        return False


def set_windows_executable_gpu_preference(executable: Path, high_performance: bool = True) -> bool:
    """Persist Windows' GPU preference for a standalone AIBrain executable."""
    if sys.platform != "win32":
        return False
    try:
        import winreg

        key_path = r"Software\Microsoft\DirectX\UserGpuPreferences"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            winreg.SetValueEx(key, str(executable.resolve()), 0, winreg.REG_SZ,
                              "GpuPreference=2;" if high_performance else "GpuPreference=0;")
        LOG.info("Requested Windows %s GPU for packaged executable %s",
                 "high-performance" if high_performance else "system-default", executable)
        return True
    except OSError as exc:
        LOG.warning("Could not set packaged executable GPU preference: %s", exc)
        return False


def should_prefer_high_performance_gpu() -> bool:
    """Return the saved rendering preference without importing Qt at startup."""
    if sys.platform != "win32":
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\AIBrain\AIBrain") as key:
            value, _ = winreg.QueryValueEx(key, "render_adapter")
            return str(value) != "system"
    except (FileNotFoundError, OSError):
        # AIBrain defaults to the discrete/high-performance renderer.
        return True
