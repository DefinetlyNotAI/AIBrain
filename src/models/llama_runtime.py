"""Load llama.cpp with the CUDA DLLs installed in this Python environment."""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path
from threading import Lock
from types import ModuleType
from typing import Any

_DLL_HANDLES: dict[Path, Any] = {}
_DLL_LOCK = Lock()


def cuda_dll_directories() -> list[Path]:
    """Find NVIDIA's CUDA 11/12 and CUDA 13 wheel layouts without importing them."""
    package_roots = {
        Path(sysconfig.get_path("purelib")),
        Path(sysconfig.get_path("platlib")),
        Path(sys.executable).parent,
    }
    directories: set[Path] = set()
    for root in package_roots:
        nvidia = root / "nvidia"
        for pattern in ("*/bin", "*/bin/x86_64", "*/bin/x64"):
            for directory in nvidia.glob(pattern):
                if directory.is_dir() and any(directory.glob("*.dll")):
                    directories.add(directory.resolve())
    return sorted(directories)


def prepare_cuda_dll_search() -> list[Path]:
    """Keep NVIDIA DLL directories available for modern and legacy loaders."""
    if sys.platform != "win32":
        return []
    directories = cuda_dll_directories()
    with _DLL_LOCK:
        for directory in directories:
            if directory not in _DLL_HANDLES:
                # Retain these handles: closing one removes its search directory.
                _DLL_HANDLES[directory] = os.add_dll_directory(str(directory))

        # llama-cpp-python uses ctypes.CDLL(..., winmode=0), which also needs the
        # process PATH. Do not change the user's persistent PATH or CUDA_PATH.
        existing = os.environ.get("PATH", "").split(os.pathsep)
        known = {os.path.normcase(os.path.normpath(path)) for path in existing if path}
        additions = [
            str(directory) for directory in directories
            if os.path.normcase(os.path.normpath(str(directory))) not in known
        ]
        if additions:
            os.environ["PATH"] = os.pathsep.join([*additions, *existing])
    return directories


def load_llama_cpp() -> ModuleType:
    """Use the same native loader in the installer and application."""
    prepare_cuda_dll_search()
    import llama_cpp

    return llama_cpp
