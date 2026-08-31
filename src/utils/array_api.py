"""One numerical-array contract with transparent CUDA acceleration."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as _numpy


def _select_backend() -> tuple[ModuleType, str, str | None]:
    try:
        import cupy as _cupy

        if _cupy.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy found no CUDA devices")
        _cupy.zeros(1, dtype=_cupy.float32).sum().item()
        return _cupy, "CuPy / CUDA", None
    except Exception as exc:
        # CUDA acceleration is optional. An unavailable package, device, or
        # runtime must never surface as a startup warning or traceback.
        return _numpy, "NumPy / CPU", str(exc)


_backend, BACKEND_NAME, CUDA_FALLBACK_REASON = _select_backend()
GPU_ACCELERATED = BACKEND_NAME.startswith("CuPy")


def to_numpy(value: Any) -> _numpy.ndarray:
    """Return a host array for persistence, native DLLs, and external APIs."""
    if GPU_ACCELERATED:
        return _backend.asnumpy(value)
    return _numpy.asarray(value)


class UnifiedArrayAPI:
    """Proxy NumPy-compatible work to CuPy while keeping file I/O portable."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_backend, name, getattr(_numpy, name, None))

    @staticmethod
    def load(*args: Any, **kwargs: Any) -> Any:
        return _numpy.load(*args, **kwargs)

    @staticmethod
    def savez_compressed(file: Any, **arrays: Any) -> None:
        _numpy.savez_compressed(
            file, **{name: to_numpy(value) for name, value in arrays.items()}
        )


array_api = UnifiedArrayAPI()

__all__ = [
    "array_api",
    "BACKEND_NAME",
    "CUDA_FALLBACK_REASON",
    "GPU_ACCELERATED",
    "to_numpy",
]
