"""One numerical-array contract with transparent CUDA acceleration."""

from __future__ import annotations

from types import ModuleType

import numpy as _numpy


def _select_backend() -> tuple[ModuleType, str, str | None]:
    try:
        import cupy as _cupy

        if _cupy.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy found no CUDA devices")
        _cupy.zeros(1, dtype=_cupy.float32).sum().item()
        return _cupy, "CuPy / CUDA", None
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError) as exc:
        # CUDA acceleration is optional. An unavailable package, device, or
        # runtime must never surface as a startup warning or traceback.
        return _numpy, "NumPy / CPU", str(exc)


_backend, BACKEND_NAME, CUDA_FALLBACK_REASON = _select_backend()
GPU_ACCELERATED = BACKEND_NAME.startswith("CuPy")


def to_numpy(value: object) -> _numpy.ndarray:
    """Return a host array for persistence, native DLLs, and external APIs."""
    if GPU_ACCELERATED:
        return _backend.asnumpy(value)
    return _numpy.asarray(value)


array_api = _backend

__all__ = [
    "BACKEND_NAME",
    "CUDA_FALLBACK_REASON",
    "GPU_ACCELERATED",
    "array_api",
    "to_numpy",
]
