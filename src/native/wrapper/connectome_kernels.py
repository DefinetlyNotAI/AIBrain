from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np


_APP_ROOT = (
    Path(sys.argv[0]).resolve().parent
    if "__compiled__" in globals()
    else Path(__file__).resolve().parents[3]
)

_DLL_PATH = _APP_ROOT / "dll" / "aibrain.connectome.dll"

_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
_INT32_PTR = ctypes.POINTER(ctypes.c_int32)
_INT16_PTR = ctypes.POINTER(ctypes.c_int16)


class NativeConnectome:
    def __init__(self) -> None:
        self.dll: ctypes.WinDLL | None = None

        if not _DLL_PATH.exists():
            return

        try:
            dll = ctypes.WinDLL(str(_DLL_PATH))

            dll.decay_and_count.argtypes = [
                _FLOAT_PTR,
                ctypes.c_size_t,
                ctypes.c_float,
                ctypes.c_float,
            ]
            dll.decay_and_count.restype = ctypes.c_size_t

            dll.edge_activity.argtypes = [
                _FLOAT_PTR,
                ctypes.c_size_t,
                _INT32_PTR,
                ctypes.c_size_t,
                _FLOAT_PTR,
            ]
            dll.edge_activity.restype = None

            dll.region_activity.argtypes = [
                _FLOAT_PTR,
                _INT16_PTR,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_float,
                _FLOAT_PTR,
            ]
            dll.region_activity.restype = ctypes.c_size_t

            self.dll = dll

        except (AttributeError, OSError):
            self.dll = None

    @property
    def available(self) -> bool:
        return self.dll is not None

    def decay(
            self,
            values: np.ndarray,
            factor: float,
            threshold: float,
    ) -> int:
        self._validate_float32(values, "values")

        if self.dll is None:
            values *= factor
            return int(np.count_nonzero(values > threshold))

        return int(
            self.dll.decay_and_count(
                values.ctypes.data_as(_FLOAT_PTR),
                values.size,
                factor,
                threshold,
            )
        )

    def edges(
            self,
            values: np.ndarray,
            edges: np.ndarray,
            output: np.ndarray,
    ) -> None:
        self._validate_float32(values, "values")
        self._validate_int32(edges, "edges")
        self._validate_float32(output, "output")

        if values.ndim != 1:
            raise ValueError(f"values must be 1D, got shape {values.shape}")

        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"edges must have shape (N, 2), got {edges.shape}")

        required_output_size = edges.shape[0] * 2

        if output.size < required_output_size:
            raise ValueError(
                f"output requires at least {required_output_size} elements, "
                f"got {output.size}"
            )

        if self.dll is None:
            source = edges[:, 0]
            destination = edges[:, 1]

            valid = (
                    (source >= 0)
                    & (destination >= 0)
                    & (source < values.size)
                    & (destination < values.size)
            )

            activity = np.zeros(edges.shape[0], dtype=np.float32)

            activity[valid] = np.maximum(
                values[source[valid]],
                values[destination[valid]],
            )

            output[:required_output_size] = np.repeat(activity, 2)
            return

        self.dll.edge_activity(
            values.ctypes.data_as(_FLOAT_PTR),
            values.size,
            edges.ctypes.data_as(_INT32_PTR),
            edges.shape[0],
            output.ctypes.data_as(_FLOAT_PTR),
        )

    def regions(
            self,
            values: np.ndarray,
            region_ids: np.ndarray,
            region_count: int,
            threshold: float = 0.1,
    ) -> tuple[np.ndarray, int]:
        self._validate_float32(values, "values")
        self._validate_int16(region_ids, "region_ids")

        if values.ndim != 1:
            raise ValueError(f"values must be 1D, got shape {values.shape}")

        if region_ids.ndim != 1:
            raise ValueError(
                f"region_ids must be 1D, got shape {region_ids.shape}"
            )

        if values.size != region_ids.size:
            raise ValueError(
                "values and region_ids must contain the same number of elements"
            )

        if region_count < 0:
            raise ValueError("region_count must be non-negative")

        sums = np.zeros(region_count, dtype=np.float32)

        if self.dll is None:
            valid = (
                    (region_ids >= 0)
                    & (region_ids < region_count)
            )

            if np.any(valid):
                sums[:] = np.bincount(
                    region_ids[valid],
                    weights=values[valid],
                    minlength=region_count,
                ).astype(np.float32, copy=False)

            return sums, int(np.count_nonzero(values > threshold))

        active = self.dll.region_activity(
            values.ctypes.data_as(_FLOAT_PTR),
            region_ids.ctypes.data_as(_INT16_PTR),
            values.size,
            region_count,
            threshold,
            sums.ctypes.data_as(_FLOAT_PTR),
        )

        return sums, int(active)

    @staticmethod
    def _validate_float32(array: np.ndarray, name: str) -> None:
        if array.dtype != np.float32:
            raise TypeError(
                f"{name} must use float32, got {array.dtype.name}"
            )

        if not array.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    @staticmethod
    def _validate_int32(array: np.ndarray, name: str) -> None:
        if array.dtype != np.int32:
            raise TypeError(
                f"{name} must use int32, got {array.dtype.name}"
            )

        if not array.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    @staticmethod
    def _validate_int16(array: np.ndarray, name: str) -> None:
        if array.dtype != np.int16:
            raise TypeError(
                f"{name} must use int16, got {array.dtype.name}"
            )

        if not array.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")


native = NativeConnectome()
