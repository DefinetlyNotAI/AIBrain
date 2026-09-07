from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as _numpy
from numpy.typing import NDArray

from src.app.ctypes_helper import (
    CFloat,
    CInt16,
    CInt32,
    CSizeT,
    NativeLibrary,
    pointer,
)

from ...utils.array_api import GPU_ACCELERATED
from ...utils.array_api import array_api as np

_APP_ROOT = (
    Path(sys.argv[0]).resolve().parent
    if "__compiled__" in globals()
    else Path(__file__).resolve().parents[3]
)

_DLL_PATH = _APP_ROOT / "dll" / "aibrain.connectome.dll"

_FLOAT_PTR = pointer(CFloat)
_INT32_PTR = pointer(CInt32)
_INT16_PTR = pointer(CInt16)

_DecayAndCount = Callable[
    [
        Any,
        int,
        float,
        float,
    ],
    int,
]

_EdgeActivity = Callable[
    [
        Any,
        int,
        Any,
        int,
        Any,
    ],
    None,
]

_RegionActivity = Callable[
    [
        Any,
        Any,
        int,
        int,
        float,
        Any,
    ],
    int,
]


def _array_module(
        array: _numpy.ndarray,
) -> ModuleType:
    """Use NumPy fallback operations for NumPy input under a CUDA runtime."""
    return (
        _numpy
        if isinstance(array, _numpy.ndarray)
        else np
    )


class NativeConnectome:
    """Optional native CPU acceleration for connectome array operations."""

    def __init__(self) -> None:
        self._decay_and_count: _DecayAndCount | None = None
        self._edge_activity: _EdgeActivity | None = None
        self._region_activity: _RegionActivity | None = None

        if GPU_ACCELERATED or not _DLL_PATH.exists():
            return

        try:
            library = NativeLibrary(
                _DLL_PATH
            )

            self._decay_and_count = cast(
                _DecayAndCount,
                library.bind(
                    "decay_and_count",
                    argtypes=(
                        _FLOAT_PTR,
                        CSizeT,
                        CFloat,
                        CFloat,
                    ),
                    restype=CSizeT,
                ),
            )

            self._edge_activity = cast(
                _EdgeActivity,
                library.bind(
                    "edge_activity",
                    argtypes=(
                        _FLOAT_PTR,
                        CSizeT,
                        _INT32_PTR,
                        CSizeT,
                        _FLOAT_PTR,
                    ),
                    restype=None,
                ),
            )

            self._region_activity = cast(
                _RegionActivity,
                library.bind(
                    "region_activity",
                    argtypes=(
                        _FLOAT_PTR,
                        _INT16_PTR,
                        CSizeT,
                        CSizeT,
                        CFloat,
                        _FLOAT_PTR,
                    ),
                    restype=CSizeT,
                ),
            )

        except (
                AttributeError,
                OSError,
        ):
            self._decay_and_count = None
            self._edge_activity = None
            self._region_activity = None

    @property
    def available(self) -> bool:
        """Return whether every native connectome function is available."""
        return (
                self._decay_and_count is not None
                and self._edge_activity is not None
                and self._region_activity is not None
        )

    def decay(
            self,
            values: NDArray[_numpy.float32],
            factor: float,
            threshold: float,
    ) -> int:
        """Decay activity values and count entries above the threshold."""
        self._validate_float32(
            values,
            "values",
        )

        decay_and_count = self._decay_and_count

        if decay_and_count is None:
            values *= factor

            return int(
                _array_module(
                    values
                ).count_nonzero(
                    values > threshold
                )
            )

        return int(
            decay_and_count(
                values.ctypes.data_as(
                    _FLOAT_PTR
                ),
                values.size,
                factor,
                threshold,
            )
        )

    def edges(
            self,
            values: NDArray[_numpy.float32],
            edges: NDArray[_numpy.int32],
            output: NDArray[_numpy.float32],
    ) -> None:
        """Calculate duplicated activity values for every connectome edge."""
        self._validate_float32(
            values,
            "values",
        )

        self._validate_int32(
            edges,
            "edges",
        )

        self._validate_float32(
            output,
            "output",
        )

        if values.ndim != 1:
            raise ValueError(
                f"values must be 1D, got shape {values.shape}"
            )

        if (
                edges.ndim != 2
                or edges.shape[1] != 2
        ):
            raise ValueError(
                f"edges must have shape (N, 2), got {edges.shape}"
            )

        required_output_size = (
                edges.shape[0] * 2
        )

        if output.size < required_output_size:
            raise ValueError(
                f"output requires at least "
                f"{required_output_size} elements, "
                f"got {output.size}"
            )

        edge_activity = self._edge_activity

        if edge_activity is None:
            array_module = _array_module(
                values
            )

            source = edges[:, 0]
            destination = edges[:, 1]

            valid = (
                    (source >= 0)
                    & (destination >= 0)
                    & (source < values.size)
                    & (destination < values.size)
            )

            activity = array_module.zeros(
                edges.shape[0],
                dtype=array_module.float32,
            )

            activity[valid] = array_module.maximum(
                values[source[valid]],
                values[destination[valid]],
            )

            output[:required_output_size] = (
                array_module.repeat(
                    activity,
                    2,
                )
            )

            return

        edge_activity(
            values.ctypes.data_as(
                _FLOAT_PTR
            ),
            values.size,
            edges.ctypes.data_as(
                _INT32_PTR
            ),
            edges.shape[0],
            output.ctypes.data_as(
                _FLOAT_PTR
            ),
        )

    def regions(
            self,
            values: NDArray[_numpy.float32],
            region_ids: NDArray[_numpy.int16],
            region_count: int,
            threshold: float = 0.1,
    ) -> tuple[
        NDArray[_numpy.float32],
        int,
    ]:
        """Aggregate activity by region and return the active-value count."""
        self._validate_float32(
            values,
            "values",
        )

        self._validate_int16(
            region_ids,
            "region_ids",
        )

        if values.ndim != 1:
            raise ValueError(
                f"values must be 1D, got shape {values.shape}"
            )

        if region_ids.ndim != 1:
            raise ValueError(
                f"region_ids must be 1D, got shape {region_ids.shape}"
            )

        if values.size != region_ids.size:
            raise ValueError(
                "values and region_ids must contain "
                "the same number of elements"
            )

        if region_count < 0:
            raise ValueError(
                "region_count must be non-negative"
            )

        array_module = _array_module(
            values
        )

        sums = array_module.zeros(
            region_count,
            dtype=array_module.float32,
        )

        region_activity = (
            self._region_activity
        )

        if region_activity is None:
            valid = (
                    (region_ids >= 0)
                    & (region_ids < region_count)
            )

            if valid.any():
                sums[:] = array_module.bincount(
                    region_ids[valid],
                    weights=values[valid],
                    minlength=region_count,
                ).astype(
                    array_module.float32,
                    copy=False,
                )

            return (
                sums,
                int(
                    array_module.count_nonzero(
                        values > threshold
                    )
                ),
            )

        active = region_activity(
            values.ctypes.data_as(
                _FLOAT_PTR
            ),
            region_ids.ctypes.data_as(
                _INT16_PTR
            ),
            values.size,
            region_count,
            threshold,
            sums.ctypes.data_as(
                _FLOAT_PTR
            ),
        )

        return (
            sums,
            int(active),
        )

    @staticmethod
    def _validate_float32(
            array: _numpy.ndarray,
            name: str,
    ) -> None:
        """Require a C-contiguous float32 array."""
        if array.dtype != np.float32:
            raise TypeError(
                f"{name} must use float32, "
                f"got {array.dtype.name}"
            )

        if not array.flags.c_contiguous:
            raise ValueError(
                f"{name} must be C-contiguous"
            )

    @staticmethod
    def _validate_int32(
            array: _numpy.ndarray,
            name: str,
    ) -> None:
        """Require a C-contiguous int32 array."""
        if array.dtype != np.int32:
            raise TypeError(
                f"{name} must use int32, "
                f"got {array.dtype.name}"
            )

        if not array.flags.c_contiguous:
            raise ValueError(
                f"{name} must be C-contiguous"
            )

    @staticmethod
    def _validate_int16(
            array: _numpy.ndarray,
            name: str,
    ) -> None:
        """Require a C-contiguous int16 array."""
        if array.dtype != np.int16:
            raise TypeError(
                f"{name} must use int16, "
                f"got {array.dtype.name}"
            )

        if not array.flags.c_contiguous:
            raise ValueError(
                f"{name} must be C-contiguous"
            )


native = NativeConnectome()
