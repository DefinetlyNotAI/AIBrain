from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_DLL_PATH = Path(__file__).parents[1] / "native" / "aibrain_connectome.dll"


class NativeConnectome:
    def __init__(self) -> None:
        self.dll = None
        if _DLL_PATH.exists():
            try:
                self.dll = ctypes.WinDLL(str(_DLL_PATH))
                self.dll.decay_and_count.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_float, ctypes.c_float]
                self.dll.decay_and_count.restype = ctypes.c_size_t
                self.dll.edge_activity.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)]
            except OSError:
                self.dll = None

    @property
    def available(self) -> bool:
        return self.dll is not None

    def decay(self, values: np.ndarray, factor: float, threshold: float) -> int:
        if not self.dll:
            values *= factor
            return int(np.count_nonzero(values > threshold))
        return int(self.dll.decay_and_count(values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(values), factor, threshold))

    def edges(self, values: np.ndarray, edges: np.ndarray, output: np.ndarray) -> None:
        if not self.dll:
            output[:] = np.repeat(np.maximum(values[edges[:, 0]], values[edges[:, 1]]), 2)
            return
        self.dll.edge_activity(values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), edges.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(edges), output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))


native = NativeConnectome()
