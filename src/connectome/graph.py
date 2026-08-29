from __future__ import annotations

from dataclasses import dataclass

from ..utils.array_api import array_api as np


@dataclass(slots=True)
class ConnectomeGraph:
    positions: np.ndarray
    regions: np.ndarray
    edges: np.ndarray
    region_names: tuple[str, ...]
