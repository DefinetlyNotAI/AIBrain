from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ConnectomeGraph:
    positions: np.ndarray
    regions: np.ndarray
    edges: np.ndarray
    region_names: tuple[str, ...]
