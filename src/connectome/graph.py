from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class ConnectomeGraph:
    positions: NDArray[np.float32]
    regions: NDArray[np.int16]
    edges: NDArray[np.int32]
    region_names: tuple[str, ...]
