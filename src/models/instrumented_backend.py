from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Mapping


class ActivitySource(str, Enum):
    REAL = "Real"
    DERIVED = "Derived"
    SIMULATION = "Simulation"


@dataclass(frozen=True, slots=True)
class ActivationFrame:
    token_id: int | None
    token_text: str
    step: int
    source: ActivitySource
    timestamp: float = field(default_factory=monotonic)
    regions: Mapping[str, float] = field(default_factory=dict)
    layers: Mapping[int, float] = field(default_factory=dict)
    logits_entropy: float | None = None


class InstrumentedBackend:
    """Extension point for a future Transformers backend with real measurements."""

    available = False
