from __future__ import annotations

from time import monotonic

import numpy as np

from .graph import ConnectomeGraph
from ..models.instrumented_backend import ActivationFrame


class ActivityField:
    def __init__(self, graph: ConnectomeGraph) -> None:
        self.graph = graph
        self.values = np.zeros(len(graph.positions), dtype=np.float32)
        self.peaks = self.values.copy()
        self.last_time = monotonic()
        self.step = 0
        self.current_token = ""

    def update(self, frame: ActivationFrame) -> None:
        self.decay()
        self.step = frame.step
        self.current_token = frame.token_text
        # Stable token/step mapping: procedural, but tied to observable inference events.
        token_key = (sum(ord(c) * (index + 1) for index, c in enumerate(frame.token_text)) + frame.step * 7919)
        rng = np.random.default_rng(token_key & 0xFFFFFFFF)
        primary = frame.step % len(self.graph.region_names)
        cascade = [(primary + offset) % len(self.graph.region_names) for offset in range(4)]
        for depth, region in enumerate(cascade):
            candidates = np.flatnonzero(self.graph.regions == region)
            if len(candidates):
                selected = rng.choice(candidates, size=min(len(candidates), 40 + depth * 30), replace=False)
                amount = 1.0 / (1 + depth * .55)
                self.values[selected] = np.maximum(self.values[selected], amount)
        self.peaks = np.maximum(self.peaks * .995, self.values)

    def decay(self) -> None:
        now = monotonic()
        delta = min(now - self.last_time, 1.0)
        self.last_time = now
        self.values *= np.exp(-delta / .30).astype(np.float32) if isinstance(delta, np.ndarray) else float(np.exp(-delta / .30))

    @property
    def active_count(self) -> int:
        return int(np.count_nonzero(self.values > .10))
