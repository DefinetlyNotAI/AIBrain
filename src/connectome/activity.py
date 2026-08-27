from __future__ import annotations

from time import monotonic

import numpy as np

from .graph import ConnectomeGraph
from ..models.instrumented_backend import ActivationFrame
from ..native.wrapper.connectome_kernels import native


class ActivityField:
    def __init__(self, graph: ConnectomeGraph) -> None:
        self.graph = graph
        self.values = np.zeros(len(graph.positions), dtype=np.float32)
        self.peaks = self.values.copy()
        self.last_time = monotonic()
        self.step = 0
        self.current_token = ""
        self.active_count_cached = 0
        self.disabled = np.zeros(len(graph.positions), dtype=bool)
        self.importance = np.ones(len(graph.positions), dtype=np.float32)

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
                self.values[selected] = np.maximum(self.values[selected], amount * self.importance[selected])
        self.peaks = np.maximum(self.peaks * .995, self.values)
        self.values[self.disabled] = 0.0

    def decay(self) -> None:
        now = monotonic()
        delta = min(now - self.last_time, 1.0)
        self.last_time = now
        self.active_count_cached = native.decay(self.values, float(np.exp(-delta / .30)), .10)

    def set_disabled(self, index: int, disabled: bool) -> None:
        self.disabled[index] = disabled
        if disabled:
            self.values[index] = 0.0

    def set_importance(self, index: int, importance: float) -> None:
        self.importance[index] = np.clip(importance, 0.0, 3.0)

    @property
    def active_count(self) -> int:
        return self.active_count_cached
