from __future__ import annotations

from time import monotonic

import numpy as np

from ..models.instrumented_backend import ActivationFrame
from ..native.wrapper.connectome_kernels import native
from .graph import ConnectomeGraph

ACTIVITY_DECAY_SECONDS = 0.11


class ActivityField:
    def __init__(self, graph: ConnectomeGraph) -> None:
        self.graph = graph
        self.values = np.zeros(len(graph.positions), dtype=np.float32)
        self.peaks = self.values.copy()
        self.last_time = monotonic()
        self.step = 0
        self.current_chunk = ""
        self.current_source = "Real-time"
        self.telemetry: dict[str, float] = {}
        self.channel_values: dict[str, float] = {}
        self.active_count_cached = 0
        self.disabled = np.zeros(len(graph.positions), dtype=bool)
        self.importance = np.ones(len(graph.positions), dtype=np.float32)

    def update(self, frame: ActivationFrame) -> None:
        self.decay()
        self.step = frame.step
        self.current_chunk = frame.chunk_text
        self.current_source = frame.source.value
        self.telemetry = {name: float(value) for name, value in frame.metrics.items()}
        self.channel_values = {
            name: float(np.clip(frame.regions.get(name, 0.0), 0.0, 1.0))
            for name in self.graph.region_names
        }
        self.values.fill(0.0)
        for region, name in enumerate(self.graph.region_names):
            nodes = self.graph.regions == region
            measured_value = self.channel_values[name]
            self.values[nodes] = measured_value * self.importance[nodes]
        self.peaks = np.maximum(self.peaks * 0.995, self.values)
        self.values[self.disabled] = 0.0
        self.active_count_cached = int(np.count_nonzero(self.values > 0.10))

    def decay(self) -> None:
        now = monotonic()
        delta = min(now - self.last_time, 1.0)
        self.last_time = now
        self.active_count_cached = native.decay(
            self.values, float(np.exp(-delta / ACTIVITY_DECAY_SECONDS)), 0.10
        )

    def set_disabled(self, index: int, disabled: bool) -> None:
        self.disabled[index] = disabled
        if disabled:
            self.values[index] = 0.0

    def set_importance(self, index: int, importance: float) -> None:
        self.importance[index] = np.clip(importance, 0.0, 3.0)

    @property
    def active_count(self) -> int:
        return self.active_count_cached
