from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .graph import ConnectomeGraph
from ..models.instrumented_backend import ActivationFrame


@dataclass(slots=True)
class AnalysisRecord:
    step: int
    token: str
    active_nodes: int
    mean_activity: float
    peak_activity: float
    dominant_region: str
    novelty: float


class ConnectomeAnalyzer:
    """Online neural-style encoder for visual activity, never model internals.

    A small adaptive projection learns recurring activity fingerprints from the
    visual graph. Its novelty score is useful for comparing token events, but
    is clearly derived from the visualization stream, not measured LLM state.
    """

    def __init__(self, graph: ConnectomeGraph, width: int = 24) -> None:
        self.graph = graph
        self.width = width
        self.region_count = len(graph.region_names)
        rng = np.random.default_rng(9137)
        self.encoder = rng.normal(0, .12, (self.region_count, width)).astype("f4")
        self.prototype = np.zeros(width, dtype="f4")
        self.records: list[AnalysisRecord] = []

    def observe(self, frame: ActivationFrame, values: np.ndarray) -> AnalysisRecord:
        regional = np.bincount(self.graph.regions, weights=values, minlength=self.region_count).astype("f4")
        counts = np.bincount(self.graph.regions, minlength=self.region_count).clip(1)
        features = regional / counts
        encoded = np.tanh(features @ self.encoder)
        novelty = float(np.linalg.norm(encoded - self.prototype))
        learning_rate = .035
        self.prototype += learning_rate * (encoded - self.prototype)
        self.encoder += learning_rate * np.outer(features, encoded - self.prototype)
        dominant = int(np.argmax(regional))
        record = AnalysisRecord(frame.step, frame.token_text, int(np.count_nonzero(values > .1)),
                                float(values.mean()), float(values.max()), self.graph.region_names[dominant], novelty)
        self.records.append(record)
        return record

    def summary(self) -> dict[str, object]:
        if not self.records:
            return {"frames": 0}
        novelty = np.array([record.novelty for record in self.records])
        return {"frames": len(self.records), "mean_novelty": float(novelty.mean()),
                "peak_novelty": float(novelty.max()), "most_active_region": max(self.records, key=lambda item: item.active_nodes).dominant_region}
