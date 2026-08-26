from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import ConnectomeGraph
from ..native.wrapper.connectome import native
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
    reconstruction_error: float
    coherence: float
    embedding: tuple[float, ...]
    regional_activity: tuple[float, ...]


class ConnectomeAnalyzer:
    """Online NumPy autoencoder that learns compact visual-activity patterns.

    The network consumes every recorded visual frame but retains only learned
    embeddings and useful event metrics. Its inputs are AIBrain's procedural
    connectome signals, never hidden LLM states.
    """

    def __init__(self, graph: ConnectomeGraph, hidden_width: int = 12) -> None:
        self.graph = graph
        self.region_count = len(graph.region_names)
        self.input_width = self.region_count + 4
        self.hidden_width = hidden_width
        rng = np.random.default_rng(9137)
        self.encoder_weights = rng.normal(0, .14, (self.input_width, hidden_width)).astype("f4")
        self.encoder_bias = np.zeros(hidden_width, dtype="f4")
        self.decoder_weights = rng.normal(0, .14, (hidden_width, self.input_width)).astype("f4")
        self.decoder_bias = np.zeros(self.input_width, dtype="f4")
        self.embedding_centroid = np.zeros(hidden_width, dtype="f4")
        self.records: list[AnalysisRecord] = []
        self._learning_rate = .025

    def observe(self, frame: ActivationFrame, values: np.ndarray) -> AnalysisRecord:
        regional, active_nodes = native.regions(values, self.graph.regions, self.region_count)
        counts = np.bincount(self.graph.regions, minlength=self.region_count).clip(1).astype("f4")
        regional_mean = regional / counts
        distribution = regional / max(float(regional.sum()), 1e-6)
        active_ratio = float(np.count_nonzero(values > .1) / len(values))
        features = np.concatenate((distribution, np.array((active_ratio, float(values.mean()), float(values.max()), float(values.std())), dtype="f4")))
        encoded_pre = features @ self.encoder_weights + self.encoder_bias
        embedding = np.tanh(encoded_pre)
        decoded_pre = embedding @ self.decoder_weights + self.decoder_bias
        reconstruction = 1.0 / (1.0 + np.exp(-decoded_pre))
        reconstruction_error = float(np.mean((reconstruction - features) ** 2))
        distance = float(np.linalg.norm(embedding - self.embedding_centroid))
        novelty = distance + reconstruction_error * 4.0
        coherence = float(1.0 / (1.0 + reconstruction_error * 40.0))
        self._train(features, embedding, reconstruction)
        self.embedding_centroid += .04 * (embedding - self.embedding_centroid)
        dominant = int(np.argmax(regional))
        record = AnalysisRecord(
            frame.step, frame.token_text, active_nodes, float(values.mean()), float(values.max()),
            self.graph.region_names[dominant], novelty, reconstruction_error, coherence,
            tuple(float(value) for value in embedding), tuple(float(value) for value in regional_mean),
        )
        self.records.append(record)
        return record

    def _train(self, features: np.ndarray, embedding: np.ndarray, reconstruction: np.ndarray) -> None:
        """One gradient-descent step for tanh encoder + sigmoid decoder."""
        gradient_output = 2.0 * (reconstruction - features) / self.input_width
        gradient_output *= reconstruction * (1.0 - reconstruction)
        gradient_decoder_weights = np.outer(embedding, gradient_output)
        gradient_embedding = gradient_output @ self.decoder_weights.T
        gradient_encoder_pre = gradient_embedding * (1.0 - embedding ** 2)
        self.decoder_weights -= self._learning_rate * gradient_decoder_weights
        self.decoder_bias -= self._learning_rate * gradient_output
        self.encoder_weights -= self._learning_rate * np.outer(features, gradient_encoder_pre)
        self.encoder_bias -= self._learning_rate * gradient_encoder_pre

    def summary(self) -> dict[str, object]:
        if not self.records:
            return {"frames": 0}
        novelty = np.array([record.novelty for record in self.records])
        reconstruction = np.array([record.reconstruction_error for record in self.records])
        coherence = np.array([record.coherence for record in self.records])
        return {
            "frames": len(self.records), "mean_novelty": float(novelty.mean()), "peak_novelty": float(novelty.max()),
            "mean_reconstruction_error": float(reconstruction.mean()), "mean_coherence": float(coherence.mean()),
            "most_active_region": max(self.records, key=lambda item: item.active_nodes).dominant_region,
        }

    def smart_report(self) -> dict[str, object]:
        """Return compact findings learned from all frames, rather than raw vectors."""
        if not self.records:
            return {"frames_processed": 0}
        regional = np.asarray([record.regional_activity for record in self.records], dtype="f4")
        region_means = regional.mean(axis=0)
        region_peaks = regional.max(axis=0)
        key_events = sorted(self.records, key=lambda item: item.novelty + item.reconstruction_error * 3, reverse=True)[:24]
        top_region = self.graph.region_names[int(region_means.argmax())]
        return {
            "frames_processed": len(self.records),
            "neural_network": {
                "architecture": f"{self.input_width} → {self.hidden_width} → {self.input_width} autoencoder",
                "activation": "tanh encoder / sigmoid decoder", "training": "online gradient descent for each visual frame",
                "learned_parameters": int(self.encoder_weights.size + self.encoder_bias.size + self.decoder_weights.size + self.decoder_bias.size),
            },
            "session_findings": {
                **self.summary(), "primary_pattern": f"Highest mean visual activity was in {top_region}.",
                "pattern_segments": self._segments(),
            },
            "regional_profile": [
                {"region": name, "mean_activity": float(region_means[index]), "peak_activity": float(region_peaks[index])}
                for index, name in enumerate(self.graph.region_names)
            ],
            "key_events": [
                {"step": record.step, "token": record.token, "dominant_region": record.dominant_region,
                 "novelty": record.novelty, "reconstruction_error": record.reconstruction_error,
                 "coherence": record.coherence, "embedding": [round(value, 5) for value in record.embedding]}
                for record in sorted(key_events, key=lambda item: item.step)
            ],
        }

    def _segments(self) -> list[dict[str, object]]:
        groups: list[dict[str, object]] = []
        for record in self.records:
            if not groups or groups[-1]["dominant_region"] != record.dominant_region:
                groups.append({"start_step": record.step, "end_step": record.step, "dominant_region": record.dominant_region,
                               "frames": 1, "mean_novelty": record.novelty})
                continue
            group = groups[-1]
            frame_count = int(group["frames"])
            group["end_step"] = record.step
            group["frames"] = frame_count + 1
            group["mean_novelty"] = (float(group["mean_novelty"]) * frame_count + record.novelty) / (frame_count + 1)
        return groups[:40]
