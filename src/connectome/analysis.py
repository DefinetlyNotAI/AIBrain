from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np

from .graph import ConnectomeGraph
from ..models.instrumented_backend import ActivationFrame
from ..native.wrapper.connectome_kernels import native

LOG = logging.getLogger(__name__)
MODEL_FILENAME = "aibrain.analyser.npz"
METRIC_HISTORY_LIMIT = 512
MATURITY_STATES = ("Baby", "Teen", "Adult", "Elder")


class PatternSegment(TypedDict):
    start_step: int
    end_step: int
    dominant_region: str
    frames: int
    mean_novelty: float


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

    def __init__(self, graph: ConnectomeGraph, hidden_width: int = 12, model_path: Path | None = None) -> None:
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
        self._previous_mean = 0.0
        self.frames_seen = 0
        # These rolling metrics are deliberately persisted separately from a
        # transient export.  They make maturity a repeatable health signal,
        # while the fixed cap keeps the NPZ small and future-compatible.
        self.reconstruction_history: list[float] = []
        self.novelty_history: list[float] = []
        self.update_magnitude_history: list[float] = []
        self.maturity_state = "Baby"
        self.transition_count = 0
        self.weights_frozen = False
        self.model_path = model_path or self.default_model_path()
        self._load_persistent_model()

    @staticmethod
    def default_model_path() -> Path:
        """Return the project-local learned-analysis model location."""
        return Path(__file__).resolve().parents[2] / "models" / MODEL_FILENAME

    @staticmethod
    def legacy_model_path() -> Path:
        """Return the previous per-user location for one-time migration."""
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return base / "AIBrain" / "analysis_model" / "connectome_autoencoder_v1.npz"

    def observe(self, frame: ActivationFrame, values: np.ndarray) -> AnalysisRecord:
        regional, active_nodes = native.regions(values, self.graph.regions, self.region_count)
        counts = np.bincount(self.graph.regions, minlength=self.region_count).clip(1).astype("f4")
        regional_mean = regional / counts
        # Remove static cluster population/topology bias: each region is
        # compared with the current graph-wide activity, not raw node totals.
        global_mean = max(float(values.mean()), 1e-6)
        relative_regional_density = regional_mean / global_mean
        distribution = relative_regional_density / (1.0 + relative_regional_density)
        active_ratio = float(np.count_nonzero(values > .1) / len(values))
        mean_delta = abs(float(values.mean()) - self._previous_mean)
        self._previous_mean = float(values.mean())
        features = np.concatenate(
            (distribution, np.array((active_ratio, mean_delta, float(values.max()), float(values.std())), dtype="f4")))
        encoded_pre = features @ self.encoder_weights + self.encoder_bias
        embedding = np.tanh(encoded_pre)
        decoded_pre = embedding @ self.decoder_weights + self.decoder_bias
        reconstruction = 1.0 / (1.0 + np.exp(-decoded_pre))
        reconstruction_error = float(np.mean((reconstruction - features) ** 2))
        distance = float(np.linalg.norm(embedding - self.embedding_centroid))
        novelty = distance + reconstruction_error * 4.0
        coherence = float(1.0 / (1.0 + reconstruction_error * 40.0))
        update_magnitude = self._train(features, embedding, reconstruction)
        self.embedding_centroid += .04 * (embedding - self.embedding_centroid)
        dominant = int(np.argmax(regional))
        record = AnalysisRecord(
            frame.step, frame.token_text, active_nodes, float(values.mean()), float(values.max()),
            self.graph.region_names[dominant], novelty, reconstruction_error, coherence,
            tuple(float(value) for value in embedding), tuple(float(value) for value in regional_mean),
        )
        self.records.append(record)
        self.frames_seen += 1
        self._append_metrics(reconstruction_error, novelty, update_magnitude)
        self._advance_maturity()
        # A process can be closed or terminated between generated responses.
        # Commit every online-learning step, rather than waiting for a clean exit
        # or an arbitrary batch threshold, so the NPZ is durable memory.
        self.save_model()
        return record

    def _load_persistent_model(self) -> None:
        if self._load_model(self.model_path):
            return
        legacy_path = self.legacy_model_path()
        if legacy_path != self.model_path and self._load_model(legacy_path):
            # Preserve learning from older installs while moving it out of the
            # application directory, which is often read-only for releases.
            self.save_model()

    def _load_model(self, path: Path) -> bool:
        try:
            with np.load(path, allow_pickle=False) as stored:
                tensors = {
                    "encoder_weights": self.encoder_weights,
                    "encoder_bias": self.encoder_bias,
                    "decoder_weights": self.decoder_weights,
                    "decoder_bias": self.decoder_bias,
                    "embedding_centroid": self.embedding_centroid,
                }
                loaded = {name: stored[name] for name in tensors}
                if any(value.shape != tensors[name].shape or not np.isfinite(value).all()
                       for name, value in loaded.items()):
                    return False
                frames_seen = int(stored["frames_seen"])
                if frames_seen < 0:
                    return False
                histories = {
                    "reconstruction_history": stored.get("reconstruction_history"),
                    "novelty_history": stored.get("novelty_history"),
                    "update_magnitude_history": stored.get("update_magnitude_history"),
                }
                has_metrics = all(value is not None for value in histories.values())
                if has_metrics and any(not np.isfinite(np.asarray(value)).all() for value in histories.values()):
                    return False
                stored_state = str(np.asarray(stored.get("maturity_state", np.array("Baby"))).reshape(-1)[0])
                transition_count = int(np.asarray(stored.get("transition_count", np.array(0))).reshape(-1)[0])
                weights_frozen = bool(np.asarray(stored.get("weights_frozen", np.array(False))).reshape(-1)[0])
        except (OSError, KeyError, TypeError, ValueError):
            return False

        for name, target in tensors.items():
            target[:] = loaded[name]
        self.frames_seen = frames_seen
        if has_metrics:
            self.reconstruction_history = [float(value) for value in
                                           np.asarray(histories["reconstruction_history"])[-METRIC_HISTORY_LIMIT:]]
            self.novelty_history = [float(value) for value in
                                    np.asarray(histories["novelty_history"])[-METRIC_HISTORY_LIMIT:]]
            self.update_magnitude_history = [float(value) for value in
                                             np.asarray(histories["update_magnitude_history"])[-METRIC_HISTORY_LIMIT:]]
            self.maturity_state = stored_state if stored_state in MATURITY_STATES else "Baby"
            self.transition_count = transition_count
            self.weights_frozen = weights_frozen
        else:
            # A legacy NPZ did not observe the conditions required for state
            # promotion.  Retain its weights and frame counter, but require
            # fresh evidence instead of inferring Adult/Elder from age alone.
            self.reconstruction_history = []
            self.novelty_history = []
            self.update_magnitude_history = []
            self.maturity_state = "Baby"
            self.transition_count = 0
            self.weights_frozen = False
        return True

    def save_model(self) -> bool:
        """Atomically persist learned weights for future analysis sessions."""
        temporary_path: Path | None = None
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("wb", dir=self.model_path.parent, delete=False) as handle:
                temporary_path = Path(handle.name)
                np.savez_compressed(handle, encoder_weights=self.encoder_weights, encoder_bias=self.encoder_bias,
                                    decoder_weights=self.decoder_weights, decoder_bias=self.decoder_bias,
                                    embedding_centroid=self.embedding_centroid, frames_seen=np.array(self.frames_seen),
                                    reconstruction_history=np.asarray(self.reconstruction_history, dtype="f4"),
                                    novelty_history=np.asarray(self.novelty_history, dtype="f4"),
                                    update_magnitude_history=np.asarray(self.update_magnitude_history, dtype="f4"),
                                    maturity_state=np.array(self.maturity_state),
                                    transition_count=np.array(self.transition_count),
                                    weights_frozen=np.array(self.weights_frozen))
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(self.model_path)
            return True
        except (OSError, ValueError) as exc:
            LOG.warning("Could not persist NN Analysis+ memory to %s: %s", self.model_path, exc)
            return False
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)

    def _train(self, features: np.ndarray, embedding: np.ndarray, reconstruction: np.ndarray) -> float:
        """One gradient-descent step for tanh encoder + sigmoid decoder."""
        if self.weights_frozen:
            return 0.0
        gradient_output = 2.0 * (reconstruction - features) / self.input_width
        gradient_output *= reconstruction * (1.0 - reconstruction)
        gradient_decoder_weights = np.outer(embedding, gradient_output)
        gradient_embedding = gradient_output @ self.decoder_weights.T
        gradient_encoder_pre = gradient_embedding * (1.0 - embedding ** 2)
        self.decoder_weights -= self._learning_rate * gradient_decoder_weights
        self.decoder_bias -= self._learning_rate * gradient_output
        self.encoder_weights -= self._learning_rate * np.outer(features, gradient_encoder_pre)
        self.encoder_bias -= self._learning_rate * gradient_encoder_pre
        return float(np.sqrt(
            np.square(self._learning_rate * gradient_output).sum()
            + np.square(self._learning_rate * gradient_encoder_pre).sum()
            + np.square(self._learning_rate * gradient_decoder_weights).sum()
        ))

    def _append_metrics(self, reconstruction_error: float, novelty: float, update_magnitude: float) -> None:
        for history, value in (
                (self.reconstruction_history, reconstruction_error),
                (self.novelty_history, novelty),
                (self.update_magnitude_history, update_magnitude),
        ):
            history.append(float(value))
            del history[:-METRIC_HISTORY_LIMIT]

    def _sustained_consistency(self) -> bool:
        if len(self.reconstruction_history) < 32 or len(self.update_magnitude_history) < 32:
            return False
        errors = np.asarray(self.reconstruction_history[-32:], dtype="f4")
        updates = np.asarray(self.update_magnitude_history[-32:], dtype="f4")
        early_updates = np.asarray(self.update_magnitude_history[-64:-32], dtype="f4")
        stable_error = float(errors.std()) <= .015 and float(errors.mean()) <= .08
        slowdown = not len(early_updates) or float(updates.mean()) <= float(early_updates.mean()) * 1.05
        return stable_error and slowdown

    def _sustained_overfitting(self) -> bool:
        if len(self.reconstruction_history) < 96:
            return False
        errors = np.asarray(self.reconstruction_history, dtype="f4")
        recent = float(errors[-48:].mean())
        previous = float(errors[-96:-48].mean())
        updates = float(np.asarray(self.update_magnitude_history[-48:], dtype="f4").mean())
        return recent > previous * 1.18 and recent - previous >= .008 and updates <= .003

    def _advance_maturity(self) -> None:
        """Promote only from persisted, sustained evidence; never demote silently."""
        if self.maturity_state == "Baby" and self.frames_seen >= 250 and self._sustained_consistency():
            self.maturity_state, self.transition_count = "Teen", self.transition_count + 1
        elif self.maturity_state == "Teen" and self.frames_seen >= 4096 and self._sustained_consistency():
            self.maturity_state, self.transition_count = "Adult", self.transition_count + 1
        elif self.maturity_state == "Adult" and self._sustained_overfitting():
            self.maturity_state, self.transition_count, self.weights_frozen = "Elder", self.transition_count + 1, True

    def maturity_report(self) -> dict[str, object]:
        """Expose transparent state requirements; this is not an accuracy score."""
        consistent = self._sustained_consistency()
        overfitting = self._sustained_overfitting()
        next_state = {"Baby": "Teen", "Teen": "Adult", "Adult": "Elder", "Elder": None}[self.maturity_state]
        required_frames = {"Baby": 250, "Teen": 4096, "Adult": 4096, "Elder": 4096}[self.maturity_state]
        unmet: list[str] = []
        if self.maturity_state == "Baby":
            if self.frames_seen < 250:
                unmet.append(f"{250 - self.frames_seen} more persisted frames")
            if not consistent:
                unmet.append("32 stable recent reconstruction/update samples")
        elif self.maturity_state == "Teen":
            if self.frames_seen < 4096:
                unmet.append(f"{4096 - self.frames_seen} more persisted frames")
            if not consistent:
                unmet.append("sustained consistency and learning slowdown")
        elif self.maturity_state == "Adult" and not overfitting:
            unmet.append("sustained post-Adult overfitting signal")
        progress = 100 if next_state is None else min(99, round(self.frames_seen / required_frames * 100))
        return {
            "state": self.maturity_state,
            "next_state": next_state,
            "progress_percent": progress,
            "unmet_conditions": unmet,
            "consistency_sustained": consistent,
            "overfitting_sustained": overfitting,
            "weights_frozen": self.weights_frozen,
            "readiness": "ready" if self.maturity_state in {"Adult", "Elder"} else "caution",
            "note": "Maturity describes persisted training health and coverage; it is not a measured accuracy guarantee.",
        }

    def summary(self) -> dict[str, object]:
        maturity = self.maturity_report()
        if not self.records:
            return {"frames": 0, "maturity": maturity}
        novelty = np.array([record.novelty for record in self.records])
        reconstruction = np.array([record.reconstruction_error for record in self.records])
        coherence = np.array([record.coherence for record in self.records])
        return {
            "frames": len(self.records), "mean_novelty": float(novelty.mean()), "peak_novelty": float(novelty.max()),
            "mean_reconstruction_error": float(reconstruction.mean()), "mean_coherence": float(coherence.mean()),
            "most_active_region": max(self.records, key=lambda item: item.active_nodes).dominant_region,
            "maturity": maturity,
        }

    def smart_report(self) -> dict[str, object]:
        """Return compact findings learned from all frames, rather than raw vectors."""
        if not self.records:
            return {"frames_processed": 0, "maturity": self.maturity_report()}
        regional = np.asarray([record.regional_activity for record in self.records], dtype="f4")
        region_means = regional.mean(axis=0)
        region_peaks = regional.max(axis=0)
        key_events = sorted(self.records, key=lambda item: item.novelty + item.reconstruction_error * 3, reverse=True)[
            :24]
        top_region = self.graph.region_names[int(region_means.argmax())]
        return {
            "frames_processed": len(self.records),
            "neural_network": {
                "architecture": f"{self.input_width} → {self.hidden_width} → {self.input_width} autoencoder",
                "activation": "tanh encoder / sigmoid decoder",
                "training": "online gradient descent for each visual frame",
                "learned_parameters": int(
                    self.encoder_weights.size +
                    self.encoder_bias.size +
                    self.decoder_weights.size +
                    self.decoder_bias.size
                ),
                "lifetime_frames_seen": self.frames_seen,
                "feature_calibration": "Region-density and temporal-change features remove fixed cluster size and "
                                       "global renderer-amplitude bias.",
                "maturity": self.maturity_report(),
            },
            "session_findings": {
                **self.summary(), "primary_pattern": f"Highest mean visual activity was in {top_region}.",
                "pattern_segments": self._segments(),
            },
            "regional_profile": [
                {"region": name, "mean_activity": float(region_means[index]),
                 "peak_activity": float(region_peaks[index])}
                for index, name in enumerate(self.graph.region_names)
            ],
            "key_events": [
                {"step": record.step, "token": record.token, "dominant_region": record.dominant_region,
                 "novelty": record.novelty, "reconstruction_error": record.reconstruction_error,
                 "coherence": record.coherence, "embedding": [round(value, 5) for value in record.embedding]}
                for record in sorted(key_events, key=lambda item: item.step)
            ],
        }

    def _segments(self) -> list[PatternSegment]:
        groups: list[PatternSegment] = []

        for record in self.records:
            if not groups or groups[-1]["dominant_region"] != record.dominant_region:
                groups.append(
                    {
                        "start_step": record.step,
                        "end_step": record.step,
                        "dominant_region": record.dominant_region,
                        "frames": 1,
                        "mean_novelty": record.novelty,
                    }
                )
                continue

            group = groups[-1]
            frame_count = group["frames"]
            group["end_step"] = record.step
            group["frames"] = frame_count + 1
            group["mean_novelty"] = (group["mean_novelty"] * frame_count + record.novelty) / (frame_count + 1)

        return groups[:40]
