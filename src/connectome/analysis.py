from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as _numpy

from .graph import ConnectomeGraph
from ..models.instrumented_backend import ActivationFrame
from ..native.wrapper.connectome_kernels import native
from ..utils.array_api import array_api as np, to_numpy

LOG = logging.getLogger(__name__)
MODEL_FILENAME = "aibrain.analyser.npz"
METRIC_HISTORY_LIMIT = 512
MATURITY_STATES = ("Baby", "Teen", "Adult", "Elder")
BABY_FRAME_FLOOR = 2_048
ADULT_FRAME_FLOOR = 32_768
CONSISTENCY_WINDOW = 128
MODEL_SAVE_RETRY_DELAYS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8)
_MODEL_SAVE_LOCK = threading.RLock()


def _normal(rng: object, mean: float, deviation: float, size: object) -> np.ndarray:
    """Sample normal values with the generator APIs shared by NumPy and CuPy."""
    return mean + deviation * rng.standard_normal(size)  # type: ignore[union-attr]


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


RecordSource = Iterable[AnalysisRecord] | Callable[[], Iterable[AnalysisRecord]]


def _record_iterator(records: RecordSource) -> Iterator[AnalysisRecord]:
    source = records() if callable(records) else records
    return iter(source)


class ConnectomeAnalyzer:
    """Online NumPy autoencoder that learns compact visual-activity patterns.

    The network consumes every recorded visual frame but retains only learned
    embeddings and useful event metrics. Its inputs are AIBrain's procedural
    connectome signals, never hidden LLM states.
    """

    def __init__(
        self,
        graph: ConnectomeGraph,
        hidden_width: int = 12,
        model_path: Path | None = None,
    ) -> None:
        self.graph = graph
        self.region_count = len(graph.region_names)
        self.input_width = self.region_count + 4
        self.hidden_width = hidden_width
        self._array = _numpy if isinstance(graph.positions, _numpy.ndarray) else np
        rng = self._array.random.default_rng(9137)
        self.encoder_weights = _normal(
            rng, 0, 0.14, (self.input_width, hidden_width)
        ).astype("f4")
        self.encoder_bias = self._array.zeros(hidden_width, dtype="f4")
        self.decoder_weights = _normal(
            rng, 0, 0.14, (hidden_width, self.input_width)
        ).astype("f4")
        self.decoder_bias = self._array.zeros(self.input_width, dtype="f4")
        self.embedding_centroid = self._array.zeros(hidden_width, dtype="f4")
        self.records: list[AnalysisRecord] = []
        self._learning_rate = 0.025
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
        base = (
            Path(local_app_data)
            if local_app_data
            else Path.home() / "AppData" / "Local"
        )
        return base / "AIBrain" / "analysis_model" / "connectome_autoencoder_v1.npz"

    def observe(self, frame: ActivationFrame, values: np.ndarray) -> AnalysisRecord:
        array = self._array
        regional, active_nodes = native.regions(
            values, self.graph.regions, self.region_count
        )
        counts = (
            array.bincount(self.graph.regions, minlength=self.region_count)
            .clip(1)
            .astype("f4")
        )
        regional_mean = regional / counts
        # Remove static cluster population/topology bias: each region is
        # compared with the current graph-wide activity, not raw node totals.
        global_mean = max(float(values.mean()), 1e-6)
        relative_regional_density = regional_mean / global_mean
        distribution = relative_regional_density / (1.0 + relative_regional_density)
        active_ratio = float(array.count_nonzero(values > 0.1) / len(values))
        mean_delta = abs(float(values.mean()) - self._previous_mean)
        self._previous_mean = float(values.mean())
        features = array.concatenate(
            (
                distribution,
                array.array(
                    (
                        active_ratio,
                        mean_delta,
                        float(values.max()),
                        float(values.std()),
                    ),
                    dtype="f4",
                ),
            )
        )
        encoded_pre = features @ self.encoder_weights + self.encoder_bias
        embedding = array.tanh(encoded_pre)
        decoded_pre = embedding @ self.decoder_weights + self.decoder_bias
        reconstruction = 1.0 / (1.0 + array.exp(-decoded_pre))
        reconstruction_error = float(array.mean((reconstruction - features) ** 2))
        distance = float(array.linalg.norm(embedding - self.embedding_centroid))
        novelty = distance + reconstruction_error * 4.0
        coherence = float(1.0 / (1.0 + reconstruction_error * 40.0))
        update_magnitude = self._train(features, embedding, reconstruction)
        self.embedding_centroid += 0.04 * (embedding - self.embedding_centroid)
        dominant = int(array.argmax(regional))
        record = AnalysisRecord(
            frame.step,
            frame.token_text,
            active_nodes,
            float(values.mean()),
            float(values.max()),
            self.graph.region_names[dominant],
            novelty,
            reconstruction_error,
            coherence,
            tuple(float(value) for value in embedding),
            tuple(float(value) for value in regional_mean),
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
            with _numpy.load(path, allow_pickle=False) as stored:
                tensors = {
                    "encoder_weights": self.encoder_weights,
                    "encoder_bias": self.encoder_bias,
                    "decoder_weights": self.decoder_weights,
                    "decoder_bias": self.decoder_bias,
                    "embedding_centroid": self.embedding_centroid,
                }
                loaded = {name: stored[name] for name in tensors}
                if any(
                    value.shape != tensors[name].shape or not _numpy.isfinite(value).all()
                    for name, value in loaded.items()
                ):
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
                if has_metrics and any(
                    not _numpy.isfinite(_numpy.asarray(value)).all()
                    for value in histories.values()
                ):
                    return False
                stored_state = str(
                    _numpy.asarray(stored.get("maturity_state", _numpy.array("Baby"))).reshape(
                        -1
                    )[0]
                )
                transition_count = int(
                    _numpy.asarray(stored.get("transition_count", _numpy.array(0))).reshape(-1)[
                        0
                    ]
                )
                weights_frozen = bool(
                    _numpy.asarray(stored.get("weights_frozen", _numpy.array(False))).reshape(
                        -1
                    )[0]
                )
        except (OSError, KeyError, TypeError, ValueError):
            return False

        try:
            for name, target in tensors.items():
                # NPZ persistence is deliberately host-backed.  Convert each
                # restored tensor to the graph's active array backend before
                # assignment; CuPy otherwise treats a NumPy RHS as a scalar
                # fill and raises while the desktop window is being built.
                target[:] = self._array.asarray(loaded[name])
        except (TypeError, ValueError):
            return False
        self.frames_seen = frames_seen
        if has_metrics:
            self.reconstruction_history = [
                float(value)
                for value in _numpy.asarray(histories["reconstruction_history"])[
                    -METRIC_HISTORY_LIMIT:
                ]
            ]
            self.novelty_history = [
                float(value)
                for value in _numpy.asarray(histories["novelty_history"])[
                    -METRIC_HISTORY_LIMIT:
                ]
            ]
            self.update_magnitude_history = [
                float(value)
                for value in _numpy.asarray(histories["update_magnitude_history"])[
                    -METRIC_HISTORY_LIMIT:
                ]
            ]
            self.maturity_state = (
                stored_state if stored_state in MATURITY_STATES else "Baby"
            )
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
        with _MODEL_SAVE_LOCK:
            return self._save_model_locked()

    def _save_model_locked(self) -> bool:
        temporary_path: Path | None = None
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "wb", dir=self.model_path.parent, delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                _numpy.savez_compressed(
                    handle,
                    encoder_weights=to_numpy(self.encoder_weights),
                    encoder_bias=to_numpy(self.encoder_bias),
                    decoder_weights=to_numpy(self.decoder_weights),
                    decoder_bias=to_numpy(self.decoder_bias),
                    embedding_centroid=to_numpy(self.embedding_centroid),
                    frames_seen=_numpy.array(self.frames_seen),
                    reconstruction_history=_numpy.asarray(
                        self.reconstruction_history, dtype="f4"
                    ),
                    novelty_history=_numpy.asarray(self.novelty_history, dtype="f4"),
                    update_magnitude_history=_numpy.asarray(
                        self.update_magnitude_history, dtype="f4"
                    ),
                    maturity_state=_numpy.array(self.maturity_state),
                    transition_count=_numpy.array(self.transition_count),
                    weights_frozen=_numpy.array(self.weights_frozen),
                )
                handle.flush()
                os.fsync(handle.fileno())
            self._replace_model_file(temporary_path)
            return True
        except (OSError, ValueError) as exc:
            LOG.warning(
                "Could not persist NN Analysis+ memory to %s: %s", self.model_path, exc
            )
            return False
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)

    def _replace_model_file(self, temporary_path: Path) -> None:
        """Wait out short Windows reader locks while retaining atomic replacement."""
        for delay in (*MODEL_SAVE_RETRY_DELAYS, None):
            try:
                os.replace(temporary_path, self.model_path)
                return
            except OSError as exc:
                sharing_violation = isinstance(exc, PermissionError) or getattr(
                    exc, "winerror", None
                ) in {5, 32}
                if not sharing_violation or delay is None:
                    raise
                time.sleep(delay)

    def _train(
        self, features: np.ndarray, embedding: np.ndarray, reconstruction: np.ndarray
    ) -> float:
        """One gradient-descent step for tanh encoder + sigmoid decoder."""
        if self.weights_frozen:
            return 0.0
        gradient_output = 2.0 * (reconstruction - features) / self.input_width
        gradient_output *= reconstruction * (1.0 - reconstruction)
        array = self._array
        gradient_decoder_weights = array.outer(embedding, gradient_output)
        gradient_embedding = gradient_output @ self.decoder_weights.T
        gradient_encoder_pre = gradient_embedding * (1.0 - embedding**2)
        self.decoder_weights -= self._learning_rate * gradient_decoder_weights
        self.decoder_bias -= self._learning_rate * gradient_output
        self.encoder_weights -= self._learning_rate * array.outer(
            features, gradient_encoder_pre
        )
        self.encoder_bias -= self._learning_rate * gradient_encoder_pre
        return float(
            array.sqrt(
                array.square(self._learning_rate * gradient_output).sum()
                + array.square(self._learning_rate * gradient_encoder_pre).sum()
                + array.square(self._learning_rate * gradient_decoder_weights).sum()
            )
        )

    def _append_metrics(
        self, reconstruction_error: float, novelty: float, update_magnitude: float
    ) -> None:
        for history, value in (
            (self.reconstruction_history, reconstruction_error),
            (self.novelty_history, novelty),
            (self.update_magnitude_history, update_magnitude),
        ):
            history.append(float(value))
            del history[:-METRIC_HISTORY_LIMIT]

    def _sustained_consistency(self) -> bool:
        if (
            len(self.reconstruction_history) < CONSISTENCY_WINDOW
            or len(self.update_magnitude_history) < CONSISTENCY_WINDOW
        ):
            return False
        errors = np.asarray(
            self.reconstruction_history[-CONSISTENCY_WINDOW:], dtype="f4"
        )
        updates = np.asarray(
            self.update_magnitude_history[-CONSISTENCY_WINDOW:], dtype="f4"
        )
        early_updates = np.asarray(
            self.update_magnitude_history[
                -2 * CONSISTENCY_WINDOW : -CONSISTENCY_WINDOW
            ],
            dtype="f4",
        )
        stable_error = float(errors.std()) <= 0.015 and float(errors.mean()) <= 0.08
        slowdown = (
            not len(early_updates)
            or float(updates.mean()) <= float(early_updates.mean()) * 1.05
        )
        return stable_error and slowdown

    def _sustained_overfitting(self) -> bool:
        if len(self.reconstruction_history) < 96:
            return False
        errors = np.asarray(self.reconstruction_history, dtype="f4")
        recent = float(errors[-48:].mean())
        previous = float(errors[-96:-48].mean())
        updates = float(
            np.asarray(self.update_magnitude_history[-48:], dtype="f4").mean()
        )
        return (
            recent > previous * 1.18 and recent - previous >= 0.008 and updates <= 0.003
        )

    def _advance_maturity(self) -> None:
        """Promote only from persisted, sustained evidence; never demote silently."""
        if (
            self.maturity_state == "Baby"
            and self.frames_seen >= BABY_FRAME_FLOOR
            and self._sustained_consistency()
        ):
            self.maturity_state, self.transition_count = (
                "Teen",
                self.transition_count + 1,
            )
        elif (
            self.maturity_state == "Teen"
            and self.frames_seen >= ADULT_FRAME_FLOOR
            and self._sustained_consistency()
        ):
            self.maturity_state, self.transition_count = (
                "Adult",
                self.transition_count + 1,
            )
        elif self.maturity_state == "Adult" and self._sustained_overfitting():
            self.maturity_state, self.transition_count, self.weights_frozen = (
                "Elder",
                self.transition_count + 1,
                True,
            )

    def maturity_report(self) -> dict[str, object]:
        """Expose transparent state requirements; this is not an accuracy score."""
        consistent = self._sustained_consistency()
        overfitting = self._sustained_overfitting()
        next_state = {"Baby": "Teen", "Teen": "Adult", "Adult": "Elder", "Elder": None}[
            self.maturity_state
        ]
        required_frames = {
            "Baby": BABY_FRAME_FLOOR,
            "Teen": ADULT_FRAME_FLOOR,
            "Adult": ADULT_FRAME_FLOOR,
            "Elder": ADULT_FRAME_FLOOR,
        }[self.maturity_state]
        unmet: list[str] = []
        if self.maturity_state == "Baby":
            if self.frames_seen < BABY_FRAME_FLOOR:
                unmet.append(
                    f"{BABY_FRAME_FLOOR - self.frames_seen:,} more persisted frames"
                )
            if not consistent:
                unmet.append(
                    f"{CONSISTENCY_WINDOW} stable reconstruction/update samples"
                )
        elif self.maturity_state == "Teen":
            if self.frames_seen < ADULT_FRAME_FLOOR:
                unmet.append(
                    f"{ADULT_FRAME_FLOOR - self.frames_seen:,} more persisted frames"
                )
            if not consistent:
                unmet.append("sustained consistency and learning slowdown")
        elif self.maturity_state == "Adult" and not overfitting:
            unmet.append("sustained post-Adult overfitting signal")
        if self.maturity_state == "Teen":
            stage_frames = max(0, self.frames_seen - BABY_FRAME_FLOOR)
            stage_required = ADULT_FRAME_FLOOR - BABY_FRAME_FLOOR
        else:
            stage_frames = self.frames_seen
            stage_required = required_frames
        progress = (
            100
            if next_state is None
            else min(99, round(stage_frames / stage_required * 100))
        )
        progress_detail = (
            "Elder is the terminal safeguarded state."
            if next_state is None
            else f"{stage_frames:,} of {stage_required:,} stage frames; "
            + (
                "consistency evidence satisfied."
                if consistent
                else f"{CONSISTENCY_WINDOW} stable samples are also required."
            )
        )
        return {
            "state": self.maturity_state,
            "next_state": next_state,
            "progress_percent": progress,
            "progress_detail": progress_detail,
            "unmet_conditions": unmet,
            "consistency_sustained": consistent,
            "overfitting_sustained": overfitting,
            "weights_frozen": self.weights_frozen,
            "readiness": (
                "ready" if self.maturity_state in {"Adult", "Elder"} else "caution"
            ),
            "note": "Maturity describes persisted training health and coverage; it is not a measured accuracy guarantee.",
        }

    def summary(self, records: RecordSource | None = None) -> dict[str, object]:
        records = self.records if records is None else records
        maturity = self.maturity_report()
        count = 0
        novelty_total = 0.0
        peak_novelty = 0.0
        reconstruction_total = 0.0
        coherence_total = 0.0
        most_active: AnalysisRecord | None = None
        for record in _record_iterator(records):
            count += 1
            novelty_total += record.novelty
            peak_novelty = max(peak_novelty, record.novelty)
            reconstruction_total += record.reconstruction_error
            coherence_total += record.coherence
            if most_active is None or record.active_nodes > most_active.active_nodes:
                most_active = record
        if not count or most_active is None:
            return {"frames": 0, "maturity": maturity}
        return {
            "frames": count,
            "mean_novelty": novelty_total / count,
            "peak_novelty": peak_novelty,
            "mean_reconstruction_error": reconstruction_total / count,
            "mean_coherence": coherence_total / count,
            "most_active_region": most_active.dominant_region,
            "maturity": maturity,
        }

    def smart_report(self, records: RecordSource | None = None) -> dict[str, object]:
        """Return compact findings learned from all frames, rather than raw vectors."""
        records = self.records if records is None else records
        count = 0
        novelty_total = 0.0
        peak_novelty = 0.0
        reconstruction_total = 0.0
        coherence_total = 0.0
        most_active: AnalysisRecord | None = None
        region_totals = _numpy.zeros(self.region_count, dtype="f8")
        region_peaks = _numpy.zeros(self.region_count, dtype="f4")
        key_events: list[AnalysisRecord] = []
        segments: list[PatternSegment] = []
        segments_complete = False

        for record in _record_iterator(records):
            count += 1
            novelty_total += record.novelty
            peak_novelty = max(peak_novelty, record.novelty)
            reconstruction_total += record.reconstruction_error
            coherence_total += record.coherence
            if most_active is None or record.active_nodes > most_active.active_nodes:
                most_active = record
            regional = _numpy.asarray(record.regional_activity, dtype="f4")
            region_totals += regional
            region_peaks = _numpy.maximum(region_peaks, regional)
            key_events.append(record)
            key_events.sort(
                key=lambda item: item.novelty + item.reconstruction_error * 3,
                reverse=True,
            )
            del key_events[24:]
            if segments_complete:
                continue
            if not segments or segments[-1]["dominant_region"] != record.dominant_region:
                if len(segments) >= 40:
                    segments_complete = True
                    continue
                segments.append(
                    {
                        "start_step": record.step,
                        "end_step": record.step,
                        "dominant_region": record.dominant_region,
                        "frames": 1,
                        "mean_novelty": record.novelty,
                    }
                )
                continue
            group = segments[-1]
            frame_count = group["frames"]
            group["end_step"] = record.step
            group["frames"] = frame_count + 1
            group["mean_novelty"] = (
                group["mean_novelty"] * frame_count + record.novelty
            ) / (frame_count + 1)

        if not count or most_active is None:
            return {"frames_processed": 0, "maturity": self.maturity_report()}
        region_means = region_totals / count
        top_region = self.graph.region_names[int(region_means.argmax())]
        summary = {
            "frames": count,
            "mean_novelty": novelty_total / count,
            "peak_novelty": peak_novelty,
            "mean_reconstruction_error": reconstruction_total / count,
            "mean_coherence": coherence_total / count,
            "most_active_region": most_active.dominant_region,
            "maturity": self.maturity_report(),
        }
        return {
            "frames_processed": count,
            "neural_network": {
                "architecture": f"{self.input_width} → {self.hidden_width} → {self.input_width} autoencoder",
                "activation": "tanh encoder / sigmoid decoder",
                "training": "online gradient descent for each visual frame",
                "learned_parameters": int(
                    self.encoder_weights.size
                    + self.encoder_bias.size
                    + self.decoder_weights.size
                    + self.decoder_bias.size
                ),
                "lifetime_frames_seen": self.frames_seen,
                "feature_calibration": "Region-density and temporal-change features remove fixed cluster size and "
                "global renderer-amplitude bias.",
                "maturity": self.maturity_report(),
            },
            "session_findings": {
                **summary,
                "primary_pattern": f"Highest mean visual activity was in {top_region}.",
                "pattern_segments": segments,
            },
            "regional_profile": [
                {
                    "region": name,
                    "mean_activity": float(region_means[index]),
                    "peak_activity": float(region_peaks[index]),
                }
                for index, name in enumerate(self.graph.region_names)
            ],
            "key_events": [
                {
                    "step": record.step,
                    "token": record.token,
                    "dominant_region": record.dominant_region,
                    "novelty": record.novelty,
                    "reconstruction_error": record.reconstruction_error,
                    "coherence": record.coherence,
                    "embedding": [round(value, 5) for value in record.embedding],
                }
                for record in sorted(key_events, key=lambda item: item.step)
            ],
        }

    def _segments(self, records: RecordSource | None = None) -> list[PatternSegment]:
        records = self.records if records is None else records
        groups: list[PatternSegment] = []

        for record in _record_iterator(records):
            if not groups or groups[-1]["dominant_region"] != record.dominant_region:
                if len(groups) >= 40:
                    break
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
            group["mean_novelty"] = (
                group["mean_novelty"] * frame_count + record.novelty
            ) / (frame_count + 1)

        return groups
