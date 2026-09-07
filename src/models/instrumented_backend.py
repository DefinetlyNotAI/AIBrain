from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic

REALTIME_REGIONS: tuple[str, ...] = (
    "Context utilisation",
    "Requested output progress",
    "Stream latency",
    "Retokenized throughput",
    "Raw-logit entropy",
    "Raw top probability",
    "Raw top-5 mass",
    "Raw confidence margin",
    "Recent output rarity",
)


class ActivitySource(str, Enum):
    REAL_TIME = "Real-time"


@dataclass(frozen=True, slots=True)
class LogitMetrics:
    """Measurements read from llama.cpp's raw next-token logits."""

    vocabulary_size: int
    context_tokens: int
    raw_entropy_bits: float
    raw_normalized_entropy: float
    raw_top_probability: float
    raw_top_five_mass: float
    raw_confidence_margin: float


@dataclass(frozen=True, slots=True)
class GenerationChunk:
    """One visible streamed text chunk and its live backend measurements."""

    text: str
    retokenized_ids: tuple[int, ...]
    logits: LogitMetrics | None


@dataclass(frozen=True, slots=True)
class ActivationFrame:
    chunk_text: str
    step: int
    source: ActivitySource
    timestamp: float = field(default_factory=monotonic)
    regions: Mapping[str, float] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)


def realtime_activation_frame(
    chunk: GenerationChunk,
    step: int,
    *,
    output_tokens: int | None = None,
    max_tokens: int,
    context_limit: int,
    stream_latency_seconds: float,
    retokenized_tokens_per_second: float,
    recent_output_occurrences: int,
) -> ActivationFrame:
    """Normalize only measured generation values into display channels."""
    token_count = len(chunk.retokenized_ids)
    output_token_count = step if output_tokens is None else max(0, output_tokens)
    latency_ms = max(0.0, stream_latency_seconds * 1000.0)
    context_tokens = chunk.logits.context_tokens if chunk.logits else 0
    entropy_bits = chunk.logits.raw_entropy_bits if chunk.logits else 0.0
    regions = {
        "Context utilisation": min(1.0, context_tokens / max(1, context_limit)),
        "Requested output progress": min(1.0, output_token_count / max(1, max_tokens)),
        # These caps are display scales. Raw values remain in frame.metrics.
        "Stream latency": min(1.0, latency_ms / 500.0),
        "Retokenized throughput": min(1.0, retokenized_tokens_per_second / 50.0),
        "Raw-logit entropy": (
            chunk.logits.raw_normalized_entropy if chunk.logits else 0.0
        ),
        "Raw top probability": (
            chunk.logits.raw_top_probability if chunk.logits else 0.0
        ),
        "Raw top-5 mass": (chunk.logits.raw_top_five_mass if chunk.logits else 0.0),
        "Raw confidence margin": (
            chunk.logits.raw_confidence_margin if chunk.logits else 0.0
        ),
        "Recent output rarity": 1.0 / (1.0 + max(0, recent_output_occurrences)),
    }
    metrics = {
        "context_tokens": float(context_tokens),
        "context_limit": float(context_limit),
        "output_tokens": float(output_token_count),
        "chunk_tokens": float(token_count),
        "stream_latency_ms": latency_ms,
        "retokenized_tokens_per_second": max(0.0, retokenized_tokens_per_second),
        "vocabulary_size": float(chunk.logits.vocabulary_size if chunk.logits else 0),
        "raw_logits_available": 1.0 if chunk.logits else 0.0,
        "raw_logit_entropy_bits": entropy_bits,
        "raw_top_probability": (
            chunk.logits.raw_top_probability if chunk.logits else 0.0
        ),
        "raw_top_five_mass": (chunk.logits.raw_top_five_mass if chunk.logits else 0.0),
        "raw_confidence_margin": (
            chunk.logits.raw_confidence_margin if chunk.logits else 0.0
        ),
        "recent_output_occurrences": float(max(0, recent_output_occurrences)),
    }
    return ActivationFrame(
        chunk.text,
        step,
        ActivitySource.REAL_TIME,
        regions=regions,
        metrics=metrics,
    )


def retokenized_throughput(token_count: int, elapsed_seconds: float) -> float:
    """Return a finite retokenized-output rate for a streamed text chunk."""
    return max(0.0, token_count / max(elapsed_seconds, 1e-6))
