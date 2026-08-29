from __future__ import annotations

import hashlib

from ..utils.array_api import array_api as np

from .graph import ConnectomeGraph

REGIONS = (
    "Input / tokens",
    "Embeddings",
    "Early processing",
    "Attention clusters",
    "MLP clusters",
    "Residual pathways",
    "Middle processing",
    "Late processing",
    "Output / logits",
)

# Region names are stable, so this map gives every procedural cluster the same
# distinct color across launches, qualities, replay, and fallback rendering.
DARK_BACKGROUND_CLUSTER_COLOR_MAP = dict(
    zip(
        REGIONS,
        (
            "#75E6FF",
            "#78B7FF",
            "#8D9DFF",
            "#A786FF",
            "#D48CFF",
            "#FF8CE2",
            "#FF91B5",
            "#FFB067",
            "#C5F28A",
        ),
        strict=True,
    )
)
# Bright colours are readable on the default dark renderer. A user-selected
# light renderer background automatically gets a distinct, lower-luminance
# map, keeping idle clusters legible without making colours user-configurable.
LIGHT_BACKGROUND_CLUSTER_COLOR_MAP = dict(
    zip(
        REGIONS,
        (
            "#007C9F",
            "#245BC1",
            "#5546BE",
            "#743CB5",
            "#9B328F",
            "#B93D72",
            "#BE4A4A",
            "#A75B12",
            "#547D17",
        ),
        strict=True,
    )
)
CLUSTER_COLOR_MAP = DARK_BACKGROUND_CLUSTER_COLOR_MAP


def cluster_colour_map_for_background(background: str) -> dict[str, str]:
    """Return a contrasting semantic-region palette for a renderer background."""
    value = background.lstrip("#")
    if len(value) != 6:
        return DARK_BACKGROUND_CLUSTER_COLOR_MAP
    try:
        red, green, blue = (int(value[offset : offset + 2], 16) for offset in (0, 2, 4))
    except ValueError:
        return DARK_BACKGROUND_CLUSTER_COLOR_MAP
    luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255
    return (
        LIGHT_BACKGROUND_CLUSTER_COLOR_MAP
        if luminance >= 0.5
        else DARK_BACKGROUND_CLUSTER_COLOR_MAP
    )


def _seed(key: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(key.encode(), digest_size=8).digest(), "little"
    )


def build_connectome(
    model_key: str, quality: str = "Medium", cluster_spacing: float = 1.0
) -> ConnectomeGraph:
    counts = {"Low": 5000, "Medium": 11000, "High": 22000}
    n = counts.get(quality, 11000)
    rng = np.random.default_rng(_seed(model_key + quality))
    region_ids = rng.choice(
        len(REGIONS),
        size=n,
        p=np.array([0.07, 0.10, 0.12, 0.14, 0.16, 0.10, 0.12, 0.12, 0.07]),
    )
    # Nine deliberately separated 2D clusters. Keeping the z coordinate
    # varied preserves depth in 3D while the flat map remains non-overlapping.
    grid = np.array(
        ((-1, 1), (0, 1), (1, 1), (-1, 0), (0, 0), (1, 0), (-1, -1), (0, -1), (1, -1)),
        dtype=np.float32,
    )
    centers = np.column_stack(
        (grid[:, 0] * 12.0, grid[:, 1] * 9.5, rng.normal(0, 3.5, len(REGIONS)))
    )
    centers *= cluster_spacing
    positions = centers[region_ids] + rng.normal(0, 1.35, size=(n, 3))
    positions += rng.normal(0, 0.25, size=(n, 1)) * np.array([1, -0.4, 0.6])
    edges_per_node = {"Low": 5, "Medium": 7, "High": 9}.get(quality, 7)
    edge_parts: list[np.ndarray] = []
    for region in range(len(REGIONS)):
        nodes = np.flatnonzero(region_ids == region)
        if len(nodes) < 2:
            continue
        source = rng.choice(nodes, size=len(nodes) * edges_per_node, replace=True)
        destination = rng.choice(nodes, size=len(source), replace=True)
        edge_parts.append(np.column_stack((source, destination)))
        # Directed-ish neighboring region bridges create activity paths.
        target_nodes = np.flatnonzero(region_ids == (region + 1) % len(REGIONS))
        if len(target_nodes):
            edge_parts.append(
                np.column_stack(
                    (
                        rng.choice(nodes, size=max(20, len(nodes) // 18)),
                        rng.choice(target_nodes, size=max(20, len(nodes) // 18)),
                    )
                )
            )
    # Sparse axon-like long-range connections.
    edge_parts.append(rng.integers(0, n, size=(max(100, n // 30), 2)))
    edges = np.vstack(edge_parts).astype(np.int32, copy=False)
    return ConnectomeGraph(
        positions.astype(np.float32), region_ids.astype(np.int16), edges, REGIONS
    )
