from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .graph import ConnectomeGraph

REGIONS = ("Input / tokens", "Embeddings", "Early processing", "Attention clusters", "MLP clusters",
           "Residual pathways", "Middle processing", "Late processing", "Output / logits")


def _seed(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")


def build_connectome(model_key: str, quality: str = "Medium", cluster_spacing: float = 1.0) -> ConnectomeGraph:
    counts = {"Low": 5000, "Medium": 11000, "High": 22000}
    n = counts.get(quality, 11000)
    rng = np.random.default_rng(_seed(model_key + quality))
    region_ids = rng.choice(len(REGIONS), size=n, p=np.array([.07, .10, .12, .14, .16, .10, .12, .12, .07]))
    centers = rng.normal(0, 4.5, size=(len(REGIONS), 3))
    # Curving global spine prevents a layered, row-of-circles appearance.
    phase = np.linspace(0, np.pi * 2.1, len(REGIONS))
    centers[:, 0] += np.cos(phase) * 8
    centers[:, 1] += np.sin(phase) * 5
    centers *= cluster_spacing
    positions = centers[region_ids] + rng.normal(0, 1.35, size=(n, 3))
    positions += rng.normal(0, .25, size=(n, 1)) * np.array([1, -.4, .6])
    edges_per_node = {"Low": 5, "Medium": 7, "High": 9}.get(quality, 7)
    edge_parts: list[np.ndarray] = []
    for region in range(len(REGIONS)):
        nodes = np.flatnonzero(region_ids == region)
        if len(nodes) < 2:
            continue
        source = rng.choice(nodes, size=len(nodes) * edges_per_node, replace=True)
        destination = rng.choice(nodes, size=len(source), replace=True)
        edge_parts.append(np.column_stack((source, destination)))
        # Directed-ish neighbouring region bridges create activity paths.
        target_nodes = np.flatnonzero(region_ids == (region + 1) % len(REGIONS))
        if len(target_nodes):
            edge_parts.append(np.column_stack((rng.choice(nodes, size=max(20, len(nodes)//18)),
                                               rng.choice(target_nodes, size=max(20, len(nodes)//18)))))
    # Sparse axon-like long-range connections.
    edge_parts.append(rng.integers(0, n, size=(max(100, n // 30), 2)))
    edges = np.vstack(edge_parts).astype(np.int32, copy=False)
    return ConnectomeGraph(positions.astype(np.float32), region_ids.astype(np.int16), edges, REGIONS)
