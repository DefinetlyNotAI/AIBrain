from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

from .analysis import ConnectomeAnalyzer
from .generator import CLUSTER_COLOR_MAP
from .graph import ConnectomeGraph


def export_nn_analysis_plus(path: Path, graph: ConnectomeGraph, analyzer: ConnectomeAnalyzer,
                            conversation: list[dict[str, object]]) -> None:
    """Export compact neural findings, never massive per-neuron frame dumps."""
    payload = {
        "schema": "aibrain.nn-analysis-plus.v2",
        "created_at": datetime.now(UTC).isoformat(),
        "integrity": "An online neural network analyzed every recorded simulated visual-connectome frame. "
                     "Findings are not measured transformer activations.",
        "conversation": conversation,
        "graph": {"nodes": len(graph.positions), "edges": len(graph.edges), "regions": graph.region_names,
                  "cluster_colours": {name: CLUSTER_COLOR_MAP[name] for name in graph.region_names}},
        "smart_analysis": analyzer.smart_report(),
    }
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(encoded)
        return
    path.write_text(encoded, encoding="utf-8")
