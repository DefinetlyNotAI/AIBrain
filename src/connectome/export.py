from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

from .analysis import ConnectomeAnalyzer
from .generator import CLUSTER_COLOR_MAP
from .graph import ConnectomeGraph


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(encoded)
        return
    path.write_text(encoded, encoding="utf-8")


def _base_payload(schema: str, graph: ConnectomeGraph, analyzer: ConnectomeAnalyzer,
                  conversation: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": schema,
        "created_at": datetime.now(UTC).isoformat(),
        "conversation": conversation,
        "graph": {"nodes": len(graph.positions), "edges": len(graph.edges), "regions": graph.region_names,
                  "cluster_colours": {name: CLUSTER_COLOR_MAP[name] for name in graph.region_names}},
        "recorded_frame_summary": analyzer.summary(),
    }


def export_session_analysis(path: Path, graph: ConnectomeGraph, analyzer: ConnectomeAnalyzer,
                            conversation: list[dict[str, object]]) -> None:
    """Export normal-chat session data without autoencoder findings."""
    _write_payload(path, _base_payload("aibrain.session-analysis.v1", graph, analyzer, conversation))


def export_nn_analysis_plus(path: Path, graph: ConnectomeGraph, analyzer: ConnectomeAnalyzer,
                            conversation: list[dict[str, object]]) -> None:
    """Export Infinite-mode session data with compact neural findings."""
    payload = _base_payload("aibrain.infinite-analysis-plus.v1", graph, analyzer, conversation)
    payload["integrity"] = "An online neural network analyzed every recorded simulated visual-connectome frame. " \
                           "Findings are not measured transformer activations."
    payload["smart_analysis"] = analyzer.smart_report()
    _write_payload(path, payload)
