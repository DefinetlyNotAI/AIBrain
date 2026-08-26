from __future__ import annotations

import csv
import gzip
import json
from datetime import UTC, datetime
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .analysis import AnalysisRecord
from .graph import ConnectomeGraph


def export_analysis(path: Path, graph: ConnectomeGraph, records: list[AnalysisRecord], summary: dict[str, object]) -> None:
    if path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()) if records else ["step"])
            writer.writeheader()
            writer.writerows(asdict(record) for record in records)
        return
    payload = {"integrity": "Derived analysis of the visual connectome activity stream; not measured transformer activations.",
               "summary": summary, "graph": {"nodes": len(graph.positions), "edges": len(graph.edges), "regions": graph.region_names},
               "events": [asdict(record) for record in records]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_nn_analysis_plus(
    path: Path,
    graph: ConnectomeGraph,
    records: list[AnalysisRecord],
    summary: dict[str, object],
    conversation: list[dict[str, object]],
    brain_signals: list[dict[str, Any]],
) -> None:
    """Write the complete recorded dialogue and visual-connectome signal stream.

    This intentionally exports the full visual arrays for every captured token
    frame. They are procedural visualization signals, not hidden-state or
    measured transformer activation data.
    """
    payload = {
        "schema": "aibrain.nn-analysis-plus.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "integrity": "Conversation and complete visual-connectome signal capture. Brain signals are simulated visual activity, not measured transformer activations.",
        "summary": summary,
        "conversation": conversation,
        "graph": {
            "nodes": len(graph.positions),
            "edges": len(graph.edges),
            "regions": graph.region_names,
            "positions": graph.positions.astype("f4").tolist(),
            "node_regions": graph.regions.astype("i4").tolist(),
            "edges_indexed": graph.edges.astype("i4").tolist(),
        },
        "derived_analysis_events": [asdict(record) for record in records],
        "brain_signals": brain_signals,
    }
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(encoded)
        return
    path.write_text(encoded, encoding="utf-8")
