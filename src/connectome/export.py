from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

from .analysis import ConnectomeAnalyzer, RecordSource
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
                  conversation: list[dict[str, object]],
                  records: RecordSource | None = None,
                  recorded_summary: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "schema": schema,
        "created_at": datetime.now(UTC).isoformat(),
        "conversation": conversation,
        "graph": {"nodes": len(graph.positions), "edges": len(graph.edges), "regions": graph.region_names,
                  "cluster_colours": {name: CLUSTER_COLOR_MAP[name] for name in graph.region_names}},
        "recorded_frame_summary": (
            analyzer.summary(records) if recorded_summary is None else recorded_summary
        ),
    }


def export_session_analysis(path: Path, graph: ConnectomeGraph, analyzer: ConnectomeAnalyzer,
                            conversation: list[dict[str, object]]) -> None:
    """Export normal-chat session data without autoencoder findings."""
    _write_payload(path, _base_payload("aibrain.session-analysis.v1", graph, analyzer, conversation))


def export_nn_analysis_plus(path: Path, graph: ConnectomeGraph, analyzer: ConnectomeAnalyzer,
                            conversation: list[dict[str, object]], *,
                            records: RecordSource | None = None,
    cache_status: dict[str, object] | None = None) -> None:
    """Export Infinite-mode session data with compact neural findings."""
    smart_analysis = analyzer.smart_report(records)
    findings = smart_analysis.get("session_findings")
    if isinstance(findings, dict):
        recorded_summary = {
            key: value
            for key, value in findings.items()
            if key not in {"primary_pattern", "pattern_segments"}
        }
    else:
        recorded_summary = {
            "frames": smart_analysis.get("frames_processed", 0),
            "maturity": smart_analysis.get("maturity", analyzer.maturity_report()),
        }
    payload = _base_payload(
        "aibrain.infinite-analysis-plus.v1",
        graph,
        analyzer,
        conversation,
        records,
        recorded_summary,
    )
    payload["integrity"] = "An online neural network analyzed every recorded simulated visual-connectome frame. " \
                           "Findings are not measured transformer activations."
    payload["smart_analysis"] = smart_analysis
    if cache_status is not None:
        payload["analysis_storage"] = cache_status
    _write_payload(path, payload)
