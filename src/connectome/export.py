from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

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
