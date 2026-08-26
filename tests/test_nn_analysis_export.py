from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.connectome.analysis import AnalysisRecord
from src.connectome.export import export_nn_analysis_plus
from src.connectome.graph import ConnectomeGraph


class NNAnalysisExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = ConnectomeGraph(
            positions=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
            regions=np.array([0, 1], dtype=np.int32),
            edges=np.array([[0, 1]], dtype=np.int32),
            region_names=("Input", "Output"),
        )
        self.records = [AnalysisRecord(1, "hello", 1, .5, 1.0, "Input", .3)]
        self.signals = [{"frame": {"step": 1, "token_text": "hello"}, "values": [.5, 0.], "peaks": [.5, 0.],
                         "active_neuron_indices": [0], "regional_activity": {"Input": .5, "Output": 0.}}]

    def test_json_contains_full_conversation_and_brain_signals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json"
            export_nn_analysis_plus(path, self.graph, self.records, {"frames": 1}, [{"role": "user", "content": "hi"}], self.signals)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], "aibrain.nn-analysis-plus.v1")
        self.assertEqual(payload["conversation"][0]["content"], "hi")
        self.assertEqual(payload["brain_signals"], self.signals)
        self.assertEqual(payload["graph"]["positions"], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def test_gzip_export_is_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json.gz"
            export_nn_analysis_plus(path, self.graph, self.records, {"frames": 1}, [], self.signals)
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)

        self.assertEqual(payload["brain_signals"][0]["frame"]["step"], 1)


if __name__ == "__main__":
    unittest.main()
