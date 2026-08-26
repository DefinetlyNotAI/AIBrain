from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.connectome.analysis import ConnectomeAnalyzer
from src.connectome.export import export_nn_analysis_plus
from src.connectome.generator import REGIONS
from src.connectome.graph import ConnectomeGraph
from src.models.instrumented_backend import ActivationFrame, ActivitySource


class NNAnalysisExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = ConnectomeGraph(
            positions=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
            regions=np.array([0, 1], dtype=np.int32),
            edges=np.array([[0, 1]], dtype=np.int32),
            region_names=REGIONS,
        )
        self.analyzer = ConnectomeAnalyzer(self.graph, hidden_width=4)
        self.analyzer.observe(ActivationFrame(1, "hello", 1, ActivitySource.SIMULATION),
                              np.array([.5, 0.], dtype=np.float32))

    def test_json_contains_compact_neural_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json"
            export_nn_analysis_plus(path, self.graph, self.analyzer, [{"role": "user", "content": "hi"}])
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], "aibrain.nn-analysis-plus.v2")
        self.assertEqual(payload["conversation"][0]["content"], "hi")
        self.assertEqual(payload["smart_analysis"]["frames_processed"], 1)
        self.assertIn("autoencoder", payload["smart_analysis"]["neural_network"]["architecture"])
        self.assertNotIn("brain_signals", payload)
        self.assertNotIn("positions", payload["graph"])

    def test_gzip_export_is_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json.gz"
            export_nn_analysis_plus(path, self.graph, self.analyzer, [])
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)

        self.assertEqual(payload["smart_analysis"]["key_events"][0]["step"], 1)

    def test_learned_model_persists_between_analysis_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learned.npz"
            first = ConnectomeAnalyzer(self.graph, hidden_width=4, model_path=path)
            first.observe(ActivationFrame(2, "learn", 2, ActivitySource.SIMULATION),
                          np.array([.3, .2], dtype=np.float32))
            first.save_model()
            second = ConnectomeAnalyzer(self.graph, hidden_width=4, model_path=path)

        self.assertEqual(second.frames_seen, 1)
        self.assertTrue(np.array_equal(second.encoder_weights, first.encoder_weights))


if __name__ == "__main__":
    unittest.main()
