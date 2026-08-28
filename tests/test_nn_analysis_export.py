from __future__ import annotations

import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.connectome.analysis import ConnectomeAnalyzer
from src.connectome.export import export_nn_analysis_plus, export_session_analysis
from src.connectome.generator import REGIONS
from src.connectome.graph import ConnectomeGraph
from src.models.instrumented_backend import ActivationFrame, ActivitySource


class NNAnalysisExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = ConnectomeGraph(
            positions=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
            regions=np.array([0, 1], dtype=np.int16),
            edges=np.array([[0, 1]], dtype=np.int32),
            region_names=REGIONS,
        )
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.model_path = Path(self.temporary_directory.name) / "learned.npz"
        self.analyzer = ConnectomeAnalyzer(self.graph, hidden_width=4, model_path=self.model_path)
        self.analyzer.observe(ActivationFrame(1, "hello", 1, ActivitySource.SIMULATION),
                              np.array([.5, 0.], dtype=np.float32))

    def test_json_contains_compact_neural_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json"
            export_nn_analysis_plus(path, self.graph, self.analyzer, [{"role": "user", "content": "hi"}])
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], "aibrain.infinite-analysis-plus.v1")
        self.assertEqual(payload["conversation"][0]["content"], "hi")
        self.assertEqual(payload["smart_analysis"]["frames_processed"], 1)
        self.assertIn("autoencoder", payload["smart_analysis"]["neural_network"]["architecture"])
        self.assertNotIn("brain_signals", payload)
        self.assertNotIn("positions", payload["graph"])

    def test_normal_session_export_excludes_neural_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            export_session_analysis(path, self.graph, self.analyzer, [{"role": "user", "content": "hi"}])
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], "aibrain.session-analysis.v1")
        self.assertIn("recorded_frame_summary", payload)
        self.assertNotIn("smart_analysis", payload)
        self.assertNotIn("integrity", payload)

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
            initial_weights = first.encoder_weights.copy()
            first.observe(ActivationFrame(2, "learn", 2, ActivitySource.SIMULATION),
                          np.array([.3, .2], dtype=np.float32))
            self.assertTrue(path.is_file())
            self.assertFalse(np.array_equal(first.encoder_weights, initial_weights))
            second = ConnectomeAnalyzer(self.graph, hidden_width=4, model_path=path)

        self.assertEqual(second.frames_seen, 1)
        self.assertTrue(np.array_equal(second.encoder_weights, first.encoder_weights))

    def test_rolling_metrics_and_readiness_persist_with_the_model(self) -> None:
        report = self.analyzer.maturity_report()
        restored = ConnectomeAnalyzer(self.graph, hidden_width=4, model_path=self.model_path)

        self.assertEqual(report["state"], "Baby")
        self.assertEqual(report["readiness"], "caution")
        self.assertEqual(len(restored.reconstruction_history), 1)
        self.assertEqual(len(restored.novelty_history), 1)
        self.assertEqual(len(restored.update_magnitude_history), 1)

    def test_maturity_requires_consistency_and_the_adult_frame_floor(self) -> None:
        self.analyzer.reconstruction_history = [.01] * 96
        self.analyzer.novelty_history = [.03] * 96
        self.analyzer.update_magnitude_history = [.002] * 96
        self.analyzer.frames_seen = 250
        self.analyzer._advance_maturity()
        self.assertEqual(self.analyzer.maturity_state, "Teen")

        self.analyzer.frames_seen = 4095
        self.analyzer._advance_maturity()
        self.assertEqual(self.analyzer.maturity_state, "Teen")
        self.analyzer.frames_seen = 4096
        self.analyzer._advance_maturity()
        self.assertEqual(self.analyzer.maturity_state, "Adult")

    def test_elder_overfit_signal_freezes_training_weights(self) -> None:
        self.analyzer.maturity_state = "Adult"
        self.analyzer.frames_seen = 4096
        self.analyzer.reconstruction_history = [.01] * 48 + [.025] * 48
        self.analyzer.update_magnitude_history = [.002] * 96
        self.analyzer._advance_maturity()
        before = self.analyzer.encoder_weights.copy()
        self.analyzer.observe(ActivationFrame(3, "frozen", 3, ActivitySource.SIMULATION),
                              np.array([.4, .1], dtype=np.float32))

        self.assertEqual(self.analyzer.maturity_state, "Elder")
        self.assertTrue(self.analyzer.weights_frozen)
        self.assertTrue(np.array_equal(before, self.analyzer.encoder_weights))

    def test_legacy_npz_is_conservatively_migrated_as_baby(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "legacy.npz"
            np.savez_compressed(
                legacy,
                encoder_weights=self.analyzer.encoder_weights,
                encoder_bias=self.analyzer.encoder_bias,
                decoder_weights=self.analyzer.decoder_weights,
                decoder_bias=self.analyzer.decoder_bias,
                embedding_centroid=self.analyzer.embedding_centroid,
                frames_seen=np.array(9000),
            )
            restored = ConnectomeAnalyzer(self.graph, hidden_width=4, model_path=legacy)

        self.assertEqual(restored.frames_seen, 9000)
        self.assertEqual(restored.maturity_state, "Baby")
        self.assertFalse(restored.reconstruction_history)

    def test_default_memory_path_is_user_writable_not_the_application_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"LOCALAPPDATA": directory}):
            path = ConnectomeAnalyzer.default_model_path()

        self.assertEqual(path, Path(__file__).resolve().parents[1] / "models" / "aibrain.analyser.npz")


if __name__ == "__main__":
    unittest.main()
