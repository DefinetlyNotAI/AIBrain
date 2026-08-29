from __future__ import annotations

import numpy as np
import tempfile
import unittest
from pathlib import Path

from src.app.analysis_window import inspect_npz_model


class AnalysisCliTests(unittest.TestCase):
    def test_npz_inspector_reports_persisted_tensor_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.npz"
            np.savez(
                path,
                encoder_weights=np.ones((2, 3)),
                encoder_bias=np.ones(3),
                decoder_weights=np.ones((3, 2)),
                decoder_bias=np.ones(2),
                embedding_centroid=np.ones(3),
                frames_seen=np.array(7),
            )
            metadata = inspect_npz_model(path)

        self.assertEqual(metadata["status"], "Healthy")
        self.assertEqual(metadata["learning"]["lifetime_frames_seen"], 7)
        self.assertEqual(metadata["learning"]["maturity"]["state"], "Baby")
        self.assertFalse(metadata["learning"]["maturity"]["metrics_persisted"])
        self.assertEqual(metadata["architecture"]["shape"], "2 -> 3 -> 2")
        self.assertEqual(metadata["tensors"]["encoder_weights"]["shape"], [2, 3])
        self.assertEqual(metadata["health"]["score_percent"], 90)
        self.assertIn("No NaN", metadata["health_checks"]["finite_tensor_values"])

    def test_npz_health_score_detects_poisoned_non_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poisoned.npz"
            weights = np.ones((2, 3))
            weights[0, 0] = np.nan
            np.savez(
                path,
                encoder_weights=weights,
                encoder_bias=np.ones(3),
                decoder_weights=np.ones((3, 2)),
                decoder_bias=np.array([np.inf, 1.0]),
                embedding_centroid=np.ones(3),
                frames_seen=np.array(7),
            )

            metadata = inspect_npz_model(path)

        self.assertEqual(metadata["status"], "Critical")
        self.assertLess(metadata["health"]["score_percent"], 70)
        self.assertIn("Poisoned", metadata["health_checks"]["finite_tensor_values"])
        self.assertGreater(metadata["tensors"]["encoder_weights"]["poisoned_values"], 0)

    def test_npz_inspector_rejects_incompatible_tensor_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.npz"
            np.savez(
                path,
                encoder_weights=np.ones((2, 3)),
                encoder_bias=np.ones(2),
                decoder_weights=np.ones((3, 2)),
                decoder_bias=np.ones(2),
                embedding_centroid=np.ones(3),
                frames_seen=np.array(7),
            )
            with self.assertRaisesRegex(ValueError, "incompatible tensor shapes"):
                inspect_npz_model(path)


if __name__ == "__main__":
    unittest.main()
