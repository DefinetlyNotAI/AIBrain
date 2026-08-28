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
            np.savez(path, encoder_weights=np.ones((2, 3)), encoder_bias=np.ones(3), decoder_weights=np.ones((3, 2)),
                     decoder_bias=np.ones(2), embedding_centroid=np.ones(3), frames_seen=np.array(7))
            metadata = inspect_npz_model(path)

        self.assertEqual(metadata["status"], "Healthy")
        self.assertEqual(metadata["learning"]["lifetime_frames_seen"], 7)
        self.assertEqual(metadata["learning"]["maturity"], "Early learning")
        self.assertEqual(metadata["architecture"]["shape"], "2 -> 3 -> 2")
        self.assertEqual(metadata["tensors"]["encoder_weights"]["shape"], [2, 3])
        self.assertEqual(metadata["health_checks"]["finite_numeric_values"], "passed")

    def test_npz_inspector_rejects_incompatible_tensor_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.npz"
            np.savez(path, encoder_weights=np.ones((2, 3)), encoder_bias=np.ones(2), decoder_weights=np.ones((3, 2)),
                     decoder_bias=np.ones(2), embedding_centroid=np.ones(3), frames_seen=np.array(7))
            with self.assertRaisesRegex(ValueError, "incompatible tensor shapes"):
                inspect_npz_model(path)


if __name__ == "__main__":
    unittest.main()
