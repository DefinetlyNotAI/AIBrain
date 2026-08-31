from __future__ import annotations

import numpy as np
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton

from src.app.analysis_window import (
    AnalysisInspectionWorker,
    AnalysisWindow,
    inspect_npz_model,
)


class AnalysisCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_analysis_tooltips_are_limited_to_controls_and_tab_labels(self) -> None:
        window = AnalysisWindow(auto_refresh=False)
        try:
            buttons = {
                button.text(): button for button in window.findChildren(QPushButton)
            }

            self.assertEqual(window.tabs.toolTip(), "")
            self.assertEqual(window.tabs.tabToolTip(0), "Model-health dashboard")
            self.assertEqual(
                window.tabs.tabToolTip(1),
                "Optional raw model metadata and export summaries",
            )
            self.assertEqual(
                buttons["Refresh model health"].toolTip(),
                "Re-read the persisted Analysis+ model and its rolling metrics",
            )
            self.assertEqual(
                buttons["Inspect exported analysis JSON"].toolTip(),
                "Open an explicit JSON or JSON.GZ export for read-only inspection",
            )
        finally:
            window.close()
            self.app.processEvents()

    def test_analysis_inspection_failures_are_logged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing_model = Path(directory) / "missing-analysis-model.npz"
            worker = AnalysisInspectionWorker(missing_model)

            with self.assertLogs("src.app.analysis_window", level="WARNING") as captured:
                worker.run()

        self.assertIn("Analysis model inspection failed", captured.output[0])
        self.assertIn(str(missing_model), captured.output[0])

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
                maturity_state=np.array("Teen"),
                weights_frozen=np.array(False),
            )
            metadata = inspect_npz_model(path)

        self.assertEqual(metadata["status"], "Healthy")
        self.assertEqual(metadata["learning"]["lifetime_frames_seen"], 7)
        self.assertEqual(metadata["learning"]["maturity"]["state"], "Teen")
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
