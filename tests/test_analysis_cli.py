from __future__ import annotations

import json
import numpy as np
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFrame, QPushButton

from src.app.analysis_window import (
    AnalysisInspectionResult,
    AnalysisInspectionWorker,
    AnalysisWindow,
    inspect_npz_contents,
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
                "Complete values persisted in the Analysis+ NPZ archive",
            )
            self.assertEqual(
                window.tabs.tabToolTip(2),
                "Complete contents of a selected JSON analysis export",
            )
            self.assertEqual(
                buttons["Refresh model health"].toolTip(),
                "Re-read the persisted Analysis+ model and its rolling metrics",
            )
            self.assertEqual(
                buttons["Inspect exported analysis JSON"].toolTip(),
                "Open an explicit JSON or JSON.GZ export for read-only inspection",
            )
            metric_cards = [
                card
                for card in window.findChildren(QFrame)
                if card.objectName() == "metricCard"
            ]
            self.assertEqual(len(metric_cards), 10)
            self.assertTrue(all(card.toolTip() for card in metric_cards))
            self.assertIn("Complete contents", window.raw.toolTip())
            self.assertIn("complete values", window.npz.toolTip())
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
        self.assertEqual(metadata["architecture"]["learned_parameters"], 17)
        self.assertEqual(metadata["tensors"]["encoder_weights"]["shape"], [2, 3])
        self.assertEqual(metadata["health"]["score_percent"], 90)
        self.assertIn("No NaN", metadata["health_checks"]["finite_tensor_values"])

    def test_npz_contents_formatter_includes_every_stored_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contents.npz"
            np.savez(
                path,
                matrix=np.array([[1.25, -2.5], [3.75, 4.0]], dtype="f4"),
                frames_seen=np.array(17, dtype="i8"),
                maturity_state=np.array("Teen"),
            )

            contents = inspect_npz_contents(path)

        self.assertIn("Entries: 3", contents)
        self.assertIn("[matrix]", contents)
        self.assertIn("dtype: float32", contents)
        self.assertIn("shape: (2, 2)", contents)
        self.assertIn("1.25", contents)
        self.assertIn("-2.5", contents)
        self.assertIn("[frames_seen]", contents)
        self.assertIn("17", contents)
        self.assertIn("[maturity_state]", contents)
        self.assertIn("Teen", contents)

    def test_npz_contents_closes_archive_before_formatting_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contents.npz"
            replacement = Path(directory) / "replacement.npz"
            np.savez(path, value=np.array([1.0]))
            np.savez(replacement, value=np.array([2.0]))
            original_formatter = np.array2string
            replaced = False

            def replace_while_formatting(value, **kwargs):  # type: ignore[no-untyped-def]
                nonlocal replaced
                if not replaced:
                    replacement.replace(path)
                    replaced = True
                return original_formatter(value, **kwargs)

            with patch(
                "src.app.analysis_window.np.array2string",
                side_effect=replace_while_formatting,
            ):
                contents = inspect_npz_contents(path)

            with np.load(path, allow_pickle=False) as stored:
                current = float(stored["value"][0])

        self.assertTrue(replaced)
        self.assertIn("1.", contents)
        self.assertEqual(current, 2.0)

    def test_completed_inspection_populates_npz_contents_tab(self) -> None:
        window = AnalysisWindow(auto_refresh=False)
        try:
            result = AnalysisInspectionResult(
                metadata={"status": "Healthy"},
                npz_contents="[frames_seen]\nvalues (1):\n17",
            )

            window._inspection_ready(result)

            self.assertEqual(window.npz.toPlainText(), result.npz_contents)
            self.assertIn('"status": "Healthy"', window.raw.toPlainText())
        finally:
            window.close()
            self.app.processEvents()

    def test_json_import_displays_complete_export_and_selects_its_tab(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json"
            payload = {
                "schema": "aibrain.infinite-analysis-plus.v1",
                "created_at": "2026-09-04T00:00:00+00:00",
                "conversation": [{"role": "world", "content": "Rain begins."}],
                "recorded_frame_summary": {"frames": 12},
                "smart_analysis": {
                    "frames_processed": 12,
                    "session_findings": {"novelty": .4},
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            window = AnalysisWindow(auto_refresh=False)
            try:
                with patch(
                    "src.app.analysis_window.QFileDialog.getOpenFileName",
                    return_value=(str(path), "Analysis data (*.json *.json.gz)"),
                ):
                    window.inspect_export()

                rendered = window.raw.toPlainText()
                self.assertIs(window.tabs.currentWidget(), window.raw)
                self.assertIn("Complete JSON contents", rendered)
                self.assertIn('"smart_analysis"', rendered)
                self.assertIn('"novelty": 0.4', rendered)
                self.assertIn('"has_nn_findings": true', rendered)
                self.assertIn(path.name, window.statusBar().currentMessage())
            finally:
                window.close()
                self.app.processEvents()

    def test_model_refresh_does_not_overwrite_an_imported_json_export(self) -> None:
        window = AnalysisWindow(auto_refresh=False)
        try:
            window._selected_export = Path("selected-analysis.json")
            window.raw.setPlainText("imported JSON remains visible")
            result = AnalysisInspectionResult(
                metadata={"status": "Healthy"},
                npz_contents="[frames_seen]\nvalues (1):\n17",
            )

            window._inspection_ready(result)

            self.assertEqual(window.raw.toPlainText(), "imported JSON remains visible")
            self.assertEqual(window.npz.toPlainText(), result.npz_contents)
        finally:
            window.close()
            self.app.processEvents()

    def test_invalid_json_import_shows_the_error_in_the_json_tab(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text("not JSON", encoding="utf-8")
            window = AnalysisWindow(auto_refresh=False)
            try:
                with patch(
                    "src.app.analysis_window.QFileDialog.getOpenFileName",
                    return_value=(str(path), "Analysis data (*.json *.json.gz)"),
                ):
                    window.inspect_export()

                self.assertIs(window.tabs.currentWidget(), window.raw)
                self.assertIn("Could not read analysis export", window.raw.toPlainText())
                self.assertIn("could not be loaded", window.statusBar().currentMessage())
            finally:
                window.close()
                self.app.processEvents()

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
