from __future__ import annotations

import inspect
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication, QDialog, QPlainTextEdit, QPushButton

from src.app.loading_window import LoadingWindow
from src.models.diagnostics import ModelDiagnostic, OllamaDiagnostics
from src.app.analysis_window import AnalysisWindow
from src.app.diagnostics_window import (
    DiagnosticsWindow,
    DiagnosticsWorker,
    LiveOutputBuffer,
    normalize_process_output,
)
from src.models.model_info import ModelInfo


class ModelDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_loader_is_compact_and_centred_instead_of_maximized(self) -> None:
        loader = LoadingWindow()
        try:
            loader.show_centered()
            self.app.processEvents()

            self.assertFalse(loader.isMaximized())
            self.assertEqual(loader.size().width(), 560)
            self.assertEqual(loader.size().height(), 270)
        finally:
            loader.finish()
            self.app.processEvents()

    def test_raw_error_trace_column_uses_open_or_not_available_actions(self) -> None:
        window = DiagnosticsWindow(auto_refresh=False)
        try:
            broken = ModelDiagnostic(
                "broken:latest",
                Path("broken-manifest"),
                None,
                False,
                "llama.cpp compatibility check failed\nTraceback: unsupported model",
            )
            healthy = ModelDiagnostic(
                "healthy:latest", Path("healthy-manifest"), None, True, "Healthy"
            )
            window._diagnostics_ready([broken, healthy])

            self.assertEqual(window.table.topLevelItem(0).text(3), "Open")
            self.assertEqual(window.table.topLevelItem(1).text(3), "N/A")
            self.assertIn(
                "Traceback",
                window.table.topLevelItem(0).data(3, window._TRACE_ROLE),
            )
        finally:
            window.close()
            self.app.processEvents()

    def test_process_output_removes_terminal_control_sequences(self) -> None:
        rendered = normalize_process_output(
            "writing manifest \x1b[K\rsuccess \x1b[K\x1b[?25h\x1b[?2026l"
        )
        self.assertEqual(rendered, "writing manifest \nsuccess ")

    def test_live_output_replaces_a_carriage_return_progress_line(self) -> None:
        buffer = LiveOutputBuffer()
        buffer.feed("pulling 10%\rpulling 20%\ncomplete\n")

        self.assertEqual(buffer.completed_lines, ["pulling 20%", "complete"])
        self.assertNotIn("10%", buffer.render())

    def test_inspection_windows_defer_close_until_their_worker_has_finished(
        self,
    ) -> None:
        diagnostics_close = inspect.getsource(DiagnosticsWindow.closeEvent)
        analysis_close = inspect.getsource(AnalysisWindow.closeEvent)

        self.assertIn("event.ignore()", diagnostics_close)
        self.assertNotIn(".wait(", diagnostics_close)
        self.assertIn("event.ignore()", analysis_close)

    def test_trace_and_manifest_columns_use_one_click_actions(self) -> None:
        source = inspect.getsource(DiagnosticsWindow._build)

        self.assertIn("itemClicked.connect(self._open_table_item)", source)
        self.assertNotIn("itemDoubleClicked", source)

    def test_backend_repair_uses_the_current_noninteractive_installer_flags(self) -> None:
        source = inspect.getsource(DiagnosticsWindow.repair_selected)

        self.assertIn('"--repair",', source)
        self.assertIn('"--repair-subsystem",', source)
        self.assertIn('"backend",', source)
        self.assertIn('"-y",', source)

    def test_hashing_progress_logs_only_the_start_and_completion(self) -> None:
        window = DiagnosticsWindow(auto_refresh=False)
        try:
            logged: list[tuple[str, str]] = []
            window._log = lambda level, message: logged.append((level, message))  # type: ignore[method-assign]

            window._diagnostics_progress(1, 1, "Hashing demo:latest (start)")
            window._diagnostics_progress(1, 1, "Hashing demo:latest (50%)")
            window._diagnostics_progress(1, 1, "Hashing demo:latest (complete)")

            self.assertEqual(
                logged,
                [
                    ("CHECK", "1/1 Hashing demo:latest (start)"),
                    ("CHECK", "1/1 Hashing demo:latest (complete)"),
                ],
            )
            self.assertIn("50%", window.output.toPlainText())
        finally:
            window.close()
            self.app.processEvents()

    def test_open_trace_action_shows_copyable_trace_text(self) -> None:
        window = DiagnosticsWindow(auto_refresh=False)
        try:
            window._diagnostics_ready(
                [
                    ModelDiagnostic(
                        "broken:latest",
                        Path("broken-manifest"),
                        None,
                        False,
                        "Traceback: unsupported architecture",
                    )
                ]
            )
            item = window.table.topLevelItem(0)
            with patch("src.app.diagnostics_window.QDialog.open") as show_dialog:
                window._open_table_item(item, 3)

            show_dialog.assert_called_once()
            dialog = window.findChildren(QDialog)[-1]
            viewer = dialog.findChild(QPlainTextEdit)
            self.assertIsNotNone(viewer)
            self.assertTrue(viewer.isReadOnly())
            self.assertIn("unsupported architecture", viewer.toPlainText())
            self.assertIsNotNone(dialog.findChild(QPushButton, "copyTraceButton"))
        finally:
            window.close()
            self.app.processEvents()

    def test_failed_repair_start_releases_the_action_lock(self) -> None:
        class FailedProcess:
            @staticmethod
            def errorString() -> str:
                return "The managed Python executable could not be started"

        window = DiagnosticsWindow(auto_refresh=False)
        try:
            window._repair_process = FailedProcess()  # type: ignore[assignment]
            window._repair_error(QProcess.ProcessError.FailedToStart)

            self.assertIsNone(window._repair_process)
        finally:
            window.close()
            self.app.processEvents()

    def test_background_diagnostics_logs_start_and_completion(self) -> None:
        worker = DiagnosticsWorker()
        with (
            patch("src.app.diagnostics_window.OllamaDiagnostics.inspect", return_value=[]),
            self.assertLogs("src.app.diagnostics_window", level="INFO") as captured,
        ):
            worker.run()

        self.assertIn("Starting background Ollama model diagnostics", captured.output[0])
        self.assertIn("completed (0 models)", captured.output[1])

    def _write_manifest(self, root: Path, *, blob_data: bytes) -> Path:
        manifest = (
            root / "manifests" / "registry.ollama.ai" / "library" / "demo" / "latest"
        )
        blob = root / "blobs" / "sha256-demo"
        manifest.parent.mkdir(parents=True)
        blob.parent.mkdir(parents=True)
        blob.write_bytes(blob_data)
        manifest.write_text(
            json.dumps(
                {
                    "layers": [
                        {
                            "digest": "sha256:demo",
                            "mediaType": "application/vnd.ollama.image.model",
                            "size": len(blob_data),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_inspect_reports_healthy_and_invalid_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_manifest(root, blob_data=b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            invalid_manifest = (
                root
                / "manifests"
                / "registry.ollama.ai"
                / "library"
                / "broken"
                / "latest"
            )
            invalid_manifest.parent.mkdir(parents=True)
            invalid_manifest.write_text("not JSON", encoding="utf-8")

            diagnostics = OllamaDiagnostics(root).inspect()

        self.assertEqual(
            [item.reference for item in diagnostics], ["broken:latest", "demo:latest"]
        )
        self.assertFalse(diagnostics[0].available)
        self.assertIn("Invalid manifest", diagnostics[0].detail)
        self.assertIn("Invalid manifest", diagnostics[0].reason)
        self.assertTrue(diagnostics[1].available)

    def test_remove_stale_manifest_leaves_shared_blob_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_manifest(root, blob_data=b"not a GGUF")
            blob = root / "blobs" / "sha256-demo"
            diagnostic = OllamaDiagnostics(root).inspect()[0]

            OllamaDiagnostics.remove_stale_manifest(diagnostic)

            self.assertFalse(manifest.exists())
            self.assertTrue(blob.exists())

    def test_validation_cache_repair_preserves_other_cache_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root / ".cache" / "validation"
            validation.mkdir(parents=True)
            (validation / "old.json").write_text("old", encoding="utf-8")
            sibling = root / ".cache" / "keep.txt"
            sibling.write_text("keep", encoding="utf-8")

            rebuilt = OllamaDiagnostics.invalidate_validation_cache(root)
            self.assertEqual(rebuilt, validation)
            self.assertTrue(sibling.exists())
            self.assertFalse((validation / "old.json").exists())

    def test_human_reason_is_distinct_from_the_raw_error(self) -> None:
        diagnostic = ModelDiagnostic(
            "demo:latest",
            Path("manifest"),
            Path("blob"),
            False,
            "llama.cpp compatibility check failed: unsupported architecture",
        )

        self.assertEqual(diagnostic.reason, "Backend incompatibility")
        self.assertNotEqual(diagnostic.reason, diagnostic.detail)

    def test_corrupt_blob_is_quarantined_before_redownload(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"LOCALAPPDATA": directory}
        ):
            root = Path(directory)
            manifest = root / "manifest"
            blob = root / "sha256-broken"
            manifest.write_text("{}", encoding="utf-8")
            blob.write_bytes(b"broken")
            diagnostic = ModelDiagnostic(
                "demo:latest", manifest, blob, False, "Invalid HASH"
            )

            quarantined = OllamaDiagnostics.quarantine_for_redownload(diagnostic)

            self.assertIsNotNone(quarantined)
            self.assertFalse(blob.exists())
            self.assertEqual(quarantined.read_bytes(), b"broken")

    @patch("src.models.diagnostics.ModelValidator.validate")
    def test_backend_diagnostics_report_a_model_that_cannot_load(
        self, validate
    ) -> None:  # type: ignore[no-untyped-def]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_manifest(root, blob_data=b"GGUF" + struct.pack("<IQQ", 3, 1, 1))

            def failed_validation(models, *_args, **_kwargs):  # type: ignore[no-untyped-def]
                model = models[0]
                return [
                    ModelInfo(
                        model.name,
                        model.tag,
                        model.blob_path,
                        available=False,
                        error="llama.cpp rejected model",
                    )
                ]

            validate.side_effect = failed_validation
            diagnostics = OllamaDiagnostics(root).inspect(verify_backend=True)

        self.assertFalse(diagnostics[0].available)
        self.assertEqual(diagnostics[0].detail, "llama.cpp rejected model")


if __name__ == "__main__":
    unittest.main()
