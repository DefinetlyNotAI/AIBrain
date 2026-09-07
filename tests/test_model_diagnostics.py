from __future__ import annotations

import inspect
import json
import os
import struct
import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QByteArray, QProcess
from PySide6.QtWidgets import QApplication, QDialog, QPlainTextEdit, QPushButton

from src.app.analysis_window import AnalysisWindow
from src.app.diagnostics_window import (
    DiagnosticsWindow,
    DiagnosticsWorker,
    LiveOutputBuffer,
    normalize_process_output,
)
from src.app.loading_window import LoadingWindow
from src.models.diagnostics import ModelDiagnostic, OllamaDiagnostics
from src.models.model_info import ModelInfo


class ModelDiagnosticsTests(unittest.TestCase):
    app: ClassVar[QApplication]

    @classmethod
    def setUpClass(cls) -> None:
        application = QApplication.instance()
        cls.app = (
            application if isinstance(application, QApplication) else QApplication([])
        )

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

            broken_item = window.table.topLevelItem(0)
            healthy_item = window.table.topLevelItem(1)
            self.assertIsNotNone(broken_item)
            self.assertIsNotNone(healthy_item)
            assert broken_item is not None and healthy_item is not None
            self.assertEqual(broken_item.text(3), "Open")
            self.assertEqual(healthy_item.text(3), "N/A")
            self.assertIn(
                "Traceback",
                broken_item.data(3, window._TRACE_ROLE),
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

    def test_backend_repair_uses_the_current_noninteractive_installer_flags(
            self,
    ) -> None:
        source = inspect.getsource(DiagnosticsWindow.repair_selected)

        self.assertIn('"--repair",', source)
        self.assertIn('"--repair-subsystem",', source)
        self.assertIn('"backend",', source)
        self.assertIn('"-y",', source)
        self.assertIn('environment.insert("PYTHONUNBUFFERED", "1")', source)

    def test_valid_but_unsupported_model_cannot_be_repaired_or_removed(self) -> None:
        window = DiagnosticsWindow(auto_refresh=False)
        try:
            diagnostic = ModelDiagnostic(
                "qwen2.5vl:3b",
                Path("vision-manifest"),
                Path("vision-blob"),
                False,
                "qwen2.5vl:3b is a vision-capable Ollama model. AIBrain currently loads text-only GGUF models.",
            )
            window._diagnostics_ready([diagnostic])
            item = window.table.topLevelItem(0)
            self.assertIsNotNone(item)
            assert item is not None
            window.table.setCurrentItem(item)
            window._update_actions()

            self.assertEqual(item.text(1), "Unsupported")
            self.assertEqual(item.text(2), "Unsupported model type")
            self.assertFalse(diagnostic.can_repair)
            self.assertFalse(diagnostic.can_remove_manifest)
            self.assertFalse(window.repair_button.isEnabled())
            self.assertFalse(window.remove_button.isEnabled())
            self.assertIn("no automatic repair", window.repair_button.toolTip())
        finally:
            window.close()
            self.app.processEvents()

    def test_live_repair_output_resets_heartbeat_and_tracks_exact_status(self) -> None:
        class OutputProcess(QProcess):
            def readAllStandardOutput(self) -> QByteArray:
                return QByteArray(b"Downloading llama.cpp wheel 42%\n")

        window = DiagnosticsWindow(auto_refresh=False)
        try:
            window._repair_process = OutputProcess()
            with patch.object(window._repair_heartbeat, "start") as restart:
                window._append_repair_output()

            restart.assert_called_once_with()
            self.assertEqual(
                window._last_repair_status, "Downloading llama.cpp wheel 42%"
            )
            self.assertIn(
                "Downloading llama.cpp wheel 42%", window.output.toPlainText()
            )
        finally:
            window._repair_process = None
            window.close()
            self.app.processEvents()

    def test_hashing_progress_logs_only_the_start_and_completion(self) -> None:
        window = DiagnosticsWindow(auto_refresh=False)
        try:
            logged: list[tuple[str, str]] = []
            with patch.object(
                    window,
                    "_log",
                    side_effect=lambda level, message: logged.append((level, message)),
            ):
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
            self.assertIsNotNone(item)
            assert item is not None
            with patch("src.app.diagnostics_window.QDialog.open") as show_dialog:
                window._open_table_item(item, 3)

            show_dialog.assert_called_once()
            dialog = window.findChildren(QDialog)[-1]
            viewer = dialog.findChild(QPlainTextEdit)
            self.assertIsNotNone(viewer)
            assert viewer is not None
            self.assertTrue(viewer.isReadOnly())
            self.assertIn("unsupported architecture", viewer.toPlainText())
            report = viewer.toPlainText()
            self.assertLess(report.index("Explanation"), report.index("Trace dump"))
            self.assertIn("Model: broken:latest", report)
            self.assertIn("Manifest: broken-manifest", report)
            copy_button = dialog.findChild(QPushButton, "copyTraceButton")
            self.assertIsNotNone(copy_button)
            assert copy_button is not None
            copy_button.click()
            self.assertEqual(self.app.clipboard().text(), report)
        finally:
            window.close()
            self.app.processEvents()

    def test_failed_repair_start_releases_the_action_lock(self) -> None:
        class FailedProcess(QProcess):
            def errorString(self) -> str:
                return "The managed Python executable could not be started"

        window = DiagnosticsWindow(auto_refresh=False)
        try:
            window._repair_process = FailedProcess()
            window._repair_error(QProcess.ProcessError.FailedToStart)

            self.assertIsNone(window._repair_process)
        finally:
            window.close()
            self.app.processEvents()

    def test_trace_report_preserves_real_exception_after_explanation(self) -> None:
        import traceback

        try:
            raise ValueError("invalid model header")
        except ValueError:
            raw = traceback.format_exc().rstrip()
        diagnostic = ModelDiagnostic(
            "broken:latest",
            Path("manifest"),
            Path("model.gguf"),
            False,
            f"The model header could not be read.\n{raw}",
        )
        report = diagnostic.trace_report
        self.assertTrue(
            report.startswith("Explanation\nThe model header could not be read.")
        )
        self.assertLess(
            report.index("Trace dump"),
            report.index("Traceback (most recent call last):"),
        )
        self.assertTrue(report.endswith(raw))

    def test_manifest_parse_failure_keeps_python_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifests" / "broken"
            manifest.parent.mkdir()
            manifest.write_text("invalid json", encoding="utf-8")
            diagnostic = OllamaDiagnostics(root).inspect()[0]
        self.assertIn("JSONDecodeError", diagnostic.trace_report)
        self.assertIn("Traceback (most recent call last):", diagnostic.trace_report)

    def test_background_diagnostics_logs_start_and_completion(self) -> None:
        worker = DiagnosticsWorker()
        compute_results = []
        worker.compute_completed.connect(lambda *result: compute_results.append(result))
        with (
            patch(
                "src.app.diagnostics_window.OllamaDiagnostics.compute_health",
                return_value=("Ready", "CUDA device operation passed"),
            ),
            patch(
                "src.app.diagnostics_window.OllamaDiagnostics.inspect", return_value=[]
            ),
            self.assertLogs("src.app.diagnostics_window", level="INFO") as captured,
        ):
            worker.run()

        self.assertIn(
            "Starting background Ollama model diagnostics", captured.output[0]
        )
        self.assertIn("completed (0 models)", captured.output[1])
        self.assertEqual(compute_results, [("Ready", "CUDA device operation passed")])

    def test_compute_card_waits_for_runtime_probe_and_is_independent_of_opengl(
            self,
    ) -> None:
        window = DiagnosticsWindow(auto_refresh=False)
        try:
            with patch.object(OllamaDiagnostics, "subsystem_health", return_value={}):
                window._refresh_subsystem_cards()
            cuda_card = window.subsystem_cards["GPU / CUDA"]
            self.assertEqual(cuda_card[0].text(), "Checking")
            window._opengl_ready("Intel", "Iris Xe")
            self.assertEqual(cuda_card[0].text(), "Checking")
            window._compute_ready("Ready", "CUDA: NVIDIA RTX; device operation passed")
            self.assertEqual(cuda_card[0].text(), "Ready")
            self.assertIn("NVIDIA RTX", cuda_card[1].text())
            self.assertIn("Intel", window.subsystem_cards["OpenGL rendering"][1].text())
            self.assertRegex(
                window.output.toPlainText(), r"CUDA\s+Ready: CUDA: NVIDIA RTX"
            )
        finally:
            window.close()
            self.app.processEvents()

    def test_compute_health_reports_cuda_and_backend_failures_separately(self) -> None:
        cases = (
            (1, None, True, None, "Ready", "device operation passed"),
            (0, None, False, None, "CPU fallback", "No CUDA devices found"),
            (
                1,
                RuntimeError("device failed"),
                True,
                None,
                "CPU fallback",
                "device failed",
            ),
            (1, None, False, None, "CPU fallback", "CPU-only backend"),
            (1, None, True, OSError("missing DLL"), "Needs repair", "missing DLL"),
        )
        for (
                count,
                device_error,
                offload,
                load_error,
                expected_state,
                expected_detail,
        ) in cases:
            with self.subTest(state=expected_state, detail=expected_detail):
                cupy = Mock()
                cupy.cuda.runtime.getDeviceCount.return_value = count
                cupy.cuda.runtime.getDeviceProperties.return_value = {
                    "name": b"NVIDIA RTX"
                }
                cupy.ones.return_value.sum.return_value.item.return_value = 1
                cupy.ones.side_effect = device_error
                backend = Mock()
                backend.llama_supports_gpu_offload.return_value = offload
                with (
                    patch.dict(sys.modules, {"cupy": cupy}),
                    patch(
                        "src.models.llama_runtime.load_llama_cpp",
                        return_value=backend,
                        side_effect=load_error,
                    ),
                ):
                    state, detail = OllamaDiagnostics.compute_health()
                self.assertEqual(state, expected_state)
                self.assertIn(expected_detail, detail)
                if state == "Ready":
                    self.assertIn("NVIDIA RTX", detail)
                    self.assertIn("model inference not tested", detail)

    def test_missing_cupy_does_not_prevent_backend_diagnostics(self) -> None:
        backend = Mock()
        backend.llama_supports_gpu_offload.return_value = True
        with (
            patch.dict(sys.modules, {"cupy": None}),
            patch("src.models.llama_runtime.load_llama_cpp", return_value=backend),
        ):
            state, detail = OllamaDiagnostics.compute_health()
        self.assertEqual(state, "CPU fallback")
        self.assertIn("ModuleNotFoundError", detail)
        self.assertIn("GPU offload supported", detail)

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
            self.assertEqual(rebuilt, validation.resolve())
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
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {"LOCALAPPDATA": directory}),
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
            assert quarantined is not None
            self.assertFalse(blob.exists())
            self.assertEqual(quarantined.read_bytes(), b"broken")

    @patch("src.models.model_validator.ModelValidator.validate")
    def test_backend_diagnostics_report_a_model_that_cannot_load(
            self, validate: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_manifest(root, blob_data=b"GGUF" + struct.pack("<IQQ", 3, 1, 1))

            def failed_validation(
                    models: Sequence[ModelInfo], *_args: object, **_kwargs: object
            ) -> list[ModelInfo]:
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
