from __future__ import annotations

import inspect
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.models.diagnostics import OllamaDiagnostics
from src.app.analysis_window import AnalysisWindow
from src.app.diagnostics_window import DiagnosticsWindow, normalize_process_output
from src.models.model_info import ModelInfo


class ModelDiagnosticsTests(unittest.TestCase):
    def test_process_output_removes_terminal_control_sequences(self) -> None:
        rendered = normalize_process_output("writing manifest \x1b[K\rsuccess \x1b[K\x1b[?25h\x1b[?2026l")
        self.assertEqual(rendered, "writing manifest \nsuccess ")

    def test_inspection_windows_defer_close_until_their_worker_has_finished(self) -> None:
        diagnostics_close = inspect.getsource(DiagnosticsWindow.closeEvent)
        analysis_close = inspect.getsource(AnalysisWindow.closeEvent)

        self.assertIn("event.ignore()", diagnostics_close)
        self.assertNotIn(".wait(", diagnostics_close)
        self.assertIn("event.ignore()", analysis_close)
    def _write_manifest(self, root: Path, *, blob_data: bytes) -> Path:
        manifest = root / "manifests" / "registry.ollama.ai" / "library" / "demo" / "latest"
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
            invalid_manifest = root / "manifests" / "registry.ollama.ai" / "library" / "broken" / "latest"
            invalid_manifest.parent.mkdir(parents=True)
            invalid_manifest.write_text("not JSON", encoding="utf-8")

            diagnostics = OllamaDiagnostics(root).inspect()

        self.assertEqual([item.reference for item in diagnostics], ["broken:latest", "demo:latest"])
        self.assertFalse(diagnostics[0].available)
        self.assertIn("Invalid manifest", diagnostics[0].detail)
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

    @patch("src.models.diagnostics.ModelValidator.validate")
    def test_backend_diagnostics_report_a_model_that_cannot_load(self,
                                                                 validate) -> None:  # type: ignore[no-untyped-def]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_manifest(root, blob_data=b"GGUF" + struct.pack("<IQQ", 3, 1, 1))

            def failed_validation(models, *_args, **_kwargs):  # type: ignore[no-untyped-def]
                model = models[0]
                return [ModelInfo(model.name, model.tag, model.blob_path, available=False,
                                  error="llama.cpp rejected model")]

            validate.side_effect = failed_validation
            diagnostics = OllamaDiagnostics(root).inspect(verify_backend=True)

        self.assertFalse(diagnostics[0].available)
        self.assertEqual(diagnostics[0].detail, "llama.cpp rejected model")


if __name__ == "__main__":
    unittest.main()
