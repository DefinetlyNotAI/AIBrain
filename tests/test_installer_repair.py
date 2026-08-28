from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cli import installer


class InstallerRepairTests(unittest.TestCase):
    def test_final_health_covers_runtime_model_cache_and_dll_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch("cli.installer.OllamaDiagnostics") as diagnostics:
            root = Path(directory)
            python = root / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.touch()
            (root / ".cache").mkdir()
            (root / "native.dll").touch()
            diagnostics.return_value.inspect.return_value = []

            checks = installer.collect_final_health(str(python), None, root)

        by_name = {check.subsystem: check for check in checks}
        self.assertEqual(by_name["Hardware"].status, "CPU fallback")
        self.assertIn("REASON:", by_name["Hardware"].reason)
        self.assertEqual(by_name["Python"].status, "Ready")
        self.assertEqual(by_name["Model manifests / blobs"].status, "Ready")
        self.assertEqual(by_name["Native DLLs"].status, "Ready")

    def test_cache_repair_only_rebuilds_the_disposable_validation_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root / ".cache" / "validation"
            validation.mkdir(parents=True)
            (validation / "old.json").write_text("old", encoding="utf-8")
            sibling = root / ".cache" / "keep.txt"
            sibling.write_text("keep", encoding="utf-8")

            installer.repair_validation_cache(root)

            self.assertTrue(validation.is_dir())
            self.assertFalse((validation / "old.json").exists())
            self.assertEqual(sibling.read_text(encoding="utf-8"), "keep")

    def test_model_repair_requires_an_explicit_model_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "--model"):
            installer.repair_selected_subsystem("models", "python", None)

    def test_dependency_repair_is_scoped_to_the_selected_subsystem(self) -> None:
        with patch("cli.installer.install_dependencies") as install_dependencies:
            repaired = installer.repair_selected_subsystem("dependencies", "managed-python", None)

        self.assertEqual(repaired, "dependencies")
        install_dependencies.assert_called_once_with("managed-python")


if __name__ == "__main__":
    unittest.main()
