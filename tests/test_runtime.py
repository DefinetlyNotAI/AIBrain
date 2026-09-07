from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils import runtime


class RuntimePreflightTests(unittest.TestCase):
    def test_outside_managed_environment_explains_activation_when_venv_exists(
            self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / ".venv" / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.touch()
            with (
                patch.object(
                    runtime, "in_managed_virtual_environment", return_value=False
                ),
                patch.object(runtime, "error") as error,
            ):
                self.assertFalse(runtime.require_managed_runtime(root, "diagnostic"))

        self.assertIn("Activate", error.call_args.args[0])

    def test_missing_core_dependency_is_fatal_before_a_gui_can_start(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runtime, "in_managed_virtual_environment", return_value=True),
            patch("src.utils.runtime.importlib.util.find_spec", return_value=None),
            patch.object(runtime, "error") as error,
        ):
            self.assertFalse(runtime.require_managed_runtime(Path(directory), "main"))

        self.assertIn("llama-cpp-python", error.call_args.args[0])

    def test_compiled_distribution_skips_development_dependency_probe(self) -> None:
        with (
            patch.object(runtime, "_is_compiled_runtime", return_value=True),
            patch("src.utils.runtime.importlib.util.find_spec") as find_spec,
        ):
            self.assertTrue(runtime.require_managed_runtime(Path("dist"), "analysis"))

        find_spec.assert_not_called()

    def test_every_non_installer_cli_entry_uses_the_shared_preflight(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for path in sorted((root / "cli").glob("*.py")):
            if path.name in {"__init__.py", "installer.py"}:
                continue
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8-sig")
                self.assertIn("require_managed_runtime", source)


if __name__ == "__main__":
    unittest.main()
