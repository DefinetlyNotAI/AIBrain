from __future__ import annotations

import tempfile
import unittest
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from cli import installer
from src.utils.console_ui import Color, strip_ansi


class InstallerRepairTests(unittest.TestCase):
    def test_final_health_covers_runtime_model_cache_and_dll_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "cli.installer.OllamaDiagnostics"
        ) as diagnostics:
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

    def test_cache_repair_only_rebuilds_the_disposable_validation_directory(
        self,
    ) -> None:
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
            repaired = installer.repair_selected_subsystem(
                "dependencies", "managed-python", None
            )

        self.assertEqual(repaired, "dependencies")
        install_dependencies.assert_called_once_with("managed-python", None)

    def test_installer_uses_the_build_runner_for_live_command_output(self) -> None:
        command = ["managed-python", "-m", "pip", "install", "PySide6>=6.7,<7"]

        with patch("cli.installer.run_with_live_output") as run_live:
            installer.run(command)

        run_live.assert_called_once_with(command)

    def test_repair_reinstalls_dependencies_and_backend(self) -> None:
        with patch("cli.installer.run") as run_command:
            installer.install_dependencies("managed-python", None, force_reinstall=True)

        dependency_command = run_command.call_args_list[-1].args[0]
        self.assertIn("--upgrade", dependency_command)
        self.assertIn("--force-reinstall", dependency_command)

        with (
            patch("cli.installer.select_wheel", return_value=("cpu", "CPU")),
            patch("cli.installer.run") as run_command,
        ):
            installer.install_llama("managed-python", None, force_reinstall=True)

        backend_command = run_command.call_args.args[0]
        self.assertIn("--upgrade", backend_command)
        self.assertIn("--force-reinstall", backend_command)

    def test_repair_is_unavailable_before_the_first_install(self) -> None:
        with self.assertRaisesRegex(ValueError, "unavailable"):
            installer.select_install_action(
                runtime_exists=False,
                repair_requested=True,
            )

    def test_automatic_action_uses_install_without_a_runtime_and_repair_with_one(self) -> None:
        self.assertEqual(
            installer.select_install_action(runtime_exists=False, assume_yes=True),
            "install",
        )
        self.assertEqual(
            installer.select_install_action(runtime_exists=True, assume_yes=True),
            "repair",
        )

    def test_empty_interactive_action_writes_the_default_in_grey(self) -> None:
        output = StringIO()
        stdin = StringIO()
        stdin.isatty = lambda: True  # type: ignore[method-assign]
        with (
            patch.object(installer.sys, "stdin", stdin),
            patch("builtins.input", return_value=""),
            redirect_stdout(output),
        ):
            action = installer.select_install_action(runtime_exists=True)

        self.assertEqual(action, "repair")
        self.assertIn(Color.GRAY, output.getvalue())
        self.assertIn("R", strip_ansi(output.getvalue()))

    def test_action_flags_select_the_requested_mode(self) -> None:
        self.assertEqual(
            installer.select_install_action(runtime_exists=True, install_requested=True),
            "install",
        )
        self.assertEqual(
            installer.select_install_action(runtime_exists=True, repair_requested=True),
            "repair",
        )

    def test_install_and_repair_flags_are_mutually_exclusive(self) -> None:
        with patch.object(sys, "argv", ["installer.py", "--install", "--repair"]):
            with self.assertRaises(SystemExit):
                installer.parse_args()

    def test_automatic_install_and_repair_flags_are_accepted(self) -> None:
        for action in ("--install", "--repair"):
            with self.subTest(action=action), patch.object(
                sys, "argv", ["installer.py", "-y", action]
            ):
                parsed = installer.parse_args()

            self.assertTrue(parsed.yes)
            self.assertEqual(parsed.install, action == "--install")
            self.assertEqual(parsed.repair, action == "--repair")

    def test_cuda_umd_banner_is_detected(self) -> None:
        query = subprocess.CompletedProcess([], 0, "NVIDIA RTX, 610.88\n", "")
        status = subprocess.CompletedProcess([], 0, "CUDA UMD Version: 13.3", "")
        with patch("cli.installer.subprocess.run", side_effect=[query, status]):
            gpu = installer.detect_nvidia()

        self.assertIsNotNone(gpu)
        self.assertEqual(gpu.cuda_version, (13, 3))
        self.assertEqual(installer.numerical_package(gpu), "cupy-cuda13x[ctk]>=14,<15")

    @patch("cli.installer.available_wheel", return_value=True)
    def test_cuda_backend_is_the_noninteractive_default(self, _available) -> None:  # type: ignore[no-untyped-def]
        gpu = installer.GpuCapability("NVIDIA RTX", "610.88", (13, 3))

        wheel, description = installer.select_wheel(gpu)

        self.assertEqual(wheel, "cu132")
        self.assertIn("CUDA", description)

    def test_cpu_array_package_is_used_without_cuda(self) -> None:
        self.assertEqual(installer.numerical_package(None), "numpy>=1.26,<3")


if __name__ == "__main__":
    unittest.main()
