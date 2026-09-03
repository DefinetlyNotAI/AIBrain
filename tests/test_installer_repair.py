from __future__ import annotations

import tempfile
import unittest
import subprocess
import sys
import json
import struct
import venv
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
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
            native_dll = root / "dll" / "aibrain.connectome.dll"
            native_dll.parent.mkdir()
            native_dll.touch()
            diagnostics.return_value.inspect.return_value = []

            checks = installer.collect_final_health(str(python), None, root)

        by_name = {check.subsystem: check for check in checks}
        self.assertEqual(by_name["Hardware"].status, "CPU fallback")
        self.assertIn("REASON:", by_name["Hardware"].reason)
        self.assertEqual(by_name["Python"].status, "Ready")
        self.assertEqual(by_name["Model manifests / blobs"].status, "Ready")
        self.assertEqual(by_name["Native DLLs"].status, "Present")
        self.assertEqual(by_name["pip / libraries"].status, "Not checked")

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
        install_dependencies.assert_called_once_with(
            "managed-python", None, force_reinstall=True
        )

    def test_installer_uses_the_build_runner_for_live_command_output(self) -> None:
        command = ["managed-python", "-m", "pip", "install", "PySide6>=6.7,<7"]

        with patch("cli.installer.run_with_live_output") as run_live:
            installer.run(command)

        run_live.assert_called_once_with(command)

    def test_repair_reinstalls_dependencies_and_backend(self) -> None:
        with (
            patch("cli.installer.run") as run_command,
            patch("cli.installer.ensure_pip"),
        ):
            installer.install_dependencies("managed-python", None, force_reinstall=True)

        dependency_command = run_command.call_args_list[-1].args[0]
        self.assertIn("--upgrade", dependency_command)
        self.assertIn("--force-reinstall", dependency_command)

        with (
            patch("cli.installer.select_wheel", return_value=("cpu", "CPU")),
            patch("cli.installer.run") as run_command,
            patch("cli.installer.probe_llama_runtime", return_value=(True, "")),
        ):
            installer.install_llama("managed-python", None)

        backend_command = run_command.call_args.args[0]
        self.assertIn("--upgrade", backend_command)
        self.assertIn("--force-reinstall", backend_command)
        self.assertIn("--no-cache-dir", backend_command)
        self.assertIn(installer.LLAMA_CPP_PYTHON_REQUIREMENT, backend_command)
        self.assertIn("--only-binary=llama-cpp-python", backend_command)

    def test_unloadable_cuda_wheel_is_replaced_with_a_verified_cpu_wheel(self) -> None:
        with (
            patch("cli.installer.select_wheel", return_value=("cu124", "CUDA")),
            patch("cli.installer.run") as run_command,
            patch(
                "cli.installer.probe_llama_runtime",
                side_effect=[(False, "missing CUDA dependency"), (True, "")],
            ),
        ):
            installed = installer.install_llama("managed-python", None)

        self.assertEqual(installed, "cpu")
        commands = [call.args[0] for call in run_command.call_args_list]
        self.assertEqual(len(commands), 3)
        self.assertIn("nvidia-cublas-cu12>=12,<13", commands[0])
        self.assertIn("/cu124", commands[1][commands[1].index("--extra-index-url") + 1])
        self.assertIn("/cpu", commands[2][commands[2].index("--extra-index-url") + 1])
        self.assertIn("--force-reinstall", commands[2])
        self.assertIn("--no-cache-dir", commands[2])

    def test_unloadable_cpu_wheel_reports_a_short_backend_error(self) -> None:
        with (
            patch("cli.installer.select_wheel", return_value=("cpu", "CPU")),
            patch("cli.installer.run"),
            patch(
                "cli.installer.probe_llama_runtime",
                return_value=(False, "RuntimeError: missing llama.dll"),
            ),
        ):
            with self.assertRaisesRegex(installer.LlamaRuntimeError, "missing llama.dll"):
                installer.install_llama("managed-python", None)

    def test_repair_is_unavailable_before_the_first_install(self) -> None:
        with self.assertRaisesRegex(ValueError, "unavailable"):
            installer.select_install_action(
                runtime_exists=False,
                repair_requested=True,
            )

    def test_automatic_action_defaults_to_install_even_with_an_existing_runtime(self) -> None:
        self.assertEqual(
            installer.select_install_action(runtime_exists=False, assume_yes=True),
            "install",
        )
        self.assertEqual(
            installer.select_install_action(runtime_exists=True, assume_yes=True),
            "install",
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

        self.assertEqual(action, "install")
        self.assertIn(Color.GRAY, output.getvalue())
        self.assertIn("Install", strip_ansi(output.getvalue()))

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

    def test_scoped_backend_repair_flag_is_accepted(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["installer.py", "--repair", "--repair-subsystem", "backend", "-y"],
        ):
            parsed = installer.parse_args()

        self.assertEqual(parsed.repair_subsystem, "backend")

    def test_scoped_backend_repair_requires_explicit_repair_authorization(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["installer.py", "--repair-subsystem", "backend"],
        ), self.assertRaises(SystemExit):
            installer.parse_args()

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

    def test_cuda_runtime_dependencies_match_the_selected_wheel(self) -> None:
        self.assertEqual(installer.cuda_runtime_packages("cpu"), ())
        for tag, expected in (
            ("cu132", "nvidia-cublas>=13,<14"),
            ("cu124", "nvidia-cublas-cu12>=12,<13"),
            ("cu118", "nvidia-cublas-cu11>=11,<12"),
        ):
            with self.subTest(tag=tag):
                packages = installer.cuda_runtime_packages(tag)
                self.assertEqual(packages[0], expected)
                self.assertIn("cuda-runtime", packages[1])

    def test_explicit_cuda_selection_does_not_silently_choose_cpu(self) -> None:
        with self.assertRaisesRegex(installer.LlamaRuntimeError, "CUDA was requested"):
            installer.select_wheel(None, preference="cuda")

        for failure in ("installation", "load"):
            with (
                self.subTest(failure=failure),
                patch("cli.installer.select_wheel", return_value=("cu132", "CUDA")),
                patch("cli.installer.run") as run_command,
                patch("cli.installer.probe_llama_runtime", return_value=(False, "missing DLL")),
                patch("cli.installer.install_cpu_fallback") as fallback,
            ):
                if failure == "installation":
                    run_command.side_effect = subprocess.CalledProcessError(1, ["pip"])
                with self.assertRaises(installer.LlamaRuntimeError):
                    installer.install_llama("managed-python", None, preference="cuda")
                fallback.assert_not_called()


class InstallerBootstrapTests(unittest.TestCase):
    def test_invalid_distribution_cleanup_preserves_normal_libraries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / ".venv"
            library = runtime / "Lib" / "site-packages"
            library.mkdir(parents=True)
            for name in ("~umpy", "~umpy.libs", "~umpy-2.4.6.dist-info", "numpy", "numpy.libs"):
                (library / name).mkdir()
                (library / name / "keep.dat").write_text("contents", encoding="utf-8")
            (library / "~broken-file").write_text("leftover", encoding="utf-8")
            (runtime / "~outside-site-packages").write_text("keep", encoding="utf-8")
            with patch.object(installer.sys, "platform", "win32"):
                removed = installer.cleanup_invalid_distributions(runtime)
                self.assertEqual(installer.cleanup_invalid_distributions(runtime), [])
            self.assertEqual(len(removed), 4)
            self.assertEqual(sorted(path.name for path in library.iterdir()), ["numpy", "numpy.libs"])
            self.assertTrue((library / "numpy" / "keep.dat").is_file())
            self.assertTrue((runtime / "~outside-site-packages").is_file())

    @unittest.skipUnless(sys.platform == "win32", "Windows junction protection")
    def test_invalid_distribution_cleanup_refuses_a_junction_to_other_data(self) -> None:
        import _winapi

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / ".venv"
            library = runtime / "Lib" / "site-packages"
            library.mkdir(parents=True)
            outside = root / "important"
            outside.mkdir()
            (outside / "keep.txt").write_text("keep", encoding="utf-8")
            _winapi.CreateJunction(str(outside), str(library / "~linked"))
            with self.assertRaisesRegex(ValueError, "linked distribution"):
                installer.cleanup_invalid_distributions(runtime)
            self.assertTrue((outside / "keep.txt").is_file())

    def test_help_needs_no_site_packages_and_does_not_initialize_logs(self) -> None:
        script = (
            "from unittest.mock import patch; from cli import installer; "
            "import sys; sys.argv = ['installer.py', '--help']; "
            "patch('cli.installer.configure_cli_logging', "
            "side_effect=AssertionError('help must not modify logs')).start(); "
            "installer.main()"
        )
        with tempfile.TemporaryDirectory() as directory:
            # Exercise direct invocation from a different working directory too.
            for command, cwd in (
                ([sys.executable, "-S", "-c", script], installer.ROOT),
                ([sys.executable, "-S", str(installer.ROOT / "cli" / "installer.py"),
                  "--help"], Path(directory)),
            ):
                with self.subTest(command=command):
                    result = subprocess.run(
                        command, cwd=cwd, capture_output=True, text=True, timeout=30
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("AIBrain Installer", result.stdout)

    def test_structural_model_health_needs_no_qt_even_with_installed_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = root / "models"
            manifest = model_root / "manifests" / "registry.ollama.ai" / "library" / "demo" / "latest"
            manifest.parent.mkdir(parents=True)
            blob = model_root / "blobs" / "sha256-demo"
            blob.parent.mkdir()
            data = b"GGUF" + struct.pack("<IQQ", 3, 1, 1)
            blob.write_bytes(data)
            manifest.write_text(json.dumps({"layers": [{
                "digest": "sha256:demo", "mediaType": "application/vnd.ollama.image.model",
                "size": len(data),
            }]}), encoding="utf-8")
            broken = manifest.parent.parent / "broken" / "latest"
            broken.parent.mkdir()
            broken.write_text("not JSON", encoding="utf-8")
            script = (
                "import sys; from pathlib import Path; from unittest.mock import patch; "
                "from cli import installer; from src.models.diagnostics import OllamaDiagnostics; "
                "diagnostics = OllamaDiagnostics(Path(sys.argv[1])); "
                "patch('cli.installer.OllamaDiagnostics', return_value=diagnostics).start(); "
                "checks = installer.collect_final_health(sys.executable, None, Path(sys.argv[2])); "
                "models = next(c for c in checks if c.subsystem == 'Model manifests / blobs'); "
                "assert models.status == 'Needs repair', models; "
                "assert 'Invalid manifest' in models.reason; "
                "assert 'src.models.model_validator' not in sys.modules; "
                "assert not any(m.startswith('PySide6') for m in sys.modules)"
            )
            result = subprocess.run(
                [sys.executable, "-S", "-c", script, str(model_root), str(root)],
                cwd=installer.ROOT, capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_empty_venv_starts_installer_and_recovers_pip_offline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aibrain-bootstrap-") as directory:
            runtime = Path(directory) / ".venv"
            venv.EnvBuilder(with_pip=False).create(runtime)
            python = runtime / "Scripts" / "python.exe"
            result = subprocess.run(
                [str(python), str(installer.ROOT / "cli" / "installer.py"), "--help"],
                cwd=installer.ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            with patch.object(installer, "VENV_DIR", runtime):
                installer.verify_managed_python(str(python))
            installer.ensure_pip(str(python))
            result = subprocess.run(
                [str(python), "-m", "pip", "--version"],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(runtime).lower(), result.stdout.lower())
            # Adding pip must not install any application packages as a side effect.
            result = subprocess.run(
                [str(python), "-c",
                 "import importlib.util; assert importlib.util.find_spec('PySide6') is None"],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_pip_is_not_rebootstrapped_when_already_available(self) -> None:
        with (
            patch("cli.installer.subprocess.run") as probe,
            patch("cli.installer.run") as command,
        ):
            installer.ensure_pip("managed-python")
        self.assertEqual(probe.call_args.args[0][0], "managed-python")
        command.assert_not_called()

    def test_managed_interpreter_check_rejects_a_different_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            installer, "VENV_DIR", Path(directory) / ".venv"
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                installer.verify_managed_python(sys.executable)

    def test_cuda_probe_rejects_a_cpu_wheel_that_imports_successfully(self) -> None:
        def probe(command, **_kwargs):
            result = subprocess.CompletedProcess(command, 0, "", "")
            output = StringIO()
            fake_backend = SimpleNamespace(llama_supports_gpu_offload=lambda: False)
            with (
                redirect_stdout(output),
                patch.dict(sys.modules, {"llama_cpp": fake_backend}),
            ):
                try:
                    exec(command[2], {})
                except SystemExit as exc:
                    result.returncode = exc.code
            result.stdout = output.getvalue()
            return result

        with patch("cli.installer.subprocess.run", side_effect=probe):
            ready, reason = installer.probe_llama_runtime("managed-python", require_cuda=True)
            cpu_ready, _ = installer.probe_llama_runtime("managed-python")
        self.assertFalse(ready)
        self.assertIn("does not support GPU offload", reason)
        self.assertTrue(cpu_ready)

    def test_verification_checks_native_backend_and_dependency_consistency(self) -> None:
        with patch("cli.installer.run") as command:
            installer.verify_installation("managed-python")
        commands = [call.args[0] for call in command.call_args_list]
        self.assertIn("llama_cpp", commands[0][2])
        self.assertEqual(commands[-1], ["managed-python", "-m", "pip", "check"])

    def test_health_does_not_treat_an_unrelated_dll_as_connectome_acceleration(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "cli.installer.OllamaDiagnostics"
        ) as diagnostics:
            root = Path(directory)
            (root / "unrelated.dll").touch()
            diagnostics.return_value.inspect.return_value = []
            checks = installer.collect_final_health("missing-python", None, root)
        by_name = {check.subsystem: check for check in checks}
        self.assertEqual(by_name["Native DLLs"].status, "Info")
        self.assertEqual(by_name["pip / libraries"].status, "Not checked")

    def test_cuda_11_array_cpu_fallback_is_expected_not_a_repair_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "cli.installer.OllamaDiagnostics"
        ) as diagnostics:
            diagnostics.return_value.inspect.return_value = []
            checks = installer.collect_final_health(
                sys.executable, installer.GpuCapability("NVIDIA", "driver", (11, 8)),
                Path(directory),
            )
        cuda = next(check for check in checks if check.subsystem == "CUDA")
        self.assertEqual(cuda.status, "CPU fallback")

    def test_targeted_repair_honors_backend_and_clears_cache_only_after_success(self) -> None:
        for failure in (False, True):
            with (
                self.subTest(failure=failure),
                patch("cli.installer.ensure_pip"),
                patch("cli.installer.install_llama", return_value="cpu") as install,
                patch("cli.installer.repair_validation_cache") as cache,
            ):
                if failure:
                    install.side_effect = installer.LlamaRuntimeError("broken backend")
                    with self.assertRaises(installer.LlamaRuntimeError):
                        installer.repair_selected_subsystem(
                            "backend", "managed-python", None, preference="cpu"
                        )
                    cache.assert_not_called()
                else:
                    self.assertEqual(installer.repair_selected_subsystem(
                        "backend", "managed-python", None, preference="cpu"
                    ), "cpu")
                    install.assert_called_once_with("managed-python", None, preference="cpu")
                    cache.assert_called_once_with()

    def test_install_and_repair_refresh_cache_only_after_verification(self) -> None:
        cases = (("install", False), ("install", True), ("repair", False), ("repair", True))
        for action, failure in cases:
            with self.subTest(action=action, failure=failure), ExitStack() as stack:
                stack.enter_context(patch.object(sys, "argv", ["installer.py", f"--{action}", "-y"]))
                stack.enter_context(patch("cli.installer.configure_cli_logging", return_value=(Path("log"), None)))
                stack.enter_context(patch("cli.installer.venv_python", return_value=Path(sys.executable)))
                for name in (
                    "clear_screen", "header", "create_environment",
                    "verify_managed_python", "install_dependencies",
                    "cleanup_invalid_distributions",
                ):
                    stack.enter_context(patch(f"cli.installer.{name}"))
                stack.enter_context(patch("cli.installer.verify_python", return_value=True))
                stack.enter_context(patch("cli.installer.detect_nvidia", return_value=None))
                stack.enter_context(patch("cli.installer.install_llama", return_value="cpu"))
                verification = stack.enter_context(patch("cli.installer.verify_installation"))
                if failure:
                    verification.side_effect = subprocess.CalledProcessError(1, ["verify"])
                cache = stack.enter_context(patch("cli.installer.repair_validation_cache"))
                health = stack.enter_context(patch("cli.installer.print_final_health"))
                complete = stack.enter_context(patch("cli.installer.completion_screen"))

                self.assertEqual(installer.main(), 1 if failure else 0)
                if failure:
                    cache.assert_not_called()
                    health.assert_not_called()
                    complete.assert_not_called()
                else:
                    cache.assert_called_once_with()
                    self.assertTrue(health.call_args.kwargs["libraries_verified"])
                    complete.assert_called_once()


if __name__ == "__main__":
    unittest.main()
