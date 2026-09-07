from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.models import llama_runtime


class LlamaRuntimeTests(unittest.TestCase):
    def test_discovers_nvidia_wheel_layouts_without_importing_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "site-packages"
            expected = [
                library / "nvidia" / "cublas" / "bin",
                library / "nvidia" / "cuda_runtime" / "bin",
                library / "nvidia" / "cu13" / "bin",
                library / "nvidia" / "cu13" / "bin" / "x86_64",
            ]
            for path in [*expected, library / "unrelated" / "bin"]:
                path.mkdir(parents=True, exist_ok=True)
                (path / "runtime.dll").touch()
            with (
                patch.object(
                    llama_runtime.sysconfig, "get_path", return_value=str(library)
                ),
                patch.object(llama_runtime.sys, "executable", str(root / "python.exe")),
            ):
                self.assertEqual(
                    llama_runtime.cuda_dll_directories(),
                    sorted(path.resolve() for path in expected),
                )

    def test_prepares_both_search_mechanisms_and_retains_handles(self) -> None:
        directory = Path("managed-nvidia-bin").resolve()
        handle = Mock()
        module = Mock()

        with (
            patch.object(llama_runtime.sys, "platform", "win32"),
            patch.object(
                llama_runtime, "cuda_dll_directories", return_value=[directory]
            ),
            patch.object(
                llama_runtime.os, "add_dll_directory", return_value=handle
            ) as register,
            patch.dict(llama_runtime._DLL_HANDLES, {}, clear=True),
            patch.dict(os.environ, {"PATH": "original-path"}),
            patch.dict(llama_runtime.sys.modules, {"llama_cpp": module}),
            patch.object(llama_runtime, "_LOG_CALLBACK", None),
        ):
            self.assertIs(llama_runtime.load_llama_cpp(), module)
            self.assertIs(llama_runtime.load_llama_cpp(), module)
            register.assert_called_once_with(str(directory))
            self.assertIs(llama_runtime._DLL_HANDLES[directory], handle)
            self.assertEqual(
                os.environ["PATH"], str(directory) + os.pathsep + "original-path"
            )
            handle.close.assert_not_called()

    def test_loader_installs_retained_logging_callback_before_runtime_probe(
        self,
    ) -> None:
        module = Mock()
        module.llama_log_callback.side_effect = lambda callback: callback
        with (
            patch.object(llama_runtime, "prepare_cuda_dll_search"),
            patch.dict(llama_runtime.sys.modules, {"llama_cpp": module}),
            patch.object(llama_runtime, "_LOG_CALLBACK", None),
        ):
            loaded = llama_runtime.load_llama_cpp()
            self.assertIs(llama_runtime._LOG_CALLBACK, llama_runtime._handle_native_log)
            loaded.llama_log_set.assert_called_once_with(
                llama_runtime._handle_native_log, None
            )
            llama_runtime.load_llama_cpp()
            module.llama_log_callback.assert_called_once()

    def test_native_levels_use_current_ggml_severities(self) -> None:
        with self.assertLogs(llama_runtime._LOG, level="DEBUG") as logs:
            llama_runtime._handle_native_log(
                2, b"ggml_cuda_init: found 1 CUDA devices\n", None
            )
            llama_runtime._handle_native_log(2, b"  Device 0: NVIDIA RTX\n", None)
            llama_runtime._handle_native_log(3, b"low memory\n", None)
            llama_runtime._handle_native_log(4, b"model load failed\n", None)
            llama_runtime._handle_native_log(1, b"debug details\n", None)
        self.assertEqual(
            [record.levelname for record in logs.records],
            ["INFO", "INFO", "WARNING", "ERROR", "DEBUG"],
        )
        self.assertTrue(
            all(
                record.getMessage().startswith("llama.cpp: ") for record in logs.records
            )
        )

    def test_routine_native_info_is_debug_and_continuations_keep_severity(self) -> None:
        with self.assertLogs(llama_runtime._LOG, level="DEBUG") as logs:
            llama_runtime._handle_native_log(2, b"print_info: model metadata\n", None)
            llama_runtime._handle_native_log(3, b"control token warning\n", None)
            llama_runtime._handle_native_log(5, b"warning continuation\n", None)

        self.assertEqual(
            [record.levelname for record in logs.records],
            ["DEBUG", "WARNING", "WARNING"],
        )

    def test_existing_path_entry_is_not_duplicated(self) -> None:
        directory = Path("managed-nvidia-bin").resolve()
        with (
            patch.object(llama_runtime.sys, "platform", "win32"),
            patch.object(
                llama_runtime, "cuda_dll_directories", return_value=[directory]
            ),
            patch.object(llama_runtime.os, "add_dll_directory"),
            patch.dict(llama_runtime._DLL_HANDLES, {}, clear=True),
            patch.dict(os.environ, {"PATH": str(directory)}),
        ):
            llama_runtime.prepare_cuda_dll_search()
            self.assertEqual(os.environ["PATH"], str(directory))

    def test_non_windows_import_does_not_change_dll_search_paths(self) -> None:
        with (
            patch.object(llama_runtime.sys, "platform", "linux"),
            patch.object(llama_runtime, "cuda_dll_directories") as discover,
        ):
            self.assertEqual(llama_runtime.prepare_cuda_dll_search(), [])
            discover.assert_not_called()


if __name__ == "__main__":
    unittest.main()
