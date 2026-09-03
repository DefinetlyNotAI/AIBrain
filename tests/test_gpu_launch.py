from __future__ import annotations

import inspect
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from cli import analysis, diagnostic, main
from src.utils import gpu
from src.connectome.renderer import ConnectomeRenderer
from src.utils.gpu import GPU_RELAUNCH_EXIT_CODE, can_request_gpu_relaunch


class GpuLaunchTests(unittest.TestCase):
    def test_startup_coordinator_uses_its_declared_constructor(self) -> None:
        source = inspect.getsource(main.main)

        self.assertIn("startup_coordinator = StartupCoordinator()", source)
        self.assertNotIn("startup_coordinator = StartupCoordinator(app)", source)

    def test_renderer_relaunch_is_limited_to_one_retry(self) -> None:
        with patch("src.utils.gpu.sys.platform", "win32"):
            with patch.dict(os.environ, {"AIBRAIN_GPU_RELAUNCH_ATTEMPT": "0"}, clear=False):
                self.assertTrue(can_request_gpu_relaunch())
            with patch.dict(os.environ, {"AIBRAIN_GPU_RELAUNCH_ATTEMPT": "1"}, clear=False):
                self.assertFalse(can_request_gpu_relaunch())

    @patch("src.utils.gpu.subprocess.Popen")
    def test_supervisor_waits_for_retry_then_returns_the_child_exit_code(self, popen: Mock) -> None:
        retrying_child = Mock()
        retrying_child.wait.return_value = GPU_RELAUNCH_EXIT_CODE
        completed_child = Mock()
        completed_child.wait.return_value = 0
        popen.side_effect = [retrying_child, completed_child]

        result = gpu.supervise_gpu_launch(["python.exe", "cli/diagnostic.py"])

        self.assertEqual(result, 0)
        self.assertEqual(popen.call_count, 2)
        first_environment = popen.call_args_list[0].kwargs["env"]
        second_environment = popen.call_args_list[1].kwargs["env"]
        self.assertEqual(first_environment["AIBRAIN_GPU_SUPERVISOR"], "1")
        self.assertEqual(first_environment["AIBRAIN_GPU_RELAUNCH_ATTEMPT"], "0")
        self.assertEqual(second_environment["AIBRAIN_GPU_RELAUNCH_ATTEMPT"], "1")

    @patch("src.utils.gpu.subprocess.Popen")
    def test_supervisor_terminates_the_child_when_ctrl_c_interrupts_waiting(self, popen: Mock) -> None:
        child = Mock()
        child.wait.side_effect = [KeyboardInterrupt, 1]
        popen.return_value = child

        self.assertEqual(gpu.supervise_gpu_launch(["python.exe", "cli/diagnostic.py"]), 130)
        child.terminate.assert_called_once_with()
        self.assertEqual(child.wait.call_count, 2)

    def test_venv_gpu_preferences_include_the_base_python_process_image(self) -> None:
        with (
            patch.object(gpu.sys, "executable", str(Path("env/Scripts/python.exe").resolve())),
            patch.object(gpu.sys, "_base_executable", str(Path("base/python.exe").resolve())),
            patch.object(gpu.sys, "argv", ["cli/diagnostic.py"]),
        ):
            hosts = gpu._gpu_host_executables()
        self.assertEqual(hosts, {
            str(Path("env/Scripts/python.exe").resolve()),
            str(Path("env/Scripts/pythonw.exe").resolve()),
            str(Path("base/python.exe").resolve()),
            str(Path("base/pythonw.exe").resolve()),
        })

    def test_packaged_gpu_preferences_only_register_the_packaged_executable(self) -> None:
        executable = str(Path("dist/diagnostic.exe").resolve())
        with (
            patch.object(gpu.sys, "executable", executable),
            patch.object(gpu.sys, "argv", [executable]),
        ):
            self.assertEqual(gpu._gpu_host_executables(), {executable})

    def test_every_desktop_launcher_returns_supervised_exit_before_creating_qt(self) -> None:
        for launcher in (main, diagnostic, analysis):
            with (
                self.subTest(launcher=launcher.__name__),
                patch.object(launcher, "configure_cli_logging", return_value=("log", "crash")),
                patch.object(launcher, "require_managed_runtime", return_value=True),
                patch.object(launcher, "prepare_gpu_launch", return_value=17),
                patch.object(launcher, "configure_opengl_surface") as configure,
            ):
                self.assertEqual(launcher.main(), 17)
                configure.assert_not_called()

    def test_gpu_preference_is_saved_before_child_launch_and_preserves_arguments(self) -> None:
        for compiled, expected in (
            (False, ["python.exe", "cli/diagnostic.py", "--example"]),
            (True, ["python.exe", "--example"]),
        ):
            with (
                self.subTest(compiled=compiled),
                patch.object(gpu.sys, "platform", "win32"),
                patch.object(gpu.sys, "executable", "python.exe"),
                patch.object(gpu.sys, "argv", ["cli/diagnostic.py", "--example"]),
                patch.dict(os.environ, {}, clear=True),
                patch.object(gpu, "should_prefer_high_performance_gpu", return_value=True),
                patch.object(gpu, "set_windows_gpu_preference", return_value=True) as save,
                patch.object(gpu, "supervise_gpu_launch") as supervise,
            ):
                def launch(command):
                    save.assert_called_once_with(True)
                    self.assertEqual(command, expected)
                    return 0
                supervise.side_effect = launch
                self.assertEqual(gpu.prepare_gpu_launch(compiled=compiled), 0)

    def test_system_default_and_supervised_children_do_not_relaunch(self) -> None:
        for prefer, supervised in ((False, False), (True, True)):
            environment = {"AIBRAIN_GPU_SUPERVISOR": "1"} if supervised else {}
            with (
                self.subTest(prefer=prefer),
                patch.object(gpu.sys, "platform", "win32"),
                patch.dict(os.environ, environment, clear=True),
                patch.object(gpu, "should_prefer_high_performance_gpu", return_value=prefer),
                patch.object(gpu, "set_windows_gpu_preference") as save,
                patch.object(gpu, "supervise_gpu_launch") as supervise,
            ):
                self.assertIsNone(gpu.prepare_gpu_launch())
                supervise.assert_not_called()
                self.assertEqual(save.call_count, int(prefer))

    def test_renderer_does_not_own_the_gpu_restart_path(self) -> None:
        self.assertFalse(hasattr(ConnectomeRenderer, "_restart_for_gpu"))

    def test_renderer_requests_a_clean_supervised_restart_for_a_first_gpu_mismatch(self) -> None:
        source = inspect.getsource(ConnectomeRenderer.initializeGL)

        self.assertIn("can_request_gpu_relaunch()", source)
        self.assertIn("gpuRestartRequested.emit", source)
        self.assertIn("restarting through the startup loader", source)
