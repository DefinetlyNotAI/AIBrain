from __future__ import annotations

import inspect
import os
import unittest
from unittest.mock import Mock, patch

from cli import main
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

    @patch("cli.main.subprocess.Popen")
    def test_supervisor_waits_for_retry_then_returns_the_child_exit_code(self, popen: Mock) -> None:
        retrying_child = Mock()
        retrying_child.wait.return_value = GPU_RELAUNCH_EXIT_CODE
        completed_child = Mock()
        completed_child.wait.return_value = 0
        popen.side_effect = [retrying_child, completed_child]

        result = main._supervise_gpu_launch()

        self.assertEqual(result, 0)
        self.assertEqual(popen.call_count, 2)
        first_environment = popen.call_args_list[0].kwargs["env"]
        second_environment = popen.call_args_list[1].kwargs["env"]
        self.assertEqual(first_environment["AIBRAIN_GPU_SUPERVISOR"], "1")
        self.assertEqual(first_environment["AIBRAIN_GPU_RELAUNCH_ATTEMPT"], "0")
        self.assertEqual(second_environment["AIBRAIN_GPU_RELAUNCH_ATTEMPT"], "1")

    @patch("cli.main.subprocess.Popen")
    def test_supervisor_terminates_the_child_when_ctrl_c_interrupts_waiting(self, popen: Mock) -> None:
        child = Mock()
        child.wait.side_effect = [KeyboardInterrupt, 1]
        popen.return_value = child

        self.assertEqual(main._supervise_gpu_launch(), 130)
        child.terminate.assert_called_once_with()
        self.assertEqual(child.wait.call_count, 2)

    def test_renderer_does_not_own_the_gpu_restart_path(self) -> None:
        self.assertFalse(hasattr(ConnectomeRenderer, "_restart_for_gpu"))
