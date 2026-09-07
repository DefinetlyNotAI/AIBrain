"""Regression checks for the desktop entry point."""

from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

from cli import main


class MainEntryPointTests(unittest.TestCase):
    def test_compiled_distribution_bypasses_development_venv_guard(self) -> None:
        with (
            patch.object(main, "__compiled__", True, create=True),
            patch.object(main.sys, "prefix", "system"),
            patch.object(main.sys, "base_prefix", "system"),
        ):
            main.require_virtual_environment()

    def test_startup_waits_for_the_worker_before_exiting_after_a_fatal_error(
            self,
    ) -> None:
        source = inspect.getsource(main.main)

        self.assertIn("except Exception:", source)
        self.assertIn("self._stop_startup(exit_code=1)", source)
        self.assertIn("self._models_finished", source)
        self.assertIn("startup_worker.finished.connect(startup_thread.quit)", source)
        self.assertIn(
            "startup_thread.finished.connect(startup_coordinator.startup_thread_finished)",
            source,
        )

    def test_keyboard_interrupt_closes_the_main_window_before_qt_exits(self) -> None:
        source = inspect.getsource(main.main)

        self.assertIn('window = getattr(app, "main_window", None)', source)
        self.assertIn("QTimer.singleShot(0, window.close)", source)
