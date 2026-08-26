"""Regression checks for the desktop entry point."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from cli import main


class MainEntryPointTests(unittest.TestCase):
    def test_compiled_distribution_bypasses_development_venv_guard(self) -> None:
        with patch.object(main, "__compiled__", True, create=True), patch.object(main.sys, "prefix", "system"), patch.object(main.sys, "base_prefix", "system"):
            main.require_virtual_environment()

