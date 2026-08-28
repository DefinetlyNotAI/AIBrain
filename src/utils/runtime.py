"""Managed-runtime checks shared by every supported CLI entry point."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .console_ui import error, header, instruction_list


REQUIRED_MODULES = {
    "PySide6": "PySide6",
    "numpy": "NumPy",
    "moderngl": "ModernGL",
    "llama_cpp": "llama-cpp-python",
}


def in_managed_virtual_environment(root: Path) -> bool:
    """Return whether this process uses the project's managed virtual environment."""
    main_module = sys.modules.get("__main__")
    if main_module is not None and "__compiled__" in main_module.__dict__:
        return True
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        return False
    try:
        return Path(sys.prefix).resolve().is_relative_to((root / ".venv").resolve())
    except OSError:
        return False


def require_managed_runtime(root: Path, feature: str) -> bool:
    """Print actionable setup guidance and return false when startup must stop."""
    if not in_managed_virtual_environment(root):
        header("AIBrain", f"{feature.replace('_', ' ').title()} requires the managed runtime")
        venv_python = root / ".venv" / "Scripts" / "python.exe"
        if venv_python.is_file():
            error("Activate the managed virtual environment before running this command.")
            instruction_list(
                [("1.", "Activate:", r".\.venv\Scripts\Activate.ps1"), ("2.", "Run again:", f"python cli\\{feature}.py")],
                stream=sys.stderr,
            )
        else:
            error("The managed virtual environment has not been installed.")
            instruction_list(
                [("1.", "Create and install:", r"py cli\installer.py"), ("2.", "Activate:", r".\.venv\Scripts\Activate.ps1")],
                stream=sys.stderr,
            )
        return False
    missing = [name for module, name in REQUIRED_MODULES.items() if importlib.util.find_spec(module) is None]
    if missing:
        header("AIBrain", "Managed runtime is incomplete")
        error("Missing required dependencies: " + ", ".join(missing) + ".")
        instruction_list([("1.", "Repair the environment:", r"py cli\installer.py")], stream=sys.stderr)
        return False
    return True
