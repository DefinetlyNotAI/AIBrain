"""Interactive, installer-style test launcher for AIBrain."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.console_ui import Color, clear_screen, command_output_box, command_preview, error, header, panel, section, status


SUITES: dict[str, tuple[str, list[str]]] = {
    "1": ("Core model and native tests", ["tests.test_model_validator", "tests.test_model_diagnostics", "tests.test_llama_backend", "tests.test_native_wrapper"]),
    "2": ("Desktop UI tests", ["tests.test_chat_scroll", "tests.test_cluster_palette", "tests.test_cluster_spacing", "tests.test_console_ui"]),
    "3": ("Launch and generation tests", ["tests.test_cli_main", "tests.test_gpu_launch", "tests.test_infinite_simulation", "tests.test_markdown", "tests.test_nn_analysis_export"]),
}


def run_suite(name: str, modules: list[str]) -> int:
    command_line = [sys.executable, "-m", "unittest", *modules]
    command_preview(command_line)
    result = subprocess.run(command_line, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    command_output_box("\n".join(part.rstrip() for part in (result.stdout, result.stderr) if part.strip()))
    status("PASS" if result.returncode == 0 else "FAIL", name, Color.GREEN if result.returncode == 0 else Color.RED)
    return result.returncode


def choose_suite() -> list[tuple[str, list[str]]]:
    panel(
        "TEST MENU",
        [(key, title) for key, (title, _modules) in SUITES.items()] + [("A", "Run every suite"), ("Q", "Exit")],
        footer="Choose a suite number, A, or Q and press Enter.",
    )
    choice = input("  Selection: ").strip().lower()
    if choice == "q":
        return []
    if choice == "a":
        return [(title, modules) for title, modules in SUITES.values()]
    selected = SUITES.get(choice)
    if selected is None:
        error("Unknown selection. Choose 1, 2, 3, A, or Q.")
        return choose_suite()
    return [selected]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AIBrain verification suites.")
    parser.add_argument("--all", action="store_true", help="Run every suite without showing the menu")
    parser.add_argument("--suite", choices=sorted(SUITES), help="Run one numbered suite without showing the menu")
    arguments = parser.parse_args()
    clear_screen()
    header("AIBrain", "Verification and test centre")
    section("Select verification", 1)
    selections = (
        [(title, modules) for title, modules in SUITES.values()]
        if arguments.all
        else [SUITES[arguments.suite]] if arguments.suite else choose_suite()
    )
    if not selections:
        return 0
    section("Run tests", 2)
    return 1 if any(run_suite(title, modules) for title, modules in selections) else 0


if __name__ == "__main__":
    raise SystemExit(main())
