"""Interactive, installer-style test launcher for AIBrain."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.console_ui import (
    Color,
    clear_screen,
    command_output_box,
    command_preview,
    error,
    header,
    panel,
    section,
    status,
)

SUITES: dict[str, tuple[str, list[str]]] = {
    "1": (
        "Core model and native tests",
        [
            "tests.test_model_validator",
            "tests.test_model_diagnostics",
            "tests.test_llama_backend",
            "tests.test_native_wrapper",
        ],
    ),
    "2": (
        "Desktop UI and presentation tests",
        [
            "tests.test_chat_scroll",
            "tests.test_cluster_palette",
            "tests.test_cluster_spacing",
            "tests.test_console_ui",
            "tests.test_markdown",
        ],
    ),
    "3": (
        "CLI and application tests",
        [
            "tests.test_cli_main",
            "tests.test_cli_test",
            "tests.test_analysis_cli",
            "tests.test_gpu_launch",
            "tests.test_infinite_simulation",
            "tests.test_nn_analysis_export",
        ],
    ),
    "4": (
        "Build and packaging tests",
        [
            "tests.test_build_dist",
            "tests.test_packaged_utilities",
        ],
    ),
    "5": (
        "Infrastructure tests",
        [
            "tests.test_logging",
            "tests.test_source_encoding",
        ],
    ),
}

ALL_TESTS = "Every test in tests/"


def module_title(module: str) -> str:
    """Convert a test module name into a readable section title."""
    name = module.rsplit(".", 1)[-1]

    if name.startswith("test_"):
        name = name[5:]

    return name.replace("_", " ").title()


def run_command(command_line: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a unittest command and display its captured output."""
    command_preview(command_line)

    result = subprocess.run(
        command_line,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    output = result.stdout.rstrip()

    if output:
        command_output_box(output)

    return result


def run_suite(name: str, modules: list[str]) -> int:
    """Run a complete suite as one unittest invocation."""
    command_line = (
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-b",
        ]
        if not modules
        else [
            sys.executable,
            "-m",
            "unittest",
            "-b",
            *modules,
        ]
    )

    result = run_command(command_line)
    passed = result.returncode == 0

    status(
        "PASS" if passed else "FAIL",
        name,
        Color.GREEN if passed else Color.RED,
    )

    print()

    return result.returncode


def run_separated_suite(
        name: str,
        modules: list[str],
        start_section: int,
) -> int:
    """Run each module in a suite under its own console section."""
    failures = 0

    for offset, module in enumerate(modules):
        title = module_title(module)

        section(
            title,
            start_section + offset,
        )

        result = run_command(
            [
                sys.executable,
                "-m",
                "unittest",
                "-b",
                module,
            ]
        )

        passed = result.returncode == 0

        status(
            "PASS" if passed else "FAIL",
            title,
            Color.GREEN if passed else Color.RED,
        )

        print()

        if not passed:
            failures += 1

    status(
        "PASS" if failures == 0 else "FAIL",
        (
            f"{name} completed successfully"
            if failures == 0
            else f"{name} completed with {failures} failed module(s)"
        ),
        Color.GREEN if failures == 0 else Color.RED,
    )

    print()

    return 1 if failures else 0


def choose_suite() -> str | None:
    """Display the verification menu and return the selected suite."""
    panel(
        "TEST MENU",
        [
            (key, title)
            for key, (title, _modules) in SUITES.items()
        ]
        + [
            ("A", "Run every suite"),
            ("Q", "Exit"),
        ],
        footer="Choose a suite number, A, or Q and press Enter.",
    )

    try:
        choice = input("  Selection: ").strip().lower()
    except EOFError:
        error("No interactive input is available. Exiting the test launcher.")
        return None
    except KeyboardInterrupt:
        print()
        raise

    if choice == "q":
        return None

    if choice == "a":
        return "a"

    if choice not in SUITES:
        error("Unknown selection. Choose 1 to 5, A, or Q.")
        return choose_suite()

    return choice


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run AIBrain verification suites."
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every suite without showing the menu",
    )

    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        help="Run one numbered suite without showing the menu",
    )

    arguments = parser.parse_args()

    clear_screen()
    header(
        "AIBrain",
        "Verification and test centre",
    )

    section(
        "Select verification",
        1,
    )

    if arguments.all:
        choice = "a"
    elif arguments.suite:
        choice = arguments.suite
    else:
        choice = choose_suite()

    if choice is None:
        return 0

    if choice == "a":
        all_modules = [
            module
            for _name, modules in SUITES.values()
            for module in modules
        ]

        return run_separated_suite(
            ALL_TESTS,
            all_modules,
            start_section=2,
        )

    name, modules = SUITES[choice]

    return run_separated_suite(
        name,
        modules,
        start_section=2,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        error("Test run cancelled by keyboard interrupt.")
        raise SystemExit(130)
