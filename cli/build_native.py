"""Build and verify AIBrain's native Windows connectome library.

Run from any directory with:

    python cli/build_native.py

The script discovers MinGW GCC, Clang, or MSVC, compiles an optimized x64 DLL,
verifies the resulting PE library and exported symbols, then commits the
generated DLL only when its contents have changed.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.console_ui import (
    Color as Colour,
)
from src.utils.console_ui import (
    CommandOutputBox,
    clear_screen,
    command_preview,
    error,
    header,
    panel,
    report_keyboard_interrupt,
    section,
    status,
)
from src.utils.logging import (
    configure_cli_logging,
    log_completed_command,
    report_exception,
)
from src.utils.runtime import require_managed_runtime

SOURCE = ROOT / "src" / "native" / "c" / "connectome_kernels.c"
OUTPUT = ROOT / "dll" / "aibrain.connectome.dll"
LIVE_HEARTBEAT_SECONDS = 1.0
CAPTURED_HEARTBEAT_SECONDS = 15.0

EXPECTED_EXPORTS = (
    "decay_and_count",
    "edge_activity",
    "region_activity",
)


@dataclass(frozen=True, slots=True)
class Compiler:
    path: Path
    family: str


def line(
        label: str,
        text: str,
        colour: str = Colour.CYAN,
) -> None:
    """Render a compact build status line."""
    status(label, text, colour)


def run_command(
        command_line: list[str],
        *,
        check: bool = False,
        show_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run an external command and render each output line as it arrives."""
    command_preview(command_line)
    process: subprocess.Popen[str] | None = None
    reader: threading.Thread | None = None
    captured: list[str] = []
    try:
        process = subprocess.Popen(
            command_line,
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Could not capture command output")
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            try:
                for output_line in process.stdout:
                    output_queue.put(output_line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(
            target=read_output,
            name="native-build-output-reader",
            daemon=True,
        )
        reader.start()
        with CommandOutputBox() as output_box:
            last_output = time.monotonic()
            last_heartbeat = last_output
            while True:
                try:
                    output_line = output_queue.get(timeout=0.1)
                except queue.Empty:
                    now = time.monotonic()
                    interval = (
                        LIVE_HEARTBEAT_SECONDS
                        if output_box.is_live
                        else CAPTURED_HEARTBEAT_SECONDS
                    )
                    if show_output and now - last_heartbeat >= interval:
                        elapsed = max(0, int(now - last_output))
                        output_box.write_progress(
                            "Compiler is still running without new output "
                            f"({elapsed}s since the last line)"
                        )
                        last_heartbeat = now
                    continue
                if output_line is None:
                    break
                captured.append(output_line)
                last_output = time.monotonic()
                last_heartbeat = last_output
                if show_output:
                    output_box.write(output_line)
        return_code = process.wait()
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        log_completed_command(
            command_line, "".join(captured).rstrip(), return_code=None, interrupted=True
        )
        raise
    finally:
        if reader is not None:
            reader.join(timeout=1.0)

    output = "".join(captured).rstrip()
    log_completed_command(command_line, output, return_code=return_code)

    if check and return_code:
        raise subprocess.CalledProcessError(
            return_code,
            command_line,
            output=output,
        )

    return subprocess.CompletedProcess(command_line, return_code, output, "")


def discover_compiler(explicit: str | None) -> Compiler:
    """Locate a supported Windows C compiler."""
    candidates = [
        explicit,
        os.environ.get("CC"),
        shutil.which("gcc"),
        shutil.which("clang"),
        shutil.which("cl"),
        r"C:\Program Files\msys64\mingw64\bin\gcc.exe",
    ]

    for candidate in candidates:
        if not candidate:
            continue

        path = Path(candidate)

        if not path.is_file():
            continue

        name = path.name.lower()
        family = "msvc" if name in {"cl", "cl.exe"} else "gnu"

        return Compiler(
            path=path,
            family=family,
        )

    raise RuntimeError(
        "No supported C compiler was found. "
        "Install Visual Studio Build Tools, LLVM, or MSYS2 MinGW-w64 GCC."
    )


def command_for(
        compiler: Compiler,
        debug: bool,
) -> list[str]:
    """Create the compiler command for the selected toolchain."""
    if compiler.family == "msvc":
        flags = [
            "/nologo",
            "/Bt+",
            "/std:c11",
            "/W4",
            "/LD",
            str(SOURCE),
            f"/Fe:{OUTPUT}",
        ]

        flags += (
            ["/Od", "/Zi"]
            if debug
            else ["/O2", "/DNDEBUG"]
        )

        return [
            str(compiler.path),
            *flags,
            "/link",
            "/VERBOSE",
        ]

    flags = [
        "-v",
        "-std=c11",
        "-shared",
        "-Wall",
        "-Wextra",
        "-o",
        str(OUTPUT),
        str(SOURCE),
    ]

    flags += (
        ["-O0", "-g"]
        if debug
        else ["-O3", "-DNDEBUG", "-march=native"]
    )

    return [
        str(compiler.path),
        *flags,
    ]


def compile_library(
        compiler: Compiler,
        debug: bool,
) -> None:
    """Compile the native connectome DLL."""
    result = run_command(
        command_for(compiler, debug),
    )

    if result.returncode:
        raise RuntimeError(
            f"Compiler exited with code {result.returncode}"
        )


def verify_library() -> None:
    """Verify the generated DLL and its required exports."""
    if not OUTPUT.is_file():
        raise RuntimeError(
            f"Compiler did not create {OUTPUT.name}"
        )

    size = OUTPUT.stat().st_size

    if size < 1024:
        raise RuntimeError(
            f"Generated DLL is unexpectedly small: {size:,} bytes"
        )

    with OUTPUT.open("rb") as library_file:
        signature = library_file.read(2)

    if signature != b"MZ":
        raise RuntimeError(
            "Generated output does not contain a valid Windows PE signature"
        )

    library = ctypes.WinDLL(str(OUTPUT))

    missing_exports: list[str] = []

    for symbol in EXPECTED_EXPORTS:
        try:
            getattr(library, symbol)
        except AttributeError:
            missing_exports.append(symbol)

    if missing_exports:
        raise RuntimeError(
            "Native library is missing required exports: "
            + ", ".join(missing_exports)
        )

    line(
        "VERIFY",
        (
            f"Loaded {OUTPUT.name} "
            f"({size:,} bytes) with "
            f"{len(EXPECTED_EXPORTS)} required exports"
        ),
        Colour.GREEN,
    )


def commit_regenerated_library() -> bool:
    """Commit the generated DLL only when its contents changed."""
    relative_output = OUTPUT.relative_to(ROOT)

    section("Update generated artifact", 2)

    run_command(
        [
            "git",
            "add",
            "--",
            str(relative_output),
        ],
        check=True,
    )

    diff_result = run_command(
        [
            "git",
            "diff",
            "--cached",
            "--quiet",
            "--",
            str(relative_output),
        ],
    )

    if diff_result.returncode == 0:
        line(
            "GIT",
            "Native DLL is unchanged. No generated-artifact commit is required.",
            Colour.GRAY,
        )
        return False

    if diff_result.returncode != 1:
        raise RuntimeError(
            "Git could not determine whether the generated DLL changed"
        )

    run_command(
        [
            "git",
            "commit",
            "-m",
            "build: regenerate native connectome DLL",
            "--",
            str(relative_output),
        ],
        check=True,
    )

    line(
        "GIT",
        "Committed regenerated native DLL",
        Colour.GREEN,
    )

    return True


def main() -> int:
    """Build, verify, and optionally commit the native library."""
    runtime_log, _ = configure_cli_logging("build_native")
    if not require_managed_runtime(ROOT, "build_native"):
        return 1
    parser = argparse.ArgumentParser(
        description="Compile AIBrain's optimized native connectome DLL."
    )

    parser.add_argument(
        "--compiler",
        help="Path to a specific C compiler executable",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Build an unoptimized debug DLL",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the existing compiled DLL before building",
    )

    arguments = parser.parse_args()

    clear_screen()

    header(
        "AIBrain",
        "Native connectome build tool - x64 Windows",
    )
    status("LOG", f"CLI output: {runtime_log}")

    section("Compile native acceleration", 1)

    if not SOURCE.is_file():
        line(
            "ERROR",
            f"Missing native source: {SOURCE}",
            Colour.RED,
        )
        return 1

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if arguments.clean and OUTPUT.exists():
        OUTPUT.unlink()

        line(
            "CLEAN",
            f"Removed existing {OUTPUT.name}",
            Colour.YELLOW,
        )

    try:
        compiler = discover_compiler(
            arguments.compiler,
        )

        line(
            "TOOLCHAIN",
            f"{compiler.family.upper()} - {compiler.path}",
            Colour.GREEN,
        )

        line(
            "PROFILE",
            "Debug" if arguments.debug else "Optimized release",
            Colour.CYAN,
        )

        compile_library(
            compiler,
            arguments.debug,
        )

        verify_library()

        committed = commit_regenerated_library()

    except (
            OSError,
            RuntimeError,
            subprocess.CalledProcessError,
    ) as exc:
        error(str(exc))
        return 1

    panel(
        "NATIVE BUILD COMPLETE",
        [
            (
                "Toolchain",
                compiler.family.upper(),
            ),
            (
                "Profile",
                "Debug" if arguments.debug else "Release",
            ),
            (
                "Library",
                f"{OUTPUT.name} ({OUTPUT.stat().st_size:,} bytes)",
            ),
            (
                "Exports",
                str(len(EXPECTED_EXPORTS)),
            ),
            (
                "Git",
                "Committed" if committed else "Unchanged",
            ),
        ],
        footer="Native acceleration is verified and ready for AIBrain.",
        tone=Colour.GREEN,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        report_keyboard_interrupt("the native build")
        raise SystemExit(130)
    except Exception as exc:
        report_exception("AIBrain native build failed", exc)
        raise SystemExit(1)
