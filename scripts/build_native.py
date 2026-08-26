"""Build and verify AIBrain's native Windows connectome library.

Run from any directory with: ``py -3.11 scripts/build_native.py``.
The script discovers MinGW GCC, Clang, or MSVC, compiles an optimized x64 DLL,
and verifies it can be loaded before reporting success.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "native" / "connectome_native.c"
OUTPUT = ROOT / "src" / "native" / "aibrain_connectome.dll"


class Colour:
    RESET = "\x1b[0m"; BOLD = "\x1b[1m"; CYAN = "\x1b[36m"; GREEN = "\x1b[32m"; YELLOW = "\x1b[33m"; RED = "\x1b[31m"


def enable_colour() -> bool:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return False
    if os.name == "nt":
        try:
            import ctypes as _ctypes
            _ctypes.windll.kernel32.SetConsoleMode(_ctypes.windll.kernel32.GetStdHandle(-11), 7)
        except (AttributeError, OSError):
            pass
    return True


@dataclass(frozen=True, slots=True)
class Compiler:
    path: Path
    family: str


def discover_compiler(explicit: str | None) -> Compiler:
    candidates = [explicit] if explicit else []
    candidates += [os.environ.get("CC"), shutil.which("gcc"), shutil.which("clang"), shutil.which("cl"), r"C:\Program Files\msys64\mingw64\bin\gcc.exe"]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            name = Path(candidate).name.lower()
            return Compiler(Path(candidate), "msvc" if name in {"cl", "cl.exe"} else "gnu")
    raise RuntimeError("No C compiler found. Install Visual Studio Build Tools, LLVM, or MSYS2 MinGW-w64 GCC.")


def line(label: str, text: str, colour: str = "") -> None:
    prefix = f"{colour}{Colour.BOLD}[{label:<7}]{Colour.RESET}" if COLOUR else f"[{label:<7}]"
    print(f"{prefix} {text}")


def command_for(compiler: Compiler, debug: bool) -> list[str]:
    if compiler.family == "msvc":
        flags = ["/nologo", "/std:c11", "/W4", "/LD", str(SOURCE), f"/Fe:{OUTPUT}"]
        flags += ["/Od", "/Zi"] if debug else ["/O2", "/DNDEBUG"]
        return [str(compiler.path), *flags]
    flags = ["-std=c11", "-shared", "-Wall", "-Wextra", "-o", str(OUTPUT), str(SOURCE)]
    flags += ["-O0", "-g"] if debug else ["-O3", "-DNDEBUG", "-march=native"]
    return [str(compiler.path), *flags]


def compile_library(compiler: Compiler, debug: bool) -> None:
    command = command_for(compiler, debug)
    line("BUILD", " ".join(command), Colour.CYAN)
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    for output in (result.stdout, result.stderr):
        for item in output.splitlines():
            line("COMPILER", item, Colour.YELLOW if result.returncode else Colour.CYAN)
    if result.returncode:
        raise RuntimeError(f"Compiler exited with code {result.returncode}")


def verify_library() -> None:
    if not OUTPUT.is_file() or OUTPUT.stat().st_size < 1024:
        raise RuntimeError("Compiler did not create a valid-sized DLL")
    if OUTPUT.read_bytes()[:2] != b"MZ":
        raise RuntimeError("Output is not a Windows PE DLL")
    library = ctypes.WinDLL(str(OUTPUT))
    for symbol in ("decay_and_count", "edge_activity"):
        getattr(library, symbol)
    line("VERIFY", f"Loaded {OUTPUT.name} ({OUTPUT.stat().st_size:,} bytes); exported symbols verified", Colour.GREEN)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile AIBrain's optimized native connectome DLL.")
    parser.add_argument("--compiler", help="Path to a specific C compiler executable")
    parser.add_argument("--debug", action="store_true", help="Build an unoptimized debug DLL")
    parser.add_argument("--clean", action="store_true", help="Remove the compiled DLL before building")
    arguments = parser.parse_args()
    line("AIBRAIN", "Native connectome build tool · x64 Windows", Colour.CYAN)
    if not SOURCE.is_file():
        line("ERROR", f"Missing source: {SOURCE}", Colour.RED); return 1
    if arguments.clean and OUTPUT.exists():
        OUTPUT.unlink(); line("CLEAN", f"Removed {OUTPUT.name}", Colour.YELLOW)
    try:
        compiler = discover_compiler(arguments.compiler)
        line("TOOLCHAIN", f"{compiler.family.upper()} · {compiler.path}", Colour.GREEN)
        compile_library(compiler, arguments.debug)
        verify_library()
    except (OSError, RuntimeError) as exc:
        line("ERROR", str(exc), Colour.RED); return 1
    line("SUCCESS", "Native acceleration is ready for AIBrain.", Colour.GREEN)
    return 0


COLOUR = enable_colour()
if __name__ == "__main__":
    raise SystemExit(main())
