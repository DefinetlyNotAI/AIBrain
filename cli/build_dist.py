"""Compile a timestamped, standalone AIBrain distribution with Nuitka."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.console_ui import command, error, header, section, success
from src.utils.gpu import set_windows_executable_gpu_preference

VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
DIST_ROOT = ROOT / "dist"
# Nuitka already bundles msvcp140.dll from the Python/PySide dependency graph.
# llama.cpp additionally needs the OpenMP runtime, which Nuitka cannot locate
# reliably on a machine with only an older Visual Studio installation.
RUNTIME_DLLS = ("vcomp140.dll",)


def runtime_dlls() -> list[Path]:
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    resolved = [system32 / name for name in RUNTIME_DLLS]
    missing = [path.name for path in resolved if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required Visual C++ runtime DLLs are unavailable: {', '.join(missing)}")
    return resolved


def build(timestamp: str | None = None) -> Path:
    if not VENV_PYTHON.is_file():
        raise RuntimeError("Managed virtual environment is missing. Run: py cli\\installer.py")
    release = DIST_ROOT / f"AIBrain_{timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if release.exists():
        raise RuntimeError(f"Refusing to overwrite existing distribution: {release}")
    section("Compile standalone application", 1)
    release.mkdir(parents=True)
    build_root = release / "_nuitka"
    target = ROOT / "cli" / "main.py"
    command_line = [
        str(VENV_PYTHON), "-m", "nuitka", "--standalone", "--assume-yes-for-downloads", "--enable-plugin=pyside6",
        "--windows-console-mode=disable", "--output-filename=AIBrain.exe", f"--output-dir={build_root}",
        "--include-package=src",
        f"--include-data-files={ROOT / 'dll' / 'aibrain_connectome.dll'}=dll/aibrain_connectome.dll", str(target),
    ]
    for runtime in runtime_dlls():
        command_line.insert(-1, f"--include-data-files={runtime}={runtime.name}")
    command(command_line)
    subprocess.run(command_line, cwd=ROOT, check=True)
    produced = build_root / "main.dist"
    if not produced.is_dir():
        candidates = list(build_root.glob("*.dist"))
        if len(candidates) != 1:
            raise RuntimeError("Nuitka did not produce a standalone distribution directory")
        produced = candidates[0]
    for item in produced.iterdir():
        shutil.move(str(item), release / item.name)
    shutil.rmtree(build_root)
    executable = release / "AIBrain.exe"
    if not executable.is_file() or executable.stat().st_size < 100_000:
        raise RuntimeError("Nuitka output is missing AIBrain.exe or is unexpectedly small")
    if not (release / "dll" / "aibrain_connectome.dll").is_file():
        raise RuntimeError("Standalone distribution is missing the native connectome DLL")
    for runtime_name in ("msvcp140.dll", "vcomp140.dll"):
        if not (release / runtime_name).is_file():
            raise RuntimeError(f"Standalone distribution is missing required runtime: {runtime_name}")
    set_windows_executable_gpu_preference(executable)
    return release


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a timestamped Nuitka standalone AIBrain distribution.")
    parser.add_argument("--timestamp", help="Override the YYYYMMDD_HHMMSS distribution suffix for reproducible builds")
    arguments = parser.parse_args()
    header("AIBrain", "Nuitka standalone distribution builder")
    try:
        release = build(arguments.timestamp)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        error(str(exc))
        return 1
    success(f"Created self-contained distribution: {release}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        error("Distribution build cancelled by keyboard interrupt.")
        raise SystemExit(130)
