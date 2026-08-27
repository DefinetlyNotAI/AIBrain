"""Compile timestamped standalone AIBrain desktop applications with Nuitka."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.console_ui import Color, clear_screen, command_output_box, command_preview, error, header, panel, section
from src.utils.gpu import set_windows_executable_gpu_preference


VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
DIST_ROOT = ROOT / "dist"
RUNTIME_DLLS = ("vcomp140.dll",)
NATIVE_LIBRARY = ROOT / "dll" / "aibrain.connectome.dll"


@dataclass(frozen=True, slots=True)
class ApplicationTarget:
    directory: str
    executable: str
    entry_point: Path
    icon: Path
    console_mode: str


APPLICATIONS = (
    ApplicationTarget("ai_brain", "ai_brain.exe", ROOT / "cli" / "main.py", ROOT / "ico" / "brain.ico", "attach"),
    ApplicationTarget("diagnostic", "diagnostic.exe", ROOT / "cli" / "diagnostic.py", ROOT / "ico" / "diagnostic.ico", "disable"),
    ApplicationTarget("analysis", "analysis.exe", ROOT / "cli" / "analysis.py", ROOT / "ico" / "analysis.ico", "attach"),
)


def run(command_line: list[str]) -> None:
    """Run a build command and present its complete output in one shared box."""
    command_preview(command_line)
    result = subprocess.run(
        command_line,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    command_output_box("\n".join(part.rstrip() for part in (result.stdout, result.stderr) if part.strip()))
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command_line, result.stdout, result.stderr)


def runtime_dlls() -> list[Path]:
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    resolved = [system32 / name for name in RUNTIME_DLLS]
    missing = [path.name for path in resolved if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required Visual C++ runtime DLLs are unavailable: {', '.join(missing)}")
    return resolved


def nuitka_command(target: ApplicationTarget, build_root: Path, runtimes: list[Path]) -> list[str]:
    """Build a reproducible standalone command for one application target."""
    command_line = [
        str(VENV_PYTHON),
        "-m",
        "nuitka",
        "--standalone",
        "--assume-yes-for-downloads",
        "--enable-plugin=pyside6",
        f"--windows-console-mode={target.console_mode}",
        f"--windows-icon-from-ico={target.icon}",
        f"--output-filename={target.executable}",
        f"--output-dir={build_root}",
        "--include-package=src",
        f"--include-data-files={NATIVE_LIBRARY}=dll/{NATIVE_LIBRARY.name}",
        str(target.entry_point),
    ]
    for runtime in runtimes:
        command_line.insert(-1, f"--include-data-files={runtime}={runtime.name}")
    return command_line


def _produced_distribution(build_root: Path) -> Path:
    candidates = list(build_root.glob("*.dist"))
    if len(candidates) != 1 or not candidates[0].is_dir():
        raise RuntimeError("Nuitka did not produce exactly one standalone distribution directory")
    return candidates[0]


def _verify_application(application: Path, target: ApplicationTarget) -> Path:
    executable = application / target.executable
    if not executable.is_file() or executable.stat().st_size < 100_000:
        raise RuntimeError(f"Standalone output is missing {target.executable} or it is unexpectedly small")
    if not (application / "dll" / NATIVE_LIBRARY.name).is_file():
        raise RuntimeError(f"{target.executable} is missing the native connectome DLL")
    for runtime_name in ("msvcp140.dll", *RUNTIME_DLLS):
        if not (application / runtime_name).is_file():
            raise RuntimeError(f"{target.executable} is missing required runtime {runtime_name}")
    return executable


def build(timestamp: str | None = None) -> Path:
    if not VENV_PYTHON.is_file():
        raise RuntimeError("Managed virtual environment is missing. Run: py cli\\installer.py")
    if not NATIVE_LIBRARY.is_file():
        raise RuntimeError("Native connectome DLL is missing. Run: py cli\\build_native.py")
    for target in APPLICATIONS:
        if not target.icon.is_file():
            raise RuntimeError(f"Required application icon is missing: {target.icon}")

    release = DIST_ROOT / f"AIBrain_{timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if release.exists():
        raise RuntimeError(f"Refusing to overwrite existing distribution: {release}")
    release.mkdir(parents=True)
    runtimes = runtime_dlls()
    try:
        for index, target in enumerate(APPLICATIONS, start=1):
            section(f"Compile {target.executable}", index)
            build_root = release / "_nuitka" / target.directory
            run(nuitka_command(target, build_root, runtimes))
            application = release / target.directory
            shutil.move(str(_produced_distribution(build_root)), application)
            executable = _verify_application(application, target)
            if target.executable == "ai_brain.exe":
                set_windows_executable_gpu_preference(executable)
    finally:
        shutil.rmtree(release / "_nuitka", ignore_errors=True)
    return release


def main() -> int:
    parser = argparse.ArgumentParser(description="Build timestamped standalone AIBrain desktop applications.")
    parser.add_argument("--timestamp", help="Override the YYYYMMDD_HHMMSS distribution suffix for reproducible builds")
    arguments = parser.parse_args()
    clear_screen()
    header("AIBrain", "Nuitka standalone distribution builder")
    try:
        release = build(arguments.timestamp)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        error(str(exc))
        return 1
    panel(
        "DISTRIBUTION COMPLETE",
        [(target.executable, str(release / target.directory / target.executable)) for target in APPLICATIONS],
        footer="Each application folder is self-contained and ready to launch.",
        tone=Color.GREEN,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        error("Distribution build cancelled by keyboard interrupt.")
        raise SystemExit(130)
