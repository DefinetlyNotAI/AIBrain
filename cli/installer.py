"""Create and verify AIBrain's managed Windows runtime environment."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import urllib.error
import urllib.request
import venv
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.console_ui import Color, command, detail, error, header, info, section, success, warning


VENV_DIR = ROOT / ".venv"
WHEEL_ROOT = "https://abetlen.github.io/llama-cpp-python/whl"
BASE_PACKAGES = ("PySide6>=6.7,<7", "numpy>=1.26,<3", "moderngl>=5.10", "nuitka>=2.5,<3")
CUDA_WHEELS = ((13, 2, "cu132"), (13, 0, "cu130"), (12, 5, "cu125"), (12, 4, "cu124"),
               (12, 3, "cu123"), (12, 2, "cu122"), (12, 1, "cu121"), (11, 8, "cu118"))


@dataclass(frozen=True, slots=True)
class GpuCapability:
    name: str
    driver: str
    cuda_version: tuple[int, int] | None


def venv_python() -> Path:
    return VENV_DIR / "Scripts" / "python.exe" if sys.platform == "win32" else VENV_DIR / "bin" / "python"


def run(command_line: list[str]) -> None:
    command(command_line)
    result = subprocess.run(command_line, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command_line, output=result.stdout)


def detect_nvidia() -> GpuCapability | None:
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=8)
        status = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True, timeout=8)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    first = query.stdout.strip().splitlines()[0].split(",", maxsplit=1)
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", status.stdout)
    return GpuCapability(first[0].strip(), first[1].strip() if len(first) > 1 else "unknown",
                         (int(match.group(1)), int(match.group(2))) if match else None)


def available_wheel(tag: str) -> bool:
    try:
        with urllib.request.urlopen(f"{WHEEL_ROOT}/{tag}/", timeout=10) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError):
        return False


def select_wheel(gpu: GpuCapability | None) -> tuple[str, str]:
    if gpu and gpu.cuda_version:
        for major, minor, tag in CUDA_WHEELS:
            if (major, minor) <= gpu.cuda_version and available_wheel(tag):
                return tag, f"NVIDIA CUDA acceleration ({tag})"
        warning("No compatible published CUDA wheel was found; using the CPU wheel.")
    return "cpu", "CPU inference"


def verify_python() -> bool:
    if sys.version_info < (3, 11):
        error(f"Python 3.11 or newer is required; detected {sys.version.split()[0]}.")
        return False
    success(f"Python {sys.version.split()[0]}")
    detail("Executable", str(Path(sys.executable)))
    return True


def create_environment() -> None:
    if venv_python().exists():
        success("Virtual environment already exists")
        detail("Location", str(VENV_DIR))
        return
    info("Creating isolated Python environment")
    venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    success("Virtual environment created")


def install_dependencies(python: str) -> None:
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", *BASE_PACKAGES])
    success("Core dependencies and Nuitka installed")


def install_llama(python: str, gpu: GpuCapability | None) -> str:
    tag, description = select_wheel(gpu)
    info("Selected backend: " + description)
    request = [python, "-m", "pip", "install", "--only-binary=llama-cpp-python", "--extra-index-url", f"{WHEEL_ROOT}/{tag}", "llama-cpp-python>=0.3.0"]
    try:
        run(request)
        return tag
    except subprocess.CalledProcessError:
        if tag == "cpu":
            raise
        warning(f"{tag} wheel failed; retrying with the official CPU wheel.")
        run([python, "-m", "pip", "install", "--only-binary=llama-cpp-python", "--extra-index-url", f"{WHEEL_ROOT}/cpu", "llama-cpp-python>=0.3.0"])
        return "cpu"


def verify_installation(python: str) -> None:
    run([python, "-c", "import llama_cpp, moderngl, nuitka, numpy; from PySide6 import QtCore; print('AIBrain dependency verification passed')"])
    success("All required dependencies are importable")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install AIBrain's managed Python runtime.")
    parser.parse_args()
    header("AIBrain", "Neural Runtime Installer")
    section("System check", 1)
    if not verify_python():
        return 1
    gpu = detect_nvidia()
    if gpu:
        success("NVIDIA GPU detected")
        detail("GPU", gpu.name)
        detail("Driver", gpu.driver)
    else:
        warning("NVIDIA CUDA capability was not detected; CPU inference will be installed.")
    try:
        section("Virtual environment", 2)
        create_environment()
        python = str(venv_python())
        section("Core dependencies", 3)
        install_dependencies(python)
        section("Inference backend", 4)
        wheel_tag = install_llama(python, gpu)
        section("Verification", 5)
        verify_installation(python)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        error(f"Installation failed: {exc}")
        return 1
    success(f"Installation complete ({wheel_tag}). Launch with: {python} cli\\main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
