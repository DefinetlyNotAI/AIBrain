"""Windows bootstrapper for AIBrain.

Run this script with an installed Python 3.11+ interpreter. It is the sole
intentional exception to AIBrain's venv-only runtime rule: its job is to create
and populate that virtual environment.
"""
from __future__ import annotations

import re
import subprocess
import sys
import urllib.error
import urllib.request
import venv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
WHEEL_ROOT = "https://abetlen.github.io/llama-cpp-python/whl"
BASE_PACKAGES = ("PySide6>=6.7,<7", "numpy>=1.26,<3", "moderngl>=5.10")
CUDA_WHEELS = ((13, 2, "cu132"), (13, 0, "cu130"), (12, 5, "cu125"), (12, 4, "cu124"),
               (12, 3, "cu123"), (12, 2, "cu122"), (12, 1, "cu121"), (11, 8, "cu118"))


@dataclass(frozen=True, slots=True)
class GpuCapability:
    name: str
    driver: str
    cuda_version: tuple[int, int] | None


def run(command: list[str]) -> None:
    print("\n> " + subprocess.list2cmdline(command))
    subprocess.run(command, check=True)


def detect_nvidia() -> GpuCapability | None:
    """Return CUDA capability exposed by the installed NVIDIA driver, if any."""
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=8,
        )
        status = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True, timeout=8)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    first_gpu = query.stdout.strip().splitlines()[0].split(",", maxsplit=1)
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", status.stdout)
    return GpuCapability(first_gpu[0].strip(), first_gpu[1].strip() if len(first_gpu) > 1 else "unknown",
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
                return tag, f"NVIDIA CUDA wheel ({tag})"
        print("No compatible published CUDA wheel was found; using the CPU wheel.")
    return "cpu", "CPU wheel"


def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def main() -> int:
    if sys.version_info < (3, 11):
        print(f"Python 3.11+ is required; found {sys.version.split()[0]}.", file=sys.stderr)
        return 1
    gpu = detect_nvidia()
    if gpu:
        cuda = f"CUDA {gpu.cuda_version[0]}.{gpu.cuda_version[1]}" if gpu.cuda_version else "CUDA version unavailable"
        print(f"Detected NVIDIA GPU: {gpu.name} (driver {gpu.driver}; {cuda})")
    else:
        print("No NVIDIA CUDA capability detected. AIBrain will install the CPU inference wheel.")
    if not venv_python().exists():
        print(f"Creating virtual environment: {VENV_DIR}")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    python = str(venv_python())
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", *BASE_PACKAGES])
    wheel_tag, description = select_wheel(gpu)
    print(f"Installing llama-cpp-python from the {description}.")
    try:
        run([python, "-m", "pip", "install", "--only-binary=llama-cpp-python",
             "--extra-index-url", f"{WHEEL_ROOT}/{wheel_tag}", "llama-cpp-python>=0.3.0"])
    except subprocess.CalledProcessError:
        if wheel_tag == "cpu":
            print("The prebuilt CPU wheel could not be installed. No source build was attempted.", file=sys.stderr)
            return 1
        print("CUDA wheel installation failed; retrying the official CPU wheel.")
        run([python, "-m", "pip", "install", "--only-binary=llama-cpp-python",
             "--extra-index-url", f"{WHEEL_ROOT}/cpu", "llama-cpp-python>=0.3.0"])
    run([python, "-c", "import llama_cpp, moderngl; from PySide6 import QtCore; print('Dependency verification passed')"])
    print("\nAIBrain is ready. Start it with:\n  .\\.venv\\Scripts\\python.exe main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
