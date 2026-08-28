"""Windows bootstrapper for AIBrain.

Run this script with an installed Python 3.11+ interpreter. It is the sole
intentional exception to AIBrain's venv-only runtime rule: its job is to create
and populate that virtual environment.
"""
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

from src.utils.console_ui import Color, clear_screen, color, command_preview, command_output_box, detail, error, header, \
    info, \
    panel, relative_path, section, success, warning

VENV_DIR = ROOT / ".venv"

WHEEL_ROOT = "https://abetlen.github.io/llama-cpp-python/whl"

BASE_PACKAGES = (
    "PySide6>=6.7,<7",
    "numpy>=1.26,<3",
    "moderngl>=5.10",
    "nuitka>=2.5,<3",
)

CUDA_WHEELS = (
    (13, 2, "cu132"),
    (13, 0, "cu130"),
    (12, 5, "cu125"),
    (12, 4, "cu124"),
    (12, 3, "cu123"),
    (12, 2, "cu122"),
    (12, 1, "cu121"),
    (11, 8, "cu118"),
)


@dataclass(frozen=True, slots=True)
class GpuCapability:
    name: str
    driver: str
    cuda_version: tuple[int, int] | None


def run(command: list[str]) -> None:
    command_preview(command)

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output_parts: list[str] = []

    if process.stdout.strip():
        output_parts.append(process.stdout.rstrip())

    if process.stderr.strip():
        output_parts.append(process.stderr.rstrip())

    if output_parts:
        command_output_box("\n".join(output_parts))

    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode,
            command,
            output=process.stdout,
            stderr=process.stderr,
        )


def detect_nvidia() -> GpuCapability | None:
    """Return CUDA capability exposed by the installed NVIDIA driver, if any."""
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=8,
        )

        status = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True,
            timeout=8,
        )

    except (OSError, subprocess.SubprocessError):
        return None

    lines = query.stdout.strip().splitlines()

    if not lines:
        return None

    first_gpu = lines[0].split(",", maxsplit=1)

    match = re.search(
        r"CUDA Version:\s*(\d+)\.(\d+)",
        status.stdout,
    )

    return GpuCapability(
        name=first_gpu[0].strip(),
        driver=(
            first_gpu[1].strip()
            if len(first_gpu) > 1
            else "unknown"
        ),
        cuda_version=(
            (
                int(match.group(1)),
                int(match.group(2)),
            )
            if match
            else None
        ),
    )


def available_wheel(tag: str) -> bool:
    info(f"Checking wheel availability: {color(tag, Color.BOLD)}")

    try:
        request = urllib.request.Request(
            f"{WHEEL_ROOT}/{tag}/",
            headers={"User-Agent": "AIBrain-installer/1"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return 200 <= response.status < 300

    except (urllib.error.URLError, TimeoutError):
        return False


def select_wheel(
        gpu: GpuCapability | None,
) -> tuple[str, str]:
    if gpu and gpu.cuda_version:
        for major, minor, tag in CUDA_WHEELS:
            if (
                    (major, minor) <= gpu.cuda_version
                    and available_wheel(tag)
            ):
                return (
                    tag,
                    f"NVIDIA CUDA acceleration ({tag})",
                )

        warning("No compatible published CUDA wheel was found.")
        warning("Falling back to CPU inference.")

    return "cpu", "CPU inference"


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"

    return VENV_DIR / "bin" / "python"


def verify_python() -> bool:
    version = sys.version_info
    installed = (
        f"{version.major}."
        f"{version.minor}."
        f"{version.micro}"
    )

    if version < (3, 11):
        error(
            f"Python 3.11 or newer is required. "
            f"Detected Python {installed}."
        )
        return False

    success(f"Python {installed}")
    detail("Executable", relative_path(sys.executable))

    return True


def print_gpu(
        gpu: GpuCapability | None,
) -> None:
    if gpu is None:
        warning("NVIDIA CUDA capability was not detected.")
        detail("Backend", "CPU")
        return

    success("NVIDIA GPU detected")
    detail("GPU", gpu.name)
    detail("Driver", gpu.driver)

    if gpu.cuda_version:
        detail(
            "CUDA",
            f"{gpu.cuda_version[0]}.{gpu.cuda_version[1]}",
        )
    else:
        detail("CUDA", "Unavailable")


def create_environment() -> None:
    if venv_python().exists():
        success("Virtual environment already exists")
        detail("Location", relative_path(VENV_DIR))
        return

    info("Creating isolated Python environment...")
    detail("Location", relative_path(VENV_DIR))

    venv.EnvBuilder(
        with_pip=True,
    ).create(VENV_DIR)

    success("Virtual environment created")


def install_dependencies(
        python: str,
) -> None:
    info("Updating Python package manager")

    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
        ]
    )

    success("pip is ready")

    print()
    info("Installing AIBrain runtime dependencies")

    for package in BASE_PACKAGES:
        detail("Package", package)

    run(
        [
            python,
            "-m",
            "pip",
            "install",
            *BASE_PACKAGES,
        ]
    )

    success("Core dependencies installed")


def install_llama(
        python: str,
        gpu: GpuCapability | None,
) -> str:
    wheel_tag, description = select_wheel(gpu)

    print()

    info(
        "Selected backend: "
        + color(
            description,
            Color.BOLD,
            Color.CYAN,
        )
    )

    detail("Wheel", wheel_tag)

    command = [
        python,
        "-m",
        "pip",
        "install",
        "--only-binary=llama-cpp-python",
        "--extra-index-url",
        f"{WHEEL_ROOT}/{wheel_tag}",
        "llama-cpp-python>=0.3.0",
    ]

    try:
        run(command)

        success(
            f"llama-cpp-python installed using {wheel_tag}"
        )

        return wheel_tag

    except subprocess.CalledProcessError:
        if wheel_tag == "cpu":
            error(
                "The prebuilt CPU wheel could not be installed."
            )
            error(
                "A source compilation was not attempted."
            )
            raise

        print()
        warning(f"{wheel_tag} installation failed.")
        warning(
            "Retrying with the official CPU wheel."
        )

        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--only-binary=llama-cpp-python",
                "--extra-index-url",
                f"{WHEEL_ROOT}/cpu",
                "llama-cpp-python>=0.3.0",
            ]
        )

        success(
            "CPU fallback installed successfully"
        )

        return "cpu"


def verify_installation(
        python: str,
) -> None:
    info(
        "Running import and runtime verification"
    )

    verification = (
        "import llama_cpp, moderngl, nuitka; "
        "from PySide6 import QtCore; "
        "import numpy; "
        "print('AIBrain dependency verification passed')"
    )

    run(
        [
            python,
            "-c",
            verification,
        ]
    )

    success(
        "All required dependencies are importable"
    )


def completion_screen(
        gpu: GpuCapability | None,
        wheel_tag: str,
) -> None:
    backend = (
        f"CUDA / {wheel_tag}"
        if wheel_tag != "cpu"
        else "CPU"
    )
    panel(
        "INSTALLATION COMPLETE",
        [
            ("Environment", relative_path(VENV_DIR)),
            ("Backend", backend),
            ("GPU", gpu.name if gpu else "Not detected"),
        ],
        footer="Launch AIBrain with:  python cli\\main.py",
        tone=Color.GREEN,
    )


def print_help_banner() -> None:
    panel(
        "AIBrain Installer",
        [("-h, --help", "Show this help screen and exit.")],
        subtitle="Available Runtime Flags",
        footer=f"Usage: python {Path(__file__).name} [options]",
    )


class BannerArgumentParser(
    argparse.ArgumentParser
):
    def print_help(
            self,
            file=None,
    ) -> None:
        print_help_banner()


def parse_args() -> argparse.Namespace:
    parser = BannerArgumentParser(
        description=(
            "Install and configure the AIBrain runtime."
        ),
        add_help=True,
    )

    return parser.parse_args()


def main() -> int:
    parse_args()

    clear_screen()
    header()

    section("System check", 1)

    if not verify_python():
        return 1

    gpu = detect_nvidia()
    print_gpu(gpu)

    section("Virtual environment", 2)

    try:
        create_environment()
    except Exception as exc:
        error(
            f"Unable to create the virtual environment: {exc}"
        )
        return 1

    python = str(venv_python())

    section("Core dependencies", 3)

    try:
        install_dependencies(python)
    except subprocess.CalledProcessError as exc:
        error(
            "Dependency installation failed with "
            f"exit code {exc.returncode}."
        )
        return 1

    section("Inference backend", 4)

    try:
        wheel_tag = install_llama(
            python,
            gpu,
        )

    except subprocess.CalledProcessError as exc:
        error(
            "llama-cpp-python installation failed "
            f"with exit code {exc.returncode}."
        )
        return 1

    section("Verification", 5)

    try:
        verify_installation(python)

    except subprocess.CalledProcessError:
        error(
            "Dependency verification failed."
        )
        return 1

    completion_screen(
        gpu,
        wheel_tag,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        error("Installation cancelled by keyboard interrupt.")
        raise SystemExit(130)
