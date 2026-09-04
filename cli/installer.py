"""Windows bootstrapper for AIBrain.

Run this script with an installed Python 3.11+ interpreter. It is the sole
intentional exception to AIBrain's venv-only runtime rule: its job is to create
and populate that virtual environment.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
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

from cli.build_dist import run as run_with_live_output
from src.models.diagnostics import OllamaDiagnostics
from src.utils.console_ui import (
    Color,
    ask_choice,
    clear_screen,
    color,
    detail,
    error,
    header,
    info,
    panel,
    relative_path,
    report_keyboard_interrupt,
    section,
    success,
    warning,
)
from src.utils.logging import configure_cli_logging, report_exception

VENV_DIR = ROOT / ".venv"

EMBEDDED_REPAIR_ENV = "AIBRAIN_EMBEDDED_REPAIR"
LLAMA_CPP_PYTHON_REQUIREMENT = "llama-cpp-python>=0.3.35,<0.4"

WHEEL_ROOT = "https://abetlen.github.io/llama-cpp-python/whl"

BASE_PACKAGES = (
    "PySide6>=6.7,<7",
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


@dataclass(frozen=True, slots=True)
class InstallerHealth:
    subsystem: str
    status: str
    reason: str


class LlamaRuntimeError(RuntimeError):
    """The selected prebuilt llama.cpp wheel installed but cannot load."""


def collect_final_health(
    python: str,
    gpu: GpuCapability | None,
    root: Path = ROOT,
    *,
    libraries_verified: bool = False,
) -> list[InstallerHealth]:
    """Collect a read-only final health report for the installed runtime."""
    cuda_status = "Unavailable"
    cuda_reason = "REASON: No CUDA-capable NVIDIA driver runtime was detected."
    if gpu and gpu.cuda_version and gpu.cuda_version[0] < 12:
        cuda_status = "CPU fallback"
        cuda_reason = "The array runtime requires CUDA 12 or newer; NumPy is selected."
    elif gpu and gpu.cuda_version:
        cuda_status = "Detected"
        cuda_reason = (
            f"Driver supports CUDA {gpu.cuda_version[0]}.{gpu.cuda_version[1]}."
        )
        if Path(python).is_file():
            try:
                probe = subprocess.run(
                    [
                        python,
                        "-c",
                        "from src.utils.array_api import BACKEND_NAME; print(BACKEND_NAME)",
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if probe.returncode == 0 and "CuPy / CUDA" in probe.stdout:
                    cuda_status = "Ready"
                    cuda_reason += " CuPy executed a device operation successfully."
                else:
                    cuda_status = "Needs repair"
                    cuda_reason += (
                        " REASON: The managed array probe did not confirm usable CUDA."
                    )
            except (OSError, subprocess.SubprocessError):
                cuda_status = "Needs repair"
                cuda_reason += " REASON: The managed CUDA runtime could not be probed."
        else:
            cuda_status = "Needs repair"
            cuda_reason += " REASON: The managed interpreter is missing."
    checks = [
        InstallerHealth(
            "Hardware",
            "Ready" if gpu and gpu.cuda_version else "CPU fallback",
            (
                "NVIDIA CUDA capability detected."
                if gpu and gpu.cuda_version
                else "REASON: CUDA was not detected; CPU inference is supported."
            ),
        ),
        InstallerHealth(
            "CUDA",
            cuda_status,
            cuda_reason,
        ),
        InstallerHealth(
            "Python",
            "Ready" if Path(python).is_file() else "Needs repair",
            (
                "Managed virtual-environment interpreter found."
                if Path(python).is_file()
                else "REASON: The managed interpreter is missing."
            ),
        ),
        InstallerHealth(
            "pip / libraries",
            "Ready" if libraries_verified else "Not checked",
            (
                "Imports and pip dependency consistency passed verification."
                if libraries_verified
                else "Run installation verification to check libraries."
            ),
        ),
    ]
    try:
        models = OllamaDiagnostics().inspect()
        broken = [item for item in models if not item.available]
        checks.append(
            InstallerHealth(
                "Model manifests / blobs",
                "Needs repair" if broken else "Ready",
                (
                    f"REASON: {broken[0].reason}"
                    if broken
                    else "Local manifests are healthy or none are installed."
                ),
            )
        )
    except OSError as exc:
        checks.append(
            InstallerHealth(
                "Model manifests / blobs",
                "Info",
                f"REASON: Could not inspect models: {exc}",
            )
        )
    cache = root / ".cache"
    checks.append(
        InstallerHealth(
            ".cache",
            "Ready" if cache.exists() else "Info",
            "Cache exists." if cache.exists() else "No cache has been created yet.",
        )
    )
    native_dll = root / "dll" / "aibrain.connectome.dll"
    checks.append(
        InstallerHealth(
            "Native DLLs",
            "Present" if native_dll.is_file() else "Info",
            (
                "Optional connectome DLL found; loading it was not checked."
                if native_dll.is_file()
                else "No native DLL found; supported Python fallback remains available."
            ),
        )
    )
    return checks


def print_final_health(
    python: str, gpu: GpuCapability | None, *, libraries_verified: bool = False
) -> list[InstallerHealth]:
    checks = collect_final_health(python, gpu, libraries_verified=libraries_verified)
    panel(
        "FINAL HEALTH CHECK",
        [(check.subsystem, f"{check.status} — {check.reason}") for check in checks],
        subtitle="Selected repairs affect only the named subsystem",
    )
    return checks


def repair_validation_cache(root: Path = ROOT) -> None:
    """Invalidate only disposable validation data; never broad cache/model data."""
    target = root / ".cache" / "validation"
    resolved_root, resolved_target = root.resolve(), target.resolve()
    if not resolved_target.is_relative_to(resolved_root):
        raise ValueError("Refusing to invalidate a cache path outside the project")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def repair_selected_subsystem(
    subsystem: str,
    python: str,
    gpu: GpuCapability | None,
    *,
    model: str | None = None,
    preference: str = "auto",
) -> str:
    """Run exactly one selected repair; destructive work is explicit and narrow."""
    if subsystem == "dependencies":
        install_dependencies(python, gpu, force_reinstall=False)
        return "dependencies"
    if subsystem == "backend":
        ensure_pip(python)
        wheel_tag = install_llama(
            python, gpu, preference=preference, force_reinstall=False
        )
        repair_validation_cache()
        return wheel_tag
    if subsystem == "native":
        run([python, str(ROOT / "cli" / "build_native.py")])
        return "native"
    if subsystem == "cache":
        repair_validation_cache()
        return "cache"
    if subsystem == "models":
        if not model:
            raise ValueError(
                "Model repair requires --model <name:tag>; no model blob is deleted implicitly."
            )
        run(["ollama", "pull", model])
        return "models"
    raise ValueError(f"Unsupported repair subsystem: {subsystem}")


def run(command: list[str]) -> None:
    """Run installer work through the shared live command-output renderer."""
    run_with_live_output(command)


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
        r"CUDA(?:\s+UMD)?\s+Version:\s*(\d+)\.(\d+)",
        status.stdout + "\n" + status.stderr,
    )

    return GpuCapability(
        name=first_gpu[0].strip(),
        driver=(first_gpu[1].strip() if len(first_gpu) > 1 else "unknown"),
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


def _ask_choice(prompt: str, choices: dict[str, str], default: str) -> str:
    """Ask one bounded installer question, defaulting safely without a TTY."""
    selected = ask_choice(prompt, choices, default=default)
    return selected if selected is not None else default


def select_install_action(
    *,
    runtime_exists: bool,
    install_requested: bool = False,
    repair_requested: bool = False,
    assume_yes: bool = False,
) -> str:
    """Select the top-level installer action before probing or changing the runtime."""
    if install_requested:
        return "install"
    if repair_requested:
        if not runtime_exists:
            raise ValueError("Repair is unavailable until the managed runtime has been installed.")
        return "repair"

    default = "i"
    if assume_yes:
        return "install"

    choices = {"i": "Install"}
    if runtime_exists:
        choices["r"] = "Repair"
    else:
        info("Repair is unavailable until the managed runtime has been installed.")
    selected = _ask_choice("Choose installer action", choices, default)
    return "repair" if selected == "r" else "install"


def select_wheel(
    gpu: GpuCapability | None,
    *,
    preference: str = "auto",
    interactive: bool = False,
) -> tuple[str, str]:
    if preference == "cpu":
        return "cpu", "CPU inference (selected by user)"
    if gpu and gpu.cuda_version:
        for major, minor, tag in CUDA_WHEELS:
            if (major, minor) <= gpu.cuda_version and available_wheel(tag):
                if interactive and preference == "auto":
                    selected = _ask_choice(
                        f"Both {tag} CUDA and CPU inference are available. Select a llama.cpp backend",
                        {"g": "GPU / CUDA", "c": "CPU"},
                        "g",
                    )
                    if selected == "c":
                        return "cpu", "CPU inference (selected by user)"
                return tag, f"NVIDIA CUDA acceleration ({tag})"

        if preference != "cuda":
            warning("No compatible published CUDA wheel was found.")
            warning("Falling back to CPU inference.")

    if preference == "cuda":
        raise LlamaRuntimeError(
            "CUDA was requested but no compatible driver and published wheel were found."
        )

    return "cpu", "CPU inference"


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"

    return VENV_DIR / "bin" / "python"


def verify_python() -> bool:
    version = sys.version_info
    installed = f"{version.major}." f"{version.minor}." f"{version.micro}"

    if version < (3, 11):
        error(f"Python 3.11 or newer is required. " f"Detected Python {installed}.")
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
        detail("CUDA", "Driver detected; exact UMD version was not reported")


def numerical_package(gpu: GpuCapability | None) -> str:
    """Select exactly one supported array runtime for this machine."""
    if gpu and gpu.cuda_version and gpu.cuda_version[0] >= 13:
        return "cupy-cuda13x[ctk]>=14,<15"
    if gpu and gpu.cuda_version and gpu.cuda_version[0] == 12:
        return "cupy-cuda12x[ctk]>=14,<15"
    return "numpy>=1.26,<3"


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


def verify_managed_python(python: str) -> None:
    """Reject broken or misplaced interpreters before invoking their pip."""
    verification = (
        "import sys; from pathlib import Path; "
        "valid = (sys.version_info >= (3, 11) "
        "and sys.prefix != sys.base_prefix "
        "and Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve()); "
        "print('Managed Python is ready' if valid else "
        "'The managed environment requires its own Python 3.11+ interpreter.'); "
        "raise SystemExit(0 if valid else 1)"
    )
    run([python, "-c", verification, str(VENV_DIR)])


def ensure_pip(python: str) -> None:
    """Bootstrap pip even when an existing venv was created without it."""
    try:
        subprocess.run(
            [python, "-m", "pip", "--version"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except subprocess.CalledProcessError:
        info("Restoring pip in the managed environment")
        run([python, "-m", "ensurepip", "--upgrade"])


def cleanup_invalid_distributions(venv_dir: Path | None = None) -> list[Path]:
    """Remove pip's tilde-prefixed leftovers only inside the managed venv."""
    runtime = (venv_dir or VENV_DIR).resolve()
    libraries = (
        [runtime / "Lib" / "site-packages"]
        if sys.platform == "win32"
        else list((runtime / "lib").glob("python*/site-packages"))
    )
    targets: list[Path] = []
    for library in libraries:
        resolved_library = library.resolve()
        if not resolved_library.is_relative_to(runtime):
            raise ValueError(f"Refusing to clean site-packages outside {runtime}")
        if not library.is_dir():
            continue
        for candidate in sorted(library.iterdir()):
            if not candidate.name.startswith("~"):
                continue
            # Validate every absolute target before deleting anything. A linked
            # package must never redirect cleanup into another package or tree.
            resolved = candidate.resolve()
            if resolved.parent != resolved_library or resolved.name != candidate.name:
                raise ValueError(f"Refusing to remove linked distribution: {candidate}")
            targets.append(candidate)

    for target in targets:
        info(f"Removing invalid distribution: {relative_path(target)}")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    return targets


def install_dependencies(
    python: str,
    gpu: GpuCapability | None = None,
    *,
    force_reinstall: bool = False,
) -> None:
    ensure_pip(python)
    if force_reinstall:
        info("Reinstalling Python package manager")
        run([
            python, "-m", "pip", "install", "--upgrade", "--force-reinstall", "pip",
        ])
        success("pip reinstalled")
    else:
        success("Existing pip installation is healthy")

    print()
    info("Installing AIBrain runtime dependencies")

    packages = (*BASE_PACKAGES, numerical_package(gpu))
    for package in packages:
        detail("Package", package)

    if not force_reinstall:
        for package in broken_dependencies(python, packages):
            warning(f"Repairing broken library: {package}")
            # Reinstall only the broken root package. The normal resolver pass
            # below restores any missing transitive dependencies without
            # replacing healthy installed distributions.
            run([python, "-m", "pip", "install", "--force-reinstall", "--no-deps", package])

    command = [python, "-m", "pip", "install"]
    if force_reinstall:
        command.extend(("--upgrade", "--force-reinstall"))
    command.extend(packages)
    run(command)

    success("Core dependencies installed")


def _dependency_probe(requirement: str) -> str:
    """Return a functional import probe for one managed root requirement."""
    name = re.split(r"[<>=!~\[]", requirement, maxsplit=1)[0].lower()
    probes = {
        "pyside6": "from PySide6 import QtCore; assert QtCore.qVersion()",
        "moderngl": "import moderngl; assert moderngl.__version__",
        "nuitka": "import nuitka; assert nuitka.__file__",
        "numpy": "import numpy; assert numpy.ones(1).sum() == 1",
        "cupy-cuda13x": "import cupy; assert cupy.ones(1).sum().item() == 1",
        "cupy-cuda12x": "import cupy; assert cupy.ones(1).sum().item() == 1",
    }
    try:
        return probes[name]
    except KeyError as exc:
        raise ValueError(f"No repair probe is defined for {requirement}") from exc


def broken_dependencies(python: str, requirements: tuple[str, ...]) -> list[str]:
    """Identify broken managed roots without modifying healthy distributions."""
    broken: list[str] = []
    for requirement in requirements:
        try:
            result = subprocess.run(
                [python, "-c", _dependency_probe(requirement)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            broken.append(requirement)
            continue
        if result.returncode:
            broken.append(requirement)
    return broken


def llama_install_command(python: str, wheel_tag: str) -> list[str]:
    """Build a cache-safe current llama.cpp installation command."""
    command = [
        python,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--upgrade",
        "--no-cache-dir",
        "--only-binary=llama-cpp-python",
        "--extra-index-url",
        f"{WHEEL_ROOT}/{wheel_tag}",
        LLAMA_CPP_PYTHON_REQUIREMENT,
    ]
    return command


def cuda_runtime_packages(wheel_tag: str) -> tuple[str, ...]:
    """Match cuBLAS and the CUDA runtime to the selected wheel's major version."""
    if wheel_tag == "cpu":
        return ()
    match = re.fullmatch(r"cu(11|12|13)\d+", wheel_tag)
    if match is None:
        raise ValueError(f"Unsupported CUDA wheel tag: {wheel_tag}")
    major = int(match.group(1))
    suffix = "" if major >= 13 else f"-cu{major}"
    return (
        f"nvidia-cublas{suffix}>={major},<{major + 1}",
        f"nvidia-cuda-runtime{suffix}>={major},<{major + 1}",
    )


def probe_llama_runtime(
    python: str, *, require_cuda: bool = False, require_cpu: bool = False
) -> tuple[bool, str]:
    """Check the installed native backend without printing an import traceback."""
    if require_cuda and require_cpu:
        raise ValueError("A backend probe cannot require both CUDA and CPU")
    cuda_check = (
        "    if not llama_cpp.llama_supports_gpu_offload():\n"
        "        raise RuntimeError('The installed wheel does not support GPU offload')\n"
        if require_cuda
        else ""
    )
    cpu_check = (
        "    if llama_cpp.llama_supports_gpu_offload():\n"
        "        raise RuntimeError('The installed wheel is CUDA-enabled, not CPU-only')\n"
        if require_cpu
        else ""
    )
    verification = (
        "try:\n"
        "    from src.models.llama_runtime import load_llama_cpp\n"
        "    llama_cpp = load_llama_cpp()\n"
        f"{cuda_check}"
        f"{cpu_check}"
        "except Exception as exc:\n"
        "    print(f'{type(exc).__name__}: {exc}')\n"
        "    raise SystemExit(1)\n"
        "print('llama-cpp-python native runtime loaded')"
    )
    try:
        result = subprocess.run(
            [python, "-c", verification],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"Could not run the backend probe: {exc}"

    if result.returncode == 0:
        return True, ""

    output = result.stderr or result.stdout
    reason = " ".join(output.split())
    return False, reason or "The native backend import exited without a diagnostic."


def install_cpu_fallback(python: str, *, reason: str) -> str:
    """Replace an unavailable CUDA backend with a freshly downloaded CPU wheel."""
    warning(f"Selected CUDA backend is unavailable: {reason}")
    warning("Retrying with the official CPU wheel.")
    run(llama_install_command(python, "cpu"))
    ready, cpu_reason = probe_llama_runtime(python)
    if not ready:
        raise LlamaRuntimeError(
            "The official CPU backend installed but could not load: " f"{cpu_reason}"
        )
    success("CPU fallback installed and loaded successfully")
    return "cpu"


def install_llama(
    python: str,
    gpu: GpuCapability | None,
    *,
    preference: str = "auto",
    interactive: bool = False,
    force_reinstall: bool = True,
) -> str:
    wheel_tag, description = select_wheel(
        gpu, preference=preference, interactive=interactive
    )

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

    if not force_reinstall:
        ready, reason = probe_llama_runtime(
            python,
            require_cuda=wheel_tag != "cpu",
            require_cpu=wheel_tag == "cpu",
        )
        if ready:
            success(f"Existing llama-cpp-python {wheel_tag} backend is healthy")
            return wheel_tag
        warning(f"Repairing broken llama-cpp-python backend: {reason}")

    try:
        runtime_packages = cuda_runtime_packages(wheel_tag)
        if runtime_packages:
            info("Ensuring the selected CUDA wheel's runtime DLLs are installed")
            run([
                python, "-m", "pip", "install", "--upgrade", "--only-binary=:all:",
                *runtime_packages,
            ])
        # CPU and CUDA wheels share a version. Even Install mode must replace
        # an existing wheel, otherwise pip can silently keep the old backend.
        run(llama_install_command(python, wheel_tag))

    except subprocess.CalledProcessError as exc:
        if wheel_tag == "cpu":
            error("The prebuilt CPU wheel could not be installed.")
            error("A source compilation was not attempted.")
            raise

        if preference == "cuda":
            raise LlamaRuntimeError(
                f"{wheel_tag} installation failed; explicit CUDA selection was preserved."
            ) from exc
        print()
        warning(f"{wheel_tag} installation failed.")
        return install_cpu_fallback(python, reason="the CUDA wheel could not be installed")

    ready, reason = probe_llama_runtime(python, require_cuda=wheel_tag != "cpu")
    if ready:
        success(f"llama-cpp-python installed and loaded using {wheel_tag}")
        return wheel_tag

    if wheel_tag == "cpu":
        raise LlamaRuntimeError(
            "The official CPU backend installed but could not load: " f"{reason}"
        )
    if preference == "cuda":
        raise LlamaRuntimeError(
            f"{wheel_tag} could not load after installing its runtime DLLs: {reason}"
        )
    return install_cpu_fallback(python, reason=reason)


def verify_installation(
    python: str,
) -> None:
    info("Running import and runtime verification")

    verification = (
        "from src.models.llama_runtime import load_llama_cpp; load_llama_cpp(); "
        "import moderngl, nuitka; "
        "from PySide6 import QtCore; "
        "from src.utils.array_api import BACKEND_NAME, array_api; "
        "print('AIBrain dependency verification passed:', BACKEND_NAME)"
    )

    run(
        [
            python,
            "-c",
            verification,
        ]
    )

    run([python, "-m", "pip", "check"])
    success("Core dependencies and the selected backend are ready")


def completion_screen(
    gpu: GpuCapability | None,
    wheel_tag: str,
    *,
    action: str,
) -> None:
    backend = (
        "Existing installed backend"
        if wheel_tag == "existing"
        else f"CUDA / {wheel_tag}" if wheel_tag != "cpu" else "CPU"
    )
    panel(
        "INSTALLATION COMPLETE" if action == "install" else "REPAIR COMPLETE",
        [
            ("Environment", relative_path(VENV_DIR)),
            ("Backend", backend),
            ("GPU", gpu.name if gpu else "Not detected"),
        ],
        footer=r"Launch AIBrain with:  .\.venv\Scripts\python.exe cli\main.py",
        tone=Color.GREEN,
    )


def print_help_banner() -> None:
    panel(
        "AIBrain Installer",
        [
            ("-h, --help", "Show this help screen and exit."),
            (
                "--install",
                "Select Install mode without showing the action menu.",
            ),
            (
                "--repair",
                "Select Repair mode; available only after a managed runtime exists.",
            ),
            (
                "--backend <auto|cuda|cpu>",
                "Choose inference acceleration; auto recommends CUDA.",
            ),
            (
                "-y, --yes",
                "Accept the recommended action without prompting.",
            ),
        ],
        subtitle="Available Runtime Flags",
        footer=r"Usage: python cli\installer.py [options]",
    )


class BannerArgumentParser(argparse.ArgumentParser):
    def print_help(
        self,
        file=None,
    ) -> None:
        print_help_banner()


def parse_args() -> argparse.Namespace:
    parser = BannerArgumentParser(
        description=("Install and configure the AIBrain runtime."),
        add_help=True,
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--install", action="store_true")
    action.add_argument("--repair", action="store_true")
    parser.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--repair-subsystem",
        choices=("backend",),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("-y", "--yes", action="store_true")

    arguments = parser.parse_args()
    if arguments.repair_subsystem and not arguments.repair:
        parser.error("--repair-subsystem requires --repair")
    return arguments


def main() -> int:
    args = parse_args()
    runtime_log, _ = configure_cli_logging("installer")

    embedded_repair = os.environ.get(EMBEDDED_REPAIR_ENV) == "1"
    if not embedded_repair:
        clear_screen()
        header()
    detail("Log file", str(runtime_log))

    existing_runtime = venv_python().exists()
    if args.repair_subsystem:
        if not existing_runtime:
            error("REASON: Backend repair requires the managed runtime.")
            return 1
        gpu = detect_nvidia()
        section("Targeted backend repair", 1)
        info("Updating llama-cpp-python only; Ollama model data is preserved")
        try:
            verify_managed_python(str(venv_python()))
            cleanup_invalid_distributions()
            wheel_tag = repair_selected_subsystem(
                args.repair_subsystem,
                str(venv_python()),
                gpu,
                preference=args.backend,
            )
        except (subprocess.CalledProcessError, LlamaRuntimeError) as exc:
            error(str(exc))
            return 1
        success(f"Updated llama-cpp-python using {wheel_tag}")
        return 0

    section("Choose installer action", 1)
    try:
        action = select_install_action(
            runtime_exists=existing_runtime,
            install_requested=args.install,
            repair_requested=args.repair,
            assume_yes=args.yes,
        )
    except ValueError as exc:
        error(f"REASON: {exc}")
        return 1
    success(f"Selected {action} mode")

    section("System check", 2)

    if not verify_python():
        return 1

    gpu = detect_nvidia()
    print_gpu(gpu)

    section("Virtual environment", 3)
    if action == "install":
        try:
            create_environment()
        except Exception as exc:
            report_exception("Unable to create the virtual environment", exc)
            return 1
    else:
        success("Existing managed runtime selected for repair")
        detail("Location", relative_path(VENV_DIR))

    python = str(venv_python())

    section("Core dependencies", 4)
    try:
        verify_managed_python(python)
        cleanup_invalid_distributions()
        install_dependencies(python, gpu, force_reinstall=action == "install")
    except subprocess.CalledProcessError as exc:
        error("Dependency installation failed with " f"exit code {exc.returncode}.")
        return 1

    section("Inference backend", 5)
    try:
        wheel_tag = install_llama(
            python,
            gpu,
            preference=args.backend,
            interactive=False,
            force_reinstall=action == "install",
        )
    except subprocess.CalledProcessError as exc:
        error(
            "llama-cpp-python installation failed " f"with exit code {exc.returncode}."
        )
        return 1
    except LlamaRuntimeError as exc:
        error(str(exc))
        return 1

    section("Verification", 6)

    try:
        verify_installation(python)

    except subprocess.CalledProcessError:
        error("Dependency verification failed.")
        return 1

    # Install can also replace an existing backend with a different CPU/CUDA
    # wheel, so either successful action invalidates prior validation results.
    repair_validation_cache()
    success("Disposable model validation cache refreshed")

    print_final_health(python, gpu, libraries_verified=True)
    completion_screen(
        gpu,
        wheel_tag,
        action=action,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        report_keyboard_interrupt("the installer")
        raise SystemExit(130)
    except Exception as exc:
        report_exception("AIBrain installer failed", exc)
        raise SystemExit(1)
