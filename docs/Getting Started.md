# Getting started

## Requirements

- Windows 10 or Windows 11.
- Python 3.11 or newer, available as `py -3.11`.
- An Ollama model already installed locally. For example:

  ```powershell
  ollama pull gemma3:4b
  ```

- Enough RAM/VRAM for the selected model and context length.

The Ollama application and server do not need to be running after the model has been installed. AIBrain reads its local
model store directly and does not copy or redownload model blobs.

## Install into the managed environment

The installer starts with an **Install** or **Repair** menu. **Install** is always the default, including with an existing
environment and when using `-y` without an explicit mode. Repair is unavailable until the managed runtime exists. The
selected mode then automatically uses the recommended dependency and CUDA/CPU backend choices, finishes with a
read-only health report for hardware/CUDA fallback, Python, pip/libraries, model manifests and blobs, `.cache`, and
native DLLs. It never deletes model blobs implicitly.

Before package installation or repair, the installer removes entries beginning with `~` from the managed environment's
`site-packages`. These are pip's incomplete-uninstall leftovers, such as `~umpy` and `~umpy-*.dist-info`.
Cleanup preserves normal packages and refuses paths redirected outside the managed package directory.

From the project root, run:

```powershell
py cli\installer.py
```

`cli/installer.py` is the sole allowed system-Python entry point. It starts with only the Python standard library:
neither the launching interpreter nor a fresh `.venv` needs PySide6 or other application packages preinstalled.
It creates `.venv`, checks that its interpreter is a working Python 3.11+ virtual environment, restores pip with
`ensurepip` if needed, and updates pip inside it. It installs PySide6, ModernGL, Nuitka, and NumPy (with CuPy for
supported CUDA drivers), then chooses a prebuilt `llama-cpp-python` wheel. The installer checks
`nvidia-smi` first: when a compatible published NVIDIA CUDA wheel is available it uses that; otherwise it installs the
official CPU wheel. It does not fall back to a local C/C++ source build.

For unattended installation or repair, choose the mode explicitly and accept the defaults:

```powershell
py cli\installer.py -y --install
py cli\installer.py -y --repair
```

The installer previews commands with PowerShell-safe quoting and streams their output live in the same framed view used
by the distribution builder. In an interactive terminal, the bottom border remains visible while the command runs:
new lines replace the old border and move it down, while progress updates replace their previous rows in place.
Each update writes the output and its border together. Redirected output uses a single closing border when the command ends.
During silent periods, heartbeat and extended-wait messages name the active command, such as `pip install`, `ensurepip`,
or a Python script. Nuitka compilation explanations appear only when Nuitka itself is running.

It requires a binary wheel for `llama-cpp-python` and reinstalls it even when switching
between CPU and CUDA builds with the same version. It confirms that the selected wheel can load its native runtime,
and that a CUDA selection supports GPU offload. In auto mode, if a
CUDA wheel installs but a required DLL cannot load, it retries a fresh official CPU wheel without using the
pip cache. Repair mode also forces a fresh dependency reinstall. Successful installation and repair clear the disposable
validation cache so an earlier backend result cannot mask the updated runtime. Both preserve Ollama model blobs.

CUDA installation includes NVIDIA's cuBLAS and CUDA runtime wheels for the selected CUDA major version. Both the
installer probe and application loader register their DLL directories and add them to the current process's `PATH`.
This includes NVIDIA's `nvidia\cu13\bin` layout; no global `PATH` or `CUDA_PATH` changes are required.

Use `--backend cuda` to require CUDA. This explicit choice fails with a diagnostic if CUDA cannot be installed or
loaded; CPU fallback applies to `--backend auto`. To repair only the inference backend:

```powershell
python cli\installer.py --repair --repair-subsystem backend --backend cuda -y
```

Final verification runs imports and `pip check` in the managed interpreter. Basic model checks inspect local manifests
and GGUF headers without importing Qt or loading models. The optional native connectome DLL is reported as present,
not as successfully loaded. `--help` exits without creating or clearing installer logs.

If PowerShell blocks activation, make the current user policy permit local scripts, then open a new terminal:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Launch

You can launch directly without activating the environment:

```powershell
.\.venv\Scripts\python.exe cli\main.py
```

Alternatively, activate the environment in each new terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python cli\main.py
```

For Command Prompt use:

```bat
.venv\Scripts\activate.bat
python cli\main.py
```

`cli/main.py` exits immediately outside a virtual environment. This is deliberate: it prevents accidental use of global
packages and ensures the application runs against the dependency set installed by `cli/installer.py`.

## Standalone distribution

Build a normal, self-contained directory distribution—never a single-file executable—with:

```powershell
.\.venv\Scripts\python.exe cli\build_dist.py
```

The result is `dist\AIBrain_YYYYMMDD_HHMMSS\` with three self-contained application folders:
`ai_brain\ai_brain.exe`, `diagnostic\diagnostic.exe`, and `analysis\analysis.exe`. Each contains its Qt, Python,
llama.cpp, native connectome, and required Visual C++ runtime files, and runs without activating the development
`.venv`.

The builder invokes Nuitka in unbuffered mode, so its carriage-return progress frames are shown live in an interactive
terminal while each application is compiling. Nuitka can be quiet during C-source generation, C compilation, or linking;
the builder prints a **Still working** heartbeat every 15 seconds during those silent phases so they are not mistaken for
a frozen build.

## Command logs

Every script in `cli\` records its terminal output in `logs\aibrain.<command>.log` while it runs. GUI launchers also
keep their terminal status visible and record it there, so startup, model-validation, and OpenGL failures can be reviewed
after their window closes. Crash traces are recorded separately as `logs\crash.<command>.log`.

## First run

At launch, AIBrain first shows a loading window while it scans local Ollama manifests and blobs, validates each
candidate GGUF with the installed backend in a background thread, and checks the selected OpenGL adapter. The main
window opens only after startup completes. Choose a validated model from the selector, enter a prompt, and send it.
The chat pane shows `Thinking…` until the first generated token arrives.

If no model is listed, see [Troubleshooting](Troubleshooting.md#no-models-are-listed).
