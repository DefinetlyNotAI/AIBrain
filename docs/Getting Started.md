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

The installer finishes with a read-only health report for hardware/CUDA fallback, Python, pip/libraries, model
manifests and blobs, `.cache`, and native DLLs. For a narrow repair, use one selected subsystem instead of reinstalling
everything: `python cli\installer.py --repair dependencies`, `--repair backend`, `--repair native`, or `--repair cache`.
Model repair requires an explicit reference, for example `python cli\installer.py --repair models --model llama3:latest`;
the installer never deletes model blobs implicitly.

From the project root, run:

```powershell
py cli\installer.py
```

`cli/installer.py` is the sole allowed system-Python entry point. It creates `.venv`, updates pip inside it, installs
PySide6, NumPy, ModernGL, and Nuitka, then chooses a prebuilt `llama-cpp-python` wheel. The installer checks
`nvidia-smi` first: when a compatible published NVIDIA CUDA wheel is available it uses that; otherwise it installs the
official CPU wheel. It does not fall back to a local C/C++ source build.

If PowerShell blocks activation, make the current user policy permit local scripts, then open a new terminal:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Launch

Activate the environment every time you open a new terminal:

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

## First run

At launch, AIBrain first shows a loading window while it scans local Ollama manifests and blobs, validates each
candidate GGUF with the installed backend in a background thread, and checks the selected OpenGL adapter. The main
window opens only after startup completes. Choose a validated model from the selector, enter a prompt, and send it.
The chat pane shows `Thinking…` until the first generated token arrives.

If no model is listed, see [Troubleshooting](Troubleshooting.md#no-models-are-listed).
