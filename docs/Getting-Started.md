# Getting started

## Requirements

- Windows 10 or Windows 11.
- Python 3.11 or newer, available as `py -3.11`.
- An Ollama model already installed locally. For example:

  ```powershell
  ollama pull gemma3:4b
  ```

- Enough RAM/VRAM for the selected model and context length.

The Ollama application and server do not need to be running after the model has been installed. AIBrain reads its local model store directly and does not copy or redownload model blobs.

## Install into the managed environment

From the project root, run:

```powershell
py cli\installer.py
```

`cli/installer.py` is the sole allowed system-Python entry point. It creates `.venv`, updates pip inside it, installs PySide6, NumPy, ModernGL, and Nuitka, then chooses a prebuilt `llama-cpp-python` wheel. The installer checks `nvidia-smi` first: when a compatible published NVIDIA CUDA wheel is available it uses that; otherwise it installs the official CPU wheel. It does not fall back to a local C/C++ source build.

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

`cli/main.py` exits immediately outside a virtual environment. This is deliberate: it prevents accidental use of global packages and ensures the application runs against the dependency set installed by `cli/installer.py`.

## Standalone distribution

Build a normal, self-contained directory distribution—never a single-file executable—with:

```powershell
.\.venv\Scripts\python.exe cli\build_dist.py
```

The result is `dist\AIBrain_YYYYMMDD_HHMMSS\AIBrain.exe` plus Qt, Python, llama.cpp, native connectome, and required Visual C++ runtime files.

## First run

At launch, AIBrain scans the local Ollama manifests and blobs, then validates each candidate GGUF with the installed backend in a background thread. Choose a validated model from the selector, enter a prompt, and send it. The chat pane shows `Thinking…` until the first generated token arrives.

If no model is listed, see [Troubleshooting](Troubleshooting.md#no-models-are-listed).
