# Troubleshooting

## The installer reports a missing PySide6 module

The installer must work before application dependencies are installed. Use the current `cli/installer.py` and
`src/models/diagnostics.py`; structural model diagnostics defer their Qt-dependent validator until backend validation
is requested. Run `python cli\installer.py` again. Installing PySide6 globally is unnecessary.

An existing empty `.venv` is supported. The installer populates it, and restores pip with Python's bundled
`ensurepip` if that environment was created without pip.

## AIBrain says it must run inside a virtual environment

Activate the managed environment before launching:

```powershell
.\.venv\Scripts\Activate.ps1
python cli\main.py
```

If `.venv` does not exist, create it with `py cli\installer.py`.

## PowerShell blocks activation

Allow locally created scripts for the current user, then open a new PowerShell window:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Alternatively use Command Prompt and `.venv\Scripts\activate.bat`.

## No models are listed

Confirm that Ollama has installed a compatible GGUF model and that the current user has a `.ollama\models` directory.
Reopen AIBrain after running, for example, `ollama pull gemma3:4b`. AIBrain hides missing, malformed, and
backend-incompatible blobs by design.

## A selected model fails to load or generate

If a CUDA wheel reports that `llama.dll` or one of its dependencies cannot be found, update the installer and runtime
loader together, then run `python cli\installer.py --repair --repair-subsystem backend --backend cuda -y`.
The installer ensures the matching NVIDIA runtime packages exist. AIBrain's loader makes their DLLs visible to both
Windows DLL loading mechanisms before importing llama.cpp; a full system CUDA Toolkit installation is not required.

Run the installer in Repair mode to reinstall the managed dependencies and selected llama.cpp backend, then review its
final health report: `py cli\installer.py -y --repair`. Repair is available only after the managed runtime has been
installed. It clears the disposable validation cache so a stale backend failure is not reused. Each failure includes a
`REASON`; model blobs and native DLLs are never removed by the standard repair flow.

Lower **Context** and set **GPU layers** to `0`, then try again. This reduces memory requirements and rules out
GPU-offload incompatibility. A valid GGUF file may still require a different llama.cpp build or more memory than the
machine has available.

To repair dependencies, run `py cli\installer.py -y --repair`. The installer repairs the managed environment rather
than the system Python.

## The renderer uses an integrated GPU

The Windows adapter preference is only a preference signal. The startup loader and live overlay record the actual
OpenGL vendor/renderer, which is the source of truth. If AIBrain reports a mismatch after a restart, add the exact
Python or packaged executable in **Windows Settings > System > Display > Graphics**, choose High performance, update the
driver, then restart. The diagnostics dashboard records the same repair path.

Choose the NVIDIA/high-performance adapter in AIBrain and restart the entire application. AIBrain writes and reads back
the Windows high-performance preference for the active Python or packaged executable before OpenGL is created, and
re-applies it if a mismatch is detected. An existing OpenGL context cannot migrate adapters. If the overlay still
reports `vendor=Intel` while an NVIDIA GPU is installed, the Qt/WGL hybrid-GPU driver decision overrode that request:
open **NVIDIA Control Panel > Manage 3D settings > Program Settings**, add the active `python.exe` (or packaged
`ai_brain.exe`), choose **High-performance NVIDIA processor**, apply, and restart.

If the overlay still identifies the integrated renderer, open **Windows Settings > System > Display > Graphics**, add
the project `.venv\Scripts\python.exe`, select **Options**, choose **High performance**, and restart AIBrain. The
overlay reports the actual selected renderer.

## The visualizer is choppy

Choose **Low** quality, reduce model GPU layers if inference competes for the same GPU, update the graphics driver, and
close other GPU-heavy programs. The right pane uses GPU batched rendering, but integrated GPUs and high-quality graphs
can still be constrained by the display adapter.

## Native DLL build fails

Run the build tool from a terminal with a supported C compiler available:

```powershell
py cli\build_native.py --compiler "C:\path\to\compiler.exe"
```

See [Native acceleration](Native%20Acceleration.md) for supported toolchains and all build options.

## Distribution build stalls while processing `glcontext`

Use the maintained distribution command, rather than invoking Nuitka directly:

```powershell
.\.venv\Scripts\python.exe cli\build_dist.py
```

The builder excludes ModernGL's optional `glcontext` packaging hook. AIBrain renders through Qt's current Windows OpenGL
context, so the hook is not needed in the distribution and excluding it avoids a known long-running Nuitka import pass.

## NN Analysis+ does nothing

Complete an Infinite-mode run first. Analysis+ processes the session's recorded visual frames through its online
autoencoder. Complete a normal chat to enable the separate non-NN **Analysis** export. Changing model, quality, or
spacing discards old replay/analysis data to prevent mixing incompatible graph layouts.
