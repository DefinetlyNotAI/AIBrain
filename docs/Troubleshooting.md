# Troubleshooting

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

Lower **Context** and set **GPU layers** to `0`, then try again. This reduces memory requirements and rules out
GPU-offload incompatibility. A valid GGUF file may still require a different llama.cpp build or more memory than the
machine has available.

To repair dependencies, run `py cli\installer.py` again. The installer repairs the managed environment rather than the
system Python.

## The renderer uses an integrated GPU

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

## NN Analysis+ does nothing

Generate a response first. NN Analysis+ processes the session's recorded visual frames through its online autoencoder.
Changing model, quality, or spacing discards old replay/analysis data to prevent mixing incompatible graph layouts.
