# Troubleshooting

## AIBrain says it must run inside a virtual environment

Activate the managed environment before launching:

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

If `.venv` does not exist, create it with `py -3.11 install.py`.

## PowerShell blocks activation

Allow locally created scripts for the current user, then open a new PowerShell window:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Alternatively use Command Prompt and `.venv\Scripts\activate.bat`.

## No models are listed

Confirm that Ollama has installed a compatible GGUF model and that the current user has a `.ollama\models` directory. Reopen AIBrain after running, for example, `ollama pull gemma3:4b`. AIBrain hides missing, malformed, and backend-incompatible blobs by design.

## A selected model fails to load or generate

Lower **Context** and set **GPU layers** to `0`, then try again. This reduces memory requirements and rules out GPU-offload incompatibility. A valid GGUF file may still require a different llama.cpp build or more memory than the machine has available.

To repair dependencies, run `py -3.11 install.py` again. The installer repairs the managed environment rather than the system Python.

## The renderer uses an integrated GPU

Choose the NVIDIA/high-performance adapter in AIBrain and restart the entire application. Adapter choice happens when Windows creates the native OpenGL context and cannot migrate an existing context.

If the overlay still identifies the integrated renderer, open **Windows Settings > System > Display > Graphics**, add the project `.venv\Scripts\python.exe`, select **Options**, choose **High performance**, and restart AIBrain. The overlay reports the actual selected renderer.

## The visualizer is choppy

Choose **Low** quality, reduce model GPU layers if inference competes for the same GPU, update the graphics driver, and close other GPU-heavy programs. The right pane uses GPU batched rendering, but integrated GPUs and high-quality graphs can still be constrained by the display adapter.

## Native DLL build fails

Run the build tool from a terminal with a supported C compiler available:

```powershell
py -3.11 scripts\build_native.py --compiler "C:\path\to\compiler.exe"
```

See [Native acceleration](Native-Acceleration.md) for supported toolchains and all build options.

## Analysis/export buttons do nothing

Generate a response first. Analysis and export operate on the token frames from the latest response only. Starting another response, changing model, quality, or spacing discards old replay/analysis data to prevent mixing incompatible graph layouts.
