# Configuration

## Runtime environment

The application requires the project `.venv` at launch. There is no settings file to edit for dependency selection; use
`cli/installer.py` to create or repair the managed environment.

The main, diagnostics, and analysis launchers request desktop OpenGL 3.3 before creating the Qt application.
When high-performance rendering is selected (the default), they write the Windows preference and start a fresh
supervised process before creating an OpenGL context. Source launches register both virtual-environment hosts
and their base `python.exe` / `pythonw.exe`: Windows venv launchers redirect to that base process image.
This preference also applies to other applications using the same base Python executable. Packaged launches
register their own executable. The renderer overlay and startup probe report the actual OpenGL adapter.

OpenGL draws the interface and connectome; CUDA runs compute workloads. On a hybrid-GPU laptop,
OpenGL may report Intel while CUDA uses NVIDIA. All desktop launchers now apply the same saved rendering
preference; choosing System default leaves selection to Windows. The `OPENGL` log describes rendering only.

Diagnostics checks CUDA separately in a background worker, even when no models are installed.
The `CUDA` log and GPU / CUDA card report a synchronized CuPy device operation and whether
llama.cpp loads with GPU offload support. CPU fallbacks and library-load failures include their
reason. Offload support alone does not prove that a particular model can run on the GPU.

## Persisted settings

Windows `QSettings` persists the following values when generation starts:

| Key              | UI control               | Default                                               |
|------------------|--------------------------|-------------------------------------------------------|
| `temperature`    | Temperature              | 0.7                                                   |
| `top_p`          | Top-p                    | 0.9                                                   |
| `max_tokens`     | Max tokens               | 256                                                   |
| `context_length` | Context                  | 4096                                                  |
| `gpu_layers`     | GPU layers               | -1                                                    |
| `speed`          | Generation speed         | 1.0                                                   |
| `render_adapter` | Rendering-GPU preference | System default, with NVIDIA preferred when discovered |

The rendering-adapter setting is applied before the Qt/OpenGL process starts. The OpenGL overlay remains the source of
truth: a non-NVIDIA adapter is explicitly reported as a GPU mismatch, rather than silently treated as the
high-performance renderer.

## Logging

`main`, `diagnostic`, and `analysis` configure logging before their workers start. INFO, DEBUG, WARNING, and ERROR
records go both to the attached console and to their feature-specific files under `logs/`; uncaught exceptions also
create a lazy feature-specific crash log. Terminal progress is rendered as a live line rather than duplicated frames.
Optional CUDA probing silently selects the NumPy CPU fallback when CuPy, a compatible device, or its runtime is not
available; it does not interrupt startup or print an import traceback.

AIBrain writes `logs\aibrain.log` and `logs\crash.log` below the project or packaged application root. Both files are
reset at startup and bounded to 5 MiB while a run is active. The terminal renders clean UI-style messages without
timestamp, severity, or source columns; the log files retain that context for diagnosis. Multi-line records preserve
their lines with an aligned continuation gutter rather than being truncated.

For diagnosis, launch AIBrain from an activated PowerShell terminal and retain the relevant log entries with the error
text.
