# Configuration

## Runtime environment

The application requires the project `.venv` at launch. There is no settings file to edit for dependency selection; use
`cli/installer.py` to create or repair the managed environment.

`cli/main.py` configures a desktop OpenGL 3.3 core-profile request before creating the Qt application. When
high-performance rendering is selected (the default), it writes the Windows preference for both virtual-environment
Python hosts and relaunches once before Qt creates an OpenGL context. The renderer overlay then verifies the actual
OpenGL adapter.

OpenGL draws the interface and connectome; CUDA runs compute workloads. On a hybrid-GPU laptop,
OpenGL may report Intel while CUDA uses NVIDIA. The standalone diagnostics window probes its own
OpenGL context using Windows' current adapter selection; it does not use the main launcher's GPU
relaunch supervisor. Its `OPENGL` log describes rendering only.

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
