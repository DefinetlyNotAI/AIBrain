# Configuration

## Runtime environment

The application requires the project `.venv` at launch. There is no settings file to edit for dependency selection; use `cli/installer.py` to create or repair the managed environment.

`cli/main.py` configures a desktop OpenGL 3.3 core-profile request before creating the Qt application. When high-performance rendering is selected (the default), it writes the Windows preference for both virtual-environment Python hosts and relaunches once before Qt creates an OpenGL context. The renderer overlay then verifies the actual OpenGL adapter.

## Persisted settings

Windows `QSettings` persists the following values when generation starts:

| Key | UI control | Default |
| --- | --- | --- |
| `temperature` | Temperature | 0.7 |
| `top_p` | Top-p | 0.9 |
| `max_tokens` | Max tokens | 256 |
| `context_length` | Context | 4096 |
| `gpu_layers` | GPU layers | -1 |
| `speed` | Generation speed | 1.0 |
| `render_adapter` | Rendering-GPU preference | System default, with NVIDIA preferred when discovered |

The rendering-adapter setting is applied before the Qt/OpenGL process starts. The OpenGL overlay remains the source of truth: a non-NVIDIA adapter is explicitly reported as a GPU mismatch, rather than silently treated as the high-performance renderer.

## Logging

AIBrain configures concise one-line logs for the terminal and a file handler when the user's writable log location is available. Messages are normalized to a single terminal-width line and include time, level, logger, and message.

For diagnosis, launch AIBrain from an activated PowerShell terminal and retain the relevant one-line log entries with the error text.
