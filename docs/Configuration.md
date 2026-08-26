# Configuration

## Runtime environment

The application requires the project `.venv` at launch. There is no settings file to edit for dependency selection; use `install.py` to create or repair the managed environment.

`main.py` configures a desktop OpenGL 3.3 core-profile request before creating the Qt application. It also requests Windows' high-performance GPU preference for the active virtual-environment Python executable.

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

The rendering-adapter setting is a Windows preference, not a guarantee. The OpenGL overlay is the source of truth for the adapter actually in use.

## Logging

AIBrain configures concise one-line logs for the terminal and a file handler when the user's writable log location is available. Messages are normalized to a single terminal-width line and include time, level, logger, and message.

For diagnosis, launch AIBrain from an activated PowerShell terminal and retain the relevant one-line log entries with the error text.
