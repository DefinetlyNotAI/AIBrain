# AIBrain

AIBrain is a native Windows desktop application for chatting with GGUF models already installed by Ollama and exploring
token-driven activity through a live connectome visualization.

It is built with Python and PySide6/Qt. The interface is a real Windows window with a native OpenGL context—there is no
React, Electron, browser, or WebView layer.

## Quick start

Requirements: Windows 10/11, Python 3.11+, and at least one locally installed Ollama GGUF model.

```powershell
py cli\installer.py
.\.venv\Scripts\Activate.ps1
python cli\main.py
```

AIBrain only runs from its managed virtual environment. The installer creates and populates `.venv`, checks NVIDIA/CUDA
capability before choosing an inference wheel, and never installs packages into the system Python.

The GUI tools (`main.py`, `diagnostic.py`, and `analysis.py`) keep their attached console for compact runtime messages and
open a loader window before their first background validation completes. Logs are feature-scoped under `logs/` (for
example, `aibrain.main.log`); a detailed `crash.<feature>.log` is created only after an uncaught exception.

## Documentation

The complete documentation is in [`docs/`](docs/Home.md):

- [Getting started](docs/Getting%20Started.md)
- [User guide](docs/User%20Guide.md)
- [Models and inference](docs/Models%20and%20Inference.md)
- [Connectome and analysis](docs/Connectome%20and%20Analysis.md)
- [JSON reference](docs/JSON%20Reference.md)
- [Native acceleration](docs/Native%20Acceleration.md)
- [Architecture and development](docs/Architecture.md)
- [Configuration](docs/Configuration.md)
- [Troubleshooting](docs/Troubleshooting.md)
- [Wiki publishing](docs/Wiki%20Publishing.md)

The `Publish documentation to Wiki` GitHub Actions workflow publishes this folder to the repository wiki after one has
been enabled.

## Development shortcuts

```powershell
# Rebuild and verify the native DLL after editing its C source.
py cli\build_native.py --clean

# Build self-contained ai_brain, diagnostic, and analysis application folders.
.\.venv\Scripts\python.exe cli\build_dist.py

# Verify Python syntax from the managed environment.
python -m compileall -q cli src tests
```

See [Native acceleration](docs/Native%20Acceleration.md) and [Architecture and development](docs/Architecture.md) for
full build, verification, and contributor guidance.
