# AIBrain

AIBrain is a native Windows desktop application for chatting with GGUF models already installed by Ollama and exploring token-driven activity through a live connectome visualization.

It is built with Python and PySide6/Qt. The interface is a real Windows window with a native OpenGL context—there is no React, Electron, browser, or WebView layer.

## Quick start

Requirements: Windows 10/11, Python 3.11+, and at least one locally installed Ollama GGUF model.

```powershell
py installer.py
.\.venv\Scripts\Activate.ps1
python main.py
```

AIBrain only runs from its managed virtual environment. The installer creates and populates `.venv`, checks NVIDIA/CUDA capability before choosing an inference wheel, and never installs packages into the system Python.

## Documentation

The complete documentation is in [`docs/`](docs/Home.md):

- [Getting started](docs/Getting-Started.md)
- [User guide](docs/User-Guide.md)
- [Models and inference](docs/Models-and-Inference.md)
- [Connectome and analysis](docs/Connectome-and-Analysis.md)
- [Native acceleration](docs/Native-Acceleration.md)
- [Architecture and development](docs/Architecture.md)
- [Configuration](docs/Configuration.md)
- [Troubleshooting](docs/Troubleshooting.md)
- [Wiki publishing](docs/Wiki-Publishing.md)

The `Publish documentation to Wiki` GitHub Actions workflow publishes this folder to the repository wiki after one has been enabled.

## Development shortcuts

```powershell
# Rebuild and verify the native DLL after editing its C source.
py scripts\build_native.py --clean

# Verify Python syntax from the managed environment.
python -m compileall -q main.py install.py src
```

See [Native acceleration](docs/Native-Acceleration.md) and [Architecture and development](docs/Architecture.md) for full build, verification, and contributor guidance.
