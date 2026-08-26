# Architecture and development

## Component map

```text
main.py
  └─ PySide6 QApplication and native OpenGL setup
       └─ MainWindow
            ├─ ChatPanel
            ├─ GenerationWorker (QThread)
            │    └─ LlamaBackend → local GGUF through llama-cpp-python
            └─ VisualizerPanel
                 ├─ connectome graph generator
                 ├─ ActivityField and ActivityMapper
                 ├─ ConnectomeRenderer (QOpenGLWidget + ModernGL)
                 └─ ConnectomeAnalyzer / exporter
```

`main.py` checks the virtual environment before importing GUI dependencies, requests desktop OpenGL, configures logging, creates the QApplication, and installs SIGINT handling.

`MainWindow` owns conversation state and coordinates UI signals. It discovers and validates models, owns the worker thread, updates the thinking indicator, and saves generation settings.

`GenerationWorker` owns streaming inference work away from the UI thread. It translates each returned text chunk into a normalized `ActivationFrame` with a data-source label. `ActivityMapper` then applies that frame to the visual activity field.

## Key modules

| Path | Responsibility |
| --- | --- |
| `installer.py` | Creates and populates the managed Python virtual environment. |
| `src/models/ollama_discovery.py` | Finds and performs lightweight validation of local Ollama blobs. |
| `src/models/model_validator.py` | Validates candidates against the installed llama.cpp backend. |
| `src/models/llama_backend.py` | Loads, streams, tokenizes, and unloads GGUF models. |
| `src/connectome/generator.py` | Creates deterministic visual graph topology. |
| `src/connectome/activity.py` | Holds decaying activity, peaks, silencing, and importance state. |
| `src/connectome/renderer.py` | Batched ModernGL draw pipeline and interaction. |
| `src/connectome/analysis.py` | Online derived visual-stream analysis. |
| `src/native/connectome_native.c` | Optional native hot-path implementation. |

## Development workflow

1. Create dependencies with `py installer.py`.
2. Activate `.venv` before running validation or the application.
3. Rebuild the DLL after changing C code.
4. Compile Python modules before handing off a change:

   ```powershell
   python -m compileall -q main.py installer.py src
   ```

5. Start the app manually to validate native-window, model-discovery, and GPU behavior on the target machine. Model loading and graphics adapter selection depend on local hardware and installed models.

## Contributing principles

- Keep the UI native Qt; do not introduce browser, React, Electron, or WebView components.
- Preserve the venv-only runtime guard.
- Keep data-source labels honest. Do not represent procedural or derived visual data as measured transformer internals.
- Keep heavy inference work off the Qt UI thread.
- Preserve NumPy fallbacks for optional native-DLL acceleration.
- Update the relevant document when user-visible behavior changes.
