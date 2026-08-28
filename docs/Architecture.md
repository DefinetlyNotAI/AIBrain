# Architecture and development

## Component map

```text
cli/main.py
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

Every GUI entry point except the installer checks the managed virtual environment before importing application modules.
`cli/main.py`, `cli/diagnostic.py`, and `cli/analysis.py` show the shared loader while their first background validation
runs, retain their attached console for runtime output, configure feature-scoped logs (`aibrain.<feature>.log`), and
create a detailed `crash.<feature>.log` only after an uncaught exception. `cli/main.py` additionally requests desktop
OpenGL and installs SIGINT handling.

`MainWindow` owns conversation state and coordinates UI signals. It discovers and validates models, owns the worker
thread, updates the thinking indicator, and saves generation settings.

`GenerationWorker` owns streaming inference work away from the UI thread. It translates each returned text chunk into a
normalized `ActivationFrame` with a data-source label. `ActivityMapper` then applies that frame to the visual activity
field.

## Key modules

| Path                               | Responsibility                                                                |
|------------------------------------|-------------------------------------------------------------------------------|
| `cli/installer.py`                 | Creates and populates the managed Python virtual environment.                 |
| `src/utils/console_ui.py`          | Shared terminal presentation, native console clearing, and safe output wrapping. |
| `cli/build_dist.py`                | Creates timestamped self-contained ai_brain, diagnostic, and analysis distributions. |
| `src/models/ollama_discovery.py`   | Finds and performs lightweight validation of local Ollama blobs.              |
| `src/models/model_validator.py`    | Validates candidates against llama.cpp and caches profile-scoped GGUF results. |
| `src/models/llama_backend.py`      | Loads, streams, tokenizes, and unloads GGUF models.                           |
| `src/connectome/generator.py`      | Creates deterministic visual graph topology.                                  |
| `src/connectome/activity.py`       | Holds decaying activity, peaks, silencing, and importance state.              |
| `src/connectome/renderer.py`       | Batched ModernGL 3D/2D draw pipeline, sector filtering, and interaction.     |
| `src/connectome/analysis.py`       | Online derived visual-stream analysis.                                        |
| `src/native/c/connectome_kernels.c` | Optional native hot-path implementation.                                      |
| `src/native/wrapper/connectome_kernels.py` | ctypes contract and NumPy fallback for `dll/aibrain.connectome.dll`.          |

## Development workflow

1. Create dependencies with `py cli\installer.py`.
2. Activate `.venv` before running validation or the application.
3. Rebuild the DLL after changing C code.
4. Compile Python modules before handing off a change:

   ```powershell
   python -m compileall -q cli src tests
   ```

5. Start the app manually to validate native-window, model-discovery, and GPU behavior on the target machine. Model
   loading and graphics adapter selection depend on local hardware and installed models.

## Contributing principles

- Keep the UI native Qt; do not introduce browser, React, Electron, or WebView components.
- Preserve the venv-only runtime guard.
- Keep data-source labels honest. Do not represent procedural or derived visual data as measured transformer internals.
- Keep heavy inference work off the Qt UI thread.
- Preserve NumPy fallbacks for optional native-DLL acceleration.
- Update the relevant document when user-visible behavior changes.
