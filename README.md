# AIBrain

AIBrain is a Windows desktop application for chatting with GGUF models already installed by Ollama. It loads the existing model blob directly through `llama-cpp-python`; it does not call the Ollama server, redownload weights, or copy multi-gigabyte files.

Alongside chat, AIBrain renders a live, interactive biological-style connectome. The graph is a visualization layer mapped to logical transformer computation—not the model's literal physical neural topology.

## Requirements

- Windows 10/11
- Python 3.11 or newer
- An installed Ollama GGUF model (for example, `ollama pull gemma3:4b`)
- A Python virtual environment. AIBrain deliberately refuses to start outside one.

The application does not need the Ollama desktop application or server running after models have been installed.

## First-time setup

Open PowerShell in this project folder and run the installer:

```powershell
py -3.11 install.py
```

The installer detects NVIDIA hardware and the CUDA capability reported by its driver *before* dependency installation. It creates `.venv`, installs PySide6/NumPy/ModernGL, then installs a compatible official prebuilt `llama-cpp-python` CUDA wheel when one is published. It automatically uses the official CPU wheel if no compatible NVIDIA wheel is available. It never attempts the fragile local C++ source build that caused the prior MinGW/OpenMP linker failure.

If PowerShell blocks activation for your user account, run this once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

For Command Prompt activation instead, use:

```bat
.venv\Scripts\activate.bat
```

## Run

Activate the environment in every new terminal, then start the program:

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

The program exits with clear setup instructions if it is launched with the global Python interpreter. This protects the system Python installation and ensures the application always uses its declared dependencies. `install.py` is the only exception: it is designed to be run by a system Python specifically to create and populate `.venv`.

## Ollama model discovery

At startup AIBrain recursively reads:

```text
%USERPROFILE%\.ollama\models\manifests
%USERPROFILE%\.ollama\models\blobs
```

For every manifest, it resolves the content-addressed blob named by the model layer, checks for the `GGUF` file header, and shows a human-readable model entry. Missing blobs, invalid JSON, and non-GGUF layers are treated as unavailable models with an actionable error instead of crashing the application.

Nothing is hardcoded to a specific username or model path. The selected GGUF stays loaded between messages and is unloaded when a different model is selected.

## Chat controls

- **Model selector** — choose any validated GGUF found in the local Ollama store.
- **Send** — submit a text-only prompt through the multiline input field.
- **Stop** — stops generation after the current native generation step.
- **Regenerate** — removes the latest assistant response and generates again from the prior user prompt.
- **Clear** — clears the in-memory conversation context and visible messages.
- **Generation settings** — temperature, top-p, maximum generated tokens, context length, and GPU layers are persisted with Windows application settings.

Inference happens in a worker thread. Chat tokens are streamed to the UI so inference does not freeze the interface.

## Connectome visualization

The right pane uses seeded procedural generation to build an organic graph with 5,000, 11,000, or 22,000 visual neurons, depending on the selected quality preset. It includes irregular spatial clusters, functional regions, local connections, cross-region paths, and sparse long-range axon-like links. The same model and quality preset reproduce the same approximate layout across runs.

Controls:

- Drag with the left mouse button to orbit.
- Use the mouse wheel to zoom.
- Click a node to inspect its visual ID, region, current activity, and peak.
- Use **Reset view**, **Pause**, and the **Low / Medium / High** quality selector in the header.

Activity decays smoothly and produces token-driven paths, pulses, and short-lived trails. It is deterministic from observed generation events, rather than meaningless per-frame random flashing.

## Visualization data modes and integrity

AIBrain always displays the current data-source mode. Never interpret the glowing graph as a literal parameter-level network diagram.

| Mode | Meaning |
| --- | --- |
| **Real** | Genuine measured internals supplied by a compatible instrumented backend. |
| **Derived** | Observable model measurements mapped onto visual graph regions. |
| **Simulation** | Procedural activity driven by observable token/model state. |

The direct GGUF/llama.cpp path currently uses **Simulation**. The public `llama-cpp-python` interface does not guarantee hidden-state, attention, or MLP activation access, so AIBrain does not pretend those values are measured. A normalized `ActivationFrame` interface is included so a future compatible Transformers/PyTorch backend can provide real or derived measurements without changing the renderer.

## GPU acceleration

The UI uses a `QOpenGLWidget` renderer and will use the OpenGL implementation available through Qt. The standard `llama-cpp-python` installation is usually CPU-only. For NVIDIA or other accelerated inference, install a current `llama-cpp-python` wheel or build matching your hardware while `.venv` is active, then set **GPU layers** in AIBrain. `-1` requests all layers where the installed backend supports it.

If GPU offload is unsupported, lower the GPU-layers setting to `0` for CPU inference. No CUDA-specific dependency is required for the application itself.

## Troubleshooting

| Problem | Resolution |
| --- | --- |
| `AIBrain must run inside a Python virtual environment` | Activate `.venv` before executing `python main.py`. |
| No models appear | Confirm `%USERPROFILE%\.ollama\models` exists and a GGUF Ollama model was pulled. |
| `llama-cpp-python is not installed` | Re-run `py -3.11 install.py` to repair the managed virtual environment. |
| Model fails to load | Confirm the selected blob is a standard GGUF and reduce context or GPU layers if memory is limited. |
| UI opens but rendering is slow | Select **Low** quality and update the graphics driver. |

## Scope and current limitations

- Text-only local chat; images, tools, RAG, and network services are intentionally out of scope.
- The loaded model must be usable by the installed `llama-cpp-python` build.
- The visualizer is a biological-style computational map, not a reconstruction of every transformer parameter or connection.
- Real activation instrumentation is intentionally optional and is not fabricated when unavailable.
