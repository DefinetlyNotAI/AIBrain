# Connectome and analysis

## What the visualizer represents

The connectome is a deterministic, biological-style computational map—not a literal reconstruction of a transformer's
physical topology or parameters. It has nine labelled regions, including token input, embeddings, early/middle/late
processing, attention and MLP clusters, residual pathways, and output/logits.

Its layout is seeded from the selected model key and quality preset, so the same selection produces the same approximate
graph across runs. The generator builds irregular clusters, local connections, adjacent-region bridges, and sparse
long-range links.

## Data modes

| Mode       | Meaning                                                               |
|------------|-----------------------------------------------------------------------|
| Real       | Measured internal data supplied by a compatible instrumented backend. |
| Derived    | Observable measurements mapped onto visual regions.                   |
| Simulation | Procedural activity tied to observable model/token state.             |

The current llama.cpp GGUF path is **Simulation**. For each generated token it creates a deterministic cascading
activity pattern, applies importance and silencing controls, and lets values decay between frames. This makes playback
repeatable while avoiding a false claim that the displayed nodes are model parameters.

## Rendering

ModernGL renders the graph inside Qt's native OpenGL widget. GPU buffers retain node positions, region IDs, and selected
edge topology; each frame only updates compact activity data. Nodes are color-coded with a stable named-region palette,
so a region retains the exact same color across graph rebuilds, replay, and the fallback renderer. Nodes and edges are
rendered as batched draw calls.

The renderer reports the actual OpenGL renderer in its overlay. AIBrain writes its high-performance preference and
relaunches before Qt creates an OpenGL context; the standalone builder records the same preference for packaged
`ai_brain.exe`. If the actual renderer still differs on the first attempt, AIBrain closes the main window cleanly and
returns to the loader for one supervised retry. On hybrid
laptops, set the active `python.exe` or packaged `ai_brain.exe` to **High-performance NVIDIA processor** in **NVIDIA
Control Panel > Manage 3D settings > Program Settings**, then restart; Windows Graphics settings are a secondary
fallback.

Use **3D View** for the normal depth-aware map. Toggle it to **2D View** for a flat map, then choose either **Full 2D**
or **Sector 2D**. Sector mode provides a region selector and filters both drawing and neuron selection to that region.

## Analysis

Analysis+ persists rolling reconstruction error, novelty, and update magnitude inside its compressed NPZ model. The
dashboard reports Baby/Teen/Adult/Elder state, unmet automatic-transition conditions, persistence age, and whether an
Elder model has frozen weights. These are learning-health signals, not measured model accuracy. Full export semantics
are in the [JSON reference](JSON%20Reference.md).

The 2D map supports drag panning, wheel zoom, and pan-aware picking. Its intentionally larger flat presentation uses a
fixed readable layout, so cluster spacing is disabled while 2D is selected. Replay and token navigation remain usable
in both projection modes and do not depend on renderer pause state.

**Analysis** is available after a completed normal chat and exports `aibrain.session-analysis.v1`: conversation,
generation/visual summary, graph metadata, and recorded-frame summary. It deliberately contains no learned-network
findings. **Analysis+** is available after a completed Infinite-mode run and exports
`aibrain.infinite-analysis-plus.v1`, adding compact autoencoder findings.

Analysis+ runs an actual online NumPy autoencoder over every visual frame, then creates a compact JSON data file.
The network uses region-density and temporal-change features, a tanh encoder, sigmoid decoder, and per-frame
gradient-descent reconstruction training. Every recorded frame is atomically saved immediately, so its learned weights
survive an unexpected close as well as a normal exit. The model is stored at
`models\aibrain.analyser.npz` and is intentionally gitignored. The older
`%LOCALAPPDATA%\AIBrain\analysis_model\connectome_autoencoder_v1.npz` location is imported once when present.
The standalone `analysis.py` tool reads this NPZ without loading a GGUF or renderer and reports its integrity, age,
architecture, tensor statistics, and transparent learning-maturity estimate. Region-density normalization removes static cluster-size and global
renderer-amplitude bias; findings remain analysis of procedural visual signals, not measured transformer activations.
The export contains the complete session conversation, network architecture and fit metrics, regional profile, pattern
segments, and a bounded set of high-novelty events. It intentionally excludes massive per-neuron frame dumps.

This is analysis of AIBrain's visual activity stream. It is not an inspection of model hidden states, attention,
weights, or reasoning.

## NN Analysis+ format

Each data file includes an integrity statement so exports remain correctly interpreted after they leave the application:

```text
An online neural network analyzed every recorded simulated visual-connectome frame. Findings are not measured transformer activations.
```
