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
`ai_brain.exe`. If the overlay still reports a non-NVIDIA renderer, it explicitly calls out the mismatch. On hybrid
laptops, set the active `python.exe` or packaged `ai_brain.exe` to **High-performance NVIDIA processor** in **NVIDIA
Control Panel > Manage 3D settings > Program Settings**, then restart; Windows Graphics settings are a secondary
fallback.

## Analysis

**NN Analysis+** runs an actual online NumPy autoencoder over every visual frame, then creates a compact JSON data file.
The network uses region-density and temporal-change features, a tanh encoder, sigmoid decoder, and per-frame
gradient-descent reconstruction training. Every recorded frame is atomically saved immediately, so its learned weights
survive an unexpected close as well as a normal exit. The NPZ lives in `%LOCALAPPDATA%\AIBrain\analysis_model\` rather
than the application folder, so upgrades and standalone releases keep the same per-user memory; an older project-local
NPZ is migrated automatically on first use. Region-density normalization removes static cluster-size and global
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
