# Connectome and analysis

## What the visualizer represents

The connectome is a deterministic, biological-style computational map—not a literal reconstruction of a transformer's physical topology or parameters. It has nine labelled regions, including token input, embeddings, early/middle/late processing, attention and MLP clusters, residual pathways, and output/logits.

Its layout is seeded from the selected model key and quality preset, so the same selection produces the same approximate graph across runs. The generator builds irregular clusters, local connections, adjacent-region bridges, and sparse long-range links.

## Data modes

| Mode | Meaning |
| --- | --- |
| Real | Measured internal data supplied by a compatible instrumented backend. |
| Derived | Observable measurements mapped onto visual regions. |
| Simulation | Procedural activity tied to observable model/token state. |

The current llama.cpp GGUF path is **Simulation**. For each generated token it creates a deterministic cascading activity pattern, applies importance and silencing controls, and lets values decay between frames. This makes playback repeatable while avoiding a false claim that the displayed nodes are model parameters.

## Rendering

ModernGL renders the graph inside Qt's native OpenGL widget. GPU buffers retain node positions, region IDs, and selected edge topology; each frame only updates compact activity data. Nodes and edges are rendered as batched draw calls.

The renderer reports the actual OpenGL renderer in its overlay. AIBrain writes its high-performance preference and relaunches before Qt creates an OpenGL context. If the overlay still reports a non-NVIDIA renderer, it explicitly calls out the mismatch; set the virtual-environment `python.exe` in **Windows Settings > System > Display > Graphics** to High performance and restart.

## Analysis

**NN Analysis+** runs a small online adaptive encoder over visual activity, then creates a detailed JSON data file. It calculates a regional feature vector, updates a learned prototype, and records a novelty score for each token frame. The export contains the complete session conversation, every captured visual value and peak array, active-neuron indices, regional aggregates, topology, and derived novelty events. Choose the compressed JSON option for large sessions.

This is analysis of AIBrain's visual activity stream. It is not an inspection of model hidden states, attention, weights, or reasoning.

## NN Analysis+ format

Each data file includes an integrity statement so exports remain correctly interpreted after they leave the application:

```text
Conversation and complete visual-connectome signal capture. Brain signals are simulated visual activity, not measured transformer activations.
```
