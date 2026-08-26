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

The renderer reports the actual OpenGL renderer in its overlay. Windows GPU preference changes only affect a newly created process/context, so restart AIBrain after choosing a high-performance adapter. If Windows still selects the integrated adapter, set the virtual-environment `python.exe` in **Windows Settings > System > Display > Graphics** and restart.

## Analysis

The **Analysis** action runs a small online adaptive encoder over visual activity. It calculates a regional feature vector, updates a learned prototype, and records a novelty score for each token frame. It reports the number of recorded frames, mean and peak novelty, and the most active visual region.

This is analysis of AIBrain's visual activity stream. It is not an inspection of model hidden states, attention, weights, or reasoning.

## Export format

Use **Export** after a response to save the latest analysis.

CSV contains one row per recorded frame with: `step`, `token`, `active_nodes`, `mean_activity`, `peak_activity`, `dominant_region`, and `novelty`.

JSON contains an `integrity` statement, a summary, graph counts/region names, and an `events` array. The integrity statement is intentionally part of every JSON export so files remain correctly interpreted after they leave the application:

```text
Derived analysis of the visual connectome activity stream; not measured transformer activations.
```
