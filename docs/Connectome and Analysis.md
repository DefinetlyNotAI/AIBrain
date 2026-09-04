# Connectome and analysis

## What the real-time connectome represents

The connectome is a real-time inference telemetry display. Every generated assistant chunk in normal chat updates it
from values observed in the active `llama.cpp` generation path. In Infinite mode, Participant chunks update it while
World chunks remain chat-only. The old token-seeded cascade simulation has been removed.

The graph itself is a deterministic display layout. Its nodes, clusters, and links make the nine telemetry channels
readable; they are not transformer neurons, learned weights, physical connections, layers, attention heads, or MLP
blocks. Layout generation uses the selected model key, quality preset, and spacing, so the same settings produce the
same visual arrangement. Layout randomness never creates or changes a telemetry value.

The **REAL-TIME** label means that the values came from the current generation as it ran. It does not claim access to
model internals that `llama-cpp-python` does not expose.

## Measurement channels

The backend installs a read-only logits processor for each completion. It inspects the model's raw next-token logits,
returns the same scores object unchanged, and therefore does not affect sampling. These logits are captured before
llama.cpp applies repetition penalties, grammar, top-k/top-p/min-p/typical filtering, temperature, and token selection.
The probability fields below are a softmax of that raw vector, not the final sampling distribution.

| Channel | Value shown from 0 to 1 | Raw value retained in frame/export records |
|---|---|---|
| Context utilisation | Current evaluated context position divided by the requested context limit. | Context position and limit. |
| Requested output progress | Retokenized visible output count divided by the requested maximum; it resets for every reply. | Current reply output count. |
| Stream latency | Time to the first visible chunk, then time since the previous visible chunk, using 500 ms as the full-scale display cap. | Milliseconds. |
| Retokenized throughput | Visible chunk text retokenized with the loaded model, divided by elapsed time, using 50 tokens/s as the full-scale display cap. | Retokenized tokens per second. |
| Raw-logit entropy | Shannon entropy of the raw-logit softmax divided by the maximum entropy for that vocabulary. | Entropy in bits and vocabulary size. |
| Raw top probability | Largest probability in the raw-logit softmax. | Ratio from 0 to 1. |
| Raw top-5 mass | Sum of the five largest probabilities in the raw-logit softmax. | Ratio from 0 to 1. |
| Raw confidence margin | Difference between the two largest raw-logit softmax probabilities. | Ratio from 0 to 1. |
| Recent output rarity | `1 / (1 + occurrences)` for the same emitted text chunk in the previous 32 visible chunks. | Exact recent occurrence count. |

If a generated chunk has no logits snapshot, its four raw-logit channels are zero. Timing, output, and repetition
measurements still come from that live chunk. Infinite-mode World narration stays in chat and does not render or train
Analysis+; only Participant inference telemetry reaches the connectome.

## Display mapping and rendering

Every signal node in a channel receives that channel's normalized value. User importance multiplies only the rendered
intensity, and hiding a node sets only that node's rendered intensity to zero. The overlay, inspector measurement, and
Analysis+ continue to use the unmodified channel value. Display values fade between chunks; measured frame values do
not get rewritten by that fade.

ModernGL renders the graph inside Qt's native OpenGL widget. GPU buffers retain node positions, channel IDs, and display
links; each frame updates signal intensity. Nodes use a stable channel palette across graph rebuilds, replay, and the
fallback renderer. Edges connect the display nodes and become visible from endpoint intensity; they do not assert a
connection inside the model.

The overlay reports the actual OpenGL renderer separately from the CUDA inference backend. AIBrain writes its
high-performance preference and relaunches before Qt creates an OpenGL context; the standalone builder records the
same preference for packaged `ai_brain.exe`. On hybrid laptops, set the active `python.exe` or packaged `ai_brain.exe`
to **High-performance NVIDIA processor** in **NVIDIA Control Panel > Manage 3D settings > Program Settings**, then
restart. Windows Graphics settings are a secondary fallback.

Use **3D View** for the depth-aware map. Toggle it to **2D View** for a flat map, then choose **Full 2D** or **Sector
2D**. Sector mode filters drawing and node selection to one measurement channel. The 2D map supports drag panning,
wheel zoom, and pan-aware picking. Replay and per-frame navigation work in both projections.

## Analysis

**Analysis** is available after a completed normal chat and exports `aibrain.session-analysis.v2`: conversation, graph
metadata, the measurement-integrity statement, and a recorded telemetry summary. It contains no learned-network
findings. **Analysis+** is available after a completed Infinite-mode run and exports
`aibrain.infinite-analysis-plus.v2`, adding compact autoencoder findings.

Analysis+ runs an online autoencoder over the nine normalized measurement channels plus four temporal summary values:
active-channel ratio, mean change, maximum, and standard deviation. It uses a tanh encoder, sigmoid decoder, and
per-frame gradient-descent reconstruction training. It learns from the telemetry values before display gain, hiding,
or decay, so presentation settings cannot contaminate its features.

Rolling reconstruction error, novelty, and update magnitude are persisted in `models\aibrain.analyser.npz`. The
dashboard reports Baby/Teen/Adult/Elder state, unmet automatic-transition conditions, persistence age, and whether an
Elder model has frozen weights. These are training-health signals, not accuracy or cognition claims. Each NPZ stores
the feature schema `aibrain.realtime-inference-telemetry.v1`. Files without that schema may contain the retired
simulated features and are not loaded into the real-time model; generating a new response replaces them with compatible
memory.

Every observed frame is saved atomically. Saves briefly retry Windows sharing violations so a short-lived inspector,
antivirus, or other reader lock does not discard current learning. The standalone `analysis.py` tool closes the NPZ
before formatting it and shows full stored values in **NPZ contents**. **Inspect exported analysis JSON** shows a
selected JSON or JSON.GZ export in the **JSON export** tab and reports invalid files there.

Analysis+ keeps a bounded rewind tail in RAM. Older compact records are written as compressed pages under
`.cache/temp` and streamed into exports. The cache defaults to 1 GB, can be changed in **Advanced settings**, and is
disabled at `0`. When full, it removes oldest pages and shows a warning. Successful export consumes the temporary
pages. Normal shutdown, Python shutdown, and the next launch after a crash remove stale cache directories.

Exports include the conversation, measurement contract, autoencoder architecture and fit metrics, channel profile,
pattern segments, and up to 24 notable events with their raw telemetry. They omit renderer arrays, per-node dumps,
model weights, hidden states, attention matrices, MLP activations, and chain-of-thought.
