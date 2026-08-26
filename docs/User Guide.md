# User guide

## Chat

Choose a validated model in the selector, write a text prompt, and press **Send**. The model streams its response into
the chat pane while the connectome responds to each received token.

### Infinite simulation

Press **∞ Inf** to start an open-ended local roleplay. AIBrain loads two separate copies of the selected GGUF: the
**World** produces only external events; the **Participant** produces only first-person responses. Every stream update
is tied to both its role and turn number, so World tokens cannot append to a Participant bubble. The UI clearly labels
this as a simulation—the model is generating roleplay text, not a real sentient being. The session continues until
**Stop** is pressed, while each model keeps only a bounded recent context window so it can run without unbounded prompt
growth. The right-side connectome renders participant tokens only.

## NN Analysis+

**NN Analysis+** replaces the compact analysis action. It creates a compact JSON or compressed JSON data file with the
full conversation and a trained online autoencoder's findings: reconstruction/coherence metrics, regional patterns,
temporal segments, and selected high-novelty events. It processes every recorded visual frame but does not dump every
neuron vector. These signals are the connectome's procedural visualization data; they are not measured model hidden
states or transformer activations.

- **Stop** requests cancellation after the current native generation step.
- **Escape** does the same from anywhere in the application.
- **Regenerate** removes the latest assistant answer and resends its preceding user prompt.
- **Clear** removes visible messages and in-memory conversation history.
- `Ctrl+C` in the launch terminal requests a clean application shutdown.

The latest completed answer has replay controls beneath it. **Replay neurons** plays its captured activity sequence;
**Token** arrows move one recorded step at a time. A new response, model selection, graph-quality change, or spacing
change replaces that one-response replay buffer.

## Generation controls

The controls at the bottom of the chat pane are saved in Windows application settings when a response starts.

| Control          | Effect                                                                                                         |
|------------------|----------------------------------------------------------------------------------------------------------------|
| Temperature      | Sampling randomness, from 0 to 2.                                                                              |
| Top-p            | Nucleus-sampling cutoff, from 0.05 to 1.                                                                       |
| Max tokens       | Upper limit for a generated answer.                                                                            |
| Context          | Context window requested from llama.cpp.                                                                       |
| GPU layers       | `-1` requests automatic/all-layer offload where the installed backend supports it; `0` requests CPU inference. |
| Generation speed | UI token pacing from 0.1x to 1.0x. It does not change model sampling.                                          |

## Connectome controls

The right pane is a native `QOpenGLWidget` renderer. It supports left-drag to orbit, mouse-wheel zoom, and **Reset
view** to restore the camera. **Pause** stops visual updates; press **Resume** to continue. **Low**, **Medium**, and
**High** quality graphs contain respectively 5,000, 11,000, and 22,000 visual neurons. The **Spacing** slider shows its
selected multiplier immediately and rebuilds the procedural cluster layout after you pause movement, avoiding repeated
expensive graph rebuilds while dragging. Region colors remain visible even before activity begins.

The GPU selector asks Windows to use the high-performance adapter for the project Python executable on the next launch.
The overlay always identifies the renderer actually selected by OpenGL.

Click a visual neuron to inspect its region, current value, and peak. You can silence it (zeroing its visual activity)
or change its importance from 0 to 3. Click the displayed **Selected neuron** number to enter an exact neuron index.
**Neuron borders** outlines every visual neuron for clearer separation; use the adjacent numeric control to tune the
outline width. Region clusters use a stable distinct color palette. All message bubbles and status/inspector text can be
highlighted and copied. These controls affect the visual simulation only; they never modify model weights or inference.

## Analysis and export

After generating a response, choose **NN Analysis+** to create its compact neural-analysis JSON. Full semantics and file
layout are documented in [Connectome and analysis](Connectome%20and%20Analysis.md). Chat bubbles render safe Markdown
for headings, emphasis, inline code, and HTTP (S) links while preserving selectable source text. Generated chat follows
the newest token only while the message list is already at the bottom; scroll up to read or copy earlier text without
being pulled back down.
