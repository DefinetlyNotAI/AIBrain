# User guide

## Chat

Choose a validated model in the selector, write a text prompt, and press **Send**. The model streams its response into the chat pane while the connectome responds to each received token.

### Infinite simulation

Press **∞ Inf** to start an open-ended local roleplay. AIBrain loads two separate copies of the selected GGUF: the **World** produces the setting and events; the **Participant** responds from inside that fictional setting. The UI clearly labels this as a simulation—the model is generating roleplay text, not a real sentient being. The session continues until **Stop** is pressed, while each model keeps only a bounded recent context window so it can run without unbounded prompt growth. The right-side connectome renders participant tokens only.

## NN Analysis+

**NN Analysis+** replaces the compact analysis action. It creates a JSON or compressed JSON data file with the full conversation, every recorded visual brain-signal frame (complete node values and peaks), active-neuron and regional summaries, graph topology, and derived novelty events. These signals are the connectome's procedural visualization data; they are not measured model hidden states or transformer activations.

- **Stop** requests cancellation after the current native generation step.
- **Escape** does the same from anywhere in the application.
- **Regenerate** removes the latest assistant answer and resends its preceding user prompt.
- **Clear** removes visible messages and in-memory conversation history.
- `Ctrl+C` in the launch terminal requests a clean application shutdown.

The latest completed answer has replay controls beneath it. **Replay neurons** plays its captured activity sequence; **Token** arrows move one recorded step at a time. A new response, model selection, graph-quality change, or spacing change replaces that one-response replay buffer.

## Generation controls

The controls at the bottom of the chat pane are saved in Windows application settings when a response starts.

| Control | Effect |
| --- | --- |
| Temperature | Sampling randomness, from 0 to 2. |
| Top-p | Nucleus-sampling cutoff, from 0.05 to 1. |
| Max tokens | Upper limit for a generated answer. |
| Context | Context window requested from llama.cpp. |
| GPU layers | `-1` requests automatic/all-layer offload where the installed backend supports it; `0` requests CPU inference. |
| Generation speed | UI token pacing from 0.1x to 1.0x. It does not change model sampling. |

## Connectome controls

The right pane is a native `QOpenGLWidget` renderer. It supports left-drag to orbit, mouse-wheel zoom, and **Reset view** to restore the camera. **Pause** stops visual updates; press **Resume** to continue. **Low**, **Medium**, and **High** quality graphs contain respectively 5,000, 11,000, and 22,000 visual neurons. The **Spacing** slider rebuilds the procedural cluster layout.

The GPU selector asks Windows to use the high-performance adapter for the project Python executable on the next launch. The overlay always identifies the renderer actually selected by OpenGL.

Click a visual neuron to inspect its region, current value, and peak. You can silence it (zeroing its visual activity) or change its importance from 0 to 3. **Neuron borders** outlines every visual neuron for clearer separation; use the adjacent numeric control to tune the outline width. These controls affect the visual simulation only; they never modify model weights or inference.

## Analysis and export

After generating a response, choose **Analysis** for a compact novelty and activity summary. Choose **Export** to save either JSON or CSV. Full semantics and file layouts are documented in [Connectome and analysis](Connectome-and-Analysis.md).
