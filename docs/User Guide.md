# User guide

## Chat

Choose a validated model in the selector, write a text prompt, and press **Send**. The model streams its response into
the chat pane while the connectome responds to each received token.

### Infinite simulation

Infinite Mode treats the compose text as the first World event. When the compose box is empty, the send action becomes
**Random** and chooses one of ten premade World events. When it contains text, the arrow sends that text directly as the
first World event. The Participant always responds first, then World and Participant turns alternate. Infinite Mode applies
an internal high-randomness configuration (temperature at least 1.25 and top-p at least .96),
regardless of the Normal Chat controls. Switching between Normal Chat and Infinite Mode clears both the conversation and
the replay. The compose action becomes Stop while either mode is running; Continue is available only for a stopped Infinite scenario.

Open **Mode Actions**, enable **Infinite Mode**, then use the compose arrow to start an open-ended local roleplay. AIBrain loads two separate copies of the selected GGUF: the
**World** produces only external events; the **Participant** produces only first-person responses. Every stream update
is tied to both its role and turn number, so World tokens cannot append to a Participant bubble. **Continue** reconstructs
the two role histories from the stopped transcript rather than restarting the scenario. The UI clearly labels
this as a simulation—the model is generating roleplay text, not a real sentient being. The session continues until
**Stop** is pressed, while each model keeps only a bounded recent context window so it can run without unbounded prompt
growth. The right-side connectome renders and records Participant activity only; World narration remains in the chat.

## NN Analysis+

**NN Analysis+** replaces the compact analysis action. It creates a compact JSON or compressed JSON data file with the
full conversation and a trained online autoencoder's findings: reconstruction/coherence metrics, regional patterns,
temporal segments, and selected high-novelty events. It processes every recorded visual frame but does not dump every
neuron vector. These signals are the connectome's procedural visualization data; they are not measured model hidden
states or transformer activations.

- **Stop** requests cancellation after the current native generation step.
- **Escape** does the same from anywhere in the application.
- **Regenerate** removes the latest assistant answer and resends its preceding user prompt.
- **Clear** removes visible messages, replay/analysis readiness, and in-memory conversation history.
- `Ctrl+C` in the launch terminal requests a clean application shutdown.

The latest completed answer has replay controls beneath it. **Replay** becomes **Stop replay** while it owns the graph;
the token arrows and Exit Rewind action are disabled during that run. Manual Previous/Next inspection holds the selected
activity frame steady until replay or rewind exits. A new response, model selection, graph-quality change, or spacing
change replaces that one-response replay buffer.

## Generation controls

The controls at the bottom of the chat pane are saved in Windows application settings when a response starts.

| Control          | Effect                                                                                                         |
|------------------|----------------------------------------------------------------------------------------------------------------|
| Temperature      | Sampling randomness, from 0 to 2.                                                                              |
| Top-p            | Nucleus-sampling cutoff, from 0.05 to 1.                                                                       |
| Max tokens       | Target upper limit; AIBrain permits a short grace window to finish the current sentence.                       |
| Context          | Context window requested from llama.cpp.                                                                       |
| GPU layers       | `-1` requests automatic/all-layer offload where the installed backend supports it; `0` requests CPU inference. |
| Analysis+ RAM    | Infinite-mode RAM budget for the rewind tail. Older compact analysis records move to the temporary cache.       |
| Analysis+ cache  | Temporary `.cache/temp` budget for paged Analysis+ records. Defaults to 1 GB; `0` disables it.                  |
| Generation speed | UI token pacing from 0.1x to 1.0x. It does not change model sampling.                                          |

## Connectome controls

Open **View settings** for Simulation Performance, Rendering GPU, Spacing, and Neuron Borders (0–1). **Reset view** is
always available beside the connectome title. In 2D, drag to pan and use the wheel to zoom; picking follows the
translated view. Full 2D keeps all nine named clusters in separate positions. Choose **Rewind** after a normal response
to lock chat actions and move Previous, Replay, and Next below the graph. The same left-side control becomes **Exit
Rewind** while inspection is active; replay works in both 2D and 3D.

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
outline width. Region clusters use a stable high-contrast palette for AIBrain's dark default background. Choose **Colour
settings** beneath the connectome title to customize every semantic interface colour with Qt's wheel, RGB, and hexadecimal
controls; settings are saved to `.cache/aibrain.color.json`. Sector/region mapping colours remain stable. All message
bubbles and status/inspector text can be highlighted and copied. These controls affect the visual simulation only; they
never modify model weights or inference.

## Analysis and export

After generating a response, choose **NN Analysis+** to create its compact neural-analysis JSON. Use **Export chat** for
a complete conversation-only JSON or readable text file. Full semantics and file layout are documented in
[Connectome and analysis](Connectome%20and%20Analysis.md). Chat bubbles render safe Markdown
for headings, emphasis, inline code, and HTTP (S) links while preserving selectable source text. Generated chat follows
the newest token only while the message list is already at the bottom; scroll up to read or copy earlier text without
being pulled back down.
