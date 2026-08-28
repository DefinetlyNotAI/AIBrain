# Models and inference

## Local Ollama discovery

AIBrain searches these directories below the current user's home folder:

```text
.ollama\models\manifests
.ollama\models\blobs
```

Each manifest is parsed to locate its content-addressed layer blob. A candidate is only presented when its blob exists,
has a valid `GGUF` header, matches an expected manifest size when supplied, and can be loaded by the installed
`llama-cpp-python` backend. Validation happens in a background Qt thread so the window remains responsive.

Invalid JSON, missing layers, non-GGUF blobs, and models the backend cannot load are rejected instead of offered for
chat. This avoids a selector full of known-broken entries.

## Loading and lifecycle

Validation saves profile-specific results in `.cache/aibrain.<model_name>.cache`. A cached result is reused only when
the exact blob path, size, and modification timestamp match and the requested validation profile is the same. Changing
the blob automatically invalidates that entry; a structural header check never substitutes for a requested full
`llama-cpp-python` compatibility check.

AIBrain loads a selected GGUF directly from its Ollama blob path using `llama-cpp-python`. It does not call the Ollama
server, duplicate a model file, or require Ollama to remain running. Switching models unloads the previous backend,
clears chat history, and rebuilds the deterministic visual graph for the new model key.

Generation runs in a worker `QThread`. Tokens stream to the main UI through Qt signals. The stop flag is checked between
generated chunks and while any user-selected speed delay is waiting.

## Inference acceleration

Inference GPU offload is distinct from connectome rendering. The installer chooses a compatible CUDA wheel only when the
driver reports NVIDIA capability and a matching official prebuilt wheel is available. GPU offload also depends on the
selected model, VRAM, and the installed llama.cpp backend.

Set **GPU layers** to `-1` to request automatic/all-layer offload where supported, `0` to force CPU inference when
accelerated loading fails or memory is insufficient, or a positive number to request that number of model layers.

If a model fails during startup validation or generation, reduce context length and GPU layers first. A GGUF that is
valid as a file can still be incompatible with a particular llama.cpp build or exceed available memory.

## Data-source honesty

The public direct GGUF interface does not guarantee access to hidden states, attention values, or MLP activations. The
current backend therefore emits `Simulation` frames based on observable token events. The visualizer presents that
status in its mode label and overlay. It never labels simulation as real model-neuron data.
