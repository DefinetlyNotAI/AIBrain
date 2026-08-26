# AIBrain documentation

AIBrain is a local Windows research interface for GGUF language models already installed by Ollama. It provides
streaming chat and a native OpenGL connectome visualization driven by observable generation events.

This documentation describes the software as it is implemented. In particular, the current direct GGUF backend does
**not** expose hidden states, attention matrices, or literal model-neuron activity. Its visual connectome and its
analysis are clearly labelled simulation or derived data, never measured model internals.

## Start here

1. [Getting started](Getting%20Started.md) — requirements, managed installation, virtual-environment setup, and first
   launch.
2. [User guide](User%20Guide.md) — chat, playback, the visualizer, and controls.
3. [Models and inference](Models%20and%20Inference.md) — Ollama discovery, GGUF validation, local loading, and
   generation settings.

## Reference

- [Connectome and analysis](Connectome%20and%20Analysis.md)
- [Configuration](Configuration.md)
- [Native acceleration](Native%20Acceleration.md)
- [Architecture](Architecture.md)
- [Troubleshooting](Troubleshooting.md)
- [Wiki publishing](Wiki%20Publishing.md)

## Scope

AIBrain is intentionally a local, text-only application. It does not provide cloud APIs, image chat, tools, retrieval
augmentation, or a browser-based UI. The loaded GGUF must be compatible with the installed `llama-cpp-python` backend.
