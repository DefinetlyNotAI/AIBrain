from __future__ import annotations

from PySide6.QtCore import QSettings

from ..models.llama_backend import GenerationConfig


def load_generation_settings() -> GenerationConfig:
    settings = QSettings()
    return GenerationConfig(
        temperature=float(settings.value("temperature", .7)), top_p=float(settings.value("top_p", .9)),
        max_tokens=int(settings.value("max_tokens", 256)), context_length=int(settings.value("context_length", 4096)),
        gpu_layers=int(settings.value("gpu_layers", -1)),
    )


def save_generation_settings(config: GenerationConfig) -> None:
    settings = QSettings()
    settings.setValue("temperature", config.temperature)
    settings.setValue("top_p", config.top_p)
    settings.setValue("max_tokens", config.max_tokens)
    settings.setValue("context_length", config.context_length)
    settings.setValue("gpu_layers", config.gpu_layers)
