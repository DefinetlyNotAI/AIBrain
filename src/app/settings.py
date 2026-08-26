from __future__ import annotations

from PySide6.QtCore import QSettings

from ..models.llama_backend import GenerationConfig


def _setting_float(settings: QSettings, key: str, default: float) -> float:
    value = settings.value(key, default)
    if isinstance(value, (int, float, str)):
        return float(value)
    return default


def _setting_int(settings: QSettings, key: str, default: int) -> int:
    value = settings.value(key, default)
    if isinstance(value, (int, str)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    return default


def load_generation_settings() -> GenerationConfig:
    settings = QSettings()

    return GenerationConfig(
        temperature=_setting_float(settings, "temperature", .7),
        top_p=_setting_float(settings, "top_p", .9),
        max_tokens=_setting_int(settings, "max_tokens", 256),
        context_length=_setting_int(settings, "context_length", 4096),
        gpu_layers=_setting_int(settings, "gpu_layers", -1),
        speed=_setting_float(settings, "speed", 1.0),
    )


def save_generation_settings(config: GenerationConfig) -> None:
    settings = QSettings()
    settings.setValue("temperature", config.temperature)
    settings.setValue("top_p", config.top_p)
    settings.setValue("max_tokens", config.max_tokens)
    settings.setValue("context_length", config.context_length)
    settings.setValue("gpu_layers", config.gpu_layers)
    settings.setValue("speed", config.speed)
