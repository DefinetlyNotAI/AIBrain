from __future__ import annotations

from .activity import ActivityField
from ..models.instrumented_backend import ActivationFrame


class ActivityMapper:
    """Maps normalized frames onto a visual graph; no claim of physical topology."""
    def __init__(self, field: ActivityField) -> None:
        self.field = field

    def apply(self, frame: ActivationFrame) -> None:
        self.field.update(frame)
