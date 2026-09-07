"""Portable exports for normal and Infinite chat transcripts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..models.message_types import TranscriptTurn


def _string_value(value: object, default: str = "") -> str:
    """Return a meaningful string value or the supplied default."""
    return value if isinstance(value, str) else default


def _turn_number(value: object) -> int | None:
    """Return a valid transcript turn number when present."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _serializable_turn(turn: TranscriptTurn) -> dict[str, Any]:
    """Convert a transcript turn into JSON-safe primitive values."""
    return {
        "role": _string_value(turn.get("role"), "unknown"),
        "content": _string_value(turn.get("content")),
        "turn": _turn_number(turn.get("turn")),
    }


def write_chat_export(
        path: Path,
        conversation: Sequence[TranscriptTurn],
        *,
        mode: str,
        model: str,
) -> None:
    """Write the complete chat as structured JSON or readable plain text."""
    path = Path(path)
    created_at = datetime.now(UTC).isoformat()

    turns = [_serializable_turn(turn) for turn in conversation]

    if path.suffix.lower() == ".txt":
        lines = [
            "AIBrain chat export",
            f"Mode: {mode}",
            f"Model: {model}",
            f"Created: {created_at}",
            "",
        ]

        for turn in turns:
            role = turn["role"].upper()
            turn_number = turn["turn"]
            heading = role if turn_number is None else f"{role} {turn_number}"

            lines.extend(
                (
                    heading,
                    turn["content"],
                    "",
                )
            )

        path.write_text("\n".join(lines), encoding="utf-8")
        return

    payload = {
        "schema": "aibrain.chat.v1",
        "created_at": created_at,
        "mode": mode,
        "model": model,
        "conversation": turns,
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
