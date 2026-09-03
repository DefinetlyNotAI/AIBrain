"""Portable exports for normal and Infinite chat transcripts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def write_chat_export(
    path: Path,
    conversation: list[dict[str, object]],
    *,
    mode: str,
    model: str,
) -> None:
    """Write the complete chat as structured JSON or readable plain text."""
    path = Path(path)
    created_at = datetime.now(UTC).isoformat()
    if path.suffix.lower() == ".txt":
        lines = [
            "AIBrain chat export",
            f"Mode: {mode}",
            f"Model: {model}",
            f"Created: {created_at}",
            "",
        ]
        for turn in conversation:
            role = str(turn.get("role", "unknown")).upper()
            turn_number = turn.get("turn")
            heading = role if turn_number is None else f"{role} {turn_number}"
            lines.extend((heading, str(turn.get("content", "")), ""))
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    payload = {
        "schema": "aibrain.chat.v1",
        "created_at": created_at,
        "mode": mode,
        "model": model,
        "conversation": conversation,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
