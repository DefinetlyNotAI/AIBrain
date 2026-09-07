"""Structured message types shared by chat, simulation, and exports."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class ChatMessage(TypedDict):
    """One text-only message accepted by the local llama.cpp backend."""

    role: Literal["system", "user", "assistant"]
    content: str


class ConversationTurn(TypedDict):
    """One persisted normal-chat or Infinite Mode transcript entry."""

    role: Literal["user", "assistant", "world", "participant"]
    content: str
    turn: NotRequired[int]


TranscriptTurn = ChatMessage | ConversationTurn
