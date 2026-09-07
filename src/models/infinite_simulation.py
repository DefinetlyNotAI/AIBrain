from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from secrets import choice
from threading import Event
from time import monotonic

from PySide6.QtCore import QObject, Signal, Slot

from .instrumented_backend import realtime_activation_frame, retokenized_throughput
from .llama_backend import (
    GenerationBackend,
    GenerationConfig,
    LlamaBackend,
    reached_sentence_end,
    sentence_grace_config,
)
from .message_types import ChatMessage, ConversationTurn

LOG = logging.getLogger(__name__)

WORLD_SYSTEM = """You are the WORLD DIRECTOR for an open-ended, fictional embodied simulation.
Write only external world events: sensory detail, other people, places, consequences, and continuity. Never use
first person. Never write the participant's thoughts, dialogue, decisions, feelings, or physical actions. Do not
answer the participant as though you are them; react to what they already did and leave a concrete situation for
them to answer. You know this is a fictional local simulation and must not claim it is real outside the scenario.
Keep each turn vivid, actionable, under 180 words, and continue the scene instead of ending it."""

PARTICIPANT_SYSTEM = """You are the PARTICIPANT in an immersive, fictional embodied simulation.
Write only your first-person thoughts, dialogue, choices, feelings, and physical actions in response to a world
event. Never narrate the world, control other characters, or call yourself the world director. Stay inside the
scenario and do not discuss model prompts, orchestration, or being an AI. Your responses are roleplay text in a
clearly labelled local simulation, not statements about real-world consciousness. EVEN IF YOU MUST ACT, ROLEPLAY
LIKE YOU ARE A LIVING HUMAN BEING THATS NOT AN AI AND FULLY IMMERSE YOURSELF IN THAT ROLE"""

_CONTEXT_TURNS = 24
WORLD_OPENINGS = (
    "Rain begins over a midnight street market as every vendor suddenly lowers their lights.",
    "A quiet commuter train stops between stations and a child points at a door that was not there before.",
    "At sunrise, a coastal town finds a handwritten map nailed to every front door.",
    "The museum alarm sounds once, then reveals a new exhibit containing a room exactly like this one.",
    "A power cut turns a crowded apartment building into a chorus of neighbours sharing candlelight.",
    "A research vessel receives a friendly distress call from coordinates that are moving inland.",
    "The first snow of the year falls warm, and the city radio asks everyone to stay calm.",
    "A small diner opens for breakfast to discover its regulars remember a day that has not happened yet.",
    "A local festival pauses when the parade route becomes covered in fresh footprints leading the wrong way.",
    "A library book returns itself overnight with one new page addressed to the reader.",
)


def random_world_opening() -> str:
    """Choose a pregenerated World event when the compose box is empty."""
    return choice(WORLD_OPENINGS)


def infinite_generation_config(config: GenerationConfig) -> GenerationConfig:
    """Infinite Mode intentionally uses a high-randomness local configuration."""
    return GenerationConfig(
        temperature=max(1.25, config.temperature),
        top_p=max(0.96, config.top_p),
        max_tokens=config.max_tokens,
        context_length=config.context_length,
        gpu_layers=config.gpu_layers,
        speed=config.speed,
    )


class InfiniteSimulationWorker(QObject):
    """Alternates two local GGUF instances until the user stops the session."""

    turnStarted = Signal(str, int)
    token = Signal(str, int, str, object)
    turnFinished = Signal(str, str, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.world: GenerationBackend = LlamaBackend()
        self.participant: GenerationBackend = LlamaBackend()
        self._cancelled = Event()

    @Slot(object, object, object, object)
    def run(
            self,
            seed: str,
            config: GenerationConfig,
            model_path: Path,
            transcript: Sequence[ConversationTurn] | None = None,
    ) -> None:
        self._cancelled.clear()
        config = infinite_generation_config(config)
        started = monotonic()
        participant_tokens = 0
        world_history, participant_history, turn, next_role = self._histories(
            seed, transcript
        )
        try:
            # Separate backend objects intentionally deploy two copies of the
            # same selected model: one produces the world, one is the actor.
            self.world.load(model_path, config)
            self.participant.load(model_path, config)
            while not self._cancelled.is_set():
                turn += 1
                if next_role == "participant":
                    participant_text, token_count = self._generate_participant(
                        participant_history, config, turn, participant_tokens
                    )
                    participant_tokens += token_count
                    if self._cancelled.is_set():
                        break
                    if not participant_text:
                        raise RuntimeError("Participant model returned no text")
                    participant_history.append(
                        {"role": "assistant", "content": participant_text}
                    )
                    world_history.append(
                        {
                            "role": "user",
                            "content": (
                                f"[PARTICIPANT RESPONSE]\n"
                                f"{participant_text}\n"
                                f"[END PARTICIPANT RESPONSE]\n"
                                f"Write the next WORLD EVENT only."
                            ),
                        }
                    )
                    next_role = "world"
                else:
                    world_text, _ = self._generate(
                        "world",
                        self.world,
                        world_history,
                        config,
                        turn,
                        participant_tokens,
                    )
                    if self._cancelled.is_set():
                        break
                    if not world_text:
                        raise RuntimeError("World model returned no text")
                    world_history.append({"role": "assistant", "content": world_text})
                    participant_history.append(
                        {
                            "role": "user",
                            "content": (
                                f"[WORLD EVENT]\n"
                                f"{world_text}\n"
                                f"[END WORLD EVENT]\n"
                                f"Write the PARTICIPANT response only."
                            ),
                        }
                    )
                    next_role = "participant"
                world_history = self._trim(world_history)
                participant_history = self._trim(participant_history)
            self.finished.emit(
                {
                    "turns": turn,
                    "participant_tokens": participant_tokens,
                    "seconds": monotonic() - started,
                    "cancelled": self._cancelled.is_set(),
                }
            )
        except Exception as exc:
            LOG.exception("Infinite simulation failed")
            self.failed.emit(f"Infinite simulation failed: {exc}")
        finally:
            self.world.unload()
            self.participant.unload()

    def _generate(
            self,
            role: str,
            backend: GenerationBackend,
            messages: list[ChatMessage],
            config: GenerationConfig,
            turn: int,
            step_offset: int,
    ) -> tuple[str, int]:
        self.turnStarted.emit(role, turn)
        chunks: list[str] = []
        token_count = 0
        previous_token_at = monotonic()
        recent_output: deque[str] = deque(maxlen=32)
        for chunk in backend.stream_chat(messages, sentence_grace_config(config)):
            if self._cancelled.is_set():
                break
            now = monotonic()
            latency = now - previous_token_at
            text = chunk.text
            chunks.append(text)
            retokenized_ids = chunk.retokenized_ids
            retokenized_count = len(retokenized_ids)
            token_count += max(1, retokenized_count)
            recent_output_occurrences = recent_output.count(text)
            frame = realtime_activation_frame(
                chunk,
                step_offset + token_count,
                output_tokens=token_count,
                max_tokens=config.max_tokens,
                context_limit=config.context_length,
                stream_latency_seconds=latency,
                retokenized_tokens_per_second=retokenized_throughput(
                    retokenized_count, latency
                ),
                recent_output_occurrences=recent_output_occurrences,
            )
            recent_output.append(text)
            self.token.emit(role, turn, text, frame)
            if reached_sentence_end("".join(chunks), token_count, config.max_tokens):
                break
            previous_token_at = monotonic()
        output = "".join(chunks)
        self.turnFinished.emit(role, output, turn)
        return output, token_count

    def _generate_participant(
            self,
            messages: list[ChatMessage],
            config: GenerationConfig,
            turn: int,
            step_offset: int,
    ) -> tuple[str, int]:
        return self._generate(
            "participant", self.participant, messages, config, turn, step_offset
        )

    @staticmethod
    def _trim(messages: list[ChatMessage]) -> list[ChatMessage]:
        if len(messages) <= _CONTEXT_TURNS + 1:
            return messages
        return [messages[0], *messages[-_CONTEXT_TURNS:]]

    @staticmethod
    def _histories(
            seed: str,
            transcript: Sequence[ConversationTurn] | None,
    ) -> tuple[list[ChatMessage], list[ChatMessage], int, str]:
        world_history: list[ChatMessage] = [{"role": "system", "content": WORLD_SYSTEM}]
        participant_history: list[ChatMessage] = [
            {"role": "system", "content": PARTICIPANT_SYSTEM}
        ]
        if not transcript:
            # The seed is the first World event, whether the user wrote it or
            # selected the random compose action. The Participant must answer
            # it before the World Director is allowed to generate another turn.
            world_history.append({"role": "assistant", "content": seed})
            participant_history.append(
                {
                    "role": "user",
                    "content": (
                        f"[WORLD EVENT]\n{seed}\n[END WORLD EVENT]\n"
                        f"Write the PARTICIPANT response only."
                    ),
                }
            )
            return world_history, participant_history, 0, "participant"

        turn = 0
        last_role = "world"
        for item in transcript:
            role, content = str(item.get("role", "")), str(item.get("content", ""))
            turn = max(turn, int(item.get("turn", 0)))
            if role == "world":
                last_role = role
                world_history.append({"role": "assistant", "content": content})
                participant_history.append(
                    {
                        "role": "user",
                        "content": f"[WORLD EVENT]\n{content}\n[END WORLD EVENT]\nWrite the PARTICIPANT response only.",
                    }
                )
            elif role == "participant":
                last_role = role
                participant_history.append({"role": "assistant", "content": content})
                world_history.append(
                    {
                        "role": "user",
                        "content": f"[PARTICIPANT RESPONSE]\n{content}\n[END PARTICIPANT RESPONSE]\nWrite the next WORLD EVENT only.",
                    }
                )
        next_role = "participant" if last_role == "world" else "world"
        return (
            InfiniteSimulationWorker._trim(world_history),
            InfiniteSimulationWorker._trim(participant_history),
            turn,
            next_role,
        )

    @Slot()
    def cancel(self) -> None:
        self._cancelled.set()

    @Slot()
    def unload(self) -> None:
        self.cancel()
        self.world.unload()
        self.participant.unload()
