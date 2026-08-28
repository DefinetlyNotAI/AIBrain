from __future__ import annotations

from pathlib import Path
from hashlib import blake2b
from threading import Event
from time import monotonic

from PySide6.QtCore import QObject, Signal, Slot

from .instrumented_backend import ActivationFrame, ActivitySource
from .llama_backend import GenerationConfig, LlamaBackend

WORLD_SYSTEM = """You are the WORLD DIRECTOR for an open-ended, fictional embodied simulation.
Write only the external world: sensory events, people, places, consequences, and continuity. Never write the
participant's thoughts, dialogue, choices, or first-person actions. You know this is a simulation and should
never claim it is real outside the scenario. Keep each turn vivid, actionable, under 180 words, and continue
the scene instead of ending it. DO NOT USE THE PARTICIPANTS THOUGHTS AS YOUR OWN!!"""

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


def select_world_opening(seed: str) -> str:
    """Choose one stable, pregenerated opening without relying on Python hash randomisation."""
    digest = blake2b(seed.encode("utf-8"), digest_size=2).digest()
    return WORLD_OPENINGS[int.from_bytes(digest, "big") % len(WORLD_OPENINGS)]


def infinite_generation_config(config: GenerationConfig) -> GenerationConfig:
    """Infinite Mode intentionally uses a high-randomness local configuration."""
    return GenerationConfig(
        temperature=max(1.25, config.temperature),
        top_p=max(.96, config.top_p),
        max_tokens=min(config.max_tokens, 512),
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
        self.world = LlamaBackend()
        self.participant = LlamaBackend()
        self._cancelled = Event()

    @Slot(object, object, object)
    def run(self, seed: str, config: GenerationConfig, model_path: Path) -> None:
        self._cancelled.clear()
        config = infinite_generation_config(config)
        started = monotonic()
        participant_tokens = 0
        turn = 0
        opening = select_world_opening(seed)
        world_history = [
            {
                "role": "system",
                "content": WORLD_SYSTEM
            },
            {
                "role": "user",
                "content": (
                    f"[SIMULATION DIRECTION]\n"
                    f"Participant direction: {seed}\n"
                    f"Deterministic world opening: {opening}\n"
                    f"[END SIMULATION DIRECTION]\n"
                    f"Write the opening WORLD EVENT only."
                )
            }
        ]
        participant_history = [
            {"role": "system", "content": PARTICIPANT_SYSTEM},
            {"role": "user", "content": (
                f"[SCENARIO]\n{seed}\n\n{opening}\n[END SCENARIO]\n"
                "Begin the scenario as the PARTICIPANT. Write your first response only."
            )},
        ]
        try:
            # Separate backend objects intentionally deploy two copies of the
            # same selected model: one produces the world, one is the actor.
            self.world.load(model_path, config)
            self.participant.load(model_path, config)
            # The participant begins the scenario; afterward the world reacts
            # and both roles alternate for as long as the user leaves it live.
            participant_text, token_count = self._generate_participant(participant_history, config, 1, 0)
            participant_tokens += token_count
            if not participant_text or self._cancelled.is_set():
                if not self._cancelled.is_set():
                    raise RuntimeError("Participant model returned no text")
            else:
                participant_history.append({"role": "assistant", "content": participant_text})
                world_history.append({"role": "user", "content": (
                    f"[PARTICIPANT RESPONSE]\n{participant_text}\n[END PARTICIPANT RESPONSE]\n"
                    "Write the opening WORLD EVENT only."
                )})
            while not self._cancelled.is_set():
                turn += 1
                world_text = self._generate("world", self.world, world_history, config, turn)
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
                        )
                    }
                )

                participant_text, token_count = self._generate_participant(
                    participant_history,
                    config,
                    turn + 1,
                    participant_tokens
                )

                participant_tokens += token_count
                if self._cancelled.is_set():
                    break
                if not participant_text:
                    raise RuntimeError("Participant model returned no text")
                participant_history.append(
                    {
                        "role": "assistant",
                        "content": participant_text
                    }
                )
                world_history.append(
                    {
                        "role": "user",
                        "content": (
                            f"[PARTICIPANT RESPONSE]\n"
                            f"{participant_text}\n"
                            f"[END PARTICIPANT RESPONSE]\n"
                            f"Write the next WORLD EVENT only."
                        )
                    }
                )
                world_history = self._trim(world_history)
                participant_history = self._trim(participant_history)
            self.finished.emit({"turns": turn, "participant_tokens": participant_tokens,
                                "seconds": monotonic() - started, "cancelled": self._cancelled.is_set()})
        except Exception as exc:
            self.failed.emit(f"Infinite simulation failed: {exc}")
        finally:
            self.world.unload()
            self.participant.unload()

    def _generate(self, role: str, backend: LlamaBackend, messages: list[dict[str, str]], config: GenerationConfig,
                  turn: int) -> str:
        self.turnStarted.emit(role, turn)
        chunks: list[str] = []
        for text in backend.stream_chat(messages, config):
            if self._cancelled.is_set():
                break
            chunks.append(text)
            self.token.emit(role, turn, text, None)
        output = "".join(chunks)
        self.turnFinished.emit(role, output, turn)
        return output

    def _generate_participant(self, messages: list[dict[str, str]], config: GenerationConfig, turn: int,
                              step_offset: int) -> tuple[str, int]:
        self.turnStarted.emit("participant", turn)
        chunks: list[str] = []
        token_count = 0
        for text in self.participant.stream_chat(messages, config):
            if self._cancelled.is_set():
                break
            chunks.append(text)
            token_ids = self.participant.tokenize(text)
            token_count += max(1, len(token_ids))
            frame = ActivationFrame(token_ids[-1] if token_ids else None, text, step_offset + token_count,
                                    ActivitySource.SIMULATION)
            self.token.emit("participant", turn, text, frame)
        output = "".join(chunks)
        self.turnFinished.emit("participant", output, turn)
        return output, token_count

    @staticmethod
    def _trim(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if len(messages) <= _CONTEXT_TURNS + 1:
            return messages
        return [messages[0], *messages[-_CONTEXT_TURNS:]]

    @Slot()
    def cancel(self) -> None:
        self._cancelled.set()

    @Slot()
    def unload(self) -> None:
        self.cancel()
        self.world.unload()
        self.participant.unload()
