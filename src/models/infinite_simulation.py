from __future__ import annotations

from pathlib import Path
from threading import Event
from time import monotonic

from PySide6.QtCore import QObject, Signal, Slot

from .instrumented_backend import ActivationFrame, ActivitySource
from .llama_backend import GenerationConfig, LlamaBackend


WORLD_SYSTEM = """You are the WORLD DIRECTOR for an open-ended, fictional embodied simulation.
Write only the external world: sensory events, people, places, consequences, and continuity. Never write the
participant's thoughts, dialogue, choices, or first-person actions. You know this is a simulation and should
never claim it is real outside the scenario. Keep each turn vivid, actionable, under 180 words, and continue
the scene instead of ending it."""

PARTICIPANT_SYSTEM = """You are the PARTICIPANT in an immersive, fictional embodied simulation.
Write only your first-person thoughts, dialogue, choices, feelings, and physical actions in response to a world
event. Never narrate the world, control other characters, or call yourself the world director. Stay inside the
scenario and do not discuss model prompts, orchestration, or being an AI. Your responses are roleplay text in a
clearly labelled local simulation, not statements about real-world consciousness."""

_CONTEXT_TURNS = 24


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
        started = monotonic()
        participant_tokens = 0
        turn = 0
        world_history = [{"role": "system", "content": WORLD_SYSTEM}, {"role": "user", "content": f"[SIMULATION DIRECTION]\n{seed}\n[END SIMULATION DIRECTION]\nWrite the opening WORLD EVENT only."}]
        participant_history = [{"role": "system", "content": PARTICIPANT_SYSTEM}]
        try:
            # Separate backend objects intentionally deploy two copies of the
            # same selected model: one produces the world, one is the actor.
            self.world.load(model_path, config)
            self.participant.load(model_path, config)
            while not self._cancelled.is_set():
                turn += 1
                world_text = self._generate("world", self.world, world_history, config, turn)
                if self._cancelled.is_set():
                    break
                if not world_text:
                    raise RuntimeError("World model returned no text")
                world_history.append({"role": "assistant", "content": world_text})
                participant_history.append({"role": "user", "content": f"[WORLD EVENT]\n{world_text}\n[END WORLD EVENT]\nWrite the PARTICIPANT response only."})

                participant_text, token_count = self._generate_participant(participant_history, config, turn, participant_tokens)
                participant_tokens += token_count
                if self._cancelled.is_set():
                    break
                if not participant_text:
                    raise RuntimeError("Participant model returned no text")
                participant_history.append({"role": "assistant", "content": participant_text})
                world_history.append({"role": "user", "content": f"[PARTICIPANT RESPONSE]\n{participant_text}\n[END PARTICIPANT RESPONSE]\nWrite the next WORLD EVENT only."})
                world_history = self._trim(world_history)
                participant_history = self._trim(participant_history)
            self.finished.emit({"turns": turn, "participant_tokens": participant_tokens,
                                "seconds": monotonic() - started, "cancelled": self._cancelled.is_set()})
        except Exception as exc:
            self.failed.emit(f"Infinite simulation failed: {exc}")
        finally:
            self.world.unload()
            self.participant.unload()

    def _generate(self, role: str, backend: LlamaBackend, messages: list[dict[str, str]], config: GenerationConfig, turn: int) -> str:
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

    def _generate_participant(self, messages: list[dict[str, str]], config: GenerationConfig, turn: int, step_offset: int) -> tuple[str, int]:
        self.turnStarted.emit("participant", turn)
        chunks: list[str] = []
        token_count = 0
        for text in self.participant.stream_chat(messages, config):
            if self._cancelled.is_set():
                break
            chunks.append(text)
            token_ids = self.participant.tokenize(text)
            token_count += max(1, len(token_ids))
            frame = ActivationFrame(token_ids[-1] if token_ids else None, text, step_offset + token_count, ActivitySource.SIMULATION)
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
