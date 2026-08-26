from __future__ import annotations

import unittest
from pathlib import Path

from src.models.infinite_simulation import InfiniteSimulationWorker, _CONTEXT_TURNS
from src.models.llama_backend import GenerationConfig


class FakeBackend:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.loaded: list[Path] = []
        self.calls: list[list[dict[str, str]]] = []
        self.unloaded = False

    def load(self, path: Path, _config: GenerationConfig) -> None:
        self.loaded.append(path)

    def stream_chat(self, messages: list[dict[str, str]], _config: GenerationConfig):  # type: ignore[no-untyped-def]
        self.calls.append([dict(message) for message in messages])
        yield from self.chunks

    def tokenize(self, text: str) -> list[int]:
        return [len(text)]

    def unload(self) -> None:
        self.unloaded = True


class InfiniteSimulationContextTests(unittest.TestCase):
    def test_context_keeps_system_message_and_recent_turns(self) -> None:
        messages = [{"role": "system", "content": "system"}]
        messages.extend({"role": "user", "content": str(index)} for index in range(_CONTEXT_TURNS + 8))

        trimmed = InfiniteSimulationWorker._trim(messages)

        self.assertEqual(trimmed[0], messages[0])
        self.assertEqual(len(trimmed), _CONTEXT_TURNS + 1)
        self.assertEqual(trimmed[1:], messages[-_CONTEXT_TURNS:])

    def test_context_does_not_duplicate_system_message_when_short(self) -> None:
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "scene"}]

        self.assertEqual(InfiniteSimulationWorker._trim(messages), messages)

    def test_two_backends_run_and_participant_emits_connectome_frames(self) -> None:
        worker = InfiniteSimulationWorker()
        world = FakeBackend(["A door opens."])
        participant = FakeBackend(["I step through it."])
        worker.world = world  # type: ignore[assignment]
        worker.participant = participant  # type: ignore[assignment]
        frames = []
        finished = []

        def observe(role: str, _turn: int, _text: str, frame: object) -> None:
            if role == "participant":
                frames.append(frame)

        def stop_after_first_exchange(role: str, turn: int) -> None:
            if role == "world" and turn == 2:
                worker.cancel()

        worker.token.connect(observe)
        worker.turnStarted.connect(stop_after_first_exchange)
        worker.finished.connect(finished.append)
        worker.run("A doorway", GenerationConfig(), Path("model.gguf"))

        self.assertEqual(world.loaded, [Path("model.gguf")])
        self.assertEqual(participant.loaded, [Path("model.gguf")])
        self.assertIn("[WORLD EVENT]", participant.calls[0][-1]["content"])
        self.assertIn("[PARTICIPANT RESPONSE]", world.calls[1][-1]["content"])
        self.assertNotIn("[WORLD EVENT]", world.calls[1][-1]["content"])
        self.assertEqual(frames[0].step, 1)
        self.assertTrue(finished[0]["cancelled"])
        self.assertTrue(world.unloaded and participant.unloaded)


if __name__ == "__main__":
    unittest.main()
