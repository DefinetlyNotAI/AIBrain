from __future__ import annotations

import unittest
from collections.abc import Iterator, Sequence
from pathlib import Path
from unittest.mock import patch

from src.models.infinite_simulation import (
    _CONTEXT_TURNS,
    WORLD_OPENINGS,
    InfiniteSimulationWorker,
    infinite_generation_config,
    random_world_opening,
)
from src.models.instrumented_backend import (
    ActivitySource,
    GenerationChunk,
    LogitMetrics,
)
from src.models.llama_backend import GenerationConfig
from src.models.message_types import ChatMessage, ConversationTurn


class FakeBackend:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.loaded: list[Path] = []
        self.calls: list[list[ChatMessage]] = []
        self.unloaded = False

    def load(self, path: Path, _config: GenerationConfig) -> None:
        self.loaded.append(path)

    def stream_chat(
            self, messages: Sequence[ChatMessage], _config: GenerationConfig
    ) -> Iterator[GenerationChunk]:
        self.calls.append([message.copy() for message in messages])
        for index, text in enumerate(self.chunks, 1):
            yield GenerationChunk(
                text,
                (len(text),),
                LogitMetrics(32_000, 20 + index, 6.0, 0.4, 0.3, 0.6, 0.1),
            )

    def tokenize(self, text: str) -> list[int]:
        return [len(text)]

    def unload(self) -> None:
        self.unloaded = True


class InfiniteSimulationContextTests(unittest.TestCase):
    def test_context_keeps_system_message_and_recent_turns(self) -> None:
        messages: list[ChatMessage] = [{"role": "system", "content": "system"}]
        messages.extend(
            {"role": "user", "content": str(index)}
            for index in range(_CONTEXT_TURNS + 8)
        )

        trimmed = InfiniteSimulationWorker._trim(messages)

        self.assertEqual(trimmed[0], messages[0])
        self.assertEqual(len(trimmed), _CONTEXT_TURNS + 1)
        self.assertEqual(trimmed[1:], messages[-_CONTEXT_TURNS:])

    def test_context_does_not_duplicate_system_message_when_short(self) -> None:
        messages: list[ChatMessage] = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "scene"},
        ]

        self.assertEqual(InfiniteSimulationWorker._trim(messages), messages)

    def test_two_backends_run_and_participant_emits_connectome_frames(self) -> None:
        worker = InfiniteSimulationWorker()
        world = FakeBackend(["A door opens."])
        participant = FakeBackend(["I step through it."])
        worker.world = world
        worker.participant = participant
        frames = []
        finished = []

        def observe(role: str, _turn: int, _text: str, frame: object) -> None:
            if role == "participant":
                frames.append(frame)

        roles = []

        def record_turn(role: str, turn: int) -> None:
            roles.append((role, turn))

        def stop_after_first_exchange(role: str, _text: str, turn: int) -> None:
            if role == "world" and turn == 2:
                worker.cancel()

        worker.token.connect(observe)
        worker.turnStarted.connect(record_turn)
        worker.turnFinished.connect(stop_after_first_exchange)
        worker.finished.connect(finished.append)
        worker.run("A doorway", GenerationConfig(), Path("model.gguf"))

        self.assertEqual(world.loaded, [Path("model.gguf")])
        self.assertEqual(participant.loaded, [Path("model.gguf")])
        self.assertEqual(roles, [("participant", 1), ("world", 2)])
        self.assertEqual(len(participant.calls), 1)
        self.assertIn("A doorway", participant.calls[0][-1]["content"])
        self.assertIn("[WORLD EVENT]", participant.calls[0][-1]["content"])
        self.assertEqual(len(world.calls), 1)
        self.assertIn("[PARTICIPANT RESPONSE]", world.calls[0][-1]["content"])
        self.assertNotIn("[WORLD EVENT]", world.calls[0][-1]["content"])
        self.assertEqual(frames[0].step, 1)
        self.assertEqual(frames[0].source, ActivitySource.REAL_TIME)
        self.assertEqual(frames[0].metrics["context_tokens"], 21)
        self.assertAlmostEqual(frames[0].regions["Raw-logit entropy"], 0.4)
        self.assertTrue(finished[0]["cancelled"])
        self.assertTrue(world.unloaded and participant.unloaded)

    def test_random_opening_uses_the_pregenerated_set(self) -> None:
        with patch(
                "src.models.infinite_simulation.choice", return_value=WORLD_OPENINGS[-1]
        ) as chooser:
            self.assertEqual(random_world_opening(), WORLD_OPENINGS[-1])
        chooser.assert_called_once_with(WORLD_OPENINGS)
        self.assertEqual(len(WORLD_OPENINGS), 10)

    def test_infinite_mode_overrides_low_randomness_settings(self) -> None:
        effective = infinite_generation_config(
            GenerationConfig(temperature=0.1, top_p=0.2, max_tokens=2048)
        )

        self.assertGreaterEqual(effective.temperature, 1.25)
        self.assertGreaterEqual(effective.top_p, 0.96)
        self.assertEqual(effective.max_tokens, 2048)

    def test_continuation_restores_both_roles_and_the_last_turn_number(self) -> None:
        transcript: list[ConversationTurn] = [
            {"role": "world", "content": "A bell rings.", "turn": 1},
            {"role": "participant", "content": "I follow the sound.", "turn": 2},
        ]

        world, participant, turn, next_role = InfiniteSimulationWorker._histories(
            "direction", transcript
        )

        self.assertEqual(turn, 2)
        self.assertEqual(next_role, "world")
        self.assertEqual(world[-1]["role"], "user")
        self.assertIn("I follow the sound.", world[-1]["content"])
        self.assertEqual(participant[-1]["role"], "assistant")
        self.assertEqual(participant[-1]["content"], "I follow the sound.")

    def test_continuation_after_world_event_resumes_with_participant(self) -> None:
        transcript: list[ConversationTurn] = [
            {"role": "world", "content": "A bell rings.", "turn": 7}
        ]

        world, participant, turn, next_role = InfiniteSimulationWorker._histories(
            "unused", transcript
        )

        self.assertEqual(turn, 7)
        self.assertEqual(next_role, "participant")
        self.assertEqual(world[-1], {"role": "assistant", "content": "A bell rings."})
        self.assertIn("A bell rings.", participant[-1]["content"])


if __name__ == "__main__":
    unittest.main()
