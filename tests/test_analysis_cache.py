from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.app.visualizer_panel import PlaybackStep, VisualizerPanel
from src.connectome.analysis import AnalysisRecord
from src.connectome.analysis_cache import AnalysisPageCache
from src.models.instrumented_backend import ActivationFrame, ActivitySource


def _record(step: int) -> AnalysisRecord:
    return AnalysisRecord(
        step=step,
        output_text=f"token-{step}",
        active_channels=step,
        mean_signal=0.2,
        peak_signal=0.8,
        dominant_channel="Raw-logit entropy",
        novelty=0.1 * step,
        reconstruction_error=0.02,
        coherence=0.9,
        embedding=(0.1, 0.2),
        channel_values=(0.3, 0.4),
        telemetry={"raw_logit_entropy_bits": 4.2},
    )


class AnalysisPageCacheTests(unittest.TestCase):
    def test_records_are_compressed_paged_and_removed_after_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".cache" / "temp"
            cache = AnalysisPageCache(
                1024 * 1024, root=root, page_records=2
            )
            cache.append(_record(1))
            cache.append(_record(2))

            self.assertIsNotNone(cache.directory)
            restored = list(cache.iter_records())
            self.assertEqual([item.step for item in restored], [1, 2])
            self.assertEqual(restored[0].output_text, "token-1")
            self.assertEqual(restored[0].telemetry["raw_logit_entropy_bits"], 4.2)
            self.assertGreater(cache.disk_bytes, 0)
            cache_directory = cache.directory
            cache.cleanup()

            self.assertFalse(cache_directory.exists())  # type: ignore[union-attr]
            self.assertEqual(cache.record_count, 0)

    def test_tiny_limit_discards_oldest_page_and_reports_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = AnalysisPageCache(
                1, root=Path(directory) / "temp", page_records=1
            )

            self.assertTrue(cache.append(_record(1)))
            self.assertTrue(cache.overflowed)
            self.assertEqual(cache.record_count, 0)
            self.assertTrue(cache.status(3)["oldest_pages_discarded"])

    def test_zero_limit_disables_cache_without_creating_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "temp"
            cache = AnalysisPageCache(0, root=root, page_records=1)

            self.assertFalse(cache.append(_record(1)))
            self.assertFalse(root.exists())
            self.assertEqual(list(cache.iter_records()), [])

    def test_stale_crash_folder_is_removed_on_next_cache_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "temp"
            stale = root / "analysis-12345-abandoned"
            stale.mkdir(parents=True)
            (stale / "page.jsonl.gz").write_bytes(b"partial")

            with patch.object(AnalysisPageCache, "_pid_is_running", return_value=False):
                AnalysisPageCache(root=root)

            self.assertFalse(stale.exists())

    def test_visualizer_moves_trimmed_analysis_into_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = AnalysisPageCache(
                1024 * 1024, root=Path(directory) / "temp", page_records=8
            )
            signal = PlaybackStep(
                ActivationFrame("token", 1, ActivitySource.REAL_TIME),
                np.ones(4, dtype="f4"),
                np.ones(4, dtype="f4"),
            )
            emitted: list[bool] = []
            panel = SimpleNamespace(
                _analysis_memory_limit_bytes=1,
                _playback=[signal],
                _analysis_retained_bytes=VisualizerPanel._signal_bytes(signal),
                analyzer=SimpleNamespace(records=[_record(1)]),
                _analysis_cache=cache,
                _analysis_memory_exceeded=False,
                analysisMemoryExceeded=SimpleNamespace(emit=emitted.append),
                _signal_bytes=VisualizerPanel._signal_bytes,
            )

            VisualizerPanel._trim_analysis_memory(panel)  # type: ignore[arg-type]

            self.assertEqual(panel._playback, [])
            self.assertEqual(panel.analyzer.records, [])
            self.assertEqual(cache.record_count, 1)
            self.assertEqual(emitted, [])


if __name__ == "__main__":
    unittest.main()
