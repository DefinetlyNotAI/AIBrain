"""Bounded temporary disk pages for inactive Analysis+ records."""

from __future__ import annotations

import atexit
import ctypes
import gzip
import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .analysis import AnalysisRecord

CACHE_ROOT = Path(__file__).resolve().parents[2] / ".cache" / "temp"
DEFAULT_CACHE_BYTES = 1024 * 1024 * 1024


def _decode_record(payload: dict[str, object]) -> AnalysisRecord:
    return AnalysisRecord(
        step=int(payload["step"]),
        token=str(payload["token"]),
        active_nodes=int(payload["active_nodes"]),
        mean_activity=float(payload["mean_activity"]),
        peak_activity=float(payload["peak_activity"]),
        dominant_region=str(payload["dominant_region"]),
        novelty=float(payload["novelty"]),
        reconstruction_error=float(payload["reconstruction_error"]),
        coherence=float(payload["coherence"]),
        embedding=tuple(float(value) for value in payload["embedding"]),  # type: ignore[union-attr]
        regional_activity=tuple(
            float(value) for value in payload["regional_activity"]  # type: ignore[union-attr]
        ),
    )


class AnalysisPageCache:
    """Write chronological compressed pages and enforce a disk byte ceiling."""

    def __init__(
        self,
        max_bytes: int = DEFAULT_CACHE_BYTES,
        *,
        root: Path = CACHE_ROOT,
        page_records: int = 64,
    ) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self.root = Path(root)
        self.page_records = max(1, int(page_records))
        self._directory: Path | None = None
        self._pages: list[tuple[Path, int, int]] = []
        self._pending: list[AnalysisRecord] = []
        self._next_page = 0
        self.overflowed = False
        self._cleanup_stale_sessions()
        atexit.register(self.cleanup)

    @property
    def enabled(self) -> bool:
        return self.max_bytes > 0

    @property
    def record_count(self) -> int:
        return sum(count for _path, count, _size in self._pages) + len(
            self._pending
        )

    @property
    def disk_bytes(self) -> int:
        return sum(size for _path, _count, size in self._pages)

    @property
    def directory(self) -> Path | None:
        return self._directory

    def set_limit(self, max_bytes: int) -> None:
        self.max_bytes = max(0, int(max_bytes))
        if not self.enabled:
            self.cleanup()
            return
        self._enforce_limit()

    def append(self, record: AnalysisRecord) -> bool:
        """Queue one record and return whether the disk ceiling lost old data."""
        if not self.enabled:
            return False
        self._pending.append(record)
        if len(self._pending) >= self.page_records:
            try:
                self._flush()
            except OSError:
                self._pending.clear()
                self.overflowed = True
        return self.overflowed

    def iter_records(self) -> Iterator[AnalysisRecord]:
        """Stream every retained page followed by the small pending page."""
        for path, _count, _size in self._pages:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield _decode_record(json.loads(line))
        yield from self._pending

    def status(self, resident_records: int) -> dict[str, object]:
        return {
            "paged_records": self.record_count,
            "resident_records": resident_records,
            "cache_bytes": self.disk_bytes,
            "cache_limit_bytes": self.max_bytes,
            "cache_enabled": self.enabled,
            "oldest_pages_discarded": self.overflowed,
        }

    def cleanup(self) -> None:
        """Remove this process session's disposable pages."""
        self._pending.clear()
        self._pages.clear()
        directory = self._directory
        self._directory = None
        self._next_page = 0
        self.overflowed = False
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
        try:
            self.root.rmdir()
        except OSError:
            pass

    def _flush(self) -> None:
        if not self._pending or not self.enabled:
            return
        if self._directory is None:
            self._directory = self.root / (
                f"analysis-{os.getpid()}-{uuid4().hex[:10]}"
            )
            self._directory.mkdir(parents=True, exist_ok=True)
        page = self._directory / f"page-{self._next_page:08d}.jsonl.gz"
        temporary = page.with_suffix(page.suffix + ".tmp")
        records = self._pending
        self._pending = []
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
            for record in records:
                handle.write(
                    json.dumps(asdict(record), ensure_ascii=False, allow_nan=False)
                    + "\n"
                )
        temporary.replace(page)
        size = page.stat().st_size
        self._pages.append((page, len(records), size))
        self._next_page += 1
        self._enforce_limit()

    def _enforce_limit(self) -> None:
        if not self.enabled:
            return
        while self._pages and self.disk_bytes > self.max_bytes:
            path, _count, _size = self._pages.pop(0)
            path.unlink(missing_ok=True)
            self.overflowed = True

    def _cleanup_stale_sessions(self) -> None:
        """Remove page folders left by processes that no longer exist."""
        if not self.root.is_dir():
            return
        for directory in self.root.glob("analysis-*"):
            if not directory.is_dir():
                continue
            try:
                pid = int(directory.name.split("-", 2)[1])
            except (IndexError, ValueError):
                continue
            if not self._pid_is_running(pid):
                shutil.rmtree(directory, ignore_errors=True)

    @staticmethod
    def _pid_is_running(pid: int) -> bool:
        if pid == os.getpid():
            return True
        if os.name == "nt":
            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
                process_query_limited_information, False, pid
            )
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
