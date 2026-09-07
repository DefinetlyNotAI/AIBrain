"""Load llama.cpp with the CUDA DLLs installed in this Python environment."""

from __future__ import annotations

import logging
import os
import sys
import sysconfig
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from types import ModuleType
from typing import Protocol, cast

from src.app.ctypes_helper import CVoidP


class _NativeLogCallback(Protocol):
    """Callable signature retained by llama.cpp for native log delivery."""

    def __call__(
            self,
            level: int,
            text: bytes | None,
            _user_data: CVoidP,
    ) -> None: ...


_DLL_HANDLES: dict[Path, object] = {}
_DLL_LOCK = Lock()

_LOG_CALLBACK: _NativeLogCallback | None = None
_LOG_LOCK = Lock()

_LOG = logging.getLogger(__name__)
_LAST_NATIVE_SEVERITY = logging.DEBUG


def _handle_native_log(
        level: int,
        text: bytes | None,
        _user_data: object,
) -> None:
    """Send native runtime messages through the same formatter as Python logs."""
    global _LAST_NATIVE_SEVERITY

    if not text:
        return

    try:
        message = text.decode(
            "utf-8",
            errors="replace",
        ).strip()

        if not message or not message.strip("."):
            return

        if level == 5:
            severity = _LAST_NATIVE_SEVERITY
        else:
            severity = {
                1: logging.DEBUG,
                2: logging.INFO,
                3: logging.WARNING,
                4: logging.ERROR,
            }.get(
                level,
                logging.DEBUG,
            )

            _LAST_NATIVE_SEVERITY = severity

        if severity == logging.INFO:
            severity = logging.DEBUG

        if (
                "ggml_cuda_init:" in message
                or message.startswith("Device ")
        ):
            severity = logging.INFO

        _LOG.log(
            severity,
            "llama.cpp: %s",
            message,
        )

    except (
            OSError,
            TypeError,
            UnicodeError,
            ValueError,
    ):
        return


def cuda_dll_directories() -> list[Path]:
    """Find NVIDIA's CUDA wheel DLL directories without importing packages."""
    package_roots = {
        Path(
            sysconfig.get_path("purelib")
        ),
        Path(
            sysconfig.get_path("platlib")
        ),
        Path(
            sys.executable
        ).parent,
    }

    directories: set[Path] = set()

    for root in package_roots:
        nvidia = (
                root
                / "nvidia"
        )

        for pattern in (
                "*/bin",
                "*/bin/x86_64",
                "*/bin/x64",
        ):
            for directory in nvidia.glob(pattern):
                if (
                        directory.is_dir()
                        and any(
                    directory.glob("*.dll")
                )
                ):
                    directories.add(
                        directory.resolve()
                    )

    return sorted(
        directories
    )


def prepare_cuda_dll_search() -> list[Path]:
    """Keep NVIDIA DLL directories available for modern and legacy loaders."""
    if sys.platform != "win32":
        return []

    directories = cuda_dll_directories()

    with _DLL_LOCK:
        for directory in directories:
            if directory not in _DLL_HANDLES:
                _DLL_HANDLES[directory] = (
                    os.add_dll_directory(
                        str(directory)
                    )
                )

        existing = (
            os.environ
            .get(
                "PATH",
                "",
            )
            .split(
                os.pathsep
            )
        )

        known = {
            os.path.normcase(
                os.path.normpath(path)
            )
            for path in existing
            if path
        }

        additions = [
            str(directory)
            for directory in directories
            if os.path.normcase(
                os.path.normpath(
                    str(directory)
                )
            )
               not in known
        ]

        if additions:
            os.environ["PATH"] = (
                os.pathsep.join(
                    [
                        *additions,
                        *existing,
                    ]
                )
            )

    return directories


def load_llama_cpp() -> ModuleType:
    """Use the same native loader in the installer and application."""
    prepare_cuda_dll_search()

    import llama_cpp

    global _LOG_CALLBACK

    with _LOG_LOCK:
        callback = _LOG_CALLBACK

        if callback is None:
            raw_callback_factory = getattr(
                llama_cpp,
                "llama_log_callback",
                None,
            )

            if callable(raw_callback_factory):
                callback_factory = cast(
                    Callable[
                        [_NativeLogCallback],
                        _NativeLogCallback,
                    ],
                    raw_callback_factory,
                )

                callback = callback_factory(
                    _handle_native_log
                )

            else:
                callback = _handle_native_log

            _LOG_CALLBACK = callback

        raw_log_set = getattr(
            llama_cpp,
            "llama_log_set",
            None,
        )

        if callable(raw_log_set):
            log_set = cast(
                Callable[
                    [
                        _NativeLogCallback,
                        object | None,
                    ],
                    None,
                ],
                raw_log_set,
            )

            log_set(
                callback,
                None,
            )

    return llama_cpp
