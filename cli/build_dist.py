"""Compile timestamped standalone AIBrain desktop applications with Nuitka."""
from __future__ import annotations

import argparse
import codecs
import ctypes
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.kernel32 import kernel32
from src.utils.console_ui import (
    COMMAND_INDENT,
    Color,
    CommandOutputBox,
    clear_screen,
    command_preview,
    error,
    header,
    info,
    panel,
    report_keyboard_interrupt,
    section,
    terminal_width,
)
from src.utils.gpu import set_windows_executable_gpu_preference
from src.utils.runtime import require_managed_runtime
from src.utils.logging import configure_cli_logging, log_completed_command, report_exception

VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
DIST_ROOT = ROOT / "dist"
SITE_PACKAGES = VENV_PYTHON.parents[1] / "Lib" / "site-packages"
NUMPY_PACKAGE_ROOT = SITE_PACKAGES / "numpy"
NUMPY_DLL_ROOT = SITE_PACKAGES / "numpy.libs"

RUNTIME_DLLS = ("vcomp140.dll",)
NATIVE_LIBRARY = ROOT / "dll" / "aibrain.connectome.dll"
BUILD_HEARTBEAT_SECONDS = 1.0
BUILD_LOG_HEARTBEAT_SECONDS = 15.0
BUILD_STALL_SECONDS = 5 * 60.0
BUILD_STALL_RECHECK_SECONDS = 5 * 60.0

# Keep Nuitka's module graph aligned with the application rather than with the
# optional feature sets advertised by dependency package hooks. ModernGL is
# supplied with Qt's current OpenGL context. NumPy is staged as CPython source
# and prebuilt extension modules so Nuitka never compiles its large package.
EXCLUDED_PACKAGING_IMPORTS = (
    "glcontext",
    "numpy",
)

# These are the only NumPy implementation branches AIBrain requires: array
# core, type metadata, random generation, linear algebra, and NPZ archive I/O.
# Deliberately omit compatibility, FFT, masked arrays, polynomial, test, f2py,
# documentation, and build-tool namespaces.
NUMPY_RUNTIME_SUBDIRECTORIES = (
    "_core",
    "_typing",
    "_utils",
    "lib",
    "linalg",
    "matrixlib",
    "random",
)

# NumPy's staged Python modules dynamically load this observed CPython support
# closure. Nuitka cannot discover it after NumPy is deliberately no-follow, so
# include these exact helpers explicitly. This is not a NumPy package include.
NUMPY_RUNTIME_SUPPORT_MODULES = (
    "_collections_abc", "_compat_pickle", "_compression", "_ctypes", "_hashlib", "_lzma", "_weakrefset",
    "abc", "ast", "base64", "bisect", "bz2", "codecs", "collections", "collections.abc", "contextlib",
    "contextvars", "copyreg", "ctypes", "ctypes._endian", "datetime", "dis", "encodings",
    "encodings.aliases", "encodings.cp1252", "encodings.cp437", "encodings.utf_8", "enum", "fnmatch",
    "functools", "genericpath", "hashlib", "hmac", "importlib", "importlib._abc", "importlib.machinery",
    "importlib.util", "inspect", "io", "ipaddress", "keyword", "linecache", "lzma", "ntpath", "numbers",
    "opcode", "operator", "os", "pathlib", "pickle", "platform", "posixpath", "random", "re",
    "re._casefix", "re._compiler", "re._constants", "re._parser", "reprlib", "secrets", "shutil", "stat",
    "struct", "textwrap", "threading", "token", "tokenize", "types", "typing", "urllib", "urllib.parse",
    "warnings", "weakref", "zipfile",
)


# ConPTY output can contain terminal control sequences. CommandOutputBox handles
# the actual presentation, so terminal cursor manipulation must not leak into it.
ANSI_ESCAPE_RE = re.compile(
    r"""
    \x1B
    (?:
        [@-Z\\-_]
        |
        \[
        [0-?]*
        [ -/]*
        [@-~]
    )
    """,
    re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class ApplicationTarget:
    directory: str
    executable: str
    entry_point: Path
    icon: Path
    console_mode: str


@dataclass(frozen=True, slots=True)
class BuildStallState:
    """Track the next non-blocking extended-silence warning."""

    next_check: float


APPLICATIONS = (
    ApplicationTarget(
        "ai_brain",
        "ai_brain.exe",
        ROOT / "cli" / "main.py",
        ROOT / "ico" / "brain.ico",
        "attach",
    ),
    ApplicationTarget(
        "diagnostic",
        "diagnostic.exe",
        ROOT / "cli" / "diagnostic.py",
        ROOT / "ico" / "diagnostic.ico",
        "attach",
    ),
    ApplicationTarget(
        "analysis",
        "analysis.exe",
        ROOT / "cli" / "analysis.py",
        ROOT / "ico" / "analysis.ico",
        "attach",
    ),
)


class _ConsoleCoord(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_short),
        ("y", ctypes.c_short),
    ]


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", wintypes.LPVOID),
        ("bInheritHandle", wintypes.BOOL),
    ]


class _StartupInfoW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(ctypes.c_ubyte)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _StartupInfoExW(ctypes.Structure):
    _fields_ = [
        ("StartupInfo", _StartupInfoW),
        ("lpAttributeList", wintypes.LPVOID),
    ]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class WindowsConPty:
    """Windows child process attached to a ConPTY terminal."""

    def __init__(
        self,
        process_handle: wintypes.HANDLE,
        process_id: int,
        pseudo_console: wintypes.HANDLE,
        input_handle: wintypes.HANDLE,
        output_handle: wintypes.HANDLE,
    ) -> None:
        self.process_handle = process_handle
        self.pid = process_id
        self.pseudo_console = pseudo_console
        self.input_handle = input_handle
        self.output_handle = output_handle
        self._terminal_closed = False
        self._closed = False

    def poll(self) -> int | None:
        exit_code = kernel32.get_exit_code_process(
            self.process_handle
        )

        if exit_code == kernel32.STILL_ACTIVE:
            return None

        return exit_code

    def wait(self) -> int:
        result = kernel32.wait_for_single_object(
            self.process_handle,
            kernel32.INFINITE,
        )

        if result != kernel32.WAIT_OBJECT_0:
            raise OSError(
                f"WaitForSingleObject returned {result}"
            )

        return kernel32.get_exit_code_process(
            self.process_handle
        )

    def close_terminal(self) -> None:
        """Close ConPTY so its output reader receives EOF."""
        if self._terminal_closed:
            return

        self._terminal_closed = True

        kernel32.close_handle(
            self.input_handle
        )
        self.input_handle = wintypes.HANDLE()

        kernel32.close_pseudo_console(
            self.pseudo_console
        )
        self.pseudo_console = wintypes.HANDLE()

    def close(self) -> None:
        """Close handles after the ConPTY reader has stopped."""
        if self._closed:
            return

        self._closed = True

        self.close_terminal()

        kernel32.close_handle(
            self.output_handle
        )
        self.output_handle = wintypes.HANDLE()

        kernel32.close_handle(
            self.process_handle
        )
        self.process_handle = wintypes.HANDLE()


def _clean_terminal_output(text: str) -> str:
    """Remove terminal control sequences while preserving visible output."""
    OSC_ESCAPE_RE = re.compile(
        r"\x1b][^\x07\x1b]*(?:\x07|\x1b\\)"
    )
    text = OSC_ESCAPE_RE.sub("", text)
    text = ANSI_ESCAPE_RE.sub("", text)

    while "\b" in text:
        text = re.sub(r".\b", "", text)

    return text


def _record_build_output(line: str) -> None:
    """Persist child output as it arrives without replaying it in the CLI."""
    if line:
        logging.getLogger("aibrain.command").info("%s", line)


def _is_option_echo(line: str) -> bool:
    """Keep Nuitka's command replay in the log without hiding its warnings."""
    prefix, separator, message = line.lstrip().partition("Nuitka-Options:")
    return bool(separator) and not prefix and not message.lstrip().startswith(("WARNING:", "ERROR:", "FATAL:"))


def _is_build_noise(line: str) -> bool:
    """Filter repetitive compiler internals while preserving all diagnostics."""
    stripped = line.lstrip()
    if re.match(r"Nuitka[^:]*:\s*(?:WARNING|ERROR|FATAL):", stripped):
        return False
    return _is_option_echo(line) or stripped.startswith((
        "Nuitka-Memory: Total memory usage",
        "Nuitka-Inclusion: Demoting module ",
        "Nuitka-Progress: Doing module local optimizations ",
        "Nuitka-Progress: Doing module dependency considerations ",
        "Nuitka-Progress: Not finished with the module ",
        "Nuitka-Progress: Not changed, but retrying ",
        "Nuitka-Progress: Finished with the module.",
    ))


def _write_completed_build_line(output_box: CommandOutputBox, line: str) -> None:
    """Keep the full log and show meaningful build work at a readable rate."""
    _record_build_output(line)
    if _is_build_noise(line):
        return
    output_box.write(line)


def _render_output_text(
        text: str,
        pending: str,
        output_box: CommandOutputBox,
) -> str:
    """Render complete lines and immediately redraw carriage-return frames."""
    new_output = pending + _clean_terminal_output(text)
    pending = ""
    live_frame: str | None = None

    while new_output:
        newline_index = new_output.find("\n")
        carriage_index = new_output.find("\r")

        if newline_index < 0:
            newline = carriage_index
        elif carriage_index < 0:
            newline = newline_index
        else:
            newline = min(newline_index, carriage_index)

        if newline < 0:
            pending = new_output
            break

        line = new_output[:newline]
        terminator = new_output[newline]
        new_output = new_output[newline + 1:]

        if terminator == "\r":
            # CRLF is a normal completed line.
            if new_output.startswith("\n"):
                _write_completed_build_line(output_box, line)
                new_output = new_output[1:]
                continue

            # A lone carriage return is a live terminal frame. Nuitka and SCons
            # use this while updating compilation progress. Retain only the
            # newest frame so a verbose child cannot bury the renderer in old
            # cursor updates.
            live_frame = line
            continue

        _write_completed_build_line(output_box, line)
        live_frame = None

    visible_partial = pending or live_frame
    if visible_partial and not _is_build_noise(visible_partial):
        if re.search(r"\d+(?:\.\d+)?%.*?\b\d+/\d+", visible_partial):
            # Keep Nuitka's own bar, percentage, counts, and current item. A
            # captured console gets periodic snapshots instead of losing it.
            output_box.write_progress(visible_partial)
        else:
            output_box.write_partial(visible_partial)

    return pending


def _render_new_build_output(
        output_path: Path,
        offset: int,
        pending: str,
        output_box: CommandOutputBox,
) -> tuple[int, str]:
    """Fallback file tailer used when ConPTY is unavailable."""
    with output_path.open(
            "r",
            encoding="utf-8",
            errors="replace",
            newline="",
    ) as output_file:
        output_file.seek(offset)
        text = output_file.read()
        offset = output_file.tell()

    if not text:
        return offset, pending

    pending = _render_output_text(
        text,
        pending,
        output_box,
    )

    return offset, pending


def _command_activity(command_line: list[str]) -> str:
    """Name the running program, module, or script without exposing its arguments."""
    executable = Path(command_line[0]).name
    name = executable.lower().removesuffix(".exe")
    if name == "nuitka":
        return "Nuitka"
    if not re.fullmatch(r"py|pythonw?(?:\d+(?:\.\d+)*)?", name):
        return executable

    arguments = iter(command_line[1:])
    for argument in arguments:
        if argument == "-m":
            module = next(arguments, "Python")
            if module == "nuitka":
                return "Nuitka"
            if module == "pip":
                operation = next(arguments, "")
                if operation in {"install", "download", "wheel", "check", "uninstall"}:
                    return f"pip {operation}"
            return module
        if argument == "-c":
            return "Python command"
        if argument in {"-W", "-X", "--check-hash-based-pycs"}:
            next(arguments, None)
        elif not argument.startswith("-"):
            return Path(argument).name
    return "Python"


def _report_build_heartbeat(
        output_box: CommandOutputBox,
        last_heartbeat: float,
        last_child_output: float,
        *,
        activity: str,
        started_at: float | None = None,
        process_id: int | None = None,
) -> float:
    """Identify the active command during silence without implying progress."""
    now = time.monotonic()
    interval = BUILD_HEARTBEAT_SECONDS if output_box.is_live else BUILD_LOG_HEARTBEAT_SECONDS
    if now - last_heartbeat < interval:
        return last_heartbeat

    silent_seconds = now - last_child_output
    explanation = (
        " Source generation, compilation, and linking can be silent."
        if activity == "Nuitka" else ""
    )
    details = ""
    if started_at is not None:
        details += f" Elapsed: {now - started_at:.0f}s."
    if process_id is not None:
        details += f" PID: {process_id}."
    message = (
        f"Still working: {activity} is running without new output "
        f"for {silent_seconds:.0f}s.{details}{explanation}"
    )
    if output_box.is_live:
        output_box.write_partial(message)
    else:
        output_box.write(message)
    return now


def _monitor_build_stall(
        last_child_output: float,
        state: BuildStallState,
        *,
        activity: str,
) -> tuple[str | None, BuildStallState]:
    """Report extended silence without blocking the output pump or Qt event loop."""
    now = time.monotonic()
    if now - last_child_output < BUILD_STALL_SECONDS or now < state.next_check:
        return None, state

    explanation = (
        "Native linking can remain silent for a long time. "
        if activity == "Nuitka" else ""
    )
    message = (
        f"No new output from {activity} for {now - last_child_output:.0f}s. "
        "The process is still alive. "
        f"{explanation}"
        "Monitoring will continue. Press Ctrl+C once to stop the process tree."
    )
    return message, BuildStallState(now + BUILD_STALL_RECHECK_SECONDS)


def _stop_process_tree(process_id: int) -> None:
    """Force-stop a process and every child process it created."""
    if os.name == "nt":
        subprocess.run(
            [
                "taskkill",
                "/PID",
                str(process_id),
                "/T",
                "/F",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _stop_interrupted_build(
        process: subprocess.Popen,
) -> None:
    """Stop Nuitka and its compiler children before temporary cleanup."""
    try:
        if process.poll() is not None:
            return
    except KeyboardInterrupt:
        pass

    if os.name == "nt" and getattr(process, "pid", None):
        _stop_process_tree(process.pid)
    else:
        process.terminate()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _supports_conpty() -> bool:
    """Keep builds on the incremental file tailer.

    Windows ConPTY can leave ClosePseudoConsole waiting on its headless
    conhost after Nuitka has already exited. The file tailer is unbuffered,
    incremental, and does not have that lifecycle deadlock.
    """
    return False


def _create_pipe() -> tuple[wintypes.HANDLE, wintypes.HANDLE]:
    """Create a non-inherited anonymous Windows pipe."""
    security = _SecurityAttributes()
    security.nLength = ctypes.sizeof(_SecurityAttributes)
    security.lpSecurityDescriptor = None
    security.bInheritHandle = False

    return kernel32.create_pipe(security)


def _create_conpty_process(
        command_line: list[str],
) -> WindowsConPty:
    """Launch a command inside a Windows ConPTY pseudo console."""
    input_read = wintypes.HANDLE()
    input_write = wintypes.HANDLE()
    output_read = wintypes.HANDLE()
    output_write = wintypes.HANDLE()
    pseudo_console = wintypes.HANDLE()

    process_information = _ProcessInformation()

    attribute_buffer: ctypes.Array[ctypes.c_char] | None = None
    attribute_list: wintypes.LPVOID | None = None

    try:
        input_read, input_write = _create_pipe()
        output_read, output_write = _create_pipe()

        coord = _ConsoleCoord(
            240,
            max(20, min(shutil.get_terminal_size((120, 40)).lines, 120)),
        )

        pseudo_console = kernel32.create_pseudo_console(
            coord,
            input_read,
            output_write,
        )

        # ConPTY owns these ends once the pseudo console is created.
        kernel32.close_handle(input_read)
        input_read = wintypes.HANDLE()

        kernel32.close_handle(output_write)
        output_write = wintypes.HANDLE()

        attribute_buffer, attribute_list = (
            kernel32.initialize_attribute_list()
        )

        kernel32.update_pseudo_console_attribute(
            attribute_list,
            pseudo_console,
        )

        startup_info = _StartupInfoExW()
        startup_info.StartupInfo.cb = ctypes.sizeof(
            _StartupInfoExW
        )
        startup_info.lpAttributeList = attribute_list

        command_text = subprocess.list2cmdline(
            command_line
        )

        mutable_command = ctypes.create_unicode_buffer(
            command_text
        )

        creation_flags = (
                kernel32.EXTENDED_STARTUPINFO_PRESENT
                | kernel32.CREATE_UNICODE_ENVIRONMENT
                | kernel32.CREATE_NEW_PROCESS_GROUP
        )

        kernel32.create_process(
            mutable_command,
            ROOT,
            startup_info,
            process_information,
            creation_flags,
        )

        kernel32.close_handle(
            process_information.hThread
        )
        process_information.hThread = wintypes.HANDLE()

        return WindowsConPty(
            process_handle=process_information.hProcess,
            process_id=int(
                process_information.dwProcessId
            ),
            pseudo_console=pseudo_console,
            input_handle=input_write,
            output_handle=output_read,
        )

    except BaseException:
        if process_information.hThread:
            kernel32.close_handle(
                process_information.hThread
            )

        if process_information.hProcess:
            kernel32.close_handle(
                process_information.hProcess
            )

        if input_read:
            kernel32.close_handle(input_read)

        if input_write:
            kernel32.close_handle(input_write)

        if output_read:
            kernel32.close_handle(output_read)

        if output_write:
            kernel32.close_handle(output_write)

        if pseudo_console:
            kernel32.close_pseudo_console(
                pseudo_console
            )

        raise

    finally:
        if attribute_list is not None:
            kernel32.delete_attribute_list(
                attribute_list
            )

        # Keeps the backing memory alive for the lifetime of the
        # attribute list above.
        _ = attribute_buffer


def _conpty_reader(
        output_handle: wintypes.HANDLE,
        output_queue: queue.Queue[bytes | None],
) -> None:
    try:
        while True:
            chunk = kernel32.read_file(
                output_handle
            )

            if chunk is None:
                break

            if chunk:
                output_queue.put(chunk)

    finally:
        output_queue.put(None)


def _run_with_conpty(
    command_line: list[str],
) -> None:
    """Run Nuitka through ConPTY so terminal progress remains live."""
    activity = _command_activity(command_line)
    process = _create_conpty_process(command_line)

    chunks: queue.Queue[bytes | None] = queue.Queue()

    reader = threading.Thread(
        target=_conpty_reader,
        args=(
            process.output_handle,
            chunks,
        ),
        name="nuitka-conpty-reader",
        daemon=True,
    )
    reader.start()

    decoder = codecs.getincrementaldecoder("utf-8")(
        errors="replace"
    )

    pending = ""
    captured_tail = ""
    reader_finished = False
    last_child_output = time.monotonic()
    started_at = last_child_output
    last_heartbeat = last_child_output
    stall_state = BuildStallState(last_child_output + BUILD_STALL_SECONDS)

    try:
        with CommandOutputBox() as output_box:
            while True:
                try:
                    chunk = chunks.get(timeout=0.05)
                except queue.Empty:
                    chunk = b""

                if chunk not in (b"", None):
                    combined = bytearray(chunk)
                    for _ in range(511):
                        try:
                            queued_chunk = chunks.get_nowait()
                        except queue.Empty:
                            break
                        if queued_chunk is None:
                            reader_finished = True
                            break
                        combined.extend(queued_chunk)
                    chunk = bytes(combined)

                if chunk is None:
                    reader_finished = True

                elif chunk:
                    last_child_output = time.monotonic()
                    last_heartbeat = last_child_output
                    stall_state = BuildStallState(
                        last_child_output + BUILD_STALL_SECONDS
                    )
                    text = decoder.decode(
                        chunk,
                        final=False,
                    )

                    if text:
                        captured_tail = (captured_tail + text)[-65536:]
                        pending = _render_output_text(
                            text,
                            pending,
                            output_box,
                        )

                return_code = process.poll()

                if return_code is None:
                    last_heartbeat = _report_build_heartbeat(
                        output_box,
                        last_heartbeat,
                        last_child_output,
                        activity=activity,
                        started_at=started_at,
                        process_id=process.pid,
                    )
                    stall_message, stall_state = _monitor_build_stall(
                        last_child_output,
                        stall_state,
                        activity=activity,
                    )
                    if stall_message:
                        output_box.write(stall_message)

                if return_code is not None:
                    process.close_terminal()

                if return_code is not None and reader_finished:
                    break

            remaining = decoder.decode(
                b"",
                final=True,
            )

            if remaining:
                captured_tail = (captured_tail + remaining)[-65536:]
                pending = _render_output_text(
                    remaining,
                    pending,
                    output_box,
                )

            if pending:
                _write_completed_build_line(output_box, pending)

        reader.join()
        return_code = process.wait()

    except KeyboardInterrupt:
        _stop_process_tree(process.pid)

        process.close_terminal()
        reader.join()

        process.wait()
        log_completed_command(
            command_line,
            "",
            return_code=None,
            interrupted=True,
        )
        raise

    finally:
        process.close_terminal()

        if reader.is_alive():
            reader.join()

        process.close()

    captured_output = captured_tail.rstrip()
    log_completed_command(
        command_line, "", return_code=return_code
    )
    if return_code:
        raise subprocess.CalledProcessError(
            return_code,
            command_line,
            captured_output,
        )


def _build_process_environment(command_line: list[str]) -> dict[str, str] | None:
    """Let Nuitka's native Rich bars render through our incremental capture."""
    if _command_activity(command_line) != "Nuitka":
        return None
    environment = os.environ.copy()
    environment.update({
        "TTY_COMPATIBLE": "1",
        "TTY_INTERACTIVE": "1",
        "TERM": "xterm-256color",
        "COLUMNS": str(max(terminal_width() - COMMAND_INDENT - 4, 16)),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    })
    return environment


def _run_with_file_tailer(
        command_line: list[str],
) -> None:
    """Fallback runner for platforms without Windows ConPTY."""
    activity = _command_activity(command_line)
    with tempfile.TemporaryDirectory(
            prefix="aibrain-nuitka-"
    ) as temporary_directory:
        output_path = (
                Path(temporary_directory)
                / "nuitka-output.txt"
        )

        with output_path.open(
                "w",
                encoding="utf-8",
                errors="replace",
                newline="",
        ) as output_file:
            process = subprocess.Popen(
                command_line,
                cwd=ROOT,
                env=_build_process_environment(command_line),
                stdout=output_file,
                stderr=subprocess.STDOUT,
                creationflags=getattr(
                    subprocess,
                    "CREATE_NEW_PROCESS_GROUP",
                    0,
                ),
            )

            try:
                offset = 0
                pending = ""
                last_child_output = time.monotonic()
                started_at = last_child_output
                last_heartbeat = last_child_output
                stall_state = BuildStallState(last_child_output + BUILD_STALL_SECONDS)

                with CommandOutputBox() as output_box:
                    while process.poll() is None:
                        output_file.flush()

                        previous_offset = offset
                        offset, pending = _render_new_build_output(
                            output_path,
                            offset,
                            pending,
                            output_box,
                        )

                        if offset != previous_offset:
                            last_child_output = time.monotonic()
                            last_heartbeat = last_child_output
                            stall_state = BuildStallState(
                                last_child_output + BUILD_STALL_SECONDS
                            )
                        else:
                            last_heartbeat = _report_build_heartbeat(
                                output_box,
                                last_heartbeat,
                                last_child_output,
                                activity=activity,
                                started_at=started_at,
                                process_id=process.pid,
                            )

                        stall_message, stall_state = _monitor_build_stall(
                            last_child_output,
                            stall_state,
                            activity=activity,
                        )
                        if stall_message:
                            output_box.write(stall_message)

                        time.sleep(0.05)

                    output_file.flush()

                    offset, pending = _render_new_build_output(
                        output_path,
                        offset,
                        pending,
                        output_box,
                    )

                    if pending:
                        _write_completed_build_line(output_box, pending)

                return_code = process.wait()

            except KeyboardInterrupt:
                _stop_interrupted_build(process)
                output_file.flush()
                log_completed_command(
                    command_line,
                    "",
                    return_code=None,
                    interrupted=True,
                )
                raise

        # Verbose builds can produce very large transcripts. The complete
        # stream has already been logged; retain only a bounded exception tail.
        with output_path.open("rb") as captured_file:
            captured_file.seek(max(0, output_path.stat().st_size - 65536))
            captured_output = captured_file.read().decode("utf-8", errors="replace").rstrip()

    log_completed_command(
        command_line, "", return_code=return_code
    )
    if return_code:
        raise subprocess.CalledProcessError(
            return_code,
            command_line,
            captured_output,
        )


def run(command_line: list[str]) -> None:
    """Run a build command with live boxed output."""
    command_preview(command_line)

    if _supports_conpty():
        _run_with_conpty(command_line)
        return

    _run_with_file_tailer(command_line)


def runtime_dlls() -> list[Path]:
    system32 = (
            Path(
                os.environ.get(
                    "SystemRoot",
                    r"C:\Windows",
                )
            )
            / "System32"
    )

    resolved = [
        system32 / name
        for name in RUNTIME_DLLS
    ]

    missing = [
        path.name
        for path in resolved
        if not path.is_file()
    ]

    if missing:
        raise RuntimeError(
            "Required Visual C++ runtime DLLs are unavailable: "
            + ", ".join(missing)
        )

    return resolved


def stage_numpy_runtime(build_root: Path) -> tuple[Path, Path]:
    """Stage only AIBrain's prebuilt NumPy runtime beside Nuitka's output.

    Nuitka follows NumPy's package configuration into optional scientific
    namespaces and then compiles their Python wrappers. Keeping these selected
    CPython files raw preserves NumPy's extension-module loader while avoiding
    that expensive and unnecessary compilation work.
    """
    if not NUMPY_PACKAGE_ROOT.is_dir():
        raise RuntimeError(
            "Managed NumPy package is missing. Run: py cli\\installer.py"
        )
    if not NUMPY_DLL_ROOT.is_dir():
        raise RuntimeError(
            "Managed NumPy native DLL directory is missing. Run: py cli\\installer.py"
        )

    runtime_root = build_root / "_numpy_runtime"
    package_target = runtime_root / "numpy"
    dll_target = runtime_root / "numpy.libs"
    package_target.mkdir(parents=True, exist_ok=True)

    info("Staging the required NumPy runtime files")
    for source in NUMPY_PACKAGE_ROOT.glob("*.py"):
        if source.name == "conftest.py":
            continue
        shutil.copy2(source, package_target / source.name)

    for name in NUMPY_RUNTIME_SUBDIRECTORIES:
        source = NUMPY_PACKAGE_ROOT / name
        if not source.is_dir():
            raise RuntimeError(f"Managed NumPy runtime submodule is missing: {name}")
        shutil.copytree(source, package_target / name, dirs_exist_ok=True)

    shutil.copytree(NUMPY_DLL_ROOT, dll_target, dirs_exist_ok=True)
    info("NumPy runtime staging complete")
    return package_target, dll_target


def nuitka_command(
        target: ApplicationTarget,
        build_root: Path,
        runtimes: list[Path],
        numpy_runtime: tuple[Path, Path] | None = None,
) -> list[str]:
    """Build a reproducible standalone Nuitka command."""
    command_line = [
        str(VENV_PYTHON),
        "-u",
        "-m",
        "nuitka",
        "--progress-bar=rich",
        "--standalone",
        "--assume-yes-for-downloads",
        "--enable-plugin=pyside6",
        f"--windows-console-mode={target.console_mode}",
        f"--windows-icon-from-ico={target.icon}",
        f"--output-filename={target.executable}",
        f"--output-dir={build_root}",
        (
            f"--include-data-files={NATIVE_LIBRARY}="
            f"dll/{NATIVE_LIBRARY.name}"
        ),
        str(target.entry_point),
    ]

    for module_name in EXCLUDED_PACKAGING_IMPORTS:
        command_line.insert(-1, f"--nofollow-import-to={module_name}")

    if numpy_runtime is not None:
        package_root, dll_root = numpy_runtime
        command_line.insert(-1, f"--include-raw-dir={package_root}=numpy")
        command_line.insert(-1, f"--include-raw-dir={dll_root}=numpy.libs")
        for module_name in NUMPY_RUNTIME_SUPPORT_MODULES:
            command_line.insert(-1, f"--include-module={module_name}")

    for runtime in runtimes:
        command_line.insert(
            -1,
            f"--include-data-files={runtime}={runtime.name}",
        )

    return command_line


def _produced_distribution(
        build_root: Path,
) -> Path:
    candidates = list(
        build_root.glob("*.dist")
    )

    if (
            len(candidates) != 1
            or not candidates[0].is_dir()
    ):
        raise RuntimeError(
            "Nuitka did not produce exactly one "
            "standalone distribution directory"
        )

    return candidates[0]


def _verify_application(
        application: Path,
        target: ApplicationTarget,
) -> Path:
    executable = (
            application
            / target.executable
    )

    if (
            not executable.is_file()
            or executable.stat().st_size < 100_000
    ):
        raise RuntimeError(
            "Standalone output is missing "
            f"{target.executable} or it is "
            "unexpectedly small"
        )

    if not (
            application
            / "dll"
            / NATIVE_LIBRARY.name
    ).is_file():
        raise RuntimeError(
            f"{target.executable} is missing "
            "the native connectome DLL"
        )

    for runtime_name in (
            "msvcp140.dll",
            *RUNTIME_DLLS,
    ):
        if not (
                application
                / runtime_name
        ).is_file():
            raise RuntimeError(
                f"{target.executable} is missing "
                f"required runtime {runtime_name}"
            )

    return executable


def build(
        timestamp: str | None = None,
        targets: tuple[
            ApplicationTarget,
            ...,
        ] = APPLICATIONS,
) -> Path:
    if not VENV_PYTHON.is_file():
        raise RuntimeError(
            "Managed virtual environment is missing. "
            r"Run: py cli\installer.py"
        )

    if not NATIVE_LIBRARY.is_file():
        raise RuntimeError(
            "Native connectome DLL is missing. "
            r"Run: py cli\build_native.py"
        )

    for target in targets:
        if not target.icon.is_file():
            raise RuntimeError(
                "Required application icon is missing: "
                f"{target.icon}"
            )

    release_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    release = DIST_ROOT / f"AIBrain_{release_timestamp}"

    if release.exists():
        raise RuntimeError(
            "Refusing to overwrite existing distribution: "
            f"{release}"
        )

    release.mkdir(parents=True)

    info(f"Distribution directory: {release}")
    info("Checking required Visual C++ runtime DLLs")
    runtimes = runtime_dlls()

    try:
        for index, target in enumerate(
                targets,
                start=2,
        ):
            section(
                f"Compile {target.executable}",
                index,
            )

            build_root = (
                    release
                    / "_nuitka"
                    / target.directory
            )
            target_started = time.monotonic()
            info(f"Target {index - 1}/{len(targets)}: {target.executable}")

            numpy_runtime = stage_numpy_runtime(build_root)

            run(
                nuitka_command(
                    target,
                    build_root,
                    runtimes,
                    numpy_runtime,
                )
            )

            application = (
                    release
                    / target.directory
            )

            shutil.move(
                str(
                    _produced_distribution(
                        build_root
                    )
                ),
                application,
            )

            info(f"Verifying executable and required DLLs: {target.executable}")
            executable = _verify_application(
                application,
                target,
            )

            if target.executable == "ai_brain.exe":
                info("Setting the application's Windows GPU preference")
                set_windows_executable_gpu_preference(
                    executable
                )
            info(f"Completed {target.executable} in {time.monotonic() - target_started:.1f}s")

    finally:
        info(f"Removing intermediate build files: {release / '_nuitka'}")
        shutil.rmtree(
            release / "_nuitka",
            ignore_errors=True,
        )

    return release


def main() -> int:
    runtime_log, _ = configure_cli_logging("build_dist")
    if not require_managed_runtime(
            ROOT,
            "build_dist",
    ):
        return 1

    parser = argparse.ArgumentParser(
        description=(
            "Build timestamped standalone "
            "AIBrain desktop applications."
        )
    )

    parser.add_argument(
        "--timestamp",
        help=(
            "Override the YYYYMMDD_HHMMSS "
            "distribution suffix for "
            "reproducible builds"
        ),
    )

    parser.add_argument(
        "--only",
        choices=[
            target.directory
            for target in APPLICATIONS
        ],
        help=(
            "Build one application target "
            "for focused package verification"
        ),
    )

    arguments = parser.parse_args()

    clear_screen()
    header(
        "AIBrain",
        "Nuitka standalone distribution builder",
    )
    section("Build session", 1)
    info(f"Log file: {runtime_log}")
    info("Native Nuitka progress bars, build stages, warnings, and results appear below")

    try:
        selected = tuple(
            target
            for target in APPLICATIONS
            if target.directory == arguments.only
        )

        release = build(
            arguments.timestamp,
            selected or APPLICATIONS,
        )

    except (
            OSError,
            RuntimeError,
            subprocess.CalledProcessError,
    ) as exc:
        error(str(exc))
        return 1

    targets = selected or APPLICATIONS

    distribution_rows = [
        (
            target.executable,
            str(release / target.directory / target.executable),
        )
        for target in targets
    ]

    panel(
        "DISTRIBUTION COMPLETE",
        distribution_rows,
        footer="Each application folder is self-contained and ready to launch.",
        tone=Color.GREEN,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except KeyboardInterrupt:
        report_keyboard_interrupt("the distribution build")
        raise SystemExit(130)
    except Exception as exc:
        report_exception("AIBrain distribution build failed", exc)
        raise SystemExit(1)
