"""Compile timestamped standalone AIBrain desktop applications with Nuitka."""
from __future__ import annotations

import argparse
import codecs
import ctypes
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
    Color,
    CommandOutputBox,
    clear_screen,
    command_preview,
    error,
    header,
    panel,
    section,
)
from src.utils.gpu import set_windows_executable_gpu_preference
from src.utils.runtime import require_managed_runtime

VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
DIST_ROOT = ROOT / "dist"

RUNTIME_DLLS = ("vcomp140.dll",)
NATIVE_LIBRARY = ROOT / "dll" / "aibrain.connectome.dll"


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


def _render_output_text(
        text: str,
        pending: str,
        output_box: CommandOutputBox,
) -> str:
    """Render complete lines and immediately redraw carriage-return frames."""
    new_output = pending + _clean_terminal_output(text)
    pending = ""

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

            if pending:
                output_box.write_partial(pending)

            break

        line = new_output[:newline]
        terminator = new_output[newline]
        new_output = new_output[newline + 1:]

        if terminator == "\r":
            # CRLF is a normal completed line.
            if new_output.startswith("\n"):
                output_box.write(line)
                new_output = new_output[1:]
                continue

            # A lone carriage return is a live terminal frame. Nuitka and SCons
            # use this while updating compilation progress.
            output_box.write_partial(line)
            continue

        output_box.write(line)

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
        output = output_file.read()

    if len(output) <= offset:
        return offset, pending

    text = output[offset:]
    offset = len(output)

    pending = _render_output_text(
        text,
        pending,
        output_box,
    )

    return offset, pending


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
    """Return whether Windows ConPTY can be used."""
    return os.name == "nt"


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

        terminal_size = shutil.get_terminal_size((120, 40))

        coord = _ConsoleCoord(
            max(60, min(terminal_size.columns, 240)),
            max(20, min(terminal_size.lines, 120)),
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
    reader_finished = False

    try:
        with CommandOutputBox() as output_box:
            while True:
                try:
                    chunk = chunks.get(timeout=0.05)
                except queue.Empty:
                    chunk = b""

                if chunk is None:
                    reader_finished = True

                elif chunk:
                    text = decoder.decode(
                        chunk,
                        final=False,
                    )

                    if text:
                        pending = _render_output_text(
                            text,
                            pending,
                            output_box,
                        )

                return_code = process.poll()

                if return_code is not None:
                    process.close_terminal()

                if return_code is not None and reader_finished:
                    break

            remaining = decoder.decode(
                b"",
                final=True,
            )

            if remaining:
                pending = _render_output_text(
                    remaining,
                    pending,
                    output_box,
                )

            if pending:
                output_box.write(pending)

        reader.join()
        return_code = process.wait()

    except KeyboardInterrupt:
        _stop_process_tree(process.pid)

        process.close_terminal()
        reader.join()

        process.wait()
        raise

    finally:
        process.close_terminal()

        if reader.is_alive():
            reader.join()

        process.close()

    if return_code:
        raise subprocess.CalledProcessError(
            return_code,
            command_line,
        )


def _run_with_file_tailer(
        command_line: list[str],
) -> None:
    """Fallback runner for platforms without Windows ConPTY."""
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

                with CommandOutputBox() as output_box:
                    while process.poll() is None:
                        output_file.flush()

                        offset, pending = _render_new_build_output(
                            output_path,
                            offset,
                            pending,
                            output_box,
                        )

                        time.sleep(0.05)

                    output_file.flush()

                    offset, pending = _render_new_build_output(
                        output_path,
                        offset,
                        pending,
                        output_box,
                    )

                    if pending:
                        output_box.write(pending)

                return_code = process.wait()

            except KeyboardInterrupt:
                _stop_interrupted_build(process)
                raise

        captured_output = output_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).rstrip()

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


def nuitka_command(
        target: ApplicationTarget,
        build_root: Path,
        runtimes: list[Path],
) -> list[str]:
    """Build a reproducible standalone Nuitka command."""
    command_line = [
        str(VENV_PYTHON),
        "-u",
        "-m",
        "nuitka",
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

    runtimes = runtime_dlls()

    try:
        for index, target in enumerate(
                targets,
                start=1,
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

            run(
                nuitka_command(
                    target,
                    build_root,
                    runtimes,
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

            executable = _verify_application(
                application,
                target,
            )

            if target.executable == "ai_brain.exe":
                set_windows_executable_gpu_preference(
                    executable
                )

    finally:
        shutil.rmtree(
            release / "_nuitka",
            ignore_errors=True,
        )

    return release


def main() -> int:
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
        error(
            "Distribution build cancelled "
            "by keyboard interrupt."
        )
        print()
        raise SystemExit(130)
