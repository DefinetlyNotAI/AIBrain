"""Shared ctypes utilities for native AIBrain bindings."""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Sequence
from ctypes import wintypes
from pathlib import Path
from threading import Lock
from types import TracebackType
from typing import Any, TypeVar, cast

# ---------------------------------------------------------------------------
# Common ctypes aliases
# ---------------------------------------------------------------------------

Structure = ctypes.Structure
NativeArray = ctypes.Array

CChar = ctypes.c_char
CFloat = ctypes.c_float
CInt16 = ctypes.c_int16
CInt32 = ctypes.c_int32
CLong = ctypes.c_long
CSizeT = ctypes.c_size_t
CVoidP = ctypes.c_void_p

Handle = wintypes.HANDLE
Bool = wintypes.BOOL
DWord = wintypes.DWORD
LPVoid = wintypes.LPVOID
LPCWStr = wintypes.LPCWSTR
LPWStr = wintypes.LPWSTR

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CTypesError(RuntimeError):
    """Raised when a native library or exported function cannot be prepared."""


# ---------------------------------------------------------------------------
# Generic ctypes helpers
# ---------------------------------------------------------------------------


def pointer(type_: type[T]) -> Any:
    """Return a ctypes pointer type."""
    return ctypes.POINTER(type_)


def byref(value: Any) -> Any:
    """Return a ctypes reference to a native value."""
    return ctypes.byref(value)


def sizeof(value: Any) -> int:
    """Return the native size of a ctypes value or type."""
    return ctypes.sizeof(value)


def cast_pointer(value: Any, type_: Any) -> Any:
    """Cast a ctypes value to another native pointer type."""
    return ctypes.cast(value, type_)


def byte_buffer(size: int) -> Any:
    """Allocate a mutable native byte buffer."""
    return ctypes.create_string_buffer(size)


def unicode_buffer(value: str) -> Any:
    """Allocate a mutable native Unicode buffer."""
    return ctypes.create_unicode_buffer(value)


# ---------------------------------------------------------------------------
# Windows error helpers
# ---------------------------------------------------------------------------


def last_error() -> int:
    """Return the calling thread's Win32 last-error value."""
    return ctypes.get_last_error()


def win_error(error_code: int | None = None) -> OSError:
    """Create the Windows exception for an error code."""
    if error_code is None:
        error_code = last_error()

    return ctypes.WinError(error_code)


def raise_last_error() -> None:
    """Raise the current Windows last-error value."""
    raise win_error()


def check_bool(result: int) -> int:
    """Raise WinError when a Win32 BOOL result indicates failure."""
    if not result:
        raise_last_error()

    return result


# ---------------------------------------------------------------------------
# Native library loading
# ---------------------------------------------------------------------------


class NativeLibrary:
    """Load and bind functions from one native dynamic library."""

    def __init__(
            self,
            path: str | Path,
            *,
            use_last_error: bool = False,
            optional: bool = False,
    ) -> None:
        self.path = path if isinstance(path, Path) else str(path)
        self.use_last_error = use_last_error
        self.optional = optional

        self._dll: ctypes.WinDLL | None = None
        self._lock = Lock()

    @property
    def loaded(self) -> bool:
        """Return whether the library has been loaded successfully."""
        return self._dll is not None

    @property
    def dll(self) -> ctypes.WinDLL:
        """Return the loaded DLL or raise when it is unavailable."""
        library = self.load()

        if library is None:
            raise CTypesError(
                f"Native library is unavailable: {self.path}"
            )

        return library

    def load(self) -> ctypes.WinDLL | None:
        """Load the library once and retain it for the process lifetime."""
        if self._dll is not None:
            return self._dll

        with self._lock:
            if self._dll is not None:
                return self._dll

            if isinstance(self.path, Path) and not self.path.exists():
                if self.optional:
                    return None

                raise CTypesError(
                    f"Native library does not exist: {self.path}"
                )

            try:
                self._dll = ctypes.WinDLL(
                    str(self.path),
                    use_last_error=self.use_last_error,
                )
            except OSError as exc:
                if self.optional:
                    return None

                raise CTypesError(
                    f"Could not load native library {self.path}: {exc}"
                ) from exc

        return self._dll

    def bind(
            self,
            name: str,
            *,
            argtypes: Sequence[Any] = (),
            restype: Any = None,
            required: bool = True,
    ) -> Callable[..., Any] | None:
        """Bind an exported function with an explicit ctypes signature."""
        library = self.load()

        if library is None:
            return None

        try:
            function = getattr(library, name)
        except AttributeError as exc:
            if not required:
                return None

            raise CTypesError(
                f"{self.path} does not export required function {name!r}"
            ) from exc

        function.argtypes = list(argtypes)
        function.restype = restype

        return cast(Callable[..., Any], function)

    def has_export(self, name: str) -> bool:
        """Return whether the loaded library exposes a named symbol."""
        library = self.load()

        return library is not None and hasattr(library, name)


# ---------------------------------------------------------------------------
# Windows handle ownership
# ---------------------------------------------------------------------------


class OwnedHandle:
    """Own a Windows HANDLE and close it exactly once."""

    def __init__(
            self,
            handle: Handle | None,
            closer: Callable[[Handle], None],
    ) -> None:
        self._handle = handle or Handle()
        self._closer = closer

    @property
    def value(self) -> Handle:
        """Return the current handle."""
        return self._handle

    @property
    def valid(self) -> bool:
        """Return whether the wrapper currently owns a valid handle."""
        return bool(self._handle)

    def detach(self) -> Handle:
        """Release ownership without closing the handle."""
        handle = self._handle
        self._handle = Handle()

        return handle

    def close(self) -> None:
        """Close the owned handle once."""
        if not self._handle:
            return

        handle = self._handle
        self._handle = Handle()

        self._closer(handle)

    def __enter__(self) -> OwnedHandle:
        """Return this handle owner for use as a context manager."""
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
    ) -> None:
        """Close the owned handle when leaving the context."""
        self.close()


# ---------------------------------------------------------------------------
# Win32 process structures
# ---------------------------------------------------------------------------


class SecurityAttributes(Structure):
    """Win32 SECURITY_ATTRIBUTES structure."""

    _fields_ = [
        ("nLength", DWord),
        ("lpSecurityDescriptor", LPVoid),
        ("bInheritHandle", Bool),
    ]


class StartupInfoW(Structure):
    """Win32 STARTUPINFOW structure."""

    _fields_ = [
        ("cb", DWord),
        ("lpReserved", LPWStr),
        ("lpDesktop", LPWStr),
        ("lpTitle", LPWStr),
        ("dwX", DWord),
        ("dwY", DWord),
        ("dwXSize", DWord),
        ("dwYSize", DWord),
        ("dwXCountChars", DWord),
        ("dwYCountChars", DWord),
        ("dwFillAttribute", DWord),
        ("dwFlags", DWord),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", pointer(ctypes.c_ubyte)),
        ("hStdInput", Handle),
        ("hStdOutput", Handle),
        ("hStdError", Handle),
    ]


class StartupInfoExW(Structure):
    """Win32 STARTUPINFOEXW structure."""

    _fields_ = [
        ("StartupInfo", StartupInfoW),
        ("lpAttributeList", LPVoid),
    ]


class ProcessInformation(Structure):
    """Win32 PROCESS_INFORMATION structure."""

    _fields_ = [
        ("hProcess", Handle),
        ("hThread", Handle),
        ("dwProcessId", DWord),
        ("dwThreadId", DWord),
    ]


# ---------------------------------------------------------------------------
# Win32 console structures
# ---------------------------------------------------------------------------


class ConsoleCoord(Structure):
    """Native Windows console coordinate."""

    _fields_ = [
        ("x", ctypes.c_short),
        ("y", ctypes.c_short),
    ]


class ConsoleSmallRect(Structure):
    """Native Windows console rectangle."""

    _fields_ = [
        ("left", ctypes.c_short),
        ("top", ctypes.c_short),
        ("right", ctypes.c_short),
        ("bottom", ctypes.c_short),
    ]


class ConsoleScreenBufferInfo(Structure):
    """Native Windows console screen-buffer information."""

    _fields_ = [
        ("size", ConsoleCoord),
        ("cursor", ConsoleCoord),
        ("attributes", ctypes.c_ushort),
        ("window", ConsoleSmallRect),
        ("maximum_window_size", ConsoleCoord),
    ]


# ---------------------------------------------------------------------------
# Windows console API
# ---------------------------------------------------------------------------


class WindowsConsole:
    """Centralize native Windows console operations."""

    STD_OUTPUT_HANDLE = -11

    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000

    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002

    OPEN_EXISTING = 3

    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004

    INVALID_HANDLE_VALUE = CVoidP(-1).value

    def __init__(self) -> None:
        self._library = NativeLibrary(
            "kernel32",
            use_last_error=True,
        )

        self._get_std_handle = self._require(
            "GetStdHandle",
            argtypes=(CLong,),
            restype=Handle,
        )

        self._get_console_screen_buffer_info = self._require(
            "GetConsoleScreenBufferInfo",
            argtypes=(
                Handle,
                pointer(ConsoleScreenBufferInfo),
            ),
            restype=Bool,
        )

        self._fill_console_output_character = self._require(
            "FillConsoleOutputCharacterW",
            argtypes=(
                Handle,
                ctypes.c_wchar,
                DWord,
                ConsoleCoord,
                pointer(DWord),
            ),
            restype=Bool,
        )

        self._fill_console_output_attribute = self._require(
            "FillConsoleOutputAttribute",
            argtypes=(
                Handle,
                wintypes.WORD,
                DWord,
                ConsoleCoord,
                pointer(DWord),
            ),
            restype=Bool,
        )

        self._set_console_cursor_position = self._require(
            "SetConsoleCursorPosition",
            argtypes=(
                Handle,
                ConsoleCoord,
            ),
            restype=Bool,
        )

        self._create_file = self._require(
            "CreateFileW",
            argtypes=(
                LPCWStr,
                DWord,
                DWord,
                LPVoid,
                DWord,
                DWord,
                Handle,
            ),
            restype=Handle,
        )

        self._close_handle = self._require(
            "CloseHandle",
            argtypes=(Handle,),
            restype=Bool,
        )

        self._get_console_mode = self._require(
            "GetConsoleMode",
            argtypes=(
                Handle,
                pointer(DWord),
            ),
            restype=Bool,
        )

        self._set_console_mode = self._require(
            "SetConsoleMode",
            argtypes=(
                Handle,
                DWord,
            ),
            restype=Bool,
        )

    def _require(
            self,
            name: str,
            *,
            argtypes: Sequence[Any],
            restype: Any,
    ) -> Callable[..., Any]:
        """Bind one required Kernel32 console export."""
        function = self._library.bind(
            name,
            argtypes=argtypes,
            restype=restype,
        )

        if function is None:
            raise CTypesError(
                f"Kernel32 export {name!r} could not be loaded"
            )

        return function

    def output_handle(self) -> Handle:
        """Return the process standard-output handle."""
        return self._get_std_handle(self.STD_OUTPUT_HANDLE)

    def console_width(self) -> int | None:
        """Return the attached Windows console viewport width."""
        handle = self.output_handle()

        if not self._valid_handle(handle):
            return None

        info = ConsoleScreenBufferInfo()

        if not self._get_console_screen_buffer_info(
                handle,
                byref(info),
        ):
            return None

        return int(
            info.window.right
            - info.window.left
            + 1
        )

    def clear_output(self) -> bool:
        """Clear the complete Windows console buffer."""
        handle = self.output_handle()
        owns_handle = False

        if not self._has_console_buffer(handle):
            handle = self._open_console_output()

            if not self._valid_handle(handle):
                return False

            owns_handle = True

        try:
            info = ConsoleScreenBufferInfo()

            if not self._get_console_screen_buffer_info(
                    handle,
                    byref(info),
            ):
                return False

            cells = int(info.size.x * info.size.y)

            if cells <= 0:
                return False

            characters_written = DWord()
            attributes_written = DWord()
            origin = ConsoleCoord(0, 0)

            characters_cleared = self._fill_console_output_character(
                handle,
                " ",
                cells,
                origin,
                byref(characters_written),
            )

            attributes_cleared = self._fill_console_output_attribute(
                handle,
                info.attributes,
                cells,
                origin,
                byref(attributes_written),
            )

            cursor_reset = self._set_console_cursor_position(
                handle,
                origin,
            )

            return bool(
                characters_cleared
                and attributes_cleared
                and cursor_reset
                and characters_written.value == cells
                and attributes_written.value == cells
            )

        finally:
            if owns_handle:
                self.close_handle(handle)

    def enable_virtual_terminal_output(self) -> bool:
        """Enable ANSI virtual-terminal processing for standard output."""
        handle = self.output_handle()

        if not self._valid_handle(handle):
            return False

        mode = DWord()

        if not self._get_console_mode(
                handle,
                byref(mode),
        ):
            return False

        if mode.value & self.ENABLE_VIRTUAL_TERMINAL_PROCESSING:
            return True

        new_mode = (
                mode.value
                | self.ENABLE_VIRTUAL_TERMINAL_PROCESSING
        )

        return bool(
            self._set_console_mode(
                handle,
                new_mode,
            )
        )

    def close_handle(self, handle: Handle) -> None:
        """Close a Windows handle when valid."""
        if not self._valid_handle(handle):
            return

        if not self._close_handle(handle):
            raise win_error()

    def _has_console_buffer(self, handle: Handle) -> bool:
        """Return whether a handle targets a usable console buffer."""
        if not self._valid_handle(handle):
            return False

        info = ConsoleScreenBufferInfo()

        return bool(
            self._get_console_screen_buffer_info(
                handle,
                byref(info),
            )
        )

    def _open_console_output(self) -> Handle:
        """Open CONOUT$ when standard output itself is redirected."""
        return self._create_file(
            "CONOUT$",
            self.GENERIC_READ | self.GENERIC_WRITE,
            self.FILE_SHARE_READ | self.FILE_SHARE_WRITE,
            None,
            self.OPEN_EXISTING,
            0,
            None,
        )

    def _valid_handle(self, handle: Handle) -> bool:
        """Return whether a Windows HANDLE is usable."""
        if not handle:
            return False

        value = getattr(
            handle,
            "value",
            handle,
        )

        return value not in (
            None,
            0,
            self.INVALID_HANDLE_VALUE,
        )


# ---------------------------------------------------------------------------
# Shared native services
# ---------------------------------------------------------------------------

windows_console = WindowsConsole()
