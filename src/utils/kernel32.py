from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from src.app.ctypes_helper import (
    Bool,
    CLong,
    CSizeT,
    DWord,
    Handle,
    LPCWStr,
    LPVoid,
    LPWStr,
    NativeLibrary,
    Structure,
    byref,
    byte_buffer,
    cast_pointer,
    last_error,
    pointer,
    sizeof,
    win_error,
)


class Kernel32:
    """Typed wrapper around the Win32 Kernel32 API used by ConPTY."""

    WAIT_OBJECT_0: Final[int] = 0x00000000
    WAIT_TIMEOUT: Final[int] = 0x00000102
    INFINITE: Final[int] = 0xFFFFFFFF
    STILL_ACTIVE: Final[int] = 259

    EXTENDED_STARTUPINFO_PRESENT: Final[int] = 0x00080000
    CREATE_UNICODE_ENVIRONMENT: Final[int] = 0x00000400
    CREATE_NEW_PROCESS_GROUP: Final[int] = 0x00000200

    ERROR_ACCESS_DENIED: Final[int] = 5
    ERROR_INVALID_HANDLE: Final[int] = 6
    ERROR_INVALID_PARAMETER: Final[int] = 87
    ERROR_BROKEN_PIPE: Final[int] = 109
    ERROR_NO_DATA: Final[int] = 232

    PROCESS_QUERY_LIMITED_INFORMATION: Final[int] = 0x1000

    PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE: Final[int] = 0x00020016

    def __init__(self) -> None:
        self._library = NativeLibrary(
            "kernel32",
            use_last_error=True,
        )

        self._wait_for_single_object = self._bind(
            "WaitForSingleObject",
            argtypes=(
                Handle,
                DWord,
            ),
            restype=DWord,
        )

        self._get_exit_code_process = self._bind(
            "GetExitCodeProcess",
            argtypes=(
                Handle,
                pointer(DWord),
            ),
            restype=Bool,
        )

        self._close_handle = self._bind(
            "CloseHandle",
            argtypes=(Handle,),
            restype=Bool,
        )

        self._open_process = self._bind(
            "OpenProcess",
            argtypes=(
                DWord,
                Bool,
                DWord,
            ),
            restype=Handle,
        )

        self._create_pipe = self._bind(
            "CreatePipe",
            argtypes=(
                pointer(Handle),
                pointer(Handle),
                LPVoid,
                DWord,
            ),
            restype=Bool,
        )

        self._read_file = self._bind(
            "ReadFile",
            argtypes=(
                Handle,
                LPVoid,
                DWord,
                pointer(DWord),
                LPVoid,
            ),
            restype=Bool,
        )

        self._create_process_w = self._bind(
            "CreateProcessW",
            argtypes=(
                LPCWStr,
                LPWStr,
                LPVoid,
                LPVoid,
                Bool,
                DWord,
                LPVoid,
                LPCWStr,
                LPVoid,
                LPVoid,
            ),
            restype=Bool,
        )

        self._initialize_proc_thread_attribute_list = self._bind(
            "InitializeProcThreadAttributeList",
            argtypes=(
                LPVoid,
                DWord,
                DWord,
                pointer(CSizeT),
            ),
            restype=Bool,
        )

        self._update_proc_thread_attribute = self._bind(
            "UpdateProcThreadAttribute",
            argtypes=(
                LPVoid,
                DWord,
                CSizeT,
                LPVoid,
                CSizeT,
                LPVoid,
                pointer(CSizeT),
            ),
            restype=Bool,
        )

        self._delete_proc_thread_attribute_list = self._bind(
            "DeleteProcThreadAttributeList",
            argtypes=(LPVoid,),
            restype=None,
        )

        self._create_pseudo_console = self._bind(
            "CreatePseudoConsole",
            argtypes=(
                LPVoid,
                Handle,
                Handle,
                DWord,
                pointer(Handle),
            ),
            restype=CLong,
        )

        self._close_pseudo_console = self._bind(
            "ClosePseudoConsole",
            argtypes=(Handle,),
            restype=None,
        )

    def _bind(
            self,
            name: str,
            *,
            argtypes: tuple[Any, ...],
            restype: Any,
    ) -> Any:
        """Bind one required Kernel32 export."""
        function = self._library.bind(
            name,
            argtypes=argtypes,
            restype=restype,
        )

        if function is None:
            raise RuntimeError(
                f"Required Kernel32 export {name!r} is unavailable"
            )

        return function

    def wait_for_single_object(
            self,
            handle: Handle,
            timeout: int,
    ) -> int:
        """Wait for a kernel object to become signaled."""
        return int(
            self._wait_for_single_object(
                handle,
                timeout,
            )
        )

    def get_exit_code_process(
            self,
            handle: Handle,
    ) -> int:
        """Return the current exit code for a process handle."""
        exit_code = DWord()

        if not self._get_exit_code_process(
                handle,
                byref(exit_code),
        ):
            raise win_error()

        return int(exit_code.value)

    def process_exists(
            self,
            pid: int,
    ) -> bool:
        """Return whether a Windows process ID still represents a live process."""
        if pid <= 0:
            return False

        handle = self._open_process(
            self.PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            pid,
        )

        if handle:
            try:
                return True
            finally:
                self.close_handle(handle)

        error_code = last_error()

        if error_code == self.ERROR_INVALID_PARAMETER:
            return False

        if error_code == self.ERROR_ACCESS_DENIED:
            return True

        # For cleanup callers, ambiguous failures should be treated as alive
        # rather than risking deletion of resources belonging to a live process.
        return True

    def close_handle(
            self,
            handle: Handle,
    ) -> None:
        """Close a valid Windows kernel handle."""
        if not handle:
            return

        if not self._close_handle(handle):
            raise win_error()

    def create_pipe(
            self,
            security_attributes: Structure,
    ) -> tuple[Handle, Handle]:
        """Create an anonymous inheritable pipe."""
        read_handle = Handle()
        write_handle = Handle()

        if not self._create_pipe(
                byref(read_handle),
                byref(write_handle),
                byref(security_attributes),
                0,
        ):
            raise win_error()

        return read_handle, write_handle

    def read_file(
            self,
            handle: Handle,
            size: int = 8192,
    ) -> bytes | None:
        """Read bytes from a native Windows handle."""
        if size <= 0:
            raise ValueError("Read size must be greater than zero")

        buffer = byte_buffer(size)
        bytes_read = DWord()

        if not self._read_file(
                handle,
                buffer,
                size,
                byref(bytes_read),
                None,
        ):
            error_code = last_error()

            if error_code in (
                    self.ERROR_INVALID_HANDLE,
                    self.ERROR_BROKEN_PIPE,
                    self.ERROR_NO_DATA,
            ):
                return None

            raise win_error(error_code)

        if bytes_read.value == 0:
            return b""

        return bytes(
            buffer.raw[:bytes_read.value]
        )

    def initialize_attribute_list(
            self,
    ) -> tuple[Any, LPVoid]:
        """Allocate and initialize one process-thread attribute list."""
        size = CSizeT()

        self._initialize_proc_thread_attribute_list(
            None,
            1,
            0,
            byref(size),
        )

        if size.value <= 0:
            raise win_error()

        buffer = byte_buffer(
            size.value
        )

        attribute_list = cast_pointer(
            buffer,
            LPVoid,
        )

        if not self._initialize_proc_thread_attribute_list(
                attribute_list,
                1,
                0,
                byref(size),
        ):
            raise win_error()

        return buffer, attribute_list

    def update_pseudo_console_attribute(
            self,
            attribute_list: LPVoid,
            pseudo_console: Handle,
    ) -> None:
        """Attach a pseudo-console handle to an extended startup attribute list."""
        if not self._update_proc_thread_attribute(
                attribute_list,
                0,
                self.PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE,
                pseudo_console,
                sizeof(Handle),
                None,
                None,
        ):
            raise win_error()

    def delete_attribute_list(
            self,
            attribute_list: LPVoid,
    ) -> None:
        """Destroy an initialized process-thread attribute list."""
        if attribute_list:
            self._delete_proc_thread_attribute_list(
                attribute_list
            )

    def create_pseudo_console(
            self,
            size: Structure,
            input_handle: Handle,
            output_handle: Handle,
    ) -> Handle:
        """Create a ConPTY pseudo console."""
        pseudo_console = Handle()

        result = int(
            self._create_pseudo_console(
                size,
                input_handle,
                output_handle,
                0,
                byref(pseudo_console),
            )
        )

        if result < 0:
            raise OSError(
                "CreatePseudoConsole failed with "
                f"HRESULT 0x{result & 0xFFFFFFFF:08X}"
            )

        return pseudo_console

    def close_pseudo_console(
            self,
            pseudo_console: Handle,
    ) -> None:
        """Close a ConPTY pseudo-console handle."""
        if pseudo_console:
            self._close_pseudo_console(
                pseudo_console
            )

    def create_process(
            self,
            command_line: Any,
            current_directory: Path,
            startup_info: Structure,
            process_information: Structure,
            creation_flags: int,
    ) -> None:
        """Create a Windows process using extended startup information."""
        startup_info_ex = getattr(
            startup_info,
            "StartupInfo",
            None,
        )

        if startup_info_ex is None:
            raise TypeError(
                "startup_info must expose a StartupInfo structure"
            )

        if not self._create_process_w(
                None,
                command_line,
                None,
                None,
                False,
                creation_flags,
                None,
                str(current_directory),
                byref(startup_info_ex),
                byref(process_information),
        ):
            raise win_error()


kernel32 = Kernel32()
