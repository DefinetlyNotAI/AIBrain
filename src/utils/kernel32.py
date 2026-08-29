from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path
from typing import Final


class Kernel32:
    """Typed wrapper around the Win32 Kernel32 API used by ConPTY."""

    WAIT_OBJECT_0: Final[int] = 0x00000000
    WAIT_TIMEOUT: Final[int] = 0x00000102
    INFINITE: Final[int] = 0xFFFFFFFF
    STILL_ACTIVE: Final[int] = 259

    EXTENDED_STARTUPINFO_PRESENT: Final[int] = 0x00080000
    CREATE_UNICODE_ENVIRONMENT: Final[int] = 0x00000400
    CREATE_NEW_PROCESS_GROUP: Final[int] = 0x00000200

    ERROR_BROKEN_PIPE: Final[int] = 109
    ERROR_NO_DATA: Final[int] = 232
    ERROR_INVALID_HANDLE: Final[int] = 6

    PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE: Final[int] = 0x00020016

    def __init__(self) -> None:
        self._dll = ctypes.WinDLL(
            "kernel32",
            use_last_error=True,
        )

        self._wait_for_single_object = getattr(
            self._dll,
            "WaitForSingleObject",
        )
        self._wait_for_single_object.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        self._wait_for_single_object.restype = wintypes.DWORD

        self._get_exit_code_process = getattr(
            self._dll,
            "GetExitCodeProcess",
        )
        self._get_exit_code_process.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._get_exit_code_process.restype = wintypes.BOOL

        self._close_handle = getattr(
            self._dll,
            "CloseHandle",
        )
        self._close_handle.argtypes = [
            wintypes.HANDLE,
        ]
        self._close_handle.restype = wintypes.BOOL

        self._create_pipe = getattr(
            self._dll,
            "CreatePipe",
        )
        self._create_pipe.argtypes = [
            ctypes.POINTER(wintypes.HANDLE),
            ctypes.POINTER(wintypes.HANDLE),
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        self._create_pipe.restype = wintypes.BOOL

        self._read_file = getattr(
            self._dll,
            "ReadFile",
        )
        self._read_file.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        self._read_file.restype = wintypes.BOOL

        self._create_process_w = getattr(
            self._dll,
            "CreateProcessW",
        )
        self._create_process_w.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.LPCWSTR,
            wintypes.LPVOID,
            wintypes.LPVOID,
        ]
        self._create_process_w.restype = wintypes.BOOL

        self._initialize_proc_thread_attribute_list = getattr(
            self._dll,
            "InitializeProcThreadAttributeList",
        )
        self._initialize_proc_thread_attribute_list.restype = wintypes.BOOL

        self._update_proc_thread_attribute = getattr(
            self._dll,
            "UpdateProcThreadAttribute",
        )
        self._update_proc_thread_attribute.restype = wintypes.BOOL

        self._delete_proc_thread_attribute_list = getattr(
            self._dll,
            "DeleteProcThreadAttributeList",
        )
        self._delete_proc_thread_attribute_list.argtypes = [
            wintypes.LPVOID,
        ]
        self._delete_proc_thread_attribute_list.restype = None

        self._create_pseudo_console = getattr(
            self._dll,
            "CreatePseudoConsole",
        )
        self._create_pseudo_console.restype = ctypes.c_long

        self._close_pseudo_console = getattr(
            self._dll,
            "ClosePseudoConsole",
        )
        self._close_pseudo_console.argtypes = [
            wintypes.HANDLE,
        ]
        self._close_pseudo_console.restype = None

    def wait_for_single_object(
            self,
            handle: wintypes.HANDLE,
            timeout: int,
    ) -> int:
        return int(
            self._wait_for_single_object(
                handle,
                timeout,
            )
        )

    def get_exit_code_process(
            self,
            handle: wintypes.HANDLE,
    ) -> int:
        exit_code = wintypes.DWORD()

        if not self._get_exit_code_process(
                handle,
                ctypes.byref(exit_code),
        ):
            raise ctypes.WinError(
                ctypes.get_last_error()
            )

        return int(exit_code.value)

    def close_handle(
            self,
            handle: wintypes.HANDLE,
    ) -> None:
        if not handle:
            return

        if not self._close_handle(handle):
            raise ctypes.WinError(
                ctypes.get_last_error()
            )

    def create_pipe(
            self,
            security_attributes: ctypes.Structure,
    ) -> tuple[wintypes.HANDLE, wintypes.HANDLE]:
        read_handle = wintypes.HANDLE()
        write_handle = wintypes.HANDLE()

        if not self._create_pipe(
                ctypes.byref(read_handle),
                ctypes.byref(write_handle),
                ctypes.byref(security_attributes),
                0,
        ):
            raise ctypes.WinError(
                ctypes.get_last_error()
            )

        return read_handle, write_handle

    def read_file(
            self,
            handle: wintypes.HANDLE,
            size: int = 8192,
    ) -> bytes | None:
        buffer = ctypes.create_string_buffer(size)
        bytes_read = wintypes.DWORD()

        if not self._read_file(
                handle,
                buffer,
                size,
                ctypes.byref(bytes_read),
                None,
        ):
            error_code = ctypes.get_last_error()

            if error_code in (
                    self.ERROR_INVALID_HANDLE,
                    self.ERROR_BROKEN_PIPE,
                    self.ERROR_NO_DATA,
            ):
                return None

            raise ctypes.WinError(error_code)

        if bytes_read.value == 0:
            return b""

        return bytes(
            buffer.raw[:bytes_read.value]
        )

    def initialize_attribute_list(
            self,
    ) -> tuple[
        ctypes.Array[ctypes.c_char],
        wintypes.LPVOID,
    ]:
        size = ctypes.c_size_t()

        self._initialize_proc_thread_attribute_list(
            None,
            1,
            0,
            ctypes.byref(size),
        )

        buffer = ctypes.create_string_buffer(
            size.value
        )

        attribute_list = ctypes.cast(
            buffer,
            wintypes.LPVOID,
        )

        if not self._initialize_proc_thread_attribute_list(
                attribute_list,
                1,
                0,
                ctypes.byref(size),
        ):
            raise ctypes.WinError(
                ctypes.get_last_error()
            )

        return buffer, attribute_list

    def update_pseudo_console_attribute(
            self,
            attribute_list: wintypes.LPVOID,
            pseudo_console: wintypes.HANDLE,
    ) -> None:
        if not self._update_proc_thread_attribute(
                attribute_list,
                0,
                self.PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE,
                pseudo_console,
                ctypes.sizeof(wintypes.HANDLE),
                None,
                None,
        ):
            raise ctypes.WinError(
                ctypes.get_last_error()
            )

    def delete_attribute_list(
            self,
            attribute_list: wintypes.LPVOID,
    ) -> None:
        self._delete_proc_thread_attribute_list(
            attribute_list
        )

    def create_pseudo_console(
            self,
            size: ctypes.Structure,
            input_handle: wintypes.HANDLE,
            output_handle: wintypes.HANDLE,
    ) -> wintypes.HANDLE:
        pseudo_console = wintypes.HANDLE()

        result = self._create_pseudo_console(
            size,
            input_handle,
            output_handle,
            0,
            ctypes.byref(pseudo_console),
        )

        if result != 0:
            raise OSError(
                "CreatePseudoConsole failed with HRESULT "
                f"0x{result & 0xFFFFFFFF:08X}"
            )

        return pseudo_console

    def close_pseudo_console(
            self,
            pseudo_console: wintypes.HANDLE,
    ) -> None:
        if pseudo_console:
            self._close_pseudo_console(
                pseudo_console
            )

    def create_process(
            self,
            command_line: ctypes.Array[ctypes.c_wchar],
            current_directory: Path,
            startup_info: ctypes.Structure,
            process_information: ctypes.Structure,
            creation_flags: int,
    ) -> None:
        """Create a Windows process using extended startup information."""
        environment = None

        if not self._create_process_w(
                None,
                command_line,
                None,
                None,
                False,
                creation_flags,
                environment,
                str(current_directory),
                ctypes.byref(startup_info.StartupInfo),
                ctypes.byref(process_information),
        ):
            raise ctypes.WinError(
                ctypes.get_last_error()
            )


kernel32 = Kernel32()
