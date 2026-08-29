"""The shared terminal presentation library for all AIBrain CLI commands."""
from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path
from typing import Callable, TextIO, cast

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WIDTH = 82
MIN_WIDTH = 60
RIGHT_EDGE_MARGIN = 4
COMMAND_INDENT = 2
ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|][^\x07]*(?:\x07|\x1b\\))")


class _ConsoleCoord(ctypes.Structure):
    _fields_ = [("x", ctypes.c_short), ("y", ctypes.c_short)]


class _ConsoleSmallRect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_short),
        ("top", ctypes.c_short),
        ("right", ctypes.c_short),
        ("bottom", ctypes.c_short),
    ]


class _ConsoleScreenBufferInfo(ctypes.Structure):
    _fields_ = [
        ("size", _ConsoleCoord),
        ("cursor", _ConsoleCoord),
        ("attributes", ctypes.c_ushort),
        ("window", _ConsoleSmallRect),
        ("maximum_window_size", _ConsoleCoord),
    ]


_GetStdHandle = ctypes.WINFUNCTYPE(wintypes.HANDLE, ctypes.c_long)
_GetConsoleScreenBufferInfo = ctypes.WINFUNCTYPE(
    wintypes.BOOL,
    wintypes.HANDLE,
    ctypes.POINTER(_ConsoleScreenBufferInfo),
)
_FillConsoleOutputCharacter = ctypes.WINFUNCTYPE(
    wintypes.BOOL,
    wintypes.HANDLE,
    ctypes.c_wchar,
    wintypes.DWORD,
    _ConsoleCoord,
    ctypes.POINTER(wintypes.DWORD),
)
_FillConsoleOutputAttribute = ctypes.WINFUNCTYPE(
    wintypes.BOOL,
    wintypes.HANDLE,
    wintypes.WORD,
    wintypes.DWORD,
    _ConsoleCoord,
    ctypes.POINTER(wintypes.DWORD),
)
_SetConsoleCursorPosition = ctypes.WINFUNCTYPE(
    wintypes.BOOL,
    wintypes.HANDLE,
    _ConsoleCoord,
)
_CreateFile = ctypes.WINFUNCTYPE(
    wintypes.HANDLE,
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
)
_CloseHandle = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE)

_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_OPEN_EXISTING = 3


class _Kernel32Bindings:
    """Explicit callable contracts for the Windows console APIs we use."""

    # noinspection bad-argument-type
    def __init__(self) -> None:
        library = ctypes.WinDLL("kernel32", use_last_error=True)
        self.get_std_handle: Callable[[int], int] = cast(
            Callable[[int], int],
            _GetStdHandle(("GetStdHandle", library)),
        )
        self.get_console_screen_buffer_info: Callable[..., int] = cast(
            Callable[..., int],
            _GetConsoleScreenBufferInfo(("GetConsoleScreenBufferInfo", library)),
        )
        self.fill_console_output_character: Callable[..., int] = cast(
            Callable[..., int],
            _FillConsoleOutputCharacter(("FillConsoleOutputCharacterW", library)),
        )
        self.fill_console_output_attribute: Callable[..., int] = cast(
            Callable[..., int],
            _FillConsoleOutputAttribute(("FillConsoleOutputAttribute", library)),
        )
        self.set_console_cursor_position: Callable[..., int] = cast(
            Callable[..., int],
            _SetConsoleCursorPosition(("SetConsoleCursorPosition", library)),
        )
        self.create_file: Callable[..., int] = cast(
            Callable[..., int],
            _CreateFile(("CreateFileW", library)),
        )
        self.close_handle: Callable[[int], int] = cast(
            Callable[[int], int],
            _CloseHandle(("CloseHandle", library)),
        )


def _kernel32_bindings() -> _Kernel32Bindings:
    """Return explicitly typed Kernel32 callables without dynamic DLL attributes."""
    return _Kernel32Bindings()


try:
    "\N{BOX DRAWINGS LIGHT ARC DOWN AND RIGHT}".encode(sys.stdout.encoding or "utf-8")
except (LookupError, UnicodeEncodeError):
    # Do not mutate an IDE, pipe, or redirected stream.  Hosts that cannot
    # encode box drawing receive the same layout with portable ASCII glyphs.
    BOX_HORIZONTAL, BOX_VERTICAL = "-", "|"
    BOX_TOP_LEFT = BOX_TOP_RIGHT = BOX_BOTTOM_LEFT = BOX_BOTTOM_RIGHT = "+"
    BOX_MID_LEFT = BOX_MID_RIGHT = "+"
    BULLET, CHECK, CROSS, PROMPT = "*", "OK", "X", ">"
else:
    BOX_HORIZONTAL = "\N{BOX DRAWINGS LIGHT HORIZONTAL}"
    BOX_VERTICAL = "\N{BOX DRAWINGS LIGHT VERTICAL}"
    BOX_TOP_LEFT = "\N{BOX DRAWINGS LIGHT ARC DOWN AND RIGHT}"
    BOX_TOP_RIGHT = "\N{BOX DRAWINGS LIGHT ARC DOWN AND LEFT}"
    BOX_BOTTOM_LEFT = "\N{BOX DRAWINGS LIGHT ARC UP AND RIGHT}"
    BOX_BOTTOM_RIGHT = "\N{BOX DRAWINGS LIGHT ARC UP AND LEFT}"
    BOX_MID_LEFT = "\N{BOX DRAWINGS LIGHT VERTICAL AND RIGHT}"
    BOX_MID_RIGHT = "\N{BOX DRAWINGS LIGHT VERTICAL AND LEFT}"
    BULLET = "\N{BLACK CIRCLE}"
    CHECK = "\N{CHECK MARK}"
    CROSS = "\N{MULTIPLICATION X}"
    PROMPT = "\N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK}"


class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


def color(text: str, *styles: str) -> str:
    """Apply ANSI styles to CLI text."""
    return "".join(styles) + text + Color.RESET


def terminal_width() -> int:
    """Return the visible terminal width with a usable minimum."""
    width = shutil.get_terminal_size((DEFAULT_WIDTH, 24)).columns
    if os.name == "nt":
        try:
            info = _ConsoleScreenBufferInfo()
            kernel32 = _kernel32_bindings()
            handle = kernel32.get_std_handle(-11)
            if handle and kernel32.get_console_screen_buffer_info(handle, ctypes.byref(info)):
                width = info.window.right - info.window.left + 1
        except (AttributeError, OSError):
            pass
    return max(width - RIGHT_EDGE_MARGIN, MIN_WIDTH)


def _clear_native_console() -> bool:
    """Clear the entire attached Windows console buffer, not just its viewport."""
    kernel32 = _kernel32_bindings()
    handle = kernel32.get_std_handle(-11)
    owns_handle = False
    info = _ConsoleScreenBufferInfo()
    if not handle or not kernel32.get_console_screen_buffer_info(handle, ctypes.byref(info)):
        # stdout may be captured by an IDE or redirected while the process still
        # owns an interactive console. CONOUT$ addresses that console directly.
        handle = kernel32.create_file(
            "CONOUT$",
            _GENERIC_READ | _GENERIC_WRITE,
            _FILE_SHARE_READ | _FILE_SHARE_WRITE,
            None,
            _OPEN_EXISTING,
            0,
            None,
        )
        owns_handle = True
        if not handle or not kernel32.get_console_screen_buffer_info(handle, ctypes.byref(info)):
            if handle:
                kernel32.close_handle(handle)
            return False
    try:
        # The second parameter is a 16-bit WCHAR value, not a string pointer.
        # Passing a pointer here produces repeated CJK glyphs from its low word.
        cells = info.size.x * info.size.y
        characters_written = wintypes.DWORD()
        attributes_written = wintypes.DWORD()
        origin = _ConsoleCoord(0, 0)
        characters_cleared = kernel32.fill_console_output_character(handle, " ", cells, origin,
                                                                    ctypes.byref(characters_written))
        attributes_cleared = kernel32.fill_console_output_attribute(handle, info.attributes, cells, origin,
                                                                    ctypes.byref(attributes_written))
        cursor_reset = kernel32.set_console_cursor_position(handle, origin)
        return bool(
            characters_cleared
            and attributes_cleared
            and cursor_reset
            and characters_written.value == cells
            and attributes_written.value == cells
        )
    finally:
        if owns_handle:
            kernel32.close_handle(handle)


_ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004


def _enable_virtual_terminal() -> bool:
    if os.name != "nt":
        return True

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE

    mode = wintypes.DWORD()

    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return False

    if mode.value & _ENABLE_VIRTUAL_TERMINAL_PROCESSING:
        return True

    return bool(
        kernel32.SetConsoleMode(
            handle,
            mode.value | _ENABLE_VIRTUAL_TERMINAL_PROCESSING,
        )
    )


def clear_screen() -> None:
    """Clear the visible terminal and its scrollback where supported."""
    if sys.stdout.isatty():
        if os.name != "nt" or _enable_virtual_terminal():
            sys.stdout.write("\x1b[2J\x1b[3J\x1b[H")
            sys.stdout.flush()
            return

    if os.name == "nt":
        try:
            _clear_native_console()
        except (AttributeError, OSError):
            pass


def rule(char: str = BOX_HORIZONTAL) -> str:
    return char * terminal_width()


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def visible_trim(text: str, max_length: int) -> str:
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return "." * max_length
    return text[: max_length - 3] + "..."


def relative_path(path: str | Path) -> str:
    path_obj = Path(path)
    try:
        resolved = path_obj.resolve()
        return str(Path(".") / resolved.relative_to(ROOT.resolve()))
    except (OSError, ValueError):
        return str(path_obj)


def shorten_command_argument(argument: str) -> str:
    root_text = str(ROOT.resolve())
    normalized = argument.replace("/", "\\")
    if normalized.lower().startswith(root_text.lower()):
        relative = normalized[len(root_text):].lstrip("\\/")
        return rf".\{relative}" if relative else "."

    # Omit an absolute executable path only when PATH resolves its basename to
    # the exact same file.  A local venv Python must remain explicit when a
    # different global Python is on PATH.
    path = Path(argument)
    if path.suffix.lower() == ".exe" and path.is_file():
        resolved_on_path = shutil.which(path.name)
        if resolved_on_path:
            try:
                if Path(resolved_on_path).resolve() == path.resolve():
                    return path.name
            except OSError:
                pass
    return argument


def display_command(command_line: list[str]) -> str:
    return subprocess.list2cmdline([shorten_command_argument(part) for part in command_line])


def shorten_output_paths(text: str) -> str:
    root = str(ROOT.resolve())
    for variant in (root, root.replace("\\", "/")):
        text = re.sub(re.escape(variant), ".", text, flags=re.IGNORECASE)
    return text


def wrap_console_line(text: str, width: int) -> list[str]:
    """Wrap console text without exceeding the available width."""
    if width <= 0 or not text:
        return [""]
    lines: list[str] = []
    while len(text) > width:
        split_at = text.rfind(" ", 0, width + 1)
        if split_at <= 0:
            split_at = width
        lines.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()
    return [*lines, text]


def wrap_prefixed_text(prefix: str, text: str, *, width: int, continuation: str | None = None) -> list[str]:
    """Wrap text with a stable continuation indentation and no ellipsis."""
    continuation_prefix = continuation if continuation is not None else " " * len(prefix)
    available = max(width - len(prefix), 1)
    continuation_available = max(width - len(continuation_prefix), 1)
    lines: list[str] = []
    remaining = text.strip()
    current_prefix = prefix
    current_width = available

    while len(remaining) > current_width:
        split_at = remaining.rfind(" ", 0, current_width + 1)
        if split_at <= 0:
            split_at = current_width
        lines.append(current_prefix + remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
        current_prefix = continuation_prefix
        current_width = continuation_available
    lines.append(current_prefix + remaining)
    return lines


def _box_line(content: str, *, width: int, tone: str) -> None:
    content_width = width - 4
    print(
        color(BOX_VERTICAL, tone) + " " +
        color(content.ljust(content_width), Color.WHITE) + " " +
        color(BOX_VERTICAL, tone)
    )


def header(title: str = "AIBrain", subtitle: str = "Neural Runtime Installer") -> None:
    width = terminal_width()
    inner = width - 2
    print()
    print(color(BOX_TOP_LEFT + BOX_HORIZONTAL * inner + BOX_TOP_RIGHT, Color.CYAN))
    _box_line(title.center(width - 4), width=width, tone=Color.CYAN)
    _box_line(subtitle.center(width - 4), width=width, tone=Color.CYAN)
    print(color(BOX_BOTTOM_LEFT + BOX_HORIZONTAL * inner + BOX_BOTTOM_RIGHT, Color.CYAN))
    print()


def section(title: str, number: int) -> None:
    print()
    print(color(f" {number:02d} ", Color.BOLD, Color.CYAN) + color(title, Color.BOLD, Color.WHITE))
    print(color(rule(), Color.GRAY))


def panel(title: str, rows: list[tuple[str, str]], *, subtitle: str | None = None, footer: str | None = None,
          tone: str = Color.CYAN) -> None:
    """Render a labeled summary panel shared by installer and build tools."""
    width = terminal_width()
    inner = width - 2
    label_width = max((len(label) for label, _ in rows), default=0)
    print()
    print(color(BOX_TOP_LEFT + BOX_HORIZONTAL * inner + BOX_TOP_RIGHT, tone))
    _box_line(title.center(width - 4), width=width, tone=tone)
    if subtitle:
        _box_line(subtitle.center(width - 4), width=width, tone=tone)
    print(color(BOX_MID_LEFT + BOX_HORIZONTAL * inner + BOX_MID_RIGHT, tone))
    for label, value in rows:
        prefix = f"  {label:<{label_width}}  "
        continuation = " " * len(prefix)
        for line in wrap_prefixed_text(prefix, value, width=width - 4, continuation=continuation):
            _box_line(line, width=width, tone=tone)
    if footer:
        print(color(BOX_MID_LEFT + BOX_HORIZONTAL * inner + BOX_MID_RIGHT, tone))
        _box_line(visible_trim(footer, width - 4), width=width, tone=tone)
    print(color(BOX_BOTTOM_LEFT + BOX_HORIZONTAL * inner + BOX_BOTTOM_RIGHT, tone))
    print()


def instruction_list(steps: list[tuple[str, str, str]], *, stream: TextIO | None = None) -> None:
    output = stream or sys.stdout
    print(file=output)
    for number, description, command_text in steps:
        print("  " + color(number, Color.CYAN, Color.BOLD) + " " + color(description, Color.GRAY) + "   " + color(
            command_text, Color.WHITE, Color.BOLD), file=output)
    print(file=output)


def info(message: str) -> None:
    print(f"  {color(BULLET, Color.CYAN)} {message}")


def success(message: str) -> None:
    print(f"  {color(CHECK, Color.GREEN, Color.BOLD)} {message}")


def warning(message: str) -> None:
    print(f"  {color('!', Color.YELLOW, Color.BOLD)} {message}")


def error(message: str) -> None:
    print(f"  {color(CROSS, Color.RED, Color.BOLD)} {message}", file=sys.stderr)


def detail(label: str, value: str) -> None:
    prefix = "     " + label.ljust(12)
    for line in wrap_prefixed_text(prefix, value, width=terminal_width()):
        print(color(line[:len(prefix)], Color.GRAY) + color(line[len(prefix):], Color.WHITE))


def status(label: str, message: str, tone: str = Color.CYAN) -> None:
    print(f"  {color(label.upper().ljust(8), Color.BOLD, tone)} {message}")


def command_preview(command_line: list[str]) -> None:
    prefix = " " * COMMAND_INDENT + PROMPT + " "
    continuation = " " * (COMMAND_INDENT + len(PROMPT) + 2)
    lines = wrap_prefixed_text(
        prefix,
        display_command(command_line),
        width=terminal_width(),
        continuation=continuation,
    )
    print()
    for index, line in enumerate(lines):
        if index == 0:
            print(color(line[:len(prefix) - 1], Color.MAGENTA, Color.BOLD) + " " + color(line[len(prefix):], Color.DIM,
                                                                                         Color.WHITE))
        else:
            print(color(line, Color.DIM, Color.WHITE))


class CommandOutputBox:
    """An indented, live-rendered container for subprocess output."""

    def __init__(self, *, indent: int = COMMAND_INDENT, live: bool | None = None) -> None:
        self.prefix = " " * indent
        self.inner = max(terminal_width() - indent, 20) - 2
        self.content_width = self.inner - 2
        self._is_open = False
        self._live = self._can_redraw_live() if live is None else live
        self._bottom_visible = False
        self._partial_rows = 0

    @staticmethod
    def _can_redraw_live() -> bool:
        """Use cursor control only for a real interactive terminal."""
        if not sys.stdout.isatty():
            return False
        return os.name != "nt" or _enable_virtual_terminal()

    def __enter__(self) -> CommandOutputBox:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def open(self) -> None:
        if self._is_open:
            return
        print(self.prefix + color(BOX_TOP_LEFT + BOX_HORIZONTAL * self.inner + BOX_TOP_RIGHT, Color.GRAY), flush=True)
        self._is_open = True
        if self._live:
            self._print_bottom()

    def _print_bottom(self) -> None:
        print(self.prefix + color(BOX_BOTTOM_LEFT + BOX_HORIZONTAL * self.inner + BOX_BOTTOM_RIGHT, Color.GRAY),
              flush=True)
        self._bottom_visible = True

    def _erase_live_rows(self, rows: int) -> None:
        """Remove the visible footer and optional partial rows before redrawing."""
        if not self._live or not self._bottom_visible:
            return
        for _ in range(rows + 1):
            sys.stdout.write("\x1b[1A\x1b[2K\r")
        sys.stdout.flush()
        self._bottom_visible = False

    def _rendered_lines(self, output: str) -> list[str]:
        rendered = shorten_output_paths(strip_ansi(output)).rstrip("\r\n")
        if not rendered:
            return []
        return [
            line
            for raw_line in rendered.splitlines()
            for line in wrap_console_line(raw_line, self.content_width)
        ]

    def _print_lines(self, lines: list[str]) -> None:
        for line in lines:
            print(
                self.prefix
                + color(BOX_VERTICAL, Color.GRAY)
                + " "
                + color(line, Color.GRAY)
                + " " * (self.content_width - len(line))
                + " "
                + color(BOX_VERTICAL, Color.GRAY),
                flush=True,
            )

    def write(self, output: str) -> None:
        """Append output immediately, preserving the framed presentation."""
        if not self._is_open:
            raise RuntimeError("The command output box must be opened before writing output")
        lines = self._rendered_lines(output)
        if not lines:
            return
        self._erase_live_rows(self._partial_rows)
        self._partial_rows = 0
        self._print_lines(lines)
        if self._live:
            self._print_bottom()

    def write_partial(self, output: str) -> None:
        """Redraw an unterminated subprocess line, including progress bars."""
        if not self._is_open:
            raise RuntimeError("The command output box must be opened before writing output")
        # Captured streams cannot erase their previous rows; delaying partial
        # output prevents duplicate fragments in IDE consoles and log capture.
        if not self._live:
            return
        lines = self._rendered_lines(output)
        self._erase_live_rows(self._partial_rows)
        self._partial_rows = len(lines)
        self._print_lines(lines)
        if self._live:
            self._print_bottom()

    def close(self) -> None:
        if not self._is_open:
            return
        if not self._bottom_visible:
            self._print_bottom()
        self._is_open = False


def command_output_box(output: str, *, indent: int = COMMAND_INDENT) -> None:
    """Render complete subprocess output in an indented grey box."""
    if not output.strip():
        return
    with CommandOutputBox(indent=indent) as box:
        box.write(output)


def command(command_line: list[str]) -> None:
    command_preview(command_line)
