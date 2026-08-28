"""The shared terminal presentation library for all AIBrain CLI commands."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TextIO


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WIDTH = 82
MIN_WIDTH = 60
COMMAND_INDENT = 2
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Keep every character emitted by the shared CLI renderer ASCII.  Windows
# terminals disagree on how redirected output and code-page changes interact;
# ASCII avoids that entire class of corrupted glyphs while preserving layout.
BOX_HORIZONTAL = "-"
BOX_VERTICAL = "|"
BOX_TOP_LEFT = "+"
BOX_TOP_RIGHT = "+"
BOX_BOTTOM_LEFT = "+"
BOX_BOTTOM_RIGHT = "+"
BOX_MID_LEFT = "+"
BOX_MID_RIGHT = "+"
BULLET = "*"
CHECK = "OK"
CROSS = "X"
PROMPT = ">"


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
            import ctypes

            class Coord(ctypes.Structure):
                _fields_ = [("x", ctypes.c_short), ("y", ctypes.c_short)]

            class SmallRect(ctypes.Structure):
                _fields_ = [("left", ctypes.c_short), ("top", ctypes.c_short), ("right", ctypes.c_short), ("bottom", ctypes.c_short)]

            class ConsoleScreenBufferInfo(ctypes.Structure):
                _fields_ = [("size", Coord), ("cursor", Coord), ("attributes", ctypes.c_ushort), ("window", SmallRect), ("maximum_window_size", Coord)]

            info = ConsoleScreenBufferInfo()
            handle = ctypes.windll.kernel32.GetStdHandle(-11)
            if handle and ctypes.windll.kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
                width = info.window.right - info.window.left + 1
        except (AttributeError, OSError):
            pass
    return max(width, MIN_WIDTH)


def clear_screen() -> None:
    """Clear an interactive terminal without invoking cmd.exe or a shell command."""
    if not sys.stdout.isatty():
        return
    if os.name == "nt":
        try:
            import ctypes

            class Coord(ctypes.Structure):
                _fields_ = [("x", ctypes.c_short), ("y", ctypes.c_short)]

            class SmallRect(ctypes.Structure):
                _fields_ = [("left", ctypes.c_short), ("top", ctypes.c_short), ("right", ctypes.c_short), ("bottom", ctypes.c_short)]

            class ConsoleScreenBufferInfo(ctypes.Structure):
                _fields_ = [("size", Coord), ("cursor", Coord), ("attributes", ctypes.c_ushort), ("window", SmallRect), ("maximum_window_size", Coord)]

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            info = ConsoleScreenBufferInfo()
            if handle and kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
                cells = info.size.x * info.size.y
                written = ctypes.c_ulong()
                origin = Coord(0, 0)
                kernel32.FillConsoleOutputCharacterW(handle, " ", cells, origin, ctypes.byref(written))
                kernel32.FillConsoleOutputAttribute(handle, info.attributes, cells, origin, ctypes.byref(written))
                kernel32.SetConsoleCursorPosition(handle, origin)
                return
        except (AttributeError, OSError):
            pass
    print("\x1b[2J\x1b[H", end="", flush=True)


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


def _box_line(content: str, *, width: int, tone: str) -> None:
    print(color(BOX_VERTICAL, tone) + " " + color(content.ljust(width - 2), Color.WHITE) + " " + color(BOX_VERTICAL, tone))


def header(title: str = "AIBrain", subtitle: str = "Neural Runtime Installer") -> None:
    width = terminal_width()
    inner = width - 2
    print()
    print(color(BOX_TOP_LEFT + BOX_HORIZONTAL * inner + BOX_TOP_RIGHT, Color.CYAN))
    _box_line(f" {title} ".center(inner), width=width, tone=Color.CYAN)
    _box_line(subtitle.center(inner), width=width, tone=Color.CYAN)
    print(color(BOX_BOTTOM_LEFT + BOX_HORIZONTAL * inner + BOX_BOTTOM_RIGHT, Color.CYAN))
    print()


def section(title: str, number: int) -> None:
    print()
    print(color(f" {number:02d} ", Color.BOLD, Color.CYAN) + color(title, Color.BOLD, Color.WHITE))
    print(color(rule(), Color.GRAY))


def panel(title: str, rows: list[tuple[str, str]], *, subtitle: str | None = None, footer: str | None = None, tone: str = Color.CYAN) -> None:
    """Render a labelled summary panel shared by installer and build tools."""
    width = terminal_width()
    inner = width - 2
    label_width = max((len(label) for label, _ in rows), default=0)
    print()
    print(color(BOX_TOP_LEFT + BOX_HORIZONTAL * inner + BOX_TOP_RIGHT, tone))
    _box_line(f" {title} ".center(inner), width=width, tone=tone)
    if subtitle:
        _box_line(subtitle.center(inner), width=width, tone=tone)
    print(color(BOX_MID_LEFT + BOX_HORIZONTAL * inner + BOX_MID_RIGHT, tone))
    for label, value in rows:
        _box_line(visible_trim(f"  {label:<{label_width}}  {value}", inner), width=width, tone=tone)
    if footer:
        print(color(BOX_MID_LEFT + BOX_HORIZONTAL * inner + BOX_MID_RIGHT, tone))
        _box_line(visible_trim(footer, inner), width=width, tone=tone)
    print(color(BOX_BOTTOM_LEFT + BOX_HORIZONTAL * inner + BOX_BOTTOM_RIGHT, tone))
    print()


def instruction_list(steps: list[tuple[str, str, str]], *, stream: TextIO | None = None) -> None:
    output = stream or sys.stdout
    print(file=output)
    for number, description, command_text in steps:
        print("  " + color(number, Color.CYAN, Color.BOLD) + " " + color(description, Color.GRAY) + "   " + color(command_text, Color.WHITE, Color.BOLD), file=output)
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
    print(f"     {color(label.ljust(12), Color.GRAY)}{color(visible_trim(value, max(terminal_width() - 20, 10)), Color.WHITE)}")


def status(label: str, message: str, tone: str = Color.CYAN) -> None:
    print(f"  {color(label.upper().ljust(8), Color.BOLD, tone)} {message}")


def command_preview(command_line: list[str]) -> None:
    command_text = visible_trim(display_command(command_line), terminal_width() - COMMAND_INDENT - 2)
    print("\n" + " " * COMMAND_INDENT + color(PROMPT, Color.MAGENTA, Color.BOLD) + " " + color(command_text, Color.DIM, Color.WHITE))


def command_output_box(output: str, *, indent: int = COMMAND_INDENT) -> None:
    """Render complete subprocess output in an indented grey box."""
    output = shorten_output_paths(strip_ansi(output)).rstrip()
    if not output:
        return
    prefix = " " * indent
    inner = max(terminal_width() - indent, 20) - 2
    content_width = inner - 2
    lines = [line for raw_line in output.splitlines() for line in wrap_console_line(raw_line, content_width)]
    print(prefix + color(BOX_TOP_LEFT + BOX_HORIZONTAL * inner + BOX_TOP_RIGHT, Color.GRAY))
    for line in lines:
        print(prefix + color(BOX_VERTICAL, Color.GRAY) + " " + color(visible_trim(line, content_width), Color.GRAY) + " " * (content_width - len(visible_trim(line, content_width))) + " " + color(BOX_VERTICAL, Color.GRAY))
    print(prefix + color(BOX_BOTTOM_LEFT + BOX_HORIZONTAL * inner + BOX_BOTTOM_RIGHT, Color.GRAY))


def command(command_line: list[str]) -> None:
    command_preview(command_line)
