"""Shared AIBrain console presentation components."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_ASCII_GLYPHS = str.maketrans(
    {"╭": "+", "╮": "+", "╰": "+", "╯": "+", "├": "+", "┤": "+", "│": "|", "─": "-", "●": "*", "✓": "OK", "✗": "X",
     "›": ">"})


def _console_text(text: str) -> str:
    """Avoid UTF-8 mojibake in legacy CMD/PowerShell code pages."""
    encoding = sys.stdout.encoding or "ascii"
    try:
        text.encode(encoding)
    except UnicodeEncodeError:
        return text.translate(_ASCII_GLYPHS)
    return text


DEFAULT_WIDTH = 82
MIN_WIDTH = 60
MAX_WIDTH = 110
COMMAND_INDENT = 2

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

if os.name == "nt":
    os.system("")


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
    return "".join(styles) + _console_text(text) + Color.RESET


def terminal_width() -> int:
    """Use the visible CMD/terminal window width rather than a fixed artwork cap."""
    width = shutil.get_terminal_size((DEFAULT_WIDTH, 24)).columns
    if os.name == "nt":
        try:
            import ctypes

            class _Coord(ctypes.Structure):
                _fields_ = [("x", ctypes.c_short), ("y", ctypes.c_short)]

            class _SmallRect(ctypes.Structure):
                _fields_ = [("left", ctypes.c_short), ("top", ctypes.c_short), ("right", ctypes.c_short),
                            ("bottom", ctypes.c_short)]

            class _ConsoleScreenBufferInfo(ctypes.Structure):
                _fields_ = [("size", _Coord), ("cursor", _Coord), ("attributes", ctypes.c_ushort),
                            ("window", _SmallRect), ("maximum_window_size", _Coord)]

            info = _ConsoleScreenBufferInfo()
            handle = ctypes.windll.kernel32.GetStdHandle(-11)
            if handle and ctypes.windll.kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
                width = info.window.right - info.window.left + 1
        except (AttributeError, OSError):
            pass
    return max(width, MIN_WIDTH)


def rule(char: str = "â”€") -> str:
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
    except OSError:
        return str(path_obj)

    try:
        relative = resolved.relative_to(ROOT.resolve())
        return str(Path(".") / relative)
    except ValueError:
        return str(resolved)


def shorten_command_argument(argument: str) -> str:
    root_text = str(ROOT.resolve())
    normalized = argument.replace("/", "\\")

    if normalized.lower().startswith(root_text.lower()):
        relative = normalized[len(root_text):].lstrip("\\/")
        return rf".\{relative}" if relative else "."

    return argument


def display_command(command: list[str]) -> str:
    shortened = [shorten_command_argument(part) for part in command]
    return subprocess.list2cmdline(shortened)


def shorten_output_paths(text: str) -> str:
    """Shorten project-local absolute paths inside subprocess output."""
    root = str(ROOT.resolve())

    variants = (
        root,
        root.replace("\\", "/"),
    )

    for variant in variants:
        text = re.sub(
            re.escape(variant),
            ".",
            text,
            flags=re.IGNORECASE,
        )

    return text


def wrap_console_line(text: str, width: int) -> list[str]:
    """Wrap console text without exceeding the available width."""
    if width <= 0:
        return [""]

    if not text:
        return [""]

    lines: list[str] = []

    while len(text) > width:
        split_at = text.rfind(" ", 0, width + 1)

        if split_at <= 0:
            split_at = width

        lines.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()

    lines.append(text)

    return lines


def header(title: str = "AIBrain", subtitle: str = "Neural Runtime Installer") -> None:
    width = terminal_width()

    title = f" {title} "
    subtitle = subtitle

    print()
    print(color("â•­" + "â”€" * (width - 2) + "â•®", Color.CYAN))
    print(
        color("â”‚", Color.CYAN)
        + color(title.center(width - 2), Color.BOLD, Color.WHITE)
        + color("â”‚", Color.CYAN)
    )
    print(
        color("â”‚", Color.CYAN)
        + color(subtitle.center(width - 2), Color.DIM, Color.CYAN)
        + color("â”‚", Color.CYAN)
    )
    print(color("â•°" + "â”€" * (width - 2) + "â•¯", Color.CYAN))
    print()


def section(title: str, number: int) -> None:
    print()
    print(
        color(f" {number:02d} ", Color.BOLD, Color.CYAN)
        + color(title, Color.BOLD, Color.WHITE)
    )
    print(color(rule(), Color.GRAY))


def info(message: str) -> None:
    print(f"  {color('â—', Color.CYAN)} {message}")


def success(message: str) -> None:
    print(f"  {color('âœ“', Color.GREEN, Color.BOLD)} {message}")


def warning(message: str) -> None:
    print(f"  {color('!', Color.YELLOW, Color.BOLD)} {message}")


def error(message: str) -> None:
    print(
        f"  {color('âœ—', Color.RED, Color.BOLD)} {message}",
        file=sys.stderr,
    )


def detail(label: str, value: str) -> None:
    max_value_width = max(terminal_width() - 20, 10)
    value = visible_trim(value, max_value_width)

    print(
        f"     {color(label.ljust(12), Color.GRAY)}"
        f"{color(value, Color.WHITE)}"
    )


def status(label: str, message: str, tone: str = Color.CYAN) -> None:
    """Compact shared status row for builder commands."""
    print(f"  {color(label.upper().ljust(8), Color.BOLD, tone)} {message}")


def command_preview(command: list[str]) -> None:
    """Print only the command line itself, without a surrounding box."""
    command_text = display_command(command)

    available = terminal_width() - COMMAND_INDENT - 2
    command_text = visible_trim(command_text, available)

    print()
    print(
        (" " * COMMAND_INDENT)
        + color("â€º", Color.MAGENTA, Color.BOLD)
        + " "
        + color(command_text, Color.DIM, Color.WHITE)
    )


def command_output_box(
        output: str,
        *,
        indent: int = COMMAND_INDENT,
) -> None:
    """Render subprocess output in an indented grey box."""
    output = strip_ansi(output)
    output = shorten_output_paths(output)
    output = output.rstrip()

    if not output:
        return

    prefix = " " * indent

    available_width = max(terminal_width() - indent, 20)
    inner_width = available_width - 2
    content_width = inner_width - 2

    rendered_lines: list[str] = []

    for raw_line in output.splitlines():
        rendered_lines.extend(
            wrap_console_line(raw_line, content_width)
        )

    print(
        prefix
        + color(
            "â•­" + "â”€" * inner_width + "â•®",
            Color.GRAY,
        )
    )

    for line in rendered_lines:
        line = visible_trim(line, content_width)

        padding = content_width - len(line)

        print(
            prefix
            + color("â”‚", Color.GRAY)
            + " "
            + color(line, Color.GRAY)
            + (" " * max(padding, 0))
            + " "
            + color("â”‚", Color.GRAY)
        )

    print(
        prefix
        + color(
            "â•°" + "â”€" * inner_width + "â•¯",
            Color.GRAY,
        )
    )


def command(command_line: list[str]) -> None:
    command_preview(command_line)
