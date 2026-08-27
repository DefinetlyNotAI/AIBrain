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


def enable_windows_utf8_console() -> None:
    """Configure Windows streams and an interactive console for UTF-8 output."""
    if os.name != "nt":
        return

    try:
        kernel32 = __import__("ctypes").windll.kernel32
        if sys.stdout.isatty():
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


enable_windows_utf8_console()


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

            class _Coord(ctypes.Structure):
                _fields_ = [("x", ctypes.c_short), ("y", ctypes.c_short)]

            class _SmallRect(ctypes.Structure):
                _fields_ = [
                    ("left", ctypes.c_short),
                    ("top", ctypes.c_short),
                    ("right", ctypes.c_short),
                    ("bottom", ctypes.c_short),
                ]

            class _ConsoleScreenBufferInfo(ctypes.Structure):
                _fields_ = [
                    ("size", _Coord),
                    ("cursor", _Coord),
                    ("attributes", ctypes.c_ushort),
                    ("window", _SmallRect),
                    ("maximum_window_size", _Coord),
                ]

            info = _ConsoleScreenBufferInfo()
            handle = ctypes.windll.kernel32.GetStdHandle(-11)
            if handle and ctypes.windll.kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
                width = info.window.right - info.window.left + 1
        except (AttributeError, OSError):
            pass
    return max(width, MIN_WIDTH)


def rule(char: str = "─") -> str:
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


def display_command(command_line: list[str]) -> str:
    shortened = [shorten_command_argument(part) for part in command_line]
    return subprocess.list2cmdline(shortened)


def shorten_output_paths(text: str) -> str:
    """Shorten project-local absolute paths inside subprocess output."""
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
    lines.append(text)
    return lines


def header(title: str = "AIBrain", subtitle: str = "Neural Runtime Installer") -> None:
    width = terminal_width()
    print()
    print(color("╭" + "─" * (width - 2) + "╮", Color.CYAN))
    print(
        color("│", Color.CYAN)
        + color(f" {title} ".center(width - 2), Color.BOLD, Color.WHITE)
        + color("│", Color.CYAN)
    )
    print(
        color("│", Color.CYAN)
        + color(subtitle.center(width - 2), Color.DIM, Color.CYAN)
        + color("│", Color.CYAN)
    )
    print(color("╰" + "─" * (width - 2) + "╯", Color.CYAN))
    print()


def section(title: str, number: int) -> None:
    print()
    print(
        color(f" {number:02d} ", Color.BOLD, Color.CYAN)
        + color(title, Color.BOLD, Color.WHITE)
    )
    print(color(rule(), Color.GRAY))


def panel(
    title: str,
    rows: list[tuple[str, str]],
    *,
    subtitle: str | None = None,
    footer: str | None = None,
    tone: str = Color.CYAN,
) -> None:
    """Render a labelled summary panel shared by installer and build tools."""
    width = terminal_width()
    inner_width = width - 2
    label_width = max((len(label) for label, _ in rows), default=0)

    print()
    print(color("╭" + "─" * inner_width + "╮", tone))
    print(
        color("│", tone)
        + color(f" {title} ".center(inner_width), Color.BOLD, tone)
        + color("│", tone)
    )
    if subtitle:
        print(
            color("│", tone)
            + color(subtitle.center(inner_width), Color.DIM, tone)
            + color("│", tone)
        )
    print(color("├" + "─" * inner_width + "┤", tone))

    for label, value in rows:
        content = visible_trim(f"  {label:<{label_width}}  {value}", inner_width - 2)
        print(
            color("│", tone)
            + " "
            + color(content.ljust(inner_width - 2), Color.WHITE)
            + " "
            + color("│", tone)
        )

    if footer:
        print(color("├" + "─" * inner_width + "┤", tone))
        content = visible_trim(footer, inner_width - 2)
        print(
            color("│", tone)
            + " "
            + color(content.ljust(inner_width - 2), Color.BOLD, Color.WHITE)
            + " "
            + color("│", tone)
        )

    print(color("╰" + "─" * inner_width + "╯", tone))
    print()


def instruction_list(
    steps: list[tuple[str, str, str]], *, stream: TextIO | None = None
) -> None:
    """Render numbered setup instructions with consistent CLI styling."""
    output = stream or sys.stdout
    print(file=output)
    for number, description, command_text in steps:
        print(
            "  "
            + color(number, Color.CYAN, Color.BOLD)
            + " "
            + color(description, Color.GRAY)
            + "   "
            + color(command_text, Color.WHITE, Color.BOLD),
            file=output,
        )
    print(file=output)


def info(message: str) -> None:
    print(f"  {color('●', Color.CYAN)} {message}")


def success(message: str) -> None:
    print(f"  {color('✓', Color.GREEN, Color.BOLD)} {message}")


def warning(message: str) -> None:
    print(f"  {color('!', Color.YELLOW, Color.BOLD)} {message}")


def error(message: str) -> None:
    print(f"  {color('✗', Color.RED, Color.BOLD)} {message}", file=sys.stderr)


def detail(label: str, value: str) -> None:
    value = visible_trim(value, max(terminal_width() - 20, 10))
    print(f"     {color(label.ljust(12), Color.GRAY)}{color(value, Color.WHITE)}")


def status(label: str, message: str, tone: str = Color.CYAN) -> None:
    """Print a compact shared status row for builder commands."""
    print(f"  {color(label.upper().ljust(8), Color.BOLD, tone)} {message}")


def command_preview(command_line: list[str]) -> None:
    """Print a command line without a surrounding output box."""
    command_text = visible_trim(
        display_command(command_line), terminal_width() - COMMAND_INDENT - 2
    )
    print(
        "\n"
        + " " * COMMAND_INDENT
        + color("›", Color.MAGENTA, Color.BOLD)
        + " "
        + color(command_text, Color.DIM, Color.WHITE)
    )


def command_output_box(output: str, *, indent: int = COMMAND_INDENT) -> None:
    """Render subprocess output in an indented grey box."""
    output = shorten_output_paths(strip_ansi(output)).rstrip()
    if not output:
        return
    prefix = " " * indent
    inner_width = max(terminal_width() - indent, 20) - 2
    content_width = inner_width - 2
    rendered_lines = [
        line
        for raw_line in output.splitlines()
        for line in wrap_console_line(raw_line, content_width)
    ]
    print(prefix + color("╭" + "─" * inner_width + "╮", Color.GRAY))
    for line in rendered_lines:
        line = visible_trim(line, content_width)
        print(
            prefix
            + color("│", Color.GRAY)
            + " "
            + color(line, Color.GRAY)
            + " " * (content_width - len(line))
            + " "
            + color("│", Color.GRAY)
        )
    print(prefix + color("╰" + "─" * inner_width + "╯", Color.GRAY))


def command(command_line: list[str]) -> None:
    command_preview(command_line)
