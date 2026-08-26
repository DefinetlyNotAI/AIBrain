"""Shared console presentation used by installation and build commands."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


def color(text: str, *styles: str) -> str:
    return "".join(styles) + text + Color.RESET


def enable_colour() -> None:
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
        except (AttributeError, OSError):
            pass


def terminal_width() -> int:
    return min(max(shutil.get_terminal_size((82, 24)).columns, 60), 110)


def header(title: str, subtitle: str) -> None:
    width = terminal_width()
    print()
    print(color("╭" + "─" * (width - 2) + "╮", Color.CYAN))
    print(color("│", Color.CYAN) + color(f" {title} ".center(width - 2), Color.BOLD, Color.WHITE) + color("│", Color.CYAN))
    print(color("│", Color.CYAN) + color(subtitle.center(width - 2), Color.DIM, Color.CYAN) + color("│", Color.CYAN))
    print(color("╰" + "─" * (width - 2) + "╯", Color.CYAN))
    print()


def section(title: str, number: int) -> None:
    print()
    print(color(f" {number:02d} ", Color.BOLD, Color.CYAN) + color(title, Color.BOLD, Color.WHITE))
    print(color("─" * terminal_width(), Color.GRAY))


def status(label: str, message: str, tone: str = Color.CYAN) -> None:
    print(f"  {color(label.upper().ljust(8), Color.BOLD, tone)} {message}")


def info(message: str) -> None:
    status("info", message)


def success(message: str) -> None:
    status("ok", message, Color.GREEN)


def warning(message: str) -> None:
    status("warning", message, Color.YELLOW)


def error(message: str) -> None:
    print(f"  {color('error'.upper().ljust(8), Color.BOLD, Color.RED)} {message}", file=sys.stderr)


def detail(label: str, value: str) -> None:
    print(f"     {color(label.ljust(12), Color.GRAY)}{color(value, Color.WHITE)}")


def command(command: list[str]) -> None:
    print("  " + color("› " + subprocess.list2cmdline(command), Color.DIM, Color.WHITE))
