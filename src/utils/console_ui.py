"""The shared terminal presentation library for all AIBrain CLI commands."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Self, TextIO

from ..app.ctypes_helper import windows_console

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WIDTH = 82
MIN_WIDTH = 60
RIGHT_EDGE_MARGIN = 4
COMMAND_INDENT = 2
MAX_PREVIEW_FLAGS = 8
ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|][^\x07]*(?:\x07|\x1b\\))")


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


def _normalise_prompt_answer(answer: str) -> str:
    """Make case, spacing, and punctuation irrelevant for menu answers."""
    return re.sub(r"[^a-z0-9]+", "", answer.casefold())


def _supports_prompt_redraw() -> bool:
    """Return whether ANSI can safely replace an input line after Enter."""
    if not sys.stdout.isatty():
        return False
    return os.name != "nt" or _enable_virtual_terminal()


def _render_prompt_answer(prompt: str, answer: str, *styles: str) -> None:
    """Show a normalized answer on the original prompt line where possible."""
    rendered = prompt + color(answer, *styles)
    if _supports_prompt_redraw():
        sys.stdout.write("\x1b[1A\r\x1b[2K" + rendered + "\n")
        sys.stdout.flush()
    else:
        print(color(answer, *styles))


def _choice_label(key: str, label: str) -> str:
    """Highlight a mnemonic key without assuming labels have one character."""
    if label.casefold().startswith(key.casefold()):
        return f"[{key.upper()}]{label[len(key):]}"
    return f"[{key.upper()}] {label}"


def ask_choice(
        question: str,
        choices: Mapping[str, str],
        *,
        default: str | None = None,
        show_choices: bool = True,
) -> str | None:
    """Read a menu answer, accepting a key or a case-insensitive full label."""
    if default is not None and default not in choices:
        raise ValueError("Prompt default must be one of the available choices")
    if not sys.stdin.isatty():
        return default

    aliases = {
        _normalise_prompt_answer(alias): key
        for key, label in choices.items()
        for alias in (key, label)
    }
    rendered_choices = " / ".join(
        _choice_label(key, label) for key, label in choices.items()
    )
    suffix = f" ({rendered_choices})" if show_choices else ""
    if default is not None:
        suffix += f" [{default.upper()}]"
    prompt = f"  {question}{suffix}: "

    while True:
        try:
            answer = input(prompt).strip()
        except EOFError:
            return default

        if not answer and default is not None:
            _render_prompt_answer(prompt, choices[default], Color.GRAY)
            return default

        selected = aliases.get(_normalise_prompt_answer(answer))
        if selected is not None:
            _render_prompt_answer(prompt, choices[selected], Color.WHITE)
            return selected

        warning(f"Choose one of: {', '.join(choices)}")


def ask_boolean(question: str, *, default: bool) -> bool:
    """Read a yes/no answer with common aliases and a colored default value."""
    if not sys.stdin.isatty():
        return default

    prompt = (
        f"  {question} ("
        f"{color('[Y]es', Color.GREEN)} / {color('[N]o', Color.RED)}"
        f") [{('Y' if default else 'N')}]: "
    )
    true_answers = {"true", "yes", "y", "1"}
    false_answers = {"false", "no", "n", "0"}

    while True:
        try:
            answer = input(prompt).strip()
        except EOFError:
            return default

        if not answer:
            _render_prompt_answer(
                prompt,
                "Yes" if default else "No",
                Color.GREEN if default else Color.RED,
            )
            return default

        normalised = _normalise_prompt_answer(answer)
        if normalised in true_answers:
            _render_prompt_answer(prompt, "Yes", Color.WHITE)
            return True
        if normalised in false_answers:
            _render_prompt_answer(prompt, "No", Color.WHITE)
            return False
        warning("Enter yes or no (yes, y, 1 / no, n, 0).")


def terminal_width() -> int:
    """Return the visible terminal width with a usable minimum."""
    width = shutil.get_terminal_size(
        (DEFAULT_WIDTH, 24)
    ).columns

    if os.name == "nt":
        try:
            native_width = windows_console.console_width()

            if native_width is not None:
                width = native_width

        except OSError:
            pass

    return max(
        width - RIGHT_EDGE_MARGIN,
        MIN_WIDTH,
    )


def _enable_virtual_terminal() -> bool:
    """Enable ANSI terminal processing where supported."""
    if os.name != "nt":
        return True

    try:
        return windows_console.enable_virtual_terminal_output()
    except OSError:
        return False


def clear_screen() -> None:
    """Clear the visible terminal and its scrollback where supported."""
    if sys.stdout.isatty() and (os.name != "nt" or _enable_virtual_terminal()):
        sys.stdout.write("\x1b[2J\x1b[3J\x1b[H")
        sys.stdout.flush()
        return

    if os.name == "nt":
        try:
            windows_console.clear_output()
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
        return str(Path(resolved.relative_to(ROOT.resolve())))
    except (OSError, ValueError):
        return str(path_obj)


def shorten_command_argument(argument: str) -> str:
    root_text = str(ROOT.resolve())
    normalized = argument.replace("/", "\\")
    if normalized.lower() == root_text.lower() or normalized.lower().startswith(
            root_text.lower() + "\\"
    ):
        relative = normalized[len(root_text):].lstrip("\\/")
        return rf".\{relative}" if relative else "."

    # Omit an absolute executable path only when PATH resolves its basename to
    # the exact same file. Keep an explicit relative venv path visible even
    # when that same interpreter is also available on PATH.
    path = Path(argument)
    if path.is_absolute() and path.suffix.lower() == ".exe" and path.is_file():
        resolved_on_path = shutil.which(path.name)
        if resolved_on_path:
            try:
                if Path(resolved_on_path).resolve() == path.resolve():
                    return path.name
            except OSError:
                pass
    return argument


def display_command(command_line: list[str]) -> str:
    """Render a command preview that can be pasted safely into PowerShell."""
    arguments = [
        shorten_output_paths(shorten_command_argument(part)) for part in command_line
    ]
    rendered = [_powershell_quote(argument) for argument in arguments]
    if rendered and rendered[0] != arguments[0]:
        return "& " + " ".join(rendered)
    return " ".join(rendered)


def _powershell_quote(argument: str) -> str:
    """Quote one PowerShell argument only when shell syntax could reinterpret it."""
    if not argument or re.search(r"[\s'\"`$&|;<>()[\]{}*,#?~@]", argument):
        return "'" + argument.replace("'", "''") + "'"
    return argument


def shorten_output_paths(text: str) -> str:
    root = str(ROOT.resolve())
    for variant in (root, root.replace("\\", "/")):
        text = re.sub(
            re.escape(variant) + r"(?=[\\/]|$|[\"'])", ".", text, flags=re.IGNORECASE
        )
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


def wrap_prefixed_text(
        prefix: str, text: str, *, width: int, continuation: str | None = None
) -> list[str]:
    """Wrap each logical line with aligned gutters and its original indentation."""
    continuation_prefix = (
        continuation if continuation is not None else " " * len(prefix)
    )
    lines: list[str] = []
    for index, raw_line in enumerate(
            strip_ansi(text).expandtabs(4).splitlines() or [""]
    ):
        indentation = raw_line[: len(raw_line) - len(raw_line.lstrip())]
        remaining = raw_line.lstrip().rstrip()
        current_prefix = (prefix if index == 0 else continuation_prefix) + indentation
        while len(remaining) > max(width - len(current_prefix), 1):
            available = max(width - len(current_prefix), 1)
            split_at = remaining.rfind(" ", 0, available + 1)
            if split_at <= 0:
                split_at = available
            lines.append(current_prefix + remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip()
            current_prefix = continuation_prefix + indentation
        lines.append(current_prefix + remaining)
    return lines


def console_message_lines(prefix: str, message: str) -> list[str]:
    """Apply project-relative paths and terminal-width wrapping to status text."""
    return wrap_prefixed_text(
        prefix, shorten_output_paths(str(message)), width=terminal_width()
    )


def _box_line(content: str, *, width: int, tone: str) -> None:
    content_width = width - 4
    print(
        color(BOX_VERTICAL, tone)
        + " "
        + color(content.ljust(content_width), Color.WHITE)
        + " "
        + color(BOX_VERTICAL, tone)
    )


def header(title: str = "AIBrain", subtitle: str = "Neural Runtime Installer") -> None:
    width = terminal_width()
    inner = width - 2
    print()
    print(color(BOX_TOP_LEFT + BOX_HORIZONTAL * inner + BOX_TOP_RIGHT, Color.CYAN))
    _box_line(title.center(width - 4), width=width, tone=Color.CYAN)
    _box_line(subtitle.center(width - 4), width=width, tone=Color.CYAN)
    print(
        color(BOX_BOTTOM_LEFT + BOX_HORIZONTAL * inner + BOX_BOTTOM_RIGHT, Color.CYAN)
    )
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
        for line in wrap_prefixed_text(
                prefix,
                shorten_output_paths(value),
                width=width - 4,
                continuation=continuation,
        ):
            _box_line(line, width=width, tone=tone)
    if footer:
        print(color(BOX_MID_LEFT + BOX_HORIZONTAL * inner + BOX_MID_RIGHT, tone))
        for line in wrap_prefixed_text(
                "", shorten_output_paths(footer), width=width - 4
        ):
            _box_line(line, width=width, tone=tone)
    print(color(BOX_BOTTOM_LEFT + BOX_HORIZONTAL * inner + BOX_BOTTOM_RIGHT, tone))
    print()


def instruction_list(
        steps: list[tuple[str, str, str]], *, stream: TextIO | None = None
) -> None:
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
    _print_status_message(BULLET, message, Color.CYAN)


def success(message: str) -> None:
    _print_status_message(CHECK, message, Color.GREEN)


def report_gui_closed(application: str) -> None:
    """Make a normal desktop-window close explicit in its launch console."""
    success(f"User closed {application}.")


def report_keyboard_interrupt(application: str) -> None:
    """Render the shared clean cancellation message used by CLI entry points."""
    print(flush=True)
    error(f"User ended {application} with KeyboardInterrupt.")


def warning(message: str) -> None:
    _print_status_message("!", message, Color.YELLOW)


def error(message: str) -> None:
    """Render an error without collapsing a multiline traceback into one row."""
    _print_status_message(CROSS, message, Color.RED, stream=sys.stderr)


def _print_status_message(
        marker: str, message: str, tone: str, *, stream: TextIO | None = None
) -> None:
    prefix = f"  {marker} "
    lines = console_message_lines(prefix, message)
    print(
        color(prefix, tone, Color.BOLD) + lines[0][len(prefix):],
        file=stream or sys.stdout,
    )
    for line in lines[1:]:
        print(line, file=stream or sys.stdout)


def detail(label: str, value: str) -> None:
    prefix = "     " + label.ljust(12)
    for line in console_message_lines(prefix, value):
        print(
            color(line[: len(prefix)], Color.GRAY)
            + color(line[len(prefix):], Color.WHITE)
        )


def status(label: str, message: str, tone: str = Color.CYAN) -> None:
    _print_status_message(label.upper().ljust(8), message, tone)


def command_preview_parts(command_line: list[str]) -> tuple[list[str], int]:
    """Keep the executable/module/script and count its attached option tokens."""
    if not command_line:
        return [], 0
    end = 1
    executable = Path(command_line[0]).name.lower().removesuffix(".exe")
    if re.fullmatch(r"py|pythonw?(?:\d+(?:\.\d+)*)?", executable):
        while end < len(command_line):
            argument = command_line[end]
            end += 1
            if argument in {"-m", "-c"}:
                end = min(end + 1, len(command_line))
                break
            if argument in {"-W", "-X", "--check-hash-based-pycs"}:
                end = min(end + 1, len(command_line))
            elif not argument.startswith("-"):
                break
    # Retain subcommands such as "pip install" or "git diff".
    while end < len(command_line) and not command_line[end].startswith("-"):
        end += 1
    flag_count = 0
    for argument in command_line[end:]:
        if argument == "--":
            break
        if re.match(r"--?[A-Za-z]", argument):
            flag_count += 1
    if flag_count <= MAX_PREVIEW_FLAGS:
        return command_line, 0
    return command_line[:end], flag_count


def command_preview(command_line: list[str]) -> None:
    logging.getLogger("aibrain.command").info(
        "Command started: %s", subprocess.list2cmdline(command_line)
    )
    preview, flag_count = command_preview_parts(command_line)
    rendered = display_command(preview)
    if flag_count:
        rendered += f" ({flag_count} flags attached - Full command in log file)"
    prefix = " " * COMMAND_INDENT + PROMPT + " "
    continuation = " " * len(prefix)
    lines = wrap_prefixed_text(
        prefix,
        rendered,
        width=terminal_width(),
        continuation=continuation,
    )
    print()
    for index, line in enumerate(lines):
        if index == 0:
            print(
                color(line[: len(prefix) - 1], Color.MAGENTA, Color.BOLD)
                + " "
                + color(line[len(prefix):], Color.DIM, Color.WHITE)
            )
        else:
            print(color(line, Color.DIM, Color.WHITE))


class CommandOutputBox:
    """An indented, live-rendered container for subprocess output."""

    def __init__(
            self, *, indent: int = COMMAND_INDENT, live: bool | None = None
    ) -> None:
        self.prefix = " " * indent
        self.inner = max(terminal_width() - indent, 20) - 2
        self.content_width = self.inner - 2
        self._is_open = False
        self._has_output = False
        self._live = self._can_redraw_live() if live is None else live
        self._bottom_visible = False
        self._partial_rows = 0
        self._last_progress_at: float | None = None

    @staticmethod
    def _can_redraw_live() -> bool:
        """Use cursor control only for a real interactive terminal."""
        if not sys.stdout.isatty():
            return False
        return os.name != "nt" or _enable_virtual_terminal()

    @property
    def is_live(self) -> bool:
        """Whether transient status can be redrawn instead of appended."""
        return self._live

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def open(self) -> None:
        if self._is_open:
            return
        self._is_open = True
        self._has_output = False
        self._bottom_visible = False
        self._partial_rows = 0
        self._last_progress_at = None

    def _border_line(self, left: str, right: str) -> str:
        return (
                self.prefix
                + color(left + BOX_HORIZONTAL * self.inner + right, Color.GRAY)
                + "\n"
        )

    def _rendered_lines(self, output: str) -> list[str]:
        rendered = shorten_output_paths(strip_ansi(output)).rstrip("\r\n")
        if not rendered.strip():
            return []
        return [
            line
            for raw_line in rendered.splitlines()
            for line in wrap_prefixed_text("", raw_line, width=self.content_width)
        ]

    def _replace_output(self, lines: list[str], *, partial: bool) -> None:
        """Replace transient rows and move the footer in one flushed frame."""
        frame = ""
        if not self._has_output:
            frame = self._border_line(BOX_TOP_LEFT, BOX_TOP_RIGHT)
        elif self._live:
            rows = self._partial_rows + int(self._bottom_visible)
            frame = "\x1b[1A\x1b[2K\r" * rows
        frame += "".join(
            self.prefix
            + color(BOX_VERTICAL, Color.GRAY)
            + " "
            + color(line, Color.GRAY)
            + " " * (self.content_width - len(line))
            + " "
            + color(BOX_VERTICAL, Color.GRAY)
            + "\n"
            for line in lines
        )
        if self._live:
            frame += self._border_line(BOX_BOTTOM_LEFT, BOX_BOTTOM_RIGHT)
        # Do not flush between erasing the old footer and drawing its replacement.
        sys.stdout.write(frame)
        sys.stdout.flush()
        self._has_output = True
        self._bottom_visible = self._live
        self._partial_rows = len(lines) if partial else 0

    def write(self, output: str) -> None:
        """Append output immediately, preserving the framed presentation."""
        if not self._is_open:
            raise RuntimeError(
                "The command output box must be opened before writing output"
            )
        lines = self._rendered_lines(output)
        if not lines:
            return
        self._replace_output(lines, partial=False)
        self._last_progress_at = None

    def write_progress(self, output: str, *, interval: float = 3.0) -> None:
        """Show current work in place, or periodically in a captured console."""
        if not self._is_open:
            raise RuntimeError(
                "The command output box must be opened before writing output"
            )
        if self._live:
            self.write_partial(output)
            return
        now = time.monotonic()
        if self._last_progress_at is None or now - self._last_progress_at >= interval:
            self.write(output)
            self._last_progress_at = now

    def write_partial(self, output: str) -> None:
        """Redraw an unterminated subprocess line, including progress bars."""
        if not self._is_open:
            raise RuntimeError(
                "The command output box must be opened before writing output"
            )
        # Captured streams cannot erase their previous rows; delaying partial
        # output prevents duplicate fragments in IDE consoles and log capture.
        if not self._live:
            return
        lines = self._rendered_lines(output)
        if not lines:
            return
        self._replace_output(lines, partial=True)

    def close(self) -> None:
        if not self._is_open:
            return
        if self._has_output and not self._bottom_visible:
            sys.stdout.write(self._border_line(BOX_BOTTOM_LEFT, BOX_BOTTOM_RIGHT))
            sys.stdout.flush()
            self._bottom_visible = True
        self._is_open = False


def command(command_line: list[str]) -> None:
    command_preview(command_line)
