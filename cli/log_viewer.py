"""Open a readable, self-contained browser view for an AIBrain log file.

Usage:
    python .\\cli\\log_viewer.py

When no path is supplied, the native file picker asks for one.  The generated
HTML is intentionally separate from the application distribution: it is a
developer-facing utility and is not one of ``cli.build_dist.APPLICATIONS``.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path
from re import compile as re_compile
from typing import Any

RECORD_START = re_compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|\s*"
    r"(?P<level>[A-Z]+)\s*\|\s*(?P<source>[^|]+?)\s*\|\s*(?P<message>.*)$"
)
CONTINUATION = re_compile(r"^\s*\|\s*\|\s*\|\s?(?P<message>.*)$")
COMMAND_MESSAGE = re_compile(r"^Command (?:completed|interrupted)(?:\s|$)")

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Log Viewer — __TITLE__</title>
  <style>
    :root {
      color-scheme: dark;
      --page: #0b1020;
      --surface: rgba(18, 27, 48, .86);
      --surface-strong: #17233d;
      --line: rgba(164, 190, 232, .15);
      --text: #edf4ff;
      --muted: #96a6c3;
      --accent: #72a7ff;
      --cyan: #62ded8;
      --warning: #ffc66d;
      --danger: #ff8292;
      --shadow: 0 22px 60px rgba(0, 0, 0, .28);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-width: 320px;
      color: var(--text);
      background:
        radial-gradient(circle at 12% -15%, #25456f 0, transparent 33rem),
        radial-gradient(circle at 90% 0, #153f54 0, transparent 30rem), var(--page);
    }
    .shell { width: min(1500px, calc(100% - 32px)); margin: 0 auto; padding: 38px 0 64px; }
    .masthead { display: flex; justify-content: space-between; gap: 28px; align-items: end; margin: 0 0 24px; }
    .eyebrow { color: var(--cyan); font-size: .72rem; font-weight: 750; letter-spacing: .15em; margin-bottom: 8px; }
    h1 { font-size: clamp(1.55rem, 3vw, 2.5rem); letter-spacing: -.045em; line-height: 1.05; margin: 0; }
    .subtle { color: var(--muted); font-size: .9rem; line-height: 1.5; margin: 10px 0 0; overflow-wrap: anywhere; }
    .summary { display: flex; align-items: center; gap: 9px; color: var(--muted); font-size: .82rem; white-space: nowrap; }
    .pulse { width: 9px; height: 9px; border-radius: 50%; background: var(--cyan); box-shadow: 0 0 0 5px rgba(98, 222, 216, .12); }
    .toolbar { display: grid; grid-template-columns: minmax(180px, 1fr) auto; gap: 14px; padding: 13px; border: 1px solid var(--line); border-radius: 16px; background: var(--surface); box-shadow: var(--shadow); backdrop-filter: blur(18px); position: sticky; top: 12px; z-index: 4; }
    .search { width: 100%; min-width: 0; padding: 12px 14px; border: 1px solid var(--line); border-radius: 10px; background: #0d1527; color: var(--text); font: inherit; outline: none; transition: border-color .18s ease, box-shadow .18s ease; }
    .search:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(114, 167, 255, .16); }
    .search::placeholder { color: #7585a2; }
    .filters { display: flex; flex-wrap: wrap; align-items: center; gap: 7px; }
    .filter, .copy { border: 1px solid var(--line); border-radius: 999px; color: var(--muted); background: transparent; cursor: pointer; font: 700 .72rem/1 inherit; letter-spacing: .04em; padding: 9px 11px; transition: background .18s ease, color .18s ease, transform .18s ease; }
    .filter:hover, .copy:hover { color: var(--text); background: rgba(114, 167, 255, .13); }
    .filter:active, .copy:active { transform: scale(.96); }
    .filter.active { color: #07101c; border-color: var(--cyan); background: var(--cyan); }
    .results { color: var(--muted); font: 700 .78rem/1 inherit; padding: 19px 2px 11px; }
    .log { display: grid; gap: 10px; }
    .entry { display: grid; grid-template-columns: 10px minmax(138px, 175px) minmax(78px, 94px) minmax(135px, 200px) minmax(0, 1fr); gap: 13px; align-items: start; padding: 14px 16px; border: 1px solid var(--line); border-radius: 13px; background: var(--surface); box-shadow: 0 8px 23px rgba(0, 0, 0, .13); animation: enter .32s ease both; transition: border-color .18s ease, transform .18s ease, background .18s ease; }
    .entry:hover { border-color: rgba(114, 167, 255, .38); background: var(--surface-strong); transform: translateY(-1px); }
    .entry[hidden] { display: none; }
    .marker { width: 8px; height: 8px; border-radius: 50%; background: #74829b; margin-top: 5px; box-shadow: 0 0 0 4px rgba(116, 130, 155, .1); }
    .entry.INFO .marker { background: var(--cyan); box-shadow: 0 0 0 4px rgba(98, 222, 216, .1); }
    .entry.WARNING .marker { background: var(--warning); box-shadow: 0 0 0 4px rgba(255, 198, 109, .12); }
    .entry.ERROR .marker, .entry.CRITICAL .marker { background: var(--danger); box-shadow: 0 0 0 4px rgba(255, 130, 146, .12); }
    .time, .source { color: var(--muted); font: .76rem/1.55 ui-monospace, SFMono-Regular, Consolas, monospace; }
    .level { width: max-content; color: #b8c5dc; font: 800 .68rem/1 ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing: .07em; padding: 6px 7px; border: 1px solid currentColor; border-radius: 5px; }
    .entry.INFO .level { color: var(--cyan); } .entry.WARNING .level { color: var(--warning); } .entry.ERROR .level, .entry.CRITICAL .level { color: var(--danger); }
    .message { min-width: 0; white-space: pre-wrap; overflow-wrap: anywhere; font: .88rem/1.56 ui-monospace, SFMono-Regular, Consolas, monospace; }
    .command { grid-column: 5; position: relative; margin-top: -2px; }
    .command pre { margin: 0; padding: 13px 42px 13px 14px; overflow: auto; border: 1px solid rgba(98, 222, 216, .22); border-radius: 9px; color: #d8f7f4; background: #07151f; font: .82rem/1.6 ui-monospace, SFMono-Regular, Consolas, monospace; }
    .copy { position: absolute; right: 8px; top: 8px; padding: 7px 9px; background: rgba(7, 21, 31, .92); }
    .raw { grid-template-columns: 10px minmax(0, 1fr); } .raw .message { grid-column: 2; }
    .empty { padding: 52px 20px; border: 1px dashed var(--line); border-radius: 13px; color: var(--muted); text-align: center; }
    @keyframes enter { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
    @media (max-width: 780px) {
      .shell { width: min(100% - 20px, 1500px); padding-top: 24px; }
      .masthead, .toolbar { display: block; } .summary { margin-top: 14px; } .filters { margin-top: 11px; }
      .entry { grid-template-columns: 10px minmax(0, 1fr); gap: 6px 10px; padding: 13px; }
      .time { grid-column: 2; } .level { grid-column: 2; } .source { grid-column: 2; } .message, .command { grid-column: 2; }
      .command { margin-top: 5px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="masthead">
      <div><div class="eyebrow">AIBRAIN / LOG READER</div><h1 id="title"></h1><p class="subtle" id="path"></p></div>
      <div class="summary"><span class="pulse"></span><span id="total"></span></div>
    </header>
    <section class="toolbar" aria-label="Log controls">
      <input class="search" id="search" type="search" autocomplete="off" placeholder="Search messages, source, time…  (press /)">
      <div class="filters" id="filters"></div>
    </section>
    <div class="results" id="results" aria-live="polite"></div>
    <section class="log" id="log"></section>
  </main>
  <script>
    const payload = __DATA__;
    const levels = ["ALL", ...new Set(payload.records.map(record => record.level).filter(Boolean))];
    let selected = "ALL";
    const log = document.querySelector("#log"), search = document.querySelector("#search"), results = document.querySelector("#results");
    document.querySelector("#title").textContent = payload.name;
    document.querySelector("#path").textContent = payload.path;
    document.querySelector("#total").textContent = `${payload.records.length.toLocaleString()} lines loaded`;
    const text = (tag, value, className = "") => { const node = document.createElement(tag); node.className = className; node.textContent = value; return node; };
    const makeEntry = (record, index) => {
      const entry = document.createElement("article"); entry.className = `entry ${record.level || "raw"}`; entry.style.animationDelay = `${Math.min(index, 24) * 11}ms`;
      entry.dataset.level = record.level || "RAW"; entry.dataset.search = [record.timestamp, record.level, record.source, record.message].join(" ").toLowerCase();
      entry.append(text("span", "", "marker"));
      if (!record.level) { entry.classList.add("raw"); entry.append(text("div", record.message, "message")); return entry; }
      entry.append(text("time", record.timestamp, "time")); entry.append(text("span", record.level, "level")); entry.append(text("span", record.source, "source"));
      if (record.command) {
        const wrap = document.createElement("div"); wrap.className = "command";
        const pre = text("pre", record.message); const button = text("button", "Copy", "copy"); button.type = "button";
        button.addEventListener("click", async () => { await navigator.clipboard.writeText(record.message); button.textContent = "Copied"; setTimeout(() => button.textContent = "Copy", 1200); });
        wrap.append(pre, button); entry.append(wrap);
      } else entry.append(text("div", record.message, "message"));
      return entry;
    };
    payload.records.forEach((record, index) => log.append(makeEntry(record, index)));
    const renderFilters = () => {
      const filters = document.querySelector("#filters");
      levels.forEach(level => { const button = text("button", level, "filter"); button.type = "button"; button.dataset.level = level; button.classList.toggle("active", level === selected); button.addEventListener("click", () => { selected = level; filters.querySelectorAll("button").forEach(item => item.classList.toggle("active", item.dataset.level === selected)); update(); }); filters.append(button); });
    };
    const update = () => { const query = search.value.trim().toLowerCase(); let count = 0; log.querySelectorAll(".entry").forEach(entry => { const visible = (selected === "ALL" || entry.dataset.level === selected) && (!query || entry.dataset.search.includes(query)); entry.hidden = !visible; if (visible) count++; }); results.textContent = `${count.toLocaleString()} of ${payload.records.length.toLocaleString()} entries shown`; };
    search.addEventListener("input", update); document.addEventListener("keydown", event => { if (event.key === "/" && document.activeElement !== search) { event.preventDefault(); search.focus(); } });
    renderFilters(); update();
  </script>
</body>
</html>
"""


def choose_log_file() -> Path | None:
    """Prompt for a log file using the platform's native file picker."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:  # pragma: no cover - depends on local Python install.
        raise RuntimeError("Tkinter is unavailable; pass the log-file path explicitly.") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askopenfilename(
            title="Choose a log file to read",
            filetypes=(("Log files", "*.log *.txt"), ("All files", "*.*")),
        )
    finally:
        root.destroy()
    return Path(selected) if selected else None


def read_log(path: Path) -> str:
    """Read common log encodings while preserving every line."""
    contents = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return contents.decode(encoding)
        except UnicodeDecodeError:
            continue
    return contents.decode("utf-8", errors="replace")


def parse_records(contents: str) -> list[dict[str, Any]]:
    """Turn aligned AIBrain records into structured entries without dropping raw text."""
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in contents.splitlines():
        match = RECORD_START.match(line)

        if match:
            record: dict[str, Any] = match.groupdict()

            message = record["message"].rstrip()
            source = record["source"].strip()

            record["message"] = message
            record["command"] = (
                    source in {"aibrain.command", "command"}
                    or bool(COMMAND_MESSAGE.match(message))
            )

            records.append(record)
            current = record
            continue

        continuation = CONTINUATION.match(line)
        if continuation and current is not None:
            current["message"] += f"\n{continuation['message'].rstrip()}"
            continue

        current = None
        records.append(
            {
                "timestamp": "",
                "level": "",
                "source": "",
                "message": line,
                "command": False,
            }
        )

    return records


def build_html(path: Path, records: list[dict[str, Any]]) -> str:
    """Return a browser document whose text is inserted through safe DOM APIs."""
    data = json.dumps(
        {"name": path.name, "path": str(path.resolve()), "records": records},
        ensure_ascii=False,
    ).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__TITLE__", html.escape(path.name)).replace(
        "__DATA__", data
    )


def output_path(source: Path, requested: Path | None) -> Path:
    """Choose a persistent temporary view unless the caller named an output file."""
    if requested is not None:
        return requested.resolve()
    directory = Path(tempfile.gettempdir()) / "aibrain-log-viewer"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return directory / f"{source.stem}-{stamp}.html"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open an AIBrain log file in a clean, searchable browser view."
    )
    parser.add_argument("log_file", nargs="?", type=Path, help="Log file to open")
    parser.add_argument(
        "-o", "--output", type=Path, help="Optional destination for the generated HTML"
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    source = arguments.log_file or choose_log_file()
    if source is None:
        print("No log file selected.", file=sys.stderr)
        return 2
    source = source.expanduser().resolve()
    if not source.is_file():
        print(f"Log file not found: {source}", file=sys.stderr)
        return 2

    destination = output_path(source, arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_html(source, parse_records(read_log(source))), encoding="utf-8")
    print(f"Opened log view: {destination}")
    if not webbrowser.open(destination.as_uri()):
        print("Could not open the browser. Open the HTML file shown above manually.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
