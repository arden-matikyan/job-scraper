"""Human-in-the-loop prompt utility for the agentic scripts.

``ask_human`` pauses an agent loop and collects a typed response from the operator,
rendering any supporting ``context`` as rich tables/panels first so the human never
has to answer a bare yes/no without the relevant data in front of them.

``filter_qa.py`` routes every human gate through this one function (it is also
exposed to the LLM as an ``ask_human`` tool). Like the rest
of the codebase it never raises on bad input — it re-prompts — and it honours a
global ``AGENT_YES_ALL=1`` escape hatch that auto-answers every prompt with its first
option, so the scripts can be smoke-tested end to end without a TTY.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)

console = Console()


def yes_all_enabled() -> bool:
    """True when AGENT_YES_ALL=1 — prompts auto-answer instead of blocking on stdin."""
    return os.getenv("AGENT_YES_ALL", "").strip() == "1"


# --------------------------------------------------------------------------- #
# Context rendering
# --------------------------------------------------------------------------- #
def _truncate(value: Any, width: int = 80) -> str:
    s = "" if value is None else str(value)
    s = s.replace("\n", " ").strip()
    return s if len(s) <= width else s[: width - 1] + "…"


def _render_rows(title: str, rows: list[dict]) -> None:
    """Render a list of dict records as a rich Table (columns = union of keys)."""
    columns: list[str] = []
    for row in rows:
        for k in row:
            if k not in columns:
                columns.append(k)
    table = Table(title=title, show_lines=False, header_style="bold cyan")
    table.add_column("#", justify="right", style="dim")
    for col in columns:
        table.add_column(str(col), overflow="fold", max_width=60)
    for i, row in enumerate(rows[:50], start=1):
        table.add_row(str(i), *[_truncate(row.get(col), 60) for col in columns])
    if len(rows) > 50:
        table.caption = f"… {len(rows) - 50} more not shown"
    console.print(table)


def _render_context(context: dict) -> None:
    """Show a structured context dict: scalars as a key/value panel, lists as tables."""
    scalars: dict[str, Any] = {}
    for key, value in context.items():
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            _render_rows(str(key), value)
        elif isinstance(value, list):
            body = "\n".join(f"• {_truncate(v, 100)}" for v in value) or "(empty)"
            console.print(Panel(body, title=str(key), border_style="dim"))
        elif isinstance(value, dict):
            _render_rows(str(key), [value])
        else:
            scalars[key] = value
    if scalars:
        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_column(style="bold")
        table.add_column(overflow="fold")
        for key, value in scalars.items():
            table.add_row(str(key), _truncate(value, 110))
        console.print(Panel(table, title="context", border_style="dim"))


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
def ask_human(
    question: str,
    context: Optional[dict] = None,
    options: Optional[list[str]] = None,
    allow_freetext: bool = True,
) -> str:
    """Pause the agent loop and collect a human response.

    - ``context`` (if given) is rendered as rich tables/panels before the question.
    - ``options`` (if given) constrains the answer: the response must match one of
      them (case-insensitive); on a mismatch the prompt is re-shown with an
      ``[Invalid — enter one of: ...]`` hint. The canonical option string is returned.
    - ``allow_freetext`` only applies when ``options`` is None: when False, an empty
      response is rejected and re-prompted.

    Returns the raw response string (the caller parses it). Never raises.
    With ``AGENT_YES_ALL=1`` it returns the first option (or "" when none) without
    reading stdin.
    """
    if context:
        try:
            _render_context(context)
        except Exception as exc:  # rendering must never break the prompt
            logger.warning("ask_human context render failed: %s", exc)

    if yes_all_enabled():
        auto = options[0] if options else ""
        console.print(
            Panel(question, title="Human input (AGENT_YES_ALL)", border_style="yellow")
        )
        console.print(f"[yellow](AGENT_YES_ALL) auto-answering: {auto!r}[/]")
        return auto

    console.print(Panel(question, title="Human input needed", border_style="cyan"))
    if options:
        console.print("Options: " + " / ".join(f"[bold]{o}[/]" for o in options))

    while True:
        try:
            resp = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            # No interactive stdin (or Ctrl-C/Ctrl-D): fall back to a safe default
            # rather than crashing the agent mid-loop.
            fallback = options[0] if options else ""
            console.print(f"[yellow]No input available — using default {fallback!r}[/]")
            return fallback

        if options is not None:
            match = next((o for o in options if o.lower() == resp.lower()), None)
            if match is not None:
                return match
            console.print(
                f"[yellow][Invalid — enter one of: {', '.join(options)}][/]"
            )
            continue

        if not resp and not allow_freetext:
            console.print("[yellow]A response is required.[/]")
            continue
        return resp
