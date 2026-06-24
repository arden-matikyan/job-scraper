"""The agentic loop core, shared by ``scraper_builder.py`` and ``filter_qa.py``.

A ``ToolRunner`` drives the standard "send messages → model emits tool calls →
execute them → feed results back → repeat" loop against an Anthropic-compatible
client (one with a ``complete(messages, tools, system)`` method returning a response
whose ``.content`` is a list of text / ``tool_use`` blocks). ``ask_human`` is just
another tool in the list — the runner gives it no special treatment.

Design goals (see the project plan):
  * tool execution errors are caught and returned to the model as the tool result,
    never crashing the loop;
  * ``max_iterations`` is a hard stop that raises ``MaxIterationsExceeded`` carrying
    the full audit trail;
  * every tool call is recorded in ``RunResult.tool_calls`` for auditability.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from rich.console import Console

logger = logging.getLogger(__name__)


@dataclass
class ToolDef:
    """A single tool the model may call: metadata + the Python function behind it."""

    name: str
    description: str
    parameters_schema: dict  # JSON Schema for the tool's parameters
    fn: Callable[..., Any]   # actual function; called as fn(**tool_input)

    def to_anthropic(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters_schema,
        }


@dataclass
class RunResult:
    success: bool
    final_message: str
    tool_calls: list[dict] = field(default_factory=list)  # audit trail
    iterations: int = 0


class MaxIterationsExceeded(Exception):
    """Raised when the loop hits ``max_iterations`` without the model ending its turn."""

    def __init__(self, tool_calls: list[dict], iterations: int):
        self.tool_calls = tool_calls
        self.iterations = iterations
        super().__init__(
            f"Tool loop exceeded {iterations} iterations "
            f"after {len(tool_calls)} tool call(s) without finishing."
        )


def _stringify_result(value: Any) -> str:
    """Serialise a tool's return value to a string for a tool_result block."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


class ToolRunner:
    def __init__(
        self,
        llm_client,
        tools: list[ToolDef],
        system_prompt: str,
        max_iterations: int = 20,
        verbose: bool = False,
        console: Optional[Console] = None,
    ):
        self.llm = llm_client
        self.tools = list(tools)
        self.tool_map = {t.name: t for t in self.tools}
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.console = console or Console()
        self._anthropic_tools = [t.to_anthropic() for t in self.tools]

    # ------------------------------------------------------------------- loop
    def run(self, user_message: str) -> RunResult:
        """Run the loop until the model ends its turn or ``max_iterations`` is hit."""
        messages: list[dict] = [{"role": "user", "content": user_message}]
        tool_calls: list[dict] = []

        for iteration in range(1, self.max_iterations + 1):
            try:
                resp = self.llm.complete(
                    messages,
                    tools=self._anthropic_tools,
                    system=self.system_prompt,
                )
            except Exception as exc:  # LLM/network failure — stop cleanly
                logger.error("LLM complete() failed on iteration %d: %s", iteration, exc)
                return RunResult(
                    success=False,
                    final_message=f"LLM call failed: {exc}",
                    tool_calls=tool_calls,
                    iterations=iteration,
                )

            text_parts, tool_uses, assistant_content = self._parse_response(resp)
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

            if self.verbose and text_parts:
                self.console.print(f"[dim]LLM:[/] {' '.join(text_parts)}")

            # No tool calls => the model is done (stop_reason == end_turn).
            if not tool_uses:
                return RunResult(
                    success=True,
                    final_message="\n".join(text_parts).strip(),
                    tool_calls=tool_calls,
                    iterations=iteration,
                )

            result_blocks = []
            for tid, name, tool_input in tool_uses:
                record, block = self._execute_tool(iteration, tid, name, tool_input)
                tool_calls.append(record)
                result_blocks.append(block)
            messages.append({"role": "user", "content": result_blocks})

        raise MaxIterationsExceeded(tool_calls, self.max_iterations)

    # -------------------------------------------------------------- internals
    @staticmethod
    def _parse_response(resp) -> tuple[list[str], list[tuple[str, str, dict]], list[dict]]:
        """Split a response into (text parts, tool_use calls, replayable content)."""
        text_parts: list[str] = []
        tool_uses: list[tuple[str, str, dict]] = []
        assistant_content: list[dict] = []
        for block in getattr(resp, "content", None) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
                assistant_content.append({"type": "text", "text": block.text})
            elif btype == "tool_use":
                tool_input = block.input if isinstance(block.input, dict) else {}
                tool_uses.append((block.id, block.name, tool_input))
                assistant_content.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": tool_input}
                )
        return text_parts, tool_uses, assistant_content

    def _execute_tool(
        self, iteration: int, tid: str, name: str, tool_input: dict
    ) -> tuple[dict, dict]:
        """Run one tool call; return (audit record, tool_result content block)."""
        if self.verbose:
            self.console.print(f"[cyan]→ tool[/] {name}({_stringify_result(tool_input)[:300]})")

        tool = self.tool_map.get(name)
        if tool is None:
            err = f"Unknown tool {name!r}. Available: {', '.join(self.tool_map)}"
            logger.warning(err)
            return (
                {"iteration": iteration, "name": name, "input": tool_input,
                 "error": err, "is_error": True},
                {"type": "tool_result", "tool_use_id": tid, "content": err, "is_error": True},
            )

        try:
            result = tool.fn(**tool_input)
            result_str = _stringify_result(result)
            if self.verbose:
                self.console.print(f"[green]← {name}[/] {result_str[:300]}")
            return (
                {"iteration": iteration, "name": name, "input": tool_input,
                 "result": result, "is_error": False},
                {"type": "tool_result", "tool_use_id": tid, "content": result_str},
            )
        except Exception as exc:  # tool failure must not crash the loop
            err = f"{type(exc).__name__}: {exc}"
            logger.warning("Tool %r raised: %s", name, err)
            return (
                {"iteration": iteration, "name": name, "input": tool_input,
                 "error": err, "is_error": True},
                {"type": "tool_result", "tool_use_id": tid, "content": err, "is_error": True},
            )
