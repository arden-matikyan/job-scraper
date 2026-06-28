"""Tests for the shared agent infrastructure: ask_human + ToolRunner.

No network and no real LLM — the Anthropic client is replaced by a scripted fake
that emits tool_use / text blocks in the same shape the SDK returns.
"""
from __future__ import annotations

import builtins

import pytest

from agent.human_input import ask_human
from agent.tool_runner import (
    MaxIterationsExceeded,
    ToolDef,
    ToolRunner,
)


# --------------------------------------------------------------------------- #
# Scripted fake LLM (mimics anthropic response blocks)
# --------------------------------------------------------------------------- #
class _Text:
    type = "text"

    def __init__(self, text):
        self.text = text


class _ToolUse:
    type = "tool_use"

    def __init__(self, id, name, input):  # noqa: A002 - match SDK field name
        self.id = id
        self.name = name
        self.input = input


class _Resp:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class FakeLLM:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []

    def complete(self, messages, tools=None, system=None):
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        return self.scripted.pop(0)


# --------------------------------------------------------------------------- #
# ask_human
# --------------------------------------------------------------------------- #
def test_yes_all_returns_first_option(monkeypatch):
    monkeypatch.setenv("AGENT_YES_ALL", "1")
    assert ask_human("proceed?", options=["yes", "no"]) == "yes"


def test_yes_all_freetext_returns_empty(monkeypatch):
    monkeypatch.setenv("AGENT_YES_ALL", "1")
    assert ask_human("how many?") == ""


def test_options_reprompt_on_invalid(monkeypatch):
    monkeypatch.delenv("AGENT_YES_ALL", raising=False)
    answers = iter(["maybe", "YES"])  # first invalid, then a case-insensitive match
    monkeypatch.setattr(builtins, "input", lambda *_: next(answers))
    assert ask_human("proceed?", options=["yes", "no"]) == "yes"  # canonical option


def test_eof_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("AGENT_YES_ALL", raising=False)

    def _raise(*_):
        raise EOFError

    monkeypatch.setattr(builtins, "input", _raise)
    assert ask_human("proceed?", options=["d", "k"]) == "d"
    assert ask_human("free text?") == ""


# --------------------------------------------------------------------------- #
# ToolRunner
# --------------------------------------------------------------------------- #
def _echo_tool(calls):
    def echo(value):
        calls.append(value)
        return {"echoed": value}

    return ToolDef(
        name="echo",
        description="echo a value",
        parameters_schema={"type": "object", "properties": {"value": {"type": "string"}}},
        fn=echo,
    )


def test_runner_executes_tool_then_finishes():
    calls = []
    llm = FakeLLM([
        _Resp([_ToolUse("t1", "echo", {"value": "hi"})], stop_reason="tool_use"),
        _Resp([_Text("all done")], stop_reason="end_turn"),
    ])
    runner = ToolRunner(llm, [_echo_tool(calls)], system_prompt="sys", max_iterations=5)
    result = runner.run("do it")
    assert result.success is True
    assert result.final_message == "all done"
    assert calls == ["hi"]
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "echo"
    assert result.tool_calls[0]["is_error"] is False
    assert result.tool_calls[0]["result"] == {"echoed": "hi"}
    # the system prompt + tool schema reached the client
    assert llm.calls[0]["system"] == "sys"
    assert llm.calls[0]["tools"][0]["name"] == "echo"
    assert "input_schema" in llm.calls[0]["tools"][0]


def test_runner_catches_tool_error_without_crashing():
    def boom():
        raise ValueError("kaboom")

    tool = ToolDef("boom", "raises", {"type": "object", "properties": {}}, boom)
    llm = FakeLLM([
        _Resp([_ToolUse("t1", "boom", {})], stop_reason="tool_use"),
        _Resp([_Text("recovered")], stop_reason="end_turn"),
    ])
    runner = ToolRunner(llm, [tool], system_prompt="sys", max_iterations=5)
    result = runner.run("go")
    assert result.success is True
    assert result.tool_calls[0]["is_error"] is True
    assert "kaboom" in result.tool_calls[0]["error"]


def test_runner_unknown_tool_is_reported_not_raised():
    llm = FakeLLM([
        _Resp([_ToolUse("t1", "ghost", {})], stop_reason="tool_use"),
        _Resp([_Text("ok")], stop_reason="end_turn"),
    ])
    runner = ToolRunner(llm, [], system_prompt="sys", max_iterations=5)
    result = runner.run("go")
    assert result.success is True
    assert result.tool_calls[0]["is_error"] is True
    assert "Unknown tool" in result.tool_calls[0]["error"]


def test_runner_raises_max_iterations_with_audit_trail():
    calls = []
    # Always returns a tool_use -> never ends the turn.
    llm = FakeLLM([
        _Resp([_ToolUse(f"t{i}", "echo", {"value": str(i)})], stop_reason="tool_use")
        for i in range(10)
    ])
    runner = ToolRunner(llm, [_echo_tool(calls)], system_prompt="sys", max_iterations=3)
    with pytest.raises(MaxIterationsExceeded) as exc:
        runner.run("loop forever")
    assert exc.value.iterations == 3
    assert len(exc.value.tool_calls) == 3
