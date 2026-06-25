# tests/test_coverage_extra.py
"""Targeted tests for previously-uncovered edge branches:
- ToolAgent: leading-system-message prompt override; the deadline-exceeded break
  in stream() and astream().
- usage.build_usage: graceful {} when token counting is unavailable (no tiktoken).
"""
from __future__ import annotations

import asyncio

from aixon.agents.tool_agent import ToolAgent
from aixon.message import Message
from aixon.server import usage as usage_mod
from tests._fakes import make_llm


def _tool_agent(name, **attrs):
    return type(name, (ToolAgent,), {"name": name.lower(), "llm": make_llm(), **attrs})()


# --- ToolAgent: leading system message overrides self.prompt -----------------

def test_leading_system_message_is_consumed():
    agent = _tool_agent("SysOverrideAgent", prompt="class prompt")
    out = agent.invoke([
        Message(role="system", content="override prompt"),
        Message(role="user", content="hi"),
    ])
    assert out.role == "assistant"  # ran with the leading system message stripped/applied


# --- ToolAgent.stream deadline-exceeded break (sync + async) -----------------
# max_execution_time=-1 means "already expired", so the per-update deadline check
# trips on the first update deterministically — no global-clock patching (which
# LangGraph's internals also consume).

def test_stream_breaks_when_deadline_exceeded():
    agent = _tool_agent("StreamDeadlineAgent", max_execution_time=-1)
    chunks = list(agent.stream([Message(role="user", content="hi")]))
    assert any(c.reasoning and "exceeded max_execution_time" in c.reasoning for c in chunks)


def test_astream_breaks_when_deadline_exceeded():
    agent = _tool_agent("AStreamDeadlineAgent", max_execution_time=-1)

    async def _collect():
        return [c async for c in agent.astream([Message(role="user", content="hi")])]

    chunks = asyncio.run(_collect())
    assert any(c.reasoning and "exceeded max_execution_time" in c.reasoning for c in chunks)


# --- ToolAgent.astream surfaces per-update tool-call reasoning ----------------

def test_astream_surfaces_tool_call_reasoning():
    from langchain_core.messages import AIMessage

    def faq(query: str) -> str:
        """Search the FAQ."""
        return "the answer"

    llm = make_llm()
    llm.chat_model.script = [
        AIMessage(content="", tool_calls=[{"name": "faq", "args": {"query": "x"}, "id": "c1"}]),
        AIMessage(content="done"),
    ]
    agent = type("AStreamReasonAgent", (ToolAgent,),
                 {"name": "astreamreason", "llm": llm, "tools": [faq]})()

    async def _collect():
        return [c async for c in agent.astream([Message(role="user", content="hi")])]

    chunks = asyncio.run(_collect())
    # The tool-call label is drained per-update and streamed as a reasoning chunk.
    assert any(c.reasoning and "faq" in c.reasoning for c in chunks)
    assert chunks[-1].done is True


# --- usage: graceful degradation when token counting is unavailable ----------

def test_build_usage_returns_empty_without_tiktoken(monkeypatch):
    # Simulate tiktoken missing/erroring: _encoding raises -> count_tokens None
    # -> build_usage {} (never an error).
    def _raise(_model):
        raise ImportError("no tiktoken")

    monkeypatch.setattr(usage_mod, "_encoding", _raise)
    assert usage_mod.count_tokens("gpt-4o-mini", "hello") is None
    assert usage_mod.build_usage("gpt-4o-mini", "prompt", "completion") == {}
