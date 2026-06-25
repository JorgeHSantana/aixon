# tests/test_async_tools.py
"""Async tool callables must actually run on the async agent path.

Regression for the coerce_tools bug: an async callable was registered as a sync
`func`, so StructuredTool called it synchronously, dropped the un-awaited
coroutine, and the tool silently never ran (in invoke OR ainvoke). Now async
callables register via `coroutine=`:
  - async tool + ainvoke/astream  -> runs
  - sync  tool + invoke/ainvoke   -> runs (unchanged)
  - async tool + sync invoke      -> raises (loud), never silently skipped
"""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage

from aixon.agents.tool_agent import ToolAgent
from aixon.message import Message
from tests._fakes import make_llm

_ran: list[str] = []


async def afetch(symbol: str) -> str:
    """Async fetch."""
    _ran.append("async-ran")
    return "quote:" + symbol


def sfetch(symbol: str) -> str:
    """Sync fetch."""
    _ran.append("sync-ran")
    return "quote:" + symbol


def _agent(name, tool):
    llm = make_llm()
    llm.chat_model.script = [
        AIMessage(content="", tool_calls=[{"name": tool.__name__, "args": {"symbol": "AAPL"}, "id": "c1"}]),
        AIMessage(content="done"),
    ]
    return type(name, (ToolAgent,), {"name": name.lower(), "llm": llm, "tools": [tool]})()


def test_async_tool_runs_under_ainvoke():
    _ran.clear()
    out = asyncio.run(_agent("AsyncToolAgent", afetch).ainvoke([Message(role="user", content="hi")]))
    assert "async-ran" in _ran          # was silently skipped before the fix
    assert out.role == "assistant"


def test_async_tool_runs_under_astream():
    _ran.clear()

    async def _drain():
        async for _ in _agent("AsyncToolStreamAgent", afetch).astream(
            [Message(role="user", content="hi")]
        ):
            pass

    asyncio.run(_drain())
    assert "async-ran" in _ran


def test_sync_tool_unaffected_in_both_paths():
    _ran.clear()
    _agent("SyncInvokeAgent", sfetch).invoke([Message(role="user", content="hi")])
    assert _ran == ["sync-ran"]
    _ran.clear()
    asyncio.run(_agent("SyncAinvokeAgent", sfetch).ainvoke([Message(role="user", content="hi")]))
    assert _ran == ["sync-ran"]


def test_async_tool_under_sync_invoke_raises_not_silent():
    _ran.clear()
    with pytest.raises(NotImplementedError):
        _agent("AsyncToolSyncAgent", afetch).invoke([Message(role="user", content="hi")])
    assert _ran == []  # never silently "ran" with a dropped coroutine
