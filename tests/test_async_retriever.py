# tests/test_async_retriever.py
"""Async retriever support: Retriever.asearch + a dual AgentTool (sync func +
async coroutine), so a retriever tool runs on BOTH the sync `invoke` path
(-> search) and the async `ainvoke` path (-> asearch). Vendor retrievers
override asearch for true non-blocking I/O; the default bridges to search."""
from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage

from aixon._interop.tools import coerce_tools
from aixon.agents.tool_agent import ToolAgent
from aixon.message import Message
from aixon.retriever import Retriever, TypeAccess
from tests._fakes import make_llm

_calls: list[str] = []


class _KbRetriever(Retriever):
    description = "kb"
    type_access = TypeAccess.READ

    def search(self, query, *, k=None):
        _calls.append("sync-search")
        return [{"text": "sync result", "metadata": {}}]

    async def asearch(self, query, *, k=None):
        _calls.append("async-search")
        return [{"text": "async result", "metadata": {}}]


class _PlainRetriever(Retriever):
    description = "plain"
    type_access = TypeAccess.READ

    def search(self, query, *, k=None):
        return [{"text": "plain", "metadata": {}}]


def _agent(name):
    llm = make_llm()
    llm.chat_model.script = [
        AIMessage(content="", tool_calls=[{"name": "kb", "args": {"query": "x"}, "id": "c1"}]),
        AIMessage(content="done"),
    ]
    return type(name, (ToolAgent,),
                {"name": name.lower(), "llm": llm,
                 "tools": [_KbRetriever().as_tool(name="kb", description="kb")]})()


def test_retriever_tool_uses_sync_search_under_invoke():
    _calls.clear()
    _agent("SyncRetrAgent").invoke([Message(role="user", content="hi")])
    assert _calls == ["sync-search"]


def test_retriever_tool_uses_native_asearch_under_ainvoke():
    # The async path awaits the coroutine -> the overridden (native) asearch,
    # NOT the sync search. This is the true non-blocking path.
    _calls.clear()
    asyncio.run(_agent("AsyncRetrAgent").ainvoke([Message(role="user", content="hi")]))
    assert _calls == ["async-search"]


def test_default_asearch_bridges_to_search():
    # A retriever that doesn't override asearch still gets a working async path.
    docs = asyncio.run(_PlainRetriever().asearch("q"))
    assert docs[0]["text"] == "plain"


def test_as_tool_produces_dual_agent_tool():
    tool = _KbRetriever().as_tool()
    assert tool.func is not None and tool.coroutine is not None
    # coerce_tools wires both into the LangChain tool (sync run + async arun).
    coerced = coerce_tools([tool])[0]
    assert coerced.coroutine is not None


def test_agent_as_tool_is_also_dual():
    # Agent.as_tool() gains the async coroutine too (calls ainvoke).
    agent = type("LeafAgent", (ToolAgent,), {"name": "leaf", "llm": make_llm()})()
    tool = agent.as_tool()
    assert tool.func is not None and tool.coroutine is not None
    assert coerce_tools([tool])[0].coroutine is not None


def test_agent_as_tool_coroutine_actually_runs():
    # Execute the coroutine body (Agent.as_tool's _arun -> ainvoke), not just
    # assert it exists.
    agent = type("LeafRunAgent", (ToolAgent,), {"name": "leafrun", "llm": make_llm()})()
    out = asyncio.run(agent.as_tool().coroutine("hi"))
    assert isinstance(out, str)


def test_agent_as_tool_func_runs():
    agent = type("LeafSyncAgent", (ToolAgent,), {"name": "leafsync", "llm": make_llm()})()
    assert isinstance(agent.as_tool().func("hi"), str)


def test_write_capable_retriever_without_write_raises_not_implemented():
    # A non-READ retriever that doesn't override write() must raise
    # NotImplementedError (defensive branch in Retriever.write).
    import pytest

    class WritableRetriever(Retriever):
        description = "w"
        type_access = TypeAccess.ALL

        def search(self, query, *, k=None):
            return []

    with pytest.raises(NotImplementedError):
        WritableRetriever().write(["doc"])
