# tests/test_async_agents.py
"""Async surface: ainvoke/astream on LLM, LLMAgent, ToolAgent, Orchestrator and
the default sync->async bridge on the base Agent. Sync stays the default; async
is additive. Tests use asyncio.run (no pytest-asyncio dependency)."""
from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage

from aixon.agent import Agent
from aixon.agents.orchestrator import Orchestrator
from aixon.agents.tool_agent import ToolAgent
from aixon.llm import LLM
from aixon.message import Chunk, Message
from aixon.registry import get_registry
from aixon.retriever import Retriever, TypeAccess
from tests._fakes import make_llm


def _run(coro):
    return asyncio.run(coro)


async def _collect(agen):
    return [c async for c in agen]


# --- LLM ---------------------------------------------------------------------

def test_llm_acomplete_and_astream():
    llm = make_llm()
    msg = _run(llm.acomplete([Message(role="user", content="hi")]))
    assert isinstance(msg, Message) and msg.role == "assistant"
    chunks = _run(_collect(llm.astream([Message(role="user", content="hi")])))
    assert chunks[-1].done is True


# --- LLMAgent (native async) -------------------------------------------------

def test_llm_agent_ainvoke_and_astream():
    from aixon.agents.llm_agent import LLMAgent

    cls = type("AsyncGreeterAgent", (LLMAgent,), {"name": "agreeter", "llm": make_llm()})
    agent = cls()
    out = _run(agent.ainvoke([Message(role="user", content="hi")]))
    assert out.role == "assistant"
    chunks = _run(_collect(agent.astream([Message(role="user", content="hi")])))
    assert chunks[-1].done is True


# --- base Agent default bridge (pure-sync custom agent gets async free) -------

def test_sync_agent_gets_async_via_default_bridge():
    class BridgeAgent(Agent):
        name = "bridge"

        def invoke(self, messages):
            return Message(role="assistant", content="S:" + messages[-1].content)

        def stream(self, messages):
            yield Chunk(content="S:" + messages[-1].content)
            yield Chunk(done=True)

    agent = get_registry().resolve("bridge")
    out = _run(agent.ainvoke([Message(role="user", content="x")]))
    assert out.content == "S:x"
    chunks = _run(_collect(agent.astream([Message(role="user", content="x")])))
    assert [c.content for c in chunks if c.content] == ["S:x"]
    assert chunks[-1].done is True


# --- ToolAgent (native async over the graph) ---------------------------------

class _KbRetriever(Retriever):
    description = "kb"
    type_access = TypeAccess.READ

    def search(self, query, *, k=None):
        return [{"text": "reset via Forgot Password", "metadata": {}}]


def test_tool_agent_ainvoke_and_astream():
    cls = type(
        "AsyncHelpAgent",
        (ToolAgent,),
        {"name": "ahelp", "llm": make_llm(),
         "tools": [_KbRetriever().as_tool(name="faq", description="faq")]},
    )
    agent = cls()
    out = _run(agent.ainvoke([Message(role="user", content="hi")]))
    assert out.role == "assistant"
    chunks = _run(_collect(agent.astream([Message(role="user", content="hi")])))
    assert chunks[-1].done is True


def test_tool_agent_ainvoke_surfaces_reasoning_on_tool_call():
    # Script the fake model: call the tool, then answer -> reasoning labels appear.
    llm = make_llm()
    llm.chat_model.script = [
        AIMessage(content="", tool_calls=[{"name": "faq", "args": {"query": "x"}, "id": "c1"}]),
        AIMessage(content="done"),
    ]
    cls = type(
        "ReasonAsyncAgent",
        (ToolAgent,),
        {"name": "areason", "llm": llm,
         "tools": [_KbRetriever().as_tool(name="faq", description="faq")]},
    )
    out = _run(cls().ainvoke([Message(role="user", content="hi")]))
    assert out.reasoning and "faq" in out.reasoning


# --- Orchestrator (Tier 1 + Tier 2 async) ------------------------------------

def test_orchestrator_tier2_ainvoke_and_astream():
    type("WorkerAgent", (ToolAgent,), {"name": "worker", "llm": make_llm()})

    class T2Orchestrator(Orchestrator):
        name = "t2"
        nodes = {"worker": get_registry().resolve("worker")}
        entry = "worker"

    o = get_registry().resolve("t2")
    out = _run(o.ainvoke([Message(role="user", content="hi")]))
    assert out.role == "assistant"
    chunks = _run(_collect(o.astream([Message(role="user", content="hi")])))
    assert chunks[-1].done is True


def test_orchestrator_tier1_ainvoke():
    type("W1Agent", (ToolAgent,), {"name": "w1", "llm": make_llm()})

    class Sup1Orchestrator(Orchestrator):
        name = "sup1"
        supervisor = make_llm()
        agents = [get_registry().resolve("w1")]

    out = _run(get_registry().resolve("sup1").ainvoke([Message(role="user", content="hi")]))
    assert out.role == "assistant"
