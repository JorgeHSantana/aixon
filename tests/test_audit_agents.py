# tests/test_audit_agents.py
"""Audit regressions for Orchestrator and ToolAgent (offline, fake-driven).

Covers four verified findings:
  1. supervisor DONE sentinel must win before worker-name substring matching;
  2. ToolAgent must wrap GraphRecursionError in AixonError on all run paths;
  3. stream/astream must not surface a tool-calling preamble as final content
     when the deadline breaks the run;
  4. orchestrator stream/astream reasoning lines need a trailing newline.
"""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage

from aixon.agent import Agent
from aixon.agents.orchestrator import Orchestrator
from aixon.agents.tool_agent import ToolAgent
from aixon.exceptions import AixonError
from aixon.llm import LLM
from aixon.message import Chunk, Message
from aixon.registry import get_registry
from tests._fakes import FakeChatModel, make_llm


# ── helpers ──────────────────────────────────────────────────────────────────

def _install(monkeypatch, llm, script):
    fake = FakeChatModel(script=script)
    monkeypatch.setattr(type(llm), "chat_model", property(lambda self: fake))


def _tool_call(name, args, id="call_1", content=""):
    return AIMessage(
        content=content, tool_calls=[{"name": name, "args": args, "id": id}]
    )


def _make_counting_worker(name: str, tag: str, calls: list):
    def invoke(self, messages: list[Message]) -> Message:
        calls.append(name)
        return Message(role="assistant", content=tag)

    def stream(self, messages: list[Message]):
        calls.append(name)
        yield Chunk(content=tag)
        yield Chunk(done=True)

    type(
        name.capitalize().replace("-", "") + "Agent",
        (Agent,),
        {
            "name": name,
            "description": f"handles {name}",
            "invoke": invoke,
            "stream": stream,
        },
    )
    return get_registry().resolve(name)


# ── Finding 1: DONE sentinel checked before worker-name matching ─────────────

class _ChattyDoneSupervisor:
    """Says DONE with an explanation that NAMES a worker — the substring
    fallback must not re-dispatch to 'billing' once the turn is answered."""

    def complete(self, messages: list[Message]) -> Message:
        if messages and messages[-1].role == "assistant":
            return Message(
                role="assistant",
                content="DONE — billing already answered this",
            )
        return Message(role="assistant", content="billing")


def test_done_with_worker_name_in_explanation_terminates():
    calls: list = []
    _make_counting_worker("billing", "[billing]", calls)

    class DoneChattyOrchestrator(Orchestrator):
        supervisor = _ChattyDoneSupervisor()
        agents = [get_registry().resolve("billing")]

    out = get_registry().resolve("donechattyorchestrator").invoke(
        [Message(role="user", content="refund please")]
    )
    assert out.content == "[billing]"
    assert calls == ["billing"]  # exactly one dispatch, no ping-pong


def test_done_with_punctuation_terminates():
    calls: list = []
    _make_counting_worker("billing", "[billing]", calls)

    class _Sup:
        def complete(self, messages):
            if messages and messages[-1].role == "assistant":
                return Message(role="assistant", content="Done.")
            return Message(role="assistant", content="billing")

    class DonePunctOrchestrator(Orchestrator):
        supervisor = _Sup()
        agents = [get_registry().resolve("billing")]

    out = get_registry().resolve("donepunctorchestrator").invoke(
        [Message(role="user", content="refund")]
    )
    assert out.content == "[billing]"
    assert calls == ["billing"]


def test_plain_worker_name_still_routes():
    # No over-trigger: a bare worker-name reply and a sentence naming only a
    # worker must still route (DONE handling must not eat these).
    calls: list = []
    _make_counting_worker("billing", "[billing]", calls)

    class _Sup:
        def complete(self, messages):
            if messages and messages[-1].role == "assistant":
                return Message(role="assistant", content="DONE")
            return Message(role="assistant", content="billing should handle it")

    class RoutesStillOrchestrator(Orchestrator):
        supervisor = _Sup()
        agents = [get_registry().resolve("billing")]

    out = get_registry().resolve("routesstillorchestrator").invoke(
        [Message(role="user", content="refund")]
    )
    assert out.content == "[billing]"
    assert calls == ["billing"]


# ── Finding 2: ToolAgent wraps GraphRecursionError on all four run paths ─────

def _looping_tool_agent(monkeypatch, cls_name: str):
    """A ToolAgent whose fake model ALWAYS tool-calls, so max_iterations
    (recursion_limit) is exhausted before a final answer."""
    def spin(n: int) -> int:
        """spin"""
        return n

    type(
        cls_name,
        (ToolAgent,),
        {
            "name": cls_name.lower(),
            "llm": LLM("fake-1", provider="fake"),
            "tools": [spin],
            "max_iterations": 1,  # recursion_limit = 3
        },
    )
    agent = get_registry().resolve(cls_name.lower())
    _install(
        monkeypatch,
        agent.llm,
        [_tool_call("spin", {"n": i}, id=f"call_{i}") for i in range(8)],
    )
    return agent


def test_invoke_wraps_recursion_error(monkeypatch):
    agent = _looping_tool_agent(monkeypatch, "LoopInvokeAgent")
    with pytest.raises(AixonError, match="max_iterations"):
        agent.invoke([Message(role="user", content="go")])


def test_stream_wraps_recursion_error(monkeypatch):
    agent = _looping_tool_agent(monkeypatch, "LoopStreamAgent")
    with pytest.raises(AixonError, match="max_iterations"):
        list(agent.stream([Message(role="user", content="go")]))


def test_ainvoke_wraps_recursion_error(monkeypatch):
    agent = _looping_tool_agent(monkeypatch, "LoopAinvokeAgent")
    with pytest.raises(AixonError, match="max_iterations"):
        asyncio.run(agent.ainvoke([Message(role="user", content="go")]))


def test_astream_wraps_recursion_error(monkeypatch):
    agent = _looping_tool_agent(monkeypatch, "LoopAstreamAgent")

    async def _drain():
        async for _ in agent.astream([Message(role="user", content="go")]):
            pass

    with pytest.raises(AixonError, match="max_iterations"):
        asyncio.run(_drain())


# ── Finding 3: deadline break must not surface a tool-call preamble as the
#    final answer (content on a tool-calling AI message is a thought, not a
#    final answer) ─────────────────────────────────────────────────────────

_PREAMBLE = "Let me query the orders DB..."


def test_stream_deadline_does_not_yield_preamble_as_content(monkeypatch):
    def lookup(q: str) -> str:
        """lookup"""
        return "rows"

    class PreambleStreamAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [lookup]
        max_execution_time = 0  # deadline hit after the first update

    agent = get_registry().resolve("preamblestreamagent")
    _install(
        monkeypatch,
        agent.llm,
        [
            _tool_call("lookup", {"q": "orders"}, content=_PREAMBLE),
            AIMessage(content="You have 3 orders."),
        ],
    )

    chunks = list(agent.stream([Message(role="user", content="my orders?")]))

    reasoning = "".join(c.reasoning for c in chunks if c.reasoning)
    assert "max_execution_time" in reasoning  # the stopped line is kept
    assert not any(c.content for c in chunks)  # no misleading final content
    assert chunks[-1].done is True


def test_astream_deadline_does_not_yield_preamble_as_content(monkeypatch):
    import time as _time

    def slow_lookup(q: str) -> str:
        """slow lookup"""
        _time.sleep(0.5)
        return "rows"

    class PreambleAstreamAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [slow_lookup]
        max_execution_time = 0.15  # model update lands; tool step overruns

    agent = get_registry().resolve("preambleastreamagent")
    _install(
        monkeypatch,
        agent.llm,
        [
            _tool_call("slow_lookup", {"q": "orders"}, content=_PREAMBLE),
            AIMessage(content="You have 3 orders."),
        ],
    )

    async def _drain() -> list:
        return [c async for c in agent.astream([Message(role="user", content="my orders?")])]

    chunks = asyncio.run(_drain())

    reasoning = "".join(c.reasoning for c in chunks if c.reasoning)
    assert "max_execution_time" in reasoning
    assert not any(c.content for c in chunks)
    assert chunks[-1].done is True


def test_stream_final_answer_content_still_streams(monkeypatch):
    # Guard: a normal run (no deadline break) still yields the final answer,
    # including when the tool-call message carried preamble content.
    def lookup(q: str) -> str:
        """lookup"""
        return "rows"

    class FinalAnswerStreamAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [lookup]

    agent = get_registry().resolve("finalanswerstreamagent")
    _install(
        monkeypatch,
        agent.llm,
        [
            _tool_call("lookup", {"q": "orders"}, content=_PREAMBLE),
            AIMessage(content="You have 3 orders."),
        ],
    )

    chunks = list(agent.stream([Message(role="user", content="my orders?")]))
    contents = [c.content for c in chunks if c.content]
    assert contents == ["You have 3 orders."]
    assert chunks[-1].done is True


# ── Finding 4: orchestrator stream/astream reasoning lines keep their "\n" ──
# (adapters serialize reasoning deltas additively; without the newline that
# tool_agent appends, consecutive lines run together on the wire).

def _make_two_thought_worker(name: str):
    from aixon import emit_reasoning

    def invoke(self, messages: list[Message]) -> Message:
        emit_reasoning("first thought")
        emit_reasoning("second thought")
        return Message(role="assistant", content="answer")

    def stream(self, messages: list[Message]):
        yield Chunk(content="answer")
        yield Chunk(done=True)

    type(
        f"{name.capitalize()}Agent",
        (Agent,),
        {"name": name, "invoke": invoke, "stream": stream},
    )
    return get_registry().resolve(name)


def _newline_orch(cls_name: str, worker_name: str):
    worker = _make_two_thought_worker(worker_name)

    type(
        cls_name,
        (Orchestrator,),
        {"supervisor": make_llm(), "agents": [worker]},
    )
    return get_registry().resolve(cls_name.lower())


def test_orchestrator_stream_reasoning_lines_end_with_newline():
    orch = _newline_orch("NewlineStreamOrchestrator", "musing")
    chunks = list(orch.stream([Message(role="user", content="hi")]))
    joined = "".join(c.reasoning for c in chunks if c.reasoning)
    assert "first thought\n" in joined
    assert "second thought\n" in joined
    assert "thoughtsecond" not in joined  # lines must not run together


def test_orchestrator_astream_reasoning_lines_end_with_newline():
    orch = _newline_orch("NewlineAstreamOrchestrator", "amusing")

    async def _drain() -> list:
        return [c async for c in orch.astream([Message(role="user", content="hi")])]

    chunks = asyncio.run(_drain())
    joined = "".join(c.reasoning for c in chunks if c.reasoning)
    assert "first thought\n" in joined
    assert "second thought\n" in joined
    assert "thoughtsecond" not in joined
