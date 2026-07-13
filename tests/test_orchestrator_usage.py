# tests/test_orchestrator_usage.py
"""Orchestrator: usage aggregated across EVERY model turn of a run — every
worker invocation plus the Tier-1 supervisor's routing calls — not just the
last worker's turn. Message.usage on the final answer must be the SUM."""
from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage

from aixon.agent import Agent
from aixon.agents.orchestrator import Orchestrator
from aixon.message import Message
from aixon.registry import get_registry
from aixon.state import END
from tests._fakes import make_llm


def _usage(p: int, c: int) -> dict:
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def make_usage_agent(name: str, content: str, usage: "dict | None"):
    """Concrete Agent whose invoke/ainvoke return a Message carrying `usage`
    (simulating a real ToolAgent/LLMAgent worker turn)."""

    def invoke(self, messages: list[Message]) -> Message:
        return Message(role="assistant", content=content, usage=usage)

    from typing import Iterator

    from aixon.message import Chunk

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        yield Chunk(content=content)
        yield Chunk(done=True)

    cls = type(f"{name.capitalize()}Agent", (Agent,),
               {"invoke": invoke, "stream": stream, "name": name})
    return get_registry().resolve(name)


USER = [Message(role="user", content="hi")]


# ── Tier 1: supervisor + worker turns sum ────────────────────────────────────

def test_tier1_sums_supervisor_and_worker_usage_sync():
    make_usage_agent("billing", "handled", _usage(10, 5))
    sup = make_llm()
    sup.chat_model.script = [
        AIMessage(content="billing",
                   usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}),
        AIMessage(content="DONE",
                   usage_metadata={"input_tokens": 6, "output_tokens": 2, "total_tokens": 8}),
    ]

    class UsageOrchestrator(Orchestrator):
        supervisor = sup
        agents = [get_registry().resolve("billing")]

    orch = get_registry().resolve("usageorchestrator")
    out = orch.invoke(USER)
    assert out.usage == _usage(10 + 3 + 6, 5 + 1 + 2)


def test_tier1_sums_supervisor_and_worker_usage_async():
    make_usage_agent("billing-a", "handled", _usage(10, 5))
    sup = make_llm()
    sup.chat_model.script = [
        AIMessage(content="billing-a",
                   usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}),
        AIMessage(content="DONE",
                   usage_metadata={"input_tokens": 6, "output_tokens": 2, "total_tokens": 8}),
    ]

    class AsyncUsageOrchestrator(Orchestrator):
        supervisor = sup
        agents = [get_registry().resolve("billing-a")]

    orch = get_registry().resolve("asyncusageorchestrator")
    out = asyncio.run(orch.ainvoke(USER))
    assert out.usage == _usage(10 + 3 + 6, 5 + 1 + 2)


def test_tier1_turn_without_usage_contributes_zero_not_none():
    # The worker reports no usage; the supervisor's two routing calls do.
    # The total must still be the supervisor's sum, not None.
    make_usage_agent("silent-worker", "handled", None)
    sup = make_llm()
    sup.chat_model.script = [
        AIMessage(content="silent-worker",
                   usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}),
        AIMessage(content="DONE",
                   usage_metadata={"input_tokens": 6, "output_tokens": 2, "total_tokens": 8}),
    ]

    class PartialUsageOrchestrator(Orchestrator):
        supervisor = sup
        agents = [get_registry().resolve("silent-worker")]

    orch = get_registry().resolve("partialusageorchestrator")
    out = orch.invoke(USER)
    assert out.usage == _usage(3 + 6, 1 + 2)


def test_tier1_no_turn_reports_usage_final_usage_is_none():
    make_usage_agent("mute-worker", "handled", None)

    class _Supervisor:
        def complete(self, messages: list[Message]) -> Message:
            if messages and messages[-1].role == "assistant":
                return Message(role="assistant", content="DONE")
            return Message(role="assistant", content="mute-worker")

    class NoUsageOrchestrator(Orchestrator):
        supervisor = _Supervisor()
        agents = [get_registry().resolve("mute-worker")]

    orch = get_registry().resolve("nousageorchestrator")
    out = orch.invoke(USER)
    assert out.usage is None


# ── Tier 2: multiple worker nodes sum (no supervisor) ────────────────────────

def test_tier2_sums_usage_across_nodes():
    make_usage_agent("triage2", "triaged", _usage(4, 2))
    make_usage_agent("respond2", "responded", _usage(7, 3))

    class FlowOrchestrator(Orchestrator):
        nodes = {"triage2": get_registry().resolve("triage2"),
                 "respond2": get_registry().resolve("respond2")}
        entry = "triage2"
        edges = [("triage2", "respond2"), ("respond2", END)]

    orch = get_registry().resolve("floworchestrator")
    out = orch.invoke(USER)
    assert out.usage == _usage(4 + 7, 2 + 3)


def test_tier2_sums_usage_across_nodes_async():
    make_usage_agent("triage2a", "triaged", _usage(4, 2))
    make_usage_agent("respond2a", "responded", _usage(7, 3))

    class AsyncFlowOrchestrator(Orchestrator):
        nodes = {"triage2a": get_registry().resolve("triage2a"),
                 "respond2a": get_registry().resolve("respond2a")}
        entry = "triage2a"
        edges = [("triage2a", "respond2a"), ("respond2a", END)]

    orch = get_registry().resolve("asyncfloworchestrator")
    out = asyncio.run(orch.ainvoke(USER))
    assert out.usage == _usage(4 + 7, 2 + 3)
