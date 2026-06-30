# tests/test_async_cancellation.py
"""ainvoke gives REAL cancellation: max_execution_time/timeout wrap the run in
asyncio.wait_for, so an overrunning run is cancelled at the next await point —
not run to completion like the sync post-hoc deadline. The test would take 30s
without cancellation; with it, it finishes in well under a second."""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aixon.agents.orchestrator import Orchestrator
from aixon.agents.tool_agent import ToolAgent
from aixon.exceptions import AixonError
from aixon.llm import LLM
from aixon.message import Message
from aixon.providers.base import Provider, register_provider
from aixon.registry import get_registry
from tests._fakes import make_llm


class _SlowModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "slow"

    def _generate(self, messages, stop=None, run_manager=None, **k) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="x"))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **k) -> ChatResult:
        await asyncio.sleep(30)  # would hang for 30s if not cancelled
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="x"))])


class _SlowProvider(Provider):
    name = "slow-cancel"
    env_key = ""

    def build(self, model: str, **p: Any) -> BaseChatModel:
        return _SlowModel()


register_provider(_SlowProvider())


def test_tool_agent_ainvoke_cancels_overrun_fast():
    cls = type(
        "SlowToolAgent",
        (ToolAgent,),
        {"name": "slowtool", "llm": LLM("m", provider="slow-cancel"),
         "max_execution_time": 0.05},
    )
    agent = cls()
    t0 = time.monotonic()
    with pytest.raises(AixonError, match="max_execution_time"):
        asyncio.run(agent.ainvoke([Message(role="user", content="hi")]))
    assert time.monotonic() - t0 < 5  # cancelled, not the full 30s sleep


def test_orchestrator_ainvoke_cancels_on_timeout():
    # The orchestrator's async graph awaits each worker's ainvoke, so a worker
    # whose model is genuinely async (the slow _agenerate) is cancelled promptly
    # when the orchestrator timeout fires — not run to completion.
    type("SlowWorkerAgent", (ToolAgent,),
         {"name": "slowworker", "llm": LLM("m", provider="slow-cancel")})

    class SlowOrchestrator(Orchestrator):
        name = "sloworch"
        nodes = {"slowworker": get_registry().resolve("slowworker")}
        entry = "slowworker"
        timeout = 1  # seconds; the worker would otherwise sleep 30s

    t0 = time.monotonic()
    with pytest.raises(AixonError, match="timeout"):
        asyncio.run(get_registry().resolve("sloworch").ainvoke(
            [Message(role="user", content="hi")]
        ))
    assert time.monotonic() - t0 < 5  # cancelled at the deadline, not after 30s


def test_sync_within_budget_still_works():
    # Sanity: a fast agent under the async path returns normally.
    cls = type("FastToolAgent", (ToolAgent,), {"name": "fasttool", "llm": make_llm()})
    out = asyncio.run(cls().ainvoke([Message(role="user", content="hi")]))
    assert out.role == "assistant"


def test_astream_stops_at_deadline_on_stalled_stream():
    # Regression for the indefinite-hang bug: a step that stalls mid-flight (the
    # slow model sleeps 30s) must NOT hang the stream. With the hard-wall
    # astream, wait_for cancels the stuck step at max_execution_time and emits a
    # "(stopped: ...)" reasoning line. WITHOUT the fix this test runs ~30s and
    # fails the elapsed assertion.
    cls = type(
        "StalledAstreamAgent",
        (ToolAgent,),
        {"name": "stalledastream", "llm": LLM("m", provider="slow-cancel"),
         "max_execution_time": 0.05},
    )
    agent = cls()

    async def _drain() -> tuple[list, float]:
        chunks: list = []
        t0 = time.monotonic()
        async for ch in agent.astream([Message(role="user", content="hi")]):
            chunks.append(ch)
        return chunks, time.monotonic() - t0

    chunks, elapsed = asyncio.run(_drain())
    assert elapsed < 5  # stopped at the 0.05s deadline, not after the 30s sleep
    assert any(
        "max_execution_time" in (getattr(c, "reasoning", "") or "") for c in chunks
    )
    assert chunks[-1].done is True  # the stream still terminates cleanly
