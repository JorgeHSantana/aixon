# tests/test_final_fix_wave.py
"""Onda de correção final do plano 1 (0.1.29): toolcache, MCP, ToolAgent."""
from __future__ import annotations

import contextvars
import warnings

import pytest

from aixon.toolcache import (
    ToolCallCache, current_tool_cache, tool_call_cache, use_tool_cache,
)


# ── item 2: reset do toolcache tolera outro Context ─────────────────────────

@pytest.mark.parametrize("factory", [tool_call_cache, lambda: use_tool_cache(ToolCallCache())])
def test_toolcache_exit_in_other_context_does_not_raise(factory):
    def body():
        scope = factory()
        scope.__enter__()
        contextvars.copy_context().run(scope.__exit__, None, None, None)  # sem ValueError

    contextvars.copy_context().run(body)
    assert current_tool_cache() is None


# ── item 7: a coroutine não nasce antes do ramo que a aguarda ───────────────

def test_mcp_discover_executor_failure_leaves_no_unawaited_coroutine(monkeypatch):
    import asyncio
    import concurrent.futures

    from aixon.exceptions import AixonError
    from aixon.mcp import MCPToolset

    class _Conn:
        base_url = "http://x"

        async def aas_tools(self, include=None, exclude=None):
            return []

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("sem threads")

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _Boom)
    ts = MCPToolset.__new__(MCPToolset)
    ts.connector, ts.include, ts.exclude = _Conn(), None, None

    async def go():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(AixonError):
                ts._discover()
            import gc
            gc.collect()
        return [w for w in caught if "never awaited" in str(w.message)]

    assert asyncio.run(go()) == []


# ── item 6: o cancelamento adiado encadeia a exceção que substitui ──────────

def test_deferred_cancel_chains_replaced_exception():
    import asyncio

    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

    from aixon.agents.tool_agent import ToolAgent
    from aixon.message import Message
    from aixon.reasoning import emit_reasoning
    from aixon.registry import get_registry
    from tests._fakes import make_llm

    @tool
    async def lenta(text: str) -> str:
        """Ferramenta lenta que avisa antes de dormir."""
        emit_reasoning("dormindo")
        await asyncio.sleep(1)
        return "ok"

    llm = make_llm()
    llm.chat_model.script = [
        AIMessage(content="", tool_calls=[{"name": "lenta", "args": {"text": "a"}, "id": "c1"}]),
        AIMessage(content="fim"),
    ]
    type("EncadeiaCancelAgent", (ToolAgent,), {
        "name": "encadeiacancel", "llm": llm, "tools": [lenta],
        "reasoning_flush_interval": 0.01})
    agent = get_registry().resolve("encadeiacancel")
    original = ValueError("substituída")

    async def go():
        gen = agent.astream([Message(role="user", content="oi")])
        async for chunk in gen:
            if chunk.reasoning and "dormindo" in chunk.reasoning:
                break  # parado num yield com a poll em voo
        # cancelamento entregue ao próprio gerador dentro do finally
        asyncio.current_task().cancel()
        await gen.athrow(original)

    with pytest.raises(asyncio.CancelledError) as info:
        asyncio.run(go())
    assert info.value.__cause__ is original


# ── item 8: aninhamento impose_model × cache da request × prediction ────────

def test_reflective_imposed_astream_inside_request_cache():
    import asyncio

    from langchain_core.messages import AIMessage

    from aixon import Agent, ReflectiveAgent
    from aixon.llm import LLM
    from aixon.message import Chunk, Message
    from aixon.registry import get_registry
    from aixon.runtime import current_imposed_model, current_prediction
    from tests._fakes import make_recording_provider

    make_recording_provider("rec", [AIMessage("1. falta fonte"), AIMessage("APROVADO")])
    seen: list[tuple] = []

    class NinhoWorkerAgent(Agent):
        name = "ninhoworker"
        hidden = True

        def invoke(self, messages):
            raise AssertionError("só astream")

        def stream(self, messages):
            raise AssertionError("só astream")

        async def astream(self, messages):
            # o que cada chamada de LLM do worker veria
            seen.append((current_imposed_model(), current_tool_cache(), current_prediction()))
            yield Chunk(reasoning="pensando")
            yield Chunk(content=f"v{len(seen)}")
            yield Chunk(done=True)

    class NinhoReflectiveAgent(ReflectiveAgent):
        name = "ninhoreflective"
        agent = get_registry().resolve("ninhoworker")
        llm = LLM("tier-x", provider="rec")
        judge_llm = LLM("juiz", provider="rec")
        judge_rubric = "1. A resposta cita a fonte."
        max_rounds = 2
        impose_model = True

    r = get_registry().resolve("ninhoreflective")
    ask = [Message(role="user", content="q")]

    async def full():
        with tool_call_cache() as request_cache:
            out = "".join([c.content or "" async for c in r.astream(ask)])
            return request_cache, out

    request_cache, out = asyncio.run(full())
    assert out == "v2"
    assert len(seen) == 2
    for imposed, cache, _ in seen:
        assert imposed is not None and imposed[0] == "tier-x"
        assert cache is request_cache  # o mesmo objeto da request
    assert seen[0][2] is None and seen[1][2] == "v1"  # prediction só no retry

    async def abandoned():
        with tool_call_cache() as request_cache:
            async for _ in r.astream(ask):
                break  # abandonado no meio, sem aclose()
            right_after = (current_imposed_model(), current_tool_cache(), current_prediction())
            await asyncio.sleep(0.05)
            later = (current_imposed_model(), current_tool_cache(), current_prediction())
            return request_cache, right_after, later

    request_cache, right_after, later = asyncio.run(abandoned())
    assert right_after == later == (None, request_cache, None)
