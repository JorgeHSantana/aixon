# tests/test_tool_shield_memo.py
"""#9 — exceção de tool vira resultado de erro relatável (shield) e
#5 — memoização de tool calls com escopo de request (toolcache).

Ambos vivem no MESMO wrapper aplicado em ``coerce_tools`` (o ponto único de
execução de tools), então qualquer composição (ToolAgent, ReflectiveAgent,
Orchestrator) se beneficia sem blindagem caso a caso."""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage

from aixon._interop.tools import coerce_tools
from aixon.agent import AgentTool
from aixon.message import Message
from aixon.toolcache import current_tool_cache, tool_call_cache


def make_agenttool(func=None, coroutine=None, memoize=True, name="t"):
    return AgentTool(
        name=name, description="d",
        func=func or (lambda text: "ok"),
        coroutine=coroutine, memoize=memoize,
    )


# ── #9: shield ───────────────────────────────────────────────────────────────

def test_shield_sync_exception_becomes_error_result():
    def boom(text: str) -> str:
        raise ValueError("boom")
    [tool] = coerce_tools([make_agenttool(boom)])
    out = tool.invoke({"text": "x"})
    assert "boom" in out
    assert "t" in out  # nomeia a tool que falhou


def test_shield_empty_str_exception_uses_repr():
    class Silent(Exception):
        def __str__(self):
            return ""  # o caso httpx.ReadTimeout
    def boom(text: str) -> str:
        raise Silent()
    [tool] = coerce_tools([make_agenttool(boom)])
    assert "Silent" in tool.invoke({"text": "x"})


def test_shield_async_exception():
    async def boom(text: str) -> str:
        raise ValueError("async-boom")
    [tool] = coerce_tools([make_agenttool(coroutine=boom)])
    out = asyncio.run(tool.ainvoke({"text": "x"}))
    assert "async-boom" in out


def test_shield_plain_callable():
    def quebrada(text: str) -> str:
        """tool que quebra"""
        raise RuntimeError("x1")
    [tool] = coerce_tools([quebrada])
    assert "x1" in tool.invoke({"text": "q"})


def test_shield_off_reraises():
    def boom(text: str) -> str:
        raise ValueError("boom")
    [tool] = coerce_tools([make_agenttool(boom)], shield_errors=False)
    with pytest.raises(ValueError):
        tool.invoke({"text": "x"})


def test_toolagent_run_survives_tool_exception():
    """E2E: a tool explode e o run do ToolAgent NÃO morre — o erro vira
    ToolMessage e o modelo produz a resposta final."""
    from aixon.agents.tool_agent import ToolAgent
    from aixon.registry import get_registry
    from tests._fakes import make_llm

    def caindo(text: str) -> str:
        """tool que sempre falha"""
        raise ValueError("infra fora")

    llm = make_llm()
    llm.chat_model.script = [
        AIMessage(content="", tool_calls=[
            {"name": "caindo", "args": {"text": "x"}, "id": "call_1"}]),
        AIMessage(content="recuperado"),
    ]
    type("ShieldedAgent", (ToolAgent,), {
        "name": "shielded", "llm": llm, "tools": [caindo]})
    agent = get_registry().resolve("shielded")
    out = agent.invoke([Message(role="user", content="oi")])
    assert out.content == "recuperado"


# ── #5: memoização ───────────────────────────────────────────────────────────

def test_memo_only_when_cache_active():
    calls = {"n": 0}
    def probe(text: str) -> str:
        calls["n"] += 1
        return f"r{calls['n']}"
    [tool] = coerce_tools([make_agenttool(probe)])
    tool.invoke({"text": "x"}); tool.invoke({"text": "x"})
    assert calls["n"] == 2                      # sem cache ativo: sem memo
    with tool_call_cache():
        a = tool.invoke({"text": "x"})
        b = tool.invoke({"text": "x"})
        assert a == b                           # memoizado
        tool.invoke({"text": "y"})              # args diferentes: executa
    assert calls["n"] == 4                      # +1 (x) +1 (y)
    tool.invoke({"text": "x"})
    assert calls["n"] == 5                      # cache morreu com o contexto


def test_memo_opt_out_per_tool():
    calls = {"n": 0}
    def probe(text: str) -> str:
        calls["n"] += 1
        return "ok"
    [tool] = coerce_tools([make_agenttool(probe, memoize=False)])
    with tool_call_cache():
        tool.invoke({"text": "x"}); tool.invoke({"text": "x"})
    assert calls["n"] == 2


def test_memo_does_not_cache_errors():
    calls = {"n": 0}
    def flaky(text: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("falhou")
        return "ok"
    [tool] = coerce_tools([make_agenttool(flaky)])
    with tool_call_cache():
        first = tool.invoke({"text": "x"})
        second = tool.invoke({"text": "x"})
    assert "falhou" in first
    assert second == "ok"                       # o erro NÃO ficou cacheado


def test_memo_async_path_shares_cache_with_sync_key():
    calls = {"n": 0}
    async def probe(text: str) -> str:
        calls["n"] += 1
        return "ok"
    [tool] = coerce_tools([make_agenttool(coroutine=probe)])
    async def run():
        with tool_call_cache():
            await tool.ainvoke({"text": "x"})
            await tool.ainvoke({"text": "x"})
    asyncio.run(run())
    assert calls["n"] == 1


def test_memo_nao_vaza_entre_tasks_concorrentes():
    calls = {"n": 0}
    async def probe(text: str) -> str:
        calls["n"] += 1
        await asyncio.sleep(0)                  # intercala as tasks
        return "ok"
    [tool] = coerce_tools([make_agenttool(coroutine=probe)])
    async def one():
        with tool_call_cache():
            await tool.ainvoke({"text": "x"})
    async def main():
        await asyncio.gather(one(), one())
    asyncio.run(main())
    assert calls["n"] == 2                      # um por task; caches isolados


def test_nested_cache_reuses_outer():
    with tool_call_cache() as outer:
        with tool_call_cache() as inner:
            assert inner is outer               # request > reflective aninhado


def test_reflective_activates_tool_cache():
    """O ReflectiveAgent abre um cache quando nenhum está ativo, para as
    rodadas do loop compartilharem resultados de tools."""
    from aixon.agents.reflective import ReflectiveAgent
    from aixon.registry import get_registry
    from tests._fakes import make_llm
    from aixon.agent import Agent

    seen: list[bool] = []

    def invoke(self, messages):
        seen.append(current_tool_cache() is not None)
        return Message(role="assistant", content="ok")

    def stream(self, messages):
        from aixon.message import Chunk
        yield Chunk(content=invoke(self, messages).content)
        yield Chunk(done=True)

    type("ProbeWorkerAgent", (Agent,),
         {"name": "probeworker", "invoke": invoke, "stream": stream})
    worker = get_registry().resolve("probeworker")
    judge = make_llm(temperature=0)
    judge.chat_model.script = [AIMessage(content="APROVADO")]
    type("ProbeReflectiveAgent", (ReflectiveAgent,), {
        "name": "probereflective", "agent": worker, "judge_llm": judge,
        "judge_rubric": "1. ok."})
    r = get_registry().resolve("probereflective")
    out = r.invoke([Message(role="user", content="q")])
    assert out.content == "ok"
    assert seen == [True]
