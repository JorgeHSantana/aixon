# tests/test_reflective_audit.py
"""Achados da varredura no ReflectiveAgent (Task 6.6): um teste por achado."""
from __future__ import annotations

import asyncio

import pytest

from aixon.agent import Agent
from aixon.agents.reflective import (
    _PATCH_RETRY_TEMPLATE, _RETRY_TEMPLATE, ReflectiveAgent,
)
from aixon.exceptions import CompositionCycleError
from aixon.message import Chunk, Message
from aixon.registry import get_registry
from aixon.runtime import current_prediction
from aixon.toolcache import current_tool_cache
from tests._fakes import make_llm
from tests.test_reflective import USER, make_reflective, make_scripted_agent


def _streaming_worker(name: str, *, tool_call_chunks: list[list[dict]] | None = None):
    """Worker cujo stream emite um reasoning antes do conteúdo (ponto de
    abandono) e, opcionalmente, tool_calls em vários chunks."""
    calls: list[list[Message]] = []

    def invoke(self, messages):
        calls.append(list(messages))
        return Message(role="assistant", content=f"v{len(calls)}")

    def stream(self, messages):
        calls.append(list(messages))
        yield Chunk(reasoning="pensando")
        if tool_call_chunks:
            for tc in tool_call_chunks:
                yield Chunk(tool_calls=tc)
        else:
            yield Chunk(content=f"v{len(calls)}")
        yield Chunk(done=True)

    async def astream(self, messages):
        for c in stream(self, messages):
            yield c

    type(f"{name.capitalize()}Agent", (Agent,), {
        "name": name, "invoke": invoke, "stream": stream, "astream": astream})
    return get_registry().resolve(name), calls


# ── reflective-1 ─────────────────────────────────────────────────────────────

def test_reflective_1():
    # um único walk, em Agent, com a mensagem unificada do Orchestrator
    from aixon.agents.orchestrator import Orchestrator
    walk = Agent.__dict__["_check_composition_cycle"].__func__
    assert "_check_composition_cycle" not in ReflectiveAgent.__dict__
    assert "_check_composition_cycle" not in Orchestrator.__dict__
    assert ReflectiveAgent._check_composition_cycle.__func__ is walk

    gen, _ = make_scripted_agent("aud-gen1", ["a"])
    a = make_reflective("aud-ciclo-a", gen, ["APROVADO"])
    b = make_reflective("aud-ciclo-b", type(a), ["APROVADO"])
    type(a).agent = type(b)
    with pytest.raises(CompositionCycleError, match="Break the cycle"):
        type("AudCicloCAgent", (ReflectiveAgent,), {
            "name": "aud-ciclo-c", "agent": type(b),
            "judge_llm": make_llm(), "judge_rubric": "x"})


# ── reflective-2 ─────────────────────────────────────────────────────────────

def test_reflective_2():
    gen, _ = make_scripted_agent("aud-gen2", ["a"])
    r = make_reflective("aud-ref2", gen, ["APROVADO"])
    answer = Message(role="assistant", content="resp")
    for template in (_RETRY_TEMPLATE, _PATCH_RETRY_TEMPLATE):
        out = r._retry_messages(USER, answer, "crit", template=template)
        assert [(m.role, m.content) for m in out] == [
            ("user", USER[0].content), ("assistant", "resp"),
            ("user", template.format(critique="crit"))]
    assert r._retry_messages(USER, answer, "crit") == r._retry_messages(
        USER, answer, "crit", template=_RETRY_TEMPLATE)
    assert not hasattr(r, "_patch_retry_messages")


# ── reflective-3 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("entry", ["invoke", "ainvoke", "stream", "astream"])
def test_reflective_3(entry, caplog):
    # igualdade de comportamento: juiz caído e rodadas esgotadas dão, nos 4
    # caminhos, a mesma linha de run e a mesma resposta
    import logging
    caplog.set_level(logging.INFO)
    gen, _ = make_scripted_agent(f"aud-gen3-{entry}", ["a", "b"])
    r = make_reflective(f"aud-ref3-{entry}", gen, ["não", "não"], rounds=2)

    def run():
        if entry == "invoke":
            return r.invoke(USER).content
        if entry == "ainvoke":
            return asyncio.run(r.ainvoke(USER)).content
        if entry == "stream":
            return "".join(c.content or "" for c in r.stream(USER))

        async def go():
            return "".join([c.content or "" async for c in r.astream(USER)])
        return asyncio.run(go())

    assert run() == "b"
    lines = [m for m in caplog.messages if "reflective_run" in m]
    assert len(lines) == 1 and "outcome=exhausted" in lines[0] and "rounds=2" in lines[0]
    assert any("rounds exhausted" in m for m in caplog.messages)


# ── reflective-4 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("entry", ["invoke", "ainvoke", "stream", "astream"])
def test_reflective_4(entry, monkeypatch):
    # erro de programação na montagem do prompt do juiz não é "juiz caído"
    gen, _ = make_scripted_agent(f"aud-gen4-{entry}", ["a"])
    r = make_reflective(f"aud-ref4-{entry}", gen, ["APROVADO"])

    def quebrado(*a, **k):
        raise KeyError("rubrica")
    monkeypatch.setattr(r, "_judge_messages", quebrado)

    with pytest.raises(KeyError):
        if entry == "invoke":
            r.invoke(USER)
        elif entry == "ainvoke":
            asyncio.run(r.ainvoke(USER))
        elif entry == "stream":
            list(r.stream(USER))
        else:
            async def go():
                return [c async for c in r.astream(USER)]
            asyncio.run(go())


# ── reflective-5 ─────────────────────────────────────────────────────────────

def test_reflective_5_abandoned_stream_does_not_leak_tool_cache():
    gen, _ = _streaming_worker("aud-gen5a")
    r = make_reflective("aud-ref5a", gen, ["APROVADO"])
    g = r.stream(USER)
    seen = []
    for chunk in g:
        seen.append(current_tool_cache())
        if chunk.reasoning:
            break  # abandonado, sem close(); a referência segue viva
    assert seen == [None]
    assert current_tool_cache() is None
    del g


def test_reflective_5_abandoned_astream_does_not_leak_prediction():
    gen, _ = _streaming_worker("aud-gen5b")
    r = make_reflective("aud-ref5b", gen, ["1. falta fonte", "APROVADO"])

    async def go():
        reasonings = 0
        async for chunk in r.astream(USER):
            if chunk.reasoning == "pensando":
                reasonings += 1
                if reasonings == 2:  # dentro do stream do worker na rodada 2
                    break  # abandonado: sem aclose()
        right_after = current_prediction()
        await asyncio.sleep(0.05)
        return right_after, current_prediction(), current_tool_cache()

    right_after, later, cache = asyncio.run(go())
    assert right_after is None and later is None and cache is None


# ── reflective-6 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("entry", ["stream", "astream"])
def test_reflective_6(entry):
    tc1 = [{"name": "a", "args": {}, "id": "c1"}]
    tc2 = [{"name": "b", "args": {}, "id": "c2"}]
    gen, _ = _streaming_worker(f"aud-gen6-{entry}", tool_call_chunks=[tc1, tc2])
    r = make_reflective(f"aud-ref6-{entry}", gen, ["APROVADO"])
    if entry == "stream":
        chunks = list(r.stream(USER))
    else:
        async def go():
            return [c async for c in r.astream(USER)]
        chunks = asyncio.run(go())
    assert [c.tool_calls for c in chunks if c.tool_calls] == [tc1 + tc2]
