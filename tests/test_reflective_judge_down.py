# tests/test_reflective_judge_down.py
"""#24 — judge_llm outage degrades gracefully: the worker's current answer is
delivered unreviewed instead of crashing the run. Covers all 4 neutral call
paths (_invoke/_stream/_ainvoke/_astream) and both failure points: the judge
raising on round 1, and raising on a RETRY round (after rejecting round 1)."""
from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aixon.message import Message
from aixon.reasoning import reasoning_channel
from tests._fakes import FakeChatModel, make_llm
from tests.test_reflective import make_scripted_agent
from tests.test_reflective_stats import reflective_log  # reuse the _ListHandler fixture

from aixon.agents.reflective import ReflectiveAgent

USER = [Message(role="user", content="qual a capital do Ceará?")]


# ── scriptable judge whose chat model can raise ──────────────────────────────

class _FailingChatModel(FakeChatModel):
    """FakeChatModel twin: script entries that are Exception INSTANCES are
    raised instead of returned as an AIMessage — models a judge_llm outage
    (429/5xx/timeout, whatever the provider throws) at the exact boundary
    ReflectiveAgent calls through (LLM.complete -> chat_model.invoke ->
    _generate; LLM.acomplete -> chat_model.ainvoke -> BaseChatModel's default
    thread-executor bridge to this same _generate)."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        i = self._idx
        item = self.script[i] if i < len(self.script) else AIMessage(content="(done)")
        object.__setattr__(self, "_idx", i + 1)
        if isinstance(item, BaseException):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])


def make_failing_judge(script: list):
    """LLM judge whose script mixes verdict strings (APROVADO / critique
    text, wrapped as AIMessage) and Exception instances (raised on that
    call). Calls beyond len(script) repeat the LAST entry."""
    judge = make_llm(temperature=0)
    judge._chat_model = _FailingChatModel(
        script=[v if isinstance(v, BaseException) else AIMessage(content=v)
                for v in script]
    )
    return judge


def make_reflective_judge(name: str, agent, judge, *, rounds: int = 3):
    cls = type(f"{name.capitalize()}Agent", (ReflectiveAgent,), {
        "name": name,
        "agent": agent,
        "judge_llm": judge,
        "judge_rubric": "1. A resposta cita a fonte.",
        "max_rounds": rounds,
    })
    from aixon.registry import get_registry
    return get_registry().resolve(name)


def _run_line(lines: list[str]) -> str:
    matches = [l for l in lines if l.startswith("reflective_run ")]
    assert len(matches) == 1, f"esperava 1 linha reflective_run, veio {len(matches)}"
    return matches[0]


# ── judge raises on ROUND 1 (the very first judge call) ──────────────────────

def test_invoke_juiz_cai_na_primeira_rodada(reflective_log):
    gen, calls = make_scripted_agent("jd-inv1-gen", ["Fortaleza."])
    judge = make_failing_judge([RuntimeError("créditos esgotados")])
    r = make_reflective_judge("jd-inv1", gen, judge)
    with reasoning_channel() as ch:
        out = r.invoke(USER)
        lines = ch.drain()
    assert out.content == "Fortaleza."          # worker answer entregue intacto
    assert len(calls) == 1                       # worker rodou só uma vez
    assert r.judge_unavailable_label in lines     # label no canal
    line = _run_line(reflective_log)
    assert "outcome=judge_error" in line and "rounds=1" in line
    warn = [l for l in reflective_log if "judge unavailable" in l]
    assert len(warn) == 1 and "créditos esgotados" in warn[0]


def test_stream_juiz_cai_na_primeira_rodada(reflective_log):
    gen, calls = make_scripted_agent("jd-str1-gen", ["Fortaleza."])
    judge = make_failing_judge([RuntimeError("créditos esgotados")])
    r = make_reflective_judge("jd-str1", gen, judge)
    chunks = list(r.stream(USER))
    reasoning = "".join(c.reasoning for c in chunks)
    content = "".join(c.content for c in chunks)
    assert content == "Fortaleza."
    assert len(calls) == 1
    assert r.judge_unavailable_label in reasoning
    assert chunks[-1].done is True
    line = _run_line(reflective_log)
    assert "outcome=judge_error" in line and "rounds=1" in line


def test_ainvoke_juiz_cai_na_primeira_rodada(reflective_log):
    gen, calls = make_scripted_agent("jd-ainv1-gen", ["Fortaleza."])
    judge = make_failing_judge([RuntimeError("créditos esgotados")])
    r = make_reflective_judge("jd-ainv1", gen, judge)
    with reasoning_channel() as ch:
        out = asyncio.run(r.ainvoke(USER))
        lines = ch.drain()
    assert out.content == "Fortaleza."
    assert len(calls) == 1
    assert r.judge_unavailable_label in lines
    line = _run_line(reflective_log)
    assert "outcome=judge_error" in line and "rounds=1" in line


def test_astream_juiz_cai_na_primeira_rodada(reflective_log):
    gen, calls = make_scripted_agent("jd-astr1-gen", ["Fortaleza."])
    judge = make_failing_judge([RuntimeError("créditos esgotados")])
    r = make_reflective_judge("jd-astr1", gen, judge)

    async def run():
        return [c async for c in r.astream(USER)]

    chunks = asyncio.run(run())
    reasoning = "".join(c.reasoning for c in chunks)
    content = "".join(c.content for c in chunks)
    assert content == "Fortaleza."
    assert len(calls) == 1
    assert r.judge_unavailable_label in reasoning
    assert chunks[-1].done is True
    line = _run_line(reflective_log)
    assert "outcome=judge_error" in line and "rounds=1" in line


# ── judge rejects round 1, then raises on the RETRY round's judge call ───────

def test_invoke_juiz_cai_na_rodada_de_retry(reflective_log):
    gen, calls = make_scripted_agent(
        "jd-inv2-gen", ["Fortaleza.", "Fortaleza (fonte: IBGE)."])
    judge = make_failing_judge(
        ["1. Falta citar a fonte.", RuntimeError("timeout")])
    r = make_reflective_judge("jd-inv2", gen, judge)
    with reasoning_channel() as ch:
        out = r.invoke(USER)
        lines = ch.drain()
    # entrega o MELHOR answer que houver: a resposta da rodada 2 (a que
    # estava em avaliação quando o juiz caiu), não a v1 rejeitada.
    assert out.content == "Fortaleza (fonte: IBGE)."
    assert len(calls) == 2
    assert r.judge_unavailable_label in lines
    line = _run_line(reflective_log)
    assert "outcome=judge_error" in line and "rounds=2" in line


def test_stream_juiz_cai_na_rodada_de_retry(reflective_log):
    gen, calls = make_scripted_agent(
        "jd-str2-gen", ["Fortaleza.", "Fortaleza (fonte: IBGE)."])
    judge = make_failing_judge(
        ["1. Falta citar a fonte.", RuntimeError("timeout")])
    r = make_reflective_judge("jd-str2", gen, judge)
    chunks = list(r.stream(USER))
    reasoning = "".join(c.reasoning for c in chunks)
    content = "".join(c.content for c in chunks)
    assert content == "Fortaleza (fonte: IBGE)."
    assert len(calls) == 2
    assert r.judge_unavailable_label in reasoning
    assert chunks[-1].done is True
    line = _run_line(reflective_log)
    assert "outcome=judge_error" in line and "rounds=2" in line


def test_ainvoke_juiz_cai_na_rodada_de_retry(reflective_log):
    gen, calls = make_scripted_agent(
        "jd-ainv2-gen", ["Fortaleza.", "Fortaleza (fonte: IBGE)."])
    judge = make_failing_judge(
        ["1. Falta citar a fonte.", RuntimeError("timeout")])
    r = make_reflective_judge("jd-ainv2", gen, judge)
    with reasoning_channel() as ch:
        out = asyncio.run(r.ainvoke(USER))
        lines = ch.drain()
    assert out.content == "Fortaleza (fonte: IBGE)."
    assert len(calls) == 2
    assert r.judge_unavailable_label in lines
    line = _run_line(reflective_log)
    assert "outcome=judge_error" in line and "rounds=2" in line


def test_astream_juiz_cai_na_rodada_de_retry(reflective_log):
    gen, calls = make_scripted_agent(
        "jd-astr2-gen", ["Fortaleza.", "Fortaleza (fonte: IBGE)."])
    judge = make_failing_judge(
        ["1. Falta citar a fonte.", RuntimeError("timeout")])
    r = make_reflective_judge("jd-astr2", gen, judge)

    async def run():
        return [c async for c in r.astream(USER)]

    chunks = asyncio.run(run())
    reasoning = "".join(c.reasoning for c in chunks)
    content = "".join(c.content for c in chunks)
    assert content == "Fortaleza (fonte: IBGE)."
    assert len(calls) == 2
    assert r.judge_unavailable_label in reasoning
    assert chunks[-1].done is True
    line = _run_line(reflective_log)
    assert "outcome=judge_error" in line and "rounds=2" in line


# ── no regression: a healthy judge behaves exactly as before ─────────────────

def test_juiz_saudavel_sem_mudanca(reflective_log):
    gen, calls = make_scripted_agent("jd-ok-gen", ["Fortaleza (fonte: IBGE)."])
    judge = make_failing_judge(["APROVADO"])
    r = make_reflective_judge("jd-ok", gen, judge)
    out = r.invoke(USER)
    assert out.content == "Fortaleza (fonte: IBGE)."
    line = _run_line(reflective_log)
    assert "outcome=approved" in line
    assert not any("judge unavailable" in l for l in reflective_log)
