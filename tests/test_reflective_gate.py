# tests/test_reflective_gate.py
"""#14 — gate should_judge: pular o juiz em respostas triviais."""
from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage

from aixon.message import Message
from tests.test_reflective import make_scripted_agent
from tests._fakes import make_llm

from aixon.agents.reflective import ReflectiveAgent

USER = [Message(role="user", content="bom dia")]


def make_gated(name: str, agent, *, threshold: int):
    judge = make_llm(temperature=0)
    judge.chat_model.script = [AIMessage(content="APROVADO")]

    def should_judge(self, messages, answer):
        return len(answer.content) > threshold

    cls = type(f"{name.capitalize()}Agent", (ReflectiveAgent,), {
        "name": name, "agent": agent, "judge_llm": judge,
        "judge_rubric": "1. Cita fonte.", "should_judge": should_judge,
    })
    from aixon.registry import get_registry
    return get_registry().resolve(name), judge


def test_gate_falso_nao_chama_o_juiz():
    gen, calls = make_scripted_agent("g14a", ["Bom dia!"])
    r, judge = make_gated("r14a", gen, threshold=100)
    out = r.invoke(USER)
    assert out.content == "Bom dia!"
    assert len(calls) == 1                      # worker rodou 1x, sem retry
    assert judge.chat_model._idx == 0           # juiz NUNCA foi chamado


def test_gate_verdadeiro_julga_normalmente():
    gen, calls = make_scripted_agent("g14b", ["Fortaleza (fonte: IBGE)."])
    r, judge = make_gated("r14b", gen, threshold=3)
    out = r.invoke(USER)
    assert out.content == "Fortaleza (fonte: IBGE)."
    assert judge.chat_model._idx == 1           # juiz rodou


def test_gate_no_stream_e_async():
    gen, _ = make_scripted_agent("g14c", ["Oi!"])
    r, judge = make_gated("r14c", gen, threshold=100)
    chunks = list(r.stream(USER))
    assert any(c.content == "Oi!" for c in chunks)
    assert chunks[-1].done
    assert judge.chat_model._idx == 0

    gen2, _ = make_scripted_agent("g14d", ["Oi de novo!"])
    r2, judge2 = make_gated("r14d", gen2, threshold=100)
    out = asyncio.run(r2.ainvoke(USER))
    assert out.content == "Oi de novo!"
    assert judge2.chat_model._idx == 0
