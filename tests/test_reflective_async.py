# tests/test_reflective_async.py
"""Native async parity for ReflectiveAgent (repo convention: sync tests,
async exercised via asyncio.run)."""
from __future__ import annotations

import asyncio

from tests.test_reflective import USER, make_reflective, make_scripted_agent


def test_ainvoke_reprova_e_aprova():
    gen, calls = make_scripted_agent("agen1", ["v1", "v2 (fonte: IBGE)"])
    r = make_reflective("aref1", gen, ["1. Falta fonte.", "APROVADO"])
    out = asyncio.run(r.ainvoke(USER))
    assert out.content == "v2 (fonte: IBGE)"
    assert len(calls) == 2


def test_ainvoke_e_nativo():
    # O método deve estar definido na classe (não herdado da ponte da base).
    from aixon.agents.reflective import ReflectiveAgent
    assert "ainvoke" in ReflectiveAgent.__dict__
    assert "astream" in ReflectiveAgent.__dict__


def test_astream_conteudo_e_done():
    gen, _ = make_scripted_agent("agen2", ["final"])
    r = make_reflective("aref2", gen, ["APROVADO"])

    async def run():
        return [c async for c in r.astream(USER)]

    chunks = asyncio.run(run())
    assert "".join(c.content for c in chunks) == "final"
    assert chunks[-1].done is True
    assert r.judge_label in "".join(c.reasoning for c in chunks)
