# tests/test_reflective_stats.py
"""#12 — log estruturado por run do ReflectiveAgent (taxa de fallback)."""
from __future__ import annotations

import logging

import pytest

from aixon.message import Message
from tests.test_reflective import make_scripted_agent, make_reflective

USER = [Message(role="user", content="qual a capital do Ceará?")]


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def reflective_log():
    # Logger do aixon usa propagate=False: anexar handler direto no nomeado.
    h = _ListHandler()
    logger = logging.getLogger("aixon.reflective")
    logger.addHandler(h)
    yield h.lines
    logger.removeHandler(h)


def _run_line(lines: list[str]) -> str:
    matches = [l for l in lines if l.startswith("reflective_run ")]
    assert len(matches) == 1, f"esperava 1 linha reflective_run, veio {len(matches)}"
    return matches[0]


def test_log_aprovado_modo_full(reflective_log):
    gen, _ = make_scripted_agent("g12a", ["Fortaleza (fonte: IBGE)."])
    r = make_reflective("r12a", gen, ["APROVADO"])
    r.invoke(USER)
    line = _run_line(reflective_log)
    assert "agent=r12a" in line and "rounds=1" in line
    assert "patch_applied=0" in line and "patch_fallback=0" in line
    assert "outcome=approved" in line


def test_log_patch_aplicado(reflective_log):
    # Rodada 1 reprova; retry em patch aplica um bloco; rodada 2 aprova.
    gen, _ = make_scripted_agent("g12b", [
        "Fortaleza.",
        "<<<<<<< SEARCH\nFortaleza.\n=======\nFortaleza (fonte: IBGE).\n>>>>>>> REPLACE",
    ])
    r = make_reflective("r12b", gen, ["1. Falta citar a fonte.", "APROVADO"])
    type(r).revision_mode = "patch"
    out = r.invoke(USER)
    assert out.content == "Fortaleza (fonte: IBGE)."
    line = _run_line(reflective_log)
    assert "patch_applied=1" in line and "patch_fallback=0" in line
    assert "outcome=approved" in line and "rounds=2" in line


def test_log_patch_fallback(reflective_log):
    # O retry devolve blocos cujo SEARCH não existe -> fallback p/ full.
    gen, _ = make_scripted_agent("g12c", [
        "Fortaleza.",
        "<<<<<<< SEARCH\nNAO EXISTE\n=======\nx\n>>>>>>> REPLACE",
        "Fortaleza (fonte: IBGE).",
    ])
    r = make_reflective("r12c", gen, ["1. Falta citar a fonte.", "APROVADO"])
    type(r).revision_mode = "patch"
    out = r.invoke(USER)
    assert out.content == "Fortaleza (fonte: IBGE)."
    line = _run_line(reflective_log)
    assert "patch_applied=0" in line and "patch_fallback=1" in line


def test_log_esgotado(reflective_log):
    gen, _ = make_scripted_agent("g12d", ["Fortaleza.", "Fortaleza?!"])
    r = make_reflective("r12d", gen, ["1. Falta fonte.", "1. Falta fonte."],
                        rounds=2)
    r.invoke(USER)
    line = _run_line(reflective_log)
    assert "outcome=exhausted" in line and "rounds=2" in line


def test_log_no_stream(reflective_log):
    gen, _ = make_scripted_agent("g12e", ["Fortaleza (fonte: IBGE)."])
    r = make_reflective("r12e", gen, ["APROVADO"])
    list(r.stream(USER))
    line = _run_line(reflective_log)
    assert "outcome=approved" in line


def test_log_no_async(reflective_log):
    import asyncio

    gen, _ = make_scripted_agent("g12f", ["Fortaleza (fonte: IBGE)."])
    r = make_reflective("r12f", gen, ["APROVADO"])
    asyncio.run(r.ainvoke(USER))
    line = _run_line(reflective_log)
    assert "outcome=approved" in line
