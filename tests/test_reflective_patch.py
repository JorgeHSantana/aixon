# tests/test_reflective_patch.py
"""#7 — modo de revisão por diff/patch no ReflectiveAgent (opt-in).

Com ``revision_mode = "patch"``, o retry pede blocos SEARCH/REPLACE em vez da
resposta inteira; o agente aplica as edições sobre a resposta anterior e
submete o RESULTADO ao próximo julgamento. Patch que não casa (ou resposta sem
blocos) → fallback para a regeneração completa (o modo "full" de sempre).
Default ``revision_mode = "full"``: comportamento byte-idêntico ao anterior."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from aixon.agents.reflective import ReflectiveAgent
from aixon.exceptions import AixonError
from aixon.message import Message
from aixon.registry import get_registry
from tests._fakes import make_llm
from tests.test_reflective import USER, make_scripted_agent

PATCH_OK = (
    "<<<<<<< SEARCH\n"
    "Fortaleza é a capital.\n"
    "=======\n"
    "Fortaleza é a capital do Ceará (fonte: IBGE).\n"
    ">>>>>>> REPLACE"
)

PATCH_NO_MATCH = (
    "<<<<<<< SEARCH\n"
    "trecho que não existe na resposta\n"
    "=======\n"
    "qualquer coisa\n"
    ">>>>>>> REPLACE"
)


def make_patch_reflective(name: str, agent, judge_script: list[str],
                          *, rounds: int = 3, mode: str = "patch"):
    judge = make_llm(temperature=0)
    judge.chat_model.script = [AIMessage(content=v) for v in judge_script]
    type(f"{name.capitalize()}Agent", (ReflectiveAgent,), {
        "name": name, "agent": agent, "judge_llm": judge,
        "judge_rubric": "1. A resposta cita a fonte.",
        "max_rounds": rounds, "revision_mode": mode,
    })
    return get_registry().resolve(name)


# ── validação / helpers ──────────────────────────────────────────────────────

def test_default_mode_is_full():
    assert ReflectiveAgent.revision_mode == "full"


def test_invalid_revision_mode_raises():
    gen, _ = make_scripted_agent("pgen0", ["x"])
    with pytest.raises(AixonError):
        make_patch_reflective("pref0", gen, ["APROVADO"], mode="inline")


def test_parse_and_apply_patches():
    parse = ReflectiveAgent._parse_patches
    apply = ReflectiveAgent._apply_patches
    patches = parse(PATCH_OK)
    assert patches == [("Fortaleza é a capital.",
                        "Fortaleza é a capital do Ceará (fonte: IBGE).")]
    assert apply("Fortaleza é a capital.", patches) == \
        "Fortaleza é a capital do Ceará (fonte: IBGE)."
    assert parse("sem blocos aqui") == []
    assert apply("outro texto", patches) is None          # search não casa
    assert apply("qualquer", []) is None                  # sem blocos = falha


# ── invoke: patch aplicado ───────────────────────────────────────────────────

def test_patch_mode_applies_edit_and_approves():
    gen, calls = make_scripted_agent("pgen1", ["Fortaleza é a capital.", PATCH_OK])
    r = make_patch_reflective("pref1", gen, ["1. Falta citar a fonte.", "APROVADO"])
    out = r.invoke(USER)
    assert out.content == "Fortaleza é a capital do Ceará (fonte: IBGE)."
    assert len(calls) == 2
    # a chamada de retry pediu blocos SEARCH/REPLACE, não a resposta completa
    retry_texts = [m.content for m in calls[1]]
    assert any("SEARCH" in t for t in retry_texts)
    assert not any("completa e corrigida" in t for t in retry_texts)


# ── invoke: fallback para regeneração completa ──────────────────────────────

def test_patch_that_does_not_match_falls_back_to_full_regeneration():
    gen, calls = make_scripted_agent(
        "pgen2",
        ["Fortaleza é a capital.", PATCH_NO_MATCH, "Fortaleza (fonte: IBGE)."])
    r = make_patch_reflective("pref2", gen, ["1. Falta citar a fonte.", "APROVADO"])
    out = r.invoke(USER)
    assert out.content == "Fortaleza (fonte: IBGE)."
    assert len(calls) == 3                       # tentativa + patch + fallback
    fallback_texts = [m.content for m in calls[2]]
    assert any("completa e corrigida" in t for t in fallback_texts)


# ── stream: patch aplicado, texto de patch nunca vira content ────────────────

def test_patch_mode_stream_emits_only_applied_answer():
    from aixon.message import Chunk

    gen, _ = make_scripted_agent("pgen3", ["Fortaleza é a capital.", PATCH_OK])
    r = make_patch_reflective("pref3", gen, ["1. Falta citar a fonte.", "APROVADO"])
    chunks = list(r.stream(USER))
    content = "".join(c.content or "" for c in chunks)
    assert content == "Fortaleza é a capital do Ceará (fonte: IBGE)."
    assert "SEARCH" not in content               # patch cru jamais vaza
    assert isinstance(chunks[-1], Chunk) and chunks[-1].done
