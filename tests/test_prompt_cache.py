# tests/test_prompt_cache.py
"""#4 — aproveitamento de prompt caching entre rodadas do ReflectiveAgent.

Duas garantias:
1. O prefixo do prompt é BYTE-IDÊNTICO entre rodadas (o retry só ACRESCENTA
   mensagens ao final) — cenário que o caching automático da OpenAI já
   aproveita, desde que nada reordene/reescreva o histórico.
2. Anthropic exige breakpoints explícitos: ``LLM(..., cache=True)`` marca
   ``cache_control`` no system e na última mensagem do wire. Marcar a última
   mensagem A CADA chamada dá caching incremental entre rodadas (o breakpoint
   da rodada N vira prefixo cacheado da rodada N+1). Providers sem suporte
   (OpenAI: caching automático; fake) ignoram o knob — wire inalterado."""
from __future__ import annotations

from aixon.message import Message
from tests._fakes import make_llm
from tests.test_reflective import USER, make_reflective, make_scripted_agent


# ── 1. estabilidade do prefixo entre rodadas ─────────────────────────────────

def test_retry_prefix_is_byte_identical_between_rounds():
    gen, calls = make_scripted_agent(
        "cachegen", ["v1", "v2", "v3"])
    r = make_reflective("cacheref", gen, ["1. Ruim.", "1. Ainda ruim.", "APROVADO"])
    r.invoke(USER)
    assert len(calls) == 3
    # cada rodada é um SUPERSET-com-prefixo-intacto da anterior
    for earlier, later in zip(calls, calls[1:]):
        assert later[: len(earlier)] == earlier
        assert len(later) == len(earlier) + 2   # + assistant + crítica


# ── 2. cache_control no wire (Anthropic) ─────────────────────────────────────

MSGS = [
    Message(role="system", content="Você é um assistente."),
    Message(role="user", content="primeira"),
    Message(role="assistant", content="resposta"),
    Message(role="user", content="segunda"),
]


def _has_cache_control(lc_message) -> bool:
    content = lc_message.content
    if not isinstance(content, list):
        return False
    return any(isinstance(b, dict) and b.get("cache_control") for b in content)


def test_anthropic_cache_marks_system_and_last_message():
    from aixon.llm import LLM

    llm = LLM("claude-sonnet-4-5", cache=True)
    wire = llm._to_wire(MSGS)
    assert _has_cache_control(wire[0])          # system
    assert _has_cache_control(wire[-1])         # última mensagem (breakpoint)
    assert not _has_cache_control(wire[1])      # meio intacto
    assert not _has_cache_control(wire[2])
    # o texto sobrevive à conversão para blocos
    assert wire[0].content[0]["text"] == "Você é um assistente."


def test_anthropic_cache_off_by_default():
    from aixon.llm import LLM

    wire = LLM("claude-sonnet-4-5")._to_wire(MSGS)
    assert all(not _has_cache_control(m) for m in wire)


def test_cache_knob_is_noop_without_provider_support():
    # fake provider não declara supports_prompt_cache → wire inalterado
    wire = make_llm(cache=True)._to_wire(MSGS)
    assert all(isinstance(m.content, str) for m in wire)
