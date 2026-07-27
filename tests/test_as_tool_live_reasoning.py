# tests/test_as_tool_live_reasoning.py
"""#22 — AgentTool._run/_arun consomem stream()/astream() do agente
embrulhado (não invoke()/ainvoke()), re-emitindo cada Chunk.reasoning no
canal ATIVO (o do chamador) ao vivo, e acumulando Chunk.content no retorno.

Critério de aceite (issue #22):
  (a) timestamps — reasoning de um subagente aninhado aparece no stream do
      PAI antes do subagente completar (idioma de test_astream_live_drain.py).
  (b) retorno byte-idêntico ao invoke() direto de hoje.
  (c) defensivo preservado — Chunk(tool_calls) sem content vira o texto
      explicativo + warning (idioma de test_as_tool.py).
  (d) audience="agent" continua anexando a moldura de subagente (idioma de
      test_as_tool_audience.py).
  (e) memoização — cache hit não roda de novo, logo não re-emite reasoning.
"""
from __future__ import annotations

import asyncio
import logging
import time

import pytest
from langchain_core.messages import AIMessage

from aixon._interop.tools import coerce_tools
from aixon.agent import Agent, _AGENT_AUDIENCE_SUFFIX
from aixon.agents.tool_agent import ToolAgent
from aixon.message import Chunk, Message
from aixon.reasoning import current_channel, emit_reasoning, reasoning_channel
from aixon.registry import get_registry
from aixon.toolcache import tool_call_cache
from tests._fakes import make_echo_agent, make_llm


# ── (a) timestamps: reasoning do subagente aninhado flui AO VIVO ────────────

def test_as_tool_astream_reasoning_flui_ao_vivo_do_subagente_aninhado():
    """Um ToolAgent (child) com uma tool lenta que emite reasoning e demora
    ~1s é exposto via as_tool() como tool de outro ToolAgent (parent). O
    reasoning interno da tool lenta deve aparecer no astream() do PAI
    ENQUANTO a tool ainda está rodando — não só depois que ela termina."""
    events: list[tuple[str, float]] = []

    async def lenta(x: str) -> str:
        """tool lenta que reporta progresso"""
        emit_reasoning("progresso interno do subagente")
        await asyncio.sleep(1.0)
        events.append(("tool_end", time.monotonic()))
        return "ok"

    child_llm = make_llm()
    child_llm.chat_model.script = [
        AIMessage(content="", tool_calls=[{"name": "lenta", "args": {"x": "1"}, "id": "c1"}]),
        AIMessage(content="child fim"),
    ]

    class Child22Agent(ToolAgent):
        name = "child22"
        llm = child_llm
        tools = [lenta]

    child = get_registry().resolve("child22")

    parent_llm = make_llm()
    parent_llm.chat_model.script = [
        AIMessage(content="", tool_calls=[{"name": "child22", "args": {"text": "vai"}, "id": "p1"}]),
        AIMessage(content="parent fim"),
    ]

    class Parent22Agent(ToolAgent):
        name = "parent22"
        llm = parent_llm
        tools = [child.as_tool()]

    parent = get_registry().resolve("parent22")

    async def run():
        async for ch in parent.astream([Message(role="user", content="vai")]):
            if ch.reasoning and "progresso interno do subagente" in ch.reasoning:
                events.append(("chunk_interno", time.monotonic()))

    asyncio.run(run())

    interno = next(t for n, t in events if n == "chunk_interno")
    fim = next(t for n, t in events if n == "tool_end")
    assert interno < fim - 0.3, (
        f"reasoning interno do subagente só chegou ao pai {interno - fim:+.2f}s "
        f"em relação ao fim da tool lenta — deveria fluir DURANTE, via as_tool "
        f"consumindo stream() (#22)")


# ── (b) retorno byte-idêntico ao invoke() direto ─────────────────────────────

def test_as_tool_retorno_byte_identico_ao_invoke_direto():
    _llm = make_llm()
    _llm.chat_model.script = [
        AIMessage(content="", tool_calls=[{"name": "helper_tool", "args": {"x": "1"}, "id": "c1"}]),
        AIMessage(content="resposta final determinística"),
    ]

    def helper_tool(x: str) -> str:
        """helper"""
        return "helper-ok"

    class Byte22Agent(ToolAgent):
        name = "byte22"
        llm = _llm
        tools = [helper_tool]

    agent = get_registry().resolve("byte22")

    direct = agent.invoke([Message(role="user", content="oi")]).content
    assert direct == "resposta final determinística"

    # Replay the SAME scripted model from the top so the as_tool() path sees
    # an identical run (same tool call, same final answer).
    object.__setattr__(_llm.chat_model, "_idx", 0)

    tool = agent.as_tool()
    via_tool = tool.func("oi")

    assert via_tool == direct


def test_as_tool_arun_retorno_byte_identico_ao_ainvoke_direto():
    _llm = make_llm()
    _llm.chat_model.script = [AIMessage(content="resposta async determinística")]

    class ByteAsync22Agent(ToolAgent):
        name = "byteasync22"
        llm = _llm
        tools = []

    agent = get_registry().resolve("byteasync22")

    direct = asyncio.run(agent.ainvoke([Message(role="user", content="oi")])).content
    assert direct == "resposta async determinística"

    object.__setattr__(_llm.chat_model, "_idx", 0)

    tool = agent.as_tool()
    via_tool = asyncio.run(tool.coroutine("oi"))

    assert via_tool == direct


# ── (c) defensivo preservado: tool_calls sem content ─────────────────────────

def _leaky_stream_agent(name: str):
    """Subagente cujo stream() emite Chunk(tool_calls=...) sem content — o
    mesmo formato que um worker client_tools="merge" produz (#18c) quando o
    modelo chama uma ferramenta do CLIENTE. as_tool() não tem canal para
    relayar tool_calls de volta a quem chamou a tool."""
    calls = [{"name": "inserir_no_documento", "args": {"texto": "x"}, "id": "c1"}]

    def stream(self, messages):
        yield Chunk(tool_calls=calls)
        yield Chunk(done=True)

    async def astream(self, messages):
        yield Chunk(tool_calls=calls)
        yield Chunk(done=True)

    return type(
        name.capitalize() + "Agent", (Agent,),
        {
            "name": name,
            "invoke": lambda self, messages: Message(role="assistant", content="", tool_calls=calls),
            "stream": stream,
            "astream": astream,
        },
    )


def test_as_tool_defensivo_stream_tool_calls_sem_content_avisa_e_nao_vaza_vazio(caplog):
    _leaky_stream_agent("leakystream1")
    tool = get_registry().resolve("leakystream1").as_tool()

    with caplog.at_level(logging.WARNING, logger="aixon.agent"):
        out = tool.func("faça algo no documento")

    assert out != ""
    assert "inserir_no_documento" in out
    assert "as_tool" in out
    assert "leakystream1" in out
    assert any("leakystream1" in m for m in caplog.messages)


def test_as_tool_defensivo_astream_tool_calls_sem_content_tambem_avisa():
    _leaky_stream_agent("leakystream2")
    tool = get_registry().resolve("leakystream2").as_tool()

    out = asyncio.run(tool.coroutine("faça algo no documento"))

    assert out != ""
    assert "inserir_no_documento" in out
    assert "leakystream2" in out


# ── (d) audience="agent" continua anexando a moldura ─────────────────────────

def test_as_tool_audience_agent_anexa_moldura_via_stream():
    echo = make_echo_agent("eco22a")
    tool = echo.as_tool(audience="agent")
    result = tool.func("qual o total?")
    assert result.startswith("qual o total?")
    assert _AGENT_AUDIENCE_SUFFIX in result


def test_as_tool_audience_default_nao_muda_nada_via_stream():
    echo = make_echo_agent("eco22b")
    tool = echo.as_tool()
    assert tool.func("oi") == "oi"


def test_as_tool_audience_agent_via_astream():
    echo = make_echo_agent("eco22c")
    tool = echo.as_tool(audience="agent")
    result = asyncio.run(tool.coroutine("qual o total?"))
    assert _AGENT_AUDIENCE_SUFFIX in result


# ── (e) memoização: cache hit não re-emite reasoning ─────────────────────────

def test_as_tool_cache_hit_nao_reemite_reasoning():
    """O cache de tool calls (aixon.toolcache) memoiza o RETORNO servindo-o
    ANTES de _run rodar de novo — logo uma segunda chamada idêntica sob
    tool_call_cache não invoca self.stream() e não produz reasoning novo
    (silêncio no cache hit é o comportamento correto/documentado)."""
    calls = {"n": 0}

    def stream(self, messages):
        calls["n"] += 1
        emit_reasoning(f"trabalhando ({calls['n']}x)")
        yield Chunk(content=f"r{calls['n']}")
        yield Chunk(done=True)

    type(
        "Cache22Agent", (Agent,),
        {
            "name": "cache22",
            "invoke": lambda self, messages: Message(role="assistant", content="r"),
            "stream": stream,
        },
    )
    agent = get_registry().resolve("cache22")
    tool = agent.as_tool()
    [lc_tool] = coerce_tools([tool])

    with reasoning_channel() as channel:
        with tool_call_cache():
            first = lc_tool.invoke({"text": "x"})
            first_lines = channel.drain()
            second = lc_tool.invoke({"text": "x"})  # cache hit
            second_lines = channel.drain()

    assert calls["n"] == 1  # stream() ran exactly once
    assert first == "r1"
    assert second == "r1"  # served from cache, byte-identical
    assert any("trabalhando" in l for l in first_lines)
    assert second_lines == []  # cache hit: total silence, no new reasoning
