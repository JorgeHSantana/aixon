# tests/test_as_tool_args_schema_render.py
"""#23 — as_tool(args_schema=..., render_input=...): estrutura de briefing
para agente-como-tool. Sem args_schema, comportamento intacto (texto livre
posicional); com args_schema, a tool coercida expõe os campos do schema ao
modelo chamador (via coerce_tools -> StructuredTool.args_schema/.args) e
func/coroutine recebem **kwargs, renderizados a texto por render_input
(default: "CAMPO: valor" por linha, chave maiúscula, pulando None/""/False)
antes de seguir o MESMO fluxo _run/_arun de sempre (audience suffix DEPOIS
do render; streaming interno do #22 preservado)."""
from __future__ import annotations

import asyncio

from aixon._interop.tools import coerce_tools
from aixon.agent import _AGENT_AUDIENCE_SUFFIX
from tests._fakes import make_echo_agent

_SCHEMA = {
    "type": "object",
    "properties": {
        "objetivo": {"type": "string", "description": "O que o subagente deve fazer"},
        "contexto": {"type": "string", "description": "Contexto resolvido"},
    },
    "required": ["objetivo"],
}


# ── (a) args_schema simples: schema exposto + kwargs -> texto default ───────

def test_args_schema_expoe_campos_na_tool_coercida():
    echo = make_echo_agent("eco23a")
    tool = echo.as_tool(args_schema=_SCHEMA)
    assert tool.args_schema is _SCHEMA

    [lc_tool] = coerce_tools([tool])
    assert set(lc_tool.args) == {"objetivo", "contexto"}


def test_args_schema_kwargs_chegam_renderizados_por_default_no_subagente():
    echo = make_echo_agent("eco23b")
    tool = echo.as_tool(args_schema=_SCHEMA)

    out = tool.func(objetivo="revisar contrato", contexto="cliente ACME")
    assert out == "OBJETIVO: revisar contrato\nCONTEXTO: cliente ACME"


def test_args_schema_default_pula_campos_vazios_none_false():
    echo = make_echo_agent("eco23c")
    tool = echo.as_tool(args_schema=_SCHEMA)

    out = tool.func(objetivo="revisar contrato", contexto=None)
    assert out == "OBJETIVO: revisar contrato"

    out2 = tool.func(objetivo="", contexto="cliente ACME")
    assert out2 == "CONTEXTO: cliente ACME"


def test_args_schema_async_path_kwargs_renderizados():
    echo = make_echo_agent("eco23d")
    tool = echo.as_tool(args_schema=_SCHEMA)

    out = asyncio.run(tool.coroutine(objetivo="mapear riscos", contexto="Q3"))
    assert out == "OBJETIVO: mapear riscos\nCONTEXTO: Q3"


def test_args_schema_via_coerce_tools_chamada_estruturada():
    echo = make_echo_agent("eco23e")
    tool = echo.as_tool(args_schema=_SCHEMA)
    [lc_tool] = coerce_tools([tool])

    out = lc_tool.invoke({"objetivo": "consultar saldo", "contexto": "conta 42"})
    assert out == "OBJETIVO: consultar saldo\nCONTEXTO: conta 42"


# ── (b) render_input custom aplicado ─────────────────────────────────────────

def test_render_input_custom_substitui_o_default():
    echo = make_echo_agent("eco23f")

    def render(kwargs: dict) -> str:
        return f"Por favor: {kwargs.get('objetivo')} (ctx={kwargs.get('contexto')})"

    tool = echo.as_tool(args_schema=_SCHEMA, render_input=render)
    out = tool.func(objetivo="fechar caixa", contexto="loja 9")
    assert out == "Por favor: fechar caixa (ctx=loja 9)"


def test_render_input_custom_no_caminho_async():
    echo = make_echo_agent("eco23g")

    def render(kwargs: dict) -> str:
        return "|".join(f"{k}={v}" for k, v in kwargs.items())

    tool = echo.as_tool(args_schema=_SCHEMA, render_input=render)
    out = asyncio.run(tool.coroutine(objetivo="x", contexto="y"))
    assert out == "objetivo=x|contexto=y"


# ── (c) composição com audience="agent" — sufixo DEPOIS do render ───────────

def test_audience_agent_sufixo_aplicado_depois_do_render_default():
    echo = make_echo_agent("eco23h")
    tool = echo.as_tool(args_schema=_SCHEMA, audience="agent")

    out = tool.func(objetivo="revisar contrato", contexto="cliente ACME")
    assert out.startswith("OBJETIVO: revisar contrato\nCONTEXTO: cliente ACME")
    assert out.endswith(_AGENT_AUDIENCE_SUFFIX)


def test_audience_agent_sufixo_aplicado_depois_do_render_custom():
    echo = make_echo_agent("eco23i")

    def render(kwargs: dict) -> str:
        return f"BRIEFING: {kwargs.get('objetivo')}"

    tool = echo.as_tool(args_schema=_SCHEMA, render_input=render, audience="agent")
    out = tool.func(objetivo="mapear riscos", contexto="Q3")
    assert out == "BRIEFING: mapear riscos" + _AGENT_AUDIENCE_SUFFIX


# ── (d) sem args_schema — comportamento atual intacto ───────────────────────

def test_sem_args_schema_continua_texto_posicional():
    echo = make_echo_agent("eco23j")
    tool = echo.as_tool()
    assert tool.args_schema is None
    assert tool.func("oi") == "oi"


def test_sem_args_schema_coerce_tools_expoe_arg_text_unico():
    echo = make_echo_agent("eco23k")
    tool = echo.as_tool()
    [lc_tool] = coerce_tools([tool])
    assert set(lc_tool.args) == {"text"}
    assert lc_tool.invoke({"text": "abc"}) == "abc"
