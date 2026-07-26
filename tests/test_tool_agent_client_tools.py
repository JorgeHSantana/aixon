# tests/test_tool_agent_client_tools.py
"""#18c — merge de client tools no ToolAgent: interna executa, do cliente borbulha."""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage

from aixon.agents.tool_agent import ToolAgent
from aixon.exceptions import AixonError
from aixon.message import Message
from aixon.runtime import client_tools
from tests._fakes import make_llm

DOC_TOOL = [{
    "type": "function",
    "function": {"name": "inserir_no_documento",
                 "description": "Insere texto no documento do usuário.",
                 "parameters": {"type": "object",
                                "properties": {"texto": {"type": "string"}},
                                "required": ["texto"]}},
}]
USER = [Message(role="user", content="soma 2+3 e insere no doc")]


def soma(a: int, b: int) -> str:
    """Soma dois inteiros."""
    return str(a + b)


def _merge_agent(name: str, script, **extra):
    llm = make_llm()
    llm.chat_model.script = script
    cls = type(f"{name.capitalize()}Agent", (ToolAgent,), {
        "name": name, "llm": llm, "tools": [soma],
        "client_tools": "merge", **extra,
    })
    from aixon.registry import get_registry
    return get_registry().resolve(name)


def test_interna_executa_e_call_do_cliente_borbulha():
    script = [
        AIMessage(content="", tool_calls=[
            {"name": "soma", "args": {"a": 2, "b": 3}, "id": "c1"}]),
        AIMessage(content="", tool_calls=[
            {"name": "inserir_no_documento", "args": {"texto": "5"},
             "id": "c2"}]),
    ]
    agent = _merge_agent("mg18a", script)
    with client_tools(DOC_TOOL):
        out = agent.invoke(USER)
    assert out.role == "assistant"
    assert out.tool_calls == [
        {"name": "inserir_no_documento", "args": {"texto": "5"}, "id": "c2"}]


def test_ignore_default_nao_muda_nada():
    llm = make_llm()
    llm.chat_model.script = [AIMessage(content="só texto")]
    cls = type("Mg18bAgent", (ToolAgent,), {
        "name": "mg18b", "llm": llm, "tools": [soma],
    })
    from aixon.registry import get_registry
    with client_tools(DOC_TOOL):
        out = get_registry().resolve("mg18b").invoke(USER)
    assert out.content == "só texto" and not out.tool_calls


def test_conflito_de_nome_erro_explicito():
    colide = [{"type": "function", "function": {"name": "soma",
               "parameters": {"type": "object"}}}]
    agent = _merge_agent("mg18c", [AIMessage(content="x")])
    with client_tools(colide):
        with pytest.raises(AixonError, match="soma"):
            agent.invoke(USER)


def test_conflito_internal_descarta_a_do_cliente():
    colide = [{"type": "function", "function": {"name": "soma",
               "parameters": {"type": "object"}}}]
    script = [
        AIMessage(content="", tool_calls=[
            {"name": "soma", "args": {"a": 1, "b": 1}, "id": "c1"}]),
        AIMessage(content="dois"),
    ]
    agent = _merge_agent("mg18d", script, client_tools_conflict="internal")
    with client_tools(colide):
        out = agent.invoke(USER)
    assert out.content == "dois" and not out.tool_calls  # interna executou


def test_filter_curadoria():
    def client_tools_filter(self, defs):
        return []  # descarta tudo

    agent = _merge_agent("mg18e", [AIMessage(content="sem proxy")],
                         client_tools_filter=client_tools_filter)
    with client_tools(DOC_TOOL):
        out = agent.invoke(USER)
    assert out.content == "sem proxy" and not out.tool_calls


def test_replace_so_tools_do_cliente():
    script = [AIMessage(content="", tool_calls=[
        {"name": "inserir_no_documento", "args": {"texto": "oi"},
         "id": "c9"}])]
    agent = _merge_agent("mg18f", script, client_tools="replace")
    with client_tools(DOC_TOOL):
        out = agent.invoke(USER)
    assert out.tool_calls[0]["id"] == "c9"


def test_retomada_com_tool_result_do_cliente():
    # 2ª request: histórico traz a call + resultado; modelo conclui.
    script = [AIMessage(content="inserido com sucesso ✅")]
    agent = _merge_agent("mg18g", script)
    historico = [
        *USER,
        Message(role="assistant", content="", tool_calls=[
            {"name": "inserir_no_documento", "args": {"texto": "5"},
             "id": "c2"}]),
        Message(role="tool", content="ok", tool_call_id="c2"),
    ]
    with client_tools(DOC_TOOL):
        out = agent.invoke(historico)
    assert out.content == "inserido com sucesso ✅"


def test_valor_invalido_de_client_tools_erro_no_registro():
    llm = make_llm()
    with pytest.raises(AixonError, match="client_tools"):
        type("Mg18hAgent", (ToolAgent,), {
            "name": "mg18h", "llm": llm, "tools": [],
            "client_tools": "mescla",
        })


def test_turno_misto_surfaceia_call_do_cliente():
    # Turno MISTO: interna + cliente na MESMA AI message. O grafo NÃO corta
    # (return_direct só encerra quando TODAS as calls do turno são
    # return_direct), o modelo continua e responde texto — mas a call do
    # cliente não pode ser engolida: ela nunca teve resultado real (o proxy
    # devolve um sentinel), então SEMPRE borbulha; o texto pós-sentinel é
    # descartado.
    script = [
        AIMessage(content="", tool_calls=[
            {"name": "soma", "args": {"a": 2, "b": 3}, "id": "c1"},
            {"name": "inserir_no_documento", "args": {"texto": "5"},
             "id": "c2"}]),
        AIMessage(content="FINAL-TEXT"),
    ]
    agent = _merge_agent("mg18i", script)
    with client_tools(DOC_TOOL):
        out = agent.invoke(USER)
    assert out.content != "FINAL-TEXT"
    assert out.tool_calls == [
        {"name": "inserir_no_documento", "args": {"texto": "5"}, "id": "c2"}]


def test_surface_client_calls_e_funcao_pura():
    # O helper recebe as mensagens e o set de nomes como PARÂMETROS — nenhum
    # estado de request pode viver na instância (o agente é singleton no
    # registry; requests concorrentes compartilham o mesmo self).
    novas = [
        AIMessage(content="", tool_calls=[
            {"name": "inserir_no_documento", "args": {"texto": "x"},
             "id": "z1"}]),
    ]
    surfaced = ToolAgent._surface_client_calls(novas, {"inserir_no_documento"})
    assert surfaced is not None and surfaced.tool_calls[0]["id"] == "z1"
    assert ToolAgent._surface_client_calls(novas, set()) is None
    assert ToolAgent._surface_client_calls([], {"inserir_no_documento"}) is None

    # E depois de um invoke, o agente NÃO pode ter ganho atributo de
    # instância com os nomes das tools do cliente (estado request-scoped em
    # self = race entre requests concorrentes).
    agent = _merge_agent("mg18j", [AIMessage(content="", tool_calls=[
        {"name": "inserir_no_documento", "args": {"texto": "oi"},
         "id": "c3"}])])
    with client_tools(DOC_TOOL):
        agent.invoke(USER)
    assert not hasattr(agent, "_client_tool_names")


def test_surfaced_preserva_reasoning_do_run():
    # O caminho de surface (#18c) não pode perder o reasoning acumulado no
    # run (labels de tool call internas) só porque a resposta virou
    # tool_calls do cliente em vez de texto.
    script = [
        AIMessage(content="", tool_calls=[
            {"name": "soma", "args": {"a": 2, "b": 3}, "id": "c1"}]),
        AIMessage(content="", tool_calls=[
            {"name": "inserir_no_documento", "args": {"texto": "5"},
             "id": "c2"}]),
    ]
    agent = _merge_agent("mg18k", script)
    with client_tools(DOC_TOOL):
        out = agent.invoke(USER)
    assert out.tool_calls == [
        {"name": "inserir_no_documento", "args": {"texto": "5"}, "id": "c2"}]
    assert out.reasoning is not None and "soma" in out.reasoning


def test_ainvoke_surfaced_preserva_reasoning_do_run():
    # Paridade async do teste acima.
    script = [
        AIMessage(content="", tool_calls=[
            {"name": "soma", "args": {"a": 2, "b": 3}, "id": "c1"}]),
        AIMessage(content="", tool_calls=[
            {"name": "inserir_no_documento", "args": {"texto": "5"},
             "id": "c2"}]),
    ]
    agent = _merge_agent("mg18l", script)
    with client_tools(DOC_TOOL):
        out = asyncio.run(agent.ainvoke(USER))
    assert out.tool_calls == [
        {"name": "inserir_no_documento", "args": {"texto": "5"}, "id": "c2"}]
    assert out.reasoning is not None and "soma" in out.reasoning
