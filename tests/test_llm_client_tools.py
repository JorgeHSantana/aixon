# tests/test_llm_client_tools.py
"""#18a — LLM.complete(tools=...) + LLMAgent(client_tools=True)."""
from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage

from aixon.message import Message
from aixon.runtime import client_tools, tool_choice_scope
from tests._fakes import make_llm

WIRE_TOOLS = [{
    "type": "function",
    "function": {"name": "inserir_texto",
                 "description": "Insere texto no documento.",
                 "parameters": {"type": "object",
                                "properties": {"texto": {"type": "string"}},
                                "required": ["texto"]}},
}]
CALL = AIMessage(content="", tool_calls=[
    {"name": "inserir_texto", "args": {"texto": "olá"}, "id": "call_9"}])
USER = [Message(role="user", content="insira olá no doc")]


def test_llm_complete_com_tools_devolve_tool_calls():
    llm = make_llm()
    llm.chat_model.script = [CALL]
    out = llm.complete(USER, tools=WIRE_TOOLS)
    assert out.tool_calls and out.tool_calls[0]["name"] == "inserir_texto"
    assert out.tool_calls[0]["args"] == {"texto": "olá"}


def test_llm_complete_sem_tools_inalterado():
    llm = make_llm()
    llm.chat_model.script = [AIMessage(content="oi")]
    out = llm.complete(USER)
    assert out.content == "oi" and not out.tool_calls


def _make_client_agent(name: str):
    from aixon.agents.llm_agent import LLMAgent

    llm = make_llm()
    llm.chat_model.script = [CALL]
    cls = type(f"{name.capitalize()}Agent", (LLMAgent,), {
        "name": name, "llm": llm, "client_tools": True,
    })
    from aixon.registry import get_registry
    return get_registry().resolve(name)


def test_llmagent_client_tools_le_o_contextvar():
    agent = _make_client_agent("ct18a")
    with client_tools(WIRE_TOOLS):
        out = agent.invoke(USER)
    assert out.tool_calls and out.tool_calls[0]["name"] == "inserir_texto"


def test_llmagent_sem_contextvar_segue_normal():
    from aixon.agents.llm_agent import LLMAgent

    llm = make_llm()
    llm.chat_model.script = [AIMessage(content="sem tools")]
    cls = type("Ct18bAgent", (LLMAgent,), {
        "name": "ct18b", "llm": llm, "client_tools": True,
    })
    from aixon.registry import get_registry
    out = get_registry().resolve("ct18b").invoke(USER)
    assert out.content == "sem tools"


def test_llmagent_async_e_stream():
    agent = _make_client_agent("ct18c")
    with client_tools(WIRE_TOOLS):
        out = asyncio.run(agent.ainvoke(USER))
    assert out.tool_calls

    agent2 = _make_client_agent("ct18d")
    with client_tools(WIRE_TOOLS):
        chunks = list(agent2.stream(USER))
    calls = [c for c in chunks if c.tool_calls]
    assert calls and calls[0].tool_calls[0]["name"] == "inserir_texto"
    assert chunks[-1].done


def test_tool_choice_scope_roundtrip():
    from aixon.runtime import current_tool_choice

    assert current_tool_choice() is None
    with tool_choice_scope({"type": "function",
                            "function": {"name": "inserir_texto"}}):
        assert current_tool_choice()["function"]["name"] == "inserir_texto"
    assert current_tool_choice() is None
