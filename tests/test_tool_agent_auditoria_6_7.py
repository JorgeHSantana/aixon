# tests/test_tool_agent_auditoria_6_7.py
"""Task 6.7 (varredura 0.1.29): achados tool_agent-3..9 do ToolAgent."""
from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import AIMessage

from aixon.agents.tool_agent import ToolAgent
from aixon.message import Message
from aixon.reasoning import current_channel, emit_reasoning, reasoning_channel
from aixon.runtime import client_tools_scope
from tests._fakes import make_llm

USER = [Message(role="user", content="oi")]
DOC_TOOL = [{
    "type": "function",
    "function": {"name": "inserir_no_documento", "description": "Insere.",
                 "parameters": {"type": "object", "properties": {}}},
}]


def soma(a: int, b: int) -> str:
    """Soma dois inteiros."""
    return str(a + b)


def _agent(name, script=None, **extra):
    llm = make_llm()
    if script is not None:
        llm.chat_model.script = script
    return type(f"{name.capitalize()}Agent", (ToolAgent,),
                {"name": name, "llm": llm, "tools": [soma], **extra})()


# ---- tool_agent-3 ----------------------------------------------------------

def test_tool_agent_3_resolve_client_tools():
    ag = _agent("aud3", client_tools="merge")
    from aixon._interop.tools import coerce_tools
    lc = coerce_tools([soma])
    assert ag._resolve_client_tools(lc) == (lc, set())  # sem client tools no escopo
    with client_tools_scope(DOC_TOOL):
        tools, names = ag._resolve_client_tools(lc)
    assert names == {"inserir_no_documento"}
    assert [t.name for t in tools] == ["soma", "inserir_no_documento"]


# ---- tool_agent-4 / 6 / 9 --------------------------------------------------

def _script():
    return [
        AIMessage(content="", tool_calls=[{"name": "soma", "args": {"a": 1, "b": 2}, "id": "c1"}]),
        AIMessage(content="3"),
    ]


def test_tool_agent_4_finish_run_igual_em_invoke_e_ainvoke():
    a = _agent("aud4a", _script()).invoke(USER)
    b = asyncio.run(_agent("aud4b", _script()).ainvoke(USER))
    assert (a.content, a.reasoning, a.usage) == (b.content, b.reasoning, b.usage)
    assert hasattr(ToolAgent, "_finish_run")


def test_tool_agent_6_emit_ai_message_reasoning_antes_dos_labels():
    ag = _agent("aud6")
    m = AIMessage(content="", additional_kwargs={"reasoning_content": "pensei"},
                  tool_calls=[{"name": "soma", "args": {}, "id": "c1"}])
    with reasoning_channel() as ch:
        ag._emit_ai_message(m)
        lines = ch.drain()
    assert lines == ["pensei", ag.tool_call_label.format(name="soma")]


def test_tool_agent_9_reasoning_scope():
    ag = _agent("aud9")
    cm, owns = ag._reasoning_scope()
    assert owns is True
    with cm as ch:
        assert current_channel() is ch
        cm2, owns2 = ag._reasoning_scope()
        assert owns2 is False
        with cm2 as ch2:
            assert ch2 is ch


# ---- tool_agent-5 ----------------------------------------------------------

def test_tool_agent_5_stream_e_astream_emitem_a_mesma_sequencia():
    s = list(_agent("aud5a", _script()).stream(USER))

    async def coleta():
        return [c async for c in _agent("aud5b", _script()).astream(USER)]

    a = asyncio.run(coleta())
    assert [(c.reasoning, c.content, c.done) for c in s] == \
        [(c.reasoning, c.content, c.done) for c in a]
    ag = _agent("aud5c")
    seen: list = []
    upd = {"model": {"messages": [AIMessage(content="fim")]}}
    assert ag._absorb_update(upd, set(), seen) == "fim"
    assert seen == []
    assert [c.content for c in ag._tail_chunks(seen, set(), "fim", False)] == ["fim", ""]
