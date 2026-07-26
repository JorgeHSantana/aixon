# tests/test_tool_prune.py
"""#16 — poda de tool results antigos do histórico (stub no lugar)."""
from __future__ import annotations

from aixon.agents.tool_agent import ToolAgent
from aixon.message import Message


def _hist():
    big = "linha;" * 500
    return [
        Message(role="user", content="vendas de maio?"),
        Message(role="assistant", content="", tool_calls=[
            {"name": "sql", "args": {"q": "..."}, "id": "c1"}]),
        Message(role="tool", content=big, tool_call_id="c1"),
        Message(role="assistant", content="Maio: R$ 10k."),
        Message(role="user", content="e junho?"),
        Message(role="assistant", content="", tool_calls=[
            {"name": "sql", "args": {"q": "..."}, "id": "c2"}]),
        Message(role="tool", content=big, tool_call_id="c2"),
        Message(role="assistant", content="Junho: R$ 12k."),
        Message(role="user", content="compare os dois"),
    ]


def test_poda_stub_no_turno_antigo_preserva_o_recente():
    msgs = _hist()
    out = ToolAgent._prune_history(msgs, keep_turns=2)
    # tool result do turno de maio (fora das 2 últimas assistants) -> stub
    assert "omitido" in out[2].content and "3000" in out[2].content
    # tool result do turno de junho (dentro da janela) -> intacto
    assert out[6].content == msgs[6].content
    # nada além de tool messages foi tocado; caller intacto
    assert out[3].content == "Maio: R$ 10k."
    assert "linha;" in msgs[2].content  # lista original não mutada


def test_janela_maior_que_o_historico_nao_poda_nada():
    msgs = _hist()
    out = ToolAgent._prune_history(msgs, keep_turns=99)
    assert all(a.content == b.content for a, b in zip(out, msgs))


def test_default_desligado_nao_poda():
    assert ToolAgent.prune_tool_results_after is None
