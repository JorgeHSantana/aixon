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
    # keep_turns=1: âncora por ROUNDS COMPLETOS (um round termina numa
    # assistant SEM tool_calls) — o round de maio (mais antigo) é estubado,
    # o de junho (o round completo mais recente) fica intacto.
    msgs = _hist()
    out = ToolAgent._prune_history(msgs, keep_turns=1)
    # tool result do round de maio (fora do último round completo) -> stub
    assert "omitido" in out[2].content and "3000" in out[2].content
    # tool result do round de junho (o último round completo) -> intacto
    assert out[6].content == msgs[6].content
    # nada além de tool messages foi tocado; caller intacto
    assert out[3].content == "Maio: R$ 10k."
    assert "linha;" in msgs[2].content  # lista original não mutada


def test_poda_por_rounds_completos_nao_estuba_tool_do_round_mais_recente():
    # Regressão do bug: âncorar em TODA assistant (inclusive a que ABRIU a
    # tool_call) fazia keep_turns=1 estubar o tool result do round MAIS
    # RECENTE (o único round aqui, ainda em andamento — sem resposta final
    # ainda). Com a âncora por rounds completos, len(finals)=1 <= keep_turns
    # (1) e nada é podado: o tool result do round corrente sobrevive intacto.
    msgs = [
        Message(role="user", content="vendas de maio?"),
        Message(role="assistant", content="", tool_calls=[
            {"name": "sql", "args": {"q": "..."}, "id": "c1"}]),
        Message(role="tool", content="dump grande de maio", tool_call_id="c1"),
        Message(role="assistant", content="Maio: R$ 10k."),
        Message(role="user", content="e junho?"),
    ]
    out = ToolAgent._prune_history(msgs, keep_turns=1)
    assert out[2].content == "dump grande de maio"


def test_janela_maior_que_o_historico_nao_poda_nada():
    msgs = _hist()
    out = ToolAgent._prune_history(msgs, keep_turns=99)
    assert all(a.content == b.content for a, b in zip(out, msgs))


def test_default_desligado_nao_poda():
    assert ToolAgent.prune_tool_results_after is None


def test_valor_invalido_rejeitado_no_registro():
    import pytest
    from aixon.exceptions import AixonError
    from tests._fakes import make_llm

    llm = make_llm()
    with pytest.raises(AixonError, match="prune_tool_results_after"):
        type("Pr16invAgent", (ToolAgent,), {
            "name": "pr16inv", "llm": llm, "tools": [],
            "prune_tool_results_after": 0,
        })


def test_invoke_com_prune_manda_stub_para_o_modelo(monkeypatch):
    """Integração: prune_tool_results_after via invoke — o histórico que
    chega ao MODELO (create_agent) já traz o stub no lugar do tool result
    antigo, não o dump original de 3000 caracteres. Captura via subclasse do
    FakeChatModel (idioma de tests/_fakes.py), como em
    test_tool_agent_invoke.py::test_toolagent_developer_role_overrides_class_prompt."""
    from langchain_core.messages import AIMessage
    from tests._fakes import FakeChatModel, make_llm

    llm = make_llm()
    fake = FakeChatModel(script=[AIMessage(content="Maio estável, junho subiu.")])
    captured: list[list] = []
    original_generate = fake._generate

    def capturing_generate(messages, stop=None, run_manager=None, **kwargs):
        captured.append(list(messages))
        return original_generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    fake._generate = capturing_generate
    monkeypatch.setattr(type(llm), "chat_model", property(lambda self: fake))

    from aixon.registry import get_registry

    type("Pr16IntAgent", (ToolAgent,), {
        "name": "pr16int", "llm": llm, "tools": [],
        "prune_tool_results_after": 1,
    })
    agent = get_registry().resolve("pr16int")

    out = agent.invoke(_hist())

    assert out.content == "Maio estável, junho subiu."
    assert captured, "o modelo nunca foi chamado"
    tool_msgs = [m for m in captured[0] if type(m).__name__ == "ToolMessage"]
    assert len(tool_msgs) == 2
    # maio (fora do último round completo) chega como stub; junho (o round
    # completo mais recente) intacto.
    assert "omitido" in tool_msgs[0].content
    assert "linha;" not in tool_msgs[0].content
    assert "linha;" in tool_msgs[1].content
