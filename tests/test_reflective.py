# tests/test_reflective.py
"""ReflectiveAgent: evaluator-optimizer loop (generator -> judge -> retry)."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from aixon.exceptions import AixonError, CompositionCycleError
from aixon.message import Chunk, Message
from tests._fakes import make_llm

from aixon.agents.reflective import ReflectiveAgent


# ── scriptable generator agent ───────────────────────────────────────────────

def make_scripted_agent(name: str, answers: list[str]):
    """Concrete Agent returning `answers` in order; records received messages."""
    from typing import Iterator

    from aixon.agent import Agent

    calls: list[list[Message]] = []

    def invoke(self, messages: list[Message]) -> Message:
        calls.append(list(messages))
        i = min(len(calls) - 1, len(answers) - 1)
        return Message(role="assistant", content=answers[i])

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        yield Chunk(content=invoke(self, messages).content)
        yield Chunk(done=True)

    cls = type(f"{name.capitalize()}Agent", (Agent,),
               {"invoke": invoke, "stream": stream, "name": name})
    from aixon.registry import get_registry
    return get_registry().resolve(name), calls


def make_reflective(name: str, agent, judge_script: list[str], *, rounds: int = 3):
    """Concrete ReflectiveAgent with a scripted fake judge."""
    judge = make_llm(temperature=0)
    judge.chat_model.script = [AIMessage(content=v) for v in judge_script]
    cls = type(f"{name.capitalize()}Agent", (ReflectiveAgent,), {
        "name": name,
        "agent": agent,
        "judge_llm": judge,
        "judge_rubric": "1. A resposta cita a fonte.",
        "max_rounds": rounds,
    })
    from aixon.registry import get_registry
    return get_registry().resolve(name)


USER = [Message(role="user", content="qual a capital do Ceará?")]


# ── happy path / loop ────────────────────────────────────────────────────────

def test_aprovado_na_primeira_rodada():
    gen, calls = make_scripted_agent("gen1", ["Fortaleza (fonte: IBGE)."])
    r = make_reflective("ref1", gen, ["APROVADO"])
    out = r.invoke(USER)
    assert out.content == "Fortaleza (fonte: IBGE)."
    assert len(calls) == 1  # gerador rodou uma vez


def test_reprova_e_aprova_na_segunda():
    gen, calls = make_scripted_agent("gen2", ["Fortaleza.", "Fortaleza (fonte: IBGE)."])
    r = make_reflective("ref2", gen, ["1. Falta citar a fonte.", "APROVADO"])
    out = r.invoke(USER)
    assert out.content == "Fortaleza (fonte: IBGE)."
    assert len(calls) == 2
    # a 2ª chamada do gerador recebeu a crítica na conversa (sem mutar USER)
    retry_msgs = calls[1]
    assert any("Falta citar a fonte" in m.content for m in retry_msgs)
    assert any(m.role == "assistant" and m.content == "Fortaleza." for m in retry_msgs)
    assert len(USER) == 1  # lista do chamador intacta


def test_esgota_rodadas_devolve_ultima_sem_excecao():
    gen, calls = make_scripted_agent("gen3", ["v1", "v2"])
    r = make_reflective("ref3", gen, ["1. Ruim.", "1. Ainda ruim."], rounds=2)
    out = r.invoke(USER)
    assert out.content == "v2"          # última tentativa
    assert len(calls) == 2              # exatamente max_rounds


def test_sentinela_e_estrita():
    # minúscula ou no meio do texto NÃO aprovam
    gen, calls = make_scripted_agent("gen4", ["a", "b"])
    r = make_reflective("ref4", gen, ["aprovado", "Isto está APROVADO"], rounds=2)
    out = r.invoke(USER)
    assert out.content == "b"           # nunca aprovou -> esgotou
    # primeira linha APROVADO com espaços em volta aprova
    gen2, _ = make_scripted_agent("gen5", ["x"])
    r2 = make_reflective("ref5", gen2, ["  APROVADO  \ndetalhes"])
    assert r2.invoke(USER).content == "x"


# ── reasoning labels ─────────────────────────────────────────────────────────

def test_labels_no_canal_de_reasoning():
    from aixon.reasoning import reasoning_channel

    gen, _ = make_scripted_agent("gen6", ["a", "b"])
    r = make_reflective("ref6", gen, ["1. Ruim.", "APROVADO"])
    with reasoning_channel() as ch:
        r.invoke(USER)
        lines = ch.drain()
    assert r.judge_label in lines
    assert any("rodada 2/3" in ln for ln in lines)


def test_exhausted_label_emitido():
    from aixon.reasoning import reasoning_channel

    gen, _ = make_scripted_agent("gen7", ["a"])
    r = make_reflective("ref7", gen, ["1. Ruim."], rounds=1)
    with reasoning_channel() as ch:
        r.invoke(USER)
        lines = ch.drain()
    assert r.exhausted_label in lines


# ── scriptable STREAMING agent (reasoning antes do conteúdo) ─────────────────

def make_streaming_agent(name: str, rounds: list[tuple[list[str], str]]):
    """Concrete Agent cujo stream() emite Chunk(reasoning=...) por linha e só
    então o conteúdo — como um ToolAgent real (labels de tool ao vivo).
    `rounds` é uma lista de (linhas_de_reasoning, resposta) por chamada."""
    from typing import Iterator

    from aixon.agent import Agent

    calls: list[list[Message]] = []

    def invoke(self, messages: list[Message]) -> Message:
        calls.append(list(messages))
        i = min(len(calls) - 1, len(rounds) - 1)
        return Message(role="assistant", content=rounds[i][1])

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        calls.append(list(messages))
        i = min(len(calls) - 1, len(rounds) - 1)
        reasoning_lines, answer = rounds[i]
        for line in reasoning_lines:
            yield Chunk(reasoning=line + "\n")
        yield Chunk(content=answer)
        yield Chunk(done=True)

    cls = type(f"{name.capitalize()}Agent", (Agent,),
               {"invoke": invoke, "stream": stream, "name": name})
    from aixon.registry import get_registry
    return get_registry().resolve(name), calls


# ── stream ───────────────────────────────────────────────────────────────────

def test_stream_reasoning_e_conteudo():
    gen, _ = make_scripted_agent("gen8", ["resposta final"])
    r = make_reflective("ref8", gen, ["APROVADO"])
    chunks = list(r.stream(USER))
    reasoning = "".join(c.reasoning for c in chunks)
    content = "".join(c.content for c in chunks)
    assert r.judge_label in reasoning
    assert content == "resposta final"
    assert chunks[-1].done is True


def test_stream_e_vivo_reasoning_do_worker_sai_antes_do_juiz():
    # O reasoning do worker deve fluir ENQUANTO ele trabalha — antes de o juiz
    # rodar — e não ser drenado só no fim do loop (streaming mudo).
    gen, _ = make_streaming_agent(
        "gen-live", [(["Consultando o banco…"], "resposta (fonte: IBGE)")])
    r = make_reflective("ref-live", gen, ["APROVADO"])
    it = r.stream(USER)
    first = next(it)
    assert first.reasoning == "Consultando o banco…\n"
    # juiz ainda não consumiu o script: nada foi julgado até aqui
    assert r.judge_llm.chat_model._idx == 0
    rest = list(it)
    assert "".join(c.content for c in rest) == "resposta (fonte: IBGE)"
    assert rest[-1].done is True


def test_stream_reprova_conteudo_reprovado_nao_vaza():
    # Rodada 1 reprovada: o conteúdo v1 NÃO pode sair como content; o reasoning
    # das DUAS rodadas aparece, com o retry label entre elas.
    gen, _ = make_streaming_agent(
        "gen-2r", [(["passo A"], "v1"), (["passo B"], "v2 (fonte: IBGE)")])
    r = make_reflective("ref-2r", gen, ["1. Falta fonte.", "APROVADO"])
    chunks = list(r.stream(USER))
    reasoning = "".join(c.reasoning for c in chunks)
    content = "".join(c.content for c in chunks)
    assert "passo A" in reasoning and "passo B" in reasoning
    assert any("rodada 2/3" in (c.reasoning or "") for c in chunks)
    assert content == "v2 (fonte: IBGE)"   # v1 não vazou
    assert chunks[-1].done is True


def test_stream_esgota_emite_exhausted_e_ultima_tentativa():
    gen, _ = make_streaming_agent(
        "gen-ex", [(["p1"], "v1"), (["p2"], "v2")])
    r = make_reflective("ref-ex", gen, ["1. Ruim.", "1. Ainda ruim."], rounds=2)
    chunks = list(r.stream(USER))
    reasoning = "".join(c.reasoning for c in chunks)
    assert r.exhausted_label in reasoning
    assert "".join(c.content for c in chunks) == "v2"
    assert chunks[-1].done is True


# ── validação (sem fantasma no registry) ─────────────────────────────────────

def _ghost_free(name: str):
    from aixon.registry import get_registry
    assert all(a.name != name for a in get_registry().all())


def test_validacao_exige_agent():
    with pytest.raises(AixonError, match="agent"):
        type("SemAgenteAgent", (ReflectiveAgent,), {
            "name": "sem-agente", "judge_llm": make_llm(),
            "judge_rubric": "x", })
    _ghost_free("sem-agente")


def test_validacao_exige_judge_llm():
    gen, _ = make_scripted_agent("gen9", ["a"])
    with pytest.raises(AixonError, match="judge_llm"):
        type("SemJuizAgent", (ReflectiveAgent,), {
            "name": "sem-juiz", "agent": gen, "judge_rubric": "x"})
    _ghost_free("sem-juiz")


def test_validacao_exige_rubrica_nao_vazia():
    gen, _ = make_scripted_agent("gen10", ["a"])
    with pytest.raises(AixonError, match="judge_rubric"):
        type("SemRubricaAgent", (ReflectiveAgent,), {
            "name": "sem-rubrica", "agent": gen, "judge_llm": make_llm(),
            "judge_rubric": "   "})
    _ghost_free("sem-rubrica")


def test_validacao_max_rounds_minimo():
    gen, _ = make_scripted_agent("gen11", ["a"])
    with pytest.raises(AixonError, match="max_rounds"):
        type("RoundsZeroAgent", (ReflectiveAgent,), {
            "name": "rounds-zero", "agent": gen, "judge_llm": make_llm(),
            "judge_rubric": "x", "max_rounds": 0})
    _ghost_free("rounds-zero")


def test_ciclo_de_composicao_detectado():
    gen, _ = make_scripted_agent("gen12", ["a"])
    a = make_reflective("ciclo-a", gen, ["APROVADO"])
    b = make_reflective("ciclo-b", type(a), ["APROVADO"])  # b embrulha a (ok)
    type(a).agent = type(b)  # fecha o ciclo por mutação: a -> b -> a
    with pytest.raises(CompositionCycleError):
        type("CicloCAgent", (ReflectiveAgent,), {
            "name": "ciclo-c", "agent": type(b),
            "judge_llm": make_llm(), "judge_rubric": "x"})
    _ghost_free("ciclo-c")


# ── integração com o resto do framework ──────────────────────────────────────

def test_registro_e_export():
    import aixon
    assert aixon.ReflectiveAgent is ReflectiveAgent
    gen, _ = make_scripted_agent("gen13", ["a"])
    r = make_reflective("ref-reg", gen, ["APROVADO"])
    from aixon.registry import get_registry
    assert get_registry().resolve("ref-reg") is r


def test_as_tool_interface_uniforme():
    gen, _ = make_scripted_agent("gen14", ["resposta via tool"])
    r = make_reflective("ref-tool", gen, ["APROVADO"])
    tool = r.as_tool(name="revisado")
    assert tool.name == "revisado"
    assert tool.func("pergunta") == "resposta via tool"


def test_retries_acumulam_historico():
    gen, calls = make_scripted_agent("gen15", ["v1", "v2", "v3"])
    r = make_reflective("ref-hist", gen,
                        ["1. Critica A.", "2. Critica B.", "APROVADO"])
    r.invoke(USER)
    # a 3ª chamada do worker vê AMBAS as críticas anteriores
    third = "\n".join(m.content for m in calls[2])
    assert "Critica A" in third and "Critica B" in third
