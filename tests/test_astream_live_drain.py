"""#20 — reasoning de dentro de uma tool aparece ANTES da tool completar."""
import asyncio, time
from langchain_core.messages import AIMessage
from aixon.agents.tool_agent import ToolAgent
from aixon.message import Message
from aixon.reasoning import emit_reasoning
from tests._fakes import make_llm

def test_reasoning_interno_flui_durante_a_tool():
    events = []
    async def lenta(x: str) -> str:
        "tool lenta"
        emit_reasoning("progresso interno")
        await asyncio.sleep(1.0)
        events.append(("tool_end", time.monotonic()))
        return "ok"
    llm = make_llm()
    llm.chat_model.script = [
        AIMessage(content="", tool_calls=[{"name": "lenta", "args": {"x": "1"}, "id": "c1"}]),
        AIMessage(content="fim"),
    ]
    cls = type("Drain20Agent", (ToolAgent,), {"name": "drain20", "llm": llm, "tools": [lenta]})
    from aixon.registry import get_registry
    agent = get_registry().resolve("drain20")
    async def run():
        async for ch in agent.astream([Message(role="user", content="vai")]):
            if ch.reasoning and "progresso interno" in ch.reasoning:
                events.append(("chunk_interno", time.monotonic()))
    asyncio.run(run())
    interno = next(t for n, t in events if n == "chunk_interno")
    fim = next(t for n, t in events if n == "tool_end")
    assert interno < fim - 0.3, (
        f"reasoning interno só saiu {interno - fim:+.2f}s em relação ao fim da tool — deveria fluir DURANTE")


def test_sem_duplicacao_nem_perda_em_multiplos_ticks():
    """Stress do drain multi-tick: 9 linhas espaçadas por ~0.15s cruzam vários
    ticks de 0.25s — cada linha sai exatamente uma vez, na ordem de emissão."""
    esperadas = [f"line-{i}" for i in range(9)]

    async def tagarela(x: str) -> str:
        "tool que emite várias linhas ao longo do tempo"
        for linha in esperadas:
            emit_reasoning(linha)
            await asyncio.sleep(0.15)
        return "ok"

    llm = make_llm()
    llm.chat_model.script = [
        AIMessage(content="", tool_calls=[{"name": "tagarela", "args": {"x": "1"}, "id": "c1"}]),
        AIMessage(content="fim"),
    ]
    cls = type("Drain20MultiAgent", (ToolAgent,),
               {"name": "drain20multi", "llm": llm, "tools": [tagarela]})
    from aixon.registry import get_registry
    agent = get_registry().resolve("drain20multi")

    linhas: list[str] = []

    async def run():
        async for ch in agent.astream([Message(role="user", content="vai")]):
            if ch.reasoning:
                linhas.extend(
                    l for l in ch.reasoning.splitlines() if l.startswith("line-"))
    asyncio.run(run())

    assert linhas == esperadas, (
        f"esperava as 9 linhas exatamente uma vez cada, na ordem; veio {linhas}")
