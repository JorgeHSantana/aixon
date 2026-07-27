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
