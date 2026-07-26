# tests/test_tool_parallel.py
"""#13 — tool calls do mesmo turno executam em paralelo no caminho async."""
from __future__ import annotations

import asyncio
import time

from langchain_core.messages import AIMessage

from aixon.agents.tool_agent import ToolAgent
from aixon.message import Message
from tests._fakes import make_llm

SLEEP = 0.4


async def lenta_a(x: str) -> str:
    """Ferramenta lenta A."""
    await asyncio.sleep(SLEEP)
    return "resultado A"


async def lenta_b(x: str) -> str:
    """Ferramenta lenta B."""
    await asyncio.sleep(SLEEP)
    return "resultado B"


def _make_agent(name: str):
    llm = make_llm()
    llm.chat_model.script = [
        AIMessage(content="", tool_calls=[
            {"name": "lenta_a", "args": {"x": "1"}, "id": "c1"},
            {"name": "lenta_b", "args": {"x": "1"}, "id": "c2"},
        ]),
        AIMessage(content="pronto"),
    ]
    cls = type(f"{name.capitalize()}Agent", (ToolAgent,), {
        "name": name, "llm": llm, "tools": [lenta_a, lenta_b],
    })
    from aixon.registry import get_registry
    return get_registry().resolve(name)


def test_duas_tools_do_mesmo_turno_em_paralelo_no_ainvoke():
    agent = _make_agent("par13")
    start = time.monotonic()
    out = asyncio.run(agent.ainvoke([Message(role="user", content="vai")]))
    elapsed = time.monotonic() - start
    assert out.content == "pronto"
    # Paralelo: ~1x SLEEP. Serial: ~2x. Corte no meio com folga de CI.
    assert elapsed < SLEEP * 1.75, (
        f"tool calls do mesmo turno rodaram em série ({elapsed:.2f}s p/ "
        f"2 tools de {SLEEP}s)"
    )
