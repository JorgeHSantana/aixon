# tests/test_orchestrator_reasoning.py
from typing import Iterator

from tests._fakes import make_llm
from aixon import emit_reasoning
from aixon.agent import Agent
from aixon.agents.orchestrator import Orchestrator
from aixon.message import Chunk, Message
from aixon.registry import get_registry


def _make_thinker(name: str, thought: str) -> type:
    def invoke(self, messages: list[Message]) -> Message:
        emit_reasoning(thought)
        last = messages[-1].content if messages else ""
        return Message(role="assistant", content=f"{name}:{last}")

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        emit_reasoning(thought)
        yield Chunk(content="x")
        yield Chunk(done=True)

    return type(
        f"{name.capitalize()}Agent",
        (Agent,),
        {"name": name, "invoke": invoke, "stream": stream},
    )


def test_node_reasoning_bubbles_to_orchestrator_stream():
    _make_thinker("thinker", "pondering the request")

    class ReasoningOrchestrator(Orchestrator):
        supervisor = make_llm()
        agents = [get_registry().resolve("thinker")]

    orch = get_registry().resolve("reasoningorchestrator")
    chunks = list(orch.stream([Message(role="user", content="hi")]))
    reasoning_text = "".join(c.reasoning for c in chunks)
    assert "pondering the request" in reasoning_text
    assert any("thinker:hi" in c.content for c in chunks)
    assert chunks[-1].done is True
