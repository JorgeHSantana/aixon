# tests/test_orchestrator_tier1.py
import asyncio

import pytest

from tests._fakes import make_llm, make_echo_agent
from aixon.agent import Agent
from aixon.agents.orchestrator import Orchestrator
from aixon.exceptions import AixonError, NamingError
from aixon.message import Chunk, Message
from aixon.registry import get_registry


def test_tier1_detected_when_supervisor_and_agents_set():
    billing = make_echo_agent("billing")
    tech = make_echo_agent("tech")

    class SupportOrchestrator(Orchestrator):
        supervisor = make_llm()
        agents = [billing.__class__]

    assert SupportOrchestrator._tier == 1


def test_tier1_runs_worker_and_returns_assistant_message():
    make_echo_agent("billing")

    class SoloOrchestrator(Orchestrator):
        supervisor = make_llm()
        agents = [get_registry().resolve("billing")]

    orch = get_registry().resolve("soloorchestrator")
    out = orch.invoke([Message(role="user", content="help")])
    assert out.role == "assistant"
    assert "help" in out.content  # the worker echoed the user content


def test_orchestrator_is_registered_with_suffix_name():
    make_echo_agent("billing")

    class RoutingOrchestrator(Orchestrator):
        supervisor = make_llm()
        agents = [get_registry().resolve("billing")]

    assert get_registry().resolve("routingorchestrator").name == "routingorchestrator"


def test_bad_suffix_raises_naming_error():
    make_echo_agent("billing")
    with pytest.raises(NamingError, match="Orchestrator"):
        type(
            "BadName",
            (Orchestrator,),
            {"supervisor": make_llm(), "agents": [get_registry().resolve("billing")]},
        )


def test_no_tier_applies_raises_aixon_error():
    with pytest.raises(AixonError, match="tier"):
        type("EmptyOrchestrator", (Orchestrator,), {})


def test_invalid_orchestrator_leaves_no_ghost_in_registry():
    """A concrete Orchestrator that fails validation (no tier) must NOT be
    registered: _validate_subclass runs BEFORE registration, so the registry
    stays clean — no register-then-validate ghost."""
    before = {a.name for a in get_registry().all()}
    with pytest.raises(AixonError, match="tier"):
        type("GhostOrchestrator", (Orchestrator,), {})
    after = {a.name for a in get_registry().all()}
    assert "ghostorchestrator" not in after
    assert after == before


def test_stream_yields_content_and_done():
    make_echo_agent("billing")

    class StreamOrchestrator(Orchestrator):
        supervisor = make_llm()
        agents = [get_registry().resolve("billing")]

    orch = get_registry().resolve("streamorchestrator")
    chunks = list(orch.stream([Message(role="user", content="hey")]))
    assert any("hey" in c.content for c in chunks)
    assert chunks[-1].done is True


def _make_client_tool_worker(name: str):
    """Worker whose invoke/stream surface a CLIENT tool_calls answer (#18c
    style): content="" with tool_calls set, exactly what a ToolAgent with
    client_tools="merge" returns when the model calls a client tool."""
    calls = [{"name": "inserir_no_documento", "args": {"texto": "5"}, "id": "c1"}]

    def invoke(self, messages: list[Message]) -> Message:
        return Message(role="assistant", content="", tool_calls=calls)

    def stream(self, messages):
        yield Chunk(tool_calls=calls)
        yield Chunk(done=True)

    type(
        name.capitalize() + "Agent", (Agent,),
        {"name": name, "invoke": invoke, "stream": stream},
    )
    return get_registry().resolve(name)


def _client_tool_orchestrator(orch_name: str, worker_name: str):
    """Single-worker Tier-1 orchestrator whose fake supervisor always routes
    to the worker, then DONE once it has answered (same duck-typed supervisor
    idiom as tests/test_orchestrator_tier1_routing.py)."""
    _make_client_tool_worker(worker_name)

    class _Sup:
        def complete(self, messages):
            if messages and messages[-1].role == "assistant":
                return Message(role="assistant", content="DONE")
            return Message(role="assistant", content=worker_name)

    cls_name = orch_name.capitalize() + "Orchestrator"
    type(cls_name, (Orchestrator,),
         {"supervisor": _Sup(), "agents": [get_registry().resolve(worker_name)]})
    return get_registry().resolve(cls_name.lower())


def test_stream_surfaces_worker_tool_calls():
    orch = _client_tool_orchestrator("stt2a", "sworker2a")
    chunks = list(orch.stream([Message(role="user", content="faça algo")]))
    tool_call_chunks = [c for c in chunks if c.tool_calls]
    assert tool_call_chunks
    assert tool_call_chunks[0].tool_calls == [
        {"name": "inserir_no_documento", "args": {"texto": "5"}, "id": "c1"}]
    # No empty/garbage content chunk swallowing the tool_calls.
    assert not any(c.content for c in chunks)
    assert chunks[-1].done is True


def test_astream_surfaces_worker_tool_calls():
    orch = _client_tool_orchestrator("stt2b", "sworker2b")

    async def run():
        return [c async for c in orch.astream([Message(role="user", content="faça algo")])]

    chunks = asyncio.run(run())
    tool_call_chunks = [c for c in chunks if c.tool_calls]
    assert tool_call_chunks
    assert tool_call_chunks[0].tool_calls == [
        {"name": "inserir_no_documento", "args": {"texto": "5"}, "id": "c1"}]
    assert not any(c.content for c in chunks)
    assert chunks[-1].done is True
