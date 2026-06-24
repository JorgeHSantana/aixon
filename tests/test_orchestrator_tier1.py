# tests/test_orchestrator_tier1.py
import pytest

from _fakes import make_llm, make_echo_agent
from aixon.agents.orchestrator import Orchestrator
from aixon.exceptions import AixonError, NamingError
from aixon.message import Message
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
