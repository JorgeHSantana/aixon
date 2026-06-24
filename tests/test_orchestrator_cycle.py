# tests/test_orchestrator_cycle.py
import pytest

from tests._fakes import make_llm, make_echo_agent
from aixon.agents.orchestrator import Orchestrator
from aixon.exceptions import CompositionCycleError
from aixon.registry import get_registry
from aixon.state import END


def test_self_inclusion_raises_composition_cycle():
    cls = type("SelfOrchestrator", (Orchestrator,), {
        "supervisor": make_llm(),
        "agents": [],
    })
    cls.agents = [cls]              # inject self-reference
    with pytest.raises(CompositionCycleError, match="cycle"):
        cls._check_composition_cycle()


def test_mutual_inclusion_raises_composition_cycle():
    make_echo_agent("leaf")

    class AOrchestrator(Orchestrator):
        supervisor = make_llm()
        agents = [get_registry().resolve("leaf")]

    class BOrchestrator(Orchestrator):
        supervisor = make_llm()
        agents = [AOrchestrator]

    AOrchestrator.agents = [BOrchestrator]   # close the loop
    with pytest.raises(CompositionCycleError, match="cycle"):
        AOrchestrator._check_composition_cycle()


def test_acyclic_composition_is_allowed():
    make_echo_agent("leaf")

    class InnerOrchestrator(Orchestrator):
        supervisor = make_llm()
        agents = [get_registry().resolve("leaf")]

    class OuterOrchestrator(Orchestrator):
        supervisor = make_llm()
        agents = [InnerOrchestrator]  # outer -> inner -> leaf, no cycle

    OuterOrchestrator._check_composition_cycle()  # no exception


def test_langgraph_internal_loop_is_not_a_composition_cycle():
    # A Tier-2 graph whose node loops back is legitimate; guard A must NOT fire.
    make_echo_agent("loopa")
    make_echo_agent("loopb")

    class LoopingOrchestrator(Orchestrator):
        nodes = {"a": get_registry().resolve("loopa"),
                 "b": get_registry().resolve("loopb")}
        entry = "a"
        edges = [("a", "b"), ("b", "a")]  # graph-level cycle, allowed

    assert LoopingOrchestrator._tier == 2  # definition did not raise
