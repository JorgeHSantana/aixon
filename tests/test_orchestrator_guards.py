import pytest

from tests._fakes import make_llm, make_echo_agent
from aixon.agents.orchestrator import Orchestrator
from aixon.exceptions import AixonError
from aixon.message import Message
from aixon.registry import get_registry


def test_default_recursion_limit_is_25():
    make_echo_agent("gw")

    class GuardOrchestrator(Orchestrator):
        supervisor = make_llm()
        agents = [get_registry().resolve("gw")]

    orch = get_registry().resolve("guardorchestrator")
    assert orch.recursion_limit == 25
    assert orch._run_config()["recursion_limit"] == 25


def test_recursion_limit_none_omits_key():
    make_echo_agent("ncw")

    class NoCapOrchestrator(Orchestrator):
        supervisor = make_llm()
        agents = [get_registry().resolve("ncw")]
        recursion_limit = None

    orch = get_registry().resolve("nocaporchestrator")
    assert "recursion_limit" not in orch._run_config()


def test_custom_recursion_limit_is_passed():
    make_echo_agent("ccw")

    class CustomCapOrchestrator(Orchestrator):
        supervisor = make_llm()
        agents = [get_registry().resolve("ccw")]
        recursion_limit = 7

    orch = get_registry().resolve("customcaporchestrator")
    assert orch._run_config()["recursion_limit"] == 7


def test_nonterminating_loop_hits_recursion_limit():
    # Tier-2 graph with a hard a<->b loop and a tiny recursion_limit.
    make_echo_agent("cyclea")
    make_echo_agent("cycleb")

    class LoopGuardOrchestrator(Orchestrator):
        nodes = {"a": get_registry().resolve("cyclea"),
                 "b": get_registry().resolve("cycleb")}
        entry = "a"
        edges = [("a", "b"), ("b", "a")]  # never reaches END
        recursion_limit = 4

    orch = get_registry().resolve("loopguardorchestrator")
    with pytest.raises(AixonError, match="recursion"):
        orch.invoke([Message(role="user", content="go")])


def test_timeout_value_is_stored_and_defaults_none():
    make_echo_agent("tw")

    class TimeoutOrchestrator(Orchestrator):
        supervisor = make_llm()
        agents = [get_registry().resolve("tw")]
        timeout = 600

    assert get_registry().resolve("timeoutorchestrator").timeout == 600

    make_echo_agent("dtw")

    class DefaultTimeoutOrchestrator(Orchestrator):
        supervisor = make_llm()
        agents = [get_registry().resolve("dtw")]

    assert get_registry().resolve("defaulttimeoutorchestrator").timeout is None
