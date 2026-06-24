# tests/test_orchestrator_tier2.py
import pytest

from tests._fakes import make_echo_agent
from aixon.agents.orchestrator import Orchestrator
from aixon.exceptions import AixonError
from aixon.message import Message
from aixon.registry import get_registry
from aixon.state import END


def test_tier2_detected_and_entry_runs_first():
    make_echo_agent("triage")
    make_echo_agent("respond")

    class FlowOrchestrator(Orchestrator):
        nodes = {"triage": get_registry().resolve("triage"),
                 "respond": get_registry().resolve("respond")}
        entry = "triage"
        edges = [("triage", "respond"), ("respond", END)]

    assert FlowOrchestrator._tier == 2
    orch = get_registry().resolve("floworchestrator")
    out = orch.invoke([Message(role="user", content="hi")])
    # entry (triage) ran first, then respond ran last -> final assistant msg
    # comes from "respond" and still carries the original user content "hi".
    assert "hi" in out.content


def test_tier2_conditional_route_picks_one_path():
    make_echo_agent("triage")
    make_echo_agent("diagnose")
    make_echo_agent("respond")

    class TriageOrchestrator(Orchestrator):
        nodes = {"triage": get_registry().resolve("triage"),
                 "diagnose": get_registry().resolve("diagnose"),
                 "respond": get_registry().resolve("respond")}
        entry = "triage"
        edges = [("diagnose", "respond"), ("respond", END)]

        def route_triage(self, state) -> str:
            return "diagnose"  # always go to diagnose for this test

    orch = get_registry().resolve("triageorchestrator")
    out = orch.invoke([Message(role="user", content="hi")])
    # path: triage -> diagnose -> respond -> END
    assert "hi" in out.content


def test_tier2_list_fanout_runs_multiple_nodes():
    make_echo_agent("split")
    make_echo_agent("left")
    make_echo_agent("right")

    class FanoutOrchestrator(Orchestrator):
        nodes = {"split": get_registry().resolve("split"),
                 "left": get_registry().resolve("left"),
                 "right": get_registry().resolve("right")}
        entry = "split"
        edges = [("left", END), ("right", END)]

        def route_split(self, state):
            return ["left", "right"]  # parallel fan-out

    orch = get_registry().resolve("fanoutorchestrator")
    state = orch._compiled().invoke(
        orch._initial_state([Message(role="user", content="go")]),
        config=orch._run_config(),
    )
    produced = [m.content for m in state["messages"] if m.role == "assistant"]
    # Both branches ran (each echoed the user content "go").
    assert sum("go" in c for c in produced) >= 3  # split + left + right


def test_tier2_node_with_both_edge_and_route_raises():
    make_echo_agent("a")
    make_echo_agent("b")
    with pytest.raises(AixonError, match="exactly one exit"):
        type(
            "DupOrchestrator",
            (Orchestrator,),
            {
                "nodes": {"a": get_registry().resolve("a"),
                          "b": get_registry().resolve("b")},
                "entry": "a",
                "edges": [("a", "b"), ("b", END)],
                "route_a": lambda self, state: "b",  # also a route for 'a' -> error
            },
        )


def test_tier2_entry_not_in_nodes_raises():
    make_echo_agent("a")
    with pytest.raises(AixonError, match="entry"):
        type(
            "BadEntryOrchestrator",
            (Orchestrator,),
            {
                "nodes": {"a": get_registry().resolve("a")},
                "entry": "missing",
                "edges": [("a", END)],
            },
        )


def test_tier2_terminal_node_without_exit_is_allowed():
    make_echo_agent("a")

    class TerminalOrchestrator(Orchestrator):
        nodes = {"a": get_registry().resolve("a")}
        entry = "a"
        edges = []  # 'a' has no exit -> terminal, allowed

    orch = get_registry().resolve("terminalorchestrator")
    out = orch.invoke([Message(role="user", content="x")])
    assert "x" in out.content
