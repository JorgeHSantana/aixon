# tests/test_orchestrator_tier3.py
from langgraph.graph import StateGraph

from tests._fakes import make_echo_agent
from aixon.agents.orchestrator import Orchestrator
from aixon.message import Message
from aixon.registry import get_registry
from aixon.state import GraphState, END


def test_tier3_detected_when_build_graph_overridden():
    make_echo_agent("worker")

    class CustomOrchestrator(Orchestrator):
        def build_graph(self):
            g = StateGraph(GraphState)
            inst = get_registry().resolve("worker")

            def node(state):
                return {"messages": inst.invoke(list(state.get("messages", [])))}

            g.add_node("only", node)
            g.set_entry_point("only")
            g.add_edge("only", END)
            return g.compile()

    assert CustomOrchestrator._tier == 3


def test_tier3_runs_user_graph():
    make_echo_agent("worker")

    class RawOrchestrator(Orchestrator):
        def build_graph(self):
            g = StateGraph(GraphState)
            inst = get_registry().resolve("worker")

            def node(state):
                return {"messages": inst.invoke(list(state.get("messages", [])))}

            g.add_node("only", node)
            g.set_entry_point("only")
            g.add_edge("only", END)
            return g.compile()

    orch = get_registry().resolve("raworchestrator")
    out = orch.invoke([Message(role="user", content="ping")])
    assert "ping" in out.content
