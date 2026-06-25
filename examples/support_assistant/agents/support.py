"""SupportOrchestrator — a Tier 2 `Orchestrator` with conditional routing.

Graph:

    triage ──route_triage──▶ orders     (terminal ▶ END)
                          └─▶ knowledge  (terminal ▶ END)

`triage` (an `LLMAgent`) classifies the request; ``route_triage`` reads its
one-word answer and dispatches to the matching specialist `ToolAgent`. The
specialists are terminal nodes — their answer is the orchestrator's answer.

This is the public entry point (the workers are ``hidden``). It carries
``aliases`` so the server/CLI accept "assistant" or "help" as well as "support".
"""

from __future__ import annotations

from aixon import Orchestrator
from aixon.state import GraphState

from agents.knowledge_agent import KnowledgeAgent
from agents.orders_agent import OrdersAgent
from agents.triage import TriageAgent


class SupportOrchestrator(Orchestrator):
    name = "support"
    aliases = ["assistant", "help"]
    description = "Acme customer-support assistant (routes to FAQ or orders)."

    # Tier 2: explicit graph. Node keys are graph-internal names.
    nodes = {
        "triage": TriageAgent,
        "knowledge": KnowledgeAgent,
        "orders": OrdersAgent,
    }
    entry = "triage"
    # 'knowledge' and 'orders' declare no edge and no route_* -> terminal (END).

    def route_triage(self, state: GraphState) -> str:
        """Read triage's one-word verdict and pick the specialist node."""
        last = state["messages"][-1]
        verdict = (last.content or "").strip().lower()
        return "orders" if "order" in verdict else "knowledge"
