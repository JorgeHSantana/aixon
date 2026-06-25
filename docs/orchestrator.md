# Orchestrator

`Orchestrator` is the `Agent` subtype for coordinating multiple agents.
Like all agents, it exposes `invoke`, `stream`, and `as_tool` — so an orchestrator
can be a node inside another orchestrator, a tool inside a `ToolAgent`, or the
top-level entry point served by the `Server`.

Three tiers handle different levels of complexity. Pick the lowest tier that
covers your case — you can always promote to a higher tier later.

```python
from aixon import Orchestrator, LLM
from aixon.state import END, GraphState
```

---

## Tier 1 — Supervisor (default)

The supervisor pattern is the simplest: an LLM decides which worker agent handles
each turn and loops until it decides the conversation is complete.

```python
class SupportOrchestrator(Orchestrator):
    description = "Routes support tickets to the right specialist"
    supervisor  = LLM("gpt-4o-mini")
    agents      = [BillingAgent, TechAgent, PlannerAgent]
```

**How it works:** the supervisor LLM receives the conversation history and the
list of available workers (names + descriptions). It selects the next worker,
routes the turn, receives the result, and decides whether to call another worker
or return the final answer.

**When to use:** the routing logic is best expressed in natural language — "send
billing questions to BillingAgent, technical questions to TechAgent."

---

## Tier 2 — Explicit graph

Use Tier 2 when the routing is deterministic (or conditionally deterministic) and
you want it expressed in code rather than natural language.

```python
class TriageOrchestrator(Orchestrator):
    description = "Triages issues with conditional routing"

    nodes = {
        "triage":   TriageAgent,
        "diagnose": DiagnosisAgent,
        "respond":  PlannerAgent,
    }
    entry = "triage"
    edges = [
        ("diagnose", "respond"),
        ("respond",  END),
    ]

    def route_triage(self, state) -> str:
        return "diagnose" if state["needs_diagnosis"] else "respond"
```

### Entry and execution order

**`entry` determines which node runs first.** The `edges` list is wiring, not
a sequence — the order of tuples in `edges` is irrelevant to execution order.

To understand what runs when, trace the graph:
1. `"triage"` runs (it is `entry`).
2. `route_triage` is called to decide the next node.
3. The chosen node (`"diagnose"` or `"respond"`) runs.
4. If `"diagnose"` ran, the fixed edge `("diagnose", "respond")` sends execution
   to `"respond"`.
5. The fixed edge `("respond", END)` terminates the graph.

### Node exit forms

Each node has **exactly one** exit form:

| Exit form | How to declare |
|---|---|
| Fixed (unconditional) edge | A tuple `(node, dst)` in `edges`. |
| Conditional/fan-out | A `route_<node>` method. |

Declaring **both** for the same node raises `AixonError` at import time (ambiguous
exit). Declaring **neither** makes the node terminal (equivalent to `→ END`).

**`END`** is a sentinel imported from `aixon.state`:

```python
from aixon.state import END
edges = [("respond", END)]
```

### Two kinds of branching via `route_<node>`

**1. Conditional — choose one next node:**

```python
def route_triage(self, state) -> str:
    return "diagnose" if state["needs_diagnosis"] else "respond"
```

The method returns a single node name. Execution continues at that node.

**2. Fan-out — run multiple nodes in parallel:**

```python
def route_research(self, state) -> list[str]:
    return ["web_search", "knowledge_base", "internal_docs"]
```

The method returns a **list** of node names. All listed nodes run in parallel;
the graph waits for all to complete before moving to the next step.

---

## Tier 3 — LangGraph escape hatch

Tier 3 gives you raw LangGraph. Override `build_graph` and return a compiled
graph. The framework runs it as-is.

```python
class WeirdOrchestrator(Orchestrator):
    description = "Custom graph with cycles and conditional edges"

    def build_graph(self):
        from langgraph.graph import StateGraph
        g = StateGraph(self.State)
        g.add_node("analyze", AnalysisAgent().invoke)
        g.add_node("refine",  RefineAgent().invoke)
        g.add_conditional_edges("analyze", lambda s: "refine" if s["needs_refinement"] else END)
        g.add_edge("refine", "analyze")   # a legitimate cycle inside the graph
        g.set_entry_point("analyze")
        return g.compile()
```

Use Tier 3 only when Tier 2's declarative surface cannot express your graph
(e.g., dynamic nodes, LangGraph-native subgraphs, custom reducers).

---

## Tier detection order

`aixon` detects the tier in this order:

1. `build_graph` is overridden → **Tier 3**
2. `nodes` is non-empty → **Tier 2**
3. `supervisor` and `agents` are set → **Tier 1**
4. None of the above on a concrete subclass → `AixonError` at import time

---

## State

`GraphState` is the default state type — carries `messages` and `reasoning`.
You rarely need to touch it.

```python
from aixon.state import GraphState

class GraphState(TypedDict, total=False):
    messages:  Annotated[list[Message], add_messages_neutral]
    reasoning: list[str]
```

Add fields by nesting a `State` class inside your orchestrator:

```python
class TriageOrchestrator(Orchestrator):
    class State(GraphState):
        needs_diagnosis: bool = False
```

Your `route_<node>` methods receive this extended state:

```python
    def route_triage(self, state) -> str:
        return "diagnose" if state["needs_diagnosis"] else "respond"
```

---

## Orchestrator as a tool

Because `Orchestrator` implements the full `Agent` interface, you can use any
orchestrator as a tool inside a `ToolAgent` or as a node in another orchestrator:

```python
class RouterAgent(ToolAgent):
    llm   = LLM("gpt-4o-mini")
    tools = [SupportOrchestrator().as_tool(description="Handle support tickets")]
```

Each `invoke` call on the wrapped orchestrator gets its own state — conversation
history never leaks between calls.

---

## Recursion guards

### A — Composition cycle detection (always on)

A composition cycle is when agent A uses agent B as a tool and agent B uses agent
A as a tool (directly or transitively). This would create an infinite expansion
at build time.

`aixon` detects composition cycles in `__init_subclass__` by walking the
composition graph — agents referenced via `agents`, `nodes`, or `tools`. If any
class appears twice on the same path, `CompositionCycleError` is raised at import
time, before the server starts:

```python
class PingAgent(ToolAgent):
    llm   = LLM("gpt-4o-mini")
    tools = [PongAgent().as_tool()]   # CompositionCycleError if PongAgent uses PingAgent

class PongAgent(ToolAgent):
    llm   = LLM("gpt-4o-mini")
    tools = [PingAgent().as_tool()]   # closes the cycle
```

Note: a cycle **within** a LangGraph graph (a node that loops back to a previous
node) is legitimate and allowed — it is bounded by Guard B below.

### B — Runtime depth / loop limit (declarative)

```python
class ResearchOrchestrator(Orchestrator):
    supervisor       = LLM("gpt-4o-mini")
    agents           = [SearchAgent, SummarizeAgent]
    recursion_limit  = 50    # LangGraph supersteps. Default: 25. None = no cap.
    timeout          = 600   # Wall-clock backstop in seconds. None = no backstop.
```

`recursion_limit` is passed to LangGraph's compiled graph config. `timeout` is
enforced as a wall-clock backstop. Setting both to `None` is allowed but not
recommended — cost and time are then unbounded.
