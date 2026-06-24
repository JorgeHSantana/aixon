# aixon/agents/orchestrator.py
"""The Orchestrator subtype: a declarative, three-tier multi-agent coordinator
backed by LangGraph 1.x. Tier 1 = supervisor; Tier 2 = explicit graph (nodes +
entry + edges/route_<node>); Tier 3 = ``build_graph`` escape hatch.

The neutral boundary holds: ``invoke``/``stream`` speak only Message/Chunk;
LangGraph lives entirely inside this module and ``aixon.state``."""

from __future__ import annotations

import time
from typing import Any, Iterator

from langgraph.graph import StateGraph

from aixon.agent import Agent
from aixon.exceptions import AixonError, CompositionCycleError
from aixon.logging import Logger
from aixon.message import Chunk, Message
from aixon.state import END, GraphState

_log = Logger("aixon.orchestrator")

# Sentinel marking the supervisor node in Tier 1 graphs.
_SUPERVISOR = "supervisor"


def _instantiate(agent: Any) -> Agent:
    """Accept an Agent subclass OR instance and return a usable instance.

    Concrete Agent subclasses auto-instantiate at definition (Plan 1), so the
    canonical instance is already in the registry. We resolve a class to its
    registered singleton; instances pass through."""
    if isinstance(agent, type):
        from aixon.registry import get_registry

        name = getattr(agent, "name", "") or agent.__name__.lower()
        try:
            return get_registry().resolve(name)
        except Exception:
            return agent()  # fallback: construct a fresh one
    return agent


class Orchestrator(Agent, abstract=True):
    _suffix = "Orchestrator"

    # Tier 1
    supervisor: Any = None          # LLM | None (typed loosely; LLM imported lazily by callers)
    agents: list = []

    # Tier 2
    nodes: dict = {}
    entry: str = ""
    edges: list = []

    # Runtime guards
    recursion_limit: int | None = 25
    timeout: int | None = None

    # Resolved at subclass-definition time.
    _tier: int = 0

    @classmethod
    def _validate_subclass(cls) -> None:
        # Runs via Agent.__init_subclass__ AFTER suffix/abstract-method checks
        # and BEFORE registration (cls()), so an invalid tier, a composition
        # cycle, or bad Tier-2 wiring raises WITHOUT leaving a ghost in the
        # registry. Do NOT override __init_subclass__ to validate after
        # super().__init_subclass__() — that registers first, then fails (the
        # register-then-validate ghost bug; see contract "Subtype validation
        # hook"). The base calls this hook only for concrete subclasses, so no
        # abstract=True guard is needed here.
        cls._tier = cls._detect_tier()
        # Composition-cycle guard (A) — always on (full impl in Task 7).
        cls._check_composition_cycle()
        # Tier-2 structural validation (full impl in Task 5).
        if cls._tier == 2:
            cls._validate_tier2()

    # ----- tier detection -------------------------------------------------

    @classmethod
    def _detect_tier(cls) -> int:
        if "build_graph" in cls.__dict__:
            return 3
        if cls.nodes:
            return 2
        if cls.supervisor is not None:
            return 1
        raise AixonError(
            f"Orchestrator '{cls.__name__}' declares no tier. Set one of: "
            f"`supervisor` (+ `agents`) for Tier 1, `nodes` (+ `entry`) for "
            f"Tier 2, or override `build_graph` for Tier 3."
        )

    # ----- guard A: structural composition-cycle detection (always on) ----

    @classmethod
    def _referenced_agent_classes(cls) -> list[type]:
        """Agent CLASSES this class composes, via `agents`, `nodes`, and any
        `tools` declared on the class. Instances are mapped to their class.
        Non-Agent entries (LLMs, plain callables, AgentTool) are ignored — only
        agent→agent composition can form a structural cycle."""
        refs: list[type] = []
        seen: set[int] = set()

        def add(obj: Any) -> None:
            klass = obj if isinstance(obj, type) else type(obj)
            if isinstance(klass, type) and issubclass(klass, Agent):
                if id(klass) not in seen:
                    seen.add(id(klass))
                    refs.append(klass)

        for entry in getattr(cls, "agents", []) or []:
            add(entry)
        for entry in (getattr(cls, "nodes", {}) or {}).values():
            add(entry)
        for entry in getattr(cls, "tools", []) or []:
            add(entry)
        return refs

    @classmethod
    def _check_composition_cycle(cls) -> None:
        path: list[type] = []

        def walk(node_cls: type) -> None:
            if node_cls in path:
                chain = " -> ".join(c.__name__ for c in path + [node_cls])
                raise CompositionCycleError(
                    f"Composition cycle detected: {chain}. An agent cannot "
                    f"(transitively) include itself as a worker/node/tool. "
                    f"Break the cycle by removing one of the references."
                )
            path.append(node_cls)
            neighbors = getattr(node_cls, "_referenced_agent_classes", None)
            if callable(neighbors):
                for nxt in node_cls._referenced_agent_classes():
                    walk(nxt)
            path.pop()

        walk(cls)

    # ----- Tier-2 validation (contract §3.2) -------------------------------

    @classmethod
    def _validate_tier2(cls) -> None:
        nodes = cls.nodes
        edge_srcs = {src for src, _ in cls.edges}
        for name in nodes:
            has_edge = name in edge_srcs
            has_route = callable(getattr(cls, f"route_{name}", None))
            if has_edge and has_route:
                raise AixonError(
                    f"Node '{name}' in Orchestrator '{cls.__name__}' declares "
                    f"both a fixed edge in `edges` and a `route_{name}` method. "
                    f"A node must have exactly one exit form — remove one."
                )
            # neither -> terminal node (allowed)
        for src, dst in cls.edges:
            if src not in nodes:
                raise AixonError(
                    f"Edge ({src!r}, ...) in '{cls.__name__}' references unknown "
                    f"node '{src}'. Known nodes: {sorted(nodes)}."
                )
            if dst is not END and dst not in nodes:
                raise AixonError(
                    f"Edge (..., {dst!r}) in '{cls.__name__}' references unknown "
                    f"node '{dst}'. Use a node name or aixon.END."
                )
        if cls.entry not in nodes:
            raise AixonError(
                f"Orchestrator '{cls.__name__}' has entry={cls.entry!r}, which is "
                f"not a node. Set `entry` to one of: {sorted(nodes)}."
            )

    # ----- declarative state ----------------------------------------------

    @property
    def State(self) -> type:
        """The state TypedDict for this orchestrator. Users may declare a
        nested ``class State(GraphState): ...``; otherwise default GraphState."""
        declared = type(self).__dict__.get("State")
        return declared if declared is not None else GraphState

    # ----- graph build (lazy, cached) -------------------------------------

    def _compiled(self):
        cached = getattr(self, "_compiled_graph", None)
        if cached is None:
            cached = self.build_graph()
            self._compiled_graph = cached
        return cached

    def build_graph(self):
        """Build & compile the LangGraph graph for this orchestrator's tier.
        Tier 3 users OVERRIDE this method to return their own compiled graph."""
        if self._tier == 1:
            return self._build_supervisor_graph()
        if self._tier == 2:
            return self._build_explicit_graph()
        raise AixonError(  # pragma: no cover - Tier 3 overrides build_graph
            f"Orchestrator '{type(self).__name__}' is Tier 3 but did not "
            f"override build_graph()."
        )

    # ----- Tier 1: minimal hand-rolled supervisor -------------------------

    def _worker_instances(self) -> dict[str, Agent]:
        out: dict[str, Agent] = {}
        for raw in self.agents:
            inst = _instantiate(raw)
            out[inst.name] = inst
        return out

    def _route_supervisor(self, state: GraphState) -> str:
        """Pick the next worker, or END. Default: first worker that has not yet
        emitted an assistant message this run, else END. A real LLM-driven
        supervisor replaces this hook; the declarative surface stays the same.

        We track which workers already ran by counting assistant messages: the
        initial state has only the user message, and each worker appends exactly
        one assistant message, so the Nth assistant message means N workers have
        run. This terminates after every worker runs once."""
        workers = list(self._worker_instances().items())
        ran = sum(
            1 for m in state.get("messages", []) if m.role == "assistant"
        )
        if ran < len(workers):
            return workers[ran][0]  # next un-run worker's node name
        return END

    def _build_supervisor_graph(self):
        workers = self._worker_instances()
        if not workers:
            raise AixonError(
                f"Tier 1 Orchestrator '{type(self).__name__}' has an empty "
                f"`agents` list. Add at least one worker Agent."
            )
        graph = StateGraph(self.State)

        def supervisor_node(state: GraphState) -> dict:
            return {}  # routing happens in the conditional edge

        graph.add_node(_SUPERVISOR, supervisor_node)
        for name, inst in workers.items():
            graph.add_node(name, self._make_worker_node(inst))
            graph.add_edge(name, _SUPERVISOR)  # back to supervisor after each worker

        graph.set_entry_point(_SUPERVISOR)
        path_map = {name: name for name in workers}
        path_map[END] = END
        graph.add_conditional_edges(_SUPERVISOR, self._route_supervisor, path_map)
        return graph.compile()

    def _make_worker_node(self, agent: Agent):
        def node(state: GraphState) -> dict:
            result = agent.invoke(list(state.get("messages", [])))
            return {"messages": result}

        return node

    # ----- Tier 2: explicit graph (nodes/entry/edges + route_<node>) ------

    def _node_instances(self) -> dict[str, Agent]:
        return {name: _instantiate(raw) for name, raw in self.nodes.items()}

    def _wrap_router(self, node_name: str):
        method = getattr(self, f"route_{node_name}")

        def router(state: GraphState):
            return method(state)  # returns str (one path) or list[str] (fan-out)

        return router

    def _build_explicit_graph(self):
        instances = self._node_instances()
        graph = StateGraph(self.State)
        for name, inst in instances.items():
            graph.add_node(name, self._make_worker_node(inst))
        graph.set_entry_point(self.entry)

        edge_srcs = {src for src, _ in self.edges}
        for src, dst in self.edges:
            graph.add_edge(src, dst)
        for name in instances:
            if name in edge_srcs:
                continue
            if callable(getattr(self, f"route_{name}", None)):
                graph.add_conditional_edges(name, self._wrap_router(name))
            else:
                graph.add_edge(name, END)  # terminal node -> END
        return graph.compile()

    # ----- run config (guard B; recursion error wrapping added in Task 8) -

    def _run_config(self) -> dict:
        config: dict[str, Any] = {}
        if self.recursion_limit is not None:
            config["recursion_limit"] = self.recursion_limit
        return config

    def _initial_state(self, messages: list[Message]) -> dict:
        return {"messages": list(messages), "reasoning": []}

    # ----- neutral interface ----------------------------------------------

    def invoke(self, messages: list[Message]) -> Message:
        graph = self._compiled()
        deadline = time.monotonic() + self.timeout if self.timeout else None
        result = graph.invoke(
            self._initial_state(messages), config=self._run_config()
        )
        if deadline is not None and time.monotonic() > deadline:
            raise AixonError(
                f"Orchestrator '{type(self).__name__}' exceeded timeout="
                f"{self.timeout}s."
            )
        out_messages = result.get("messages", [])
        for m in reversed(out_messages):
            if m.role == "assistant":
                return m
        return Message(role="assistant", content="")

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        # Minimal streaming: run to completion, then emit the final assistant
        # content as one Chunk + done. (Reasoning propagation lands in Task 9.)
        final = self.invoke(messages)
        if final.reasoning:
            yield Chunk(reasoning=final.reasoning)
        yield Chunk(content=final.content)
        yield Chunk(done=True)
