# aixon/agents/orchestrator.py
"""The Orchestrator subtype: a declarative, three-tier multi-agent coordinator
backed by LangGraph 1.x. Tier 1 = supervisor; Tier 2 = explicit graph (nodes +
entry + edges/route_<node>); Tier 3 = ``build_graph`` escape hatch.

The neutral boundary holds: ``invoke``/``stream`` speak only Message/Chunk;
LangGraph lives entirely inside this module and ``aixon.state``."""

from __future__ import annotations

import re
import time
from typing import Any, AsyncIterator, Iterator

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
        # Tier-1 structural validation: a worker named "supervisor" collides
        # with the internal supervisor node LangGraph builds for this tier.
        if cls._tier == 1:
            cls._validate_tier1()
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

    # ----- Tier-1 validation -------------------------------------------------

    @classmethod
    def _validate_tier1(cls) -> None:
        """A Tier-1 worker named 'supervisor' collides with the internal
        supervisor node LangGraph builds for this graph (_SUPERVISOR), which
        surfaces as an opaque LangGraph error on the first invoke rather than
        a clear message at definition time. Workers are already instantiated
        here (concrete Agent subclasses auto-instantiate at definition), so
        this can resolve names without building the graph.

        NOT symmetric with ``_validate_tier2``: this validation instantiates
        every worker to read its resolved ``.name`` (cheap and side-effect-
        free, since concrete Agent subclasses already auto-instantiate at
        class-definition time). Tier-2 validation never instantiates
        anything — it only inspects the declarative ``nodes``/``edges``
        dicts and ``route_<node>`` method names on the class."""
        for raw in cls.agents:
            inst = _instantiate(raw)
            if inst.name == _SUPERVISOR:
                raise AixonError(
                    f"Tier-1 worker name 'supervisor' collides with the "
                    f"internal supervisor node — rename the agent."
                )

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

    def _acompiled(self):
        """Async-node graph: worker nodes ``await agent.ainvoke`` so the whole
        chain is genuinely async (LangGraph can then cancel it at await points).
        Tier 3 reuses the user's compiled graph (it already exposes ``ainvoke``)."""
        cached = getattr(self, "_compiled_agraph", None)
        if cached is None:
            if self._tier == 3 or "build_graph" in type(self).__dict__:
                cached = self._compiled()
            elif self._tier == 1:
                cached = self._build_supervisor_graph(node_factory=self._make_aworker_node)
            else:
                cached = self._build_explicit_graph(node_factory=self._make_aworker_node)
            self._compiled_agraph = cached
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
        """Ask the supervisor LLM which worker should act next, or END.

        The supervisor sees the full conversation (including any worker answers
        appended earlier this run) plus the worker roster, and returns a worker
        name or DONE. Termination is the LLM's decision, backstopped by
        ``recursion_limit``. Safety net: we never END on an *unanswered* user
        turn — if the LLM declines to route while the last message is still the
        user's, dispatch to the first worker so the turn is always handled."""
        workers = self._worker_instances()
        messages = list(state.get("messages", []))
        choice = self._supervisor_choose(messages, workers)
        if choice in workers:
            return choice
        last_role = messages[-1].role if messages else "user"
        if last_role != "assistant" and workers:
            return next(iter(workers))  # don't strand an unanswered user turn
        return END

    def _supervisor_choose(self, messages: list[Message], workers: dict) -> str:
        """Run the supervisor LLM and return a worker name, or ``""`` for DONE.

        Parsing requires the reply to name exactly ONE worker (whole-word
        match; exact reply wins; DONE always wins). A reply naming zero or
        several workers gets one strict retry; if that also fails, ``""`` is
        returned and ``_route``'s safety net decides. Requires
        ``self.supervisor`` to expose ``complete(list[Message]) -> Message``."""
        roster = "\n".join(
            f"- {name}: {inst.description or 'no description'}"
            for name, inst in workers.items()
        )
        system = Message(
            role="system",
            content=(
                "You are a routing supervisor coordinating a team of workers. "
                "Read the conversation and decide who should act next.\n\n"
                f"Workers:\n{roster}\n\n"
                "Reply with ONLY the name of the worker that should handle the "
                "conversation next, or the single word DONE if the user's "
                "request has been fully answered. If no worker has answered the "
                "latest request yet, you MUST pick a worker."
            ),
        )
        reply = self.supervisor.complete([system, *messages])
        choice = self._parse_choice(reply.content, workers)
        if choice is not None:
            return choice
        # One strict retry for a reply that names zero or several workers:
        # cheaper than mis-routing, and a supervisor that still can't comply
        # falls through to _route's safety net ("" -> DONE / first worker).
        strict = Message(
            role="system",
            content=(
                f"Your previous reply ({(reply.content or '').strip()!r}) did "
                f"not name exactly one worker. Reply with EXACTLY one of: "
                f"{', '.join(workers)} — or the single word DONE."
            ),
        )
        reply = self.supervisor.complete([system, *messages, strict])
        choice = self._parse_choice(reply.content, workers)
        return choice if choice is not None else ""

    def _parse_choice(self, content: str | None, workers: dict) -> str | None:
        """Parse one supervisor reply: a worker name, ``""`` for DONE, or
        ``None`` when the reply names zero or several workers (caller re-asks
        once). Name matching is whole-word — bounded by anything that is not a
        word character or hyphen — so a name never fires inside a longer token
        ("order" does not match inside "order-history", "billing" does not
        match inside "billings")."""
        text = (content or "").strip().lower()
        # DONE wins BEFORE any name matching: a reply like "DONE — billing
        # already answered this" must terminate, not re-dispatch billing.
        if re.match(r"done\b", text):
            return ""
        for name in workers:                 # exact match wins
            if text == name.lower():
                return name
        matched = [
            name
            for name in workers
            if re.search(rf"(?<![\w-]){re.escape(name.lower())}(?![\w-])", text)
        ]
        if len(matched) == 1:
            return matched[0]
        return None  # zero or ambiguous — not a routing decision

    def _build_supervisor_graph(self, node_factory=None):
        node_factory = node_factory or self._make_worker_node
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
            graph.add_node(name, node_factory(inst))
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

    def _make_aworker_node(self, agent: Agent):
        """Async worker node: ``await agent.ainvoke`` so the async graph chain is
        genuinely async (cancellable at await points)."""
        async def node(state: GraphState) -> dict:
            result = await agent.ainvoke(list(state.get("messages", [])))
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

    def _build_explicit_graph(self, node_factory=None):
        node_factory = node_factory or self._make_worker_node
        instances = self._node_instances()
        graph = StateGraph(self.State)
        for name, inst in instances.items():
            graph.add_node(name, node_factory(inst))
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
        from langgraph.errors import GraphRecursionError

        graph = self._compiled()
        deadline = time.monotonic() + self.timeout if self.timeout else None
        try:
            result = graph.invoke(
                self._initial_state(messages), config=self._run_config()
            )
        except GraphRecursionError as exc:
            raise AixonError(
                f"Orchestrator '{type(self).__name__}' hit its recursion limit "
                f"({self.recursion_limit}). The graph looped without reaching "
                f"END. Raise `recursion_limit`, fix the routing, or set a "
                f"terminal edge. (LangGraph: {exc})"
            ) from exc
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
        from aixon.reasoning import reasoning_channel

        with reasoning_channel() as channel:
            final = self.invoke(messages)
            for line in channel.drain():
                yield Chunk(reasoning=line + "\n")
        if final.reasoning:
            yield Chunk(reasoning=final.reasoning)
        yield Chunk(content=final.content)
        yield Chunk(done=True)

    # ----- async neutral interface ----------------------------------------

    async def ainvoke(self, messages: list[Message]) -> Message:
        """Async invoke over the graph's ``ainvoke`` — does not block the event
        loop (worker nodes run via the executor). When ``timeout`` is set the run
        is wrapped in ``asyncio.wait_for``, so it is cancelled between supersteps
        if it overruns (a real deadline, not the sync post-hoc check)."""
        import asyncio

        from langgraph.errors import GraphRecursionError

        graph = self._acompiled()
        try:
            coro = graph.ainvoke(
                self._initial_state(messages), config=self._run_config()
            )
            if self.timeout:
                result = await asyncio.wait_for(coro, timeout=self.timeout)
            else:
                result = await coro
        except asyncio.TimeoutError:
            raise AixonError(
                f"Orchestrator '{type(self).__name__}' exceeded timeout="
                f"{self.timeout}s; the run was cancelled."
            )
        except GraphRecursionError as exc:
            raise AixonError(
                f"Orchestrator '{type(self).__name__}' hit its recursion limit "
                f"({self.recursion_limit}). The graph looped without reaching "
                f"END. Raise `recursion_limit`, fix the routing, or set a "
                f"terminal edge. (LangGraph: {exc})"
            ) from exc
        out_messages = result.get("messages", [])
        for m in reversed(out_messages):
            if m.role == "assistant":
                return m
        return Message(role="assistant", content="")

    async def astream(self, messages: list[Message]) -> "AsyncIterator[Chunk]":
        from aixon.reasoning import reasoning_channel

        with reasoning_channel() as channel:
            final = await self.ainvoke(messages)
            for line in channel.drain():
                yield Chunk(reasoning=line + "\n")
        if final.reasoning:
            yield Chunk(reasoning=final.reasoning)
        yield Chunk(content=final.content)
        yield Chunk(done=True)
