# aixon Orchestrator Implementation Plan (Plan 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `Orchestrator` subtype of `Agent` — a declarative, three-tier multi-agent coordinator backed by **LangGraph 1.x** — plus the default `GraphState` and the two recursion guards. The three tiers (Tier 1 supervisor, Tier 2 explicit graph, Tier 3 `build_graph` escape hatch) all share the same `Agent` interface (`invoke`/`stream`/`as_tool`), so the server, registry, and protocol layer never need to know which tier an orchestrator uses.

**Architecture:** `Orchestrator(Agent, abstract=True)` sets `_suffix = "Orchestrator"`. Concrete subclasses are suffix-validated and auto-registered by Plan 1's `Agent.__init_subclass__`. Tier detection and Tier-2 structural validation run in `Orchestrator._validate_subclass()` — the hook `Agent.__init_subclass__` invokes AFTER the suffix/abstract-method checks and BEFORE registration — so an invalid tier or composition cycle raises without leaving a ghost in the registry. (Do NOT validate inside an `__init_subclass__` override after `super().__init_subclass__()`: that registers first, then fails — the register-then-validate ghost bug; see contract "Subtype validation hook".) The compiled LangGraph graph is built lazily on first `invoke`/`stream` and cached. Nodes are `Agent` instances; each node wraps an agent's neutral `invoke` into a LangGraph node function that reads `state["messages"]`, runs the agent, and appends the result via the `add_messages_neutral` reducer. The neutral boundary holds: `Orchestrator.invoke`/`stream` speak only `Message`/`Chunk`; LangGraph/LangChain types live strictly inside `orchestrator.py` and `state.py`.

**Two recursion guards (distinct):**
- **(A) Composition cycle (structural), always on:** walk the static composition graph of nested agents (referenced via `agents`, `nodes`, or any agent declaring `tools`) at subclass-definition time; revisiting a class already on the current DFS path raises `CompositionCycleError`. A loop *inside* one LangGraph graph (a node edging back) is legitimate and is bounded by guard B, not flagged by guard A.
- **(B) Runtime depth/loop:** `recursion_limit` is passed into the compiled graph's run config (`graph.invoke(state, config={"recursion_limit": N})`); `timeout` is a wall-clock backstop enforced around the run.

**Tech Stack:** Python 3.11+, **langgraph 1.x** (validated at langgraph 1.2; `langchain` 1.3 / `langchain-core` 1.4), the neutral types from Plan 1, and the hermetic fakes (`tests/_fakes.py`: `make_llm`, `make_echo_agent`, `FakeChatModel`) plus `LLM`/`LLMAgent`/`ToolAgent`/`emit_reasoning`/`reasoning_channel` from Plans 2–3 (all merged before this plan — see contract §9.3).

## Global Constraints

- `requires-python >= 3.11` — copied verbatim from the contract / spec packaging section.
- **Neutral boundary (binding):** `Orchestrator.invoke`/`stream` and the public API speak ONLY `Message`/`Chunk`. LangGraph/LangChain objects may be used INTERNALLY (inside `aixon/state.py` and `aixon/agents/orchestrator.py`) but never cross the `Agent` interface. `state.py`/`orchestrator.py` must NOT import from `aixon.server` or `aixon.providers`.
- **Hermetic, offline tests:** no test may require a real provider SDK or network. Orchestrator nodes are fake-LLM agents from `tests/_fakes.py` (built on `LLM("fake-1", provider="fake")` — the hermetic handle established in Plan 2, contract §1.5/§1.6/§9.1). `tests/conftest.py` already provides an autouse `reset_registry` fixture (Plan 1). NO `tests/__init__.py`.
- **`tests/_fakes.py` is owned by Plan 2 (contract §9.1) — do NOT redefine it.** Plan 4 imports from it directly:
  `from tests._fakes import make_llm, make_echo_agent`. Its binding surface (contract §9.1):
  - `make_llm(**params) -> LLM` — returns `LLM("fake-1", provider="fake", **params)`.
  - `make_echo_agent(name: str = "echo", *, hidden: bool = False)` — returns/registers a concrete `Agent` subclass whose `invoke` echoes the last message and whose `stream` yields one content `Chunk` then `done`. **The echo content is the registered agent's `name` followed by the echoed user text** (this plan's tests only depend on the agent's `name` and the last-user content appearing in the output; see Task 3 for the exact contract used and a fallback if Plan 2's echo format differs).
  - `FakeChatModel`, `register_fake_provider`, `FAKE_MODEL`, `FAKE_PROVIDER` also live there.
- **Plans 2 and 3 are merged before Plan 4 (contract §9.3): NO try/except fallback shims.** Import `LLM`, `emit_reasoning`, `reasoning_channel` directly from `aixon`; import `make_llm`/`make_echo_agent` directly from `tests._fakes` (the package-qualified path Plans 1-3 already use — see the Task 3 note for why the unqualified `_fakes` import must NOT be used). If an import fails, that is a real ordering bug to fix, not something to paper over.
- **Abstract subtype:** `Orchestrator` is declared `class Orchestrator(Agent, abstract=True)` and sets **`_suffix = "Orchestrator"`**. Concrete user subclasses inherit, get suffix-validated (`*Orchestrator`) and auto-registered by Plan 1's machinery.
- **`recursion_limit` default = 25** (matches LangGraph's own default). `None` = no cap (still bounded by `timeout`). `timeout` default `None`.
- **Composition-cycle guard (A) is ALWAYS ON and NOT disableable** — it runs in `_validate_subclass()` (before registration) for every concrete `Orchestrator` subclass and raises `CompositionCycleError` on a structural cycle.
- **Error tone:** state what was got and how to fix it (mirror Plan 1).
- **`END` sentinel:** re-export LangGraph's `END` from `aixon.state`; concrete orchestrators and the public API use `aixon.END` / `aixon.state.END`, never `langgraph.graph.END` directly.
- **Dependencies (contract §9.2 — binding):** `langgraph` lives in the **`llm` extra** (ToolAgent and Orchestrator both need it). There is **NO separate `orchestration` extra**. Plan 2 introduces the `llm` extra as `["langchain>=1.0", "langchain-core>=1.0", "langgraph>=1.0"]`; if (and only if) `langgraph>=1.0` is somehow absent from `llm`, Plan 4 adds it there and ensures it is in `all`. **Never pin a `<1` ceiling anywhere.**
- **LangGraph 1.x API (validated at langgraph 1.2.6 — use these EXACT imports):**
  - `from langgraph.graph import StateGraph, END` (both present).
  - `from langchain.agents import create_agent` (for any agent worker; NOT `AgentExecutor`, NOT the deprecated `langgraph.prebuilt.create_react_agent`). Plan 4 does not call `create_agent` directly — nodes invoke neutral `Agent.invoke` — but any agent worker construction elsewhere uses this import.
  - Compiled graph: `graph = builder.compile()`; run `graph.invoke(state)` / `graph.stream(state)`; pass guards via `graph.invoke(state, config={"recursion_limit": N})`.
- **Dedicated virtualenv (contract §9.5 — REQUIRED, do not reuse another project's venv):**
  ```bash
  cd /Users/jorge/Documents/Git/aixon
  python3 -m venv .venv                      # .venv is git-ignored; created once in Plan 2 Task 0
  .venv/bin/python -m pip install -e ".[dev,llm]"
  ```
  Every run/install step in this plan uses `.venv/bin/python -m pytest ...` and `.venv/bin/python -m pip ...` — NEVER a bare `pytest` (the console script can carry a stale shebang) and NEVER another project's interpreter.
- **Commits:** Co-Authored-By trailer per repo convention.

---

### Task 1: Confirm `langgraph` is in the `llm` extra (no new extra)

**Files:**
- Verify (and only if necessary, modify): `pyproject.toml`

**Interfaces:**
- Consumes: existing `[project.optional-dependencies]` table (Plan 2 added the `llm` extra).
- Produces: `langgraph>=1.0` present in the `llm` extra and in `all`. **No `orchestration` extra is created** (contract §9.2).

- [ ] **Step 1: Write the test**

```python
# tests/test_orchestration_packaging.py
import tomllib
from pathlib import Path


def _pyproject() -> dict:
    root = Path(__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


def test_langgraph_lives_in_the_llm_extra():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "llm" in extras, "Plan 2 must have created the 'llm' extra"
    assert any(dep.startswith("langgraph") for dep in extras["llm"]), (
        "langgraph must live in the 'llm' extra (contract §9.2) — "
        "there is NO separate 'orchestration' extra"
    )


def test_no_orchestration_extra_exists():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "orchestration" not in extras, (
        "langgraph belongs in 'llm', not a separate 'orchestration' extra "
        "(contract §9.2)"
    )


def test_all_extra_includes_langgraph():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert any(dep.startswith("langgraph") for dep in extras["all"])
```

- [ ] **Step 2: Run test to verify current state**

Run: `.venv/bin/python -m pytest tests/test_orchestration_packaging.py -v`
Expected: PASS if Plan 2 already declared `llm = [..., "langgraph>=1.0"]` and added it to `all` (the normal case). If `test_langgraph_lives_in_the_llm_extra` or `test_all_extra_includes_langgraph` FAILS, proceed to Step 3; if `test_no_orchestration_extra_exists` fails, an upstream plan wrongly created `orchestration` — remove it and fold `langgraph` into `llm`.

- [ ] **Step 3: Edit `pyproject.toml` only if a test failed**

The `llm` extra must read exactly (contract §9.2; do not remove anything Plan 2/3 added):

```toml
llm = ["langchain>=1.0", "langchain-core>=1.0", "langgraph>=1.0"]
```

Ensure `langgraph>=1.0` is also present in the `all` list (union of every extra). Do NOT add a `<1` ceiling. Do NOT create an `orchestration` extra.

- [ ] **Step 4: Install the llm extra into the dedicated venv**

Run: `cd /Users/jorge/Documents/Git/aixon && .venv/bin/python -m pip install -e ".[dev,llm]"`
Expected: `langgraph` (1.x) installed; `Successfully installed ... aixon`.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_orchestration_packaging.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/test_orchestration_packaging.py
git commit -m "test(orchestrator): assert langgraph lives in the llm extra (no orchestration extra)"
```

---

### Task 2: `GraphState` + `add_messages_neutral` reducer + `END` re-export

**Files:**
- Create: `aixon/state.py`
- Modify: `aixon/__init__.py` (export `GraphState`, `END`)
- Test: `tests/test_state.py`

**Interfaces:**
- Consumes: `aixon.message.Message`; `langgraph.graph.END`.
- Produces:
  - `aixon.state.add_messages_neutral(existing: list[Message] | None, new: list[Message] | Message | None) -> list[Message]` — a LangGraph reducer that appends neutral `Message` objects. Accepts a single `Message` or a list (LangGraph passes node return values straight to the reducer); `None` left operand treated as `[]`; returns a NEW list (never mutates `existing`).
  - `aixon.state.GraphState(TypedDict, total=False)` with `messages: Annotated[list[Message], add_messages_neutral]` and `reasoning: list[str]`.
  - `aixon.state.END` — re-export of `langgraph.graph.END`.
  - `GraphState` and `END` re-exported from `aixon`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_state.py
from typing import get_args, get_type_hints

from aixon.state import GraphState, add_messages_neutral, END
from aixon.message import Message


def test_reducer_appends_list_to_existing():
    existing = [Message(role="user", content="a")]
    out = add_messages_neutral(existing, [Message(role="assistant", content="b")])
    assert [m.content for m in out] == ["a", "b"]


def test_reducer_accepts_single_message():
    out = add_messages_neutral([], Message(role="assistant", content="solo"))
    assert [m.content for m in out] == ["solo"]


def test_reducer_treats_none_left_as_empty():
    out = add_messages_neutral(None, [Message(role="user", content="x")])
    assert [m.content for m in out] == ["x"]


def test_reducer_does_not_mutate_existing():
    existing = [Message(role="user", content="a")]
    add_messages_neutral(existing, [Message(role="assistant", content="b")])
    assert [m.content for m in existing] == ["a"]  # unchanged


def test_reducer_none_right_is_noop():
    existing = [Message(role="user", content="a")]
    out = add_messages_neutral(existing, None)
    assert [m.content for m in out] == ["a"]


def test_graphstate_messages_field_uses_reducer():
    hints = get_type_hints(GraphState, include_extras=True)
    annotated = hints["messages"]  # Annotated[list[Message], add_messages_neutral]
    args = get_args(annotated)
    assert add_messages_neutral in args


def test_graphstate_is_total_false():
    state: GraphState = {"messages": [Message(role="user", content="hi")]}
    assert state["messages"][0].content == "hi"


def test_end_is_reexported_from_langgraph():
    from langgraph.graph import END as LG_END
    assert END is LG_END
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aixon.state'`.

- [ ] **Step 3: Write the implementation**

```python
# aixon/state.py
"""Default LangGraph state for aixon Orchestrators. Carries the neutral
conversation (``messages``) and accumulated ``reasoning``. Users subclass
``GraphState`` to add fields (declared as ``class State(GraphState): ...``
inside their Orchestrator).

This module is the ONE place outside ``agents/orchestrator.py`` that touches
LangGraph, and only to re-export the ``END`` sentinel so concrete orchestrators
never import ``langgraph`` directly."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import END as END  # re-export; aixon.END is the public name

from aixon.message import Message


def add_messages_neutral(
    existing: list[Message] | None,
    new: list[Message] | Message | None,
) -> list[Message]:
    """LangGraph reducer for the neutral ``messages`` channel.

    Appends neutral ``Message`` objects without mutating ``existing``. LangGraph
    passes a node's return value (``state["messages"]`` update) as ``new``; that
    value may be a single ``Message`` or a list. ``None`` on either side is
    treated as empty so partial state updates are safe.
    """
    base: list[Message] = list(existing) if existing else []
    if new is None:
        return base
    if isinstance(new, Message):
        return base + [new]
    return base + list(new)


class GraphState(TypedDict, total=False):
    """Default orchestrator state. ``total=False`` makes every key optional, so
    nodes may return partial updates and subclasses may add fields freely."""

    messages: Annotated[list[Message], add_messages_neutral]
    reasoning: list[str]
```

In `aixon/__init__.py`, add the import line (alphabetically near the others) and append the two names to `__all__`. **Do not remove anything Plans 2/3 already added.** Task 4 adds `Orchestrator` to the same file.

```python
# aixon/__init__.py  — add to the import block:
from aixon.state import END, GraphState
# aixon/__init__.py  — append to __all__ (keep grouped): "GraphState", "END"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_state.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add aixon/state.py aixon/__init__.py tests/test_state.py
git commit -m "feat(orchestrator): GraphState, add_messages_neutral reducer, END re-export"
```

---

### Task 3: Smoke-test the shared fakes (owned by Plan 2)

**Files:**
- Test: `tests/test_orchestrator_fakes_smoke.py` (consumes `tests/_fakes.py` — already present from Plan 2)

> `tests/_fakes.py` exists (Plan 2 created it). This task does NOT create or modify it — it confirms the surface Plan 4 relies on (`make_llm`, `make_echo_agent`) behaves as the contract describes, so later tasks build on a known foundation. Tests import `from tests._fakes import ...` (the package-qualified path — `tests` has no `__init__.py` but resolves as a Python 3 implicit namespace package, since the repo root is on `sys.path`). **Always use this qualified form, never the unqualified `from _fakes import ...`**: pytest's default rootdir import mode also makes the bare `_fakes` name resolvable, and Plans 1-3's existing tests already import the qualified `tests._fakes` form — mixing the two forms loads `tests/_fakes.py` as two distinct module objects with two distinct `FakeChatModel` classes, breaking `isinstance` checks across files in the same test session. There is still NO `tests/__init__.py` file on disk.

**Interfaces (contract §9.1 — binding):**
- `make_llm(**params) -> LLM` → `LLM("fake-1", provider="fake", **params)`.
- `make_echo_agent(name: str = "echo", *, hidden: bool = False)` → defines + registers a concrete `Agent` whose `invoke` echoes the last message and whose `stream` yields one content `Chunk` then `done`.

> **Echo-content contract used by Plan 4 (resolve any ambiguity here):** Plan 4's Tier tests only assert that (a) the resolving agent runs and (b) the last user content reaches the node. To stay robust against Plan 2's exact echo wording, every later test asserts on **substrings derived from the input** (e.g. the user text `"hi"` appears in the output) and on the registered **agent name**, NOT on a `tag:` prefix. This makes the tests independent of whether Plan 2's echo prepends the agent name, the word "echo", or nothing. The smoke test below pins exactly what we depend on.

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_orchestrator_fakes_smoke.py
from tests._fakes import make_llm, make_echo_agent
from aixon.llm import LLM
from aixon.message import Chunk, Message
from aixon.registry import get_registry


def test_make_llm_returns_fake_handle():
    llm = make_llm()
    assert isinstance(llm, LLM)
    assert llm.model == "fake-1"
    assert llm._provider_name == "fake"


def test_make_echo_agent_registers_and_echoes_last_user():
    make_echo_agent("alpha")
    agent = get_registry().resolve("alpha")
    out = agent.invoke([Message(role="user", content="ping")])
    assert out.role == "assistant"
    assert "ping" in out.content  # last user content is echoed back


def test_make_echo_agent_stream_yields_content_then_done():
    make_echo_agent("beta")
    agent = get_registry().resolve("beta")
    chunks = list(agent.stream([Message(role="user", content="go")]))
    assert isinstance(chunks[-1], Chunk)
    assert chunks[-1].done is True
    assert any("go" in c.content for c in chunks)


def test_make_echo_agent_distinct_names_are_distinct_agents():
    make_echo_agent("one")
    make_echo_agent("two")
    assert get_registry().resolve("one").name == "one"
    assert get_registry().resolve("two").name == "two"


def test_make_echo_agent_hidden_flag():
    make_echo_agent("seen")
    make_echo_agent("unseen", hidden=True)
    public_names = {a.name for a in get_registry().public()}
    assert "seen" in public_names
    assert "unseen" not in public_names
```

- [ ] **Step 2: Run the smoke test**

Run: `.venv/bin/python -m pytest tests/test_orchestrator_fakes_smoke.py -v`
Expected: PASS (5 tests). If any fail, the shared `tests/_fakes.py` (Plan 2) does not match contract §9.1 — fix it in Plan 2's module (it is the single owner), do NOT shadow it here.

- [ ] **Step 3: Commit**

```bash
git add tests/test_orchestrator_fakes_smoke.py
git commit -m "test(orchestrator): smoke-test shared _fakes surface relied on by Plan 4"
```

---

### Task 4: `Orchestrator` skeleton — abstract subtype, tier detection, lazy build (Tier 1 supervisor)

**Files:**
- Create: `aixon/agents/orchestrator.py`
- Modify: `aixon/__init__.py` (export `Orchestrator`)
- Test: `tests/test_orchestrator_tier1.py`

> `aixon/agents/__init__.py` already exists (Plan 2/3 created it for `llm_agent.py`/`tool_agent.py`). If for any reason it is absent, create an empty one so `aixon.agents.orchestrator` is importable.

**Interfaces (contract §3.2 — verbatim):**

```python
class Orchestrator(Agent, abstract=True):
    _suffix = "Orchestrator"

    # Tier 1 (supervisor):
    supervisor: LLM | None = None
    agents: list = []            # any Agent subclasses/instances (workers)

    # Tier 2 (explicit graph):
    nodes: dict = {}             # name -> Agent
    entry: str = ""
    edges: list = []             # list of (src, dst) fixed edges; dst may be END
    # conditional edges: methods named route_<node>(self, state) -> str | list[str]

    # Tier 3 (escape hatch): override build_graph(self) -> compiled graph

    # Runtime guards:
    recursion_limit: int | None = 25
    timeout: int | None = None

    def invoke(self, messages: list[Message]) -> Message: ...
    def stream(self, messages: list[Message]) -> Iterator[Chunk]: ...
```

**Tier detection order** (computed in `_validate_subclass()`, stored as `cls._tier`): `build_graph` overridden on the subclass → Tier 3; else `nodes` non-empty → Tier 2; else `supervisor` set → Tier 1. None apply on a concrete subclass → `AixonError`.

This task implements: the abstract subtype, tier detection, Tier 1 build + run, and the "no tier applies" error. Tier 2 validation is Task 5; Tier 3 is Task 6; composition-cycle guard is Task 7; runtime-guard wiring assertions are Task 8; reasoning propagation is Task 9.

**Tier 1 approach (chosen, hand-rolled minimal supervisor — keeps tests hermetic, no real supervisor LLM):** Build a `StateGraph(self.State)` with one node per worker (`agents`) plus a `"supervisor"` node. To stay hermetic and avoid binding a real `supervisor` LLM in tests, the supervisor's routing uses a small, **overridable** hook `self._route_supervisor(state) -> str` defaulting to: route to the first worker that has not yet produced an assistant message this run, else `END`. A real LLM-driven supervisor is a drop-in replacement of this hook in a later refinement; the declarative surface `supervisor=LLM(...)` / `agents=[...]` is fixed by the contract and is what tests assert on. Each worker node runs `agent.invoke(state["messages"])` and returns `{"messages": result}`; after a worker runs, control returns to `"supervisor"` (the loop is bounded by guard B).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orchestrator_tier1.py
import pytest

from tests._fakes import make_llm, make_echo_agent
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_orchestrator_tier1.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aixon.agents.orchestrator'`.

- [ ] **Step 3: Write the implementation**

```python
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

    # ----- guard A placeholder (real impl in Task 7) ----------------------

    @classmethod
    def _check_composition_cycle(cls) -> None:
        return None

    # ----- Tier-2 validation placeholder (real impl in Task 5) ------------

    @classmethod
    def _validate_tier2(cls) -> None:
        return None

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
```

In `aixon/__init__.py`, add the import and export. Per contract §9.4, layers whose deps live in an extra may be guarded behind `try/except ImportError` so a bare `import aixon` works without the `llm` extra — match the pattern Plan 2/3 used for `LLMAgent`/`ToolAgent`:

```python
# aixon/__init__.py  — add the Orchestrator export (mirror the LLMAgent/ToolAgent guard if Plans 2/3 used one)
try:
    from aixon.agents.orchestrator import Orchestrator
    __all__.append("Orchestrator")
except ImportError:  # langgraph not installed (no 'llm' extra)
    pass
```

> If Plans 2/3 export `LLMAgent`/`ToolAgent` unguarded (because `import aixon` already requires the `llm` extra in this project), follow that same convention instead — add `from aixon.agents.orchestrator import Orchestrator` to the import block and `"Orchestrator"` to `__all__`. Match whatever pattern is already in the file; do not introduce an inconsistent style.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_orchestrator_tier1.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: PASS (all prior plans' tests + packaging + state + fakes smoke + Tier 1).

- [ ] **Step 6: Commit**

```bash
git add aixon/agents/orchestrator.py aixon/__init__.py tests/test_orchestrator_tier1.py
git commit -m "feat(orchestrator): Orchestrator subtype, tier detection, Tier 1 supervisor"
```

---

### Task 5: Tier 2 — explicit graph (nodes/entry/edges + route_<node> conditional & list fan-out) + structural validation

**Files:**
- Modify: `aixon/agents/orchestrator.py` (implement `_validate_tier2` + `_build_explicit_graph` + `_wrap_router` + `_node_instances`)
- Test: `tests/test_orchestrator_tier2.py`

**Interfaces (contract §3.2 Tier-2 rules — binding):**
- Each node has exactly ONE exit form: a fixed edge in `edges` (`(src, dst)`, `dst` may be `END`) OR a `route_<node>` method. **Both for the same node → `AixonError`. Neither → terminal node (allowed).**
- `route_<node>(self, state) -> str` = conditional (one next node). `-> list[str]` = parallel fan-out (LangGraph runs them and joins).
- `entry` must name a node in `nodes` → else `AixonError`.
- `edges` textual order is irrelevant; `entry` + topology drives execution.

**Implementation notes:**
- A `route_<node>` method is detected by `callable(getattr(cls, f"route_{node}", None))`.
- Fixed edges for a node = any `(src, dst)` in `edges` with `src == node`.
- Validation per node: `has_edge = node in {src for src,_ in edges}`; `has_route = callable(getattr(cls, f"route_{node}", None))`. If both → error (ambiguous exit). If neither → terminal (allowed). Also validate every edge endpoint references a real node or `END`, and `entry in nodes`.
- Build: `add_node(name, worker_node)` for each; `set_entry_point(entry)`; for fixed edges `add_edge(src, dst)`; for `route_<node>` use `add_conditional_edges(node, self._wrap_router(node))`. LangGraph 1.x natively supports a router returning a single node name (conditional) or a list of node names (fan-out), so no `path_map` is required — pass the bound router directly. `END` returned by a router is honored.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_orchestrator_tier2.py -v`
Expected: FAIL (validation/build not implemented; the both-exit case won't raise, fan-out won't run).

- [ ] **Step 3: Write the implementation**

Replace the `_validate_tier2` placeholder and add `_node_instances`, `_wrap_router`, `_build_explicit_graph` in `aixon/agents/orchestrator.py`:

```python
# aixon/agents/orchestrator.py  — replace the _validate_tier2 placeholder

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
```

```python
# aixon/agents/orchestrator.py  — add these methods to the Orchestrator class

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
```

> **LangGraph 1.x router return contract:** `add_conditional_edges(source, path_func)` invokes `path_func(state)` and routes to the returned node name(s). A returned `str` routes to that one node; a returned `list[str]` fans out to all listed nodes in parallel and the graph joins them at their next common point. We pass the bound router directly (no `path_map`), so both branching kinds work from the single `route_<node>` convention. `END` returned by a router is honored.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_orchestrator_tier2.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add aixon/agents/orchestrator.py tests/test_orchestrator_tier2.py
git commit -m "feat(orchestrator): Tier 2 explicit graph, route_<node> conditional + list fan-out, exit-form validation"
```

---

### Task 6: Tier 3 — `build_graph` escape hatch

**Files:**
- Test: `tests/test_orchestrator_tier3.py`

> No production edit is needed: Task 4 already routes `build_graph` overrides through `_compiled()` and detects Tier 3 (`"build_graph" in cls.__dict__`). This task LOCKS that behavior with tests. If a test fails, the fix belongs in `orchestrator.py`.

**Interfaces:**
- A subclass that overrides `build_graph(self) -> CompiledGraph` is Tier 3. `_compiled()` calls the user's `build_graph` (the override shadows the framework dispatch). `invoke`/`stream` then run that compiled graph through the same neutral boundary and run config.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it passes (or fails)**

Run: `.venv/bin/python -m pytest tests/test_orchestrator_tier3.py -v`
Expected: PASS if Task 4 is correct. If it FAILS, the bug is that `_compiled()` calls the framework `build_graph` instead of the override — fix by ensuring `_compiled()` calls `self.build_graph()` (instance dispatch picks the override automatically) and that `_detect_tier` checks `"build_graph" in cls.__dict__`. (Both are already specified in Task 4.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_orchestrator_tier3.py
git commit -m "test(orchestrator): lock Tier 3 build_graph escape hatch"
```

---

### Task 7: Composition-cycle guard (A) — structural, always on

**Files:**
- Modify: `aixon/agents/orchestrator.py` (implement `_referenced_agent_classes` + `_check_composition_cycle`)
- Test: `tests/test_orchestrator_cycle.py`

**Interfaces (contract §3.2 guard A + spec):**
- At subclass-definition time, walk the **static composition graph** of nested agents. The neighbors of an orchestrator class are the agent classes it references via `agents`, `nodes`, and any `tools` declared on referenced agents. Revisiting a class already on the current DFS path → `CompositionCycleError`.
- A loop *inside one LangGraph graph* (a node edging back to an earlier node) is NOT a composition cycle and must not be flagged — guard A only follows **agent→agent composition references**, never LangGraph edges.

**Implementation notes:**
- Collect referenced agent CLASSES: from `cls.agents`, `cls.nodes.values()`, and `cls.tools` (each entry → its class if it is an instance; non-Agent entries ignored).
- DFS from `cls` over `_referenced_agent_classes(node_cls)`; maintain a `path` list. Revisiting a class on `path` → raise with the cycle chain in the message.
- The walk uses CLASSES so self-inclusion (`agents=[Self]`) and mutual inclusion (`A.agents=[B]`, `B.agents=[A]`) are caught. `AgentTool` instances (from `Agent.as_tool()`) are intentionally NOT traced — they wrap an already-instantiated agent and carry only a callable, so they cannot reintroduce a class-level cycle the declarative attributes don't already express.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_orchestrator_cycle.py -v`
Expected: FAIL (guard is a no-op placeholder; cycle tests do not raise).

- [ ] **Step 3: Write the implementation**

Replace the `_check_composition_cycle` placeholder in `aixon/agents/orchestrator.py`:

```python
# aixon/agents/orchestrator.py  — replace the _check_composition_cycle placeholder

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
```

> **Why the walk reads `_referenced_agent_classes` per node:** only `Orchestrator` subclasses expose that classmethod, plus any agent declaring a `tools` attribute (e.g. a `ToolAgent` whose tool is another agent class). A plain `LLMAgent`/`ToolAgent` leaf with no agent-class tools returns nothing structural to follow, so the walk terminates. `AgentTool` instances are not traced (see notes above).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_orchestrator_cycle.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add aixon/agents/orchestrator.py tests/test_orchestrator_cycle.py
git commit -m "feat(orchestrator): always-on composition-cycle guard (CompositionCycleError)"
```

---

### Task 8: Runtime guard (B) — wire `recursion_limit` + `timeout` into the compiled graph config

**Files:**
- Modify: `aixon/agents/orchestrator.py` (harden `invoke` to translate LangGraph's recursion error)
- Test: `tests/test_orchestrator_guards.py`

**Interfaces (contract §3.2 guard B + spec):**
- `recursion_limit` (default 25) is passed to the compiled graph as `graph.invoke(state, config={"recursion_limit": N})`. `None` → key omitted (LangGraph then uses its own default unless capped by timeout). `_run_config` (Task 4) already builds exactly this dict.
- `timeout` (default `None`) is a wall-clock backstop enforced around the run (the deadline check in `invoke`, Task 4).
- A graph that would exceed `recursion_limit` raises LangGraph 1.x's `GraphRecursionError` (`from langgraph.errors import GraphRecursionError`); the orchestrator re-raises it as `AixonError` with a clear message. We test that a low `recursion_limit` on a non-terminating loop raises `AixonError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orchestrator_guards.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_orchestrator_guards.py -v`
Expected: FAIL on `test_nonterminating_loop_hits_recursion_limit` (LangGraph raises `GraphRecursionError`, not `AixonError`, until we wrap it).

- [ ] **Step 3: Harden `invoke` to wrap the recursion error**

Replace the `invoke` method in `aixon/agents/orchestrator.py`:

```python
# aixon/agents/orchestrator.py  — replace invoke()

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
```

(The `_run_config` and `timeout` deadline from Task 4 are unchanged and already correct; this edit only adds the `GraphRecursionError` translation.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_orchestrator_guards.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add aixon/agents/orchestrator.py tests/test_orchestrator_guards.py
git commit -m "feat(orchestrator): wire recursion_limit + timeout runtime guards (guard B)"
```

---

### Task 9: Reasoning propagation through orchestrator nodes

**Files:**
- Modify: `aixon/agents/orchestrator.py` (open a reasoning channel in `stream`)
- Test: `tests/test_orchestrator_reasoning.py`

**Interfaces (contract §3.3 + §2.1):**
- Orchestrator nodes that are Agents run inside the active `ReasoningChannel`; their `emit_reasoning(...)` bubbles to the orchestrator's stream as `Chunk(reasoning=...)`.
- `Orchestrator.stream` activates `reasoning_channel()` (Plan 3) for the run, executes the graph, and emits collected reasoning lines as reasoning chunks before the final content chunk.

**Implementation notes:**
- Plan 3 (merged) provides `from aixon import emit_reasoning, reasoning_channel` (and `current_channel`). The `ReasoningChannel` collects reasoning text emitted during a run. `stream` opens `with reasoning_channel() as channel:` around the graph run; nodes' `emit_reasoning` populate `channel`. After the run, drain the channel's collected lines into `Chunk(reasoning=line)`.
- The `ReasoningChannel` uses `contextvars` (Plan 3 §2.1), so the channel activated by `stream()` is the one node `emit_reasoning` calls target — even across LangGraph's executor — without threading it through state.
- **Accessor:** read the collected lines via Plan 3's canonical `ReasoningChannel` accessor. Plan 3's `ReasoningChannel` exposes a `drain()` method that returns and clears the collected lines. Use `channel.drain()` directly. (If Plan 3 named it differently, use that exact name — the test asserts behavior, not the accessor name.)
- This test needs a node that emits reasoning. Since the shared `make_echo_agent` (contract §9.1) has no reasoning hook, the test defines a tiny inline `Agent` that calls `emit_reasoning(...)` inside `invoke` — fully hermetic, no LLM.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orchestrator_reasoning.py
from typing import Iterator

from tests._fakes import make_llm
from aixon import emit_reasoning
from aixon.agent import Agent
from aixon.agents.orchestrator import Orchestrator
from aixon.message import Chunk, Message
from aixon.registry import get_registry


def _make_thinker(name: str, thought: str) -> type:
    def invoke(self, messages: list[Message]) -> Message:
        emit_reasoning(thought)
        last = messages[-1].content if messages else ""
        return Message(role="assistant", content=f"{name}:{last}")

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        emit_reasoning(thought)
        yield Chunk(content="x")
        yield Chunk(done=True)

    return type(
        f"{name.capitalize()}Agent",
        (Agent,),
        {"name": name, "invoke": invoke, "stream": stream},
    )


def test_node_reasoning_bubbles_to_orchestrator_stream():
    _make_thinker("thinker", "pondering the request")

    class ReasoningOrchestrator(Orchestrator):
        supervisor = make_llm()
        agents = [get_registry().resolve("thinker")]

    orch = get_registry().resolve("reasoningorchestrator")
    chunks = list(orch.stream([Message(role="user", content="hi")]))
    reasoning_text = "".join(c.reasoning for c in chunks)
    assert "pondering the request" in reasoning_text
    assert any("thinker:hi" in c.content for c in chunks)
    assert chunks[-1].done is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_orchestrator_reasoning.py -v`
Expected: FAIL (current `stream` only forwards `final.reasoning`, which the thinker node does not set on the Message — the reasoning was emitted to the channel, not attached to the Message).

- [ ] **Step 3: Write the implementation**

Replace the `stream` method in `aixon/agents/orchestrator.py`:

```python
# aixon/agents/orchestrator.py  — replace stream()

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        from aixon.reasoning import reasoning_channel

        with reasoning_channel() as channel:
            final = self.invoke(messages)
            for line in channel.drain():
                yield Chunk(reasoning=line)
        if final.reasoning:
            yield Chunk(reasoning=final.reasoning)
        yield Chunk(content=final.content)
        yield Chunk(done=True)
```

> **Boundary note:** `invoke()` does NOT open a channel (single-shot, returns one Message); per the contract, reasoning on the non-streaming path rides on `Message.reasoning` if a node sets it. The streaming path opens the `contextvars`-backed channel so node `emit_reasoning` calls bubble up even across LangGraph's executor. If Plan 3's `ReasoningChannel` exposes a different accessor than `drain()`, call that one — the test asserts the reasoning text appears, not the accessor name.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_orchestrator_reasoning.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: PASS (packaging, state, fakes smoke, all three tiers, validation errors, cycle guard, runtime guards, reasoning — plus every prior plan's suite).

- [ ] **Step 6: Commit**

```bash
git add aixon/agents/orchestrator.py tests/test_orchestrator_reasoning.py
git commit -m "feat(orchestrator): propagate node reasoning to orchestrator stream"
```

---

## Self-Review

**Spec coverage (Orchestrator section of the design spec + contract §3):**
- **3-tier declarative API** → Tier 1 (Task 4), Tier 2 (Task 5), Tier 3 (Task 6). ✓
- **Tier detection order** (`build_graph` override → Tier 3; `nodes` → Tier 2; `supervisor` → Tier 1; none → `AixonError`) → `_detect_tier` (Task 4), tested incl. the no-tier error. ✓
- **`entry` + topology vs. textual `edges` order** → Task 5 builds from `set_entry_point(entry)` + edges/routers; `edges` order is never used as sequence (`test_tier2_detected_and_entry_runs_first` proves entry drives execution). ✓
- **One exit form per node** (fixed edge XOR `route_<node>`; both → error; neither → terminal) → `_validate_tier2` (Task 5), tested for the both-exit error, the entry-not-in-nodes error, and the terminal-node-allowed case. ✓
- **Two branching kinds via `route_<node>`** — conditional (`-> str`) and parallel fan-out (`-> list[str]`) → Task 5, both tested. ✓
- **Recursion guard A (composition cycle, structural, always on, not disableable)** → `_check_composition_cycle` (Task 7), tested for self-inclusion, mutual inclusion, acyclic-allowed, and that a legitimate LangGraph-internal loop is NOT flagged. ✓
- **Recursion guard B (runtime depth/loop)** → `recursion_limit` (default 25, `None` omits the cap) + `timeout` wall-clock backstop wired into `graph.invoke(state, config={"recursion_limit": N})` (Tasks 4 + 8), tested incl. a non-terminating loop raising via `recursion_limit`. ✓
- **`GraphState` default state + user subclassing via nested `class State(GraphState)`** → Task 2 (`GraphState`, `add_messages_neutral`) + the `State` property in Task 4. ✓
- **Orchestrator `as_tool()` / subgraph-as-tool, state isolation, reasoning propagation** → `as_tool` inherited unchanged from Plan 1; state isolation automatic (fresh `_initial_state` per `invoke`); reasoning via the `contextvars` channel → Task 9. ✓
- **`END` re-export** → Task 2 (`from langgraph.graph import END`, re-exported as `aixon.state.END` and `aixon.END`). ✓
- **Exports `Orchestrator`, `GraphState`, `END`** → Tasks 2 + 4. ✓

**LangGraph 1.x targeting (the rewrite's purpose):**
- Imports are `from langgraph.graph import StateGraph, END` and `from langgraph.errors import GraphRecursionError` — both validated at langgraph 1.2.6. No `langchain` 0.x APIs, no `AgentExecutor`, no deprecated `langgraph.prebuilt.create_react_agent`. ✓
- Guards passed via `graph.invoke(state, config={"recursion_limit": N})`. ✓
- Dependency: `langgraph>=1.0` lives in the **`llm` extra** (contract §9.2) — NO `orchestration` extra; Task 1 asserts this and asserts the absence of `orchestration`. No `<1` ceiling anywhere. ✓

**No fallback shims (the other rewrite purpose — contract §9.3/§9.1):**
- `tests/_fakes.py` is imported, never redefined; Plan 4 uses `make_llm` / `make_echo_agent` from it with the contract §9.1 signatures (`make_echo_agent(name, *, hidden)`, `make_llm(**params)`). ✓
- `LLM`, `emit_reasoning`, `reasoning_channel` imported directly (no `try/except`). The orchestrator drains reasoning via `channel.drain()` directly (no compatibility shim). ✓
- The previous version's `_drain_reasoning` shim and `_fakes.py` `try/except LLM/emit_reasoning` stand-ins are GONE. ✓

**Dedicated venv (contract §9.5):** every run/install step uses `.venv/bin/python -m pytest` / `.venv/bin/python -m pip` (never bare `pytest`, never another project's interpreter). The standalone install command `.venv/bin/python -m pip install -e ".[dev,llm]"` is in Global Constraints and Task 1. ✓

**Placeholder scan:** Every code step is complete and runnable; no `TODO`/`...`/"add later" in production code. The Task-4 `_check_composition_cycle`/`_validate_tier2` stubs are replaced with full implementations in Tasks 7 and 5 respectively (the stub→impl progression is the TDD spine, not leftover placeholders). ✓

**Type consistency vs. contract (§3):** `GraphState` fields (`messages: Annotated[list[Message], add_messages_neutral]`, `reasoning: list[str]`, `total=False`) match §3.1 verbatim. `Orchestrator` class attributes (`supervisor`, `agents`, `nodes`, `entry`, `edges`, `recursion_limit: int|None = 25`, `timeout: int|None = None`) and `_suffix = "Orchestrator"` match §3.2 verbatim; `invoke`/`stream` match the `Agent` neutral interface from §0. `route_<node>(self, state) -> str | list[str]` matches §3.2. `CompositionCycleError` / `AixonError` are the Plan 1 exception types (§0). `END` is langgraph's, re-exported per §3.2/§3.5. Nodes are fake-LLM agents from `tests/_fakes.py` per §1.5/§3.4/§9.1. No `aixon.server`/`aixon.providers` import in `state.py`/`orchestrator.py` — neutral boundary intact. ✓
