# aixon ToolAgent + Reasoning Channel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build aixon's tool-calling agent subtype (`ToolAgent`) and the
reasoning channel it streams through. A `ToolAgent` runs a **langgraph-native**
tool-calling loop built with `langchain.agents.create_agent` over its
declarative `llm`/`prompt`/`tools`, returns a neutral `Message` from `invoke`,
and yields neutral `Chunk`s from `stream` — with tool-step labels surfaced as
`Chunk(reasoning=...)`. The reasoning channel is a `contextvars`-based current
channel so a nested agent's `emit_reasoning(...)` bubbles up to the outermost
`stream()` across sync, async, and LangGraph execution.

**Architecture:** This plan implements contract §2 and CONSUMES Plan 1
(`Agent`, `AgentTool`, `Message`, `Chunk`, `Registry`, `Logger`, `AixonError`)
and Plan 2 (`LLM` with its lazy `.chat_model`; the `fake` provider +
`LLM("fake-1", provider="fake")` hermetic handle and the validated
`FakeChatModel`/`make_llm` established in `tests/_fakes.py`; and
`aixon/_langchain.py::{to_langchain, from_langchain}`). Three new pieces:
`aixon/reasoning.py` (the channel), `aixon/_tools.py::coerce_tools` (neutral
tool entries → LangChain `BaseTool`), and `aixon/agents/tool_agent.py`
(`ToolAgent`). One confirmation guard keeps `Agent.as_tool` neutral (contract
§2.4). The neutral boundary is absolute: `ToolAgent.invoke`/`stream` speak only
`Message`/`Chunk`; LangChain/LangGraph objects exist only INSIDE the methods.

**Tech Stack:** Python 3.11+, **LangChain 1.x** (`langchain>=1.0`,
`langchain-core>=1.0`, `langgraph>=1.0`) via `langchain.agents.create_agent`
(validated against langchain 1.3 / langchain-core 1.4 / langgraph 1.2),
`contextvars` (stdlib), `pytest`, `hatchling`. The removed 0.x
`create_tool_calling_agent` + `AgentExecutor` are NOT used; the deprecated
`langgraph.prebuilt.create_react_agent` is NOT used.

## Global Constraints

Copied verbatim / binding from the contract and Plan 1:

- **Python 3.11+**, build backend `hatchling`, package name `aixon`. Editable
  install already present from Plan 1; this plan adds NO packaging change beyond
  what Plan 2 already declared in the `llm` extra (see Task 0).
- **LangChain 1.x is the agent API (contract §1.7, §2.2, §9.5):** the old 0.x
  `create_tool_calling_agent` + `AgentExecutor` are **removed**; the ToolAgent
  uses `from langchain.agents import create_agent` (validated against
  langchain 1.3 / langchain-core 1.4 / langgraph 1.2).
  `langgraph.prebuilt.create_react_agent` is **DEPRECATED in langgraph 1.0** —
  do NOT use it. **Do NOT pin a `<1` ceiling anywhere** (contract §9.5).
  The validated construction is exactly:

  ```python
  from langchain.agents import create_agent          # NOT langgraph.prebuilt.create_react_agent (deprecated)
  agent = create_agent(self.llm.chat_model, coerce_tools(self.tools),
                       system_prompt=self.prompt or None)
  result = agent.invoke({"messages": to_langchain(messages)})
  final = from_langchain(result["messages"][-1])     # result["messages"] is [Human, AI(tool_calls), Tool, AI(final)]
  ```

- **Dedicated virtualenv (contract §9.5 — REQUIRED, do NOT reuse another
  project's venv):** all install/run steps use this project's own `.venv`. The
  controller creates it ONCE before Plan 2:

  ```bash
  cd /Users/jorge/Documents/Git/aixon
  python3 -m venv .venv                      # .venv is git-ignored
  .venv/bin/python -m pip install -e ".[dev,llm,openai,anthropic,google]"
  # (server/cli extras add nothing this plan needs; `retrieval` is Plan 6's
  #  extra and does not exist yet — do NOT request it here.)
  ```

  If running Plan 3 standalone (Plan 2 already merged but the venv lacks the llm
  deps), install with `.venv/bin/python -m pip install -e ".[dev,llm]"`. Every
  run step uses `.venv/bin/python -m pytest ...` — NEVER a bare `pytest` (the
  console script can carry a stale shebang) and NEVER another project's
  interpreter. langgraph/langchain are validated at 1.x; do not pin a `<1`
  ceiling.
- **Neutral boundary (contract §0 "Established conventions"):**
  `Agent.invoke`/`stream` and the public API speak ONLY `Message`/`Chunk`.
  LangChain/LangGraph objects may be used INTERNALLY but MUST be converted at
  the boundary. The neutral↔LangChain message conversion helpers are
  `aixon/_langchain.py::{to_langchain, from_langchain}` (built by Plan 2).
  `ToolAgent` imports langchain lazily inside its methods, never at module top
  level — importing `aixon` must not require langchain.
- **`as_tool` stays neutral (contract §2.4):** `Agent.as_tool()` returns the
  neutral `AgentTool` dataclass. It MUST NOT return a LangChain tool.
  Tool→LangChain conversion happens ONLY in `coerce_tools` (§2.3). This plan
  adds a regression test asserting this; it makes NO change to `as_tool`.
- **Reasoning channel uses `contextvars.ContextVar`, NOT thread-local**
  (contract §2.1). olympus used a `threading.local()` thought queue; this plan
  deliberately converts that mechanism to a `ContextVar` so it composes with
  sync, async, and LangGraph execution. Do not reintroduce thread-local state.
- **Test fakes — single owner `tests/_fakes.py` (contract §9.1):** Plan 2 owns
  `tests/_fakes.py`. Plan 3 **imports** `FakeChatModel`, `make_llm`,
  `register_fake_provider`, `FAKE_MODEL`, `FAKE_PROVIDER` from it and does
  **NOT** redefine them. The `FakeChatModel` is the contract's validated class:
  a `BaseChatModel` whose `script` is a list of `AIMessage`s returned one per
  LLM call; setting `tool_calls` on a scripted `AIMessage` drives a real tool
  step through `create_agent` offline. (See §9.1 of the contract for the exact
  class.)
- **Hermetic tests (contract §0, §1.6, §2.5):** NO test may require a real
  provider SDK, real API key, or network. Reuse Plan 2's
  `LLM("fake-1", provider="fake")` and `FakeChatModel`. A `FakeChatModel` whose
  `script` is `[AIMessage(content="", tool_calls=[{...}]), AIMessage(content="final")]`
  drives the REAL `create_agent` graph through a tool call then a final answer
  fully offline (mirrors the validated probe). A NESTED agent's
  `emit_reasoning` bubbling to the parent `stream()` via the active
  `ReasoningChannel` is tested explicitly (Task 5).
- **Dependencies:** core `dependencies = []`. langchain/langgraph live in the
  `llm` extra (Plan 2, contract §9.2: `langchain>=1.0`, `langchain-core>=1.0`,
  `langgraph>=1.0`). This plan adds NO new dependency and NO version ceiling.
- **Logging (Plan 1):** lifecycle/diagnostic events use `Logger("aixon.<area>")`.
  Streaming the agent's own content/reasoning to a human is the CLI's job
  (Plan 7), NOT logging.
- **Error tone:** state what was got and how to fix it.
- **Commits:** Co-Authored-By trailer per repo convention.

> **Resolved ambiguity — `max_iterations` / `max_execution_time` mapping.**
> The contract (§2.2) declares `max_iterations: int = 15` and
> `max_execution_time: int = 600` on `ToolAgent`, and says they "map to
> langgraph's recursion/time config where supported; if `create_agent` exposes
> no direct knob, document the mapping and pass what it accepts (do not invent a
> parameter)." `create_agent` in langchain 1.x compiles a LangGraph graph whose
> per-invocation runtime guard is the **`recursion_limit`** passed in the
> `config` dict to `invoke`/`stream` (NOT a constructor parameter). LangGraph
> counts one super-step per node visit; a single tool round-trip (model → tool →
> model) is ~3 super-steps. So this plan maps `max_iterations` →
> `config={"recursion_limit": 2 * max_iterations + 1}` (model+tool per
> iteration, plus the final model turn), which is the documented, conservative
> mapping — it bounds the loop without inventing a parameter `create_agent`
> doesn't accept. There is **no built-in wall-clock knob** on the compiled
> graph; `max_execution_time` is enforced by aixon as a wall-clock backstop
> (a deadline checked in the stream loop / via a `time.monotonic()` guard around
> `invoke`). Both attributes stay on the declarative surface exactly as the
> contract pins them; only their *enforcement mechanism* is documented here.

---

### Task 0: Verify the `llm` extra + dedicated venv, baseline green

**Files:**
- (No file change expected. Confirm Plan 2's `pyproject.toml` `llm` extra. Only
  edit `pyproject.toml` if the `llm` extra is missing or pins a `<1` ceiling.)

**Interfaces:**
- Consumes: the `llm` extra declared by Plan 2
  (`llm = ["langchain>=1.0", "langchain-core>=1.0", "langgraph>=1.0"]`,
  contract §9.2) and the dedicated `.venv` (contract §9.5).
- Produces: a verified environment where `langchain.agents.create_agent`
  resolves. No public API change.

> Why a task: every `ToolAgent` runtime path and every test in this plan depends
> on the langchain 1.x `create_agent` API and the dedicated venv. Confirming
> them first makes the whole plan reproducible. This task adds NO ceiling — a
> `<1` pin is forbidden by contract §9.5.

- [ ] **Step 1: Confirm the venv exists and the `llm` extra is langchain 1.x**

Run:
```bash
cd /Users/jorge/Documents/Git/aixon
ls .venv/bin/python && echo "venv OK"
.venv/bin/python -m pip show langchain langchain-core langgraph 2>/dev/null | grep -E '^(Name|Version)'
```
Open `pyproject.toml` and confirm Plan 2 declared, under
`[project.optional-dependencies]`:

```toml
llm = ["langchain>=1.0", "langchain-core>=1.0", "langgraph>=1.0"]
```

If the `llm` extra is missing entirely (Plan 2 not yet merged when running this
plan standalone), add exactly that line. If it pins a `<1` ceiling anywhere,
REMOVE the ceiling (contract §9.5 forbids it). Leave the vendor extras
(`openai`/`anthropic`/`google`), `server`, `cli`, and `all` exactly as Plan 2
left them. (There is no `retrieval` extra yet — it arrives in Plan 6; do not add
it here.)

- [ ] **Step 2: Install (or refresh) the llm extra into the dedicated venv**

Run (standalone form — the controller normally installs the full set once before
Plan 2, see Global Constraints):
```bash
cd /Users/jorge/Documents/Git/aixon
.venv/bin/python -m pip install -e ".[dev,llm]"
```
Expected: resolves `langchain` `1.x`, `langchain-core` `1.x`, `langgraph` `1.x`,
no `<1` downgrade.

- [ ] **Step 3: Verify the langgraph-native API resolves**

Run:
```bash
.venv/bin/python -c "from langchain.agents import create_agent; print('create_agent OK')"
.venv/bin/python -c "from langchain_core.tools import StructuredTool, BaseTool; print('tools OK')"
.venv/bin/python -c "from langchain_core.language_models.chat_models import BaseChatModel; print('basechatmodel OK')"
```
Expected:
```
create_agent OK
tools OK
basechatmodel OK
```
No `ImportError`. (If `create_tool_calling_agent` were imported instead it would
fail — it is removed in 1.x. Do NOT import it.)

- [ ] **Step 4: Confirm `aixon` still imports without langchain at module top**

Run: `.venv/bin/python -c "import aixon; print(sorted(aixon.__all__))"`
Expected: prints Plan 1/2 exports, no error. (Sanity that nothing forces
langchain at import time before we add the new modules.)

- [ ] **Step 5: Run the existing suite (baseline green)**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — all Plan 1 + Plan 2 tests green before we add anything.

- [ ] **Step 6: Commit (only if `pyproject.toml` changed)**

If you edited `pyproject.toml` in Step 1:
```bash
git add pyproject.toml
git commit -m "chore(#3): ensure llm extra targets langchain 1.x (create_agent), no <1 ceiling"
```
Otherwise skip — there is nothing to commit for this task.

---

### Task 1: Reasoning channel (`aixon/reasoning.py`)

**Files:**
- Create: `aixon/reasoning.py`
- Modify: `aixon/__init__.py` (export `emit_reasoning`, `reasoning_channel`)
- Test: `tests/test_reasoning.py`

**Interfaces (contract §2.1):**
- Consumes: nothing (stdlib `contextvars` + `contextlib`).
- Produces:
  - `aixon.reasoning.ReasoningChannel` — collects reasoning lines emitted
    during a run. Methods: `emit(self, text: str) -> None` (append a line),
    `drain(self) -> list[str]` (return and clear buffered lines), and a `lines`
    property returning a copy (does not clear) for tests.
  - `aixon.reasoning.current_channel() -> ReasoningChannel | None` — the
    `ContextVar`-backed active channel, or `None` if none is active.
  - `aixon.reasoning.emit_reasoning(text: str) -> None` — push `text` to the
    current channel if one is active; **no-op otherwise**. Nested agents call
    this so their reasoning bubbles up to the parent's stream.
  - `aixon.reasoning.reasoning_channel() -> Iterator[ReasoningChannel]` — a
    `@contextmanager` that activates a fresh channel for the duration of a
    `stream()`, resetting the `ContextVar` token on exit (LIFO-safe for
    nesting). Yields the channel so the streaming loop can `drain()` it.
  - Re-exported from `aixon`: `emit_reasoning`, `reasoning_channel`.

> Design note: use a single `ContextVar[ReasoningChannel | None]` named
> `_current` with `default=None`. `reasoning_channel()` sets a NEW channel and
> stores the returned token, then `reset(token)` in `finally` — the contextvars
> idiom that composes with nesting and async. NOT `threading.local()`
> (contract §2.1).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reasoning.py
import contextvars

from aixon.reasoning import (
    ReasoningChannel,
    current_channel,
    emit_reasoning,
    reasoning_channel,
)


def test_emit_reasoning_is_noop_without_active_channel():
    # No channel active: must not raise, must not store anywhere.
    assert current_channel() is None
    emit_reasoning("ignored")  # no-op
    assert current_channel() is None


def test_channel_collects_emitted_lines():
    with reasoning_channel() as ch:
        assert current_channel() is ch
        emit_reasoning("step one")
        emit_reasoning("step two")
        assert ch.lines == ["step one", "step two"]


def test_drain_returns_and_clears():
    with reasoning_channel() as ch:
        emit_reasoning("a")
        emit_reasoning("b")
        assert ch.drain() == ["a", "b"]
        assert ch.lines == []
        emit_reasoning("c")
        assert ch.drain() == ["c"]


def test_channel_is_reset_after_context_exits():
    with reasoning_channel():
        assert current_channel() is not None
    assert current_channel() is None


def test_nested_channels_restore_outer_on_exit():
    with reasoning_channel() as outer:
        emit_reasoning("outer-1")
        with reasoning_channel() as inner:
            assert current_channel() is inner
            emit_reasoning("inner-1")
            assert inner.lines == ["inner-1"]
        # Inner exited: the outer channel is active again, unpolluted.
        assert current_channel() is outer
        assert outer.lines == ["outer-1"]


def test_contextvar_isolation_across_independent_contexts():
    # A copied context sees its own channel; the parent context is unaffected.
    results = {}

    def run_in_child():
        with reasoning_channel() as ch:
            emit_reasoning("child")
            results["child"] = ch.lines

    ctx = contextvars.copy_context()
    ctx.run(run_in_child)
    # Parent context never had a channel.
    assert current_channel() is None
    assert results["child"] == ["child"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_reasoning.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aixon.reasoning'`.

- [ ] **Step 3: Write the implementation**

```python
# aixon/reasoning.py
"""Reasoning channel. Collects reasoning text emitted during an agent run and
makes it available to the streaming layer, propagating across nested agents.

Backed by a ``contextvars.ContextVar`` (NOT thread-local) so it composes with
sync code, async code, and LangGraph execution. The olympus framework used a
``threading.local()`` thought queue; aixon deliberately uses a ContextVar so a
copied/forked execution context carries its own channel correctly.

Usage:
    with reasoning_channel() as channel:   # activates a channel for this run
        ...                                 # nested agents call emit_reasoning()
        for line in channel.drain():        # streaming loop pulls lines out
            yield Chunk(reasoning=line + "\\n")
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator, Optional


class ReasoningChannel:
    """Buffers reasoning lines for one agent run."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def emit(self, text: str) -> None:
        """Append one reasoning line."""
        self._lines.append(text)

    def drain(self) -> list[str]:
        """Return all buffered lines and clear the buffer."""
        lines = self._lines
        self._lines = []
        return lines

    @property
    def lines(self) -> list[str]:
        """A copy of the currently buffered lines (does not clear)."""
        return list(self._lines)


# The active channel for the current execution context. None when no run is
# streaming through a channel.
_current: contextvars.ContextVar[Optional[ReasoningChannel]] = contextvars.ContextVar(
    "aixon_reasoning_channel", default=None
)


def current_channel() -> Optional[ReasoningChannel]:
    """Return the channel active in this execution context, or None."""
    return _current.get()


def emit_reasoning(text: str) -> None:
    """Push a reasoning line to the current channel if one is active.

    No-op when no channel is active, so a nested agent invoked outside any
    stream() (e.g. a bare ``agent.invoke``) never raises. Nested agents call
    this and their reasoning bubbles to the parent's stream because the
    parent's stream() set the active channel.
    """
    channel = _current.get()
    if channel is not None:
        channel.emit(text)


@contextmanager
def reasoning_channel() -> Iterator[ReasoningChannel]:
    """Activate a fresh ReasoningChannel for the duration of a stream().

    Drained by the streaming loop into ``Chunk(reasoning=...)``. The ContextVar
    token is reset on exit so nested ``reasoning_channel()`` blocks restore the
    outer channel (LIFO).
    """
    channel = ReasoningChannel()
    token = _current.set(channel)
    try:
        yield channel
    finally:
        _current.reset(token)
```

Add exports to `aixon/__init__.py`. Read the file first, then insert the import
(alphabetically near the other `aixon.*` imports) and add the two names to
`__all__` in the logical group with other runtime helpers:

```python
# aixon/__init__.py  — add this import line
from aixon.reasoning import emit_reasoning, reasoning_channel
```

```python
# aixon/__init__.py  — add to __all__ (keep the file's existing grouping/sorting)
    "emit_reasoning",
    "reasoning_channel",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_reasoning.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Confirm package import is unaffected**

Run: `.venv/bin/python -c "import aixon; assert 'emit_reasoning' in aixon.__all__ and 'reasoning_channel' in aixon.__all__; print('exports OK')"`
Expected: `exports OK`.

- [ ] **Step 6: Commit**

```bash
git add aixon/reasoning.py aixon/__init__.py tests/test_reasoning.py
git commit -m "feat(#3): contextvars-based ReasoningChannel + emit_reasoning/reasoning_channel"
```

---

### Task 2: Guardrail — Plan 2's `FakeChatModel` drives `create_agent` offline

**Files:**
- Test: `tests/test_fake_drives_create_agent.py` (proves the contract's
  `FakeChatModel` from `tests/_fakes.py` drives a REAL `create_agent` graph
  through a tool call then a final answer — the linchpin for Tasks 4–5)
- (No production change. NO new fake is defined — `FakeChatModel` is owned by
  Plan 2's `tests/_fakes.py`, contract §9.1. This task only proves the import +
  pattern work, mirroring the validated probe.)

**Interfaces:**
- Consumes: `tests._fakes.FakeChatModel` (contract §9.1, validated against
  langchain 1.3 / core 1.4 / langgraph 1.2); `langchain.agents.create_agent`;
  `langchain_core.messages.AIMessage`; `langchain_core.tools.tool`.
- Produces (test-only): a guardrail asserting the hermetic fake drives the
  langgraph-native `create_agent` graph offline, so Tasks 4–5 can rely on it.

> This mirrors the VALIDATED reference (probe2.py): a `FakeChatModel` whose
> `script` is `[AIMessage(content="", tool_calls=[{...}]), AIMessage("final")]`
> drives `create_agent(fake, [tool], system_prompt=...)` through one tool call
> then a final answer, with `result["messages"]` ending `[Human, AI(tool_calls),
> Tool, AI(final)]` — all offline, NO API key, NO network. If this test fails,
> stop and fix the environment / `tests/_fakes.py` before Task 4: every later
> test depends on this exact pattern.

- [ ] **Step 1: Confirm Plan 2's `tests/_fakes.py` provides `FakeChatModel`**

Read `tests/_fakes.py` and confirm it defines `FakeChatModel`, `make_llm`,
`register_fake_provider`, `FAKE_MODEL`, `FAKE_PROVIDER` (contract §9.1). Do NOT
modify it. If `FakeChatModel` is missing, Plan 2 is incomplete — STOP and
resolve Plan 2 first (Plan 3 must not redefine it; §9.1 single-owner rule).

- [ ] **Step 2: Write the guardrail test**

```python
# tests/test_fake_drives_create_agent.py
"""Guardrail: Plan 2's FakeChatModel (tests/_fakes.py, contract §9.1) drives the
langgraph-native langchain.agents.create_agent graph offline through a tool call
then a final answer. Mirrors the validated probe. This is the linchpin pattern
the ToolAgent tests rely on."""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from tests._fakes import FakeChatModel


@tool
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"sunny in {city}"


def test_fake_chat_model_drives_create_agent_through_tool_then_answer():
    fake = FakeChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_weather", "args": {"city": "Recife"}, "id": "call_1"}
                ],
            ),
            AIMessage(content="The weather in Recife is sunny."),
        ]
    )
    agent = create_agent(fake, [get_weather], system_prompt="You are helpful.")
    result = agent.invoke({"messages": [("user", "weather in Recife?")]})

    final = result["messages"][-1]
    assert "sunny" in final.content.lower()
    # The graph really ran a tool step: Human, AI(tool_calls), Tool, AI(final).
    types = [type(m).__name__ for m in result["messages"]]
    assert "ToolMessage" in types
    assert types[-1] == "AIMessage"


def test_fake_chat_model_streams_updates_offline():
    fake = FakeChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_weather", "args": {"city": "Recife"}, "id": "c1"}
                ],
            ),
            AIMessage(content="It is sunny in Recife."),
        ]
    )
    agent = create_agent(fake, [get_weather], system_prompt="You are helpful.")

    updates = list(
        agent.stream(
            {"messages": [("user", "weather?")]}, stream_mode="updates"
        )
    )
    # Each update is {node_name: {"messages": [...]}}; collect every message.
    seen = []
    for upd in updates:
        for node_state in upd.values():
            for m in node_state.get("messages", []):
                seen.append(m)
    contents = [getattr(m, "content", "") for m in seen]
    assert any("sunny" in c.lower() for c in contents)
```

- [ ] **Step 3: Run the guardrail (it should PASS immediately — no production code yet)**

Run: `.venv/bin/python -m pytest tests/test_fake_drives_create_agent.py -v`
Expected: PASS (2 tests). This proves the hermetic offline pattern before we
build `ToolAgent` on top of it. If it fails:
- `ImportError: cannot import name 'FakeChatModel'` → Plan 2's `tests/_fakes.py`
  is missing it (see Step 1) — fix Plan 2, do not redefine here.
- a langchain `ImportError` → re-run Task 0 Steps 2–3 (env not installed).

- [ ] **Step 4: Commit**

```bash
git add tests/test_fake_drives_create_agent.py
git commit -m "test(#3): guardrail — FakeChatModel drives create_agent offline (invoke + stream)"
```

---

### Task 3: Tool coercion (`aixon/_tools.py::coerce_tools`)

**Files:**
- Create: `aixon/_tools.py`
- Test: `tests/test_tools.py`

**Interfaces (contract §2.3):**
- Consumes: `aixon.agent.AgentTool` (Plan 1); `langchain_core.tools.BaseTool`,
  `langchain_core.tools.StructuredTool` (lazy import inside the function).
- Produces:
  - `aixon._tools.coerce_tools(tools: list) -> list[BaseTool]` — convert each
    entry to a LangChain `BaseTool`:
    - an `AgentTool` (from `Agent.as_tool()` / `Retriever.as_tool()`) →
      `StructuredTool.from_function(func=tool.func, name=tool.name, description=tool.description)`
    - a LangChain `BaseTool` (incl. `@tool`-decorated functions, which ARE
      `BaseTool` instances) → passed through unchanged
    - a plain callable → wrapped via `StructuredTool.from_function(callable)`
    - anything else → `AixonError` stating the bad type and the accepted forms.

> The function imports langchain lazily so importing `aixon` never forces
> langchain. An `AgentTool.func(text: str) -> str` becomes a `StructuredTool`
> whose only param is named after the function's parameter (`from_function`
> introspects the signature), giving a clean one-string tool input — matching
> how `Agent.as_tool` builds `_run(text: str)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools.py
import pytest
from langchain_core.tools import BaseTool, StructuredTool, tool

from aixon.agent import AgentTool
from aixon.exceptions import AixonError
from aixon._tools import coerce_tools


def test_agenttool_becomes_structuredtool():
    at = AgentTool(name="greeter", description="says hi", func=lambda text: "hi " + text)
    [coerced] = coerce_tools([at])
    assert isinstance(coerced, BaseTool)
    assert coerced.name == "greeter"
    assert coerced.description == "says hi"
    # The wrapped func runs through the LangChain tool.
    assert coerced.invoke({"text": "bob"}) == "hi bob"


def test_langchain_basetool_passes_through_unchanged():
    @tool
    def echo(text: str) -> str:
        """Echo the text."""
        return text

    result = coerce_tools([echo])
    assert result == [echo]  # same object, not re-wrapped


def test_plain_callable_is_wrapped():
    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    [coerced] = coerce_tools([multiply])
    assert isinstance(coerced, BaseTool)
    assert coerced.name == "multiply"
    assert coerced.invoke({"a": 3, "b": 4}) == 12


def test_mixed_list_preserves_order_and_types():
    at = AgentTool(name="t1", description="d1", func=lambda text: text)

    @tool
    def t2(text: str) -> str:
        """second"""
        return text

    def t3(text: str) -> str:
        """third"""
        return text

    out = coerce_tools([at, t2, t3])
    assert [t.name for t in out] == ["t1", "t2", "t3"]
    assert all(isinstance(t, BaseTool) for t in out)
    assert out[1] is t2  # passthrough preserved


def test_unsupported_entry_raises_aixon_error():
    with pytest.raises(AixonError, match="cannot be used as a tool"):
        coerce_tools([42])


def test_empty_list_returns_empty():
    assert coerce_tools([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aixon._tools'`.

- [ ] **Step 3: Write the implementation**

```python
# aixon/_tools.py
"""Coerce neutral tool entries into LangChain BaseTools for the tool-calling
loop. This is the ONLY place neutral AgentTool -> LangChain conversion happens
(the neutral boundary, contract §2.3/§2.4): Agent.as_tool stays neutral and
returns an AgentTool; coercion to a LangChain tool occurs here, inside the
ToolAgent runtime. langchain is imported lazily so importing ``aixon`` never
requires it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aixon.agent import AgentTool
from aixon.exceptions import AixonError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.tools import BaseTool


def coerce_tools(tools: list) -> list["BaseTool"]:
    """Convert each entry of ``tools`` to a LangChain ``BaseTool``.

    Accepted entry forms:
      * ``AgentTool`` (from ``Agent.as_tool()`` / ``Retriever.as_tool()``) ->
        wrapped with ``StructuredTool.from_function``.
      * a LangChain ``BaseTool`` (incl. ``@tool``-decorated functions) ->
        passed through unchanged.
      * a plain callable -> wrapped with ``StructuredTool.from_function``.

    Raises ``AixonError`` for any other type.
    """
    from langchain_core.tools import BaseTool, StructuredTool

    coerced: list[BaseTool] = []
    for entry in tools:
        if isinstance(entry, BaseTool):
            coerced.append(entry)
        elif isinstance(entry, AgentTool):
            coerced.append(
                StructuredTool.from_function(
                    func=entry.func,
                    name=entry.name,
                    description=entry.description,
                )
            )
        elif callable(entry):
            coerced.append(StructuredTool.from_function(entry))
        else:
            raise AixonError(
                f"Tool entry {entry!r} (type {type(entry).__name__}) cannot be "
                f"used as a tool. Provide an AgentTool (agent.as_tool() / "
                f"retriever.as_tool()), a LangChain BaseTool / @tool function, "
                f"or a plain callable."
            )
    return coerced
```

> Note: `aixon._tools` is a private module — it is NOT exported from `aixon`
> (only `ToolAgent`, `emit_reasoning`, `reasoning_channel` are public per §2.6).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tools.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add aixon/_tools.py tests/test_tools.py
git commit -m "feat(#3): coerce_tools — neutral AgentTool/callable/BaseTool -> BaseTool"
```

---

### Task 4: `ToolAgent.invoke` over `create_agent` (langgraph-native)

**Files:**
- Create: `aixon/agents/tool_agent.py`
- Modify: `aixon/__init__.py` (export `ToolAgent`)
- Test: `tests/test_tool_agent_invoke.py`

**Interfaces (contract §2.2):**
- Consumes: `aixon.agent.Agent` (Plan 1); `aixon.message.{Message, Chunk}`
  (Plan 1); `aixon.llm.LLM` with its `.chat_model` property (Plan 2);
  `aixon._langchain.{to_langchain, from_langchain}` (Plan 2);
  `aixon._tools.coerce_tools` (Task 3); `aixon.reasoning.{reasoning_channel,
  emit_reasoning}` (Task 1); `aixon.exceptions.AixonError`;
  `aixon.logging.Logger`; `langchain.agents.create_agent` (lazy import).
- Produces:
  - `aixon.agents.tool_agent.ToolAgent(Agent, abstract=True)` with declarative
    class attributes:
    - `llm: LLM` — **REQUIRED** (validated by overriding the
      `Agent._validate_subclass()` classmethod hook — NOT `__init_subclass__`;
      raise `AixonError` if absent, mirroring §1.5's `LLMAgent`. The base calls
      the hook before registration, so a missing `llm` fails without registering
      a ghost; see contract "Subtype validation hook").
    - `prompt: str = ""` — system prompt (passed to `create_agent` as
      `system_prompt=self.prompt or None`).
    - `tools: list = []` — entries accepted by `coerce_tools`.
    - `max_iterations: int = 15` (mapped to `recursion_limit` — see Global
      Constraints' resolved-ambiguity note).
    - `max_execution_time: int = 600` (wall-clock backstop — see same note).
  - `_suffix = "Agent"`.
  - `invoke(self, messages: list[Message]) -> Message` — build the graph with
    `create_agent(self.llm.chat_model, coerce_tools(self.tools),
    system_prompt=self.prompt or None)`, invoke with
    `{"messages": to_langchain(messages)}` under the recursion config, convert
    the final message back with `from_langchain(result["messages"][-1])`, and
    set any reasoning collected during the run on `Message.reasoning`.
  - `stream` is implemented fully in THIS task (so `ToolAgent` is instantiable —
    `Agent.__init_subclass__` rejects unimplemented abstract methods). Task 5
    adds the streaming + nested-propagation tests against this same code.

> Decision: implement `invoke` AND `stream` fully now; this task's TESTS cover
> only `invoke`. Task 5 adds the streaming + reasoning-propagation tests.

Implementation shape (langgraph-native, neutral types + contextvars):

- `lc_tools = coerce_tools(list(self.tools))`.
- A leading neutral `system` message, if present, overrides `self.prompt`.
- `agent = create_agent(self.llm.chat_model, lc_tools, system_prompt=system_prompt or None)`.
- `config = {"recursion_limit": 2 * self.max_iterations + 1}`.
- Run inside `reasoning_channel()` so a nested agent's `emit_reasoning` is
  captured even when this agent is invoked directly (no outer stream).
- The reasoning labels for the PARENT's own tool calls are emitted by inspecting
  the AI messages that carry `tool_calls` as they appear in the result/stream
  (langgraph has no per-action callback equivalent to olympus' `on_agent_action`;
  the tool-call labels are derived from the graph's message stream instead — a
  documented mapping, not an invented parameter).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tool_agent_invoke.py
import pytest

from aixon.agents.tool_agent import ToolAgent
from aixon.exceptions import AixonError
from aixon.llm import LLM
from aixon.message import Message
from aixon.registry import get_registry

from langchain_core.messages import AIMessage
from tests._fakes import FakeChatModel


def _install_fake(monkeypatch, llm, script):
    """Force llm.chat_model to return our scripted fake (no provider/network)."""
    fake = FakeChatModel(script=script)
    monkeypatch.setattr(type(llm), "chat_model", property(lambda self: fake))


def _tool_call(name, args, id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])


def test_toolagent_requires_llm():
    with pytest.raises(AixonError, match="llm"):
        # Concrete subclass missing the required `llm` attribute.
        type("NoLLMAgent", (ToolAgent,), {"tools": []})


def test_toolagent_suffix_enforced():
    from aixon.exceptions import NamingError

    with pytest.raises(NamingError, match="Agent"):
        type("BadTool", (ToolAgent,), {"llm": LLM("fake-1", provider="fake")})


def test_toolagent_invoke_runs_tool_then_returns_final_message(monkeypatch):
    calls = {"n": 0}

    def adder(a: int, b: int) -> int:
        """Add two integers."""
        calls["n"] += 1
        return a + b

    class MathAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        prompt = "You do math."
        tools = [adder]

    agent = get_registry().resolve("mathagent")
    _install_fake(
        monkeypatch,
        agent.llm,
        [
            _tool_call("adder", {"a": 2, "b": 3}),
            AIMessage(content="The answer is 5."),
        ],
    )

    result = agent.invoke([Message(role="user", content="add 2 and 3")])

    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "The answer is 5."
    assert calls["n"] == 1


def test_toolagent_invoke_sets_reasoning_on_message(monkeypatch):
    def adder(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    class ReasonAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [adder]

    agent = get_registry().resolve("reasonagent")
    _install_fake(
        monkeypatch,
        agent.llm,
        [
            _tool_call("adder", {"a": 1, "b": 1}),
            AIMessage(content="Sum is 2."),
        ],
    )

    result = agent.invoke([Message(role="user", content="add")])

    # A tool-call step label was collected as reasoning.
    assert result.reasoning is not None
    assert "adder" in result.reasoning


def test_toolagent_is_neutral_in_and_out(monkeypatch):
    def noop(text: str) -> str:
        """noop"""
        return text

    class NeutralAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [noop]

    agent = get_registry().resolve("neutralagent")
    _install_fake(monkeypatch, agent.llm, [AIMessage(content="done immediately")])

    # Pass only neutral Messages; receive a neutral Message.
    result = agent.invoke([Message(role="user", content="hi")])
    assert type(result).__name__ == "Message"
    assert result.content == "done immediately"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tool_agent_invoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aixon.agents.tool_agent'`.

- [ ] **Step 3: Write the implementation**

First ensure the `aixon/agents/` package exists with an `__init__.py` (Plan 2
created it for `llm_agent.py`; if `aixon/agents/__init__.py` is missing, create
an empty one). Then:

```python
# aixon/agents/tool_agent.py
"""ToolAgent — the tool-calling agent subtype, langgraph-native.

Builds a LangGraph agent with ``langchain.agents.create_agent`` (LangChain 1.x;
the removed 0.x ``create_tool_calling_agent`` + ``AgentExecutor`` and the
deprecated ``langgraph.prebuilt.create_react_agent`` are NOT used) over
``self.llm`` and the coerced tools, speaking ONLY neutral Message/Chunk at its
boundary.

Reasoning is surfaced through the contextvars-based ReasoningChannel (contract
§2.1): parent tool-call labels are derived from the graph's AI messages, and a
nested agent's ``emit_reasoning`` bubbles up because it targets the active
channel. langchain is imported lazily inside methods so importing ``aixon``
never requires it."""

from __future__ import annotations

import time
from typing import Iterator

from aixon.agent import Agent
from aixon.exceptions import AixonError
from aixon.logging import Logger
from aixon.message import Chunk, Message
from aixon.reasoning import emit_reasoning, reasoning_channel
from aixon._tools import coerce_tools

_log = Logger("aixon.tool_agent")


class ToolAgent(Agent, abstract=True):
    """Tool-calling agent. Declarative attributes:

        class Diagnosis(ToolAgent):
            llm = LLM("gpt-5.4", temperature=0.1)
            prompt = "..."
            tools = [LibraryRetriever.as_tool(), check_battery]

    ``max_iterations`` maps to LangGraph's per-invocation ``recursion_limit``
    (a model+tool pair plus the final model turn per iteration);
    ``max_execution_time`` is a wall-clock backstop enforced here (LangGraph's
    compiled graph has no built-in time knob)."""

    _suffix = "Agent"

    llm = None  # REQUIRED LLM instance on concrete subclasses
    prompt: str = ""
    tools: list = []
    max_iterations: int = 15
    max_execution_time: int = 600

    @classmethod
    def _validate_subclass(cls) -> None:
        # Validate the required declarative LLM on concrete subclasses. This
        # overrides Agent._validate_subclass (a hook the base calls AFTER suffix/
        # abstract-method checks and BEFORE registration), so a missing `llm`
        # raises without leaving a ghost in the registry. Do NOT override
        # __init_subclass__ to validate after super() — that registers first,
        # then fails (the register-then-validate ghost bug). The hook fires only
        # for concrete subclasses, so no abstract=True guard is needed here.
        if getattr(cls, "llm", None) is None:
            raise AixonError(
                f"ToolAgent subclass '{cls.__name__}' must declare an `llm` "
                f"attribute (e.g. `llm = LLM(\"gpt-5.4\")`). It was missing or None."
            )

    # ---- internal: build the langgraph agent + neutral message prep -------

    def _build_agent(self, messages: list[Message]):
        """Return (compiled_agent, lc_messages, config). A leading neutral
        system message overrides self.prompt."""
        from langchain.agents import create_agent
        from aixon._langchain import to_langchain

        system_prompt = self.prompt or None
        if messages and messages[0].role == "system":
            system_prompt = messages[0].content or system_prompt
            messages = messages[1:]

        lc_tools = coerce_tools(list(self.tools))
        agent = create_agent(self.llm.chat_model, lc_tools, system_prompt=system_prompt)
        lc_messages = to_langchain(messages)
        config = {"recursion_limit": 2 * self.max_iterations + 1}
        return agent, lc_messages, config

    @staticmethod
    def _emit_tool_call_labels(message) -> None:
        """If an AI message carries tool calls, emit one reasoning label per
        call into the active ReasoningChannel (the langgraph-native equivalent
        of olympus' on_agent_action callback)."""
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            if name:
                emit_reasoning(f"Calling {name}...")

    # ---- neutral boundary: invoke ---------------------------------------

    def invoke(self, messages: list[Message]) -> Message:
        """Run the tool-calling graph to completion; return a neutral Message.

        Reasoning collected during the run (tool-step labels, plus reasoning a
        nested agent emitted) is set on Message.reasoning. Runs inside a
        reasoning_channel so a nested agent's emit_reasoning is captured even
        when this agent is invoked directly (no outer stream)."""
        from aixon._langchain import from_langchain

        agent, lc_messages, config = self._build_agent(messages)
        deadline = time.monotonic() + self.max_execution_time
        with reasoning_channel() as channel:
            result = agent.invoke({"messages": lc_messages}, config=config)
            # Derive parent tool-call labels from the AI messages in the result.
            for m in result["messages"]:
                if getattr(m, "type", "") == "ai":
                    self._emit_tool_call_labels(m)
            if time.monotonic() > deadline:
                _log.warning(
                    f"agent '{self.name}' exceeded max_execution_time "
                    f"({self.max_execution_time}s)"
                )
            reasoning_lines = channel.drain()
        final = from_langchain(result["messages"][-1])
        if reasoning_lines:
            final.reasoning = "\n".join(reasoning_lines)
        _log.info(f"agent '{self.name}' completed ({len(reasoning_lines)} step(s))")
        return final

    # ---- neutral boundary: stream ---------------------------------------

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        """Stream the run: Chunk(reasoning=...) for tool-step labels (parent +
        nested) and Chunk(content=...) for the final answer; final
        Chunk(done=True)."""
        agent, lc_messages, config = self._build_agent(messages)
        deadline = time.monotonic() + self.max_execution_time
        final_content = ""
        with reasoning_channel() as channel:
            for update in agent.stream(
                {"messages": lc_messages}, config=config, stream_mode="updates"
            ):
                # Each update is {node_name: {"messages": [...]}}.
                for node_state in update.values():
                    for m in node_state.get("messages", []) or []:
                        if getattr(m, "type", "") == "ai":
                            self._emit_tool_call_labels(m)
                            if getattr(m, "content", ""):
                                final_content = m.content
                # Surface reasoning accrued since the last update (parent labels
                # + any nested-agent emit_reasoning) before yielding content.
                for line in channel.drain():
                    yield Chunk(reasoning=line + "\n")
                if time.monotonic() > deadline:
                    emit_reasoning(
                        f"(stopped: exceeded max_execution_time "
                        f"{self.max_execution_time}s)"
                    )
                    for line in channel.drain():
                        yield Chunk(reasoning=line + "\n")
                    break
            # Any trailing reasoning emitted after the last update.
            for line in channel.drain():
                yield Chunk(reasoning=line + "\n")
        if final_content:
            yield Chunk(content=final_content)
        yield Chunk(done=True)
```

> Note on `stream_mode="updates"`: each yielded update is a dict keyed by node
> name whose value carries the messages produced at that step. We inspect the AI
> messages to (a) emit tool-call labels and (b) capture the final answer
> content. Draining the channel after each update interleaves parent labels with
> any nested-agent reasoning emitted while a tool ran. The `from_langchain`
> conversion is only needed in `invoke`; in `stream` we surface
> `m.content` directly as the neutral `Chunk(content=...)`.

Add the export to `aixon/__init__.py` (read it first):

```python
# aixon/__init__.py  — add this import
from aixon.agents.tool_agent import ToolAgent
```

```python
# aixon/__init__.py  — add to __all__ in the agents group (near LLMAgent)
    "ToolAgent",
```

> Export guard (contract §9.4): if a top-level `from aixon.agents.tool_agent
> import ToolAgent` would force langchain at `import aixon` time, wrap the import
> in `try/except ImportError` exactly as Plan 2/5 do. (It should NOT — langchain
> is imported lazily inside the methods — but verify with Step 5's import check;
> if `import aixon` fails on a bare install, apply the guard.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tool_agent_invoke.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Confirm `import aixon` still works (neutral boundary)**

Run: `.venv/bin/python -c "import aixon; print('ToolAgent' in aixon.__all__)"`
Expected: `True`. (If `import aixon` raises an `ImportError` referencing
langchain, apply the §9.4 try/except guard around the `ToolAgent` import in
`aixon/__init__.py` and re-run.)

- [ ] **Step 6: Run the suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — Plan 1, Plan 2, and Tasks 1–4 all green.

- [ ] **Step 7: Commit**

```bash
git add aixon/agents/tool_agent.py aixon/__init__.py tests/test_tool_agent_invoke.py
git commit -m "feat(#3): ToolAgent.invoke over langchain.agents.create_agent (langgraph-native)"
```

---

### Task 5: `ToolAgent.stream` + nested-agent reasoning propagation

**Files:**
- Test: `tests/test_tool_agent_stream.py`
- Test: `tests/test_nested_reasoning.py`
- (No production change expected — `stream` was implemented in Task 4. If a
  test reveals a gap, fix it in `aixon/agents/tool_agent.py` per
  superpowers:test-driven-development and note it.)

**Interfaces:**
- Consumes: everything from Task 4 plus `aixon.reasoning.emit_reasoning`
  (to drive a nested worker's reasoning) and the `FakeChatModel` from
  `tests/_fakes.py`.
- Produces: verification that `stream` yields `Chunk(reasoning=...)` for tool
  steps and `Chunk(content=...)` for the final answer, ending with
  `Chunk(done=True)`; and that a NESTED agent's `emit_reasoning` bubbles to the
  parent `stream()`.

> Nested propagation mechanism (validated offline): the parent's `stream()`
> activates a `ReasoningChannel` via the `ContextVar`. The parent's compiled
> graph calls a tool whose `func` runs the nested agent (or emits directly); the
> nested code calls `emit_reasoning(...)`, which targets the SAME active channel
> because the ContextVar is still set in that call stack. The parent's stream
> loop drains those lines as `Chunk(reasoning=...)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tool_agent_stream.py
from aixon.agents.tool_agent import ToolAgent
from aixon.llm import LLM
from aixon.message import Chunk, Message
from aixon.registry import get_registry

from langchain_core.messages import AIMessage
from tests._fakes import FakeChatModel


def _install(monkeypatch, llm, script):
    fake = FakeChatModel(script=script)
    monkeypatch.setattr(type(llm), "chat_model", property(lambda self: fake))


def _tool_call(name, args, id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])


def test_stream_yields_reasoning_then_content_then_done(monkeypatch):
    def adder(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    class StreamAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [adder]

    agent = get_registry().resolve("streamagent")
    _install(
        monkeypatch,
        agent.llm,
        [
            _tool_call("adder", {"a": 2, "b": 2}),
            AIMessage(content="The total is 4."),
        ],
    )

    chunks = list(agent.stream([Message(role="user", content="add 2 and 2")]))

    # All chunks are neutral Chunks.
    assert all(isinstance(c, Chunk) for c in chunks)
    # A reasoning chunk mentions the tool.
    assert any("adder" in c.reasoning for c in chunks if c.reasoning)
    # A content chunk carries the final answer.
    assert any("The total is 4." in c.content for c in chunks if c.content)
    # The stream terminates with done=True.
    assert chunks[-1].done is True


def test_stream_no_tool_still_streams_content_and_done(monkeypatch):
    def noop(text: str) -> str:
        """noop"""
        return text

    class DirectAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [noop]

    agent = get_registry().resolve("directagent")
    _install(monkeypatch, agent.llm, [AIMessage(content="immediate answer")])

    chunks = list(agent.stream([Message(role="user", content="hi")]))

    assert any("immediate answer" in c.content for c in chunks if c.content)
    assert chunks[-1].done is True
```

```python
# tests/test_nested_reasoning.py
"""A nested agent's emit_reasoning must bubble to the parent's stream() via the
active contextvars ReasoningChannel (contract §2.3 / §2.5)."""

from aixon.agents.tool_agent import ToolAgent
from aixon.llm import LLM
from aixon.message import Message
from aixon.reasoning import emit_reasoning
from aixon.registry import get_registry

from langchain_core.messages import AIMessage
from tests._fakes import FakeChatModel


def _tool_call(name, args, id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])


def test_nested_worker_reasoning_bubbles_to_parent_stream(monkeypatch):
    # The nested "agent" is a tool whose body emits reasoning exactly as a
    # nested ToolAgent would (its steps call emit_reasoning against the active
    # channel set by the parent's stream()).
    def nested_worker(text: str) -> str:
        """A nested worker that reasons before answering."""
        emit_reasoning("nested: analysing the request")
        emit_reasoning("nested: producing an answer")
        return "nested-result"

    class ParentAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [nested_worker]

    parent = get_registry().resolve("parentagent")
    fake = FakeChatModel(
        script=[
            _tool_call("nested_worker", {"text": "go"}),
            AIMessage(content="Parent done."),
        ]
    )
    monkeypatch.setattr(type(parent.llm), "chat_model", property(lambda self: fake))

    chunks = list(parent.stream([Message(role="user", content="please work")]))
    reasoning_text = "".join(c.reasoning for c in chunks if c.reasoning)

    # The parent's own tool-call label AND the nested worker's two lines all
    # surfaced through the parent stream.
    assert "Calling nested_worker..." in reasoning_text
    assert "nested: analysing the request" in reasoning_text
    assert "nested: producing an answer" in reasoning_text
    assert any("Parent done." in c.content for c in chunks if c.content)
    assert chunks[-1].done is True


def test_nested_toolagent_as_tool_propagates_reasoning(monkeypatch):
    # End-to-end with a REAL nested ToolAgent wired via as_tool() + coerce_tools.
    # Each agent gets its OWN scripted FakeChatModel via per-instance _chat_model
    # (LLM.chat_model is a cached property over self._chat_model per contract §1.3),
    # so root and child run independent scripts.
    def leaf_tool(text: str) -> str:
        """leaf"""
        return "leaf:" + text

    class ChildAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [leaf_tool]

    child = get_registry().resolve("childagent")
    child_fake = FakeChatModel(
        script=[
            _tool_call("leaf_tool", {"text": "x"}),
            AIMessage(content="child answer"),
        ]
    )
    object.__setattr__(child.llm, "_chat_model", child_fake)

    class RootAgent(ToolAgent):
        llm = LLM("fake-1", provider="fake")
        tools = [child.as_tool()]

    root = get_registry().resolve("rootagent")
    root_fake = FakeChatModel(
        script=[
            _tool_call("childagent", {"text": "delegate"}),
            AIMessage(content="root answer"),
        ]
    )
    object.__setattr__(root.llm, "_chat_model", root_fake)

    chunks = list(root.stream([Message(role="user", content="do it")]))
    reasoning_text = "".join(c.reasoning for c in chunks if c.reasoning)

    # Root labelled its call to the child; the child labelled its call to leaf.
    assert "Calling childagent..." in reasoning_text
    assert "Calling leaf_tool..." in reasoning_text
    assert any("root answer" in c.content for c in chunks if c.content)
    assert chunks[-1].done is True
```

> **On per-instance `_chat_model` patching:** `LLM.chat_model` is a cached
> property over `self._chat_model` (contract §1.3). Patching the *type's*
> property would make root and child share one fake; instead set
> `object.__setattr__(agent.llm, "_chat_model", fake)` per instance so each
> agent runs its own script. Verify by reading `aixon/llm.py` before writing
> this test — if Plan 2 caches under a different attribute name, use that exact
> name (keep ONE form, not both).

- [ ] **Step 2: Run tests (stream tests should pass; nested tests exercise propagation first)**

Run: `.venv/bin/python -m pytest tests/test_tool_agent_stream.py tests/test_nested_reasoning.py -v`
Expected: the stream tests pass (stream was implemented in Task 4); the nested
tests run propagation for the first time. If any assertion fails, debug with
superpowers:systematic-debugging and fix `aixon/agents/tool_agent.py` (most
likely cause: draining reasoning at the wrong point in the stream loop, or the
`_chat_model` attribute name differing from §1.3 — read `aixon/llm.py`).

- [ ] **Step 3: Make them pass**

Read `aixon/llm.py` to confirm the `_chat_model` cached-attribute name and
finalize the nested test's per-instance patch to match. If a production gap
exists, the minimal fix is the `stream` drain ordering in Task 4's code; adjust
there per TDD.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tool_agent_stream.py tests/test_nested_reasoning.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add aixon/agents/tool_agent.py tests/test_tool_agent_stream.py tests/test_nested_reasoning.py
git commit -m "feat(#3): ToolAgent.stream + nested-agent reasoning propagation via ReasoningChannel"
```

---

### Task 6: Confirm `Agent.as_tool` stays neutral (contract §2.4)

**Files:**
- Test: `tests/test_as_tool_neutral.py`
- (No production change — this is the §2.4 guard. `aixon/agent.py` MUST be left
  exactly as Plan 1 built it: `as_tool` returns a neutral `AgentTool`.)

**Interfaces:**
- Consumes: `aixon.agent.{Agent, AgentTool}`; `aixon._tools.coerce_tools`.
- Produces: a regression test pinning that `as_tool` returns the neutral
  `AgentTool` dataclass (NOT a LangChain tool) and that the LangChain
  conversion happens only via `coerce_tools`.

> The contract assigns the `as_tool`-neutrality decision to Plan 3 ("one edit,
> owned by Plan 3") but the edit is to KEEP it neutral — i.e. make NO change.
> This task records that decision as an executable guard so a later change that
> accidentally returns a LangChain tool from `as_tool` fails CI.

- [ ] **Step 1: Write the test**

```python
# tests/test_as_tool_neutral.py
"""Guard: Agent.as_tool() returns the NEUTRAL AgentTool (contract §2.4).
LangChain conversion is exclusively coerce_tools' job."""

from langchain_core.tools import BaseTool

from aixon.agent import Agent, AgentTool
from aixon.message import Chunk, Message
from aixon.registry import get_registry
from aixon._tools import coerce_tools


def _concrete(name_cls, reply):
    return type(
        name_cls,
        (Agent,),
        {
            "invoke": lambda self, messages: Message(
                role="assistant", content=reply + ":" + messages[-1].content
            ),
            "stream": lambda self, m: iter([Chunk(done=True)]),
        },
    )


def test_as_tool_returns_neutral_agenttool_not_langchain():
    _concrete("PlainAgent", "p")
    tool = get_registry().resolve("plainagent").as_tool()
    assert isinstance(tool, AgentTool)
    assert not isinstance(tool, BaseTool)


def test_as_tool_output_is_consumable_by_coerce_tools():
    _concrete("PlainAgent", "p")
    tool = get_registry().resolve("plainagent").as_tool()
    [lc_tool] = coerce_tools([tool])
    assert isinstance(lc_tool, BaseTool)
    # Round-trip: the LangChain tool runs the neutral agent.
    assert lc_tool.invoke({"text": "ping"}) == "p:ping"
```

- [ ] **Step 2: Run test to verify it passes immediately**

Run: `.venv/bin/python -m pytest tests/test_as_tool_neutral.py -v`
Expected: PASS (2 tests) with NO change to `aixon/agent.py`. (If it fails
because `as_tool` returns a LangChain tool, that is a regression introduced
elsewhere — restore Plan 1's neutral `as_tool`.)

- [ ] **Step 3: Full suite + import-without-langchain sanity**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — all tests across Plans 1–3.

Run (sanity that the neutral boundary holds — `aixon` imports without langchain
forced at top level; private langchain imports are lazy):
```bash
.venv/bin/python -c "import aixon; print('import ok'); print('ToolAgent' in aixon.__all__, 'emit_reasoning' in aixon.__all__, 'reasoning_channel' in aixon.__all__)"
```
Expected:
```
import ok
True True True
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_as_tool_neutral.py
git commit -m "test(#3): guard Agent.as_tool stays neutral (AgentTool, not a LangChain tool)"
```

---

## Self-Review

**Spec coverage (contract §2 only):**
- §2.1 `aixon/reasoning.py` — `ReasoningChannel`, `current_channel`,
  `emit_reasoning` (no-op when no channel), `reasoning_channel`
  `@contextmanager`, backed by `contextvars.ContextVar` (NOT thread-local),
  nesting-safe via token reset → Task 1. ✓
- §2.2 `aixon/agents/tool_agent.py` — `ToolAgent(Agent, abstract=True)`,
  `_suffix="Agent"`, declarative `llm`/`prompt`/`tools`/`max_iterations`/
  `max_execution_time`, required-`llm` validation, `invoke` built with
  `langchain.agents.create_agent(self.llm.chat_model, coerce_tools(self.tools),
  system_prompt=self.prompt or None)`, invoked as
  `agent.invoke({"messages": to_langchain(messages)})` and converted back with
  `from_langchain(result["messages"][-1])`, returning a neutral `Message` with
  `.reasoning` set; `stream` over `agent.stream(..., stream_mode="updates")`
  yielding `Chunk(reasoning=...)` + `Chunk(content=...)` + final
  `Chunk(done=True)` → Tasks 4–5. NOT `create_tool_calling_agent`/
  `AgentExecutor` (removed in 1.x), NOT `langgraph.prebuilt.create_react_agent`
  (deprecated). ✓
- §2.3 `aixon/_tools.py::coerce_tools` — `AgentTool`→`StructuredTool`,
  `BaseTool`/`@tool` passthrough, plain callable wrapped, else `AixonError`;
  nested-agent reasoning reaches the parent because `emit_reasoning` targets the
  active channel → Task 3 + Task 5 (`test_nested_*`). ✓
- §2.4 `Agent.as_tool` stays neutral — recorded as an executable guard with NO
  production edit → Task 6. ✓
- §2.5 Tests — hermetic, offline; reuse Plan 2's `FakeChatModel`/`make_llm`/
  `LLM("fake-1", provider="fake")` from `tests/_fakes.py` (NOT redefined,
  §9.1); a scripted `FakeChatModel` drives the REAL `create_agent` graph through
  a tool call then a final answer (Task 2 guardrail mirrors probe2.py);
  explicitly tests nested-agent `emit_reasoning` bubbling to the parent
  `stream()` → Tasks 2, 4, 5. ✓
- §2.6 Exports — `ToolAgent`, `emit_reasoning`, `reasoning_channel` from `aixon`
  → Tasks 1, 4. ✓

**Cross-plan reconciliation (§9):**
- §9.1 single-owner fakes — Plan 3 imports `FakeChatModel`/`make_llm` from
  `tests/_fakes.py`; defines none. ✓
- §9.2 extras — no new extra; langchain/langgraph already in `llm`; no ceiling
  added (Task 0). ✓
- §9.4 export guard — Task 4 Step 5 verifies `import aixon` works bare; applies
  the `try/except ImportError` guard if needed. ✓
- §9.5 dedicated venv — every install/run step uses `.venv/bin/python` (NOT bare
  `pytest`, NOT another project's interpreter); NO `<1` ceiling anywhere. ✓

**Contract ambiguity resolved:** `max_iterations`/`max_execution_time` have no
direct `create_agent` constructor knob in langchain 1.x; documented mapping —
`max_iterations` → `config={"recursion_limit": 2 * max_iterations + 1}` (per
super-step), `max_execution_time` → an aixon-enforced wall-clock backstop. No
parameter is invented; both attributes stay on the declarative surface verbatim
(Global Constraints note + Task 4). ✓

**Placeholder scan:** No `TODO`/`TBD`/"add error handling"/"similar to above"
left. Every code step is complete and runnable. The one verify-then-pick is the
`_chat_model` cached-attribute name in Task 5's nested test (read `aixon/llm.py`,
use the single matching form) — a bounded verification against Plan 2, not a
placeholder. ✓

**Type consistency vs contract:**
- `emit_reasoning(text: str) -> None`, `reasoning_channel() -> Iterator[ReasoningChannel]`,
  `current_channel() -> ReasoningChannel | None` — match §2.1 verbatim.
- `coerce_tools(tools: list) -> list[BaseTool]` — matches §2.3; the three
  accepted forms + `StructuredTool.from_function` mapping are exact.
- `ToolAgent` attribute names/defaults (`prompt=""`, `tools=[]`,
  `max_iterations=15`, `max_execution_time=600`, `_suffix="Agent"`) and method
  signatures (`invoke(messages: list[Message]) -> Message`,
  `stream(messages: list[Message]) -> Iterator[Chunk]`) match §2.2.
- Consumes Plan 1 (`Agent`, `AgentTool(name, description, func)`, `Message`/
  `Chunk` field names incl. `Message.reasoning` and
  `Chunk.{content,reasoning,done}`, `get_registry`) and Plan 2
  (`LLM(...).chat_model`, `aixon._langchain.{to_langchain, from_langchain}`,
  `LLM("fake-1", provider="fake")`, `FakeChatModel`) exactly as the contract
  pins them. `ToolAgent` validates `llm` on concrete subclasses by overriding
  `Agent._validate_subclass()` (the base hook, called before registration) — NOT
  `__init_subclass__` — mirroring §1.5's `LLMAgent` rule and avoiding the
  register-then-validate ghost. ✓
- Neutral boundary held: langchain imported lazily inside methods/`coerce_tools`;
  `aixon` imports without langchain; `invoke`/`stream` accept and return only
  `Message`/`Chunk`; `as_tool` returns neutral `AgentTool` (Task 6 guard). ✓
</content>
</invoke>
