# Olympus → aixon: differences and known problems

This document compares **aixon** (the reusable framework) against **olympus-ai-server**
(the application it was extracted from), and records the problems found in aixon
while auditing the whole codebase.

It is the sibling of the restmcp ← mcp-financial-server extraction: same idea, a
real app distilled into a framework. The goal of the extraction was to keep the
good ideas and turn the messy parts into deliberate, tested abstractions. Mostly
it succeeded; the open issues below are what's left.

> Scope of the audit: every module under `aixon/`, all of `docs/`, the
> `examples/support_assistant/` showcase, `pyproject.toml`, and the 50-file test
> suite (`288 passed, 4 skipped`). Olympus was read in full for the comparison.

---

## 1. The big picture

| | olympus-ai-server | aixon |
|---|---|---|
| **Nature** | Concrete app (Athena, Saori, Detetive… agents) | Reusable framework, agents live in consumer projects |
| **Web layer** | Flask singleton, OpenAI wire format **hardcoded** | FastAPI/ASGI + pluggable `ProtocolAdapter` (OpenAI + Anthropic) |
| **Agent runtime** | `Model` = `AgentExecutor` (`langchain-classic`), tightly LangChain-coupled | `LLMAgent` / `ToolAgent` / `Orchestrator` over LangGraph `create_agent` |
| **Boundary** | LangChain/OpenAI types leak across layers | **Neutral boundary**: only `Message[]` / `Chunk` cross `invoke`/`stream` |
| **Streaming "thoughts"** | Thread-local `thought_queue`, manual drains, keepalive state machine | `ReasoningChannel` via `ContextVar`, additive `Chunk.reasoning` |
| **Multi-agent** | Nested `Model.agents` exposed as `as_tool()` (ad-hoc) | First-class `Orchestrator` with 3 tiers + composition-cycle guard |
| **Registration** | `__init_subclass__` + eager instantiation at import | `__init_subclass__`, validate-before-register, `abstract=True` opt-out |
| **Models named** | Fictional `gpt-5.4/5.5/...` everywhere | Real `gpt-4o-mini` (in source) |
| **Tests** | 3 integration smoke files, no unit coverage of core | 50 files, 288 tests, conftest registry reset |
| **Packaging** | `requirements.txt`, Docker/Cloud Run app | `pyproject.toml` with 8 optional extras + `aixon` console script |

---

## 2. What aixon improved (the wins)

These are real, verified improvements over olympus.

**2.1 The neutral boundary is genuine.** In olympus, LangChain message objects,
OpenAI JSON shapes, and `AgentExecutor` internals flow across layers; an agent
"is" a LangChain executor. In aixon, `Agent.invoke(list[Message]) -> Message` and
`stream() -> Iterator[Chunk]` are the *only* types that cross. Provider SDKs stay
inside `LLM`/`providers/`, wire formats stay inside `server/adapters/`. This is
what lets you swap an agent's LLM (OpenAI→Anthropic) without touching an
Orchestrator that calls it, and mount a new wire format without touching any
agent. It is enforced, not aspirational.

**2.2 Protocol is pluggable instead of hardcoded.** Olympus bakes the OpenAI
chunk/error/usage JSON into `server/server.py`. aixon factors this into
`ProtocolAdapter` (`aixon/server/protocol.py`) with two concrete adapters
(`OpenAIAdapter`, `AnthropicAdapter`). Adding a wire format is a subclass, not a
server edit. The Anthropic adapter exists specifically to *prove* the neutral
types aren't secretly OpenAI types.

**2.3 Thought tracking went from fragile to principled.** Olympus passes a
`thread-local thought_queue` between nested agents and drains it manually with
keepalive logic — concurrent streams can interfere. aixon replaces this with a
`ReasoningChannel` built on `contextvars.ContextVar` (`aixon/reasoning.py`),
with correct LIFO token reset and a safe no-op when no channel is active. This
is the single biggest reduction in accidental complexity.

**2.4 Multi-agent is first-class and guarded.** Olympus orchestration is "a model
lists other models in `agents` and they become tools." aixon promotes this to an
`Orchestrator` base with three tiers, an explicit graph, and a **composition-cycle
guard** (`orchestrator.py`, DFS with back-edge detection over referenced agent
classes) that catches `A→B→A` at build time instead of at runtime stack overflow.

**2.5 Safer registration.** aixon validates the suffix/naming rule *before*
registering (`agent.py`), so a mis-named class raises `NamingError` at import
without leaving a ghost half-registration, and correctly compensates for
`ABCMeta` populating `__abstractmethods__` after `__init_subclass__`.
`abstract=True` intermediate classes opt out cleanly.

**2.6 Auth is correct.** `_AuthMiddleware` (`server/server.py`) is pure-ASGI (no
body buffering, so SSE is unaffected), constant-time (`hmac.compare_digest`),
no-op when `AUTH_API_KEY` is unset, supports comma-separated keys, and has an
exact-path public allowlist computed from each adapter's *mounted* path.

**2.7 Real models, real tests, real packaging.** The fictional `gpt-5.4` family
is gone from source; the project ships proper extras and a CLI; the test suite
went from ~3 smoke files to broad coverage of adapters, auth, orchestrator tiers,
CLI, and the reasoning channel.

---

## 3. Problems found in aixon

Ordered by severity. File/line citations are against the current tree. The two
HIGH items were directly verified by reading the code; the rest come from the
audit with line cites and are worth confirming as you fix them.

### HIGH

**3.1 — Tier 1 Orchestrator ignores the `supervisor` LLM (documented behavior is not implemented).**
`aixon/agents/orchestrator.py:216-231`. The whole premise of Tier 1
(`docs/orchestrator.md`: "the supervisor **LLM** decides which worker handles
each turn") is not real. `_route_supervisor` never consults `self.supervisor`;
it counts assistant messages and runs workers in fixed order
(`ran = sum(... m.role == "assistant")`, then `workers[ran]`, else `END`). The
docstring even admits "A real LLM-driven supervisor replaces this hook." A user
declaring `supervisor = LLM("gpt-4o-mini")` with three workers gets deterministic
round-robin, not routing — the `supervisor` is required to enter Tier 1 and then
discarded.
*Compounding:* the round-robin counter assumes the input has **zero** assistant
messages and each worker appends **exactly one**. In the normal server case
(multi-turn history passed into `invoke`), any prior assistant message makes it
skip workers or terminate immediately. Off-by-N routing on real conversations.
**Fix:** either implement LLM routing, or downgrade the docs to "round-robin"
and stop requiring an unused `supervisor`. And count workers via a dedicated
state field, not by scanning message roles.

**3.2 — `format_stream_chunk` silently drops data on multi-field chunks.**
`aixon/server/adapters/openai.py:75-86` and
`aixon/server/adapters/anthropic.py` (same shape). Both use a mutually-exclusive
ladder: `if chunk.done: return delta={} …; if chunk.content: …; if chunk.reasoning: …`.
But `Chunk` (`message.py`) explicitly allows `content` + `reasoning` + `done` to
co-occur. Verified consequences:
- `Chunk(content="final", done=True)` emits only `{"delta": {}, "finish_reason": "stop"}` — **the final content is lost.**
- `Chunk(content="hi", reasoning="thinking")` emits only the content — **reasoning is lost.**

It's latent today *only* because the built-in `LLM`/`ToolAgent`/`Orchestrator`
streams happen to emit content, reasoning, and done in separate chunks — an
undocumented coupling between the runtime and the adapter contract. Any custom
agent that fills the dataclass as documented gets truncated streams.
**Fix:** build the `delta` additively (include `content` and `reasoning` when
present) and treat `done` as an additional flag, not an exclusive branch. There
is no test covering a both-fields chunk (test gap).

### MEDIUM

**3.3 — Orchestrator `timeout` and ToolAgent `max_execution_time` are no-ops in `invoke`.**
`orchestrator.py` (deadline computed, only checked *after* `graph.invoke` returns)
and `tool_agent.py` (same: `agent.invoke()` runs to completion, deadline checked
afterward and only *logs*). The docs sell these as a "wall-clock backstop" against
runaway graphs; in the non-streaming path they bound nothing — a one-hour run
runs the full hour, then raises/logs. (`stream` is better: it breaks between
updates, though it still can't interrupt one long tool call.)
**Fix:** pass the budget into the LangGraph/agent execution (recursion/step
limits or a cancel token), or document them as post-hoc and stop calling them a
backstop.

**3.4 — Tier 2 fan-out (`route_<node>` returning a list) likely doesn't do what the docs claim.**
`orchestrator.py` builds `add_conditional_edges(name, self._wrap_router(name))`
**with no path_map**, while `docs/orchestrator.md` promises "all listed nodes run
in parallel" when a router returns a list. Compare the Tier 1 supervisor edge,
which *does* pass a `path_map`. LangGraph parallel fan-out generally needs `Send`
objects, and string-target conditional routing usually needs a path map. The
code/doc mismatch is certain; the exact runtime failure should be confirmed with
a test against langgraph≥1.0.

**3.5 — `Registry.resolve` single-agent fallback masks typos.**
`aixon/registry.py:46-47`. When exactly one agent is registered, `resolve(name)`
returns it **regardless of the name** — `resolve("does-not-exist")` returns the
lone agent instead of raising `AgentNotFoundError`. A single-agent server never
reports a bad `model` field. Undocumented and surprising.

**3.6 — `Agent._registered` class flag desyncs from `reset_registry()`.**
`aixon/agent.py` sets `type(self)._registered = True`. `reset_registry()`
(`registry.py`) rebinds a fresh `Registry`, but the class still believes it's
registered, so re-instantiating short-circuits and **never re-registers**. The
registry is empty while the class thinks it isn't. Tests that define agent
classes at import time and rely on the autouse reset can get silently empty
registries. (`clear()` mutates in place; `reset_registry()` replaces the instance
— the two reset paths are inconsistent.)

**3.7 — `from_langchain` loses `tool_call_id`/`name` on round-trip.**
`aixon/_interop/messages.py`. `to_langchain` maps a tool `Message` to
`ToolMessage(tool_call_id=…, name=…)`, but `from_langchain` never reads those
back. A `Message → LangChain → Message` round-trip on a tool message drops its
`tool_call_id`/`name`, and the returned `role="tool"` message has an empty
`tool_call_id`, which can break the next `to_langchain` (ToolMessage requires
one). Multi-turn tool conversations through the boundary can produce malformed
messages.

**3.8 — `aixon serve` is OpenAI-only; the multi-adapter feature is unreachable from the CLI.**
`aixon/cli.py` calls `Server.get_instance()`, which builds the singleton with the
default `[OpenAIAdapter()]` and offers no `--adapter`/`--anthropic` flag. The
`mount_prefix` multi-dialect capability — the headline fix for the old adapter
collision — can only be used via the Python API. The `Server` singleton also
silently *discards* its `adapters` argument if an instance already exists, so
importing a `main.py` that does `Server()` before `serve` locks in OpenAI-only.
A singleton that silently ignores constructor args is a footgun.

**3.9 — Non-streaming responses always report empty `usage`.**
`server/server.py` passes `usage={}`; the neutral `Message` has no token-count
field and `agent.invoke` returns none, so `usage` can never be populated. OpenAI
clients get `"usage": {}` where they expect `prompt_tokens`/`completion_tokens`.
Either wire counts through or document that usage is intentionally empty.

### LOW

**3.10 — Fictional `gpt-5.4` survives in the *design* docs and a test.** Source is
clean, but `docs/superpowers/aixon-interface-contract.md`, the spec, several plan
files, and `tests/test_providers.py` still use `gpt-5.4`. The interface contract
is the doc people read to learn the framework — same confusion ISSUES.md #1 set
out to kill.

**3.11 — Docs reference a `Provider` *enum* that doesn't exist.** `docs/agents.md`
shows a `Provider` enum (`OPENAI/ANTHROPIC/GOOGLE`) and
`register_provider(provider, cls)`. In reality `Provider` is an ABC
(`providers/base.py`), provider names are lowercase strings, and
`register_provider` takes a single instance. Copy-pasting the documented
signature raises `TypeError`.

**3.11b — Stale "Flask" docstring in the OpenAI adapter.** `aixon/server/adapters/openai.py:5`
describes itself as a `"Flask handler to pure neutral translation"`. aixon uses
**no Flask** — the server is FastAPI/ASGI + uvicorn (`server/server.py`,
`server/__init__.py`). This is leftover text copied from olympus (which *was*
Flask) and never updated; the only occurrence of the word "Flask" in the whole
source tree, and it's wrong.

**3.12 — Anthropic adapter never emits the `message_start` event the docs claim.**
`docs/server.md` lists `message_start` among demonstrated Anthropic SSE events;
the adapter only emits `content_block_delta`, `message_delta`, `message_stop`. A
strict Anthropic SSE client expecting `message_start` breaks.

**3.13 — Version / maturity drift.** `pyproject.toml` is `version = "0.0.1"` with
`Development Status :: 3 - Alpha`, while the README presents a feature-complete
framework and says `pip install aixon`. There is no `__version__` and no
`aixon --version` (the README/RTK convention treats `--version` as a basic
sanity check).

**3.14 — Smaller smells.** `_dim(_text)` in `cli.py` ignores its argument and is
mis-named (it answers "should we dim?", returning `isatty()`); the `aixon new`
scaffold double-declares `uvicorn` (already pinned by the `server` extra); the
example README references a `.env.example` that isn't in the directory; the
README `Orchestrator` snippet uses undefined `BillingAgent/TechAgent/PlannerAgent`
(fine as a fragment, `NameError` if pasted).

### Note on ISSUES.md

The repo's `ISSUES.md` (3 entries, all RESOLVED) is accurate and the fixes are
real — the adapter `mount_prefix`, the CLI `_ensure_cwd_on_path`, and the
`gpt-5.4` removal all landed and are tested. This document does **not** duplicate
those; everything above is newly found.

---

## 4. Suggested priority

1. **3.1 + 3.2** (HIGH) — these are the gap between documented and actual behavior
   for the two headline features (orchestration, streaming). Fix the code or fix
   the docs, but close the gap; add the missing both-fields-chunk and
   multi-turn-routing tests.
2. **3.3 / 3.4** — make the safety knobs real or stop advertising them; confirm
   Tier 2 fan-out against langgraph≥1.0.
3. **3.5 / 3.6** — the registry footguns; cheap to fix, surprising in production.
4. The LOW doc fixes (3.10–3.12) — they actively mislead new users.

---

## 5. Resolution log

Status after the `fix/tier1-and-docs` branch (full suite: **294 passed, 4 skipped**).

| # | Status | Notes |
|---|---|---|
| **3.1** | ✅ FIXED | Tier 1 now does real LLM routing: `_route_supervisor`/`_supervisor_choose` consult `self.supervisor` (conversation + worker roster), pick the next worker or DONE, with a safety net for unanswered user turns. The message-counting hack (and its multi-turn skip / stale-return bugs) is gone. 6 regression tests in `test_orchestrator_tier1_routing.py` encode all three old failure modes. |
| **3.4** | ❎ FALSE POSITIVE | Tier 2 list fan-out works. `test_tier2_list_fanout_runs_multiple_nodes` returns `["left","right"]` with no path_map and asserts ≥3 assistant messages — it passes. LangGraph's conditional edge accepts a list of node names for fan-out. Doc was correct. |
| **3.10** | ✅ FIXED (partial) | `gpt-5.4` removed from the interface contract and `test_providers.py`. Dated `docs/superpowers/plans|specs` left as historical process artifacts (not living docs). |
| **3.11** | ✅ FIXED | `docs/agents.md` now describes `Provider` as an ABC with lowercase string names and `register_provider(instance)` — was a fictional enum + 2-arg signature (a Plan 8 doc error). |
| **3.11b** | ✅ FIXED | Stale "Flask handler" docstring removed from `adapters/openai.py`; no "Flask" remains in source. |
| **3.12** | ✅ FIXED | `docs/server.md` SSE events corrected to `content_block_delta`/`message_delta`/`message_stop` (the adapter never emits `message_start`). |
| 3.2, 3.3, 3.5, 3.6, 3.7, 3.8, 3.9, 3.13 | ⏳ OPEN | Not in this branch's scope. 3.2 (stream multi-field drop) and 3.8 (CLI can't reach multi-adapter) are the highest-value remaining. |
| **3.14** | ⚠️ CORRECTION | The `.env.example` sub-claim is **stale** — the file exists in `examples/support_assistant/`. The `_dim(_text)` and duplicate-`uvicorn` sub-claims still hold. |
