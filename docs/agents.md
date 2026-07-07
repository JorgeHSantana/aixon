# Agents

An **`Agent`** is the single executable unit in `aixon`. Every agent — regardless
of subtype — exposes the same interface:

```python
agent.invoke(messages: list[Message]) -> Message
agent.stream(messages: list[Message]) -> Iterator[Chunk]
agent.as_tool(name=None, description=None) -> AgentTool
```

This uniformity means a `ToolAgent` can be a node in an `Orchestrator`, an
`Orchestrator` can be a tool inside a `ToolAgent`, and the `Server` never needs
to know which subtype it is calling.

---

## Declaring an agent

Subclass one of the concrete types and set class attributes. The agent
self-registers when Python processes the class body — no call to a registration
function required.

### Common attributes (all subtypes)

| Attribute | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | class name lowercased | Registry key and API `model` field. |
| `description` | `str` | `""` | Human-readable purpose; shown in `aixon list` and the chat menu. |
| `aliases` | `list[str]` | `[]` | Alternate registry names. |
| `hidden` | `bool` | `False` | Exclude from `get_registry().public()` and the `aixon chat` menu. |
| `owned_by` | `str` | `"aixon"` | Shown in `/v1/models` response. |

---

## LLMAgent — direct LLM call

Use `LLMAgent` when you want a single LLM call with no tool loop — the simplest
path from question to answer.

```python
from aixon import LLMAgent, LLM

class PlannerAgent(LLMAgent):
    llm         = LLM("gpt-4o-mini", temperature=0.2)
    description = "Breaks complex goals into step-by-step plans"
    prompt      = "You are a concise strategic planner. Use numbered lists."
```

**Additional `LLMAgent` attributes:**

| Attribute | Type | Required | Description |
|---|---|---|---|
| `llm` | `LLM` | **Yes** | The language model. Missing `llm` on a concrete subclass raises `AixonError` at import time. |
| `prompt` | `str` | No | System prompt prepended to every `invoke`/`stream` call. |

**How it works:** `invoke` prepends the system prompt (if any) as a
`Message(role="system", content=self.prompt)` and delegates to
`self.llm.complete(messages)`. `stream` delegates to `self.llm.stream(messages)`,
yielding `Chunk` deltas and a final `Chunk(done=True)`.

### LLM — declaring a language model

```python
from aixon import LLM

# Explicit provider
llm = LLM("claude-3-5-haiku-20241022", provider="anthropic", temperature=0.3)

# Inferred provider (model prefix → provider):
#   gpt-* / o[0-9]* / text-*  →  openai
#   claude-*                   →  anthropic
#   gemini-*                   →  google
llm = LLM("gpt-4o-mini", temperature=0.2, max_tokens=4096)
```

The `LLM` object is lazy — it builds the underlying LangChain `BaseChatModel`
only on first use, so constructing an agent never requires a network call or an
API key to be present at import time.

**Provider inference table** (the model name's prefix selects the provider):

| Model prefix | Provider name |
|---|---|
| `gpt-*`, `o[0-9]*`, `text-*` | `"openai"` |
| `claude-*` | `"anthropic"` |
| `gemini-*` | `"google"` |

Provider names are lowercase strings, not an enum. To override inference, pass
`provider=` explicitly: `LLM("some-model", provider="openai")`.

For a custom backend, subclass the `Provider` ABC (`aixon.providers.base`) and
register a single instance before first use:

```python
from aixon.providers.base import Provider, register_provider

class MyProvider(Provider):
    name = "myvendor"
    env_key = "MYVENDOR_API_KEY"
    def build(self, model: str, **params):
        from my_sdk import ChatModel        # lazy import
        return ChatModel(model=model, **params)

register_provider(MyProvider())             # one instance, keyed by .name
# then: LLM("my-model", provider="myvendor")
```

---

## ToolAgent — LLM + tool-calling loop

Use `ToolAgent` when your agent needs to call external functions, query a
`Retriever`, or invoke another agent as a tool, then loop until it has a final
answer.

```python
from aixon import ToolAgent, LLM
from langchain_community.tools import DuckDuckGoSearchRun  # pip install langchain-community

from retrievers.library import LibraryRetriever

class ResearchAgent(ToolAgent):
    llm                = LLM("gpt-4o-mini", temperature=0.1)
    description        = "Researches topics using web search and the knowledge base"
    prompt             = "Always cite your sources. Think step by step."
    tools              = [LibraryRetriever, DuckDuckGoSearchRun()]
    max_iterations     = 15
    max_execution_time = 600
```

**Additional `ToolAgent` attributes:**

| Attribute | Type | Default | Description |
|---|---|---|---|
| `llm` | `LLM` | **Required** | The language model driving the loop. |
| `prompt` | `str` | `""` | System prompt. |
| `tools` | `list` | `[]` | Mix of `AgentTool`, `Retriever`, LangChain `@tool` functions, or any callable. All are coerced to `BaseTool` internally via `coerce_tools`. |
| `max_iterations` | `int` | `15` | Maximum tool-call rounds before the loop stops. |
| `max_execution_time` | `int` | `600` | Wall-clock timeout in seconds. |
| `tool_call_label` | `str` | `"Calling {name}..."` | `{name}`-templated reasoning label emitted before each tool call. Override for a friendlier phrase or i18n, e.g. `"Chamando {name}..."`. Consecutive duplicate labels are emitted once (a run calling the same tool N times in a row shows a single line). |

**Tool coercion:** anything in `tools` is normalized at runtime:
- An `AgentTool` (from `Agent.as_tool()` or `Retriever.as_tool()`) → `StructuredTool`
- A LangChain `BaseTool` or `@tool`-decorated function → passed through
- A plain callable → wrapped via `StructuredTool.from_function`

This means you can mix library tools, custom functions, and other agents freely.

### Nesting agents as tools

Any `Agent` exposes itself as a tool via `as_tool()`. The result is a neutral
`AgentTool` — coerced to a LangChain tool inside `ToolAgent` automatically.

```python
from aixon import ToolAgent, LLM

class OrchestratorAgent(ToolAgent):
    llm   = LLM("gpt-4o-mini")
    tools = [
        PlannerAgent().as_tool(description="Break the goal into steps"),
        ResearchAgent().as_tool(),
    ]
```

**Reasoning propagation:** when a nested agent emits reasoning (via the
`ReasoningChannel`), that reasoning bubbles up through the outer `stream()` as
`Chunk(reasoning=...)` deltas — so callers see the full chain of thought even
across nesting levels.

---

## ReflectiveAgent — evaluator-optimizer loop

Use `ReflectiveAgent` when a single generation pass isn't reliable enough:
it wraps a worker `Agent` in a review loop — a judge LLM scores each answer
against an objective rubric, and a rejected answer goes back to the worker
together with the judge's critique, up to `max_rounds` attempts.

```python
from aixon import LLM, ReflectiveAgent
from agents.gerente import GerenteAgent

class GerenteRevisadoAgent(ReflectiveAgent):
    name = "gerente-revisado"
    agent = GerenteAgent                 # class OR instance (like Orchestrator nodes)
    judge_llm = LLM("gpt-5.4-mini", temperature=0)
    judge_rubric = (
        "1. Every SQL statement returned was validated (no non-existent column).\n"
        "2. Any number quoted matches what the tools returned.\n"
        "3. The answer addresses the entire question."
    )
    max_rounds = 3
```

**`ReflectiveAgent` attributes:**

| Attribute | Type | Required | Description |
|---|---|---|---|
| `agent` | `Agent` (class or instance) | **Yes** | The worker that produces answers. Resolved once, at `__init__`, with the same `_instantiate` helper `Orchestrator` uses for its nodes. |
| `judge_llm` | `LLM` | **Yes** | The model that grades each answer. Often a cheaper/faster model than the worker's — judging is a classification task, not generation. |
| `judge_rubric` | `str` | **Yes** | Objective approval criteria, non-empty. See "Write an objective rubric" below. |
| `max_rounds` | `int` | No (default `3`) | Worker attempts before giving up, `>= 1`. |
| `judge_label` | `str` | No | Reasoning-channel label emitted before each judge call. Default: `"Avaliando a resposta…"`. |
| `retry_label` | `str` | No | Reasoning-channel label emitted before a retry. `{round}`/`{max}` are interpolated. Default: `"Refinando a resposta (rodada {round}/{max})…"`. |
| `exhausted_label` | `str` | No | Reasoning-channel label emitted when `max_rounds` is reached without approval. Default: `"Rodadas esgotadas — entregando a melhor tentativa."`. |

Missing `agent`/`judge_llm`, an empty `judge_rubric`, or `max_rounds < 1` on a
concrete subclass raises `AixonError` at import time — before registration
(the same validate-before-register precept as every other subtype), so a
misconfigured `ReflectiveAgent` never leaves a ghost entry in the registry.

**How it works — the loop:**

1. `invoke` runs the worker (`agent.invoke`) to get a first answer.
2. `emit_reasoning(judge_label)`, then the judge grades it: `judge_llm.complete`
   is called with the rubric and the question/answer pair.
3. The verdict is a text sentinel, following the `DELEGAR`/`END` precedent: if
   its first line (after `strip()`) is exactly `APROVADO`, the answer is
   returned as-is.
4. Otherwise the verdict IS the critique. If rounds remain,
   `emit_reasoning(retry_label)` and the worker is re-invoked with the
   critique appended to the conversation (a new message list — the caller's
   is never mutated).
5. If `max_rounds` is reached without an `APROVADO`, `exhausted_label` is
   emitted and the **last attempt is returned** — exhausting the rounds is
   *not* an exception. A quality shortfall must not crash a run that produced
   an answer; the caller decides what to do with a possibly-imperfect result.

`stream`/`astream` mirror `Orchestrator`: they run the loop under a fresh
reasoning channel, drain it as `Chunk(reasoning=...)` deltas, then yield the
final `Chunk(content=...)` and `Chunk(done=True)`. `ainvoke`/`astream` are
native (`agent.ainvoke` + `judge_llm.acomplete`), not thread-bridged.

**Cost and latency.** Each round re-runs the full worker call (and, on
rejection, a fresh judge call too) — a `max_rounds=3` run can cost up to 3×
the worker's tokens/latency plus the judge overhead. Keep `max_rounds` as low
as the rubric allows, and prefer a cheap `judge_llm`.

**Write an objective rubric.** `judge_rubric` should state checkable facts,
not vibes — "does it cite a source?", "do the numbers match the tool
results?", "is every requested field present?". A vague rubric ("sounds
right", "is helpful") degenerates into the judge approving on the first pass
regardless of quality, defeating the point of the loop.

A complete runnable example (scripted judge + worker, no API key needed) is
at [examples/reflective_review](../examples/reflective_review).

---

## Agent.as_tool — the neutral tool descriptor

```python
@dataclass
class AgentTool:
    name: str
    description: str
    func: Callable[[str], str]
    coroutine: Callable[[str], Awaitable[str]] | None = None  # optional async path
```

```python
tool = agent.as_tool()
tool = agent.as_tool(name="planner", description="Decomposes goals")
```

`func` wraps `agent.invoke`: each call creates a fresh
`[Message(role="user", content=text)]` — the agent's state never leaks between
tool calls. `as_tool()` also sets `coroutine` (wrapping `ainvoke`), so the tool
is **dual**: `coerce_tools` registers both, and the tool runs on the sync
(`invoke` → `func`) and async (`ainvoke` → `coroutine`) paths. The same
`AgentTool` shape is returned by `Retriever.as_tool()`, so
`ToolAgent.tools` handles both uniformly.

---

## Suffix rule reference

| Base class | `_suffix` | Valid example | Invalid (raises `NamingError`) |
|---|---|---|---|
| `LLMAgent` | `"Agent"` | `PlannerAgent` | `Planner`, `PlannerLLM` |
| `ToolAgent` | `"Agent"` | `ResearchAgent` | `Research`, `ResearchTool` |
| `Orchestrator` | `"Orchestrator"` | `SupportOrchestrator` | `Support`, `SupportAgent` |

**Abstract subtypes** (your own base classes) bypass the suffix check by passing
`abstract=True`. Their concrete subclasses are then validated:

```python
class BaseSupportAgent(ToolAgent, abstract=True):
    llm   = LLM("gpt-4o-mini")
    tools = [check_ticket]

class BillingAgent(BaseSupportAgent):     # valid: ends with "Agent"
    prompt = "You handle billing issues."

class TechAgent(BaseSupportAgent):        # valid
    prompt = "You handle technical issues."
```

---

## Invoke and stream examples

```python
from aixon.message import Message

# invoke — returns a Message
reply = PlannerAgent().invoke([Message(role="user", content="Plan a product launch")])
print(reply.content)

# stream — yields Chunk deltas
for chunk in ResearchAgent().stream([Message(role="user", content="Latest on LLMs")]):
    if chunk.reasoning:
        print("[reasoning]", chunk.reasoning)
    elif chunk.content:
        print(chunk.content, end="", flush=True)
```

## Async — `ainvoke` / `astream`

Every agent also exposes async methods. **Sync is the default; async is purely
additive** — existing sync code is untouched, and you opt into async only where
you want it.

```python
reply = await PlannerAgent().ainvoke([Message(role="user", content="Plan a launch")])

async for chunk in ResearchAgent().astream([Message(role="user", content="...")]):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

- `LLMAgent`, `ToolAgent` and `Orchestrator` implement `ainvoke`/`astream`
  **natively** over LangGraph's async path (`ainvoke`/`astream`), so they never
  block the event loop.
- A purely sync custom `Agent` (one that only implements `invoke`/`stream`)
  still gets working `ainvoke`/`astream` for free — the base bridges them to a
  worker thread.
- The neutral types are unchanged: `ainvoke` returns a `Message`, `astream`
  yields `Chunk`s.

**Async tools.** A `ToolAgent` tool may be an `async def` callable — it runs on
the async path (`ainvoke`/`astream`) and does real non-blocking I/O (e.g. an MCP
call via `Connector.aget`). An async tool requires that path: calling it from
sync `invoke` raises `NotImplementedError` (it is never silently skipped). Sync
tool callables work on **both** paths (under `ainvoke` they run in a thread
executor). So: use **sync** tools if you need the agent to work via both `invoke`
and `ainvoke`; use **async** tools when you commit to the async path and want
non-blocking I/O.

**Real timeouts (cancellation).** On the async path, `ToolAgent.max_execution_time`
and `Orchestrator.timeout` wrap the run in `asyncio.wait_for`, so an overrun is
**cancelled at the next await point** — provided the chain is genuinely async
(an async model, async tools). Sync work bridged to a thread cannot be
interrupted mid-call; bound that at the tool/IO layer (e.g. `Connector.timeout`).
The server (`docs/server.md`) awaits `ainvoke`/`astream`, so concurrent requests
no longer serialize.

---

## Registry helpers

```python
from aixon import get_registry

registry = get_registry()
registry.public()           # list of non-hidden agents
registry.all()              # every registered agent
registry.resolve("planner") # by name or alias
```

Agents with `hidden = True` remain callable but are excluded from `public()` and
the `aixon chat` selection menu.

---

## See also

- [Architecture overview](architecture.md) — how agents, retrievers, and the server compose
- [Retrieval](retrieval.md) — `Retriever.as_tool()` and the same `AgentTool` contract
