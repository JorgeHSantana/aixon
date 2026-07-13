# Architecture

## Layers

```
HTTP / SSE
    │
ProtocolAdapter          ← translates wire ↔ neutral types (Message[], Chunk)
    │
Server (Registry)        ← resolves agent name from request's "model" field
    │
Agent.invoke / stream    ← speaks ONLY neutral types; no wire type crosses here
    │
    ├── LLMAgent  → LLM → Provider → vendor SDK (OpenAI / Anthropic / Google)
    ├── ToolAgent → create_agent (LangGraph) → tools: Retriever · Connector · Agent.as_tool()
    └── Orchestrator → LangGraph → nodes (Agents) → ...
```

Each layer depends only on the layer directly below it. Provider SDKs, LangChain
internals, and wire-format objects are contained within their layer and never
cross upward.

---

## The neutral boundary

`Agent.invoke` and `Agent.stream` speak two neutral types only:

```python
# aixon/message.py — the only types that cross the boundary

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Role = Literal["system", "developer", "user", "assistant", "tool"]

@dataclass
class Message:
    role: Role
    content: str = ""
    name: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    reasoning: Optional[str] = None
    usage: Optional[dict[str, int]] = None

    def to_dict(self) -> dict[str, Any]: ...   # omits empty optional fields

@dataclass
class Chunk:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
```

`Message` carries a conversation turn. `Chunk` is a streaming delta — `content`
and `reasoning` are additive text; the final `Chunk` has `done=True`. `developer`
is OpenAI's system-role alias: every conversion point (`to_langchain`, `LLMAgent`,
`ToolAgent`) treats it exactly like `system` (a leading `developer` message wins
over the agent's class-level `prompt`, same as `system` would). `Message.usage`,
when present, carries the provider's real token usage in OpenAI shape
(`{"prompt_tokens", "completion_tokens", "total_tokens"}`); `None` means the
provider reported none, so a consumer (e.g. the Server) may fall back to
estimating. `Chunk.tool_calls` carries neutral tool-call dicts the agent wants
surfaced to the client for execution (see [server.md](server.md), client tools).

`Message.reasoning`/`Chunk.reasoning` have two sources: `ToolAgent`'s own
tool-call step labels (unchanged), and — when the agent's `LLM` has the
[reasoning knob](agents.md#reasoning-extended-thinking--reasoning-effort) on —
the model's own native reasoning (Anthropic `thinking` blocks, or the
`reasoning_content` convention some OpenAI-compatible providers use),
extracted by `aixon._interop.messages.reasoning_from_message` and merged into
the same `ReasoningChannel` ahead of the labels it led to. `LLMAgent`/`LLM.stream`
surface the same extraction directly, one `Chunk(reasoning=...)` per delta
before the matching `Chunk(content=...)` of that delta.

**What the neutral boundary prevents:** a `ToolAgent` can swap its LLM from
OpenAI to Anthropic without touching the `Orchestrator` that calls it as a node.
The `Server` can mount a new `ProtocolAdapter` without touching any `Agent`.
Provider types (`langchain_openai.ChatOpenAI`) stay inside `LLM` and never reach
an agent's `invoke` signature.

---

## Protocol decoupling

`ProtocolAdapter` is the seam between wire formats and the neutral runtime. The
server mounts one or more adapters; each handles its own routes:

```
OpenAI client  ──→  OpenAIAdapter.parse_request  ──→  Message[]
                                                         ↓
                                                    agent.invoke
                                                         ↓
                ←── OpenAIAdapter.format_response  ←── Message

Anthropic SDK  ──→  AnthropicAdapter.parse_request ──→  Message[]
                                                         ↓
                                                    agent.invoke
                                                         ↓
                ←── AnthropicAdapter.format_response ←── Message
```

Adding a new wire format = adding a new `ProtocolAdapter` subclass. Nothing in
`Agent`, `LLM`, or `Registry` changes.

`aixon` ships two adapters:
- **`OpenAIAdapter`** — full OpenAI-compatible (`/v1/chat/completions`, `/v1/models`).
- **`AnthropicAdapter`** — full production dialect (`/v1/messages`), proof that
  the neutral types are not secretly OpenAI types — Anthropic's structurally
  different wire format (typed content blocks, `stop_reason`, named SSE events:
  `message_start`/`content_block_start`/`content_block_delta`/
  `content_block_stop`/`message_delta`/`message_stop`) translates through the
  same `Message`/`Chunk` boundary, including a stateful per-request stream
  session that sequences blocks and closes them cleanly on a mid-stream error.

See [server.md](server.md) for the adapter API.

---

## Request flow (end to end)

```
HTTP POST /v1/chat/completions
  body: {"model": "planneragent", "messages": [...], "stream": true}
  │
  ▼
ProtocolAdapter.parse_request(body)
  → ParsedRequest(model="planneragent", messages=[Message(...)], stream=True)
  │
  ▼
get_registry().resolve("planneragent")
  → PlannerAgent instance
  │
  ▼
agent.stream(messages)
  → Iterator[Chunk(content="..."), ..., Chunk(done=True)]
  │
  ▼
ProtocolAdapter.format_stream_chunk / format_stream_done
  → SSE: data: {"choices": [{"delta": {"content": "..."}}]}
  │
  ▼
HTTP response (streaming)
```

---

## Auto-registration

Agents self-register at class definition time. `autodiscover(package)` imports
every non-underscore module in a package, which triggers each class body —
and therefore each registration — without any explicit list to maintain.

```python
from aixon import autodiscover
from aixon.registry import get_registry

autodiscover("agents")           # imports agents/hello.py, agents/support.py, …
agent = get_registry().resolve("helloagent")
```

The registry is a process-global singleton. The `autouse` pytest fixture in
`tests/conftest.py` calls `reset_registry()` between tests so they stay
isolated.

---

## Suffix enforcement

Every concrete subclass of a base type must end with the declared `_suffix`.
The check runs in `Agent.__init_subclass__` — before the class is instantiated,
before the server starts.

```python
class Greeter(LLMAgent):      # ← NamingError raised here, at import time
    llm = LLM("gpt-4o-mini")

class GreeterAgent(LLMAgent): # ← fine
    llm = LLM("gpt-4o-mini")
```

Abstract intermediate classes opt out with `abstract=True` and are never
registered:

```python
class BaseResearchAgent(LLMAgent, abstract=True):
    prompt = "You are a research assistant."
    # no llm declared — subclasses must supply it

class WebResearchAgent(BaseResearchAgent):
    llm = LLM("gpt-4o-mini")   # ← registered as "webresearchagent"
```
