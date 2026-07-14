# Server

`aixon` ships a FastAPI/ASGI server (`Server`) backed by a pluggable
`ProtocolAdapter` layer. Adapters translate wire formats to neutral `Message`/`Chunk`
types — the agents never see a wire type. Adding a new wire format means adding a
new adapter class; no agent code changes.

---

## Install

```bash
pip install 'aixon[server]'
```

---

## Quick start

### Python

```python
from aixon import Server, autodiscover

autodiscover("agents")           # register all agents
server = Server()                # default: [OpenAIAdapter()]
server.serve(host="0.0.0.0", port=8000)
```

### CLI

```bash
aixon serve --host 0.0.0.0 --port 8000 --package agents
```

### `main.py` scaffold (production)

```python
from aixon import Server, autodiscover

autodiscover("agents")

server = Server()
app = server.app  # ASGI app — for production: uvicorn main:app --workers 4

if __name__ == "__main__":
    server.serve(host="0.0.0.0", port=8000)
```

---

## `Server` class

```python
class Server:
    @classmethod
    def get_instance(cls) -> "Server": ...

    def __init__(self, adapters: list[ProtocolAdapter] | None = None):
        """Create a server. Defaults to [OpenAIAdapter()] if adapters is None."""

    @property
    def app(self):
        """The underlying FastAPI ASGI application."""

    def serve(self, host: str = "0.0.0.0", port: int = 8000):
        """Start uvicorn. Blocks until interrupted."""
```

`Server` is a singleton — `Server.get_instance()` returns the existing instance
or creates one. Multiple `Server()` calls in the same process share state.
Adapters are fixed on first construction; later `Server(adapters=[...])` calls
silently reuse the first instance's adapters.

**Built-in routes (always public, no auth required):**

| Route | Method | Description |
|---|---|---|
| `/health` | GET | Returns `{"status": "healthy", "server": "aixon", "timestamp": "..."}`. Liveness check. |

GET routes registered by each adapter (model-list routes) are also always public.

---

## `ProtocolAdapter`

```python
from abc import ABC, abstractmethod
from aixon.server.protocol import ParsedRequest, ProtocolAdapter

class ProtocolAdapter(ABC):
    name: str     # e.g. "openai", "anthropic"

    @abstractmethod
    def parse_request(self, body: dict, *, path: str) -> ParsedRequest: ...

    @abstractmethod
    def format_response(self, *, model: str, message: Message, usage: dict) -> dict: ...

    @abstractmethod
    def format_stream_chunk(self, *, model: str, chunk: Chunk) -> str:
        """Return one SSE 'data: {...}\\n\\n' line, or '' to skip."""

    @abstractmethod
    def format_stream_done(self, *, model: str) -> str: ...

    @abstractmethod
    def format_models(self, agents: list) -> dict: ...

    @abstractmethod
    def routes(self) -> list[tuple[str, str]]:
        """[(http_method, path)] served by this adapter."""
```

```python
@dataclass
class ParsedRequest:
    model:    str                  # the requested agent name / alias
    messages: list[Message]
    params:   dict                 # temperature, max_tokens, etc. (transport fields already stripped)
    stream:   bool
    tools:    list[dict] | None = None
```

`tools` carries the tool definitions the **client** declared on the request,
**always normalized to the OpenAI wire shape**
(`{"type": "function", "function": {"name", "description", "parameters"}}`)
regardless of which adapter parsed the request — `AnthropicAdapter` converts
its `{name, description, input_schema}` defs before returning `ParsedRequest`,
so `aixon.runtime.current_client_tools()` is dialect-neutral for every
consumer, or `None` if the client sent none.

---

## `OpenAIAdapter`

Full OpenAI-compatible wire format. Served routes:

| Route | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Non-streaming and streaming (SSE) completions. |
| `/chat/completions` | POST | Same as above, for clients that omit the `/v1` prefix. |
| `/v1/models` | GET | List registered agents in OpenAI `model` object format. |
| `/models` | GET | Same as above, without the `/v1` prefix. |

> **`usage`.** On a non-streaming response, the **provider's real usage wins**:
> when the LLM call reports `usage_metadata` (surfaced on the neutral
> `Message.usage`, summed over every model turn for a `ToolAgent` run — a
> multi-step tool loop bills every turn, not just the final answer), the
> server reports it as-is. Only when the provider reported none does the
> server fall back to a `tiktoken` estimate (`pip install aixon[tiktoken]`),
> counted in the Server layer — the neutral `Message`/`Chunk` types still carry
> no token counts of their own. Without `tiktoken` AND no provider usage,
> the response carries an empty usage object (`"usage": {}`) — never an
> error. Streaming keeps the estimate-only path
> (provider usage isn't accumulated mid-stream); add
> `"stream_options": {"include_usage": true}` to get a final usage chunk before
> `[DONE]`.

> **`thought_stream_mode`** (request body, OpenAI adapter) controls how an
> agent's reasoning reaches the wire on a stream:
> - `custom` (default) — reasoning on a separate `delta.reasoning` field;
>   `delta.content` is never mutated, so programmatic consumers parse it safely.
> - `content` — reasoning wrapped in a `<think>...</think>` block inside
>   `delta.content` (opt-in for chat UIs that render think-blocks).
> - `hidden` — reasoning dropped; content only.
>
> The server-side default is configurable per deploy:
> `OpenAIAdapter(default_thought_mode="content")` for chat UIs that render
> think-blocks. A per-request `thought_stream_mode` always wins.

> **Generation params.** Per-request `temperature`, `top_p`, `max_tokens`,
> `presence_penalty`, `frequency_penalty`, `stop`, and `reasoning_effort` are
> forwarded to the model automatically (allow-listed), overriding the agent's
> class-level `LLM(...)` defaults for that request. `reasoning_effort`
> specifically overrides the class-level `reasoning=` knob (see
> [agents.md](agents.md#reasoning-extended-thinking--reasoning-effort)) for
> that one build, translated as `{"effort": reasoning_effort}`.

> **Client tools.** Agentic clients (editors, IDEs) may send OpenAI `tools` on
> the request; the adapter extracts them into `ParsedRequest.tools` and the
> Server publishes them per request via `aixon.runtime.client_tools` — agents
> that support client-executed tools read them with
> `aixon.runtime.current_client_tools()` and answer with `Message.tool_calls`
> (or `Chunk.tool_calls` on a stream). The adapter emits OpenAI-shaped
> `tool_calls` with `finish_reason: "tool_calls"`, and parses the follow-up
> history (`assistant.tool_calls` + `role: "tool"` results) back into neutral
> form. Agents that ignore client tools keep working unchanged. Runnable demo:
> `examples/client_tools/`.
>
> The client-tool **round-trip** also works over the Anthropic dialect
> (`/v1/messages`), using the SAME `Message.tool_calls` / `Chunk.tool_calls`
> neutral shapes. Outbound, `Message.tool_calls` becomes `tool_use` content
> blocks (`stop_reason: "tool_use"`); on a stream, each call opens a
> `content_block_start` (`type: "tool_use"`), one `input_json_delta` carrying
> the full arguments, and a `content_block_stop` — the final `message_delta`
> uses `stop_reason: "tool_use"` when the stream emitted one. Inbound, an
> `assistant` history message with `tool_use` blocks parses back into
> `Message.tool_calls`, and a `user` message with `tool_result` blocks parses
> into one `role: "tool"` neutral message per block (`tool_call_id` from
> `tool_use_id`).
>
> **Limitation.** Anthropic extended thinking (`reasoning=` on a
> `claude-*` `LLM`) does not round-trip through a CLIENT-executed tool loop:
> the neutral boundary drops `thinking` content on the way back into
> LangChain messages, so a follow-up request carrying the client's tool
> result — but not the matching signed thinking block — gets rejected by
> Anthropic's API. This only affects tool calls the CLIENT executes across
> separate HTTP requests; a normal `ToolAgent` (aixon runs the tool loop
> in-process, within one request) is unaffected. See
> [agents.md](agents.md#reasoning-extended-thinking--reasoning-effort) for
> detail.

> **Non-blocking.** The server `await`s `agent.ainvoke` / `agent.astream`, so an
> in-flight LLM call does not block the event loop — concurrent requests overlap
> instead of serializing. Native-async agents run truly async; a purely sync
> agent is bridged to a worker thread. You write agents the same way.

Any OpenAI-compatible client works out of the box:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")

# Non-streaming
response = client.chat.completions.create(
    model="planneragent",
    messages=[{"role": "user", "content": "Plan a sprint."}],
)
print(response.choices[0].message.content)

# Streaming
with client.chat.completions.stream(
    model="researchagent",
    messages=[{"role": "user", "content": "Research LangGraph."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

**Reasoning field:** `Chunk.reasoning` (emitted by `ToolAgent`/nested agents via
the `ReasoningChannel`, and — when the agent's `LLM(model, reasoning=...)` knob
is on — the model's own native thinking, see
[agents.md](agents.md#reasoning-extended-thinking--reasoning-effort)) is
surfaced in the streaming response via a configurable mode: hidden (default), a
vendor extension field, or inline `<think>…</think>` tags.

---

## `AnthropicAdapter`

Full production dialect serving Anthropic's Messages API from the SAME neutral
`Message`/`Chunk` types the OpenAI adapter uses — proof that the neutral
boundary is genuinely dialect-neutral, not OpenAI types in disguise.

Served routes:

| Route | Method | Description |
|---|---|---|
| `/v1/messages` | POST | Non-streaming and streaming Anthropic messages. |
| `/v1/models` | GET | List registered agents in Anthropic `model` object format. |

Wire-format differences handled by the adapter (agents see none of these):

- System prompt is outside the `messages` array (a separate top-level field),
  hoisted into a leading neutral `system` `Message`.
- Response body uses typed content blocks (`[{"type": "text", "text": "..."}]`).
- Stop reason field is `stop_reason` instead of `finish_reason`.
- Client-declared `tools` ({name, description, input_schema}) are normalized to
  the OpenAI wire shape before reaching `ParsedRequest.tools` (see above).

**Streaming envelope.** `AnthropicAdapter.open_stream` returns a stateful
`_AnthropicStreamSession` (not the stateless default) that emits the real
Anthropic SSE sequence a production SDK expects, tracking per-block indices
across the whole request:

```
message_start
content_block_start (index 0, "thinking" or "text")
content_block_delta  (thinking_delta | text_delta) ...
content_block_stop   (index 0)
content_block_start (index 1, the other kind — only if the run interleaves)
...
message_delta  (stop_reason, usage.output_tokens)
message_stop
```

Blocks are a true **sequence**, not a fixed thinking-then-text pair: whichever
modality (reasoning vs. content) is *not* the currently open block closes the
open one and opens a fresh block at the next index — indices are never
reused. A mid-stream failure closes whatever block is currently open
(`content_block_stop`) *before* the `error` event, so the client's SDK (which
tracks block state) never sees a delta/stop against a block it doesn't know
is still open; `message_stop` still follows to close the request.

### Serving more than one dialect

Both built-in adapters declare `GET /v1/models`, so mounting them at their
canonical paths would collide. Give one a `mount_prefix` — the Server prepends
it to every route from that adapter's `routes()`:

```python
from aixon import Server
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.adapters.anthropic import AnthropicAdapter

server = Server(adapters=[
    OpenAIAdapter(),                          # /v1/chat/completions, /v1/models
    AnthropicAdapter(mount_prefix="/anthropic"),  # /anthropic/v1/messages, /anthropic/v1/models
])
server.serve()
```

`mount_prefix` defaults to `""` (canonical paths), so a single adapter is
unaffected. If two adapters end up claiming the same `(method, path)`, the
Server raises a clear `AixonError` at app-build time instead of silently
shadowing one route — set a prefix to disambiguate.

---

## Auth

Set `AUTH_API_KEY` to enable Bearer token authentication. Unset = no auth.

```bash
AUTH_API_KEY=my-secret-key aixon serve
```

- `/health` and model-list (GET) routes are always public.
- All other routes require `Authorization: Bearer my-secret-key`.
- Multiple keys: comma-separated (`AUTH_API_KEY=key1,key2`).

```python
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="my-secret-key",
)
```

---

## Errors

Every chat route returns a dialect-appropriate JSON error body (never a raw
traceback) with the matching HTTP status:

| Status | When | Body shape |
|---|---|---|
| `400` | Request body isn't a JSON object, or `adapter.parse_request` raises (e.g. a malformed `messages` entry). | `{"error": {"message": ..., "type": "invalid_request_error"}}` |
| `404` | `model` doesn't resolve to a registered agent (`AgentNotFoundError`). | `{"error": {"message": ..., "type": "model_not_found"}}` |
| `500` | The agent raised while running (non-streaming). The real exception is logged server-side only; the client gets a generic message. | `{"error": {"message": "The agent failed to process the request.", "type": "server_error"}}` |

**Mid-stream failures** cannot use an HTTP status — the `200`/
`text/event-stream` headers already went out. Instead, once streaming has
started, an exception from `agent.astream` is caught and turned into one
final dialect-shaped SSE error event (never propagated into Starlette, which
would otherwise abort the response mid-stream) — the full exception goes to
the server log, never to the client:

- OpenAI: `data: {"error": {"message": "The server encountered an error while generating the response.", "type": "server_error"}}\n\n`
- Anthropic: `event: error\ndata: {"type": "error", "error": {"type": "api_error", "message": "..."}}\n\n`, with any
  currently-open content block closed first (`content_block_stop`) so the
  client's SDK never sees a delta/stop against a block it doesn't know is
  still open.

The stream's terminal event (`[DONE]` / `message_stop`) is still emitted after
the error event either way.

---

## Custom adapter

Implement `ProtocolAdapter` to support any wire format:

```python
from aixon.server.protocol import ProtocolAdapter, ParsedRequest
from aixon import Message, Chunk

class MyAdapter(ProtocolAdapter):
    name = "myformat"

    def parse_request(self, body: dict, *, path: str) -> ParsedRequest:
        return ParsedRequest(
            model=body["agent"],
            messages=[Message(role="user", content=body["input"])],
            params={},
            stream=body.get("stream", False),
        )

    def format_response(self, *, model, message, usage) -> dict:
        return {"output": message.content}

    def format_stream_chunk(self, *, model, chunk) -> str:
        if chunk.content:
            return f"data: {chunk.content}\n\n"
        return ""

    def format_stream_done(self, *, model) -> str:
        return "data: [DONE]\n\n"

    def format_models(self, agents) -> dict:
        return {"agents": [a.name for a in agents]}

    def routes(self) -> list[tuple[str, str]]:
        return [("POST", "/my/chat")]
```

```python
server = Server(adapters=[MyAdapter()])
server.serve()
```

---

## Request flow

```
ASGI
  -> adapter.parse_request        # wire body -> ParsedRequest (neutral)
  -> get_registry().resolve(model) # name / alias lookup
  -> agent.invoke | agent.stream  # neutral Message[] / Chunk only
  -> adapter.format_*             # neutral -> wire HTTP / SSE
```

The Server is dialect-agnostic. Every wire detail lives in the adapter.

---

## See also

- [agents.md](agents.md) — defining and registering agents
- [architecture.md](architecture.md) — overall design
