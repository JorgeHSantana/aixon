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
    model:    str            # the requested agent name / alias
    messages: list[Message]
    params:   dict           # temperature, max_tokens, stream, etc.
    stream:   bool
```

---

## `OpenAIAdapter`

Full OpenAI-compatible wire format. Served routes:

| Route | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Non-streaming and streaming (SSE) completions. |
| `/v1/models` | GET | List registered agents in OpenAI `model` object format. |

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

**Reasoning field:** `Chunk.reasoning` (emitted by `ToolAgent` and nested agents
via the `ReasoningChannel`) is surfaced in the streaming response via a
configurable mode: hidden (default), a vendor extension field, or inline
`<think>…</think>` tags.

---

## `AnthropicAdapter`

Thin proof-of-concept adapter serving Anthropic's structurally different wire
format. Demonstrates that neutral types are genuinely neutral — not OpenAI types
in disguise.

Served routes:

| Route | Method | Description |
|---|---|---|
| `/v1/messages` | POST | Non-streaming and streaming Anthropic messages. |
| `/v1/models` | GET | List registered agents in Anthropic `model` object format. |

Wire-format differences handled by the adapter (agents see none of these):

- System prompt is outside the `messages` array (a separate top-level field).
- Response body uses typed content blocks (`[{"type": "text", "text": "..."}]`).
- Stop reason field is `stop_reason` instead of `finish_reason`.
- Streaming uses named event types (`message_start`, `content_block_delta`, `message_stop`).

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
