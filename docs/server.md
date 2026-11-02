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

    def __init__(
        self,
        adapters: list[ProtocolAdapter] | None = None,
        *,
        request_hook: RequestHookLike | None = None,
        budget_exceeded_message: str = DEFAULT_BUDGET_EXCEEDED_MESSAGE,
        rate_limited_message: str = DEFAULT_RATE_LIMITED_MESSAGE,
    ):
        """Create a server. Defaults to [OpenAIAdapter()] if adapters is None.
        The two messages are what the user reads when the gateway refuses the call.
        request_hook decides, per request, the imposed model, the identity and
        the refusal (see "Per-request context")."""

    @property
    def app(self):
        """The underlying FastAPI ASGI application."""

    def serve(self, host: str = "0.0.0.0", port: int = 8000):
        """Start uvicorn. Blocks until interrupted."""
```

`budget_exceeded_message` and `rate_limited_message` are what the user is told
when the `gateway` provider's call is refused: its budget is spent, or it is
rate limiting the caller (see [Errors](#errors)). Their defaults are:

- `budget_exceeded_message`: `Your usage budget has been exhausted. Please try again after it resets or contact your administrator.`
- `rate_limited_message`: `Too many requests to the model right now. Please wait a moment and try again.`

`Server` is a singleton — `Server.get_instance()` returns the existing instance
or creates one. Multiple `Server()` calls in the same process share state, and
the **first construction fixes** the adapters, the two messages and the
`request_hook`. A later
construction:

- with no arguments, or with arguments equal to what is stored (an equivalent
  adapter set — the same adapter classes and mount prefixes, in the same order —
  or the stored message), returns the same instance, unchanged;
- with a **different** adapter set or a **different** message **raises
  `AixonError`**, instead of silently ignoring the argument. Call
  `Server._reset()` first to build the singleton again with the new
  configuration;
- passing a message's default text counts as having no opinion: a bare
  `Server()` keeps working after the application configured the messages, and
  keeps the ones stored. The same holds for `request_hook=None`: it means "no
  opinion" (keep the stored hook, or use the default one), never "no hook", so
  `aixon serve` and a `main.py` that builds the `Server` live together.

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
> think-blocks. Agents may pin their own mode with the class attribute
> `thought_mode = "hidden"` — made for programmatic clients (protocol parsers,
> editor plugins) that must never see `<think>` in content, immune to a
> chat-UI-oriented server default. Precedence: request `thought_stream_mode`
> > agent `thought_mode` > adapter `default_thought_mode`. Non-stream
> responses honor only EXPLICIT modes (request/agent); the adapter default
> stays streaming-only, preserving the historical non-stream shape.

> **Debug tap (`AIXON_DEBUG_REQUESTS`).** Opt-in diagnostic recorder for
> client integrations: when set, each chat POST is appended as one JSONL
> record — verbatim body, resolved agent, and the response (non-stream
> payload, or the raw SSE lines of a stream) — under
> `AIXON_DEBUG_REQUESTS_DIR` (default `./aixon-debug/`). Headers are never
> recorded, so `Authorization` can't leak. Values `1`/`true`/`yes` record
> every agent; anything else is a comma-separated **allowlist of agent
> names** (`AIXON_DEBUG_REQUESTS="OnlyOffice,Redator"`), keeping agents with
> sensitive conversations off the disk while one integration is under
> diagnosis. The env is checked per request (toggle without rebuild); tap
> failures never break the request. It is a temporary diagnostic tool: run it
> locally or in short windows, then unset the env and DELETE the directory —
> never leave it on in production.

> **Generation params.** Per-request `temperature`, `top_p`, `max_tokens`,
> `presence_penalty`, `frequency_penalty`, `stop`, and `reasoning_effort` are
> forwarded to the model automatically (allow-listed), overriding the agent's
> class-level `LLM(...)` defaults for that request. `reasoning_effort`
> specifically overrides the class-level `reasoning=` knob (see
> [agents.md](agents.md#reasoning-extended-thinking--reasoning-effort)) for
> that one build, translated as `{"effort": reasoning_effort}`.

> **Client tools.** Agentic clients (editors, IDEs) may send OpenAI `tools`
> (and optionally `tool_choice`) on the request; the adapter extracts them
> into `ParsedRequest.tools`/`ParsedRequest.tool_choice` and the Server
> publishes them per request via `aixon.runtime.client_tools_scope` /
> `aixon.runtime.tool_choice_scope`. `tool_choice` **requires** `tools` on the
> same request — sending it alone is rejected with `400` (a client asking the
> model to pick a tool that doesn't exist is a request error, not a silent
> no-op). Two opt-in ways for an agent to participate:
>
> - **Raw passthrough** — `LLMAgent(client_tools="passthrough")` (#18a) or any agent
>   that reads `aixon.runtime.current_client_tools()` /
>   `current_tool_choice()` directly and decides for itself when to answer
>   with `Message.tool_calls` (or `Chunk.tool_calls` on a stream).
> - **First-class merge** — `ToolAgent(client_tools="merge" | "replace")`
>   (#18c): the client's tool defs are proxied into the SAME tool-calling loop
>   as the agent's own (internal) tools. The model can call either kind in the
>   same run:
>   - a call to an **internal** tool executes server-side, same as always —
>     the loop continues normally;
>   - a call to a **client** tool immediately ends the turn: the run returns
>     `Message(role="assistant", content="", tool_calls=[...])` (the wire
>     adapter emits `finish_reason: "tool_calls"` from that alone) instead of
>     looping further, because the client — not aixon — must execute it.
>
>   | `client_tools` | Effect |
>   |---|---|
>   | `"ignore"` (default) | Client tool defs are never read. Zero behavior change. |
>   | `"merge"` | Client defs are ADDED to the agent's own tools for this request. |
>   | `"replace"` | ONLY the client's defs are exposed for this request (the agent's own `tools` are not bound). |
>
>   A client def can collide by name with one of the agent's own tools;
>   `client_tools_conflict` picks the resolution at request time:
>
>   | `client_tools_conflict` | Effect |
>   |---|---|
>   | `"error"` (default) | `AixonError` at graph build time, naming the colliding tool(s) — an explicit failure instead of guessing which one the model meant. |
>   | `"internal"` | The CLIENT's def is dropped; the agent's own tool wins and executes server-side as usual. |
>   | `"client"` | The agent's INTERNAL tool is dropped for this request; only the client's def (as a proxy) is exposed under that name. |
>
>   Override `client_tools_filter(self, defs) -> defs` to curate which client
>   defs are even considered (e.g. drop anything without an expected prefix)
>   before the merge/replace/conflict logic runs; the default keeps all of them.
>
>   **Resuming after the client executes its tool.** The client runs the call
>   locally and POSTs again with the result appended to history — same shape
>   as the raw-passthrough path:
>
>   ```
>   request 1  → user message
>            ←  finish_reason: "tool_calls" (client tool)
>   [client executes the tool itself]
>   request 2  → ...history..., assistant(tool_calls=[...]), role="tool" (the result)
>            ←  finish_reason: "stop" (or another tool_calls turn)
>   ```
>
>   The follow-up request's history round-trips through the SAME neutral
>   `Message`/LangChain conversion (`to_langchain`/`from_langchain`) as any
>   other tool result — no special-casing needed on the client side.
>
>   **Mixed turns (internal + client call in the SAME model turn).** The
>   FIRST turn of the run that calls a client tool is what surfaces: its
>   client call(s) become `Message.tool_calls`, and the internal call(s) of
>   that same turn have already executed server-side (LangGraph's `ToolNode`
>   runs the whole turn's calls). Because LangChain's `return_direct` only
>   ends the graph when ALL of a turn's calls are return-direct, a mixed
>   turn keeps the loop going — the model sees the proxy's placeholder
>   "result" and may generate further turns; everything it produced AFTER
>   that first client call is **discarded** (those model turns are paid for
>   but never reach the client: a real result for the client's tool did not
>   exist yet, so any answer built on the placeholder would be fabricated).
>   On the next request (with the client's tool result in the history) the
>   model re-plans from real data; an internal call it re-emits simply
>   re-executes. To reduce the wasted post-call turns, prompt the agent to
>   put document/client-side actions in their own turn, separate from
>   internal lookups.
>
>   **SIDE EFFECTS (beyond wasted cost/fabricated text).** The turns
>   generated AFTER that first client call are discarded as a *response*,
>   but any INTERNAL tool the model calls in those discarded turns actually
>   EXECUTES — LangGraph's `ToolNode` runs the call before the turn's text
>   is thrown away. An internal tool with a side effect (write to a file,
>   trigger an export, send a notification) based on the proxy's placeholder
>   "result" fires anyway, and it cannot be undone afterward. Mitigation: on
>   agents with `client_tools` enabled, avoid mixing internal tools that
>   have side effects into the same `tools` list, or prompt the model to
>   keep client-side (document) actions in their own turn, BEFORE any
>   internal tool that writes/exports/notifies.
>
>   Runnable demos: `examples/client_tools/main.py` (raw passthrough, #18a)
>   and `examples/client_tools/merge_demo.py` (first-class merge + resume,
>   #18c).
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
| `429` | The gateway refused the call (non-streaming): its budget is spent (an HTTP 422 mentioning "budget") or it is rate limiting the caller (an HTTP 429). The `message` is `budget_exceeded_message` or `rate_limited_message`. **Only an error from a request to `AIXON_GATEWAY_BASE_URL` counts**: a 429 from any other provider is not the gateway's and stays a `500`. The response carries `x-should-retry: false`, so a client on the `openai` SDK does not retry it (each retry would run the whole agent tree again against the same refusal). | `{"error": {"message": ..., "type": "insufficient_quota" \| "rate_limit_exceeded", "code": "budget_exceeded" \| "rate_limited"}}` |
| `500` | The agent raised while running (non-streaming), anything but a gateway refusal. The real exception is logged server-side only; the client gets a generic message. | `{"error": {"message": "The agent failed to process the request.", "type": "server_error"}}` |

The gateway error is found in the exception that reached the Server or in its
`__cause__`/`__context__` chain, so a refusal wrapped by another exception on
the way still gets its `429`. It is logged as a `warning`, not an `error`, and
the log line carries only the verdict (`GatewayBudgetExceeded` or
`GatewayRateLimited`), the agent, the adapter and the HTTP status of the answer:
never the gateway's own text, which (in LiteLLM, for one) names the end user and
the spend.

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

**The exception to the error event: a gateway refusal.** When the exception is a
refusal by the gateway (budget spent or rate limited, classified as above), the
stream gets **no** error event. The message goes out as the last content of a
stream that ends as usual:

- OpenAI: a `delta.content` carrying `budget_exceeded_message` or
  `rate_limited_message` (it closes a `<think>` block that the failed run left
  open), then the closing chunk with `finish_reason: "stop"`, then `[DONE]`.
- Anthropic: a `text_delta` block with the message, then `message_delta` and
  `message_stop`, with no `event: error`.

A chat UI shows it as the assistant's reply instead of a generic error. The
refusal is logged the same way as without streaming (a `warning` with the
verdict only, never the gateway's text). Only a gateway refusal is treated this
way: any other exception, a 429 from another
provider included, keeps the generic error event above.

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
  -> request_hook(ctx)            # once: RequestDecision | None, or RequestRejected
  -> request scopes               # in order: observe_request (trace identity) -> generation_params -> user_email_scope -> imposed_model_scope (only with impose) -> client_tools_scope -> tool_choice_scope -> tool_call_cache
  -> agent.invoke | agent.stream  # neutral Message[] / Chunk only
  -> gateway refusal?             # budget / rate limit -> the quota message (see Errors)
  -> adapter.format_*             # neutral -> wire HTTP / SSE
```

The Server is dialect-agnostic. Every wire detail lives in the adapter.

**The user's e-mail.** Right before the agent runs, the Server publishes the
`user_email` of the request hook's decision. With the default hook
(`open_webui_request_hook`, see [Per-request context](#per-request-context-request_hook))
that is the `x-openwebui-user-email` header (what Open WebUI sends with
`ENABLE_FORWARD_USER_INFO_HEADERS=true`); the Server publishes it for the whole call,
streaming or not, with `aixon.runtime.user_email_scope(email)`; an absent or
blank header publishes no e-mail. The chain is:

```
RequestDecision.user_email (default hook: x-openwebui-user-email header)
  -> user_email_scope(email) / current_user_email()
  -> the gateway header (AIXON_GATEWAY_USER_HEADER, default X-OpenWebUI-User-Email)
```

Only the `gateway` provider reads `current_user_email()`: it stamps the e-mail on
every HTTP request it sends, so the gateway meters spend and limits per person
(see [agents.md](agents.md#llm--declaring-a-language-model)). No other provider
ever receives it. Only the **e-mail** is forwarded, never the Open WebUI
user-id: a Langfuse trace falls back to the user-id when there is no e-mail (see
[tracing.md](tracing.md)), but the gateway never sees it. A request without the
header runs with no e-mail, and the gateway then gets no header at all; refusing
such a request is the application's job: a `request_hook` that raises
`RequestRejected` when there is no e-mail.

What this identity is worth:

- **ASCII only.** The e-mail travels as an HTTP header, so it must be ASCII; an
  internationalized e-mail address is not supported in this version.
- **One header name.** The default hook reads only `x-openwebui-user-email`. If Open
  WebUI is configured to rename it (`FORWARD_USER_INFO_HEADER_USER_EMAIL`) or to
  send a signed JWT instead (`FORWARD_USER_INFO_HEADER_JWT_SECRET`), the Server
  publishes no e-mail (`None`), and it is up to the application to refuse the
  request.
- **Trust boundary.** The header is taken as is from whoever passes
  `AUTH_API_KEY` (or from anyone, without it): any caller holding a key can
  claim any e-mail. Per-person quotas at the gateway are only as good as that
  boundary, so keep the Server behind Open WebUI and the key secret.
- **Context, not threads.** The e-mail is a `ContextVar`: code run on a thread
  that does not copy the context (`threading.Thread`,
  `ThreadPoolExecutor.submit`, `loop.run_in_executor`) sees no e-mail, and its
  gateway calls go out without the header. A `user_email_scope` held across a
  `yield` is also seen by the generator's consumer: open it around the call.

---

## Per-request context (`request_hook`)

The application decides, once per request, **which model the agent's whole tree
runs on**, **who the user is** and **whether the request gets in** — from the
headers, without touching the agent classes. Pass a function, or an instance of
a `RequestHook` subclass (see [Hook as a class](#hook-as-a-class-0130)), to
`Server(request_hook=...)`:

```python
# aixon/server/context.py
@dataclass(frozen=True)
class RequestContext:
    """O que o hook enxerga de uma request: modelo pedido, agente resolvido,
    cabeçalhos (chaves minúsculas, somente leitura), parâmetros, adapter e
    se a resposta é em stream."""

    model: str
    agent_name: str
    headers: Mapping[str, str]
    params: Mapping[str, Any]
    adapter: str
    stream: bool


@dataclass(frozen=True)
class RequestDecision:
    """A decisão do hook. Todo campo ``None`` significa "não mexer"."""

    impose: tuple[str, str] | None = None
    user_email: str | None = None
    trace_user_id: str | None = None
    trace_session_id: str | None = None


RequestHookLike = Callable[[RequestContext], "RequestDecision | None"]

# aixon/exceptions.py
class RequestRejected(AixonError):
    """O ``request_hook`` recusou a request; o Server responde com ``status``
    e ``code`` no envelope de erro do protocolo."""

    def __init__(
        self, message: str, *, status: int = 403, code: str = "request_rejected"
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
```

```python
from aixon import Server
from aixon.exceptions import RequestRejected
from aixon.server.context import RequestContext, RequestDecision

TIERS = {"fast": ("gpt-5.4-mini", "openai"), "deep": ("gpt-5.4", "openai")}

def tier_hook(ctx: RequestContext) -> RequestDecision:
    user = ctx.headers.get("x-user")
    if not user:
        raise RequestRejected("x-user header is required", status=401, code="no_user")
    return RequestDecision(impose=TIERS.get(ctx.headers.get("x-tier", "")),
                           user_email=user)

server = Server(request_hook=tier_hook)
```

**Lifecycle.** `parse → resolve → hook → scopes → agent`. The adapter parses the
body, the registry resolves the agent, and only then the Server calls the hook,
**once**, before any scope opens. `ctx.headers` has lower-case keys (on a duplicate
header the first value wins, like `starlette.Headers.get`) and is read only; `ctx.params` is a read-only copy of the generation params the adapter
extracted. The Server then opens the scopes of the decision around the agent
call, on the streaming and the non-streaming path alike:

| Field | Scope the Server opens |
|---|---|
| `impose=(model, provider)` | `imposed_model_scope(model, provider, by="server")` — `current_imposed_by()` returns `"server"` |
| `user_email` | `user_email_scope(email)` (the gateway header, see below) |
| `trace_user_id` / `trace_session_id` | the Langfuse trace identity; `trace_user_id=None` with a `user_email` falls back to the e-mail |

The scopes open in this order: `observe_request` (trace identity) →
`generation_params` → `user_email_scope` → `imposed_model_scope` (only with
`impose`) → `client_tools_scope` → `tool_choice_scope` → `tool_call_cache`.

`None` from the hook is the same as `RequestDecision()`: nothing imposed, no
identity. Since the outermost imposed scope wins, an agent with
`impose_model = True` under a server-imposed model does not open its own (see
[agents.md](agents.md#modelo-imposto-impose_model)).

**Refusal.** A hook that raises `RequestRejected` stops the request before any
LLM call. Without streaming, the answer is HTTP `status` (default `403`) with
`{"error": {"message": ..., "type": "request_rejected", "code": ...}}`; on a
stream, the message is delivered as the last chat message, the same path the
quota message uses. Any other exception raised by the hook becomes a `500`
`server_error` (the log names only the exception type). An `impose` naming a
provider that is not registered also ends in `500`.

**The hook is synchronous.** A network call inside it blocks the request; that
is the hook author's call.

**The default hook, deprecated.** Without `request_hook`, the Server uses
`aixon.server.compat.open_webui_request_hook` — what aixon did up to 0.1.28: it
reads `x-openwebui-user-email` as `user_email` and, for the trace,
`x-openwebui-user-email` (falling back to `x-openwebui-user-id`) and
`x-openwebui-chat-id`. It is **deprecated** and becomes opt-in in 0.2; write
your own. A hook of your own replaces it entirely: there is no fallback to the
Open WebUI headers. `aixon.observability.request_identity` is deprecated with
it.

**Warning: `LLM.chat_model` under a server-imposed model.** The server imposes
on agents that never declared `impose_model`. Under the imposed scope,
`llm.chat_model` (the model the `LLM` was *configured* with) raises
`ImposedModelViolation`; code that reads it directly must go through
`llm.request_chat_model()`, `complete`/`stream` or declare
`LLM(..., pinned=True)`. The non-portable kwargs of each `LLM` (`reasoning`,
`timeout`, `max_retries`...) are dropped as under any imposition, and
`LLM.request_chat_model` logs a WARNING once per (LLM, imposed model) naming
them.

### Hook as a class (0.1.30)

`RequestHookLike` is still what the Server accepts: any callable. `RequestHook`
is a base class that organises one into phases. Its `__call__` runs `check` →
`identity` → `imposition`, in that order, and assembles the `RequestDecision`.
Any phase may raise `RequestRejected`; a refusal in `check` skips the others.
Override only the phases you need. All of these are exported from
`aixon.server` (`OpenWebUIRequestHook` from `aixon.server.compat`).

```python
# aixon/server/context.py
@dataclass(frozen=True)
class Identity:
    user_email: str | None = None
    trace_user_id: str | None = None
    trace_session_id: str | None = None


class RequestHook:
    def check(self, ctx: RequestContext) -> None: ...            # base: accepts
    def identity(self, ctx: RequestContext) -> Identity: ...     # base: Identity()
    def imposition(self, ctx: RequestContext) -> tuple[str, str] | None: ...  # base: None
    def __call__(self, ctx: RequestContext) -> RequestDecision: ...


class HeaderIdentityHook(RequestHook):
    def __init__(
        self,
        *,
        email_header: str,
        user_id_header: str | None = None,
        session_header: str | None = None,
    ) -> None: ...


class ChainRequestHook(RequestHook):
    def __init__(self, hooks: Sequence[RequestHookLike]) -> None: ...

# aixon/server/compat.py
class OpenWebUIRequestHook(HeaderIdentityHook):
    def __init__(self) -> None: ...   # x-openwebui-user-email / -user-id / -chat-id

open_webui_request_hook = OpenWebUIRequestHook()
```

- `HeaderIdentityHook`: `user_email` is the raw value of `email_header`;
  `trace_user_id` is the e-mail or, when empty/absent, `user_id_header` (if
  configured); `trace_session_id` comes from `session_header` (if configured).
  Header names are lower-cased. An empty `email_header` raises `AixonError`.
- `OpenWebUIRequestHook`: the default hook as a class. Subclass it to add
  `check` or `imposition` while keeping the Open WebUI identity.
- `ChainRequestHook`: runs the hooks in order; a `RequestRejected` from any of
  them stops the chain. Decisions merge field by field: the first non-`None`
  value wins; a hook returning `None` counts as `RequestDecision()`.

The tier hook above, as a subclass:

```python
from aixon.exceptions import RequestRejected
from aixon.server import HeaderIdentityHook, RequestContext
from aixon.server.server import Server

TIERS = {"fast": ("gpt-5.4-mini", "openai"), "deep": ("gpt-5.4", "openai")}

class TierHook(HeaderIdentityHook):
    def __init__(self) -> None:
        super().__init__(email_header="x-user")

    def check(self, ctx: RequestContext) -> None:
        if not ctx.headers.get("x-user", "").strip():
            raise RequestRejected("x-user header is required", status=401, code="no_user")

    def imposition(self, ctx: RequestContext) -> tuple[str, str] | None:
        return TIERS.get(ctx.headers.get("x-tier", ""))

server = Server(request_hook=TierHook())
```

Example: [examples/request_hook](../examples/request_hook) — two agents, an
`x-tier` header choosing the model and an `x-user` header that is required.

---

## See also

- [agents.md](agents.md) — defining and registering agents
- [architecture.md](architecture.md) — overall design
