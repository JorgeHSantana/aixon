# aixon Server + ProtocolAdapter + Adapters + Auth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build aixon's HTTP boundary on **FastAPI/ASGI**: the protocol-decoupling seam (`ParsedRequest` + the `ProtocolAdapter` ABC), a full **`OpenAIAdapter`** (`/v1/chat/completions` stream + non-stream, `/v1/models`), a thin proof-of-decoupling **`AnthropicAdapter`** (`/v1/messages`, system outside the array, typed `content[]` blocks, `stop_reason`, named stream events), and the **`Server`** singleton that mounts adapters, resolves agents via the registry, and enforces Bearer auth via the `AUTH_API_KEY` env var. The defining guarantee: **no vendor/wire type ever crosses into `Agent.invoke`/`stream`** — the agent runtime only ever sees neutral `Message[]`.

**Architecture:** Mirrors restmcp's FastAPI app construction and env-based Bearer auth, and olympus's OpenAI chat-completions wire shapes (translated from Flask to FastAPI). The request flow is: `ASGI → adapter.parse_request(body, path) → ParsedRequest → get_registry().resolve(model) → agent.invoke|stream` (neutral types only) `→ adapter.format_response|format_stream_chunk|format_stream_done → HTTP/SSE`. Each `ProtocolAdapter` owns its wire dialect and its route list; the `Server` is dialect-agnostic. New wire styles are new `*Adapter` subclasses — never edits to `Server` or the agent runtime. This is what the `AnthropicAdapter` proves: a structurally different envelope (system hoisted out of `messages`, content blocks, named SSE events) is served by the same neutral `Message[]`/`Chunk` the OpenAI dialect uses.

**Tech Stack:** Python 3.11+, **FastAPI/ASGI (NOT Flask)**, `uvicorn[standard]` for `serve()`, `httpx` for the hermetic `fastapi.testclient.TestClient`. `pydantic` is already in the `server` extra. SSE via Starlette `StreamingResponse` (`media_type="text/event-stream"`) — no extra dependency. All tests run against the in-process ASGI app through `TestClient`; **no network, no API keys, no provider SDK** — hermetic fake agents are defined in `tests/_server_fakes.py`.

## Global Constraints

- **Python 3.11+.** Use `from __future__ import annotations` wherever `X | Y` unions appear in annotations.
- **FastAPI/ASGI, NOT Flask.** olympus is Flask; aixon deliberately uses FastAPI + uvicorn, mirroring restmcp. Translate olympus's *wire shapes*, never its framework.
- **Neutral boundary is absolute.** No vendor/wire type (OpenAI dict, Anthropic block, `ParsedRequest`, raw request body) may cross into `Agent.invoke`/`Agent.stream`. Adapters translate wire → neutral `Message[]` on the way in and neutral `Message`/`Chunk` → wire on the way out. `aixon/server/protocol.py` re-exports the neutral types so adapters import them from one place. The agent runtime (`aixon/agent.py`, `aixon/message.py`) must never import from `aixon.server`.
- **Bearer auth via `AUTH_API_KEY` env, disabled if unset.** When `AUTH_API_KEY` is unset/empty, every route is open. When set (comma-separated keys supported, mirroring restmcp), all routes require `Authorization: Bearer <key>` **except** `/health` and every adapter's model-list route (e.g. `/v1/models`), which stay public. Use a constant-time compare (`hmac.compare_digest`).
- **Hermetic tests via `fastapi.testclient.TestClient`** + fake-LLM agents (defined in `tests/_server_fakes.py`, NOT relying on a real provider SDK or network).
- **`server` extra owns the deps.** Add `fastapi`, `uvicorn[standard]`, `httpx` to the `server` optional-dependency group (`httpx` powers `TestClient`); `pydantic` is already there. Keep `all` in sync. The server modules import FastAPI/Starlette **lazily** inside methods where practical so that importing `aixon` (or `aixon.server.protocol`) never hard-requires FastAPI — matching restmcp's pattern (`import uvicorn` inside `start`).
- **Registry is the single source of agent routing.** Resolve every request's `model` through `get_registry().resolve(model)` (name → alias → single-agent default → `AgentNotFoundError`). The server never holds its own agent table.
- **Logging:** log the resolved agent name and the matched route at `INFO` through `Logger("aixon.server")` (the Plan 1 logger). Do not log message contents.
- **Error tone:** state what was got and how to fix it (mirrors restmcp / Plan 1).
- **Commits:** `git add <specific files>` — never `git add -A`. End commit messages with the repo's `Co-Authored-By` trailer.
- **Run all tests** from the repo root (`/Users/jorge/Documents/Git/aixon`) with `python -m pytest`. Every pre-existing foundation test must still pass after each task.

> **Dependency note (read before Task 1).** This plan depends ONLY on the **Plan 1 / foundation** public surface that is already merged: `aixon.agent.Agent`, `aixon.message.{Message, Chunk, Role}`, `aixon.registry.{get_registry, reset_registry, Registry}`, `aixon.exceptions.{AixonError, AgentNotFoundError}`, `aixon.logging.Logger`. It does **not** depend on `LLM`/`LLMAgent`/providers (Plan 2), so it can land whether or not Plan 2 is merged. Tests therefore define their own concrete `Agent` subclasses (fake LLMs) in `tests/_server_fakes.py` rather than importing `LLMAgent` or any provider fake — this keeps Plan 5 hermetic and decoupled from Plan 2's state. (Contract §4 says "use the fake-LLM agents"; since `tests/_fakes.py` does not exist in the repo, this plan establishes its own minimal fakes as plain `Agent` subclasses, which is strictly sufficient because the server only consumes the neutral `Agent` interface.)

---

### Task 1: `server` extra deps + protocol seam (`ParsedRequest` + `ProtocolAdapter` ABC + neutral re-exports)

**Files:**
- Modify: `pyproject.toml` (add `httpx` to `server` and `all`)
- Create: `aixon/server/__init__.py`
- Create: `aixon/server/protocol.py`
- Create: `aixon/server/adapters/__init__.py` (empty package marker; adapters land in Tasks 3–4)
- Test: `tests/test_protocol.py`

**Interfaces:**
- Consumes: `aixon.message.{Message, Chunk, Role}`.
- Produces (contract §4.1, verbatim signatures):
  - `aixon.server.protocol.Message`, `Chunk`, `Role` — **re-exported** from `aixon.message` so adapters import neutral types from one place.
  - `aixon.server.protocol.ParsedRequest` — dataclass: `model: str`, `messages: list[Message]`, `params: dict`, `stream: bool`.
  - `aixon.server.protocol.ProtocolAdapter` — ABC with `name: str` class attribute and abstract methods:
    - `parse_request(self, body: dict, *, path: str) -> ParsedRequest`
    - `format_response(self, *, model: str, message: Message, usage: dict) -> dict`
    - `format_stream_chunk(self, *, model: str, chunk: Chunk) -> str` — returns one SSE `"data: {...}\n\n"` line (or `""` to skip)
    - `format_stream_done(self, *, model: str) -> str`
    - `format_models(self, agents: list) -> dict`
    - `routes(self) -> list[tuple[str, str]]` — `[(http_method, path)]` this adapter serves, e.g. `[("POST","/v1/chat/completions"), ("GET","/v1/models")]`

> The ABC must not import FastAPI. It is pure stdlib + neutral types, so `aixon.server.protocol` is importable with zero server deps installed (only the concrete adapters/server need FastAPI). This keeps the seam testable on a bare install.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_protocol.py
from __future__ import annotations

import pytest

from aixon.server.protocol import (
    Chunk,
    Message,
    ParsedRequest,
    ProtocolAdapter,
)


def test_neutral_types_are_reexported_from_message_module():
    # protocol.py must re-export the SAME objects as aixon.message, not copies.
    import aixon.message as m

    assert Message is m.Message
    assert Chunk is m.Chunk


def test_parsed_request_fields():
    pr = ParsedRequest(
        model="echo",
        messages=[Message(role="user", content="hi")],
        params={"temperature": 0.2},
        stream=True,
    )
    assert pr.model == "echo"
    assert pr.messages[0].content == "hi"
    assert pr.params["temperature"] == 0.2
    assert pr.stream is True


def test_protocol_adapter_is_abstract():
    with pytest.raises(TypeError):
        ProtocolAdapter()  # all six methods abstract -> cannot instantiate


def test_concrete_adapter_must_implement_all_methods():
    # A subclass missing any abstract method is still abstract.
    class Partial(ProtocolAdapter):
        name = "partial"

        def parse_request(self, body, *, path):
            return ParsedRequest(model="x", messages=[], params={}, stream=False)

    with pytest.raises(TypeError):
        Partial()


def test_fully_concrete_adapter_instantiates():
    class Full(ProtocolAdapter):
        name = "full"

        def parse_request(self, body, *, path):
            return ParsedRequest(model="x", messages=[], params={}, stream=False)

        def format_response(self, *, model, message, usage):
            return {}

        def format_stream_chunk(self, *, model, chunk):
            return ""

        def format_stream_done(self, *, model):
            return "data: [DONE]\n\n"

        def format_models(self, agents):
            return {"data": []}

        def routes(self):
            return [("POST", "/x")]

    adapter = Full()
    assert adapter.name == "full"
    assert adapter.routes() == [("POST", "/x")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_protocol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aixon.server'`.

- [ ] **Step 3: Update `pyproject.toml`**

Replace the `[project.optional-dependencies]` block so `server` and `all` include `httpx` (the `TestClient` transport). Leave `dev`/`cli` (and any provider extras from Plan 2, if present) untouched — match whatever is currently in the file, only adding `httpx` to `server` and `all`:

```toml
[project.optional-dependencies]
dev = ["pytest", "pytest-cov"]
server = ["fastapi>=0.100", "uvicorn[standard]>=0.20", "httpx>=0.27", "pydantic>=2.0"]
cli = ["click>=8.0"]
all = ["fastapi>=0.100", "uvicorn[standard]>=0.20", "httpx>=0.27", "pydantic>=2.0", "click>=8.0"]
```

> If Plan 2 has already been merged it will have added `openai`/`anthropic`/`google`/`providers` extras and folded them into `all`; in that case keep those entries and simply ensure `httpx>=0.27` is present in both `server` and `all`. The only required change here is adding `httpx` to those two groups.

- [ ] **Step 4: Create the server package skeleton**

```python
# aixon/server/__init__.py
"""aixon HTTP boundary — protocol-decoupled FastAPI/ASGI server.

Public surface lands incrementally across this plan's tasks. The neutral
protocol seam (ParsedRequest, ProtocolAdapter) is import-safe with no server
deps; the concrete adapters and Server require the ``server`` extra
(``pip install aixon[server]``)."""
```

```python
# aixon/server/adapters/__init__.py
"""Concrete ProtocolAdapters (OpenAI, Anthropic). Each translates a wire
dialect to and from the neutral Message/Chunk types and declares the routes it
serves. New wire styles are new modules here — never edits to the Server."""
```

```python
# aixon/server/protocol.py
"""The protocol-decoupling seam.

``ProtocolAdapter`` translates a wire format (OpenAI, Anthropic, ...) to and
from aixon's neutral types. The agent runtime speaks ONLY ``Message``/``Chunk``;
no vendor/wire detail crosses this boundary inward. Neutral types are
re-exported here so adapters import them from one place. This module is pure
stdlib + neutral types — it does NOT import FastAPI, so the seam is importable
and testable on a bare install."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# Re-export the neutral types (the SAME objects, not copies) so adapter modules
# can do `from aixon.server.protocol import Message, Chunk`.
from aixon.message import Chunk, Message, Role

__all__ = ["Message", "Chunk", "Role", "ParsedRequest", "ProtocolAdapter"]


@dataclass
class ParsedRequest:
    """A wire request reduced to neutral terms. The Server consumes only this —
    it never sees the raw vendor body.

    - ``model``: the requested agent name/alias (the wire ``model`` field).
    - ``messages``: neutral conversation handed straight to ``Agent.invoke``.
    - ``params``: passthrough knobs (temperature, max_tokens, ...) minus the
      transport-level fields the adapter already consumed (model, messages,
      stream, system).
    - ``stream``: whether the client asked for an SSE stream.
    """

    model: str
    messages: list[Message]
    params: dict
    stream: bool


class ProtocolAdapter(ABC):
    """Translates one wire dialect <-> neutral types. New wire styles = new
    subclass. NO neutral type leaks a vendor/wire detail."""

    name: str = ""  # e.g. "openai", "anthropic"

    @abstractmethod
    def parse_request(self, body: dict, *, path: str) -> ParsedRequest:
        """Reduce a raw request body to a neutral ``ParsedRequest``. ``path`` is
        the matched route, so one adapter can serve several paths."""

    @abstractmethod
    def format_response(self, *, model: str, message: Message, usage: dict) -> dict:
        """Wrap a final neutral ``Message`` in the dialect's non-stream envelope."""

    @abstractmethod
    def format_stream_chunk(self, *, model: str, chunk: Chunk) -> str:
        """Return one SSE ``'data: {...}\\n\\n'`` line for a neutral ``Chunk``
        (or ``''`` to emit nothing for this chunk)."""

    @abstractmethod
    def format_stream_done(self, *, model: str) -> str:
        """Return the terminal SSE line(s) that close the stream."""

    @abstractmethod
    def format_models(self, agents: list) -> dict:
        """Render the model-listing payload from registered agents."""

    @abstractmethod
    def routes(self) -> list[tuple[str, str]]:
        """``[(http_method, path)]`` this adapter serves, e.g.
        ``[("POST","/v1/chat/completions"), ("GET","/v1/models")]``."""
```

- [ ] **Step 5: Install the server extra**

Run: `cd /Users/jorge/Documents/Git/aixon && python -m pip install -e ".[server,dev]"`
Expected: installs `fastapi`, `uvicorn[standard]`, `httpx`, `pydantic`; `Successfully installed ... aixon`.

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_protocol.py -v`
Expected: PASS (5 tests).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml aixon/server/__init__.py aixon/server/protocol.py aixon/server/adapters/__init__.py tests/test_protocol.py
git commit -m "feat(server): protocol seam — ParsedRequest, ProtocolAdapter ABC, neutral re-exports; httpx in server extra"
```

---

### Task 2: Hermetic fake agents for server tests

**Files:**
- Create: `tests/_server_fakes.py`
- Test: `tests/test_server_fakes.py`

**Interfaces:**
- Consumes: `aixon.agent.Agent`, `aixon.message.{Message, Chunk}`.
- Produces (test-only helpers — not part of the public API):
  - `tests._server_fakes.EchoAgent` — concrete `Agent`; `invoke` returns `Message(role="assistant", content="echo:" + last_user_text)`; `stream` yields two content `Chunk`s then `Chunk(done=True)`; **records the exact `messages` list it last received** on `self.seen` so a test can assert the agent only ever saw neutral `Message[]`.
  - `tests._server_fakes.ReasoningAgent` — concrete `Agent`; `stream` yields a `Chunk(reasoning=...)` then content chunks then `Chunk(done=True)`; `invoke` returns a `Message` with `reasoning` set. Used to prove reasoning survives both dialects.
  - `tests._server_fakes.make_echo(name, *, aliases=(), hidden=False, description="")` — define-and-register a fresh `EchoAgent` subclass at call time with the given metadata, returning the registered instance. (Defining at call time means each test's autouse `reset_registry` starts clean.)

> These are plain `Agent` subclasses — no `LLM`, no provider, no network. They are the "fake-LLM agents" the contract calls for, scoped to Plan 5 so it stays independent of Plan 2.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server_fakes.py
from __future__ import annotations

from aixon.message import Chunk, Message
from aixon.registry import get_registry
from tests._server_fakes import EchoAgent, ReasoningAgent, make_echo


def test_make_echo_registers_and_resolves():
    inst = make_echo("alpha", aliases=["a1"], description="d")
    assert get_registry().resolve("alpha") is inst
    assert get_registry().resolve("a1") is inst
    assert inst.description == "d"


def test_echo_invoke_echoes_last_user_and_records_messages():
    inst = make_echo("alpha")
    msgs = [Message(role="system", content="s"), Message(role="user", content="hi")]
    out = inst.invoke(msgs)
    assert isinstance(out, Message)
    assert out.role == "assistant"
    assert out.content == "echo:hi"
    # It recorded exactly the neutral list it was handed.
    assert inst.seen is msgs
    assert all(isinstance(m, Message) for m in inst.seen)


def test_echo_stream_yields_content_then_done():
    inst = make_echo("alpha")
    chunks = list(inst.stream([Message(role="user", content="hi")]))
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[-1].done is True
    text = "".join(c.content for c in chunks if not c.done)
    assert text == "echo:hi"


def test_reasoning_agent_emits_reasoning():
    class _R(ReasoningAgent):
        name = "thinker"

    inst = get_registry().resolve("thinker")
    chunks = list(inst.stream([Message(role="user", content="hi")]))
    assert any(c.reasoning for c in chunks)
    assert chunks[-1].done is True
    assert inst.invoke([Message(role="user", content="hi")]).reasoning
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_server_fakes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests._server_fakes'`.

- [ ] **Step 3: Write the implementation**

```python
# tests/_server_fakes.py
"""Hermetic fake agents for server tests. Plain Agent subclasses — no LLM, no
provider SDK, no network. They stand in for real LLM agents so the FastAPI
boundary can be exercised end-to-end with TestClient.

``EchoAgent.seen`` captures the exact ``messages`` list passed to ``invoke`` so
a test can assert the agent only ever received neutral ``Message[]`` (the
no-vendor-leak guarantee)."""

from __future__ import annotations

from typing import Iterator

from aixon.agent import Agent
from aixon.message import Chunk, Message


def _last_user_text(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


class EchoAgent(Agent, abstract=True):
    """Abstract fake: echoes the last user message. Concrete via make_echo or a
    named subclass. ``seen`` records the last ``messages`` list received."""

    seen: list[Message] | None = None

    def invoke(self, messages: list[Message]) -> Message:
        self.seen = messages
        return Message(role="assistant", content="echo:" + _last_user_text(messages))

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        self.seen = messages
        text = "echo:" + _last_user_text(messages)
        # Two content deltas to prove deltas concatenate on the wire.
        yield Chunk(content=text[: len(text) // 2])
        yield Chunk(content=text[len(text) // 2 :])
        yield Chunk(done=True)


class ReasoningAgent(Agent, abstract=True):
    """Abstract fake that also emits reasoning, to prove reasoning survives a
    round trip through each dialect."""

    def invoke(self, messages: list[Message]) -> Message:
        return Message(
            role="assistant",
            content="answer",
            reasoning="because",
        )

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        yield Chunk(reasoning="thinking...")
        yield Chunk(content="ans")
        yield Chunk(content="wer")
        yield Chunk(done=True)


def make_echo(name, *, aliases=(), hidden=False, description=""):
    """Define + register a fresh EchoAgent subclass at call time and return the
    registered instance. Defining at call time keeps each test's autouse
    reset_registry clean."""
    from aixon.registry import get_registry

    cls = type(
        "MadeEchoAgent",
        (EchoAgent,),
        {
            "name": name,
            "aliases": list(aliases),
            "hidden": hidden,
            "description": description,
        },
    )
    return get_registry().resolve(cls.name)
```

> `ReasoningAgent`/`EchoAgent` are declared `abstract=True` so importing the module does NOT auto-register them (which would otherwise collide across tests). Concrete subclasses — `make_echo`'s generated class, or `class _R(ReasoningAgent)` in a test — register on definition, under the autouse `reset_registry` fixture.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_server_fakes.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/_server_fakes.py tests/test_server_fakes.py
git commit -m "test(server): hermetic fake agents (EchoAgent, ReasoningAgent, make_echo)"
```

---

### Task 3: `OpenAIAdapter` — pure translation, no HTTP yet

**Files:**
- Create: `aixon/server/adapters/openai.py`
- Test: `tests/test_adapter_openai.py`

**Interfaces:**
- Consumes: `aixon.server.protocol.{ParsedRequest, ProtocolAdapter, Message, Chunk}`, `aixon.message.Message`.
- Produces:
  - `aixon.server.adapters.openai.OpenAIAdapter(ProtocolAdapter)` with `name = "openai"` and:
    - `parse_request(body, *, path) -> ParsedRequest` — reads `body["model"]` (default `""`), `body["messages"]` → `list[Message]` (each `{role, content, name?, tool_call_id?}` → `Message`), `stream=bool(body.get("stream"))`, and `params` = the remaining knobs (everything except `model`/`messages`/`stream`, e.g. `temperature`, `max_tokens`, `top_p`).
    - `format_response(*, model, message, usage) -> dict` — the `chat.completion` envelope: `{id, object:"chat.completion", created, model, choices:[{index:0, message:{role, content}, finish_reason:"stop"}], usage}`. If `message.reasoning` is set, include it as `choices[0].message["reasoning"]` (reasoning-field mode; default).
    - `format_stream_chunk(*, model, chunk) -> str` — one `chat.completion.chunk` SSE line. Content delta → `delta:{"content": ...}`; reasoning delta → `delta:{"reasoning": ...}`; a `done=True` chunk → the finish line `delta:{}, finish_reason:"stop"`. An empty chunk (no content, no reasoning, not done) → `""` (skip).
    - `format_stream_done(*, model) -> str` — `"data: [DONE]\n\n"`.
    - `format_models(agents) -> dict` — `{"object":"list","data":[{id, object:"model", created, owned_by}, ...]}`, one entry per agent **plus one per alias**, mirroring olympus.
    - `routes() -> list[tuple[str,str]]` — `[("POST","/v1/chat/completions"), ("GET","/v1/models")]`.
  - A stable stream id per response is generated by the Server (Task 5), not the adapter; the adapter formats whatever id/created the Server threads in. To keep `format_stream_chunk` stateless and pure, the adapter generates a fresh `id`/`created` lazily is NOT done — instead the chunk line carries a per-call `id`/`created`. **Decision:** the OpenAI stream id may differ per chunk in a hermetic test and clients tolerate it, but to be correct the adapter exposes `new_stream_id() -> tuple[str,int]` and `format_stream_chunk`/`format_stream_done` accept the **model only** (per the contract signature), generating `id`/`created` internally per line. Tests assert `object`/`choices`/`delta` shape, not id stability across lines.

> Contract fidelity: the abstract signatures are fixed (`format_stream_chunk(self, *, model, chunk)` and `format_stream_done(self, *, model)`), so id/created are generated inside each call. This matches how olympus assigns them once per response, with the only observable difference being that the id is not constant across lines — irrelevant to OpenAI clients, which key off `object`/`choices`. (Ambiguity resolved: contract pins the method signatures, so per-line id generation is the only conformant option.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adapter_openai.py
from __future__ import annotations

import json

from aixon.message import Chunk, Message
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.protocol import ParsedRequest


def _data(line: str) -> dict:
    assert line.startswith("data: ")
    assert line.endswith("\n\n")
    return json.loads(line[len("data: ") : -2])


class TestParseRequest:
    def test_parses_model_messages_stream(self):
        a = OpenAIAdapter()
        pr = a.parse_request(
            {
                "model": "echo",
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "hi"},
                ],
                "stream": True,
                "temperature": 0.3,
            },
            path="/v1/chat/completions",
        )
        assert isinstance(pr, ParsedRequest)
        assert pr.model == "echo"
        assert [m.role for m in pr.messages] == ["system", "user"]
        assert all(isinstance(m, Message) for m in pr.messages)
        assert pr.stream is True
        assert pr.params == {"temperature": 0.3}

    def test_defaults_when_fields_absent(self):
        a = OpenAIAdapter()
        pr = a.parse_request({"messages": []}, path="/v1/chat/completions")
        assert pr.model == ""
        assert pr.messages == []
        assert pr.stream is False
        assert pr.params == {}


class TestFormatResponse:
    def test_chat_completion_envelope(self):
        a = OpenAIAdapter()
        out = a.format_response(
            model="echo",
            message=Message(role="assistant", content="hello"),
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
        assert out["object"] == "chat.completion"
        assert out["model"] == "echo"
        assert out["id"].startswith("chatcmpl-")
        choice = out["choices"][0]
        assert choice["index"] == 0
        assert choice["finish_reason"] == "stop"
        assert choice["message"] == {"role": "assistant", "content": "hello"}
        assert out["usage"]["total_tokens"] == 2

    def test_reasoning_included_when_present(self):
        a = OpenAIAdapter()
        out = a.format_response(
            model="echo",
            message=Message(role="assistant", content="x", reasoning="why"),
            usage={},
        )
        assert out["choices"][0]["message"]["reasoning"] == "why"


class TestFormatStream:
    def test_content_chunk(self):
        a = OpenAIAdapter()
        d = _data(a.format_stream_chunk(model="echo", chunk=Chunk(content="hi")))
        assert d["object"] == "chat.completion.chunk"
        assert d["model"] == "echo"
        assert d["choices"][0]["delta"] == {"content": "hi"}
        assert d["choices"][0]["finish_reason"] is None

    def test_reasoning_chunk(self):
        a = OpenAIAdapter()
        d = _data(a.format_stream_chunk(model="echo", chunk=Chunk(reasoning="r")))
        assert d["choices"][0]["delta"] == {"reasoning": "r"}

    def test_done_chunk_is_finish_line(self):
        a = OpenAIAdapter()
        d = _data(a.format_stream_chunk(model="echo", chunk=Chunk(done=True)))
        assert d["choices"][0]["delta"] == {}
        assert d["choices"][0]["finish_reason"] == "stop"

    def test_empty_chunk_skipped(self):
        a = OpenAIAdapter()
        assert a.format_stream_chunk(model="echo", chunk=Chunk()) == ""

    def test_stream_done_is_done_sentinel(self):
        a = OpenAIAdapter()
        assert a.format_stream_done(model="echo") == "data: [DONE]\n\n"


class TestFormatModelsAndRoutes:
    def test_models_lists_agents_and_aliases(self):
        a = OpenAIAdapter()

        class _Fake:
            def __init__(self, name, aliases, owned_by):
                self.name = name
                self.aliases = aliases
                self.owned_by = owned_by

        out = a.format_models([_Fake("echo", ["e1"], "aixon")])
        assert out["object"] == "list"
        ids = [d["id"] for d in out["data"]]
        assert ids == ["echo", "e1"]
        assert all(d["object"] == "model" for d in out["data"])
        assert out["data"][0]["owned_by"] == "aixon"

    def test_routes(self):
        a = OpenAIAdapter()
        assert a.routes() == [
            ("POST", "/v1/chat/completions"),
            ("GET", "/v1/models"),
        ]
        assert a.name == "openai"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_adapter_openai.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aixon.server.adapters.openai'`.

- [ ] **Step 3: Write the implementation**

```python
# aixon/server/adapters/openai.py
"""OpenAI-compatible ProtocolAdapter — the full, primary dialect.

Wire shapes mirror OpenAI's Chat Completions API (translated from olympus's
Flask handler to pure neutral translation): ``/v1/chat/completions`` (stream +
non-stream) and ``/v1/models``. Reasoning is surfaced in the ``message``/``delta``
``reasoning`` field (reasoning-field mode)."""

from __future__ import annotations

import json
import time
import uuid

from aixon.server.protocol import Chunk, Message, ParsedRequest, ProtocolAdapter

# Transport-level fields the adapter consumes itself; everything else in the
# body is a passthrough param handed to the agent's params.
_TRANSPORT_FIELDS = frozenset({"model", "messages", "stream"})


class OpenAIAdapter(ProtocolAdapter):
    name = "openai"

    # --- inbound ---------------------------------------------------------
    def parse_request(self, body: dict, *, path: str) -> ParsedRequest:
        raw_messages = body.get("messages") or []
        messages: list[Message] = []
        for m in raw_messages:
            messages.append(
                Message(
                    role=m.get("role", "user"),
                    content=m.get("content") or "",
                    name=m.get("name"),
                    tool_call_id=m.get("tool_call_id"),
                )
            )
        params = {k: v for k, v in body.items() if k not in _TRANSPORT_FIELDS}
        return ParsedRequest(
            model=body.get("model") or "",
            messages=messages,
            params=params,
            stream=bool(body.get("stream", False)),
        )

    # --- outbound (non-stream) ------------------------------------------
    def format_response(self, *, model: str, message: Message, usage: dict) -> dict:
        msg: dict = {"role": "assistant", "content": message.content}
        if message.reasoning is not None:
            msg["reasoning"] = message.reasoning
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "message": msg, "finish_reason": "stop"}
            ],
            "usage": usage,
        }

    # --- outbound (stream) ----------------------------------------------
    def _chunk_line(self, *, model: str, delta: dict, finish_reason) -> str:
        payload = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def format_stream_chunk(self, *, model: str, chunk: Chunk) -> str:
        if chunk.done:
            return self._chunk_line(model=model, delta={}, finish_reason="stop")
        if chunk.content:
            return self._chunk_line(
                model=model, delta={"content": chunk.content}, finish_reason=None
            )
        if chunk.reasoning:
            return self._chunk_line(
                model=model, delta={"reasoning": chunk.reasoning}, finish_reason=None
            )
        return ""  # nothing to emit for an empty chunk

    def format_stream_done(self, *, model: str) -> str:
        return "data: [DONE]\n\n"

    # --- model listing ---------------------------------------------------
    def format_models(self, agents: list) -> dict:
        created = int(time.time())
        data = []
        for agent in agents:
            owned_by = getattr(agent, "owned_by", "aixon")
            data.append(
                {"id": agent.name, "object": "model", "created": created, "owned_by": owned_by}
            )
            for alias in getattr(agent, "aliases", []) or []:
                data.append(
                    {"id": alias, "object": "model", "created": created, "owned_by": owned_by}
                )
        return {"object": "list", "data": data}

    # --- routing ---------------------------------------------------------
    def routes(self) -> list[tuple[str, str]]:
        return [("POST", "/v1/chat/completions"), ("GET", "/v1/models")]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_adapter_openai.py -v`
Expected: PASS (12 tests).

- [ ] **Step 5: Commit**

```bash
git add aixon/server/adapters/openai.py tests/test_adapter_openai.py
git commit -m "feat(server): OpenAIAdapter — chat.completions + chunk deltas + models translation"
```

---

### Task 4: `AnthropicAdapter` — thin proof of decoupling

**Files:**
- Create: `aixon/server/adapters/anthropic.py`
- Test: `tests/test_adapter_anthropic.py`

**Interfaces:**
- Consumes: `aixon.server.protocol.{ParsedRequest, ProtocolAdapter, Message, Chunk}`.
- Produces:
  - `aixon.server.adapters.anthropic.AnthropicAdapter(ProtocolAdapter)` with `name = "anthropic"` and:
    - `parse_request(body, *, path) -> ParsedRequest` — **`system` is a top-level field, NOT in `messages[]`**. If `body["system"]` is present (a string), prepend a `Message(role="system", content=system)` to the parsed `messages`. `model` ← `body["model"]`; `messages` ← `body["messages"]` (each `{role, content}` where `content` may be a string or a list of `{type:"text", text}` blocks → flatten to text); `stream` ← `bool(body.get("stream"))`; `params` ← remaining knobs minus `model`/`messages`/`stream`/`system` (e.g. `max_tokens`, `temperature`).
    - `format_response(*, model, message, usage) -> dict` — the Messages envelope: `{id, type:"message", role:"assistant", model, content:[{type:"text", text: message.content}], stop_reason:"end_turn", stop_sequence:None, usage:{input_tokens, output_tokens}}`. Map neutral `usage` (`prompt_tokens`/`completion_tokens`) → Anthropic (`input_tokens`/`output_tokens`) when present.
    - `format_stream_chunk(*, model, chunk) -> str` — **named SSE events**, not bare `data:`. A content delta → an `event: content_block_delta` line plus its `data:` line carrying `{type:"content_block_delta", index:0, delta:{type:"text_delta", text:...}}`. A reasoning delta → `event: content_block_delta` with `delta:{type:"thinking_delta", thinking:...}`. A `done=True` chunk → `event: message_delta` with `data:{type:"message_delta", delta:{stop_reason:"end_turn"}}`. Empty chunk → `""`.
    - `format_stream_done(*, model) -> str` — `event: message_stop\ndata: {"type":"message_stop"}\n\n` (the named terminal event; Anthropic has no `[DONE]` sentinel).
    - `format_models(agents) -> dict` — Anthropic has no `/v1/models`; return `{"data":[{"type":"model","id":agent.name} ...]}` (aliases too) so the Server can still expose a public listing on this adapter's GET route.
    - `routes() -> list[tuple[str,str]]` — `[("POST","/v1/messages"), ("GET","/v1/models")]`. (The GET `/v1/models` is the public model-list route for this adapter; the Server treats every adapter's GET-list route as public — see Task 5.)

> This adapter exists to PROVE the neutral types are not OpenAI-in-disguise: it hoists `system` out of `messages`, emits typed `content[]` blocks, a `stop_reason` envelope, and *named* SSE events — all from the same neutral `Message`/`Chunk`. (Ambiguity resolved: the contract's `format_models(agents)` is required on every adapter; Anthropic's real API lacks a models endpoint, so this adapter serves a minimal Anthropic-flavored listing on `GET /v1/models` purely so the Server's "model-list is public" rule has a route to attach to. Documented deviation, kept thin.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adapter_anthropic.py
from __future__ import annotations

import json

from aixon.message import Chunk, Message
from aixon.server.adapters.anthropic import AnthropicAdapter


def _event_blocks(text: str) -> list[tuple[str, dict]]:
    """Split an SSE payload into (event_name, data_dict) pairs."""
    out = []
    for block in [b for b in text.split("\n\n") if b.strip()]:
        lines = block.split("\n")
        event = next(l[len("event: ") :] for l in lines if l.startswith("event: "))
        data = next(json.loads(l[len("data: ") :]) for l in lines if l.startswith("data: "))
        out.append((event, data))
    return out


class TestParseRequest:
    def test_system_is_hoisted_into_messages(self):
        a = AnthropicAdapter()
        pr = a.parse_request(
            {
                "model": "echo",
                "system": "you are terse",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
            path="/v1/messages",
        )
        assert pr.model == "echo"
        assert pr.messages[0].role == "system"
        assert pr.messages[0].content == "you are terse"
        assert pr.messages[1].role == "user"
        assert pr.params == {"max_tokens": 100}
        assert pr.stream is False

    def test_content_blocks_flattened_to_text(self):
        a = AnthropicAdapter()
        pr = a.parse_request(
            {
                "model": "echo",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "a"},
                                                 {"type": "text", "text": "b"}]}
                ],
            },
            path="/v1/messages",
        )
        assert pr.messages[0].content == "ab"


class TestFormatResponse:
    def test_messages_envelope(self):
        a = AnthropicAdapter()
        out = a.format_response(
            model="echo",
            message=Message(role="assistant", content="hello"),
            usage={"prompt_tokens": 3, "completion_tokens": 5},
        )
        assert out["type"] == "message"
        assert out["role"] == "assistant"
        assert out["model"] == "echo"
        assert out["content"] == [{"type": "text", "text": "hello"}]
        assert out["stop_reason"] == "end_turn"
        assert out["usage"] == {"input_tokens": 3, "output_tokens": 5}


class TestFormatStream:
    def test_content_delta_event(self):
        a = AnthropicAdapter()
        (event, data), = _event_blocks(
            a.format_stream_chunk(model="echo", chunk=Chunk(content="hi"))
        )
        assert event == "content_block_delta"
        assert data["delta"] == {"type": "text_delta", "text": "hi"}

    def test_reasoning_delta_event(self):
        a = AnthropicAdapter()
        (event, data), = _event_blocks(
            a.format_stream_chunk(model="echo", chunk=Chunk(reasoning="r"))
        )
        assert event == "content_block_delta"
        assert data["delta"] == {"type": "thinking_delta", "thinking": "r"}

    def test_done_emits_message_delta(self):
        a = AnthropicAdapter()
        (event, data), = _event_blocks(
            a.format_stream_chunk(model="echo", chunk=Chunk(done=True))
        )
        assert event == "message_delta"
        assert data["delta"]["stop_reason"] == "end_turn"

    def test_empty_chunk_skipped(self):
        a = AnthropicAdapter()
        assert a.format_stream_chunk(model="echo", chunk=Chunk()) == ""

    def test_stream_done_is_named_message_stop(self):
        a = AnthropicAdapter()
        (event, data), = _event_blocks(a.format_stream_done(model="echo"))
        assert event == "message_stop"
        assert data["type"] == "message_stop"


class TestModelsAndRoutes:
    def test_models_listing(self):
        a = AnthropicAdapter()

        class _Fake:
            name = "echo"
            aliases = ["e1"]
            owned_by = "aixon"

        out = a.format_models([_Fake()])
        ids = [d["id"] for d in out["data"]]
        assert ids == ["echo", "e1"]
        assert all(d["type"] == "model" for d in out["data"])

    def test_routes(self):
        a = AnthropicAdapter()
        assert a.routes() == [("POST", "/v1/messages"), ("GET", "/v1/models")]
        assert a.name == "anthropic"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_adapter_anthropic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aixon.server.adapters.anthropic'`.

- [ ] **Step 3: Write the implementation**

```python
# aixon/server/adapters/anthropic.py
"""Anthropic Messages-API ProtocolAdapter — the thin PROOF that aixon's neutral
types are not OpenAI-in-disguise.

Structural differences from OpenAI, served from the SAME neutral Message/Chunk:
- ``system`` is a top-level request field, hoisted into a neutral system Message.
- responses use a typed ``content[]`` block envelope with ``stop_reason``.
- streaming uses *named* SSE events (content_block_delta / message_delta /
  message_stop), not a bare ``data:`` line + ``[DONE]`` sentinel."""

from __future__ import annotations

import json
import time
import uuid

from aixon.server.protocol import Chunk, Message, ParsedRequest, ProtocolAdapter

_TRANSPORT_FIELDS = frozenset({"model", "messages", "stream", "system"})


def _flatten_content(content) -> str:
    """Anthropic content may be a string or a list of typed blocks. Flatten the
    text blocks to neutral plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class AnthropicAdapter(ProtocolAdapter):
    name = "anthropic"

    # --- inbound ---------------------------------------------------------
    def parse_request(self, body: dict, *, path: str) -> ParsedRequest:
        messages: list[Message] = []
        system = body.get("system")
        if isinstance(system, str) and system:
            messages.append(Message(role="system", content=system))
        for m in body.get("messages") or []:
            messages.append(
                Message(role=m.get("role", "user"), content=_flatten_content(m.get("content")))
            )
        params = {k: v for k, v in body.items() if k not in _TRANSPORT_FIELDS}
        return ParsedRequest(
            model=body.get("model") or "",
            messages=messages,
            params=params,
            stream=bool(body.get("stream", False)),
        )

    # --- outbound (non-stream) ------------------------------------------
    def format_response(self, *, model: str, message: Message, usage: dict) -> dict:
        out_usage = {}
        if "prompt_tokens" in usage or "completion_tokens" in usage:
            out_usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }
        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": message.content}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": out_usage,
        }

    # --- outbound (stream) ----------------------------------------------
    def format_stream_chunk(self, *, model: str, chunk: Chunk) -> str:
        if chunk.done:
            return _event(
                "message_delta",
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
            )
        if chunk.content:
            return _event(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0,
                 "delta": {"type": "text_delta", "text": chunk.content}},
            )
        if chunk.reasoning:
            return _event(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0,
                 "delta": {"type": "thinking_delta", "thinking": chunk.reasoning}},
            )
        return ""

    def format_stream_done(self, *, model: str) -> str:
        return _event("message_stop", {"type": "message_stop"})

    # --- model listing ---------------------------------------------------
    def format_models(self, agents: list) -> dict:
        data = []
        for agent in agents:
            data.append({"type": "model", "id": agent.name})
            for alias in getattr(agent, "aliases", []) or []:
                data.append({"type": "model", "id": alias})
        return {"data": data}

    # --- routing ---------------------------------------------------------
    def routes(self) -> list[tuple[str, str]]:
        return [("POST", "/v1/messages"), ("GET", "/v1/models")]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_adapter_anthropic.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add aixon/server/adapters/anthropic.py tests/test_adapter_anthropic.py
git commit -m "feat(server): AnthropicAdapter — system-hoist, content[] blocks, named SSE events (decoupling proof)"
```

---

### Task 5: `Server` — FastAPI app, singleton, mount adapters, auth, serve()

**Files:**
- Create: `aixon/server/server.py`
- Modify: `aixon/__init__.py` (export the public server surface)
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `aixon.server.protocol.{ProtocolAdapter, ParsedRequest, Message, Chunk}`, `aixon.server.adapters.openai.OpenAIAdapter`, `aixon.registry.get_registry`, `aixon.exceptions.AgentNotFoundError`, `aixon.logging.Logger`, FastAPI/Starlette (lazy).
- Produces (contract §4.3):
  - `aixon.server.server.Server`:
    - `__init__(self, adapters: list[ProtocolAdapter] | None = None)` — default `[OpenAIAdapter()]`. Singleton: repeated construction returns the same instance and ignores new `adapters` (matching restmcp's `__new__`-based singleton). The instance builds the FastAPI app once.
    - `@property app` — the FastAPI ASGI app, with every adapter's routes mounted, plus a public `GET /health`.
    - `serve(self, host="0.0.0.0", port=8000)` — `uvicorn.run(self.app, ...)` (lazy `import uvicorn`).
    - `@classmethod get_instance(cls) -> "Server"` — return the singleton (constructing the default if needed).
    - `@classmethod _reset(cls)` — drop the singleton (tests; mirrors restmcp).
  - Request flow per POST route: `body = await request.json()` → `adapter.parse_request(body, path=...)` → `get_registry().resolve(pr.model)` (catch `AgentNotFoundError` → 404 with a `{"error": {...}}` body) → if `pr.stream`: `StreamingResponse` iterating `agent.stream(pr.messages)` through `adapter.format_stream_chunk` then `adapter.format_stream_done` (`media_type="text/event-stream"`); else `agent.invoke(pr.messages)` → `adapter.format_response(model=pr.model or resolved.name, message=..., usage={})`. Log `INFO` the resolved agent name + route.
  - Per GET model-list route: `adapter.format_models(get_registry().public())`.
  - **Auth:** a pure-ASGI Bearer middleware (mirroring restmcp's `AuthMiddleware`) wraps the app when `AUTH_API_KEY` is set; `public_paths` = `{"/health"}` ∪ every adapter's GET (model-list) path. No-op when `AUTH_API_KEY` is unset.
  - Exported from `aixon`: `Server`, `ProtocolAdapter`, `OpenAIAdapter`, `AnthropicAdapter`, `ParsedRequest`.

> **`usage`** is passed as `{}` (token counting is out of scope for Plan 5 — olympus's counter depends on a provider model object aixon's neutral boundary deliberately hides; the envelope key is present and adapters tolerate an empty dict). Documented deviation from olympus, intentional.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server.py
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from aixon.message import Message
from aixon.server.adapters.anthropic import AnthropicAdapter
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.server import Server
from tests._server_fakes import ReasoningAgent, make_echo


@pytest.fixture(autouse=True)
def _reset_server():
    Server._reset()
    yield
    Server._reset()


@pytest.fixture
def client():
    make_echo("echo", aliases=["echo-alias"], description="d")
    return TestClient(Server(adapters=[OpenAIAdapter()]).app)


# --- health + models -----------------------------------------------------
def test_health_is_public_and_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_models_lists_registered_agent_and_alias(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = [d["id"] for d in r.json()["data"]]
    assert "echo" in ids and "echo-alias" in ids


# --- OpenAI non-stream ---------------------------------------------------
def test_chat_completions_non_stream(client):
    r = client.post("/v1/chat/completions", json={
        "model": "echo",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "echo:hi"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_model_resolution_via_alias(client):
    r = client.post("/v1/chat/completions", json={
        "model": "echo-alias",
        "messages": [{"role": "user", "content": "yo"}],
    })
    assert r.json()["choices"][0]["message"]["content"] == "echo:yo"


def test_unknown_model_with_multiple_agents_is_404():
    make_echo("a")
    make_echo("b")
    c = TestClient(Server(adapters=[OpenAIAdapter()]).app)
    r = c.post("/v1/chat/completions", json={
        "model": "nope", "messages": [{"role": "user", "content": "x"}],
    })
    assert r.status_code == 404
    assert "error" in r.json()


# --- OpenAI stream -------------------------------------------------------
def _sse_data_lines(text: str) -> list[str]:
    return [l[len("data: ") :] for l in text.splitlines() if l.startswith("data: ")]


def test_chat_completions_stream(client):
    r = client.post("/v1/chat/completions", json={
        "model": "echo",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    datas = _sse_data_lines(r.text)
    assert datas[-1] == "[DONE]"
    parsed = [json.loads(d) for d in datas if d != "[DONE]"]
    assert all(p["object"] == "chat.completion.chunk" for p in parsed)
    content = "".join(
        p["choices"][0]["delta"].get("content", "") for p in parsed
    )
    assert content == "echo:hi"
    assert parsed[-1]["choices"][0]["finish_reason"] == "stop"


# --- neutral boundary: no vendor type leaks into the agent ---------------
def test_agent_only_ever_receives_neutral_messages(client):
    client.post("/v1/chat/completions", json={
        "model": "echo",
        "messages": [{"role": "user", "content": "leak-check"}],
    })
    agent = make_echo  # not used; resolve the live instance instead
    from aixon.registry import get_registry
    inst = get_registry().resolve("echo")
    assert inst.seen is not None
    assert all(isinstance(m, Message) for m in inst.seen)
    # No dict / vendor body slipped through.
    assert not any(isinstance(m, dict) for m in inst.seen)


# --- Anthropic adapter mounted on the same server ------------------------
@pytest.fixture
def anthropic_client():
    make_echo("echo")
    return TestClient(Server(adapters=[AnthropicAdapter()]).app)


def test_anthropic_messages_non_stream(anthropic_client):
    r = anthropic_client.post("/v1/messages", json={
        "model": "echo",
        "system": "be terse",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 50,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message"
    assert body["content"][0]["text"] == "echo:hi"
    assert body["stop_reason"] == "end_turn"


def test_anthropic_messages_stream_named_events(anthropic_client):
    r = anthropic_client.post("/v1/messages", json={
        "model": "echo",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    assert r.status_code == 200
    assert "event: content_block_delta" in r.text
    assert "event: message_stop" in r.text
    assert "[DONE]" not in r.text  # Anthropic has no [DONE] sentinel


# --- auth ON / OFF -------------------------------------------------------
def test_auth_off_when_env_unset(monkeypatch):
    monkeypatch.delenv("AUTH_API_KEY", raising=False)
    make_echo("echo")
    c = TestClient(Server(adapters=[OpenAIAdapter()]).app)
    r = c.post("/v1/chat/completions", json={
        "model": "echo", "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200


def test_auth_on_rejects_missing_and_bad_bearer(monkeypatch):
    monkeypatch.setenv("AUTH_API_KEY", "secret123")
    make_echo("echo")
    c = TestClient(Server(adapters=[OpenAIAdapter()]).app)
    payload = {"model": "echo", "messages": [{"role": "user", "content": "hi"}]}
    assert c.post("/v1/chat/completions", json=payload).status_code == 401
    assert c.post(
        "/v1/chat/completions", json=payload, headers={"Authorization": "Bearer wrong"}
    ).status_code == 401


def test_auth_on_accepts_good_bearer_and_keeps_public_routes_open(monkeypatch):
    monkeypatch.setenv("AUTH_API_KEY", "secret123")
    make_echo("echo")
    c = TestClient(Server(adapters=[OpenAIAdapter()]).app)
    payload = {"model": "echo", "messages": [{"role": "user", "content": "hi"}]}
    assert c.post(
        "/v1/chat/completions", json=payload, headers={"Authorization": "Bearer secret123"}
    ).status_code == 200
    # public routes stay open even with auth on
    assert c.get("/health").status_code == 200
    assert c.get("/v1/models").status_code == 200
```

> **Singleton + TestClient note:** because `Server` is a singleton, each test that needs a *different* adapter set or a fresh app must run under the autouse `Server._reset()` fixture (provided above) so `Server(adapters=[...])` rebuilds. Auth is read live from the env on every request by the ASGI middleware, so `monkeypatch.setenv` after app construction still takes effect (the middleware checks `os.getenv` per call, exactly like restmcp).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aixon.server.server'`.

- [ ] **Step 3: Write the implementation**

```python
# aixon/server/server.py
"""The aixon ASGI Server.

A singleton FastAPI app that mounts one or more ProtocolAdapters over the agent
Registry. Request flow:

    ASGI -> adapter.parse_request -> get_registry().resolve(model)
         -> agent.invoke|stream  (NEUTRAL Message[]/Chunk only)
         -> adapter.format_*      -> HTTP / SSE

The Server is dialect-agnostic: every wire detail lives in the adapter. Bearer
auth (AUTH_API_KEY env) wraps the whole app when set; /health and each
adapter's model-list route stay public. Mirrors restmcp's FastAPI + ASGI-auth
construction."""

from __future__ import annotations

import datetime as dt
import hmac
import os
from typing import Optional

from aixon.exceptions import AgentNotFoundError
from aixon.logging import Logger
from aixon.registry import get_registry
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.protocol import ProtocolAdapter

_log = Logger("aixon.server")


def _valid_token(raw: str) -> bool:
    keys = [k.strip() for k in os.getenv("AUTH_API_KEY", "").split(",") if k.strip()]
    return bool(raw) and any(hmac.compare_digest(raw, k) for k in keys)


class _AuthMiddleware:
    """Pure-ASGI Bearer middleware. No-op when AUTH_API_KEY is unset. Does not
    buffer the body, so SSE streaming is unaffected. ``public_paths`` are
    matched by exact path."""

    def __init__(self, app, public_paths):
        self.app = app
        self.public = frozenset(public_paths)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not os.getenv("AUTH_API_KEY"):
            return await self.app(scope, receive, send)
        if scope.get("path", "") in self.public:
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode("utf-8", "ignore")
        token = auth.split(" ", 1)[1] if auth.startswith("Bearer ") else None
        if not token or not _valid_token(token):
            from starlette.responses import JSONResponse

            resp = JSONResponse({"error": "Unauthorized"}, status_code=401)
            return await resp(scope, receive, send)
        return await self.app(scope, receive, send)


class Server:
    _instance: Optional["Server"] = None

    def __new__(cls, adapters: list[ProtocolAdapter] | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, adapters: list[ProtocolAdapter] | None = None):
        if self._initialized:
            return
        self._adapters: list[ProtocolAdapter] = adapters or [OpenAIAdapter()]
        self._app = None
        self._initialized = True

    # --- app construction ------------------------------------------------
    @property
    def app(self):
        if self._app is None:
            self._app = self._build_app()
        return self._app

    def _public_paths(self) -> set[str]:
        public = {"/health"}
        for adapter in self._adapters:
            for method, path in adapter.routes():
                if method.upper() == "GET":
                    public.add(path)  # model-list routes stay public
        return public

    def _build_app(self):
        from fastapi import FastAPI, Request
        from starlette.middleware.cors import CORSMiddleware

        app = FastAPI()
        cors = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
        app.add_middleware(
            CORSMiddleware, allow_origins=cors, allow_methods=["*"], allow_headers=["*"]
        )

        @app.get("/health")
        def health():
            return {
                "status": "healthy",
                "server": "aixon",
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            }

        for adapter in self._adapters:
            self._mount_adapter(app, adapter)

        if os.getenv("AUTH_API_KEY"):
            return _AuthMiddleware(app, public_paths=self._public_paths())
        # Wrap unconditionally so the middleware can react to AUTH_API_KEY set
        # AFTER construction (tests, hot-reload). The middleware is a no-op when
        # the env is unset, so this is safe and matches restmcp's per-request check.
        return _AuthMiddleware(app, public_paths=self._public_paths())

    def _mount_adapter(self, app, adapter: ProtocolAdapter) -> None:
        from fastapi import Request
        from starlette.responses import JSONResponse, StreamingResponse

        for method, path in adapter.routes():
            if method.upper() == "GET":
                self._mount_models(app, adapter, path)
            else:
                self._mount_chat(app, adapter, path, Request, JSONResponse, StreamingResponse)

    def _mount_models(self, app, adapter: ProtocolAdapter, path: str) -> None:
        async def list_models():
            _log.info(f"{adapter.name}: GET {path} (model list)")
            return adapter.format_models(get_registry().public())

        app.add_api_route(path, list_models, methods=["GET"])

    def _mount_chat(self, app, adapter, path, Request, JSONResponse, StreamingResponse) -> None:
        async def chat(request: Request):
            body = await request.json()
            pr = adapter.parse_request(body, path=path)
            try:
                agent = get_registry().resolve(pr.model)
            except AgentNotFoundError as exc:
                return JSONResponse(
                    {"error": {"message": exc.message, "type": "model_not_found"}},
                    status_code=404,
                )
            _log.info(f"{adapter.name}: {path} -> agent '{agent.name}' (stream={pr.stream})")
            model = pr.model or agent.name

            if pr.stream:
                def gen():
                    for chunk in agent.stream(pr.messages):
                        line = adapter.format_stream_chunk(model=model, chunk=chunk)
                        if line:
                            yield line
                    yield adapter.format_stream_done(model=model)

                return StreamingResponse(gen(), media_type="text/event-stream")

            message = agent.invoke(pr.messages)
            return adapter.format_response(model=model, message=message, usage={})

        app.add_api_route(path, chat, methods=["POST"])

    # --- lifecycle -------------------------------------------------------
    def serve(self, host: str = "0.0.0.0", port: int = 8000):
        import uvicorn

        uvicorn.run(self.app, host=host, port=port)

    @classmethod
    def get_instance(cls) -> "Server":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def _reset(cls):
        cls._instance = None
```

Now update `aixon/__init__.py` to export the server surface. Append the server imports and extend `__all__` (do NOT remove any existing exports — match the current file and add to it):

```python
# aixon/__init__.py  — add these imports alongside the existing ones
from aixon.server.adapters.anthropic import AnthropicAdapter
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.protocol import ParsedRequest, ProtocolAdapter
from aixon.server.server import Server
```

```python
# aixon/__init__.py  — add these names to __all__
    "Server",
    "ProtocolAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "ParsedRequest",
```

> **Import-safety caveat:** these top-level imports pull FastAPI in at `import aixon` time (because `server.server` imports FastAPI inside `_build_app` only, but the module itself is imported eagerly here). FastAPI is NOT a core dependency. To keep `import aixon` working on a bare install, guard the server block: wrap the four server imports in a `try/except ImportError` that leaves the names unset and omits them from `__all__` when the `server` extra is absent. Concretely:
>
> ```python
> # aixon/__init__.py — server surface (optional; requires aixon[server])
> try:
>     from aixon.server.adapters.anthropic import AnthropicAdapter
>     from aixon.server.adapters.openai import OpenAIAdapter
>     from aixon.server.protocol import ParsedRequest, ProtocolAdapter
>     from aixon.server.server import Server
>     __all__ += ["Server", "ProtocolAdapter", "OpenAIAdapter", "AnthropicAdapter", "ParsedRequest"]
> except ImportError:  # aixon[server] not installed
>     pass
> ```
>
> `aixon.server.server` itself only needs FastAPI when `.app` is first built, but `_AuthMiddleware`/route mounting reference it lazily, so the module imports cleanly whenever FastAPI is present. Keep `protocol.py` import-safe (no FastAPI) so `aixon.server.protocol` is always importable for the seam tests. (Ambiguity resolved: the contract says export "from `aixon` (or a documented `aixon.server` namespace)"; this plan exports from top-level `aixon` but behind an optional-import guard, so a bare install still imports cleanly — documented deviation that honors both the contract and the "server deps live in an extra" constraint.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_server.py -v`
Expected: PASS (12 tests).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS (all foundation + Plan 5 tests).

- [ ] **Step 6: Smoke test app construction (no port bind)**

Run:
```bash
cd /Users/jorge/Documents/Git/aixon && python - <<'EOF'
from fastapi.testclient import TestClient
from aixon import Server, OpenAIAdapter, AnthropicAdapter
from aixon.agent import Agent
from aixon.message import Message, Chunk

class EchoAgent(Agent):
    name = "echo"
    def invoke(self, messages):
        return Message(role="assistant", content="echo:" + messages[-1].content)
    def stream(self, messages):
        yield Chunk(content="echo:" + messages[-1].content); yield Chunk(done=True)

app = Server(adapters=[OpenAIAdapter(), AnthropicAdapter()]).app
c = TestClient(app)
print("health:", c.get("/health").json()["status"])
print("models:", [m["id"] for m in c.get("/v1/models").json()["data"]])
print("openai:", c.post("/v1/chat/completions",
      json={"model": "echo", "messages": [{"role": "user", "content": "hi"}]}
      ).json()["choices"][0]["message"]["content"])
print("anthropic:", c.post("/v1/messages",
      json={"model": "echo", "messages": [{"role": "user", "content": "hi"}]}
      ).json()["content"][0]["text"])
print("smoke OK")
EOF
```
Expected: prints `health: healthy`, the model ids, `openai: echo:hi`, `anthropic: echo:hi`, `smoke OK`.

- [ ] **Step 7: Commit**

```bash
git add aixon/server/server.py aixon/__init__.py tests/test_server.py
git commit -m "feat(server): Server — FastAPI app, singleton, adapter mounting, Bearer auth, serve()"
```

---

## Self-Review

**Spec coverage — protocol-decoupling + adapter requirements (spec §"Desacoplamento de protocolo", contract §4):**
- Protocol seam separating agent runtime (neutral `Message[]`/`Chunk`) from the wire dialect → `ProtocolAdapter` ABC + `ParsedRequest` in `server/protocol.py` (Task 1). ✓
- Neutral types re-exported from `protocol.py` so adapters import from one place → Task 1 (`Message`/`Chunk`/`Role` re-export, asserted `is` identical). ✓
- **`OpenAIAdapter` — full:** `/v1/chat/completions` non-stream (Task 5 `test_chat_completions_non_stream`) AND stream with `chat.completion.chunk` deltas (Task 3 unit + Task 5 `test_chat_completions_stream`), `/v1/models` (Tasks 3 + 5). Reasoning surfaced via the `reasoning` field. ✓
- **`AnthropicAdapter` — thin proof:** `/v1/messages`, **system outside the array** (hoisted to a neutral system `Message`), typed `content[]` blocks, `stop_reason` envelope, **named** stream events (`content_block_delta`/`message_delta`/`message_stop`, no `[DONE]`) → Task 4 unit + Task 5 (`test_anthropic_messages_non_stream`, `test_anthropic_messages_stream_named_events`). Proves the neutral types aren't OpenAI-in-disguise. ✓
- **Model resolution via registry** (name → alias → single-agent default → 404) → Task 5 (`test_model_resolution_via_alias`, `test_unknown_model_with_multiple_agents_is_404`). ✓
- **Auth ON/OFF:** 401 without bearer and with bad bearer; 200 with good bearer; public routes (`/health`, model-list) stay open; full no-op when `AUTH_API_KEY` unset → Task 5 (`test_auth_off_when_env_unset`, `test_auth_on_rejects_missing_and_bad_bearer`, `test_auth_on_accepts_good_bearer_and_keeps_public_routes_open`). Constant-time compare + comma-separated keys mirror restmcp. ✓
- **No vendor type leaks inward:** `test_agent_only_ever_receives_neutral_messages` asserts the live agent's recorded `seen` list is all `Message` and contains no dicts; adapters never hand the agent a raw body or a `ParsedRequest` (only `pr.messages`). ✓
- **FastAPI/ASGI not Flask:** `Server` builds a `FastAPI` app, SSE via Starlette `StreamingResponse`, `serve()` via `uvicorn.run` — wire shapes translated from olympus's Flask handler. ✓
- **Hermetic tests:** all via `fastapi.testclient.TestClient` + `tests/_server_fakes.py` plain-`Agent` fakes; no network, no API keys, no provider SDK. ✓
- **`server` extra deps:** `fastapi`, `uvicorn[standard]`, `httpx` added (pydantic already present); `all` kept in sync → Task 1. ✓
- **Exports** `Server`, `ProtocolAdapter`, `OpenAIAdapter`, `AnthropicAdapter`, `ParsedRequest` from `aixon` (behind an optional-import guard so a bare install still imports cleanly) → Task 5. ✓

**Placeholder scan:** No `TODO`/`TBD`/`pass`-as-stub/"similar to above" left. Every code block is complete and runnable; every test has explicit assertions. The lone `try/except ImportError` in `__init__.py` is a deliberate optional-import guard, not a placeholder. ✓

**Type consistency vs contract §4:**
- `ParsedRequest(model: str, messages: list[Message], params: dict, stream: bool)` — exact. ✓
- `ProtocolAdapter` methods match the contract verbatim: `parse_request(self, body, *, path)`, `format_response(self, *, model, message, usage)`, `format_stream_chunk(self, *, model, chunk)`, `format_stream_done(self, *, model)`, `format_models(self, agents)`, `routes(self)`; `name` class attr. ✓
- `Server.__init__(adapters=None)` default `[OpenAIAdapter()]`, `@property app`, `serve(host, port)`, `@classmethod get_instance` — match §4.3. (`port` default 8000 per contract.) ✓
- Neutral types consumed exactly as Plan 1 defines them (`Message.role/content/name/tool_call_id/reasoning`, `Chunk.content/reasoning/done`); `agent.name`/`agent.aliases`/`agent.owned_by` read via the Plan 1 `Agent` surface; resolution via `get_registry().resolve` / listing via `.public()`. ✓

**Resolved ambiguities (recorded inline at the point of decision):**
1. Contract references `tests/_fakes.py` (Plan 2) which does not exist in the repo, and Plan 2 itself is written-but-unmerged; this plan defines its own `tests/_server_fakes.py` plain-`Agent` fakes so Plan 5 is hermetic and independent of Plan 2 (Task 2 header note).
2. `format_stream_chunk`/`format_stream_done` take only `model` (fixed contract signatures), so the OpenAI stream `id`/`created` are generated per line; clients key off `object`/`choices`, so non-constant ids are conformant (Task 3 note).
3. Anthropic has no real `/v1/models`; the adapter serves a thin Anthropic-flavored listing on `GET /v1/models` so the Server's "model-list is public" rule has a route to attach (Task 4 note).
4. `usage` is `{}` for now — token counting is out of Plan 5 scope and depends on a provider object the neutral boundary hides; the envelope key stays present (Task 5 note).
5. Server surface is exported from top-level `aixon` behind a `try/except ImportError` guard, honoring both "export from `aixon`" and "server deps live in an extra" (Task 5 note).
