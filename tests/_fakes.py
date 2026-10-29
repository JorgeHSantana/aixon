# tests/_fakes.py
"""Single owner of hermetic test doubles for aixon (contract §9.1).

Imported by Plan 2 tests and by Plans 3, 4, 5, 7. DO NOT redefine these
elsewhere. Everything here is offline: no API key, no network.

FakeChatModel is copied VERBATIM from the interface contract §9.1 and is
validated against langchain 1.3 / langchain-core 1.4 / langgraph 1.2 — it
drives langchain.agents.create_agent through a tool call then a final answer.
"""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aixon.agent import Agent
from aixon.message import Chunk, Message
from aixon.providers.base import Provider, register_provider

if TYPE_CHECKING:
    import httpx
    import openai
    import pytest

    from aixon.llm import LLM


FAKE_MODEL = "fake-1"
FAKE_PROVIDER = "fake"


# ── FakeChatModel — VERBATIM from contract §9.1 (do not edit) ────────────────

class FakeChatModel(BaseChatModel):
    """Scriptable offline chat model. `script` is a list of AIMessages returned
    one per LLM call (set tool_calls on an AIMessage to drive a tool step)."""

    script: list = []
    _idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeChatModel":
        return self  # tools ignored; script drives calls

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        i = self._idx
        msg = self.script[i] if i < len(self.script) else AIMessage(content="(done)")
        object.__setattr__(self, "_idx", i + 1)
        return ChatResult(generations=[ChatGeneration(message=msg)])


# Example script for a tool-calling test (used by Plan 3):
#   FakeChatModel(script=[
#       AIMessage(content="", tool_calls=[{"name":"get_weather","args":{"city":"Recife"},"id":"call_1"}]),
#       AIMessage(content="The weather in Recife is sunny."),
#   ])


# ── Fake provider ────────────────────────────────────────────────────────────

class FakeProvider(Provider):
    """Provider named 'fake' whose build() returns a FakeChatModel.

    The returned model's `script` can be overridden per test, e.g.:
        from tests._fakes import make_llm
        llm = make_llm()
        llm.chat_model.script = [AIMessage(content="hi")]
    A bare FakeChatModel() with an empty script echoes "(done)" for each call,
    which is enough for the LLM.complete / LLM.stream smoke tests below.
    """

    name = FAKE_PROVIDER
    env_key = "FAKE_API_KEY"

    def build(self, model: str, **params: Any) -> FakeChatModel:
        return FakeChatModel()


def register_fake_provider() -> None:
    """Register the 'fake' provider. Idempotent — safe to call repeatedly."""
    register_provider(FakeProvider())


# Register at import time so `LLM("fake-1", provider="fake")` works for any
# importer without an explicit call.
register_fake_provider()


# ── Recording provider (stands in for the model a gateway imposes) ───────────

class RecordingChatModel(FakeChatModel):
    """FakeChatModel that also keeps the messages of every call it served:
    `calls[i]` is what its i-th `_generate` received, so a test can assert what
    an agent really sent (system prompt, history, tool results)."""

    calls: list[list[BaseMessage]] = []

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        return super()._generate(messages, stop, run_manager, **kwargs)


class RecordingProvider(Provider):
    """Provider whose `build()` records `(model, params)` in `builds` and always
    answers with the SAME `model`, a RecordingChatModel. Sharing one model means
    sharing one `script`, consumed in the order the calls happen across a whole
    agent tree, whichever agent or LLM built it. `make_recording_provider`
    registers one under a name; the `rec` fixture hands out the default one."""

    env_key = "REC_API_KEY"

    def __init__(self, name: str = "rec", script: list | None = None) -> None:
        self.name = name
        self.builds: list[tuple[str, dict]] = []
        self.model = RecordingChatModel(script=list(script or []))

    def build(self, model: str, **params: Any) -> RecordingChatModel:
        self.builds.append((model, dict(params)))
        return self.model


def make_recording_provider(
    name: str = "rec", script: list | None = None
) -> RecordingProvider:
    """A new RecordingProvider, registered as `name` (replacing any earlier one)."""
    provider = RecordingProvider(name, script)
    register_provider(provider)
    return provider


# ── Convenience factories (used by Plans 3/4/5/7) ────────────────────────────

def make_llm(**params: Any) -> "LLM":
    """Return an LLM bound to the fake provider/model."""
    register_fake_provider()
    from aixon.llm import LLM  # local import: aixon.llm depends on providers

    return LLM(FAKE_MODEL, provider=FAKE_PROVIDER, **params)


def make_echo_agent(name: str = "echo", *, hidden: bool = False):
    """Define + register a concrete Agent that echoes the last message.

    invoke([... , Message(content="x")]) -> Message(role="assistant", content="x")
    stream(...) yields one content Chunk then Chunk(done=True).
    Returns the registered agent instance. Used by server/CLI/orchestrator
    tests that need an Agent but not a real LLM.
    """
    from typing import Iterator

    def invoke(self, messages: list[Message]) -> Message:
        last = messages[-1].content if messages else ""
        return Message(role="assistant", content=last)

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        last = messages[-1].content if messages else ""
        yield Chunk(content=last)
        yield Chunk(done=True)

    cls_name = f"{name.capitalize()}Agent"
    cls = type(
        cls_name,
        (Agent,),
        {"invoke": invoke, "stream": stream, "name": name, "hidden": hidden},
    )
    # Agent.__init_subclass__ already instantiated + registered it; fetch it.
    from aixon.registry import get_registry

    return get_registry().resolve(name)


# ── Gateway status errors (OpenAI-compatible gateway, HTTP 4xx/5xx) ──────────

# What a gateway answers when an end user is over budget (HTTP 422).
BUDGET = "ExceededBudget: End User=ana@example.com over budget. Spend=1.2, Budget=1.0"


def make_gateway_status_error(
    status: int,
    message: str,
    *,
    body: dict | None = None,
    url: str = "http://gateway.test/v1/chat/completions",
) -> "openai.APIStatusError":
    """The `openai.APIStatusError` the SDK raises for an HTTP `status` answer.

    Picks the same subclass the SDK does (422 -> UnprocessableEntityError,
    429 -> RateLimitError, anything else -> APIStatusError) and builds it on a
    real `httpx.Response` for a POST to `url`. `body` is the JSON the server
    answered (default: `{"error": {"message": <message>}}`); as in the SDK,
    the error's own `.body` is its `"error"` member, and `str(error)` is
    `message`. Nothing here touches the network.
    """
    import httpx
    import openai

    payload = body if body is not None else {"error": {"message": message}}
    response = httpx.Response(status, request=httpx.Request("POST", url), json=payload)
    cls: type[openai.APIStatusError] = {
        422: openai.UnprocessableEntityError,
        429: openai.RateLimitError,
    }.get(status, openai.APIStatusError)
    return cls(message, response=response, body=payload.get("error", payload))


# ── The gateway on the wire (OpenAI-compatible HTTP, offline) ─────────────────

class FakeGatewayWire:
    """An OpenAI-compatible gateway played on the HTTP wire, not stubbed at
    the model.

    `install()` configures the `gateway` provider (URL and key set, header
    override cleared) and swaps the two transports it builds its HTTP clients
    on (`aixon.providers.gateway._transports`) for `httpx.MockTransport`s. The
    REAL `ChatOpenAI` and the REAL `openai` SDK run on top, so every request
    that would have left the process lands in `requests` exactly as the SDK
    wrote it (URL, headers, body), on the sync path and on the async one.

    Answers come from a FIFO queue (`queue()`); with the queue empty the
    gateway answers 200 `completion("ok")`. A request that asks for
    `"stream": true` gets the equivalent SSE. Nothing touches the network.
    The provider builds no custom client when the user header is turned off
    (`AIXON_GATEWAY_USER_HEADER=""`), so the wire sees nothing in that case.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        # Seconds the ASYNC transport waits before it answers (the sync one
        # answers at once): keeps requests in flight for the test.
        self.delay = 0.0
        self._queue: list[tuple[int, dict, dict]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeGatewayWire:
        import httpx

        monkeypatch.setenv("AIXON_GATEWAY_BASE_URL", "http://gateway.test/v1")
        monkeypatch.setenv("AIXON_GATEWAY_API_KEY", "gw-test-key")
        monkeypatch.delenv("AIXON_GATEWAY_USER_HEADER", raising=False)

        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return self._answer(request)

        async def ahandle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            await asyncio.sleep(self.delay)
            return self._answer(request)

        monkeypatch.setattr(
            "aixon.providers.gateway._transports",
            lambda: (httpx.MockTransport(handle), httpx.MockTransport(ahandle)),
        )
        return self

    def queue(
        self,
        status: int = 200,
        body: dict | None = None,
        headers: dict | None = None,
    ) -> None:
        """Enqueue the gateway's next answer. `body` defaults to
        `completion("ok")` for a 200 and to an OpenAI-style `{"error": ...}`
        for any other status; `headers` are extra response headers."""
        if body is None:
            body = (
                self.completion("ok")
                if status == 200
                else {"error": {"message": f"the fake gateway answered {status}"}}
            )
        self._queue.append((status, body, dict(headers or {})))

    @staticmethod
    def completion(
        content: str | None = None, tool_calls: list[dict] | None = None
    ) -> dict:
        """The body of a non-stream `chat.completion`. Each `tool_calls` item
        is `{"name": ..., "args": {...}}` and goes out in the wire shape: ids
        `call_1`..`call_n`, `arguments` as a JSON string."""
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": f"call_{n}",
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["args"]),
                    },
                }
                for n, call in enumerate(tool_calls, start=1)
            ]
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "tier-x",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def _answer(self, request: httpx.Request) -> httpx.Response:
        import httpx

        status, body, headers = (
            self._queue.pop(0) if self._queue else (200, self.completion("ok"), {})
        )
        if status == 200 and self._wants_stream(request):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream", **headers},
                content=self._sse(body),
            )
        return httpx.Response(status, json=body, headers=headers)

    @staticmethod
    def _wants_stream(request: httpx.Request) -> bool:
        try:
            payload = json.loads(request.content)
        except ValueError:
            return False
        return isinstance(payload, dict) and payload.get("stream") is True

    @staticmethod
    def _sse(body: dict) -> bytes:
        """`body` (a `completion()`) as an SSE stream: one chunk with the
        content or the tool_calls, one with the finish_reason, then [DONE]."""
        choice = body["choices"][0]
        message = choice["message"]
        delta: dict[str, Any] = {"role": "assistant"}
        if message.get("content") is not None:
            delta["content"] = message["content"]
        if message.get("tool_calls"):
            delta["tool_calls"] = [
                {"index": n, **call} for n, call in enumerate(message["tool_calls"])
            ]
        chunk = {
            "id": body["id"],
            "object": "chat.completion.chunk",
            "created": body["created"],
            "model": body["model"],
        }
        events = [
            {**chunk, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {
                **chunk,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}
                ],
            },
        ]
        sse = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        return (sse + "data: [DONE]\n\n").encode()
