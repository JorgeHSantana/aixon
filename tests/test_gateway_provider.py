# tests/test_gateway_provider.py
"""0.1.28 — the `gateway` provider and the request's e-mail on a ContextVar.

The provider builds a REAL `langchain_openai.ChatOpenAI` on a REAL `openai`
client; only the HTTP transport is swapped (`tests/_fakes.FakeGatewayWire`),
so the URL, headers and body asserted below are what the SDK puts on the
wire. Offline: nothing resolves `gateway.test`.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import subprocess
import sys
from pathlib import Path

import openai
import pytest

from aixon.exceptions import (
    AixonError,
    GatewayBudgetExceeded,
    GatewayRateLimited,
    classify_gateway_error,
)
from aixon.llm import LLM
from aixon.message import Message
from aixon.providers.base import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT_S, get_provider
from aixon.providers.gateway import DEFAULT_USER_HEADER
from aixon.runtime import current_user_email, user_email_scope
from tests._fakes import FakeGatewayWire

USER = [Message(role="user", content="oi")]


@pytest.fixture
def gw_env(monkeypatch, gateway_base_url):
    """A configured gateway, plus the OPENAI_* variables the provider must
    never read: any of them showing up in the built model is a leak."""
    monkeypatch.setenv("AIXON_GATEWAY_API_KEY", "gw-test-key")
    monkeypatch.delenv("AIXON_GATEWAY_USER_HEADER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-must-not-leak")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai-base-url.test/v1")
    monkeypatch.setenv("OPENAI_API_BASE", "http://openai-api-base.test/v1")


# ── build ─────────────────────────────────────────────────────────────────────

def test_build_ignores_openai_env(gw_env):
    m = get_provider("gateway").build("tier-x")

    assert m.openai_api_key.get_secret_value() == "gw-test-key"
    assert m.root_client.api_key == m.root_async_client.api_key == "gw-test-key"
    assert str(m.root_client.base_url).rstrip("/") == "http://gateway.test/v1"
    assert str(m.root_async_client.base_url).rstrip("/") == "http://gateway.test/v1"
    assert m.use_responses_api is False
    assert (m.request_timeout, m.max_retries) == (DEFAULT_TIMEOUT_S, DEFAULT_MAX_RETRIES)


def test_only_by_explicit_provider():
    assert get_provider("gateway").name == "gateway"
    # no inference rule: a model name never selects the gateway
    with pytest.raises(AixonError, match="Cannot infer"):
        LLM("tier-x").chat_model


@pytest.mark.parametrize("value", [None, ""], ids=["absent", "empty"])
@pytest.mark.parametrize("env", ["AIXON_GATEWAY_BASE_URL", "AIXON_GATEWAY_API_KEY"])
def test_missing_env_names_it(gw_env, monkeypatch, env, value):
    if value is None:
        monkeypatch.delenv(env)
    else:
        monkeypatch.setenv(env, value)

    with pytest.raises(AixonError, match=env) as caught:
        get_provider("gateway").build("tier-x")
    # and it says why OPENAI_API_KEY (set by gw_env) was not used instead
    assert "OPENAI_API_KEY" in str(caught.value)


def test_reasoning_effort_verbatim_and_no_prediction(gw_env):
    provider = get_provider("gateway")

    # "xhigh" is not in aixon's low/medium/high table: the gateway decides
    assert provider.build("tier-x", reasoning={"effort": "low"}).reasoning_effort == "low"
    assert provider.build("tier-x", reasoning={"effort": "xhigh"}).reasoning_effort == "xhigh"
    assert provider.build("tier-x").reasoning_effort is None
    assert getattr(provider, "supports_prediction", False) is False


def test_reasoning_effort_reaches_the_wire(monkeypatch):
    # Chat Completions shape (a flat `reasoning_effort`), not the Responses
    # API's `reasoning: {...}`: the gateway only speaks the former.
    wire = FakeGatewayWire().install(monkeypatch)

    LLM("tier-x", provider="gateway", reasoning={"effort": "xhigh"}).complete(USER)

    assert json.loads(wire.requests[0].content)["reasoning_effort"] == "xhigh"
    assert wire.requests[0].url.path == "/v1/chat/completions"


def test_importing_aixon_does_not_load_the_openai_sdk():
    # `aixon.providers` imports every provider module; the gateway one must
    # keep langchain-openai (an optional extra) inside build(), or a bare
    # `import aixon` breaks for whoever did not install it.
    probe = (
        "import sys, aixon; "
        "print(sorted(m for m in ('openai', 'langchain_openai') if m in sys.modules))"
    )
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


# ── the e-mail header ─────────────────────────────────────────────────────────

def test_default_header_name():
    assert DEFAULT_USER_HEADER == "X-OpenWebUI-User-Email"


def test_email_header_on_sync_and_async_paths(monkeypatch):
    wire = FakeGatewayWire().install(monkeypatch)
    llm = LLM("tier-x", provider="gateway")

    with user_email_scope("ana@example.com"):
        llm.complete(USER)
        asyncio.run(llm.acomplete(USER))

    assert [r.headers.get("x-openwebui-user-email") for r in wire.requests] == [
        "ana@example.com"
    ] * 2
    assert {r.headers["authorization"] for r in wire.requests} == {"Bearer gw-test-key"}
    assert {str(r.url) for r in wire.requests} == {"http://gateway.test/v1/chat/completions"}
    assert json.loads(wire.requests[0].content)["model"] == "tier-x"


def test_email_header_on_streaming_paths(monkeypatch):
    wire = FakeGatewayWire().install(monkeypatch)
    wire.queue(body=wire.completion("sync"))
    wire.queue(body=wire.completion("async"))
    llm = LLM("tier-x", provider="gateway")

    async def collect() -> str:
        return "".join([chunk.content async for chunk in llm.astream(USER)])

    with user_email_scope("ana@example.com"):
        streamed = ["".join(chunk.content for chunk in llm.stream(USER)), asyncio.run(collect())]

    assert streamed == ["sync", "async"]
    assert [json.loads(r.content)["stream"] for r in wire.requests] == [True, True]
    assert [r.headers.get("x-openwebui-user-email") for r in wire.requests] == [
        "ana@example.com"
    ] * 2


def test_concurrent_requests_keep_their_own_email(monkeypatch):
    # One LLM, hence one cached model and one HTTP client, serves two users at
    # once, as an Agent does. The header must come from the CALLER's context at
    # send time: baked in when the model was built, the second user would be
    # billed as the first.
    wire = FakeGatewayWire().install(monkeypatch)
    wire.delay = 0.02  # keep both requests in flight together
    llm = LLM("tier-x", provider="gateway")

    async def ask(email: str) -> None:
        with user_email_scope(email):
            await llm.acomplete([Message(role="user", content=email)])

    async def both() -> None:
        await asyncio.gather(ask("ana@example.com"), ask("bob@example.com"))

    asyncio.run(both())

    sent = {
        json.loads(r.content)["messages"][0]["content"]: r.headers["x-openwebui-user-email"]
        for r in wire.requests
    }
    assert sent == {"ana@example.com": "ana@example.com", "bob@example.com": "bob@example.com"}


@pytest.mark.parametrize("email", [None, "", "   "], ids=["none", "empty", "blank"])
def test_no_email_no_header(monkeypatch, email):
    wire = FakeGatewayWire().install(monkeypatch)
    llm = LLM("tier-x", provider="gateway")

    with user_email_scope(email):
        llm.complete(USER)
        asyncio.run(llm.acomplete(USER))
    llm.complete(USER)  # and outside any scope at all

    assert len(wire.requests) == 3
    assert all("x-openwebui-user-email" not in r.headers for r in wire.requests)


def test_custom_header_name(monkeypatch):
    wire = FakeGatewayWire().install(monkeypatch)
    monkeypatch.setenv("AIXON_GATEWAY_USER_HEADER", "X-End-User")
    llm = LLM("tier-x", provider="gateway")

    with user_email_scope("ana@example.com"):
        llm.complete(USER)
        asyncio.run(llm.acomplete(USER))

    assert len(wire.requests) == 2
    for request in wire.requests:
        assert request.headers["x-end-user"] == "ana@example.com"
        assert "x-openwebui-user-email" not in request.headers


def test_empty_header_env_disables_forwarding(gw_env, monkeypatch):
    monkeypatch.setenv("AIXON_GATEWAY_USER_HEADER", "")

    m = get_provider("gateway").build("tier-x")

    assert m.http_client is None and m.http_async_client is None


@pytest.mark.parametrize("fixture, model", [
    ("fake_openai", "gpt-5.4"),
    ("fake_xai", "grok-4.5"),
    ("fake_zai", "glm-5.2"),
], ids=["openai", "xai", "zai"])
def test_other_providers_never_get_identity(request, fixture, model):
    # The other OpenAI-compatible providers build their client the usual way,
    # with the e-mail in scope: nothing here may carry it to a third party.
    provider = request.getfixturevalue(fixture)

    with user_email_scope("ana@example.com"):
        kwargs = provider.build(model).kwargs

    assert not {"http_client", "http_async_client", "default_headers"} & kwargs.keys()


# ── what comes back over the wire ─────────────────────────────────────────────

@pytest.mark.parametrize("status, body, error, expected", [
    (
        422,
        {"error": {"message": "ExceededBudget: End User=ana@example.com over budget."}},
        openai.UnprocessableEntityError,
        GatewayBudgetExceeded,
    ),
    (
        429,
        {"error": {"message": "Rate limit reached for tier-x"}},
        openai.RateLimitError,
        GatewayRateLimited,
    ),
], ids=["422-budget", "429-rate-limit"])
def test_wire_errors_classify(monkeypatch, status, body, error, expected):
    wire = FakeGatewayWire().install(monkeypatch)
    wire.queue(status, body)
    wire.queue(status, body)
    llm = LLM("tier-x", provider="gateway", max_retries=0)

    with pytest.raises(error) as sync_error:
        llm.complete(USER)
    with pytest.raises(error) as async_error:
        asyncio.run(llm.acomplete(USER))

    assert classify_gateway_error(sync_error.value) is expected
    assert classify_gateway_error(async_error.value) is expected


def test_tool_calls_come_back_as_json_and_as_sse(monkeypatch):
    wire = FakeGatewayWire().install(monkeypatch)
    call = {"name": "get_weather", "args": {"city": "Recife"}}
    wire.queue(body=wire.completion(tool_calls=[call]))
    wire.queue(body=wire.completion(tool_calls=[call]))
    llm = LLM("tier-x", provider="gateway")
    weather = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather of a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }

    answered = llm.complete(USER, tools=[weather])
    streamed = [c.tool_calls for c in llm.stream(USER, tools=[weather]) if c.tool_calls]

    expected = [("get_weather", {"city": "Recife"}, "call_1")]
    assert [(t["name"], t["args"], t["id"]) for t in answered.tool_calls] == expected
    assert [[(t["name"], t["args"], t["id"]) for t in calls] for calls in streamed] == [expected]
    # one answered as JSON, one as SSE: both shapes of the fake were parsed
    assert [bool(json.loads(r.content).get("stream")) for r in wire.requests] == [False, True]


# ── the ContextVar ────────────────────────────────────────────────────────────

def test_user_email_scope_normalizes_and_restores():
    assert current_user_email() is None

    with user_email_scope("  ana@example.com ") as outer:
        assert outer == "ana@example.com"
        assert current_user_email() == "ana@example.com"

        with user_email_scope(None) as inner:
            assert inner is None
            assert current_user_email() is None

        assert current_user_email() == "ana@example.com"  # the outer one is back

        with user_email_scope("   ") as blank:
            assert blank is None
            assert current_user_email() is None

    assert current_user_email() is None


def test_user_email_scope_exit_in_another_context_does_not_raise():
    # Same hazard as reasoning_channel() (#21): an abandoned async generator is
    # finalized in a different Context, where ContextVar.reset() raises
    # ValueError. Run in a Context of its own: the failed reset leaves the
    # value stale there, which must not leak into the other tests.
    def body() -> None:
        scope = user_email_scope("ana@example.com")
        scope.__enter__()
        contextvars.copy_context().run(scope.__exit__, None, None, None)  # no ValueError

        with user_email_scope("bob@example.com"):  # the var is still usable
            assert current_user_email() == "bob@example.com"

    contextvars.copy_context().run(body)
