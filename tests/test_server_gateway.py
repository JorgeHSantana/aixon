# tests/test_server_gateway.py
"""0.1.28 — what the Server does about the gateway: it publishes the e-mail of
the request's user for the call, and it turns a refusal by the gateway (budget
spent, rate limit) into a message the user can read — a chat message on a
stream, an HTTP 429 without one — instead of the generic server error.

Offline: the errors are built by hand (`tests/_fakes.make_gateway_status_error`)
and raised by fake agents; nothing is sent anywhere."""
from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from aixon.exceptions import AixonError
from aixon.server.adapters.anthropic import AnthropicAdapter
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.server import Server
from tests._fakes import BUDGET, make_gateway_status_error as status_error
from tests._server_fakes import make_failing, make_whoami, sse_content, sse_datas

pytestmark = pytest.mark.usefixtures("gateway_base_url")

GENERIC_STREAM_ERROR = {
    "message": "The server encountered an error while generating the response.",
    "type": "server_error",
}
GENERIC_500 = {
    "error": {"message": "The agent failed to process the request.", "type": "server_error"}
}


@pytest.fixture(autouse=True)
def _reset_server():
    Server._reset()
    yield
    Server._reset()


def _client(adapter=None) -> TestClient:
    return TestClient(
        Server(
            adapters=[adapter or OpenAIAdapter()],
            budget_exceeded_message="SEM COTA",
            rate_limited_message="DEVAGAR",
        ).app
    )


def _body(model: str, *, stream: bool, **extra) -> dict:
    return {
        "model": model,
        "stream": stream,
        "messages": [{"role": "user", "content": "q"}],
        **extra,
    }


def _wrapping(cause: BaseException) -> AixonError:
    """An ``AixonError("x")`` raised ``from cause``, as an error shield would."""
    try:
        raise AixonError("x") from cause
    except AixonError as wrapped:
        return wrapped


class _FinishAdapter(OpenAIAdapter):
    """An OpenAI adapter whose stream sessions end with the given ``finish``
    (what ``include_usage`` makes the real one do: a tail after the last chunk)."""

    def __init__(self, finish):
        super().__init__()
        self._finish = finish

    def open_stream(self, *, model, request):
        session = super().open_stream(model=model, request=request)
        session.finish = self._finish
        return session


# --- the message goes to the user, not to the error handler ----------------
def test_stream_gateway_error_closes_think_block():
    # The label of the specialist opened a <think> block before the gateway
    # refused the call: the message must come AFTER the block is closed.
    make_failing("falha", status_error(422, BUDGET))
    r = _client().post(
        "/v1/chat/completions", json=_body("falha", stream=True, thought_stream_mode="content")
    )
    assert r.status_code == 200
    assert sse_content(r.text) == "<think>\nCalling especialista...\n</think>\nSEM COTA"
    datas = sse_datas(r.text)
    assert json.loads(datas[-2])["choices"][0]["finish_reason"] == "stop"
    assert datas[-1] == "[DONE]"
    assert "server_error" not in r.text


@pytest.mark.parametrize(
    ("exc", "body"),
    [
        pytest.param(
            status_error(422, BUDGET),
            {
                "error": {
                    "message": "SEM COTA",
                    "type": "insufficient_quota",
                    "code": "budget_exceeded",
                }
            },
            id="budget",
        ),
        pytest.param(
            status_error(429, "slow"),
            {
                "error": {
                    "message": "DEVAGAR",
                    "type": "rate_limit_exceeded",
                    "code": "rate_limited",
                }
            },
            id="rate-limited",
        ),
    ],
)
def test_non_stream_gateway_error_is_429(exc, body):
    make_failing("falha", exc)
    r = _client().post("/v1/chat/completions", json=_body("falha", stream=False))
    assert r.status_code == 429
    assert r.json() == body
    # the openai SDK retries a 429 (twice, by default) unless told not to, and
    # each retry would run the whole tree again against the same refusal
    assert r.headers["x-should-retry"] == "false"


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        pytest.param(status_error(422, BUDGET), "SEM COTA", id="budget"),
        pytest.param(status_error(429, "slow"), "DEVAGAR", id="rate-limited"),
    ],
)
def test_stream_gateway_error_becomes_content(exc, message):
    make_failing("falha", exc)
    r = _client().post("/v1/chat/completions", json=_body("falha", stream=True))
    assert r.status_code == 200
    assert sse_content(r.text) == message
    datas = sse_datas(r.text)
    assert json.loads(datas[-2])["choices"][0]["finish_reason"] == "stop"
    assert datas[-1] == "[DONE]"


def test_wrapped_gateway_error_still_classified():
    make_failing("falha", _wrapping(status_error(429, "slow")))
    r = _client().post("/v1/chat/completions", json=_body("falha", stream=False))
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "rate_limited"


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(ValueError("boom"), id="value-error"),
        # a 429 of another API is none of the gateway's business
        pytest.param(
            status_error(429, "slow", url="https://api.openai.com/v1/chat/completions"),
            id="429-from-another-api",
        ),
    ],
)
def test_other_errors_unchanged(exc):
    make_failing("falha", exc)
    c = _client()

    r = c.post("/v1/chat/completions", json=_body("falha", stream=True))
    assert r.status_code == 200
    datas = sse_datas(r.text)
    events = [json.loads(d) for d in datas[:-1]]
    assert [e["error"] for e in events if "error" in e] == [GENERIC_STREAM_ERROR]
    assert datas[-1] == "[DONE]"
    assert "SEM COTA" not in r.text and "DEVAGAR" not in r.text

    r = c.post("/v1/chat/completions", json=_body("falha", stream=False))
    assert r.status_code == 500
    assert r.json() == GENERIC_500


def test_anthropic_stream_gateway_error():
    make_failing("falha", status_error(422, BUDGET))
    r = _client(AnthropicAdapter()).post(
        "/v1/messages", json=_body("falha", stream=True, max_tokens=50)
    )
    assert r.status_code == 200
    assert "event: error" not in r.text
    deltas = [
        event["delta"]
        for event in map(json.loads, sse_datas(r.text))
        if event.get("type") == "content_block_delta"
    ]
    assert {"type": "text_delta", "text": "SEM COTA"} in deltas
    assert "event: message_stop" in r.text


# --- the branch itself: tail, failure inside it, log -----------------------
def test_stream_gateway_error_emits_the_session_tail():
    make_failing("falha", status_error(422, BUDGET))
    adapter = _FinishAdapter(lambda: 'data: {"tail": true}\n\n')
    r = _client(adapter).post("/v1/chat/completions", json=_body("falha", stream=True))
    datas = sse_datas(r.text)
    assert sse_content(r.text) == "SEM COTA"
    assert json.loads(datas[-3])["choices"][0]["finish_reason"] == "stop"
    assert datas[-2:] == ['{"tail": true}', "[DONE]"]


def test_failure_inside_gateway_branch_falls_to_generic_error():
    def finish():
        raise RuntimeError("finish exploded")

    make_failing("falha", status_error(422, BUDGET))
    # TestClient re-raises whatever escapes the app: reaching the asserts means
    # the failure did not propagate to Starlette.
    r = _client(_FinishAdapter(finish)).post(
        "/v1/chat/completions", json=_body("falha", stream=True)
    )
    assert r.status_code == 200
    datas = sse_datas(r.text)
    assert sse_content(r.text) == "SEM COTA"  # what was already sent stays sent
    assert json.loads(datas[-2]) == {"error": GENERIC_STREAM_ERROR}
    assert datas[-1] == "[DONE]"
    assert "finish exploded" not in r.text


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "invoke"])
def test_gateway_error_is_logged_as_warning(caplog, stream):
    make_failing("falha", status_error(422, BUDGET))
    with caplog.at_level(logging.INFO, logger="aixon.server"):
        _client().post("/v1/chat/completions", json=_body("falha", stream=stream))
    levels = {r.levelname for r in caplog.records if r.name == "aixon.server"}
    assert "WARNING" in levels
    assert "ERROR" not in levels  # the gateway said no; the server did not fail


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "invoke"])
def test_gateway_refusal_log_carries_the_verdict_not_the_gateway_text(caplog, stream):
    # The gateway's text names the end user and the spend (LiteLLM: "End
    # User=<e-mail> ... Spend=... Budget=..."): the log keeps only the verdict.
    make_failing("falha", status_error(422, BUDGET))
    with caplog.at_level(logging.INFO, logger="aixon.server"):
        _client().post("/v1/chat/completions", json=_body("falha", stream=stream))

    for record in caplog.records:
        for secret in ("End User=", "ana@example.com", "Spend=", "Budget="):
            assert secret not in record.getMessage()
    (warning,) = [
        r for r in caplog.records if r.name == "aixon.server" and r.levelno == logging.WARNING
    ]
    message = warning.getMessage()
    # the verdict, the agent, the adapter and the HTTP status the client gets
    for part in ("GatewayBudgetExceeded", "'falha'", "openai", "HTTP 200" if stream else "HTTP 429"):
        assert part in message


# --- the messages of the singleton ------------------------------------------
def test_default_messages_exact():
    server = Server()
    assert server.budget_exceeded_message == (
        "Your usage budget has been exhausted. "
        "Please try again after it resets or contact your administrator."
    )
    assert server.rate_limited_message == (
        "Too many requests to the model right now. Please wait a moment and try again."
    )


def test_singleton_message_rules():
    first = Server(adapters=[OpenAIAdapter()], budget_exceeded_message="SEM COTA")
    # the default (a bare Server()) keeps what the first construction stored
    assert Server() is first
    assert first.budget_exceeded_message == "SEM COTA"
    # so does asking for the stored value again
    Server(budget_exceeded_message="SEM COTA")
    # a value that is neither the stored one nor the default is refused...
    with pytest.raises(AixonError, match="_reset"):
        Server(budget_exceeded_message="OUTRA")
    with pytest.raises(AixonError, match="_reset"):
        Server(rate_limited_message="OUTRA")
    # ...and nothing changed
    assert Server().budget_exceeded_message == "SEM COTA"
    assert Server().rate_limited_message != "OUTRA"


# --- the e-mail of the request's user ---------------------------------------
@pytest.mark.parametrize("stream", [False, True], ids=["invoke", "stream"])
@pytest.mark.parametrize(
    ("headers", "seen"),
    [
        pytest.param({"x-openwebui-user-email": "ana@example.com"}, "ana@example.com", id="email"),
        pytest.param({}, "None", id="no-header"),
        pytest.param({"x-openwebui-user-email": "   "}, "None", id="blank"),
        # the user-id is never an e-mail
        pytest.param({"x-openwebui-user-id": "u-1"}, "None", id="user-id-only"),
    ],
)
def test_email_scope_per_request(stream, headers, seen):
    make_whoami("quem")
    r = _client().post("/v1/chat/completions", json=_body("quem", stream=stream), headers=headers)
    assert r.status_code == 200
    answer = sse_content(r.text) if stream else r.json()["choices"][0]["message"]["content"]
    assert answer == seen
