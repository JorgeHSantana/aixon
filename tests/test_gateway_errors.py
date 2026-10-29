# tests/test_gateway_errors.py
"""0.1.28 — GatewayBudgetExceeded / GatewayRateLimited and
`classify_gateway_error`. Everything here is offline: the SDK errors are built
by hand (`tests/_fakes.make_gateway_status_error`) and no request is sent."""
from __future__ import annotations

import threading

import pytest

from aixon.exceptions import (
    AixonError,
    GatewayBudgetExceeded,
    GatewayRateLimited,
    classify_gateway_error,
)
from tests._fakes import BUDGET, make_gateway_status_error as status_error

pytestmark = pytest.mark.usefixtures("gateway_base_url")


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        # 422 mentioning "budget" -> out of budget
        pytest.param(status_error(422, BUDGET), GatewayBudgetExceeded, id="422-budget-message"),
        pytest.param(
            status_error(422, "Budget has been exceeded! Current cost: 5.1"),
            GatewayBudgetExceeded,
            id="422-budget-prose",
        ),
        pytest.param(status_error(422, "BUDGET"), GatewayBudgetExceeded, id="422-budget-uppercase"),
        pytest.param(GatewayBudgetExceeded("sem cota"), GatewayBudgetExceeded, id="own-budget"),
        # 429 -> rate limited
        pytest.param(
            status_error(429, "Rate limit reached for tier-x"), GatewayRateLimited, id="429"
        ),
        pytest.param(GatewayRateLimited("devagar"), GatewayRateLimited, id="own-rate-limited"),
        # everything else is not a gateway verdict
        pytest.param(
            status_error(422, "Unprocessable Entity: 'tools' is invalid"),
            None,
            id="422-without-budget",
        ),
        pytest.param(status_error(400, BUDGET), None, id="400-budget"),
        pytest.param(status_error(500, BUDGET), None, id="500-budget"),
        pytest.param(ValueError("429 budget"), None, id="foreign-exception"),
    ],
)
def test_classify_matrix(exc, expected):
    assert classify_gateway_error(exc) is expected


def test_only_gateway_requests_classify(monkeypatch):
    # a 429/422 from any other API (here: OpenAI itself) is none of our business
    other = "https://api.openai.com/v1/chat/completions"
    assert classify_gateway_error(status_error(429, "slow", url=other)) is None
    assert classify_gateway_error(status_error(422, BUDGET, url=other)) is None

    # trailing slash on the env is irrelevant
    monkeypatch.setenv("AIXON_GATEWAY_BASE_URL", "http://gateway.test/v1/")
    assert classify_gateway_error(status_error(429, "slow")) is GatewayRateLimited

    # no gateway configured: no SDK error counts, aixon's own types still do
    monkeypatch.delenv("AIXON_GATEWAY_BASE_URL")
    assert classify_gateway_error(status_error(429, "slow")) is None
    assert classify_gateway_error(GatewayRateLimited("devagar")) is GatewayRateLimited

    # httpx lowercases the host and drops the default port of the request URL;
    # the env gets the same normalization
    monkeypatch.setenv("AIXON_GATEWAY_BASE_URL", "http://GATEWAY.test:80/v1")
    assert classify_gateway_error(status_error(429, "slow")) is GatewayRateLimited

    # an invalid URL in the env must not raise: it just matches nothing
    monkeypatch.setenv("AIXON_GATEWAY_BASE_URL", "http://gateway.test:abc/v1")
    assert classify_gateway_error(status_error(429, "slow")) is None


def test_base_requires_path_boundary():
    # "/v10" is not under "/v1": the base must be followed by "/"
    sibling = "http://gateway.test/v10/chat/completions"
    assert classify_gateway_error(status_error(429, "slow", url=sibling)) is None
    assert classify_gateway_error(status_error(422, BUDGET, url=sibling)) is None


@pytest.mark.parametrize("base", ["https://", "http://", "http:///v1", "gateway.test/v1"],
                         ids=["https-only", "http-only", "no-host", "no-scheme"])
def test_base_without_scheme_or_host_matches_nothing(monkeypatch, base):
    monkeypatch.setenv("AIXON_GATEWAY_BASE_URL", base)
    for url in ("http://gateway.test/v1/chat/completions",
                "https://api.openai.com/v1/chat/completions"):
        assert classify_gateway_error(status_error(429, "slow", url=url)) is None


def test_status_error_without_a_request_never_raises():
    # the SDK always sets `.request`; a hand-built or mangled error may not
    missing = status_error(429, "slow")
    del missing.request
    nulled = status_error(429, "slow")
    nulled.request = None

    for broken in (missing, nulled):
        assert classify_gateway_error(broken) is None
        # the walk goes on past it: a gateway error behind it still counts
        broken.__cause__ = status_error(422, BUDGET)
        assert classify_gateway_error(broken) is GatewayBudgetExceeded


def test_status_error_whose_str_raises_never_raises():
    import openai

    class Mangled(openai.UnprocessableEntityError):
        def __str__(self) -> str:
            raise RuntimeError("no str for you")

    good = status_error(422, BUDGET)
    mangled = Mangled("x", response=good.response, body=good.body)
    assert classify_gateway_error(mangled) is None
    mangled.__cause__ = status_error(429, "slow")
    assert classify_gateway_error(mangled) is GatewayRateLimited


def test_openai_import_that_fails_otherwise_never_raises(monkeypatch):
    import builtins

    sdk_error = status_error(429, "slow")
    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "openai":
            raise RuntimeError("the SDK blew up while importing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    assert classify_gateway_error(sdk_error) is None
    assert classify_gateway_error(GatewayRateLimited("devagar")) is GatewayRateLimited


def test_without_the_env_the_request_is_never_touched(monkeypatch):
    # who does not use the gateway pays nothing: not even a look at the URL
    class Untouchable:
        @property
        def url(self):
            raise AssertionError("read the request URL with no gateway configured")

    monkeypatch.delenv("AIXON_GATEWAY_BASE_URL")
    exc = status_error(429, "slow")
    exc.request = Untouchable()
    assert classify_gateway_error(exc) is None


def test_suppressed_context_is_not_walked():
    # `raise X from None` asks to drop what was being handled: it is not a
    # gateway verdict any more
    try:
        try:
            raise status_error(429, "slow")
        except Exception:
            raise RuntimeError("degradado de propósito") from None
    except RuntimeError as wrapped:
        assert wrapped.__context__ is not None and wrapped.__suppress_context__
        assert classify_gateway_error(wrapped) is None

    # an explicit cause still counts, and the context it suppresses does not
    try:
        try:
            raise status_error(429, "slow")
        except Exception:
            raise RuntimeError("outro motivo") from status_error(422, BUDGET)
    except RuntimeError as wrapped:
        assert classify_gateway_error(wrapped) is GatewayBudgetExceeded
        wrapped.__cause__ = ValueError("not the gateway")
        assert classify_gateway_error(wrapped) is None


def test_budget_found_in_body_only():
    exc = status_error(422, "Error code: 422", body={"error": {"type": "budget_exceeded"}})
    assert "budget" not in str(exc).lower()
    assert classify_gateway_error(exc) is GatewayBudgetExceeded


def test_walks_explicit_cause_and_implicit_context():
    # explicit: `raise ... from <gateway error>`, outside any `except`, links
    # the two through __cause__ alone
    try:
        raise AixonError("subagente falhou") from status_error(429, "slow")
    except AixonError as wrapped:
        assert wrapped.__context__ is None
        assert classify_gateway_error(wrapped) is GatewayRateLimited

    # implicit: raising inside the `except` block links them through
    # __context__ alone
    try:
        try:
            raise status_error(422, BUDGET)
        except Exception:
            raise RuntimeError("falha ao tratar o erro")
    except RuntimeError as wrapped:
        assert wrapped.__cause__ is None
        assert classify_gateway_error(wrapped) is GatewayBudgetExceeded


def test_cycle_safe():
    a, b = ValueError("a"), ValueError("b")
    a.__context__, b.__context__ = b, a
    # A walk that loops on the cycle must fail this test, not hang the run
    # (the suite has no pytest-timeout): run it on a daemon thread.
    outcome: list[type[AixonError] | None] = []
    walker = threading.Thread(
        target=lambda: outcome.append(classify_gateway_error(a)), daemon=True
    )
    walker.start()
    walker.join(timeout=5)
    assert not walker.is_alive(), "classify_gateway_error does not stop on a cycle"
    assert outcome == [None]


def test_looks_inside_exception_groups():
    group = ExceptionGroup("tools", [ValueError("x"), status_error(429, "slow")])
    assert classify_gateway_error(group) is GatewayRateLimited


def test_gateway_errors_are_aixon_errors():
    assert issubclass(GatewayBudgetExceeded, AixonError)
    assert issubclass(GatewayRateLimited, AixonError)
    assert GatewayRateLimited("devagar").message == "devagar"
