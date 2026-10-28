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
