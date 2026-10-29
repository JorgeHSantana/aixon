"""Exception hierarchy for aixon. Every error subclasses ``AixonError`` and
carries a human-readable ``message``."""

import os
from typing import Any


class AixonError(Exception):
    """Base exception for aixon."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class NamingError(AixonError):
    """A subclass violated a required class-name suffix."""


class RegistrationError(AixonError):
    """An agent could not be registered (duplicate name or alias clash)."""


class AgentNotFoundError(AixonError):
    """No registered agent matches the requested name or alias."""


class CompositionCycleError(AixonError):
    """A cycle was detected in the agent composition graph (A uses B as a
    tool and B uses A, directly or transitively)."""


class GatewayBudgetExceeded(AixonError):
    """The gateway's budget is spent (HTTP 422 mentioning "budget").

    aixon does not raise it: the gateway's answer reaches the caller as the
    SDK's ``openai.UnprocessableEntityError``, and ``classify_gateway_error(exc)``
    returns this class for it. A tool may raise it to signal a spent quota."""


class GatewayRateLimited(AixonError):
    """The gateway is rate-limiting the caller (HTTP 429).

    aixon does not raise it: the gateway's answer reaches the caller as the
    SDK's ``openai.RateLimitError``, and ``classify_gateway_error(exc)`` returns
    this class for it. A tool may raise it to signal a rate limit."""


class ImposedModelViolation(AixonError):
    """Code asked an ``LLM`` for the model it was configured with while another
    model is imposed on the call (``aixon.runtime.imposed_model_scope``).
    Raised instead of silently running the call on the configured model."""


def _gateway_base() -> str | None:
    """``AIXON_GATEWAY_BASE_URL`` as httpx normalizes a request URL (lowercase
    host, no default port), without the final ``/``. ``None`` when it is unset,
    malformed, or lacks a scheme or a host (a bare ``"https://"`` would
    otherwise be the prefix of every request of that scheme)."""
    raw = os.getenv("AIXON_GATEWAY_BASE_URL", "").strip()
    if not raw:
        return None
    try:
        import httpx

        url = httpx.URL(raw)
        if not url.scheme or not url.host:
            return None
        return str(url).rstrip("/")
    except Exception:  # no httpx, or httpx.InvalidURL: classifying never raises
        return None


def _sdk_verdict(node: BaseException, openai: Any, base: str) -> type[AixonError] | None:
    """The verdict of ``node`` if it is an ``openai.APIStatusError`` of a
    request under ``base`` (the base followed by ``/``)."""
    if not isinstance(node, openai.APIStatusError):
        return None
    request = getattr(node, "request", None)
    if request is None or not str(request.url).startswith(base + "/"):
        return None
    if node.status_code == 429:
        return GatewayRateLimited
    if node.status_code == 422 and "budget" in (str(node) + " " + str(node.body)).lower():
        return GatewayBudgetExceeded
    return None


def classify_gateway_error(exc: BaseException) -> type[AixonError] | None:
    """The gateway error ``exc`` stands for, or ``None`` if it is not one.

    aixon itself never raises ``GatewayBudgetExceeded`` / ``GatewayRateLimited``:
    a refusal of the gateway reaches the caller as the SDK's own error
    (``openai.UnprocessableEntityError``, ``openai.RateLimitError``), and this
    function says which of the two it stands for.

    Walks ``exc``, then its ``__cause__``, its ``__context__`` (unless
    ``__suppress_context__`` is set, as ``raise ... from None`` does: that is
    an explicit request to drop it) and the members of exception groups,
    depth first and each node once (cycles are harmless). The first node that
    matches wins:

    * an instance of ``GatewayBudgetExceeded`` / ``GatewayRateLimited`` (which
      a tool may raise to signal a quota): its class;
    * an ``openai.APIStatusError`` raised by a request to the gateway
      (``AIXON_GATEWAY_BASE_URL``): 429 is ``GatewayRateLimited``; 422 with
      "budget" in the message or body (any case) is ``GatewayBudgetExceeded``.

    Errors from any other API never match, so a 429 from OpenAI itself keeps
    behaving as it always did. Safe to call from an error handler: it never
    raises. A missing or malformed gateway URL just means that no SDK error
    matches (and without it no request URL is even read), and a node that
    cannot be inspected is skipped.
    """
    base = _gateway_base()
    openai: Any = None
    if base is not None:
        # Lazy: a top-level import would make every `import aixon` load the SDK.
        try:
            import openai as sdk

            openai = sdk
        except Exception:  # no SDK, or one that fails to import: only aixon's types match
            openai = None

    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))

        if isinstance(node, GatewayBudgetExceeded):
            return GatewayBudgetExceeded
        if isinstance(node, GatewayRateLimited):
            return GatewayRateLimited
        if openai is not None and base is not None:
            try:
                verdict = _sdk_verdict(node, openai, base)
            except Exception:  # a mangled error (no request, a __str__ that raises...)
                verdict = None
            if verdict is not None:
                return verdict

        children = [node.__cause__]
        if not getattr(node, "__suppress_context__", False):
            children.append(node.__context__)
        if isinstance(node, BaseExceptionGroup):
            children.extend(node.exceptions)
        # `pending` is a stack: push in reverse so the cause is visited first.
        pending.extend(c for c in reversed(children) if c is not None)
    return None
