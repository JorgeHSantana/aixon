"""Exception hierarchy for aixon. Every error subclasses ``AixonError`` and
carries a human-readable ``message``."""

import os


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
    """The gateway's budget is spent (HTTP 422 mentioning "budget")."""


class GatewayRateLimited(AixonError):
    """The gateway is rate-limiting the caller (HTTP 429)."""


def _is_gateway_url(url: str) -> bool:
    """Is ``url`` under ``AIXON_GATEWAY_BASE_URL``? Never, if the env is unset."""
    raw = os.getenv("AIXON_GATEWAY_BASE_URL", "").strip()
    if not raw:
        return False
    try:
        import httpx

        # httpx lowercases the host and drops the default port of the request
        # URL, so the env gets the same normalization before the comparison.
        base = str(httpx.URL(raw)).rstrip("/")
    except Exception:  # no httpx, or httpx.InvalidURL: classifying never raises
        base = raw.rstrip("/")
    return url.startswith(base + "/")


def classify_gateway_error(exc: BaseException) -> type[AixonError] | None:
    """The gateway error ``exc`` stands for, or ``None`` if it is not one.

    Walks ``exc``, then its ``__cause__``, its ``__context__`` and the members
    of exception groups, depth first and each node once (cycles are harmless).
    The first node that matches wins:

    * an instance of ``GatewayBudgetExceeded`` / ``GatewayRateLimited``: its class;
    * an ``openai.APIStatusError`` raised by a request to the gateway
      (``AIXON_GATEWAY_BASE_URL``): 429 is ``GatewayRateLimited``; 422 with
      "budget" in the message or body (any case) is ``GatewayBudgetExceeded``.

    Errors from any other API never match, so a 429 from OpenAI itself keeps
    behaving as it always did. Safe to call from an error handler: a missing
    or malformed gateway URL just means that no SDK error matches.
    """
    # Lazy: a top-level import would make every `import aixon` load the SDK.
    try:
        import openai
    except ImportError:  # no SDK installed: only aixon's own types can match
        openai = None  # type: ignore[assignment]

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
        if (
            openai is not None
            and isinstance(node, openai.APIStatusError)
            and _is_gateway_url(str(node.request.url))
        ):
            if node.status_code == 429:
                return GatewayRateLimited
            if node.status_code == 422 and "budget" in (
                str(node) + " " + str(node.body)
            ).lower():
                return GatewayBudgetExceeded

        children = [node.__cause__, node.__context__]
        if isinstance(node, BaseExceptionGroup):
            children.extend(node.exceptions)
        # `pending` is a stack: push in reverse so the cause is visited first.
        pending.extend(c for c in reversed(children) if c is not None)
    return None
