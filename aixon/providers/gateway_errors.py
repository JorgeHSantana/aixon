"""Classificação dos erros do gateway (provedor ``gateway``).

Único ponto que lê e normaliza ``AIXON_GATEWAY_BASE_URL``; o provedor
(``aixon.providers.gateway``) e o classificador usam ``gateway_base_url``.
Contrato de cota/limite (o do proxy LiteLLM, por exemplo): 429 é limite de
taxa, 422 com "budget" na mensagem ou no corpo é cota esgotada. Importa só
``aixon.exceptions`` (nunca o contrário no topo, para não criar ciclo)."""
from __future__ import annotations

import os
from typing import Any

from aixon.exceptions import AixonError, GatewayBudgetExceeded, GatewayRateLimited

GATEWAY_BASE_URL_ENV = "AIXON_GATEWAY_BASE_URL"

# Contrato do gateway (o do LiteLLM, por exemplo): status e substring (sem caixa) que marcam cota esgotada.
BUDGET_STATUS = 422
BUDGET_MARKER = "budget"
# Contrato do gateway: status de limite de taxa.
RATE_LIMIT_STATUS = 429


def required_env_error(name: str) -> AixonError:
    """O erro de env obrigatória ausente do provedor gateway."""
    return AixonError(
        f"{name} is missing or empty. Refusing to build the gateway model — "
        f"this provider never falls back to OPENAI_API_KEY, OPENAI_BASE_URL "
        f"or OPENAI_API_BASE."
    )


def gateway_base_url(required: bool = False) -> str | None:
    """``AIXON_GATEWAY_BASE_URL`` sem espaços nas pontas; ``None`` se vazia.
    Com ``required=True``, vazia levanta o ``AixonError`` do provedor."""
    value = os.getenv(GATEWAY_BASE_URL_ENV, "").strip()
    if not value:
        if required:
            raise required_env_error(GATEWAY_BASE_URL_ENV)
        return None
    return value


def _gateway_base() -> str | None:
    """``AIXON_GATEWAY_BASE_URL`` as httpx normalizes a request URL (lowercase
    host, no default port), without the final ``/``. ``None`` when it is unset,
    malformed, or lacks a scheme or a host (a bare ``"https://"`` would
    otherwise be the prefix of every request of that scheme)."""
    raw = gateway_base_url()
    if raw is None:
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
    if node.status_code == RATE_LIMIT_STATUS:
        return GatewayRateLimited
    if node.status_code == BUDGET_STATUS and BUDGET_MARKER in (str(node) + " " + str(node.body)).lower():
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
      (``AIXON_GATEWAY_BASE_URL``): ``RATE_LIMIT_STATUS`` (429) is
      ``GatewayRateLimited``; ``BUDGET_STATUS`` (422) with ``BUDGET_MARKER``
      ("budget") in the message or body (any case) is
      ``GatewayBudgetExceeded``. That is the gateway contract (the LiteLLM
      proxy's, for example).

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
