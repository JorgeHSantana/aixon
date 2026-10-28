"""Gateway provider — an OpenAI-compatible gateway that imposes the model.

Some deployments send every LLM call through one gateway (a LiteLLM proxy, for
instance): the caller names a tier, the gateway picks the real model and meters
the spend per end user. It speaks the OpenAI Chat Completions contract, so this
provider reuses langchain_openai.ChatOpenAI pointed at it. Self-registers as
'gateway' at import time. There is no model-name inference rule: it is reached
only through ``provider="gateway"``. The langchain_openai, pydantic and httpx
imports are LAZY (inside build()) so importing this module never raises
ImportError if langchain-openai is not installed; only build() will fail then.

Environment:

    AIXON_GATEWAY_BASE_URL     the gateway's OpenAI-compatible URL (required)
    AIXON_GATEWAY_API_KEY      its key (required)
    AIXON_GATEWAY_USER_HEADER  name of the header that carries the end user's
                               e-mail (default X-OpenWebUI-User-Email); empty
                               turns the forwarding off

It NEVER reads OPENAI_API_KEY, OPENAI_BASE_URL or OPENAI_API_BASE: it hands
ChatOpenAI its own key and URL, so the SDK has nothing to fill in from those.

The end user's e-mail travels as that header on every call made while
``aixon.runtime.user_email_scope(email)`` is active. It is read when the request
is SENT, from the caller's context, not when the model is built: the LLM caches
one model, and that model serves many users. Without an e-mail no header goes
out at all; refusing such a request is the application's job, not this one's.

Both ``http_client`` and ``http_async_client`` get the hook, on purpose.
ChatOpenAI keeps one SDK client per path (sync for invoke/stream, async for
ainvoke/astream) and a lone ``http_client`` is ignored, with no error, on the
async path, which is the one the Server runs agents on. The header would never
leave, and the gateway would bill every user to the shared key.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from aixon.exceptions import AixonError
from aixon.providers.base import (
    Provider,
    apply_resilience_defaults,
    register_provider,
    resolve_reasoning_spec,
)
from aixon.runtime import current_user_email

if TYPE_CHECKING:
    import httpx
    from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_USER_HEADER = "X-OpenWebUI-User-Email"


def _transports() -> tuple[httpx.BaseTransport | None, httpx.AsyncBaseTransport | None]:
    """The transports under the sync and async HTTP clients: always
    ``(None, None)``, i.e. httpx's own, in production. It exists as a seam for
    the tests, which swap in mock transports to see what the SDK would send."""
    return None, None


def _required_env(name: str) -> str:
    # Stripped like classify_gateway_error() strips the URL, so the two read
    # AIXON_GATEWAY_BASE_URL the same way.
    value = os.getenv(name, "").strip()
    if not value:
        raise AixonError(
            f"{name} is missing or empty. Refusing to build the gateway model — "
            f"this provider never falls back to OPENAI_API_KEY, OPENAI_BASE_URL "
            f"or OPENAI_API_BASE."
        )
    return value


def _identity_clients(header: str) -> dict[str, Any]:
    """The sync/async httpx client pair that stamps the end user's e-mail, if
    any is in scope, onto every request as *header*."""
    import httpx  # lazy import

    def hook(request: httpx.Request) -> None:
        email = current_user_email()
        if email:
            request.headers[header] = email

    async def ahook(request: httpx.Request) -> None:
        # httpx.AsyncClient awaits its event hooks: a plain function would fail.
        hook(request)

    sync_transport, async_transport = _transports()
    return {
        "http_client": httpx.Client(
            transport=sync_transport, event_hooks={"request": [hook]}
        ),
        "http_async_client": httpx.AsyncClient(
            transport=async_transport, event_hooks={"request": [ahook]}
        ),
    }


class GatewayProvider(Provider):
    name = "gateway"
    env_key = "AIXON_GATEWAY_API_KEY"
    supports_reasoning = True
    # No `supports_prediction`: Predicted Outputs is an OpenAI feature, and
    # here the gateway, not the caller, picks the model that answers.

    def build(self, model: str, **params: Any) -> "BaseChatModel":
        from langchain_openai import ChatOpenAI  # lazy import
        from pydantic import SecretStr

        base_url = _required_env("AIXON_GATEWAY_BASE_URL")
        api_key = _required_env(self.env_key)
        apply_resilience_defaults(params)
        spec = resolve_reasoning_spec(params)
        if spec is not None:
            # The gateway hands the effort to whichever model it picked and is
            # the authority on the values that model accepts: forward it as is.
            params["reasoning_effort"] = spec["effort"]

        # An empty header name turns the forwarding off: plain SDK clients.
        header = os.getenv("AIXON_GATEWAY_USER_HEADER", DEFAULT_USER_HEADER)
        clients = _identity_clients(header) if header else {}

        # ChatOpenAI's `api_key` field is SecretStr | Callable | None, not a
        # bare str — wrap explicitly (api_key is non-empty, see above).
        return ChatOpenAI(
            model=model,
            api_key=SecretStr(api_key),
            base_url=base_url,
            # A gateway speaks Chat Completions; never let ChatOpenAI pick /responses.
            use_responses_api=False,
            **clients,
            **params,
        )


register_provider(GatewayProvider())
