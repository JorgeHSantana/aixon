"""Optional Langfuse observability (#26). OFF by default, zero-cost when off.

Activates when BOTH ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are
set (``LANGFUSE_HOST`` selects the instance; the SDK defaults to Langfuse
Cloud). The Server then wraps each chat request in ``observe_request``, which:

- opens one Langfuse root span per request named after the resolved agent, so
  EVERY internal model turn (workers, orchestrator routing, judge) lands in
  the SAME trace — without the span, each LangChain root run (e.g. the
  reflective judge's separate ``acomplete``) would become its own trace;
- propagates trace attributes (``user_id``/``session_id`` from the
  ``X-OpenWebUI-User-*``/``X-OpenWebUI-Chat-Id`` headers when the frontend
  forwards them) to all observations in the trace;
- publishes the Langfuse ``CallbackHandler`` through a langchain-core
  *configure hook* (``register_configure_hook``): every callback manager the
  LangChain runtime builds — graphs, chat models, langgraph fork threads —
  attaches the handler automatically, deduped by identity. NO aixon call
  site changes; the same mechanism LangSmith's ``tracing_v2_enabled`` uses.
  The ContextVar is only ever ``.set()`` in the request's root context
  (never inside a langgraph fork), so reads propagate into forks via
  ``copy_context`` — the ``tool_call_cache`` lesson.

Per-model aggregation comes for free: the handler records each generation
with the provider model that served it (the worker's model, the judge's
model, ...), not the wire ``model`` field (which is the agent name).

Failure policy: Langfuse being missing, misconfigured or unreachable must
NEVER break a request — import/creation failures disable the integration for
the process (one warning), and flush errors are swallowed (the OTel exporter
already only logs). Flush runs at the end of each request by default so
batched spans survive serverless CPU throttling (Cloud Run pauses the CPU
after the response; a background-only flush may never run);
``AIXON_LANGFUSE_FLUSH_PER_REQUEST=0`` disables it for always-on deploys."""

from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from aixon.logging import Logger

_log = Logger("aixon.observability")

# The handler published to langchain-core's configure hook. Module-level and
# request-scoped-by-value: the SAME singleton handler instance is set per
# request (the hook dedups by identity); per-request state lives in the OTel
# context, not on the handler.
_handler_var: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "aixon_langfuse_handler", default=None
)

_hook_registered = False
_handler: Any = None
_disabled = False  # sticky after an import/creation failure


def langfuse_enabled() -> bool:
    """True when the Langfuse env pair is present (host is optional)."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _ensure_handler() -> Any:
    """Lazily import langfuse and build the singleton CallbackHandler.

    Returns None (and warns ONCE) when the env pair is absent, the optional
    dependency is not installed, or client creation fails — the sticky
    ``_disabled`` latch keeps the failure path at zero cost afterwards."""
    global _hook_registered, _handler, _disabled
    if _disabled or not langfuse_enabled():
        return None
    if _handler is not None:
        return _handler
    try:
        from langchain_core.tracers.context import register_configure_hook
        from langfuse.langchain import CallbackHandler

        if not _hook_registered:
            register_configure_hook(_handler_var, inheritable=True)
            _hook_registered = True
        _handler = CallbackHandler()
    except Exception as exc:
        _disabled = True
        _log.warning(
            "Langfuse desabilitado (falha ao criar o handler — instale "
            f"aixon[langfuse] e confira as envs LANGFUSE_*): {exc}"
        )
        return None
    return _handler


def request_identity(headers: Any) -> dict:
    """Extract trace identity from request headers (case-insensitive mapping,
    e.g. Starlette's). Open WebUI sends these when
    ``ENABLE_FORWARD_USER_INFO_HEADERS=true``; absent headers simply yield
    an anonymous trace."""
    get = getattr(headers, "get", lambda *_: None)
    user = get("x-openwebui-user-email") or get("x-openwebui-user-id")
    session = get("x-openwebui-chat-id")
    return {"user_id": user, "session_id": session}


@contextmanager
def _handler_scope(handler: Any) -> Iterator[None]:
    """Publish ``handler`` to langchain-core's configure hook for the block.
    Split from ``observe_request`` so tests can drive the propagation with a
    dummy handler and no langfuse installed."""
    global _hook_registered
    if not _hook_registered:
        from langchain_core.tracers.context import register_configure_hook

        register_configure_hook(_handler_var, inheritable=True)
        _hook_registered = True
    token = _handler_var.set(handler)
    try:
        yield
    finally:
        _handler_var.reset(token)


@contextmanager
def observe_request(agent_name: str, *, user_id: str | None = None,
                    session_id: str | None = None,
                    metadata: dict | None = None) -> Iterator[bool]:
    """Wrap one chat request in a Langfuse trace (yields True) or no-op
    (yields False) when the integration is off.

    Observability must never break a request: enter failures (client
    creation, span/attribute context managers) degrade to a no-op with a
    warning, and exit/flush failures are swallowed with a warning. The
    agent's own exception still propagates untouched."""
    from contextlib import ExitStack

    handler = _ensure_handler()
    if handler is None:
        yield False
        return
    stack = ExitStack()
    try:
        from langfuse import get_client, propagate_attributes

        client = get_client()
        stack.enter_context(_handler_scope(handler))
        stack.enter_context(client.start_as_current_observation(
            name=agent_name, as_type="agent", metadata=metadata
        ))
        stack.enter_context(propagate_attributes(
            trace_name=agent_name, user_id=user_id, session_id=session_id
        ))
    except Exception as exc:
        try:
            stack.close()
        except Exception:
            pass
        _log.warning(f"Langfuse: falha ao abrir o trace (request segue sem): {exc}")
        yield False
        return
    try:
        yield True
    finally:
        try:
            stack.close()
        except Exception as exc:
            _log.warning(f"Langfuse: falha ao fechar o trace: {exc}")
        if os.getenv("AIXON_LANGFUSE_FLUSH_PER_REQUEST", "1") != "0":
            try:
                client.flush()
            except Exception as exc:
                _log.warning(f"Langfuse: flush falhou (spans podem atrasar): {exc}")
