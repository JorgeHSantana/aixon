"""Per-request runtime context for the aixon server.

Generation params (temperature, max_tokens, ...) arrive on each HTTP request but
the Agent public API is ``invoke(messages)`` / ``stream(messages)`` — fixed and
shared. Rather than thread params through every signature, the Server publishes
them on a ContextVar around the agent call and the LLM reads them at call time
and binds them onto the chat model. ContextVar (not thread-local) so it composes
with async + LangGraph, mirroring aixon.reasoning."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

# Wire params we forward to the provider. Everything else in the request body
# (thought_stream_mode, stream_options, user, n, ...) is NOT a generation knob.
GENERATION_PARAMS = frozenset(
    {"temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty", "stop"}
)

_gen_params: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "aixon_generation_params", default={}
)


def current_generation_params() -> dict:
    """Return the generation params active for the current request (or {})."""
    return _gen_params.get()


@contextmanager
def generation_params(params: dict | None) -> Iterator[dict]:
    """Activate allow-listed generation params for the duration of the block."""
    filtered = {
        k: v for k, v in (params or {}).items() if k in GENERATION_PARAMS and v is not None
    }
    token = _gen_params.set(filtered)
    try:
        yield filtered
    finally:
        _gen_params.reset(token)
