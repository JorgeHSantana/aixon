"""Per-request runtime context for the aixon server.

Generation params (temperature, max_tokens, ...) arrive on each HTTP request but
the Agent public API is ``invoke(messages)`` / ``stream(messages)`` — fixed and
shared. Rather than thread params through every signature, the Server publishes
them on a ContextVar around the agent call and the LLM reads them at call time
and binds them onto the chat model. ContextVar (not thread-local) so it composes
with async + LangGraph, mirroring aixon.reasoning."""

from __future__ import annotations

import contextvars
import copy
from contextlib import contextmanager
from typing import Iterator

# Wire params we forward to the provider. Everything else in the request body
# (thought_stream_mode, stream_options, user, n, ...) is NOT a generation knob.
GENERATION_PARAMS = frozenset(
    {
        "temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty",
        "stop",
        # Per-request override of the declarative `reasoning` knob (R1 rule 6):
        # each provider's build() pops it via resolve_reasoning_spec and
        # translates {"effort": reasoning_effort} the same way as any other
        # effort-only spec, taking precedence over the class-level knob for
        # this one build.
        "reasoning_effort",
    }
)

# Provider-dialect spellings normalized to the canonical key above so downstream
# models always receive kwargs every LangChain chat model understands. The
# canonical key wins when both are present.
_PARAM_ALIASES = {
    "max_completion_tokens": "max_tokens",  # modern OpenAI clients
    "stop_sequences": "stop",               # Anthropic dialect
}

# default=None (not a mutable {}) so no shared dict can leak across contexts.
_gen_params: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "aixon_generation_params", default=None
)


def current_generation_params() -> dict:
    """Return the generation params active for the current request (or {}).

    Always a fresh copy: mutating the returned dict never pollutes the
    ContextVar state."""
    params = _gen_params.get()
    return dict(params) if params else {}


@contextmanager
def generation_params(params: dict | None) -> Iterator[dict]:
    """Activate allow-listed generation params for the duration of the block."""
    raw = params or {}
    filtered = {k: v for k, v in raw.items() if k in GENERATION_PARAMS and v is not None}
    for alias, canonical in _PARAM_ALIASES.items():
        value = raw.get(alias)
        if value is not None and canonical not in filtered:
            filtered[canonical] = value
    token = _gen_params.set(filtered)
    try:
        yield filtered
    finally:
        _gen_params.reset(token)


# Tool definitions the CLIENT declared on the request (agentic clients bring
# their own tools and expect tool_calls back — e.g. editors, IDEs). Published
# per request like generation params; agents that support client-executed
# tools read them via current_client_tools(), everyone else ignores them.
_client_tools: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "aixon_client_tools", default=None
)


def current_client_tools() -> list[dict]:
    """Return the client-declared tool definitions for the current request
    (or ``[]``). Always a DEEP copy: a shallow ``dict(t)`` still shares the
    nested ``function`` dict with the ContextVar's stored value, so mutating
    ``result[0]["function"]`` would silently corrupt state seen by later reads
    (and by concurrent requests/tasks sharing the same underlying tool def)."""
    tools = _client_tools.get()
    return copy.deepcopy(tools) if tools else []


@contextmanager
def client_tools(tools: list[dict] | None) -> Iterator[list[dict]]:
    """Publish the client's tool definitions for the duration of the block."""
    value = copy.deepcopy([t for t in tools if isinstance(t, dict)]) if tools else None
    token = _client_tools.set(value)
    try:
        yield value or []
    finally:
        _client_tools.reset(token)
