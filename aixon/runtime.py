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

# The part of GENERATION_PARAMS that means the same thing to ANY model: the
# sampling and length knobs. Only these cross over to an imposed model (see
# imposed_model_scope). Everything else is a setting for the model the agent
# was written against, and is dropped: the class's `reasoning`, the request's
# `reasoning_effort`, `timeout`, `seed`, `prediction`, any vendor kwarg.
PORTABLE_PARAMS = frozenset(
    {"temperature", "top_p", "max_tokens", "stop", "presence_penalty", "frequency_penalty"}
)

# Provider-dialect spellings normalized to the canonical key above so downstream
# models always receive kwargs every LangChain chat model understands. The
# canonical key wins when both are present.
_PARAM_ALIASES = {
    "max_completion_tokens": "max_tokens",  # modern OpenAI clients
    "stop_sequences": "stop",               # Anthropic dialect
}

def normalize_param_aliases(params: dict) -> dict:
    """Cópia de *params* com cada grafia de dialeto trocada pela canônica.

    O valor do alias só entra quando a chave canônica falta ou é None (a
    canônica vence), e um alias None é descartado. A entrada não é mutada.
    Regra única usada por ``generation_params``, ``LLM.request_chat_model`` e
    ``LLM.settings_lost_under_imposition``."""
    out = {k: v for k, v in params.items() if k not in _PARAM_ALIASES}
    for alias, canonical in _PARAM_ALIASES.items():
        value = params.get(alias)
        if value is not None and out.get(canonical) is None:
            out[canonical] = value
    return out


def _reset_quietly(var: contextvars.ContextVar, token: contextvars.Token) -> None:
    """``var.reset(token)`` que tolera a saída num Context diferente (#21).

    ``reset()`` levanta ValueError quando a limpeza roda em outro Context, por
    exemplo na finalização por GC de um async generator abandonado depois de
    uma desconexão abrupta do cliente. A variável morre com aquele Context;
    engolir aqui só cala o log de exceção não levantável."""
    try:
        var.reset(token)
    except ValueError:
        pass


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
    raw = normalize_param_aliases(params or {})
    filtered = {k: v for k, v in raw.items() if k in GENERATION_PARAMS and v is not None}
    token = _gen_params.set(filtered)
    try:
        # Uma cópia, como current_generation_params(): mutar o valor do
        # `with ... as p` não altera o estado guardado na ContextVar.
        yield dict(filtered)
    finally:
        _reset_quietly(_gen_params, token)


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
def client_tools_scope(tools: list[dict] | None) -> Iterator[list[dict]]:
    """Publish the client's tool definitions for the duration of the block.

    Named ``*_scope`` (#25) for consistency with ``tool_choice_scope``/
    ``generation_params`` — renamed from ``client_tools`` (no back-compat
    alias: internal-only usage, server.py + tests, pre-0.2.0)."""
    value = copy.deepcopy([t for t in tools if isinstance(t, dict)]) if tools else None
    token = _client_tools.set(value)
    try:
        # Cópia profunda, como current_client_tools(): o bloco `with` não
        # recebe o objeto guardado na ContextVar.
        yield copy.deepcopy(value) if value else []
    finally:
        _reset_quietly(_client_tools, token)


# tool_choice of the client (#18a): published per request alongside
# client_tools when the request declares one (OpenAI wire shape, e.g.
# {"type": "function", "function": {"name": ...}}, or the literal "auto"/
# "none"/"required"). Agents/LLM calls that bind client tools read this to
# forward the client's preference to the provider; everyone else ignores it.
_tool_choice: contextvars.ContextVar = contextvars.ContextVar(
    "aixon_tool_choice", default=None
)


def current_tool_choice():
    """Return the client's ``tool_choice`` for the current request, or
    ``None`` when absent. Always a deep copy: mirrors ``current_client_tools``
    so a caller mutating the result never pollutes the ContextVar state."""
    return copy.deepcopy(_tool_choice.get())


@contextmanager
def tool_choice_scope(value) -> Iterator:
    """Publish (or clear, with ``None``) the client's ``tool_choice`` for the
    duration of the block. Mirrors ``client_tools_scope``'s pattern."""
    token = _tool_choice.set(copy.deepcopy(value) if value is not None else None)
    try:
        yield value
    finally:
        _reset_quietly(_tool_choice, token)


# Predicted output (#6): the previous attempt's full text, published by the
# ReflectiveAgent around a RETRY call only. Providers that support OpenAI's
# Predicted Outputs read it via current_prediction() and attach it to the
# invocation; everyone else ignores it. Latency win only — rejected predicted
# tokens are still billed as output.
_prediction: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aixon_prediction", default=None
)


def current_prediction() -> str | None:
    """The predicted-output text active for this context, or None."""
    return _prediction.get()


@contextmanager
def prediction_scope(text: str | None) -> Iterator[str | None]:
    """Publish (or clear, with None) the predicted-output text for the block."""
    token = _prediction.set(text or None)
    try:
        yield text or None
    finally:
        _reset_quietly(_prediction, token)


# The e-mail of the end user the current call acts for, published around the
# agent call by whoever serves the request. Quem o lê e como ele sai do
# processo é assunto de quem lê (ver aixon.providers.gateway).
_user_email: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aixon_user_email", default=None
)


def current_user_email() -> str | None:
    """The end user's e-mail active for this context, or None."""
    return _user_email.get()


@contextmanager
def user_email_scope(email: str | None) -> Iterator[str | None]:
    """Publish (or clear, with None) the end user's e-mail for the block.

    The e-mail is stripped and a blank one becomes None, so readers never
    tell "" from absent. On exit the previous value comes back.

    Like any ContextVar
    scope, it is seen only by what runs in this context (asyncio tasks and
    ``asyncio.to_thread`` copy it; a bare thread does not), and a scope held
    across a ``yield`` is also seen by the generator's consumer between two
    items: open it around the call, not around a ``yield``."""
    value = (email or "").strip() or None
    token = _user_email.set(value)
    try:
        yield value
    finally:
        _reset_quietly(_user_email, token)


# The model imposed on the current call, as a (model, provider) pair, published
# by whoever runs the agent tree. While it is set, every LLM that is not
# `pinned` builds THAT model instead of the one it was declared with, handing it
# only the PORTABLE_PARAMS, and refuses to give out its configured chat_model
# (a silent fallback would run the call on the wrong model).
_imposed_model: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "aixon_imposed_model", default=None
)


def current_imposed_model() -> tuple[str, str] | None:
    """The ``(model, provider)`` imposed on this context, or None."""
    return _imposed_model.get()


@contextmanager
def imposed_model_scope(model: str, provider: str) -> Iterator[tuple[str, str]]:
    """Impose ``model`` of ``provider`` on the LLM calls made in the block.

    Always activates the pair it is given, and on exit the previous value comes
    back. Which scope wins when they nest is not decided here: an agent that
    finds one already active chooses whether to open its own.
    """
    value = (model, provider)
    token = _imposed_model.set(value)
    try:
        yield value
    finally:
        _reset_quietly(_imposed_model, token)
