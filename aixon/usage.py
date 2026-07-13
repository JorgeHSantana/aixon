"""Usage accumulator — same ContextVar pattern as ``aixon.reasoning`` (a
mutable object stored in a ``contextvars.ContextVar`` so it composes with
sync code, async code, and LangGraph execution, and so nested/concurrent runs
each carry their own accumulator instead of colliding on shared instance
state).

Exists because not every model turn of a multi-turn run lands in the neutral
Message list a caller can sum after the fact. An Orchestrator's Tier-1
supervisor asks ``LLM.complete``/``acomplete`` to pick the next worker — that
reply is a routing decision, never appended to ``state["messages"]`` — so its
usage would be lost if the only sink were "diff the resulting message list".
``add_usage`` lets any call site (worker node, supervisor routing call, judge
call, ...) contribute its turn's usage to the run's total, regardless of
whether that turn also produced a Message the caller keeps.

``merge_usage`` is also exposed standalone for callers that already hold each
turn's usage dict in hand (e.g. ``ReflectiveAgent``, which runs entirely
within one synchronous/async method with no hidden graph turns) and would
rather fold usage locally than open a scope.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator, Optional

_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def merge_usage(a: "dict[str, int] | None", b: "dict[str, int] | None") -> "dict[str, int] | None":
    """Sum two neutral OpenAI-shaped usage dicts.

    A side that reports no usage (``None``) contributes zero and does NOT
    erase a total already accumulated from the other side. Only when BOTH
    sides are ``None`` does the merge stay ``None`` (nothing was ever
    reported, so the server's estimate-based fallback should still apply)."""
    if a is None and b is None:
        return None
    out = {k: 0 for k in _KEYS}
    for src in (a, b):
        if src:
            for k in _KEYS:
                out[k] += src.get(k, 0)
    return out


class UsageAccumulator:
    """Buffers usage across every model turn of one run."""

    def __init__(self) -> None:
        self._total: "dict[str, int] | None" = None

    def add(self, usage: "dict[str, int] | None") -> None:
        self._total = merge_usage(self._total, usage)

    @property
    def total(self) -> "dict[str, int] | None":
        return dict(self._total) if self._total is not None else None


# The active accumulator for the current execution context. None when no run
# has opened a usage_scope().
_current: contextvars.ContextVar[Optional[UsageAccumulator]] = contextvars.ContextVar(
    "aixon_usage_accumulator", default=None
)


def current_usage_accumulator() -> Optional[UsageAccumulator]:
    """Return the accumulator active in this execution context, or None."""
    return _current.get()


def add_usage(usage: "dict[str, int] | None") -> None:
    """Contribute one turn's usage to the current accumulator, if any.

    No-op when no scope is active, so a call site can call this unconditionally
    (mirrors ``emit_reasoning``'s no-op-when-inactive contract) without needing
    to know whether it is running inside a usage_scope()."""
    acc = _current.get()
    if acc is not None:
        acc.add(usage)


@contextmanager
def usage_scope() -> Iterator[UsageAccumulator]:
    """Activate a fresh UsageAccumulator for the duration of one run.

    The ContextVar token is reset on exit so a nested usage_scope() (a nested
    Orchestrator run invoked as a worker, say) restores the outer accumulator
    (LIFO) — nested runs do not leak their turns into the outer total, matching
    ``reasoning_channel``'s nesting behavior."""
    acc = UsageAccumulator()
    token = _current.set(acc)
    try:
        yield acc
    finally:
        _current.reset(token)
