"""Request-scoped memoization for tool calls (issue #5).

A ``ToolCallCache`` lives for one request (or one ReflectiveAgent run when no
request scope is active) and maps ``(tool_name, normalized_args)`` to the tool's
result. It exists to stop retry loops (evaluator-optimizer rounds) from
re-executing identical tool calls — repeated DB queries, web searches — which
wastes cost/latency AND can produce answers inconsistent with the critique that
motivated the retry.

Backed by a ``contextvars.ContextVar`` (NOT thread-local) so it composes with
sync code, async code and LangGraph execution — the same design as
``aixon.reasoning``. No TTL, no invalidation: the cache dies with its context.

Usage:
    with tool_call_cache():          # request scope (Server) / run scope
        ...                          # tool executions memoize transparently

Nesting REUSES the outer cache (a ReflectiveAgent inside a served request
shares the request's cache instead of shadowing it)."""

from __future__ import annotations

import contextvars
import json
from contextlib import contextmanager
from typing import Any, Iterator, Optional


class ToolCallCache:
    """Maps (tool_name, normalized_args) -> result for one context."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], Any] = {}

    @staticmethod
    def key(name: str, args: tuple, kwargs: dict) -> tuple[str, str]:
        """Normalize a call into a hashable key. ``default=repr`` keeps weird
        argument types from crashing normalization (a cache must never break
        the call it memoizes)."""
        normalized = json.dumps([list(args), kwargs], sort_keys=True, default=repr)
        return (name, normalized)

    def get(self, key: tuple[str, str]) -> Any:
        return self._store.get(key)

    def has(self, key: tuple[str, str]) -> bool:
        return key in self._store

    def set(self, key: tuple[str, str], value: Any) -> None:
        self._store[key] = value


_current: contextvars.ContextVar[Optional[ToolCallCache]] = contextvars.ContextVar(
    "aixon_tool_call_cache", default=None
)


def current_tool_cache() -> Optional[ToolCallCache]:
    """The cache active in this execution context, or None (memoization off)."""
    return _current.get()


@contextmanager
def tool_call_cache() -> Iterator[ToolCallCache]:
    """Activate a tool-call cache for the block. If one is ALREADY active,
    yield it unchanged — nesting shares the outer scope (request wins over an
    inner ReflectiveAgent run) instead of fragmenting the memoization."""
    existing = _current.get()
    if existing is not None:
        yield existing
        return
    cache = ToolCallCache()
    token = _current.set(cache)
    try:
        yield cache
    finally:
        _current.reset(token)
