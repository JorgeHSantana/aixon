"""Reasoning channel. Collects reasoning text emitted during an agent run and
makes it available to the streaming layer, propagating across nested agents.

Backed by a ``contextvars.ContextVar`` (NOT thread-local) so it composes with
sync code, async code, and LangGraph execution. The olympus framework used a
``threading.local()`` thought queue; aixon deliberately uses a ContextVar so a
copied/forked execution context carries its own channel correctly.

Usage:
    with reasoning_channel() as channel:   # activates a channel for this run
        ...                                 # nested agents call emit_reasoning()
        for line in channel.drain():        # streaming loop pulls lines out
            yield Chunk(reasoning=line + "\\n")
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator, Optional


class ReasoningChannel:
    """Buffers reasoning lines for one agent run."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def emit(self, text: str) -> None:
        """Append one reasoning line."""
        self._lines.append(text)

    def drain(self) -> list[str]:
        """Return all buffered lines and clear the buffer."""
        lines = self._lines
        self._lines = []
        return lines

    @property
    def lines(self) -> list[str]:
        """A copy of the currently buffered lines (does not clear)."""
        return list(self._lines)


# The active channel for the current execution context. None when no run is
# streaming through a channel.
_current: contextvars.ContextVar[Optional[ReasoningChannel]] = contextvars.ContextVar(
    "aixon_reasoning_channel", default=None
)


def current_channel() -> Optional[ReasoningChannel]:
    """Return the channel active in this execution context, or None."""
    return _current.get()


def emit_reasoning(text: str) -> None:
    """Push a reasoning line to the current channel if one is active.

    No-op when no channel is active, so a nested agent invoked outside any
    stream() (e.g. a bare ``agent.invoke``) never raises. Nested agents call
    this and their reasoning bubbles to the parent's stream because the
    parent's stream() set the active channel.
    """
    channel = _current.get()
    if channel is not None:
        channel.emit(text)


@contextmanager
def reasoning_channel() -> Iterator[ReasoningChannel]:
    """Activate a fresh ReasoningChannel for the duration of a stream().

    Drained by the streaming loop into ``Chunk(reasoning=...)``. The ContextVar
    token is reset on exit so nested ``reasoning_channel()`` blocks restore the
    outer channel (LIFO).
    """
    channel = ReasoningChannel()
    token = _current.set(channel)
    try:
        yield channel
    finally:
        _current.reset(token)
