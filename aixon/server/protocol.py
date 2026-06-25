"""The protocol-decoupling seam.

``ProtocolAdapter`` translates a wire format (OpenAI, Anthropic, ...) to and
from aixon's neutral types. The agent runtime speaks ONLY ``Message``/``Chunk``;
no vendor/wire detail crosses this boundary inward. Neutral types are
re-exported here so adapters import them from one place. This module is pure
stdlib + neutral types — it does NOT import FastAPI, so the seam is importable
and testable on a bare install."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# Re-export the neutral types (the SAME objects, not copies) so adapter modules
# can do `from aixon.server.protocol import Message, Chunk`.
from aixon.message import Chunk, Message, Role

__all__ = ["Message", "Chunk", "Role", "ParsedRequest", "ProtocolAdapter", "StreamSession"]


@dataclass
class ParsedRequest:
    """A wire request reduced to neutral terms. The Server consumes only this —
    it never sees the raw vendor body.

    - ``model``: the requested agent name/alias (the wire ``model`` field).
    - ``messages``: neutral conversation handed straight to ``Agent.invoke``.
    - ``params``: passthrough knobs (temperature, max_tokens, ...) minus the
      transport-level fields the adapter already consumed (model, messages,
      stream, system).
    - ``stream``: whether the client asked for an SSE stream.
    """

    model: str
    messages: list[Message]
    params: dict
    stream: bool


class StreamSession:
    """Per-request streaming state. Base is stateless: it delegates to the
    adapter's ``format_stream_chunk`` / ``format_stream_done``. Dialects that
    need per-request state (think-mode wrapping, usage accumulation) subclass
    this and are returned by ``ProtocolAdapter.open_stream``."""

    def __init__(self, adapter: "ProtocolAdapter", *, model: str, request: "ParsedRequest"):
        self.adapter = adapter
        self.model = model
        self.request = request

    def chunk(self, chunk: "Chunk") -> str:
        """SSE line(s) for one neutral Chunk ('' to emit nothing)."""
        return self.adapter.format_stream_chunk(model=self.model, chunk=chunk)

    def finish(self) -> str:
        """Extra SSE line(s) emitted after the last chunk, before done()."""
        return ""

    def done(self) -> str:
        """Terminal SSE line(s)."""
        return self.adapter.format_stream_done(model=self.model)


class ProtocolAdapter(ABC):
    """Translates one wire dialect <-> neutral types. New wire styles = new
    subclass. NO neutral type leaks a vendor/wire detail."""

    name: str = ""  # e.g. "openai", "anthropic"

    def __init__(self, *, mount_prefix: str = "") -> None:
        """``mount_prefix`` is prepended to every path from ``routes()`` when the
        Server mounts this adapter (default ``""`` = the adapter's canonical
        paths). Use it to serve two dialects whose ``routes()`` would otherwise
        collide, e.g. ``AnthropicAdapter(mount_prefix="/anthropic")`` so its
        ``/v1/models`` does not clash with OpenAI's."""
        self.mount_prefix = mount_prefix.rstrip("/")

    @abstractmethod
    def parse_request(self, body: dict, *, path: str) -> ParsedRequest:
        """Reduce a raw request body to a neutral ``ParsedRequest``. ``path`` is
        the matched route, so one adapter can serve several paths."""

    @abstractmethod
    def format_response(self, *, model: str, message: Message, usage: dict) -> dict:
        """Wrap a final neutral ``Message`` in the dialect's non-stream envelope."""

    @abstractmethod
    def format_stream_chunk(self, *, model: str, chunk: Chunk) -> str:
        """Return one SSE ``'data: {...}\\n\\n'`` line for a neutral ``Chunk``
        (or ``''`` to emit nothing for this chunk)."""

    @abstractmethod
    def format_stream_done(self, *, model: str) -> str:
        """Return the terminal SSE line(s) that close the stream."""

    @abstractmethod
    def format_models(self, agents: list) -> dict:
        """Render the model-listing payload from registered agents."""

    @abstractmethod
    def routes(self) -> list[tuple[str, str]]:
        """``[(http_method, path)]`` this adapter serves, e.g.
        ``[("POST","/v1/chat/completions"), ("GET","/v1/models")]``."""

    def open_stream(self, *, model: str, request: "ParsedRequest") -> "StreamSession":
        """Return a per-request StreamSession. Default = stateless passthrough;
        override to add dialect-specific per-request stream state."""
        return StreamSession(self, model=model, request=request)
