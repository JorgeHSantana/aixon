# aixon/server/adapters/anthropic.py
"""Anthropic Messages-API ProtocolAdapter — the thin PROOF that aixon's neutral
types are not OpenAI-in-disguise.

Structural differences from OpenAI, served from the SAME neutral Message/Chunk:
- ``system`` is a top-level request field, hoisted into a neutral system Message.
- responses use a typed ``content[]`` block envelope with ``stop_reason``.
- streaming uses *named* SSE events (content_block_delta / message_delta /
  message_stop), not a bare ``data:`` line + ``[DONE]`` sentinel."""

from __future__ import annotations

import json
import uuid

from aixon.server.protocol import Chunk, Message, ParsedRequest, ProtocolAdapter, StreamSession

_TRANSPORT_FIELDS = frozenset({"model", "messages", "stream", "system", "tools"})


def _flatten_content(content) -> str:
    """Anthropic content may be a string or a list of typed blocks. Flatten the
    text blocks to neutral plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _openai_tools(tools) -> list[dict] | None:
    """Anthropic tool defs ({name, description, input_schema}) -> the OpenAI
    wire shape, so ParsedRequest.tools is dialect-neutral (always
    OpenAI-shaped) no matter which adapter parsed the request."""
    out = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        if "function" in t:            # already OpenAI-shaped: pass through
            out.append(t)
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {},
            },
        })
    return out or None


class AnthropicAdapter(ProtocolAdapter):
    name = "anthropic"

    # --- inbound ---------------------------------------------------------
    def parse_request(self, body: dict, *, path: str) -> ParsedRequest:
        messages: list[Message] = []
        system = body.get("system")
        if isinstance(system, str) and system:
            messages.append(Message(role="system", content=system))
        for m in body.get("messages") or []:
            if not isinstance(m, dict):
                raise ValueError("Each entry in 'messages' must be a JSON object.")
            messages.append(
                Message(role=m.get("role", "user"), content=_flatten_content(m.get("content")))
            )
        params = {k: v for k, v in body.items() if k not in _TRANSPORT_FIELDS}
        return ParsedRequest(
            model=body.get("model") or "",
            messages=messages,
            params=params,
            stream=bool(body.get("stream", False)),
            # ParsedRequest.tools is always OpenAI-shaped (see _openai_tools);
            # this converts the Anthropic wire dialect (input_schema, ...) so
            # current_client_tools() is dialect-neutral for every consumer.
            tools=_openai_tools(body.get("tools")),
        )

    # --- outbound (non-stream) ------------------------------------------
    def format_response(self, *, model: str, message: Message, usage: dict) -> dict:
        out_usage = {}
        if "prompt_tokens" in usage or "completion_tokens" in usage:
            out_usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }
        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": message.content}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": out_usage,
        }

    # --- outbound (stream) ----------------------------------------------
    def format_stream_chunk(self, *, model: str, chunk: Chunk) -> str:
        # A Chunk may carry reasoning AND content AND done at once (message.py
        # allows it). Emit one named SSE event per present field rather than an
        # exclusive ladder that would drop the others (e.g. Chunk(content=...,
        # done=True) losing the content).
        parts: list[str] = []
        if chunk.reasoning:
            parts.append(_event(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0,
                 "delta": {"type": "thinking_delta", "thinking": chunk.reasoning}},
            ))
        if chunk.content:
            parts.append(_event(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0,
                 "delta": {"type": "text_delta", "text": chunk.content}},
            ))
        if chunk.done:
            parts.append(_event(
                "message_delta",
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
            ))
        return "".join(parts)

    def format_stream_done(self, *, model: str) -> str:
        return _event("message_stop", {"type": "message_stop"})

    def format_stream_error(self, exc: Exception) -> str:
        return _event("error", {"type": "error",
                                "error": {"type": "api_error",
                                          "message": "The server encountered an "
                                                     "error while generating the "
                                                     "response."}})

    def open_stream(self, *, model: str, request: ParsedRequest) -> StreamSession:
        return _AnthropicStreamSession(self, model=model, request=request)

    # --- model listing ---------------------------------------------------
    def format_models(self, agents: list) -> dict:
        data = []
        for agent in agents:
            data.append({"type": "model", "id": agent.name})
            for alias in getattr(agent, "aliases", []) or []:
                data.append({"type": "model", "id": alias})
        return {"data": data}

    # --- routing ---------------------------------------------------------
    def routes(self) -> list[tuple[str, str]]:
        return [("POST", "/v1/messages"), ("GET", "/v1/models")]


class _AnthropicStreamSession(StreamSession):
    """Anthropic streaming with the full Messages-API SSE envelope: a real
    Anthropic SDK parses ``message_start``/``content_block_start``/
    ``content_block_stop`` and tracks per-block indices — it cannot parse the
    bare ``content_block_delta`` stream the stateless ``format_stream_chunk``
    emits (kept for compat; no longer used by the server route).

    Block sequencing: thinking (if any) opens first, closes when content
    starts; text opens next. Indices are assigned by an incrementing counter
    so thinking and text never share one, matching real multi-block Claude
    streams."""

    def __init__(self, adapter, *, model, request):
        super().__init__(adapter, model=model, request=request)
        self._started = False
        self._next_index = 0
        self._thinking_index: int | None = None
        self._text_index: int | None = None
        self._open_index: int | None = None  # currently open content block, if any
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []

    def _message_start(self) -> str:
        if self._started:
            return ""
        self._started = True
        return _event("message_start", {
            "type": "message_start",
            "message": {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

    def _close_open_block(self) -> str:
        if self._open_index is None:
            return ""
        out = _event("content_block_stop",
                     {"type": "content_block_stop", "index": self._open_index})
        self._open_index = None
        return out

    def chunk(self, chunk: Chunk) -> str:
        out = self._message_start()
        if chunk.reasoning:
            # KNOWN LIMITATION: if reasoning arrives AFTER the text block has
            # already opened (and therefore closed the thinking block — see
            # the `chunk.content` branch below), this re-enters here with
            # `_thinking_index` already set and emits a thinking_delta against
            # that now-CLOSED block instead of reopening a new one. Real
            # providers stream all reasoning before any text, so this
            # interleave is rare; accepted rather than adding block-reopen
            # bookkeeping for a case that shouldn't occur in practice.
            self._reasoning_parts.append(chunk.reasoning)
            if self._thinking_index is None:
                self._thinking_index = self._next_index
                self._next_index += 1
                out += _event("content_block_start", {
                    "type": "content_block_start", "index": self._thinking_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                })
                self._open_index = self._thinking_index
            out += _event("content_block_delta", {
                "type": "content_block_delta", "index": self._thinking_index,
                "delta": {"type": "thinking_delta", "thinking": chunk.reasoning},
            })
        if chunk.content:
            self._content_parts.append(chunk.content)
            if self._open_index == self._thinking_index and self._open_index is not None:
                out += self._close_open_block()
            if self._text_index is None:
                self._text_index = self._next_index
                self._next_index += 1
                out += _event("content_block_start", {
                    "type": "content_block_start", "index": self._text_index,
                    "content_block": {"type": "text", "text": ""},
                })
                self._open_index = self._text_index
            out += _event("content_block_delta", {
                "type": "content_block_delta", "index": self._text_index,
                "delta": {"type": "text_delta", "text": chunk.content},
            })
        if chunk.done:
            out += self._close_open_block()
            from aixon.server.usage import build_usage

            completion_text = "".join(self._content_parts)
            if self._reasoning_parts:
                completion_text += "\n" + "".join(self._reasoning_parts)
            usage = build_usage(self.model, "", completion_text)
            output_tokens = usage.get("completion_tokens", 0) if usage else 0
            out += _event("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            })
        return out
