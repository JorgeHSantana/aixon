# aixon/server/adapters/anthropic.py
"""Anthropic Messages-API ProtocolAdapter — a full production dialect served
from the SAME neutral Message/Chunk aixon's OpenAI adapter uses, proof that
the neutral types are not OpenAI-in-disguise.

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


def _tool_use_id(tc: dict) -> str:
    """Neutral tool-call id -> Anthropic ``tool_use.id``; an empty/missing
    neutral id (e.g. an agent that didn't bother minting one) gets a fresh
    ``toolu_`` one rather than shipping an empty id on the wire."""
    return tc.get("id") or f"toolu_{uuid.uuid4().hex}"


def _assistant_message_from_blocks(content: list) -> Message:
    """Assistant history content[] -> one neutral Message: text blocks flatten
    into ``content`` (same as ``_flatten_content``), ``tool_use`` blocks become
    ``Message.tool_calls`` entries in the neutral ``{name, args, id, type}``
    shape (mirrors ``_neutral_tool_calls`` in the OpenAI adapter)."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            # Non-dict input (valid JSON like a list or string) must degrade
            # to {} — a dict is required by AIMessage.tool_calls, and one
            # malformed history entry must not 500 the whole request (same
            # guard as _neutral_tool_calls in the OpenAI adapter).
            args = block.get("input")
            tool_calls.append({
                "name": block.get("name", ""),
                "args": args if isinstance(args, dict) else {},
                "id": block.get("id", ""),
                "type": "tool_call",
            })
    return Message(role="assistant", content="".join(text_parts), tool_calls=tool_calls)


def _user_messages_from_tool_result_blocks(content: list) -> list[Message]:
    """User history content[] with tool_result blocks -> one neutral
    ``Message(role="tool", ...)`` PER tool_result block (a tool_result's
    ``content`` may be a string or a list of text blocks — flatten either
    way), interleaved with a normal ``role="user"`` Message for any text
    blocks found alongside them, in the order the blocks appear."""
    out: list[Message] = []
    text_buffer: list[str] = []

    def _flush_text() -> None:
        if text_buffer:
            out.append(Message(role="user", content="".join(text_buffer)))
            text_buffer.clear()

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_buffer.append(block.get("text", ""))
        elif btype == "tool_result":
            _flush_text()
            out.append(Message(
                role="tool",
                tool_call_id=block.get("tool_use_id", ""),
                content=_flatten_content(block.get("content")),
            ))
    _flush_text()
    return out


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
        if "input_schema" not in t:
            # Anthropic SERVER tools (e.g. {"type": "web_search_20250305",
            # "name": "web_search", "max_uses": 5}) have no input_schema and
            # cannot be expressed as a client function tool — skip rather
            # than emit a bogus empty-parameters function def.
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
            role = m.get("role", "user")
            content = m.get("content")
            if role == "assistant" and isinstance(content, list):
                messages.append(_assistant_message_from_blocks(content))
                continue
            if role == "user" and isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            ):
                messages.extend(_user_messages_from_tool_result_blocks(content))
                continue
            messages.append(Message(role=role, content=_flatten_content(content)))
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
        content: list[dict] = [{"type": "text", "text": message.content}]
        stop_reason = "end_turn"
        if message.tool_calls:
            # message.content only becomes a text block when non-empty (a
            # tool-calls-only turn, the common case, has no text preamble);
            # `_flatten_content` intact behavior is preserved by the else path.
            content = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            for tc in message.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": _tool_use_id(tc),
                    "name": tc.get("name", ""),
                    "input": tc.get("args") or {},
                })
            stop_reason = "tool_use"
        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
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

    Block sequencing: blocks are a SEQUENCE, not a fixed thinking-then-text
    pair. Whichever modality (reasoning vs content) is NOT the currently open
    block closes the open one (``content_block_stop``) and opens a fresh block
    at the next index when it needs to emit — so thinking -> text -> thinking
    (an interleave real providers don't produce today, but the wire format
    allows) still yields a valid, ever-increasing index per block instead of a
    delta against an already-closed one."""

    def __init__(self, adapter, *, model, request):
        super().__init__(adapter, model=model, request=request)
        self._started = False
        self._next_index = 0
        self._open_index: int | None = None  # currently open content block, if any
        self._open_kind: str | None = None  # "thinking" | "text" | None
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._had_tool_use = False  # flips the final message_delta.stop_reason

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
        self._open_kind = None
        return out

    def _open_block(self, kind: str, content_block: dict) -> str:
        """Ensure a block of ``kind`` ("thinking"/"text") is open at
        ``self._open_index``. If a DIFFERENT kind is currently open, close it
        first (``content_block_stop``) and allocate a brand-new index for the
        one being opened — indices are never reused, matching how a real
        Claude stream never reopens a closed block."""
        out = ""
        if self._open_kind is not None and self._open_kind != kind:
            out += self._close_open_block()
        if self._open_index is None:
            index = self._next_index
            self._next_index += 1
            out += _event("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": content_block,
            })
            self._open_index = index
            self._open_kind = kind
        return out

    def chunk(self, chunk: Chunk) -> str:
        out = self._message_start()
        if chunk.reasoning:
            self._reasoning_parts.append(chunk.reasoning)
            out += self._open_block("thinking", {"type": "thinking", "thinking": ""})
            out += _event("content_block_delta", {
                "type": "content_block_delta", "index": self._open_index,
                "delta": {"type": "thinking_delta", "thinking": chunk.reasoning},
            })
        if chunk.content:
            self._content_parts.append(chunk.content)
            out += self._open_block("text", {"type": "text", "text": ""})
            out += _event("content_block_delta", {
                "type": "content_block_delta", "index": self._open_index,
                "delta": {"type": "text_delta", "text": chunk.content},
            })
        if chunk.tool_calls:
            # A tool_use block is never additive (unlike text/thinking deltas):
            # the full call is already known, so it opens, gets ONE
            # input_json_delta with the whole JSON, and closes immediately —
            # whatever text/thinking block was open closes first (a new kind
            # always forces the current block shut, see _open_block).
            out += self._close_open_block()
            for tc in chunk.tool_calls:
                index = self._next_index
                self._next_index += 1
                out += _event("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "tool_use", "id": _tool_use_id(tc),
                                      "name": tc.get("name", ""), "input": {}},
                })
                out += _event("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta",
                             "partial_json": json.dumps(tc.get("args") or {},
                                                        ensure_ascii=False)},
                })
                out += _event("content_block_stop",
                             {"type": "content_block_stop", "index": index})
            self._had_tool_use = True
        if chunk.done:
            out += self._close_open_block()
            from aixon.server.usage import build_usage

            completion_text = "".join(self._content_parts)
            if self._reasoning_parts:
                completion_text += "\n" + "".join(self._reasoning_parts)
            usage = build_usage(self.model, "", completion_text)
            output_tokens = usage.get("completion_tokens", 0) if usage else 0
            stop_reason = "tool_use" if self._had_tool_use else "end_turn"
            out += _event("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            })
        return out

    def error(self, exc: Exception) -> str:
        """A mid-stream failure must close whatever block is open BEFORE the
        error event — the client's SDK tracks block state and would choke on
        a delta/stop it never sees after an `error` event closes the request
        conceptually. `_message_start` is idempotent, so this stays safe even
        if the agent raised before yielding a single chunk."""
        out = self._message_start()
        out += self._close_open_block()
        out += self.adapter.format_stream_error(exc)
        return out
