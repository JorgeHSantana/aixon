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

from aixon.server.protocol import Chunk, Message, ParsedRequest, ProtocolAdapter

_TRANSPORT_FIELDS = frozenset({"model", "messages", "stream", "system"})


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


class AnthropicAdapter(ProtocolAdapter):
    name = "anthropic"

    # --- inbound ---------------------------------------------------------
    def parse_request(self, body: dict, *, path: str) -> ParsedRequest:
        messages: list[Message] = []
        system = body.get("system")
        if isinstance(system, str) and system:
            messages.append(Message(role="system", content=system))
        for m in body.get("messages") or []:
            messages.append(
                Message(role=m.get("role", "user"), content=_flatten_content(m.get("content")))
            )
        params = {k: v for k, v in body.items() if k not in _TRANSPORT_FIELDS}
        return ParsedRequest(
            model=body.get("model") or "",
            messages=messages,
            params=params,
            stream=bool(body.get("stream", False)),
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
