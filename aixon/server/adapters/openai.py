# aixon/server/adapters/openai.py
"""OpenAI-compatible ProtocolAdapter — the full, primary dialect.

Wire shapes mirror OpenAI's Chat Completions API as pure neutral translation:
``/v1/chat/completions`` (stream + non-stream) and ``/v1/models``. Reasoning is
surfaced in the ``message``/``delta`` ``reasoning`` field (reasoning-field
mode)."""

from __future__ import annotations

import json
import time
import uuid

from aixon.server.protocol import Chunk, Message, ParsedRequest, ProtocolAdapter

# Transport-level fields the adapter consumes itself; everything else in the
# body is a passthrough param handed to the agent's params.
_TRANSPORT_FIELDS = frozenset({"model", "messages", "stream"})


class OpenAIAdapter(ProtocolAdapter):
    name = "openai"

    # --- inbound ---------------------------------------------------------
    def parse_request(self, body: dict, *, path: str) -> ParsedRequest:
        raw_messages = body.get("messages") or []
        messages: list[Message] = []
        for m in raw_messages:
            messages.append(
                Message(
                    role=m.get("role", "user"),
                    content=m.get("content") or "",
                    name=m.get("name"),
                    tool_call_id=m.get("tool_call_id"),
                )
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
        msg: dict = {"role": "assistant", "content": message.content}
        if message.reasoning is not None:
            msg["reasoning"] = message.reasoning
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "message": msg, "finish_reason": "stop"}
            ],
            "usage": usage,
        }

    # --- outbound (stream) ----------------------------------------------
    def _chunk_line(self, *, model: str, delta: dict, finish_reason) -> str:
        payload = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def format_stream_chunk(self, *, model: str, chunk: Chunk) -> str:
        # Build the delta additively: a Chunk may carry content AND reasoning AND
        # done together (message.py allows it), so an exclusive if/return ladder
        # would silently drop fields (e.g. Chunk(content=..., done=True) losing
        # the content). Include every field that is present.
        delta: dict = {}
        if chunk.reasoning:
            delta["reasoning"] = chunk.reasoning
        if chunk.content:
            delta["content"] = chunk.content
        if not delta and not chunk.done:
            return ""  # nothing to emit for an empty chunk
        finish_reason = "stop" if chunk.done else None
        return self._chunk_line(model=model, delta=delta, finish_reason=finish_reason)

    def format_stream_done(self, *, model: str) -> str:
        return "data: [DONE]\n\n"

    # --- model listing ---------------------------------------------------
    def format_models(self, agents: list) -> dict:
        created = int(time.time())
        data = []
        for agent in agents:
            owned_by = getattr(agent, "owned_by", "aixon")
            data.append(
                {"id": agent.name, "object": "model", "created": created, "owned_by": owned_by}
            )
            for alias in getattr(agent, "aliases", []) or []:
                data.append(
                    {"id": alias, "object": "model", "created": created, "owned_by": owned_by}
                )
        return {"object": "list", "data": data}

    # --- routing ---------------------------------------------------------
    def routes(self) -> list[tuple[str, str]]:
        # Serve both the canonical /v1/* paths and the bare aliases (/chat/
        # completions, /models) for OpenAI clients that omit the version prefix.
        return [
            ("POST", "/v1/chat/completions"),
            ("POST", "/chat/completions"),
            ("GET", "/v1/models"),
            ("GET", "/models"),
        ]
