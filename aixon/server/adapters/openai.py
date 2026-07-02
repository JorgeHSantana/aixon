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

from aixon.server.protocol import Chunk, Message, ParsedRequest, ProtocolAdapter, StreamSession

# Transport-level fields the adapter consumes itself; everything else in the
# body is a passthrough param handed to the agent's params.
_TRANSPORT_FIELDS = frozenset({"model", "messages", "stream"})


def _flatten_content(content) -> str:
    """OpenAI content may be a string or a list of typed parts. Flatten the
    text parts to neutral plain text (same shape as the Anthropic adapter)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


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
                    content=_flatten_content(m.get("content")),
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

    def _usage_chunk_line(self, *, model: str, usage: dict) -> str:
        payload = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [],
            "usage": usage,
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def open_stream(self, *, model: str, request: ParsedRequest) -> StreamSession:
        return _OpenAIStreamSession(self, model=model, request=request)

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


class _OpenAIStreamSession(StreamSession):
    """OpenAI streaming with thought_stream_mode + optional usage.

    Modes (request param ``thought_stream_mode``, default ``content``):
      - content: reasoning wrapped in a single <think>...</think> block inside
        delta.content; closed before the first real content delta.
      - custom:  reasoning in delta.reasoning (aixon's native behavior).
      - hidden:  reasoning dropped; content only.
    Usage (when ``stream_options.include_usage`` is true) is emitted as a final
    choices=[] chunk before [DONE]."""

    def __init__(self, adapter, *, model, request):
        super().__init__(adapter, model=model, request=request)
        params = request.params or {}
        self.mode = params.get("thought_stream_mode") or "content"
        stream_options = params.get("stream_options") or {}
        self.include_usage = bool(stream_options.get("include_usage"))
        self._think_open = False
        self._content_started = False
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []

    def _line(self, delta, finish_reason=None):
        return self.adapter._chunk_line(model=self.model, delta=delta, finish_reason=finish_reason)

    def chunk(self, chunk: Chunk) -> str:
        out = ""
        if chunk.reasoning:
            self._reasoning_parts.append(chunk.reasoning)
            if self.mode == "custom":
                out += self._line({"reasoning": chunk.reasoning})
            elif self.mode == "content":
                # "<think>\n" once on open; later reasoning deltas are raw
                # token-level fragments — no separator between them.
                text = ("" if self._think_open else "<think>\n") + chunk.reasoning
                self._think_open = True
                out += self._line({"content": text})
            # hidden: drop reasoning
        if chunk.content:
            self._content_parts.append(chunk.content)
            text = chunk.content
            if self.mode == "content" and self._think_open and not self._content_started:
                text = "</think>\n" + text
                self._think_open = False
            self._content_started = True
            out += self._line({"content": text})
        if chunk.done:
            if self.mode == "content" and self._think_open:
                out += self._line({"content": "</think>\n"})
                self._think_open = False
            out += self._line({}, finish_reason="stop")
        return out

    def finish(self) -> str:
        if not self.include_usage:
            return ""
        from aixon.server.usage import build_usage

        prompt_text = "\n".join(m.content for m in self.request.messages)
        completion_text = "".join(self._content_parts)
        if self._reasoning_parts:
            completion_text += "\n" + "".join(self._reasoning_parts)
        usage = build_usage(self.model, prompt_text, completion_text)
        if not usage:
            return ""
        return self.adapter._usage_chunk_line(model=self.model, usage=usage)
