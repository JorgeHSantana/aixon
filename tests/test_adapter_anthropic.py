# tests/test_adapter_anthropic.py
from __future__ import annotations

import json

from aixon.message import Chunk, Message
from aixon.server.adapters.anthropic import AnthropicAdapter


def _event_blocks(text: str) -> list[tuple[str, dict]]:
    """Split an SSE payload into (event_name, data_dict) pairs."""
    out = []
    for block in [b for b in text.split("\n\n") if b.strip()]:
        lines = block.split("\n")
        event = next(l[len("event: ") :] for l in lines if l.startswith("event: "))
        data = next(json.loads(l[len("data: ") :]) for l in lines if l.startswith("data: "))
        out.append((event, data))
    return out


class TestParseRequest:
    def test_system_is_hoisted_into_messages(self):
        a = AnthropicAdapter()
        pr = a.parse_request(
            {
                "model": "echo",
                "system": "you are terse",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
            path="/v1/messages",
        )
        assert pr.model == "echo"
        assert pr.messages[0].role == "system"
        assert pr.messages[0].content == "you are terse"
        assert pr.messages[1].role == "user"
        assert pr.params == {"max_tokens": 100}
        assert pr.stream is False

    def test_content_blocks_flattened_to_text(self):
        a = AnthropicAdapter()
        pr = a.parse_request(
            {
                "model": "echo",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "a"},
                                                 {"type": "text", "text": "b"}]}
                ],
            },
            path="/v1/messages",
        )
        assert pr.messages[0].content == "ab"


class TestFormatResponse:
    def test_messages_envelope(self):
        a = AnthropicAdapter()
        out = a.format_response(
            model="echo",
            message=Message(role="assistant", content="hello"),
            usage={"prompt_tokens": 3, "completion_tokens": 5},
        )
        assert out["type"] == "message"
        assert out["role"] == "assistant"
        assert out["model"] == "echo"
        assert out["content"] == [{"type": "text", "text": "hello"}]
        assert out["stop_reason"] == "end_turn"
        assert out["usage"] == {"input_tokens": 3, "output_tokens": 5}


class TestFormatStream:
    def test_content_delta_event(self):
        a = AnthropicAdapter()
        (event, data), = _event_blocks(
            a.format_stream_chunk(model="echo", chunk=Chunk(content="hi"))
        )
        assert event == "content_block_delta"
        assert data["delta"] == {"type": "text_delta", "text": "hi"}

    def test_reasoning_delta_event(self):
        a = AnthropicAdapter()
        (event, data), = _event_blocks(
            a.format_stream_chunk(model="echo", chunk=Chunk(reasoning="r"))
        )
        assert event == "content_block_delta"
        assert data["delta"] == {"type": "thinking_delta", "thinking": "r"}

    def test_done_emits_message_delta(self):
        a = AnthropicAdapter()
        (event, data), = _event_blocks(
            a.format_stream_chunk(model="echo", chunk=Chunk(done=True))
        )
        assert event == "message_delta"
        assert data["delta"]["stop_reason"] == "end_turn"

    def test_empty_chunk_skipped(self):
        a = AnthropicAdapter()
        assert a.format_stream_chunk(model="echo", chunk=Chunk()) == ""

    def test_stream_done_is_named_message_stop(self):
        a = AnthropicAdapter()
        (event, data), = _event_blocks(a.format_stream_done(model="echo"))
        assert event == "message_stop"
        assert data["type"] == "message_stop"


class TestModelsAndRoutes:
    def test_models_listing(self):
        a = AnthropicAdapter()

        class _Fake:
            name = "echo"
            aliases = ["e1"]
            owned_by = "aixon"

        out = a.format_models([_Fake()])
        ids = [d["id"] for d in out["data"]]
        assert ids == ["echo", "e1"]
        assert all(d["type"] == "model" for d in out["data"])

    def test_routes(self):
        a = AnthropicAdapter()
        assert a.routes() == [("POST", "/v1/messages"), ("GET", "/v1/models")]
        assert a.name == "anthropic"
