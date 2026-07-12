# tests/test_anthropic_client_tools.py
"""N1 — Anthropic outbound client-tools round-trip.

Closes the gap where the Anthropic adapter normalized inbound tool defs
(M2, see test_maturity_tools_dialect.py) but never emitted ``tool_use``
blocks nor ingested ``tool_result`` history. After this, client tools
round-trip in both the OpenAI dialect (test_client_tools.py) and the
Anthropic dialect (this file).
"""
from __future__ import annotations

import json

from aixon.message import Chunk, Message
from aixon.server.adapters.anthropic import AnthropicAdapter
from aixon.server.protocol import ParsedRequest


def _event_blocks(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in [b for b in text.split("\n\n") if b.strip()]:
        lines = block.split("\n")
        event = next(l[len("event: ") :] for l in lines if l.startswith("event: "))
        data = next(json.loads(l[len("data: ") :]) for l in lines if l.startswith("data: "))
        out.append((event, data))
    return out


class TestFormatResponse:
    def test_tool_use_block_with_stop_reason_tool_use(self):
        a = AnthropicAdapter()
        out = a.format_response(
            model="echo",
            message=Message(
                role="assistant", content="",
                tool_calls=[{"name": "open_file", "args": {"path": "/tmp/x"}, "id": "c1"}],
            ),
            usage={},
        )
        assert out["content"] == [
            {"type": "tool_use", "id": "c1", "name": "open_file", "input": {"path": "/tmp/x"}}
        ]
        assert out["stop_reason"] == "tool_use"

    def test_text_block_precedes_tool_use_when_content_present(self):
        a = AnthropicAdapter()
        out = a.format_response(
            model="echo",
            message=Message(
                role="assistant", content="let me check",
                tool_calls=[{"name": "open_file", "args": {}, "id": "c1"}],
            ),
            usage={},
        )
        assert out["content"][0] == {"type": "text", "text": "let me check"}
        assert out["content"][1]["type"] == "tool_use"

    def test_missing_id_generates_toolu_prefixed_id(self):
        a = AnthropicAdapter()
        out = a.format_response(
            model="echo",
            message=Message(role="assistant", tool_calls=[{"name": "f", "args": {}, "id": ""}]),
            usage={},
        )
        tool_use_id = out["content"][0]["id"]
        assert tool_use_id.startswith("toolu_")
        assert len(tool_use_id) == len("toolu_") + 32  # uuid4().hex length

    def test_no_tool_calls_behavior_intact(self):
        a = AnthropicAdapter()
        out = a.format_response(
            model="echo", message=Message(role="assistant", content="hello"), usage={},
        )
        assert out["content"] == [{"type": "text", "text": "hello"}]
        assert out["stop_reason"] == "end_turn"


class TestStreamToolUse:
    def _session(self):
        a = AnthropicAdapter()
        request = ParsedRequest(model="m", messages=[], params={}, stream=True)
        return a.open_stream(model="m", request=request)

    def test_tool_call_emits_start_delta_stop(self):
        s = self._session()
        raw = s.chunk(Chunk(tool_calls=[{"name": "open_file", "args": {"path": "/tmp/x"},
                                         "id": "c1"}]))
        events = _event_blocks(raw)
        kinds = [e for e, _ in events]
        assert "message_start" in kinds
        assert "content_block_start" in kinds
        assert "content_block_delta" in kinds
        assert "content_block_stop" in kinds

        start = next(d for e, d in events if e == "content_block_start")
        assert start["content_block"] == {
            "type": "tool_use", "id": "c1", "name": "open_file", "input": {}
        }
        delta = next(d for e, d in events if e == "content_block_delta")
        assert delta["delta"]["type"] == "input_json_delta"
        assert json.loads(delta["delta"]["partial_json"]) == {"path": "/tmp/x"}
        stop = next(d for e, d in events if e == "content_block_stop")
        assert stop["index"] == start["index"]

    def test_missing_id_generates_toolu_prefixed_id(self):
        s = self._session()
        raw = s.chunk(Chunk(tool_calls=[{"name": "f", "args": {}, "id": ""}]))
        start = next(d for e, d in _event_blocks(raw) if e == "content_block_start")
        assert start["content_block"]["id"].startswith("toolu_")

    def test_open_text_block_closed_before_tool_use_block(self):
        s = self._session()
        raw = s.chunk(Chunk(content="checking"))
        raw += s.chunk(Chunk(tool_calls=[{"name": "f", "args": {}, "id": "c1"}]))
        events = _event_blocks(raw)
        kinds = [e for e, _ in events]
        text_stop_i = kinds.index("content_block_stop")
        tool_start_i = kinds.index("content_block_start", text_stop_i)
        assert text_stop_i < tool_start_i
        tool_start = events[tool_start_i][1]
        text_start = next(d for e, d in events if e == "content_block_start"
                          and d["content_block"]["type"] == "text")
        assert tool_start["index"] != text_start["index"]

    def test_message_delta_stop_reason_tool_use_after_tool_call(self):
        s = self._session()
        raw = s.chunk(Chunk(tool_calls=[{"name": "f", "args": {}, "id": "c1"}]))
        raw += s.chunk(Chunk(done=True))
        message_delta = next(d for e, d in _event_blocks(raw) if e == "message_delta")
        assert message_delta["delta"]["stop_reason"] == "tool_use"

    def test_message_delta_stop_reason_end_turn_without_tool_use(self):
        s = self._session()
        raw = s.chunk(Chunk(content="hi"))
        raw += s.chunk(Chunk(done=True))
        message_delta = next(d for e, d in _event_blocks(raw) if e == "message_delta")
        assert message_delta["delta"]["stop_reason"] == "end_turn"


class TestParseRequestToolRoundTrip:
    def _parse(self, messages):
        a = AnthropicAdapter()
        return a.parse_request({"model": "m", "messages": messages}, path="/v1/messages")

    def test_assistant_tool_use_block_becomes_neutral_tool_calls(self):
        pr = self._parse([
            {"role": "assistant", "content": [
                {"type": "text", "text": "let me check"},
                {"type": "tool_use", "id": "c1", "name": "open_file",
                 "input": {"path": "/tmp/x"}},
            ]},
        ])
        msg = pr.messages[0]
        assert msg.role == "assistant"
        assert msg.content == "let me check"
        assert msg.tool_calls == [
            {"name": "open_file", "args": {"path": "/tmp/x"}, "id": "c1", "type": "tool_call"}
        ]

    def test_user_tool_result_block_becomes_tool_message(self):
        pr = self._parse([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "42"},
            ]},
        ])
        assert len(pr.messages) == 1
        msg = pr.messages[0]
        assert msg.role == "tool"
        assert msg.tool_call_id == "c1"
        assert msg.content == "42"

    def test_tool_result_content_list_of_text_blocks_flattened(self):
        pr = self._parse([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": [
                    {"type": "text", "text": "a"}, {"type": "text", "text": "b"},
                ]},
            ]},
        ])
        assert pr.messages[0].content == "ab"

    def test_multiple_tool_result_blocks_each_become_separate_message(self):
        pr = self._parse([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "1"},
                {"type": "tool_result", "tool_use_id": "c2", "content": "2"},
            ]},
        ])
        assert len(pr.messages) == 2
        assert [m.tool_call_id for m in pr.messages] == ["c1", "c2"]
        assert [m.content for m in pr.messages] == ["1", "2"]

    def test_surrounding_text_blocks_still_become_normal_content(self):
        pr = self._parse([
            {"role": "user", "content": [
                {"type": "text", "text": "note"},
                {"type": "tool_result", "tool_use_id": "c1", "content": "ok"},
            ]},
        ])
        assert len(pr.messages) == 2
        assert pr.messages[0].role == "user"
        assert pr.messages[0].content == "note"
        assert pr.messages[1].role == "tool"
        assert pr.messages[1].tool_call_id == "c1"
        assert pr.messages[1].content == "ok"
