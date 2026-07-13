"""M3 — Anthropic production stream session: interleave block reopening, error
closes the envelope before message_stop, and an e2e proof over TestClient."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from aixon.agent import Agent
from aixon.message import Chunk
from aixon.server.adapters.anthropic import AnthropicAdapter
from aixon.server.protocol import ParsedRequest
from aixon.server.server import Server


@pytest.fixture(autouse=True)
def _reset_server():
    Server._reset()
    yield
    Server._reset()


def _session():
    request = ParsedRequest(model="m", messages=[], params={}, stream=True)
    return AnthropicAdapter().open_stream(model="m", request=request)


def _events(raw: str) -> list[dict]:
    out = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for line in block.split("\n"):
            if line.startswith("data: "):
                out.append(json.loads(line[len("data: "):]))
    return out


# --- 1. interleave: reasoning arriving AFTER text is open reopens a NEW ----
# thinking block at the next index instead of emitting against the closed one.
def test_reasoning_after_text_reopens_new_thinking_block():
    s = _session()
    raw = s.chunk(Chunk(content="answer"))
    raw += s.chunk(Chunk(reasoning="afterthought"))
    raw += s.chunk(Chunk(done=True))
    raw += s.done()

    events = _events(raw)
    starts = [e for e in events if e["type"] == "content_block_start"]
    assert [b["content_block"]["type"] for b in starts] == ["text", "thinking"]
    text_start, thinking_start = starts
    assert text_start["index"] != thinking_start["index"]

    # The text block must have been closed BEFORE the new thinking block
    # opened (block sequencing, not a delta against a closed block).
    stops = [e for e in events if e["type"] == "content_block_stop"]
    text_stop = next(e for e in stops if e["index"] == text_start["index"])
    assert events.index(text_stop) < events.index(thinking_start)

    # The reopened thinking delta targets the NEW index, not the stale one.
    thinking_deltas = [
        e for e in events
        if e["type"] == "content_block_delta" and e["delta"]["type"] == "thinking_delta"
    ]
    assert thinking_deltas
    assert thinking_deltas[0]["index"] == thinking_start["index"]

    # Both blocks get closed by the time the stream ends.
    assert len(stops) == 2


# --- 2. error() closes the open block, then done() still emits message_stop -
def test_error_closes_open_block_then_done_emits_message_stop():
    s = _session()
    raw = s.chunk(Chunk(content="partial"))
    err = s.error(RuntimeError("internal secret"))
    tail = s.done()

    events = _events(raw + err + tail)
    types = [e["type"] for e in events]

    assert types[0] == "message_start"
    # The block opened by the content chunk is closed by error(), BEFORE the
    # error event, and message_stop from done() still comes after.
    assert "content_block_stop" in types
    stop_i = types.index("content_block_stop")
    error_i = types.index("error")
    stop_event_i = next(i for i, t in enumerate(types) if t == "message_stop")
    assert stop_i < error_i < stop_event_i

    error_event = next(e for e in events if e["type"] == "error")
    assert "internal secret" not in json.dumps(error_event)


def test_error_is_noop_safe_when_no_block_is_open():
    s = _session()
    # Nothing streamed yet: error() must not blow up on a fresh session.
    err = s.error(RuntimeError("boom"))
    events = _events(err)
    assert events[-1]["type"] == "error"


# --- 3. e2e: TestClient against an Anthropic-mounted server, agent raises ---
# mid-stream -> SSE body carries the full ordered envelope, error is generic.
def _register_boom_anthropic():
    class BoomAnthropicAgent(Agent):
        name = "boom-anthropic"

        def invoke(self, messages):
            raise RuntimeError("provider exploded")

        def stream(self, messages):
            yield Chunk(content="partial")
            raise RuntimeError("provider exploded")

    return BoomAnthropicAgent


def _anthropic_client():
    return TestClient(
        Server(adapters=[AnthropicAdapter()]).app, raise_server_exceptions=False
    )


def test_e2e_anthropic_stream_error_envelope_order_and_is_generic():
    _register_boom_anthropic()
    r = _anthropic_client().post(
        "/v1/messages",
        json={
            "model": "boom-anthropic",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    events = _events(r.text)
    types = [e["type"] for e in events]

    assert types[0] == "message_start"
    assert "content_block_start" in types
    assert "content_block_stop" in types
    assert "error" in types
    assert types[-1] == "message_stop"

    start_i = types.index("content_block_start")
    stop_i = types.index("content_block_stop")
    error_i = types.index("error")
    assert start_i < stop_i < error_i < len(types) - 1

    error_event = next(e for e in events if e["type"] == "error")
    assert "provider exploded" not in json.dumps(error_event)
    assert error_event["error"]["message"] == (
        "The server encountered an error while generating the response."
    )


# ── final-review 0.1.14: pins for hand-verified paths + malformed input ──────

def test_parse_non_dict_tool_use_input_degrades_to_empty_dict():
    # Valid JSON that isn't an object (list/string) must become args={} —
    # one malformed history entry must not 500 the request (same contract
    # as _neutral_tool_calls on the OpenAI adapter).
    pr = AnthropicAdapter().parse_request(
        {"model": "m", "messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "f", "input": [1, 2]},
                {"type": "tool_use", "id": "t2", "name": "g", "input": "oops"},
                {"type": "tool_use", "id": "t3", "name": "h",
                 "input": {"x": 1}},
            ]},
        ]},
        path="/v1/messages")
    calls = pr.messages[0].tool_calls
    assert [c["args"] for c in calls] == [{}, {}, {"x": 1}]


def test_stream_error_after_tool_use_emits_clean_envelope():
    # tool_use blocks self-close, so an error right after one must emit the
    # error event without a dangling content_block_stop.
    s = _session()
    raw = s.chunk(Chunk(tool_calls=[{"name": "f", "args": {}, "id": "t1"}]))
    raw += s.error(RuntimeError("boom"))
    raw += s.done()
    text = raw
    assert text.index("content_block_stop") < text.index("event: error")
    assert text.index("event: error") < text.index("message_stop")
    assert "boom" not in text  # generic payload, no internal leak


def test_stream_reasoning_after_tool_use_opens_new_block():
    s = _session()
    raw = s.chunk(Chunk(content="answer"))
    raw += s.chunk(Chunk(tool_calls=[{"name": "f", "args": {}, "id": "t1"}]))
    raw += s.chunk(Chunk(reasoning="thinking more"))
    raw += s.chunk(Chunk(done=True))
    events = _events(raw)
    starts = [(e["index"], e["content_block"]["type"])
              for e in events if e["type"] == "content_block_start"]
    # text(0) -> tool_use(1) -> thinking(2): indices strictly increasing,
    # no reuse of a closed block's index.
    assert starts == [(0, "text"), (1, "tool_use"), (2, "thinking")]
