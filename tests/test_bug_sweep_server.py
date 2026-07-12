"""Regression tests for the 2026-07-09 bug-sweep audit — server/adapters."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from aixon.message import Chunk
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.server import Server
from tests._server_fakes import make_echo


@pytest.fixture(autouse=True)
def _reset_server():
    Server._reset()
    yield
    Server._reset()


@pytest.fixture
def client_with_echo():
    make_echo("echo")
    return TestClient(
        Server(adapters=[OpenAIAdapter()]).app, raise_server_exceptions=False
    )


def _openai_session(mode: str | None = None, *, include_usage: bool = False):
    from aixon.server.protocol import ParsedRequest

    params: dict = {}
    if mode:
        params["thought_stream_mode"] = mode
    if include_usage:
        params["stream_options"] = {"include_usage": True}
    request = ParsedRequest(model="m", messages=[], params=params, stream=True)
    return OpenAIAdapter().open_stream(model="m", request=request)


def _deltas(raw: str) -> list[dict]:
    out = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block.startswith("data: "):
            continue
        payload = block[len("data: "):]
        if payload == "[DONE]":
            continue
        out.append(json.loads(payload))
    return out


def _anthropic_events(raw: str) -> list[dict]:
    out = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for line in block.split("\n"):
            if line.startswith("data: "):
                out.append(json.loads(line[len("data: "):]))
    return out


# S1 — arguments "null"/"[1,2]" degradam para {}
def test_neutral_tool_calls_non_object_arguments_degrade_to_empty_dict():
    from aixon.server.adapters.openai import _neutral_tool_calls
    calls = [
        {"id": "a", "function": {"name": "f", "arguments": "null"}},
        {"id": "b", "function": {"name": "g", "arguments": "[1, 2]"}},
        {"id": "c", "function": {"name": "h", "arguments": "{\"x\": 1}"}},
    ]
    out = _neutral_tool_calls(calls)
    assert out[0]["args"] == {}
    assert out[1]["args"] == {}
    assert out[2]["args"] == {"x": 1}


# S2 — role developer mapeia para SystemMessage
def test_to_langchain_developer_role_maps_to_system():
    from aixon._interop.messages import to_langchain
    from aixon.message import Message
    from langchain_core.messages import SystemMessage
    out = to_langchain([Message(role="developer", content="be terse")])
    assert isinstance(out[0], SystemMessage)


# S3 — corpo com messages malformadas -> 400 JSON, não 500
def test_malformed_message_entry_returns_400(client_with_echo):
    resp = client_with_echo.post(
        "/v1/chat/completions",
        json={"model": "echo", "messages": ["hi"]},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_unknown_role_returns_400(client_with_echo):
    resp = client_with_echo.post(
        "/v1/chat/completions",
        json={"model": "echo",
              "messages": [{"role": "banana", "content": "hi"}]},
    )
    assert resp.status_code == 400


# S4 — finish_reason volta a "stop" quando o stream termina em texto
def test_finish_reason_stop_when_content_follows_tool_calls():
    s = _openai_session()
    raw = s.chunk(Chunk(tool_calls=[{"name": "f", "args": {}, "id": "c1"}]))
    raw += s.chunk(Chunk(content="answer"))
    raw += s.chunk(Chunk(done=True))
    finishes = [
        d["choices"][0]["finish_reason"]
        for d in _deltas(raw)
        if d.get("choices") and d["choices"][0].get("finish_reason") is not None
    ]
    assert finishes[-1] == "stop"


# S5 — modo content: reasoning depois de content fecha o <think> no próximo content
def test_think_block_closes_when_content_resumes_after_reasoning():
    s = _openai_session(mode="content")
    raw = s.chunk(Chunk(reasoning="think1"))
    raw += s.chunk(Chunk(content="answer1"))
    raw += s.chunk(Chunk(reasoning="think2"))
    raw += s.chunk(Chunk(content="answer2"))
    raw += s.chunk(Chunk(done=True))
    content = "".join(
        d["choices"][0]["delta"].get("content", "")
        for d in _deltas(raw)
        if d.get("choices")
    )
    assert "</think>\nanswer2" in content


# S6 — id e created estáveis dentro do mesmo stream
def test_stream_chunk_id_stable_within_session():
    s = _openai_session()
    raw = s.chunk(Chunk(content="a"))
    raw += s.chunk(Chunk(content="b"))
    raw += s.chunk(Chunk(done=True))
    ids = {d["id"] for d in _deltas(raw)}
    assert len(ids) == 1


# S7 — primeiro delta carrega role assistant
def test_first_stream_delta_carries_assistant_role():
    s = _openai_session()
    raw = s.chunk(Chunk(content="hi"))
    deltas = _deltas(raw)
    assert deltas[0]["choices"][0]["delta"]["role"] == "assistant"


# S8 — include_usage: chunks intermediários com "usage": null
def test_include_usage_intermediate_chunks_have_null_usage():
    s = _openai_session(include_usage=True)
    raw = s.chunk(Chunk(content="hi"))
    raw += s.chunk(Chunk(done=True))
    deltas = _deltas(raw)
    assert deltas  # sanity: something was emitted
    for d in deltas:
        assert "usage" in d
        assert d["usage"] is None


# S9 — erro mid-stream usa o dialeto do adapter e não vaza str(exc)
def test_stream_error_event_is_adapter_shaped_and_generic():
    from aixon.server.adapters.anthropic import AnthropicAdapter

    exc = RuntimeError("super secret internal detail")

    oa_line = OpenAIAdapter().format_stream_error(exc)
    assert oa_line.startswith("data: ")
    oa_payload = json.loads(oa_line[len("data: "):].strip())
    assert oa_payload["error"]["type"] == "server_error"
    assert "secret internal detail" not in oa_line

    an_line = AnthropicAdapter().format_stream_error(exc)
    assert an_line.startswith("event: error")
    assert "secret internal detail" not in an_line


# S10 — stream Anthropic emite o envelope completo da spec
def test_anthropic_stream_emits_message_start_and_block_envelope():
    from aixon.server.adapters.anthropic import AnthropicAdapter
    from aixon.server.protocol import ParsedRequest

    a = AnthropicAdapter()
    request = ParsedRequest(model="m", messages=[], params={}, stream=True)
    s = a.open_stream(model="m", request=request)

    raw = s.chunk(Chunk(reasoning="think"))
    raw += s.chunk(Chunk(content="answer"))
    raw += s.chunk(Chunk(done=True))
    raw += s.done()

    events = _anthropic_events(raw)
    types = [e["type"] for e in events]

    assert types[0] == "message_start"
    assert types[-1] == "message_stop"

    starts = [e for e in events if e["type"] == "content_block_start"]
    thinking_start = next(e for e in starts if e["content_block"]["type"] == "thinking")
    text_start = next(e for e in starts if e["content_block"]["type"] == "text")
    assert thinking_start["index"] != text_start["index"]
    assert events.index(thinking_start) < events.index(text_start)

    stops = [e for e in events if e["type"] == "content_block_stop"]
    assert len(stops) >= 2  # thinking block closed, then text block closed at done

    message_deltas = [e for e in events if e["type"] == "message_delta"]
    assert message_deltas
    assert "output_tokens" in message_deltas[0]["usage"]
    assert message_deltas[0]["delta"]["stop_reason"] == "end_turn"


# S11 — tools é transport field no adapter Anthropic
def test_anthropic_parse_request_extracts_tools_not_params():
    from aixon.server.adapters.anthropic import AnthropicAdapter
    pr = AnthropicAdapter().parse_request(
        {"model": "m", "messages": [],
         "tools": [{"name": "t", "input_schema": {"type": "object"}}]},
        path="/v1/messages")
    # M2: ParsedRequest.tools is always OpenAI-shaped, regardless of adapter.
    assert pr.tools == [{"type": "function",
                         "function": {"name": "t", "description": "",
                                      "parameters": {"type": "object"}}}]
    assert "tools" not in pr.params
