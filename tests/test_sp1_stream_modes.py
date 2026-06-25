from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.server import Server
from tests._server_fakes import ReasoningAgent
from aixon.registry import get_registry


@pytest.fixture(autouse=True)
def _reset_server():
    Server._reset()
    yield
    Server._reset()


def _register_reasoning(name="rz"):
    cls = type("RzAgent", (ReasoningAgent,), {"name": name})
    return get_registry().resolve(cls.name)


def _deltas(text: str) -> list[dict]:
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block.startswith("data: "):
            continue
        payload = block[len("data: "):]
        if payload == "[DONE]":
            continue
        out.append(json.loads(payload))
    return out


def _stream(client, body):
    body = {**body, "stream": True}
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        return r.read().decode()


@pytest.fixture
def client():
    _register_reasoning("rz")
    return TestClient(Server(adapters=[OpenAIAdapter()]).app)


def test_content_mode_wraps_reasoning_in_think(client):
    # ReasoningAgent streams: reasoning "thinking...", content "ans","wer", done.
    raw = _stream(client, {"model": "rz", "messages": [{"role": "user", "content": "q"}],
                           "thought_stream_mode": "content"})
    content = "".join(d["choices"][0]["delta"].get("content", "")
                      for d in _deltas(raw) if d["choices"])
    assert "<think>" in content and "</think>" in content
    assert "thinking..." in content
    assert content.index("</think>") < content.index("answer".replace("answer", "ans"))


def test_custom_mode_uses_reasoning_delta(client):
    raw = _stream(client, {"model": "rz", "messages": [{"role": "user", "content": "q"}],
                           "thought_stream_mode": "custom"})
    deltas = [d["choices"][0]["delta"] for d in _deltas(raw) if d["choices"]]
    assert any("reasoning" in d for d in deltas)
    assert not any("<think>" in d.get("content", "") for d in deltas)


def test_hidden_mode_drops_reasoning(client):
    raw = _stream(client, {"model": "rz", "messages": [{"role": "user", "content": "q"}],
                           "thought_stream_mode": "hidden"})
    deltas = [d["choices"][0]["delta"] for d in _deltas(raw) if d["choices"]]
    assert not any("reasoning" in d for d in deltas)
    assert not any("<think>" in d.get("content", "") for d in deltas)
    content = "".join(d.get("content", "") for d in deltas)
    assert content == "answer"


def test_default_mode_is_content(client):
    raw = _stream(client, {"model": "rz", "messages": [{"role": "user", "content": "q"}]})
    content = "".join(d["choices"][0]["delta"].get("content", "")
                      for d in _deltas(raw) if d["choices"])
    assert "<think>" in content


def test_include_usage_emits_usage_chunk(client):
    pytest.importorskip("tiktoken")
    raw = _stream(client, {"model": "rz", "messages": [{"role": "user", "content": "q"}],
                           "stream_options": {"include_usage": True}})
    usage_chunks = [d for d in _deltas(raw) if d.get("usage")]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["usage"]["total_tokens"] > 0
    assert usage_chunks[0]["choices"] == []
