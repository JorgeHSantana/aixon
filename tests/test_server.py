# tests/test_server.py
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from aixon.message import Message
from aixon.server.adapters.anthropic import AnthropicAdapter
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.server import Server
from tests._server_fakes import ReasoningAgent, make_echo


@pytest.fixture(autouse=True)
def _reset_server():
    Server._reset()
    yield
    Server._reset()


@pytest.fixture
def client():
    make_echo("echo", aliases=["echo-alias"], description="d")
    return TestClient(Server(adapters=[OpenAIAdapter()]).app)


# --- health + models -----------------------------------------------------
def test_health_is_public_and_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_models_lists_registered_agent_and_alias(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = [d["id"] for d in r.json()["data"]]
    assert "echo" in ids and "echo-alias" in ids


# --- OpenAI non-stream ---------------------------------------------------
def test_chat_completions_non_stream(client):
    r = client.post("/v1/chat/completions", json={
        "model": "echo",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "echo:hi"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_model_resolution_via_alias(client):
    r = client.post("/v1/chat/completions", json={
        "model": "echo-alias",
        "messages": [{"role": "user", "content": "yo"}],
    })
    assert r.json()["choices"][0]["message"]["content"] == "echo:yo"


def test_unknown_model_with_multiple_agents_is_404():
    make_echo("a")
    make_echo("b")
    c = TestClient(Server(adapters=[OpenAIAdapter()]).app)
    r = c.post("/v1/chat/completions", json={
        "model": "nope", "messages": [{"role": "user", "content": "x"}],
    })
    assert r.status_code == 404
    assert "error" in r.json()


# --- OpenAI stream -------------------------------------------------------
def _sse_data_lines(text: str) -> list[str]:
    return [l[len("data: ") :] for l in text.splitlines() if l.startswith("data: ")]


def test_chat_completions_stream(client):
    r = client.post("/v1/chat/completions", json={
        "model": "echo",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    datas = _sse_data_lines(r.text)
    assert datas[-1] == "[DONE]"
    parsed = [json.loads(d) for d in datas if d != "[DONE]"]
    assert all(p["object"] == "chat.completion.chunk" for p in parsed)
    content = "".join(
        p["choices"][0]["delta"].get("content", "") for p in parsed
    )
    assert content == "echo:hi"
    assert parsed[-1]["choices"][0]["finish_reason"] == "stop"


# --- neutral boundary: no vendor type leaks into the agent ---------------
def test_agent_only_ever_receives_neutral_messages(client):
    client.post("/v1/chat/completions", json={
        "model": "echo",
        "messages": [{"role": "user", "content": "leak-check"}],
    })
    agent = make_echo  # not used; resolve the live instance instead
    from aixon.registry import get_registry
    inst = get_registry().resolve("echo")
    assert inst.seen is not None
    assert all(isinstance(m, Message) for m in inst.seen)
    # No dict / vendor body slipped through.
    assert not any(isinstance(m, dict) for m in inst.seen)


# --- Anthropic adapter mounted on the same server ------------------------
@pytest.fixture
def anthropic_client():
    make_echo("echo")
    return TestClient(Server(adapters=[AnthropicAdapter()]).app)


def test_anthropic_messages_non_stream(anthropic_client):
    r = anthropic_client.post("/v1/messages", json={
        "model": "echo",
        "system": "be terse",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 50,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message"
    assert body["content"][0]["text"] == "echo:hi"
    assert body["stop_reason"] == "end_turn"


def test_anthropic_messages_stream_named_events(anthropic_client):
    r = anthropic_client.post("/v1/messages", json={
        "model": "echo",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    assert r.status_code == 200
    assert "event: content_block_delta" in r.text
    assert "event: message_stop" in r.text
    assert "[DONE]" not in r.text  # Anthropic has no [DONE] sentinel


# --- auth ON / OFF -------------------------------------------------------
def test_auth_off_when_env_unset(monkeypatch):
    monkeypatch.delenv("AUTH_API_KEY", raising=False)
    make_echo("echo")
    c = TestClient(Server(adapters=[OpenAIAdapter()]).app)
    r = c.post("/v1/chat/completions", json={
        "model": "echo", "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200


def test_auth_on_rejects_missing_and_bad_bearer(monkeypatch):
    monkeypatch.setenv("AUTH_API_KEY", "secret123")
    make_echo("echo")
    c = TestClient(Server(adapters=[OpenAIAdapter()]).app)
    payload = {"model": "echo", "messages": [{"role": "user", "content": "hi"}]}
    assert c.post("/v1/chat/completions", json=payload).status_code == 401
    assert c.post(
        "/v1/chat/completions", json=payload, headers={"Authorization": "Bearer wrong"}
    ).status_code == 401


def test_auth_on_accepts_good_bearer_and_keeps_public_routes_open(monkeypatch):
    monkeypatch.setenv("AUTH_API_KEY", "secret123")
    make_echo("echo")
    c = TestClient(Server(adapters=[OpenAIAdapter()]).app)
    payload = {"model": "echo", "messages": [{"role": "user", "content": "hi"}]}
    assert c.post(
        "/v1/chat/completions", json=payload, headers={"Authorization": "Bearer secret123"}
    ).status_code == 200
    # public routes stay open even with auth on
    assert c.get("/health").status_code == 200
    assert c.get("/v1/models").status_code == 200
