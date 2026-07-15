# tests/test_agent_thought_mode.py
"""Agent.thought_mode — modo de reasoning POR AGENTE (issue #11).

Precedência: ``thought_stream_mode`` da request > ``Agent.thought_mode`` >
``default_thought_mode`` do adapter.

Não-streaming: o default do SERVIDOR nunca se aplica (comportamento histórico:
reasoning em campo separado, content limpo) — só modos explícitos (request ou
agente) mudam a resposta.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from aixon.registry import get_registry
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.server import Server
from tests._server_fakes import ReasoningAgent


@pytest.fixture(autouse=True)
def _reset_server():
    Server._reset()
    yield
    Server._reset()


def make_reasoning(name: str, *, thought_mode: str | None = None):
    attrs: dict = {"name": name}
    if thought_mode is not None:
        attrs["thought_mode"] = thought_mode
    type("MadeReasoningAgent", (ReasoningAgent,), attrs)
    return get_registry().resolve(name)


def client_with(default_thought_mode: str = "content") -> TestClient:
    return TestClient(
        Server(adapters=[OpenAIAdapter(default_thought_mode=default_thought_mode)]).app
    )


def collect_stream(resp) -> tuple[str, str]:
    """(content agregado, reasoning agregado em delta.reasoning) do SSE."""
    content, reasoning = "", ""
    for line in resp.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        delta = json.loads(line[6:])["choices"][0].get("delta", {})
        content += delta.get("content") or ""
        reasoning += delta.get("reasoning") or ""
    return content, reasoning


def post(client, model, *, stream, **extra):
    return client.post("/v1/chat/completions", json={
        "model": model,
        "stream": stream,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    })


# --- streaming -----------------------------------------------------------

def test_stream_agent_hidden_beats_server_content_default():
    make_reasoning("prog", thought_mode="hidden")
    client = client_with(default_thought_mode="content")
    content, reasoning = collect_stream(post(client, "prog", stream=True))
    assert "<think>" not in content
    assert content == "answer"
    assert reasoning == ""


def test_stream_request_param_beats_agent_mode():
    make_reasoning("prog", thought_mode="hidden")
    client = client_with(default_thought_mode="content")
    content, _ = collect_stream(
        post(client, "prog", stream=True, thought_stream_mode="content")
    )
    assert content.startswith("<think>")
    assert "thinking..." in content


def test_stream_agent_without_mode_inherits_server_default():
    make_reasoning("chatty")  # sem thought_mode
    client = client_with(default_thought_mode="content")
    content, _ = collect_stream(post(client, "chatty", stream=True))
    assert content.startswith("<think>")


# --- não-streaming --------------------------------------------------------

def test_non_stream_agent_hidden_drops_reasoning_field():
    make_reasoning("prog", thought_mode="hidden")
    client = client_with(default_thought_mode="content")
    msg = post(client, "prog", stream=False).json()["choices"][0]["message"]
    assert msg["content"] == "answer"
    assert "reasoning" not in msg


def test_non_stream_request_content_embeds_think():
    make_reasoning("chatty")
    client = client_with(default_thought_mode="content")
    msg = post(
        client, "chatty", stream=False, thought_stream_mode="content"
    ).json()["choices"][0]["message"]
    assert msg["content"].startswith("<think>")
    assert "because" in msg["content"]
    assert "reasoning" not in msg


def test_non_stream_without_modes_is_unchanged():
    # default do SERVIDOR não vaza para o não-streaming: reasoning continua
    # em campo separado e o content limpo (comportamento histórico).
    make_reasoning("chatty")
    client = client_with(default_thought_mode="content")
    msg = post(client, "chatty", stream=False).json()["choices"][0]["message"]
    assert msg["content"] == "answer"
    assert msg["reasoning"] == "because"
