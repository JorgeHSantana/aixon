# tests/test_tool_choice_transport.py
"""#18b — tool_choice vira transporte: publicado no contextvar, 400 sem tools."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aixon.message import Message
from aixon.runtime import current_tool_choice
from aixon.server.server import Server
from tests._fakes import make_echo_agent

WIRE_TOOLS = [{"type": "function",
               "function": {"name": "f1", "parameters": {"type": "object"}}}]


@pytest.fixture(autouse=True)
def _reset_server(monkeypatch):
    monkeypatch.delenv("AUTH_API_KEY", raising=False)
    Server._reset()
    yield
    Server._reset()


def _client_and_probe():
    """Echo agent que grava o tool_choice visto durante o invoke."""
    seen: dict = {}
    agent = make_echo_agent("tc18probe")
    original_invoke = type(agent).invoke

    def probing_invoke(self, messages: list[Message]) -> Message:
        seen["tool_choice"] = current_tool_choice()
        return original_invoke(self, messages)

    type(agent).invoke = probing_invoke
    return TestClient(Server().app), seen


def test_tool_choice_publicado_no_contextvar():
    client, seen = _client_and_probe()
    resp = client.post("/v1/chat/completions", json={
        "model": "tc18probe",
        "messages": [{"role": "user", "content": "oi"}],
        "tools": WIRE_TOOLS,
        "tool_choice": {"type": "function", "function": {"name": "f1"}},
    })
    assert resp.status_code == 200
    assert seen["tool_choice"]["function"]["name"] == "f1"


def test_tool_choice_sem_tools_da_400_explicito():
    client, _ = _client_and_probe()
    resp = client.post("/v1/chat/completions", json={
        "model": "tc18probe",
        "messages": [{"role": "user", "content": "oi"}],
        "tool_choice": "auto",
    })
    assert resp.status_code == 400
    assert "tool_choice" in resp.text


def test_sem_tool_choice_contextvar_none():
    client, seen = _client_and_probe()
    resp = client.post("/v1/chat/completions", json={
        "model": "tc18probe",
        "messages": [{"role": "user", "content": "oi"}],
    })
    assert resp.status_code == 200
    assert seen["tool_choice"] is None
