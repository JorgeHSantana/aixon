from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.server import Server
from tests._server_fakes import make_echo


@pytest.fixture(autouse=True)
def _reset_server():
    Server._reset()
    yield
    Server._reset()


@pytest.fixture
def client():
    make_echo("echo")
    return TestClient(Server(adapters=[OpenAIAdapter()]).app)


def test_routes_include_non_v1_aliases():
    assert ("POST", "/chat/completions") in OpenAIAdapter().routes()
    assert ("GET", "/models") in OpenAIAdapter().routes()
    assert ("POST", "/v1/chat/completions") in OpenAIAdapter().routes()
    assert ("GET", "/v1/models") in OpenAIAdapter().routes()


def test_non_v1_chat_completions_works(client):
    r = client.post("/chat/completions", json={
        "model": "echo", "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "echo:hi"


def test_non_v1_models_is_public_and_ok(client):
    r = client.get("/models")
    assert r.status_code == 200
    assert r.json()["object"] == "list"
