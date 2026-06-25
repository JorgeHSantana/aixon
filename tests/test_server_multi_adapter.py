# tests/test_server_multi_adapter.py
"""Mounting more than one ProtocolAdapter on one Server (Issue #2).

`mount_prefix` lets two dialects coexist whose ``routes()`` would otherwise
collide on a shared path (both adapters declare ``GET /v1/models``). Without a
prefix the Server fails loudly instead of silently shadowing one route.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aixon.exceptions import AixonError
from aixon.server.adapters.anthropic import AnthropicAdapter
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.server import Server
from tests._server_fakes import make_echo


@pytest.fixture(autouse=True)
def _reset_server():
    Server._reset()
    yield
    Server._reset()


def _both():
    return Server(
        adapters=[OpenAIAdapter(), AnthropicAdapter(mount_prefix="/anthropic")]
    ).app


def test_two_dialects_coexist_under_distinct_prefixes():
    make_echo("echo", description="d")
    client = TestClient(_both())

    # OpenAI stays at its canonical paths.
    assert client.get("/v1/models").status_code == 200
    r = client.post(
        "/v1/chat/completions",
        json={"model": "echo", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "echo:hi"

    # Anthropic lives under /anthropic — its /v1/models no longer clashes.
    assert client.get("/anthropic/v1/models").status_code == 200
    r2 = client.post(
        "/anthropic/v1/messages",
        json={"model": "echo", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r2.status_code == 200
    assert r2.json()["type"] == "message"
    assert r2.json()["content"][0]["text"] == "echo:hi"


def test_each_models_route_keeps_its_own_dialect_shape():
    make_echo("echo", description="d")
    client = TestClient(_both())
    openai_models = client.get("/v1/models").json()
    anthropic_models = client.get("/anthropic/v1/models").json()
    assert openai_models["object"] == "list"        # OpenAI shape
    assert "object" not in anthropic_models          # Anthropic shape: {"data": [...]}


def test_colliding_adapters_raise_a_clear_error():
    make_echo("echo", description="d")
    # Both adapters claim GET /v1/models with no prefix.
    server = Server(adapters=[OpenAIAdapter(), AnthropicAdapter()])
    with pytest.raises(AixonError, match="GET /v1/models"):
        _ = server.app


def test_models_route_is_public_at_its_mounted_path():
    make_echo("echo", description="d")
    _both()  # build the app on the singleton with both adapters
    # Public-path set is what the auth middleware exempts; both mounted GET
    # routes must be in it (their prefixed paths, not the bare /v1/models).
    server = Server()  # singleton — same instance just built
    public = server._public_paths()
    assert "/v1/models" in public
    assert "/anthropic/v1/models" in public


def test_default_single_adapter_is_unchanged():
    make_echo("echo", description="d")
    client = TestClient(Server(adapters=[OpenAIAdapter()]).app)
    assert client.get("/v1/models").status_code == 200
    assert client.get("/anthropic/v1/models").status_code == 404
