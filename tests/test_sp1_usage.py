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


def test_build_usage_counts_tokens():
    pytest.importorskip("tiktoken")
    from aixon.server.usage import build_usage

    usage = build_usage("gpt-5.4", "hello world", "hi there")
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_build_usage_empty_when_counter_unavailable(monkeypatch):
    from aixon.server import usage as usage_mod

    monkeypatch.setattr(usage_mod, "count_tokens", lambda model, text: None)
    assert usage_mod.build_usage("gpt-5.4", "a", "b") == {}


def test_non_stream_response_includes_usage():
    pytest.importorskip("tiktoken")
    make_echo("echo")
    client = TestClient(Server(adapters=[OpenAIAdapter()]).app)
    r = client.post("/v1/chat/completions", json={
        "model": "echo", "messages": [{"role": "user", "content": "hi there"}],
    })
    usage = r.json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
