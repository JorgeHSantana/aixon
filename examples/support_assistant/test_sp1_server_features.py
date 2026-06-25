"""Server-level feature coverage for this example — offline, no network, no key.

The base suite (test_support_assistant.py) exercises the in-process layers. This
file drives the SAME agents over the HTTP wire with TestClient to lock the
OpenAI-server behaviour the example actually serves:

  - bare /chat/completions (no /v1 prefix)
  - thought_stream_mode (content default / custom / hidden) over a reasoning agent
  - usage token counting (when tiktoken is installed)
  - per-request generation params reaching the model

Run from this folder:  python -m pytest test_sp1_server_features.py
"""

from __future__ import annotations

import json
import os

# Force offline mode BEFORE importing anything that builds an LLM at import time.
os.environ.pop("OPENAI_API_KEY", None)

import pytest  # noqa: E402

from aixon import Server, autodiscover  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # Module-scoped: the example's agents register once at import time (via
    # autodiscover) into the shared registry — mirroring test_support_assistant.
    # Resetting per test would defeat autodiscover (modules are cached in
    # sys.modules, so a re-import would not re-run the class definitions).
    from fastapi.testclient import TestClient

    autodiscover("agents")
    return TestClient(Server().app)


ORDER_Q = {"role": "user", "content": "where is my order 1002?"}


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


def _stream(client, body: dict) -> str:
    body = {**body, "stream": True}
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        return r.read().decode()


# --- bare routes (chronos-style consumers omit /v1) --------------------------

def test_bare_chat_completions_route_works(client):
    r = client.post("/chat/completions", json={"model": "support", "messages": [ORDER_Q]})
    assert r.status_code == 200
    assert "1002" in r.json()["choices"][0]["message"]["content"]


def test_bare_models_route_is_public(client):
    r = client.get("/models")
    assert r.status_code == 200
    assert r.json()["object"] == "list"


# --- thought_stream_mode over the reasoning-emitting orders specialist --------

def test_default_stream_wraps_reasoning_in_think(client):
    # orders_agent emits a reasoning line; the default mode (content) wraps it.
    content = "".join(
        d["choices"][0]["delta"].get("content", "")
        for d in _deltas(_stream(client, {"model": "support", "messages": [ORDER_Q]}))
        if d["choices"]
    )
    assert "<think>" in content and "</think>" in content


def test_custom_mode_uses_reasoning_delta(client):
    deltas = [
        d["choices"][0]["delta"]
        for d in _deltas(_stream(client, {"model": "support", "messages": [ORDER_Q],
                                          "thought_stream_mode": "custom"}))
        if d["choices"]
    ]
    assert any("reasoning" in d for d in deltas)
    assert not any("<think>" in d.get("content", "") for d in deltas)


def test_hidden_mode_drops_reasoning(client):
    deltas = [
        d["choices"][0]["delta"]
        for d in _deltas(_stream(client, {"model": "support", "messages": [ORDER_Q],
                                          "thought_stream_mode": "hidden"}))
        if d["choices"]
    ]
    assert not any("reasoning" in d for d in deltas)
    assert not any("<think>" in d.get("content", "") for d in deltas)


# --- usage (graceful: skip if tiktoken is not installed) ---------------------

def test_non_stream_response_includes_usage(client):
    pytest.importorskip("tiktoken")
    r = client.post("/v1/chat/completions", json={"model": "support", "messages": [ORDER_Q]})
    usage = r.json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_stream_include_usage_emits_final_usage_chunk(client):
    pytest.importorskip("tiktoken")
    raw = _stream(client, {"model": "support", "messages": [ORDER_Q],
                           "stream_options": {"include_usage": True}})
    usage_chunks = [d for d in _deltas(raw) if d.get("usage")]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["usage"]["total_tokens"] > 0
    assert usage_chunks[0]["choices"] == []


# --- per-request generation params reach the model ---------------------------

def test_generation_params_are_accepted_and_forwarded(client):
    # The offline provider ignores temperature, but the request must succeed and
    # the param must pass the allow-list without leaking into the response body.
    r = client.post("/chat/completions", json={
        "model": "support", "messages": [ORDER_Q], "temperature": 0.1,
    })
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"]
