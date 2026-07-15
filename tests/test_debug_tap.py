# tests/test_debug_tap.py
"""Debug tap de requests no Server (issue #10).

Ligado por env ``AIXON_DEBUG_REQUESTS`` (truthy), grava um registro JSONL por
POST de chat em ``AIXON_DEBUG_REQUESTS_DIR`` (default ``./aixon-debug``):
body verbatim + agente resolvido + resposta (payload não-stream ou linhas SSE).
Headers NUNCA são gravados (Authorization jamais toca o arquivo).
"""
from __future__ import annotations

import json
from pathlib import Path

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


def records(dirpath: Path) -> list[dict]:
    lines = []
    for f in sorted(dirpath.glob("*.jsonl")):
        lines += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return lines


def post(client, *, stream: bool):
    return client.post(
        "/v1/chat/completions",
        json={"model": "echo", "stream": stream,
              "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer super-secret-token"},
    )


def test_tap_disabled_by_default_writes_nothing(client, tmp_path, monkeypatch):
    monkeypatch.delenv("AIXON_DEBUG_REQUESTS", raising=False)
    monkeypatch.setenv("AIXON_DEBUG_REQUESTS_DIR", str(tmp_path))
    assert post(client, stream=False).status_code == 200
    assert records(tmp_path) == []


def test_tap_records_non_stream_request_and_response(client, tmp_path, monkeypatch):
    monkeypatch.setenv("AIXON_DEBUG_REQUESTS", "1")
    monkeypatch.setenv("AIXON_DEBUG_REQUESTS_DIR", str(tmp_path))
    assert post(client, stream=False).status_code == 200
    recs = records(tmp_path)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["agent"] == "echo"
    assert rec["body"]["messages"][0]["content"] == "hi"
    assert "echo:hi" in json.dumps(rec["response"])


def test_tap_records_stream_chunks(client, tmp_path, monkeypatch):
    monkeypatch.setenv("AIXON_DEBUG_REQUESTS", "1")
    monkeypatch.setenv("AIXON_DEBUG_REQUESTS_DIR", str(tmp_path))
    assert post(client, stream=True).status_code == 200
    recs = records(tmp_path)
    assert len(recs) == 1
    assert recs[0]["stream"] is True
    # o record guarda as linhas SSE cruas: reconstruir o content dos deltas
    content = ""
    for sse in recs[0]["response"]:
        for line in sse.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                delta = json.loads(line[6:])["choices"][0].get("delta", {})
                content += delta.get("content") or ""
    assert content == "echo:hi"


def test_tap_never_records_authorization(client, tmp_path, monkeypatch):
    monkeypatch.setenv("AIXON_DEBUG_REQUESTS", "1")
    monkeypatch.setenv("AIXON_DEBUG_REQUESTS_DIR", str(tmp_path))
    post(client, stream=False)
    post(client, stream=True)
    raw = "".join(f.read_text() for f in tmp_path.glob("*.jsonl"))
    assert "super-secret-token" not in raw
    assert "uthorization" not in raw


# --- filtro por agente (reduz blast radius em servidor compartilhado) ------

def test_tap_agent_filter_records_only_named_agents(tmp_path, monkeypatch):
    make_echo("alpha")
    make_echo("beta")
    client = TestClient(Server(adapters=[OpenAIAdapter()]).app)
    monkeypatch.setenv("AIXON_DEBUG_REQUESTS", "alpha")
    monkeypatch.setenv("AIXON_DEBUG_REQUESTS_DIR", str(tmp_path))
    for model in ("alpha", "beta", "alpha"):
        r = client.post("/v1/chat/completions", json={
            "model": model, "stream": False,
            "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
    recs = records(tmp_path)
    assert [r["agent"] for r in recs] == ["alpha", "alpha"]


def test_tap_agent_filter_accepts_comma_list_with_spaces(tmp_path, monkeypatch):
    make_echo("alpha")
    make_echo("beta")
    client = TestClient(Server(adapters=[OpenAIAdapter()]).app)
    monkeypatch.setenv("AIXON_DEBUG_REQUESTS", "alpha, beta")
    monkeypatch.setenv("AIXON_DEBUG_REQUESTS_DIR", str(tmp_path))
    for model in ("alpha", "beta"):
        client.post("/v1/chat/completions", json={
            "model": model, "stream": False,
            "messages": [{"role": "user", "content": "hi"}]})
    assert sorted(r["agent"] for r in records(tmp_path)) == ["alpha", "beta"]
