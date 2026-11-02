"""0.1.29 — o exemplo ``examples/request_hook`` roda contra um provedor fake:
os três ``curl`` do README viram chamadas do TestClient."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aixon.providers import base as providers_base
from aixon.server.server import Server
from tests._fakes import RecordingProvider

MAIN = Path(__file__).resolve().parents[1] / "examples" / "request_hook" / "main.py"
URL = "/v1/chat/completions"


@pytest.fixture
def rec_openai(monkeypatch):
    """Substitui o provedor "openai" por um que grava os builds: sem rede,
    sem chave de API."""
    rec = RecordingProvider("openai")
    monkeypatch.setitem(providers_base._registry, "openai", rec)
    return rec


@pytest.fixture
def client(rec_openai):
    Server._reset()
    spec = importlib.util.spec_from_file_location("example_request_hook_main", MAIN)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield TestClient(module.server.app)
    Server._reset()


def _post(client, headers):
    return client.post(URL, headers=headers, json={
        "model": "helper", "messages": [{"role": "user", "content": "hi"}],
    })


def test_tier_deep_impoe_modelo_grande(client, rec_openai):
    r = _post(client, {"x-user": "ana@example.com", "x-tier": "deep"})
    assert r.status_code == 200, r.text
    assert [m for m, _ in rec_openai.builds] == ["gpt-5.4"]


def test_tier_fast_impoe_modelo_pequeno(client, rec_openai):
    r = _post(client, {"x-user": "ana@example.com", "x-tier": "fast"})
    assert r.status_code == 200, r.text
    assert [m for m, _ in rec_openai.builds] == ["gpt-5.4-mini"]


def test_sem_usuario_e_401(client, rec_openai):
    r = _post(client, {"x-tier": "deep"})
    assert r.status_code == 401
    assert "x-user header is required" in r.text
    assert "no_user" in r.text
    assert rec_openai.builds == []
