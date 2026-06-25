# tests/test_connector_async.py
"""Connector.aget/apost use httpx.AsyncClient (audit 3.3 async story). Mocked
so no network is touched."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from aixon.connector import Connector


class _ApiConnector(Connector):
    base_url_env = "X_URL"
    auth_token_env = "X_TOKEN"


def _mock_async_client(monkeypatch, captured):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True, "method": captured["method"]}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kwargs):
            captured.update(method="GET", url=url, headers=kwargs.get("headers", {}))
            return _Resp()

        async def post(self, url, json=None, **kwargs):
            captured.update(method="POST", url=url, json=json, headers=kwargs.get("headers", {}))
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)


def test_aget(monkeypatch):
    captured: dict = {}
    _mock_async_client(monkeypatch, captured)
    conn = _ApiConnector(base_url="https://api.example.com", auth_token="t")
    out = asyncio.run(conn.aget("/orders/7"))
    assert out["ok"] is True and out["method"] == "GET"
    assert captured["url"] == "https://api.example.com/orders/7"
    assert captured["headers"]["Authorization"] == "Bearer t"


def test_apost(monkeypatch):
    captured: dict = {}
    _mock_async_client(monkeypatch, captured)
    conn = _ApiConnector(base_url="https://api.example.com")
    out = asyncio.run(conn.apost("/orders", json={"x": 1}))
    assert out["method"] == "POST"
    assert captured["json"] == {"x": 1}
