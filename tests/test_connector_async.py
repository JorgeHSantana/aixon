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


# ── loop affinity: a pooled client bound to a dead loop must be rebuilt ─────

def _loop_bound_async_client_cls(created: list):
    """A fake httpx.AsyncClient that mimics the real one's loop affinity:
    ``get`` raises "Event loop is closed" if called from a DIFFERENT running
    loop than the one active when the client was constructed."""

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    class _Client:
        def __init__(self, *a, **k):
            self.loop = asyncio.get_event_loop()
            self.is_closed = False
            created.append(self)

        async def get(self, url, **kwargs):
            if asyncio.get_running_loop() is not self.loop:
                raise RuntimeError("Event loop is closed")
            return _Resp()

        async def aclose(self):
            self.is_closed = True

    return _Client


def test_async_client_rebuilds_across_separate_asyncio_run_calls(monkeypatch):
    """Two successive `asyncio.run()` calls each spin up a NEW event loop.
    A client cached from the first call must not be reused (and blow up) on
    the second — the second call must get a fresh client bound to its loop."""
    created: list = []
    monkeypatch.setattr(httpx, "AsyncClient", _loop_bound_async_client_cls(created))

    conn = _ApiConnector(base_url="https://api.example.com")
    asyncio.run(conn.aget("/a"))  # loop #1
    asyncio.run(conn.aget("/b"))  # loop #2 — must rebuild, must not raise

    assert len(created) == 2
    assert created[0] is not created[1]


def test_async_client_pooled_within_a_single_loop(monkeypatch):
    """Within ONE event loop (the normal case — an async tool making several
    calls during one request), the client is still reused, not rebuilt on
    every call."""
    created: list = []
    monkeypatch.setattr(httpx, "AsyncClient", _loop_bound_async_client_cls(created))

    conn = _ApiConnector(base_url="https://api.example.com")

    async def _both():
        await conn.aget("/a")
        await conn.aget("/b")

    asyncio.run(_both())
    assert len(created) == 1
