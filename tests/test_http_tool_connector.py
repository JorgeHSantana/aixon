# tests/test_http_tool_connector.py
"""HttpToolConnector: HTTP-JSON tool-server client over Connector. httpx mocked
(sync get/post + AsyncClient) — no network."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from aixon.connector import HttpToolConnector
from aixon.exceptions import AixonError


class _ToolConnector(HttpToolConnector):
    base_url_env = "X_URL"
    auth_token_env = "X_TOKEN"


def _mock_sync(monkeypatch, captured, payload):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    def fake_get(url, **kwargs):
        captured.update(method="GET", url=url, params=kwargs.get("params"),
                        headers=kwargs.get("headers", {}))
        return _Resp()

    def fake_post(url, json=None, **kwargs):
        captured.update(method="POST", url=url, json=json,
                        headers=kwargs.get("headers", {}))
        return _Resp()

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)


def _mock_async(monkeypatch, captured, payload):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kwargs):
            captured.update(method="GET", url=url, params=kwargs.get("params"))
            return _Resp()

        async def post(self, url, json=None, **kwargs):
            captured.update(method="POST", url=url, json=json)
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)


def test_call_post_filters_none_and_unwraps_result(monkeypatch):
    captured: dict = {}
    _mock_sync(monkeypatch, captured, {"tool": "t", "result": {"n": 1}, "success": True})
    conn = _ToolConnector(base_url="https://svc.example.com", auth_token="tok")
    out = conn.call("POST", "/mcp/tools/t", a=1, b=None)
    assert out == {"n": 1}                      # unwrapped result
    assert captured["url"] == "https://svc.example.com/mcp/tools/t"
    assert captured["json"] == {"a": 1}          # None filtered
    assert captured["headers"]["Authorization"] == "Bearer tok"


def test_call_get_uses_params(monkeypatch):
    captured: dict = {}
    _mock_sync(monkeypatch, captured, {"result": {"ok": True}, "success": True})
    conn = _ToolConnector(base_url="https://svc.example.com")
    out = conn.call("GET", "/mcp/tools/list", q="x", empty=None)
    assert out == {"ok": True}
    assert captured["method"] == "GET"
    assert captured["params"] == {"q": "x"}


def test_unwrap_raises_on_success_false(monkeypatch):
    captured: dict = {}
    _mock_sync(monkeypatch, captured, {"tool": "t", "error": "boom", "success": False})
    conn = _ToolConnector(base_url="https://svc.example.com")
    with pytest.raises(AixonError, match="boom"):
        conn.call("POST", "/mcp/tools/t", a=1)


def test_unwrap_overridable(monkeypatch):
    captured: dict = {}
    _mock_sync(monkeypatch, captured, {"data": 42})

    class _CustomConnector(HttpToolConnector):
        def _unwrap(self, resp):
            return resp["data"]

    out = _CustomConnector(base_url="https://svc.example.com").call("POST", "/x", a=1)
    assert out == 42


def test_acall_async(monkeypatch):
    captured: dict = {}
    _mock_async(monkeypatch, captured, {"result": {"async": True}, "success": True})
    conn = _ToolConnector(base_url="https://svc.example.com", auth_token="tok")
    out = asyncio.run(conn.acall("POST", "/mcp/tools/t", a=1, b=None))
    assert out == {"async": True}
    assert captured["url"] == "https://svc.example.com/mcp/tools/t"
    assert captured["json"] == {"a": 1}


def test_list_tools_cached(monkeypatch):
    calls = {"n": 0}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            calls["n"] += 1
            return {"tools": [{"name": "t1"}, {"name": "t2"}], "server": "restmcp"}

    monkeypatch.setattr(httpx, "get", lambda url, **k: _Resp())
    conn = _ToolConnector(base_url="https://svc.example.com")
    first = conn.list_tools()
    second = conn.list_tools()
    assert [t["name"] for t in first] == ["t1", "t2"]
    assert second is first or second == first
    assert calls["n"] == 1                       # cached: HTTP hit once
