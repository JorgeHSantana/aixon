# tests/test_audit_task_6_3.py
"""Achados mcp-1..4: limpeza de params num lugar, cache inicializado, API
pública na descoberta do toolset e descoberta fora da trava."""
from __future__ import annotations

import asyncio
import threading
import time

from aixon.connector import Connector
from aixon.mcp import MCPConnector, MCPToolset
from tests.test_mcp_connector import _ProbeMCPConnector


def test_mcp_1(monkeypatch):
    """A regra 'None é omitido' é Connector._clean_params, usada pelo MCP."""
    assert Connector._clean_params({"a": 1, "b": None, "c": 0}) == {"a": 1, "c": 0}
    seen = []
    real = Connector._clean_params

    def spy(params):
        seen.append(dict(params))
        return real(params)

    monkeypatch.setattr(Connector, "_clean_params", staticmethod(spy))
    conn = _ProbeMCPConnector()
    assert conn.call("echo", text="oi", extra=None) == "oi"
    assert seen == [{"text": "oi", "extra": None}]


def test_mcp_2():
    """_tools_cache nasce em __init__ (None), sem depender de getattr."""
    conn = MCPConnector(base_url="http://x")
    assert "_tools_cache" in vars(conn) and conn._tools_cache is None


def test_mcp_3(monkeypatch):
    """O toolset descobre pela API pública aas_tools, nos dois ramos."""
    conn = _ProbeMCPConnector()
    calls = []
    real = conn.aas_tools

    def spy(**kw):
        calls.append(kw)
        return real(**kw)

    monkeypatch.setattr(conn, "aas_tools", spy)
    assert len(MCPToolset(conn, include=["add"]).resolve_tools()) == 1

    async def from_loop():
        return MCPToolset(conn, exclude=["boom"]).resolve_tools()

    assert {t.name for t in asyncio.run(from_loop())} == {"add", "echo"}
    assert calls == [
        {"include": ["add"], "exclude": None},
        {"include": None, "exclude": ["boom"]},
    ]


def test_mcp_4(monkeypatch):
    """Descoberta fora da trava: uma thread rápida não espera a lenta; o
    primeiro a terminar publica e o outro descarta o seu resultado."""
    ts = MCPToolset(MCPConnector(base_url="http://x"))
    slow_started = threading.Event()
    lists = {}

    def fake_discover():
        if threading.current_thread().name == "slow":
            slow_started.set()
            time.sleep(0.6)
            lists["slow"] = ["slow"]
            return lists["slow"]
        lists["fast"] = ["fast"]
        return lists["fast"]

    monkeypatch.setattr(ts, "_discover", fake_discover)
    out = {}

    def run(name):
        t0 = time.monotonic()
        out[name] = (ts.resolve_tools(), time.monotonic() - t0)

    slow = threading.Thread(target=run, args=("slow",), name="slow")
    slow.start()
    assert slow_started.wait(2)
    fast = threading.Thread(target=run, args=("fast",), name="fast")
    fast.start()
    fast.join(2)
    slow.join(2)
    assert out["fast"][1] < 0.3  # não ficou presa na descoberta da outra
    assert out["fast"][0] is lists["fast"]
    assert out["slow"][0] is lists["fast"]  # a lenta descartou a sua
    assert ts.resolve_tools() is lists["fast"]
