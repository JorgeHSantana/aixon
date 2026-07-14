# tests/test_mcp_connector.py
"""MCPConnector: point it at an MCP server and the LLM drives the catalog.

Exercised against a REAL in-memory MCP server (FastMCP + the SDK's connected
client session) — full protocol roundtrip, no network, no hand-rolled mocks.
The ``_session()`` seam is overridden to yield the in-memory session."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from aixon.agent import AgentTool
from aixon.exceptions import AixonError, NamingError
from aixon.mcp import MCPConnector
from aixon._interop.tools import coerce_tools


# ----- in-memory MCP server -------------------------------------------------

def _server():
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("probe")

    @server.tool(description="Soma dois números")
    def add(a: int, b: int) -> int:
        return a + b

    @server.tool(description="Ecoa o texto")
    def echo(text: str) -> str:
        return text

    @server.tool(description="Sempre falha")
    def boom() -> str:
        raise RuntimeError("kaputt")

    return server


class _ProbeMCPConnector(MCPConnector):
    """MCPConnector wired to the in-memory server through the _session seam."""

    def __init__(self, **kwargs):
        super().__init__(base_url="http://in-memory", **kwargs)
        self._mcp_server = _server()
        self.sessions_opened = 0

    @asynccontextmanager
    async def _session(self):
        from mcp.shared.memory import create_connected_server_and_client_session

        self.sessions_opened += 1
        async with create_connected_server_and_client_session(
            self._mcp_server._mcp_server
        ) as session:
            yield session


# ----- discovery ------------------------------------------------------------

def test_list_tools_discovers_catalog_and_caches():
    conn = _ProbeMCPConnector()
    first = conn.list_tools()
    second = conn.list_tools()
    assert [t["name"] for t in first] == ["add", "echo", "boom"]
    assert first[0]["description"] == "Soma dois números"
    assert first[0]["inputSchema"]["required"] == ["a", "b"]
    assert second is first
    assert conn.sessions_opened == 1            # cached: one discovery session


def test_as_tools_returns_agenttools_with_schema():
    conn = _ProbeMCPConnector()
    tools = conn.as_tools()
    assert all(isinstance(t, AgentTool) for t in tools)
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {"add", "echo", "boom"}
    assert by_name["add"].description == "Soma dois números"
    assert by_name["add"].args_schema["required"] == ["a", "b"]
    assert by_name["add"].coroutine is not None  # async path wired


def test_as_tools_include_exclude():
    conn = _ProbeMCPConnector()
    assert {t.name for t in conn.as_tools(include=["add"])} == {"add"}
    assert {t.name for t in conn.as_tools(exclude=["boom"])} == {"add", "echo"}


def test_as_tools_unknown_include_raises():
    conn = _ProbeMCPConnector()
    with pytest.raises(AixonError, match="nope"):
        conn.as_tools(include=["nope"])


# ----- execution ------------------------------------------------------------

def test_call_routes_and_returns_text():
    conn = _ProbeMCPConnector()
    assert conn.call("add", a=2, b=3) == "5"


def test_acall_async_path():
    conn = _ProbeMCPConnector()
    assert asyncio.run(conn.acall("echo", text="oi")) == "oi"


def test_call_drops_none_params():
    conn = _ProbeMCPConnector()
    # b=None would fail the server-side schema; the connector drops it first
    # (same contract as HttpToolConnector).
    with pytest.raises(AixonError):
        conn.call("add", a=2, b=None)


def test_error_result_raises_aixon_error():
    conn = _ProbeMCPConnector()
    with pytest.raises(AixonError, match="kaputt"):
        conn.call("boom")


# ----- end-to-end: as_tools -> coerce_tools -> LangChain invoke ---------------

def test_tools_execute_through_coerce_tools():
    conn = _ProbeMCPConnector()
    tools = coerce_tools(conn.as_tools(include=["add"]))
    [tool] = tools
    assert tool.invoke({"a": 20, "b": 22}) == "42"
    assert asyncio.run(tool.ainvoke({"a": 1, "b": 1})) == "2"


# ----- conventions ------------------------------------------------------------

def test_subclass_naming_enforced():
    with pytest.raises(NamingError):
        class Wrong(MCPConnector):  # noqa: F841 — must end with 'Connector'
            pass


def test_default_session_wires_url_and_auth(monkeypatch):
    """The default transport is streamable HTTP at base_url with Bearer auth."""
    import mcp.client.streamable_http as sh

    captured: dict = {}

    def fake_client(url, headers=None, **kwargs):
        captured.update(url=url, headers=headers or {})
        raise RuntimeError("captured")           # abort before any real IO

    monkeypatch.setattr(sh, "streamablehttp_client", fake_client)

    class _WiredMCPConnector(MCPConnector):
        pass

    conn = _WiredMCPConnector(base_url="https://svc.example.com/mcp",
                              auth_token="tok")
    with pytest.raises(RuntimeError, match="captured"):
        conn.list_tools()
    assert captured["url"] == "https://svc.example.com/mcp"
    assert captured["headers"]["Authorization"] == "Bearer tok"


def test_mcp_optional_extra_declared():
    import pathlib
    import tomllib

    root = pathlib.Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert any(dep.startswith("mcp") for dep in extras["mcp"])
    assert any(dep.startswith("mcp") for dep in extras["all"])
