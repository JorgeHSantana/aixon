"""MCP Tools — point MCPConnector at an MCP server; the LLM drives the catalog.

``MCPConnector`` is the plug-and-play counterpart of ``HttpToolConnector``:
instead of hand-writing one typed method per tool, the server publishes its
catalog (``tools/list``) and ``toolset()``/``as_tools()`` turn every entry
into a neutral ``AgentTool`` whose ``args_schema`` is the server's own JSON
Schema — the LLM sees the published contract, not a free-text wrapper.

Two ways to get there, and this example demonstrates BOTH:

* ``toolset()`` — the pattern for a ``ToolAgent`` class body (see
  ``WeatherAgent`` below). A class body runs at ``autodiscover()``/server-boot
  time; ``toolset()`` does NO network I/O at construction — it just records
  the connector plus include/exclude — so an unreachable MCP server can never
  block server boot. Discovery happens lazily, the first time the agent is
  actually invoked: ``coerce_tools`` (what ``ToolAgent._build_agent`` calls,
  per invoke) expands the toolset into real tools right before the LangGraph
  agent is built.
* ``as_tools()`` — eager, for runtime/script code (the ``main()`` function
  below) that wants the catalog immediately and isn't sitting in a class
  body.

This example runs **fully offline**: the "server" is an in-memory FastMCP
instance wired through the ``_session()`` seam (the same seam the test suite
uses), so ``python main.py`` needs no API key and no network. A real
deployment only swaps the seam for configuration:

    class WeatherMCPConnector(MCPConnector):
        base_url_env   = "MCP_WEATHER_URL"     # streamable-HTTP endpoint
        auth_token_env = "MCP_WEATHER_TOKEN"   # optional Bearer token

Run it:

    cd examples/mcp_tools
    python main.py

Expected output: the discovered catalog with schemas, two direct tool calls
(sync and async), a proof that ``toolset()`` does zero I/O until the first
``coerce_tools`` expansion (the lazy-discovery fix), and the same tools
invoked through ``coerce_tools(conn.as_tools(...))`` — the exact integration
point a ``ToolAgent`` uses.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from aixon import LLM, MCPConnector, ToolAgent
from aixon._interop.tools import coerce_tools

# ── the MCP server: in-memory, offline ───────────────────────────────────────
# A real server would live in its own process behind an HTTP endpoint; FastMCP
# here plays that role in-process. Note there is NO aixon code on this side —
# any MCP server works, including ones you don't own.

server = FastMCP("weather")


@server.tool(description="Previsão do tempo para uma cidade (dias à frente)")
def forecast(city: str, days: int = 1) -> str:
    return f"{city}: sol com nuvens pelos próximos {days} dia(s), 24°C."


@server.tool(description="Temperatura atual em uma cidade")
def current_temp(city: str) -> str:
    return f"{city}: 24°C agora."


# ── the connector: aixon side ─────────────────────────────────────────────────
# Overriding _session() wires the in-memory server in place of the default
# streamable-HTTP transport. Everything else (discovery, caching, unwrapping,
# toolset/as_tools) is the production code path.


class WeatherMCPConnector(MCPConnector):
    @asynccontextmanager
    async def _session(self):
        from mcp.shared.memory import create_connected_server_and_client_session

        async with create_connected_server_and_client_session(
            server._mcp_server
        ) as session:
            yield session


# ── the agent: what a real deployment looks like ─────────────────────────────
# `tools = [WeatherMCPConnector().toolset(...)]` is the recommended pattern:
# this class body runs at import/autodiscover() time, and toolset() performs
# NO network I/O here — see the "toolset() does zero I/O" block in main()
# below for a direct demonstration. Discovery only happens once this agent is
# actually invoked; this example never invokes it (that needs a real LLM
# call, and this demo stays fully offline), so this class exists purely to
# show the declaration shape.


class WeatherAgent(ToolAgent):
    llm = LLM("gpt-4o-mini")
    tools = [WeatherMCPConnector().toolset()]


def main() -> None:
    conn = WeatherMCPConnector(base_url="http://in-memory")

    print("── catálogo descoberto (tools/list, cacheado por instância) ──")
    for spec in conn.list_tools():
        required = spec["inputSchema"].get("required", [])
        print(f"  {spec['name']}: {spec['description']} — args {required}")

    print("\n── chamada direta (sync) ──")
    print(" ", conn.call("forecast", city="Fortaleza", days=3))

    print("\n── chamada direta (async) ──")
    print(" ", asyncio.run(conn.acall("current_temp", city="Recife")))

    print("\n── toolset(): zero I/O até o primeiro coerce_tools() ──")
    # A FRESH connector — no discovery has happened on it yet. toolset()
    # just records (connector, include, exclude) and returns; nothing below
    # touches the (in-memory) network until coerce_tools() resolves it.
    fresh_conn = WeatherMCPConnector(base_url="http://in-memory")
    lazy_toolset = fresh_conn.toolset(include=["current_temp"])
    print("  toolset() construído — nenhuma chamada de rede ainda")
    lazy_tools = coerce_tools([lazy_toolset])  # <- discovery happens HERE
    print(f"  coerce_tools([toolset]) resolveu: {[t.name for t in lazy_tools]}")
    print("  invoke:", lazy_tools[0].invoke({"city": "Belém"}))

    print("\n── as_tools() -> coerce_tools: uso eager (scripts/runtime) ──")
    # `toolset()` is for agent class bodies (see WeatherAgent above); when the
    # catalog is wanted immediately in a script, as_tools() is the direct
    # equivalent of what coerce_tools() would otherwise resolve lazily.
    for tool in coerce_tools(conn.as_tools(include=["forecast"])):
        print(f"  tool '{tool.name}' — args expostos ao LLM: {list(tool.args)}")
        print("  invoke:", tool.invoke({"city": "Natal", "days": 2}))


if __name__ == "__main__":
    main()
