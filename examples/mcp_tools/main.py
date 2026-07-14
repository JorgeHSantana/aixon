"""MCP Tools — point MCPConnector at an MCP server; the LLM drives the catalog.

``MCPConnector`` is the plug-and-play counterpart of ``HttpToolConnector``:
instead of hand-writing one typed method per tool, the server publishes its
catalog (``tools/list``) and ``as_tools()`` turns every entry into a neutral
``AgentTool`` whose ``args_schema`` is the server's own JSON Schema — the LLM
sees the published contract, not a free-text wrapper.

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
(sync and async), and the same tools invoked through ``coerce_tools`` — the
exact integration point a ``ToolAgent`` uses.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from aixon import MCPConnector
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
# as_tools) is the production code path.


class WeatherMCPConnector(MCPConnector):
    @asynccontextmanager
    async def _session(self):
        from mcp.shared.memory import create_connected_server_and_client_session

        async with create_connected_server_and_client_session(
            server._mcp_server
        ) as session:
            yield session


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

    print("\n── as_tools() -> coerce_tools: o que um ToolAgent enxerga ──")
    # In a real agent this is simply:
    #     class WeatherAgent(ToolAgent):
    #         llm   = LLM("gpt-4o-mini")
    #         tools = [*WeatherMCPConnector().as_tools()]
    # coerce_tools is the exact conversion ToolAgent applies to that list.
    for tool in coerce_tools(conn.as_tools(include=["forecast"])):
        print(f"  tool '{tool.name}' — args expostos ao LLM: {list(tool.args)}")
        print("  invoke:", tool.invoke({"city": "Natal", "days": 2}))


if __name__ == "__main__":
    main()
