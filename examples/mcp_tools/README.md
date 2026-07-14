# MCP Tools — a runnable `MCPConnector` example

A minimal, **fully offline** demonstration of `MCPConnector`, documented in
[docs/retrieval.md](../../docs/retrieval.md#mcpconnector--mcp-servers): point
it at an MCP server and the server's published catalog (`tools/list`) becomes
the agent's toolbox — each tool's JSON Schema is what the LLM sees, no
hand-written wrapper per tool.

No API key, no network call: the "server" is an in-memory FastMCP instance
wired through the `_session()` seam (the same seam the test suite uses), so
`python main.py` just works.

## What it demonstrates

| Element | Where |
|---|---|
| An MCP server publishing tools with schemas (zero aixon code on that side) | [main.py](main.py) — `FastMCP("weather")` |
| `class WeatherMCPConnector(MCPConnector)` + the `_session()` seam | [main.py](main.py) |
| `tools = [WeatherMCPConnector().toolset()]` — the recommended `ToolAgent` class-body pattern (zero I/O at class/import time) | `class WeatherAgent(ToolAgent)` |
| Catalog discovery (`list_tools`), cached per instance | `main()` — first block |
| Direct execution, sync `call` and async `acall` | `main()` — middle blocks |
| `toolset()` performing no I/O until `coerce_tools()` resolves it (the lazy-discovery fix) | `main()` — "toolset(): zero I/O" block |
| `as_tools(include=...)` → `coerce_tools` → LangChain `invoke` — the eager, script/runtime path | `main()` — last block |

## `HttpToolConnector` or `MCPConnector`?

They are complements, not generations:

- **`HttpToolConnector`** — the flow is decided **in code**: each typed method
  is a deliberate tool you shaped (signature, normalization, encoding).
- **`MCPConnector`** — the flow is decided **by the LLM**: the server publishes
  the catalog and the model works from the published schemas. Plug-and-play
  for servers you don't own.

## Run it

```bash
cd examples/mcp_tools
python main.py
```

Install the dependencies first (or use the repo venv, where `aixon` is
already importable):

```bash
pip install -r requirements.txt   # aixon[mcp] — the MCP SDK extra
```

## Real deployment

Swap the in-memory seam for configuration — that's the whole diff:

```python
class WeatherMCPConnector(MCPConnector):
    base_url_env   = "MCP_WEATHER_URL"     # streamable-HTTP endpoint
    auth_token_env = "MCP_WEATHER_TOKEN"   # optional Bearer token

class WeatherAgent(ToolAgent):
    llm   = LLM("gpt-4o-mini")
    tools = [WeatherMCPConnector().toolset(exclude=["dangerous_tool"])]
```

Note the class body uses `toolset()`, not `as_tools()`: a `ToolAgent` class
body runs at `autodiscover()`/server-boot time, and `toolset()` performs no
network I/O there — discovery is deferred to the agent's first invoke, so one
unreachable MCP server can't take down the whole server's boot. `as_tools()`
(eager) is for runtime/script code, like this example's `main()`.
