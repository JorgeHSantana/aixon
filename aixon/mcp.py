"""MCPConnector — point at an MCP server and let the LLM drive its catalog.

Counterpart of ``HttpToolConnector``, NOT its replacement. The two cover
opposite ends of one trade-off:

* ``HttpToolConnector`` — the flow is decided IN CODE: each typed method is a
  deliberate tool you shaped (signature, normalization, encoding).
* ``MCPConnector`` — the flow is decided BY THE LLM: the server publishes its
  catalog via ``tools/list`` and the model works from the published JSON
  Schemas. No hand-written wrapper per tool.

Transport is MCP **streamable HTTP** at ``base_url`` (the full endpoint URL,
e.g. ``https://host/mcp``), with Bearer auth from ``auth_token`` — both resolve
from env vars exactly like any ``Connector``. The ``mcp`` SDK is imported
LAZILY so ``import aixon`` works without it (install extra: ``aixon[mcp]``).

Discovery (``list_tools``) is cached per instance. Execution opens one fresh
session per call — stateless, so there is no event-loop affinity to manage
(contrast ``Connector._async_client``). The sync paths (``list_tools``/
``call``/``as_tools``) run the async ones via ``asyncio.run`` and therefore
must NOT be called from a running event loop — use ``alist_tools``/``acall``
there.

Example::

    class MetabaseMCPConnector(MCPConnector):
        base_url_env = "MCP_METABASE_URL"
        auth_token_env = "MCP_METABASE_TOKEN"

    class AnalistaAgent(ToolAgent):
        tools = [*MetabaseMCPConnector().as_tools(exclude=["delete_card"])]
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from aixon.agent import AgentTool
from aixon.connector import Connector
from aixon.exceptions import AixonError


class MCPConnector(Connector):
    """Client for an MCP server (streamable HTTP transport).

    Declarative class attributes (inherited from ``Connector``):
        base_url_env:   Env var with the MCP endpoint URL.
        auth_token_env: Env var with the Bearer token.

    ``as_tools()`` turns the server's catalog into ``AgentTool``s whose
    ``args_schema`` is each tool's published ``inputSchema`` — the LLM sees the
    server's own contract, not a free-text wrapper."""

    @staticmethod
    def _sdk():
        """Lazily import the ``mcp`` SDK (same pattern as ``Connector._httpx``)."""
        try:
            from mcp import ClientSession
            from mcp.client import streamable_http
        except ImportError as exc:  # pragma: no cover - bare install without [mcp]
            raise ImportError(
                "MCPConnector requires the 'mcp' SDK. Install it with: "
                "pip install 'aixon[mcp]'"
            ) from exc
        return ClientSession, streamable_http

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[Any]:
        """Open, initialize and yield one MCP client session (per operation).

        The single seam between this class and the wire: tests override it to
        yield an in-memory session; the default speaks streamable HTTP."""
        ClientSession, streamable_http = self._sdk()
        headers = (
            {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else None
        )
        async with streamable_http.streamablehttp_client(
            self.base_url, headers=headers
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    # ----- discovery (cached per instance) -----------------------------------

    async def alist_tools(self) -> list[dict]:
        """Fetch the server catalog (``tools/list``) once, cached per instance.

        Returns neutral dicts: ``{name, description, inputSchema}``."""
        if getattr(self, "_tools_cache", None) is None:
            async with self._session() as session:
                result = await session.list_tools()
            self._tools_cache = [
                {
                    "name": tool.name,
                    "description": tool.description or tool.name,
                    "inputSchema": tool.inputSchema,
                }
                for tool in result.tools
            ]
        return self._tools_cache

    def list_tools(self) -> list[dict]:
        """Sync ``alist_tools`` (not callable from a running event loop)."""
        if getattr(self, "_tools_cache", None) is None:
            asyncio.run(self.alist_tools())
        return self._tools_cache

    # ----- execution (one session per call) ----------------------------------

    async def acall(self, name: str, **params: Any) -> str:
        """Call one server tool (``tools/call``). ``None`` params are dropped
        (same contract as ``HttpToolConnector``)."""
        clean = {k: v for k, v in params.items() if v is not None}
        async with self._session() as session:
            result = await session.call_tool(name, clean)
        return self._unwrap(result)

    def call(self, name: str, **params: Any) -> str:
        """Sync ``acall`` (not callable from a running event loop)."""
        return asyncio.run(self.acall(name, **params))

    @staticmethod
    def _unwrap(result: Any) -> str:
        """Neutral view of a ``CallToolResult``: error -> ``AixonError``; text
        content joined for the LLM; ``structuredContent`` as JSON fallback."""
        texts = [
            block.text
            for block in result.content
            if getattr(block, "text", None) is not None
        ]
        if result.isError:
            raise AixonError("\n".join(texts) or "MCP tool call failed")
        if texts:
            return "\n".join(texts)
        if result.structuredContent is not None:
            return json.dumps(result.structuredContent, ensure_ascii=False)
        return ""

    # ----- agent surface ------------------------------------------------------

    def as_tools(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[AgentTool]:
        """Expose the server catalog as neutral ``AgentTool``s.

        Args:
            include: only these tool names (error if the server lacks one —
                     a typo must not silently shrink an agent's toolbox).
            exclude: drop these tool names (unknown names are ignored).

        Triggers discovery on first use (sync path — see class docstring)."""
        catalog = self.list_tools()
        names = {spec["name"] for spec in catalog}
        unknown = set(include or ()) - names
        if unknown:
            raise AixonError(
                f"MCP server at '{self.base_url}' does not expose tool(s): "
                f"{sorted(unknown)}. Available: {sorted(names)}."
            )
        dropped = set(exclude or ())
        return [
            self._make_tool(spec)
            for spec in catalog
            if (include is None or spec["name"] in include)
            and spec["name"] not in dropped
        ]

    def _make_tool(self, spec: dict) -> AgentTool:
        name = spec["name"]

        def _run(**kwargs: Any) -> str:
            return self.call(name, **kwargs)

        async def _arun(**kwargs: Any) -> str:
            return await self.acall(name, **kwargs)

        return AgentTool(
            name=name,
            description=spec["description"],
            func=_run,
            coroutine=_arun,
            args_schema=spec["inputSchema"],
        )
