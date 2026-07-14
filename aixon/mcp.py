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

Example (class-body / import-time position — a ``ToolAgent`` subclass body
runs at ``autodiscover()`` time, so this position must never do network I/O;
use the deferred ``toolset()`` marker here, NOT eager ``as_tools()``)::

    class MetabaseMCPConnector(MCPConnector):
        base_url_env = "MCP_METABASE_URL"
        auth_token_env = "MCP_METABASE_TOKEN"

    class AnalistaAgent(ToolAgent):
        tools = [MetabaseMCPConnector().toolset(exclude=["delete_card"])]

``toolset()`` returns an ``MCPToolset`` — a tiny holder of
``(connector, include, exclude)`` that performs NO network I/O at
construction. ``coerce_tools`` (``aixon._interop.tools``) detects any tool-list
entry with a ``resolve_tools()`` method and expands it in place, so discovery
runs lazily at the first agent invoke (when ``ToolAgent._build_agent`` calls
``coerce_tools``) — never at import time, and the blast radius of an
unreachable server is contained to that one invoke, not the whole server boot.

``as_tools()``/``aas_tools()`` stay for runtime/script code that wants the
catalog immediately (see their docstrings for the sync/async split); prefer
``toolset()`` for anything declared in a class body.

Known limitation (deferred, not fixed here): there is no session reuse across
calls — ``_session()`` opens a fresh MCP handshake (transport connect +
``initialize``) for every ``list_tools``/``call``, so a chatty tool loop pays
the full handshake cost each time. See ``docs/retrieval.md`` MCP section.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
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

    def __init__(
        self,
        *,
        base_url: str | None = None,
        auth_token: str | None = None,
        timeout: float | None = 30.0,
    ) -> None:
        super().__init__(base_url=base_url, auth_token=auth_token, timeout=timeout)
        # Guards `_tools_cache` (double-checked locking, mirroring
        # `Connector._async_client`'s `_aclient_lock`) so concurrent first-use
        # from multiple threads triggers exactly one discovery session.
        self._tools_lock = threading.Lock()

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
        yield an in-memory session; the default speaks streamable HTTP.

        ``self.timeout`` (seconds, from ``Connector``) is passed straight
        through as the transport's ``timeout`` kwarg: the installed ``mcp``
        SDK's ``streamablehttp_client(url, headers=None, timeout: float |
        datetime.timedelta = 30, sse_read_timeout: float | datetime.timedelta
        = 300, ...)`` accepts a bare float (seconds) directly — no
        seconds->timedelta conversion needed. ``sse_read_timeout`` (the
        long-poll read timeout on the SSE stream) is left at the SDK default;
        only the connector's own ``timeout`` attribute is threaded through.
        ``None`` means "use the SDK default" (no kwarg is passed)."""
        ClientSession, streamable_http = self._sdk()
        headers = (
            {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else None
        )
        kwargs: dict[str, Any] = {}
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        async with streamable_http.streamablehttp_client(
            self.base_url, headers=headers, **kwargs
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    # ----- discovery (cached per instance) -----------------------------------

    async def alist_tools(self) -> list[dict]:
        """Fetch the server catalog (``tools/list``) once, cached per instance.

        Returns neutral dicts: ``{name, description, inputSchema}``.

        Double-checked locking around the cache write (mirroring
        ``Connector._async_client``) so concurrent first-use from multiple
        threads triggers exactly one discovery session instead of one per
        thread."""
        if getattr(self, "_tools_cache", None) is None:
            with self._tools_lock:
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

    def toolset(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> "MCPToolset":
        """Deferred marker for a ``ToolAgent`` class body: ``tools = [conn.toolset(...)]``.

        Performs NO network I/O here — just records ``(self, include,
        exclude)``. ``coerce_tools`` (``aixon._interop.tools``) expands any
        tool-list entry exposing ``resolve_tools()`` in place, so discovery
        happens lazily at the first agent invoke (when ``coerce_tools`` runs),
        never at import/``autodiscover()`` time. This is the recommended way
        to declare MCP tools on an agent; use ``as_tools()``/``aas_tools()``
        for runtime/script code that wants the catalog immediately."""
        return MCPToolset(connector=self, include=include, exclude=exclude)

    async def _aas_tools_core(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[AgentTool]:
        """Shared include/exclude filtering logic for ``as_tools``/``aas_tools``
        (and ``MCPToolset.resolve_tools``, via the connector). Async because
        discovery (``alist_tools``) is async."""
        catalog = await self.alist_tools()
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

    async def aas_tools(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[AgentTool]:
        """Async counterpart of ``as_tools()`` — safe to call (``await``) from
        a running event loop. Same include/exclude contract."""
        return await self._aas_tools_core(include=include, exclude=exclude)

    def as_tools(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[AgentTool]:
        """Expose the server catalog as neutral ``AgentTool``s (eager — for
        runtime/script code, NOT for a ``ToolAgent`` class body; use
        ``toolset()`` there instead).

        Args:
            include: only these tool names (error if the server lacks one —
                     a typo must not silently shrink an agent's toolbox).
            exclude: drop these tool names (unknown names are ignored).

        Triggers discovery on first use (sync path via ``asyncio.run`` — see
        class docstring). Raises ``AixonError`` (instead of letting
        ``asyncio.run`` raise a bare ``RuntimeError`` and leak an un-awaited
        coroutine) when called from inside a running event loop — use
        ``await aas_tools(...)`` there, or ``toolset(...)`` for a deferred,
        class-body-safe declaration."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise AixonError(
                "MCPConnector.as_tools() cannot run inside an active event "
                "loop (it uses asyncio.run() internally, which raises if a "
                "loop is already running). Use `await connector.aas_tools(...)` "
                "from async code, or `connector.toolset(...)` to declare MCP "
                "tools on an agent's `tools = [...]` (deferred, resolved lazily "
                "at first invoke)."
            )
        return asyncio.run(self._aas_tools_core(include=include, exclude=exclude))

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


@dataclass
class MCPToolset:
    """Deferred marker returned by ``MCPConnector.toolset()`` — put it
    directly in a ``ToolAgent``'s ``tools = [...]`` list.

    Holds ``(connector, include, exclude)``. Construction does NO network
    I/O. ``aixon._interop.tools.coerce_tools`` detects any tool-list entry
    with a ``resolve_tools()`` method (duck-typed — no import of this class
    needed there) and expands it in place, extending the coerced list with
    each resolved ``AgentTool``. Discovery therefore happens lazily, at the
    first ``coerce_tools`` call — i.e. the first agent invoke
    (``ToolAgent._build_agent`` calls ``coerce_tools(list(self.tools))`` per
    invoke) — never at class-body/import (``autodiscover()``) time. A single
    unreachable MCP server thus cannot fail server boot; the failure (wrapped
    as ``AixonError``) is contained to whichever agent's request first needed
    it, and is NOT cached — a later invoke retries discovery.

    The resolved catalog IS cached on this instance once it succeeds
    (double-checked locking, mirroring ``Connector._async_client``), so
    repeat invokes of the same agent do not re-discover."""

    connector: "MCPConnector"
    include: list[str] | None = None
    exclude: list[str] | None = None
    _resolved: list[AgentTool] | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def resolve_tools(self) -> list[AgentTool]:
        """Run discovery on first call (from sync OR async context — see
        ``_discover``); cached on success. Any failure (unreachable server,
        protocol error) surfaces as ``AixonError``, scoped to this call —
        nothing is cached, so a later call can retry."""
        if self._resolved is None:
            with self._lock:
                if self._resolved is None:
                    self._resolved = self._discover()
        return self._resolved

    def _discover(self) -> list[AgentTool]:
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No running loop (plain sync call, e.g. a sync ToolAgent
                # invoke building its tool list): asyncio.run is safe here.
                return asyncio.run(
                    self.connector._aas_tools_core(
                        include=self.include, exclude=self.exclude
                    )
                )
            else:
                # A running loop means resolve_tools() is being called
                # synchronously from inside async agent code (coerce_tools /
                # _build_agent are sync, called from ainvoke/astream without
                # awaiting). asyncio.run() cannot nest inside a running loop,
                # so discovery runs on a worker thread with its own loop;
                # `.result()` blocks the calling thread until it completes —
                # a bounded block, same class as any sync tool invoked from an
                # async agent loop.
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(
                        asyncio.run,
                        self.connector._aas_tools_core(
                            include=self.include, exclude=self.exclude
                        ),
                    ).result()
        except AixonError:
            raise
        except Exception as exc:
            raise AixonError(
                f"MCPToolset discovery failed for "
                f"'{type(self.connector).__name__}' at "
                f"'{self.connector.base_url}': {exc}"
            ) from exc
