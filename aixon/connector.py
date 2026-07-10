"""Connector base class for HTTP microservice clients.

Concrete subclasses must end with 'Connector' (raises ``NamingError``).
Each subclass declares ``base_url_env`` and ``auth_token_env`` as class
attributes; constructor overrides take precedence over environment variables.

HTTP calls use ``httpx`` (the ``retrieval`` extra), imported LAZILY inside
``get``/``post`` via ``_httpx()`` so that ``import aixon`` and defining/subclassing
``Connector`` work on a bare install without the extra (the core has no hard deps).
JSON is returned directly; non-2xx responses raise ``httpx.HTTPStatusError``.

Example::

    class DiagnosisConnector(Connector):
        base_url_env = "MCP_DIAGNOSIS_BASE_URL"
        auth_token_env = "MCP_DIAGNOSIS_AUTH_TOKEN"

        def get_status(self) -> dict:
            return self.get("/health")
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any

from aixon.exceptions import AixonError, NamingError

# NOTE: httpx is imported lazily inside _httpx() — NOT at module level — so the
# neutral-boundary guarantee holds (import aixon works without [retrieval]).


class Connector:
    """Base HTTP client for an external microservice.

    Declarative class attributes:
        base_url_env:   Env var name for the service base URL.
        auth_token_env: Env var name for the Bearer token.

    Constructor kwargs override env vars. ``timeout`` defaults to 30 seconds.
    """

    base_url_env: str = ""
    auth_token_env: str = ""

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if abstract:
            return
        if not cls.__name__.endswith("Connector"):
            raise NamingError(
                f"Connector subclass '{cls.__name__}' must end with 'Connector' "
                f"(rename to '{cls.__name__}Connector')."
            )

    def __init__(
        self,
        *,
        base_url: str | None = None,
        auth_token: str | None = None,
        timeout: float | None = 30.0,
    ) -> None:
        self.base_url = (
            base_url
            or (os.getenv(self.base_url_env) if self.base_url_env else None)
            or ""
        ).rstrip("/")

        self.auth_token = (
            auth_token
            or (os.getenv(self.auth_token_env) if self.auth_token_env else None)
            or ""
        )

        self.timeout = timeout
        self._aclient = None
        self._aclient_loop: asyncio.AbstractEventLoop | None = None
        self._aclient_lock = threading.Lock()

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    @staticmethod
    def _httpx():
        """Lazily import httpx. Keeps `import aixon` working on a bare install;
        a clear error tells the user which extra to install if it's missing."""
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - bare install without [retrieval]
            raise ImportError(
                "Connector requires httpx. Install it with: pip install 'aixon[retrieval]'"
            ) from exc
        return httpx

    def get(self, path: str, **kwargs: Any) -> dict:
        """Issue a GET request to ``base_url + path``.

        ``headers``/``timeout`` in ``kwargs`` are merged with (not clobbered
        by) the defaults: extra headers add to ``_headers()`` and an explicit
        timeout overrides ``self.timeout``.

        Raises:
            httpx.HTTPStatusError: on non-2xx response.

        Returns:
            Parsed JSON dict.
        """
        httpx = self._httpx()
        url = self.base_url + path
        headers = {**self._headers(), **(kwargs.pop("headers", None) or {})}
        timeout = kwargs.pop("timeout", self.timeout)
        response = httpx.get(
            url,
            headers=headers,
            timeout=timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()

    def post(self, path: str, json: dict | None = None, **kwargs: Any) -> dict:
        """Issue a POST request to ``base_url + path`` with a JSON body.

        ``headers``/``timeout`` in ``kwargs`` are merged with the defaults
        (see ``get``).

        Raises:
            httpx.HTTPStatusError: on non-2xx response.

        Returns:
            Parsed JSON dict.
        """
        httpx = self._httpx()
        url = self.base_url + path
        headers = {**self._headers(), **(kwargs.pop("headers", None) or {})}
        timeout = kwargs.pop("timeout", self.timeout)
        response = httpx.post(
            url,
            json=json,
            headers=headers,
            timeout=timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()

    def _aclient_needs_rebuild(self, loop: asyncio.AbstractEventLoop) -> bool:
        """True when the cached client is missing/closed, or was built for a
        DIFFERENT event loop than the one currently running."""
        client = self._aclient
        stored_loop = self._aclient_loop
        return (
            client is None
            or client.is_closed
            or stored_loop is None
            or stored_loop is not loop
            or stored_loop.is_closed()
        )

    def _async_client(self):
        """Lazily-created pooled AsyncClient (keep-alive across tool calls
        made from the SAME event loop).

        ``httpx.AsyncClient``'s connection pool is bound to the event loop
        that created it. Caching one instance across separate ``asyncio.run()``
        calls (each of which spins up a brand-new loop) eventually raises
        "Event loop is closed" — and ``client.is_closed`` alone does not catch
        this, since the client itself was never closed, only orphaned from a
        loop that has since gone away. Rebuild whenever the running loop
        differs from the one that built the cached client (or that loop is
        closed, or the client itself is closed). Double-checked locking
        guards concurrent first-builds (same pattern as
        ``WeaviateRetriever._ensure``)."""
        loop = asyncio.get_running_loop()
        if self._aclient_needs_rebuild(loop):
            with self._aclient_lock:
                if self._aclient_needs_rebuild(loop):
                    httpx = self._httpx()
                    self._aclient = httpx.AsyncClient(timeout=self.timeout)
                    self._aclient_loop = loop
        return self._aclient

    async def aclose(self) -> None:
        """Close the pooled async client (idempotent)."""
        if getattr(self, "_aclient", None) is not None:
            await self._aclient.aclose()
            self._aclient = None
            self._aclient_loop = None

    async def aget(self, path: str, **kwargs: Any) -> dict:
        """Async GET via a pooled ``httpx.AsyncClient`` (does not block the
        event loop; the client is kept alive across calls — see ``aclose``).
        Same contract as ``get``, including the ``headers``/``timeout`` merge.
        Use from an async tool / ``ainvoke`` path."""
        client = self._async_client()
        headers = {**self._headers(), **(kwargs.pop("headers", None) or {})}
        timeout = kwargs.pop("timeout", self.timeout)
        response = await client.get(
            self.base_url + path, headers=headers, timeout=timeout, **kwargs
        )
        response.raise_for_status()
        return response.json()

    async def apost(self, path: str, json: dict | None = None, **kwargs: Any) -> dict:
        """Async POST via a pooled ``httpx.AsyncClient``. Same contract as
        ``post`` (see ``aget`` for the pooling/merge behavior)."""
        client = self._async_client()
        headers = {**self._headers(), **(kwargs.pop("headers", None) or {})}
        timeout = kwargs.pop("timeout", self.timeout)
        response = await client.post(
            self.base_url + path, json=json, headers=headers, timeout=timeout,
            **kwargs
        )
        response.raise_for_status()
        return response.json()


class HttpToolConnector(Connector):
    """HTTP-JSON client for a tool server (``POST /<path>`` + a
    ``{success, result, error}`` response envelope).

    Typed methods on subclasses pass the explicit ``path``/``method`` — the tool
    name is NOT assumed to equal the URL (servers may route arbitrarily). This is
    NOT the MCP wire protocol (stdio/SSE); for that, use langchain-mcp-adapters ->
    BaseTool -> coerce_tools.

    ``call``/``acall`` drop ``None`` params before sending (POST -> json, GET ->
    query). ``_unwrap`` understands the default ``{success, result, error}``
    envelope and is overridable per consumer. ``list_tools`` is discovery only
    (NOT routing) and is cached per instance.
    """

    tools_path = "/mcp/tools"

    def list_tools(self) -> list[dict]:
        """Fetch the tool catalog (``GET tools_path``) once, cached per instance."""
        if getattr(self, "_tools_cache", None) is None:
            self._tools_cache = self.get(self.tools_path).get("tools", [])
        return self._tools_cache

    def call(self, method: str, path: str, **params: Any) -> Any:
        """Sync tool call. ``None`` params are dropped."""
        clean = {k: v for k, v in params.items() if v is not None}
        if method.upper() == "POST":
            resp = self.post(path, json=clean)
        else:
            resp = self.get(path, params=clean)
        return self._unwrap(resp)

    async def acall(self, method: str, path: str, **params: Any) -> Any:
        """Async tool call (non-blocking IO). ``None`` params are dropped."""
        clean = {k: v for k, v in params.items() if v is not None}
        if method.upper() == "POST":
            resp = await self.apost(path, json=clean)
        else:
            resp = await self.aget(path, params=clean)
        return self._unwrap(resp)

    def _unwrap(self, resp: dict) -> Any:
        """Default envelope: ``{success, result, error}``. Overridable.

        Raises ``AixonError`` when ``success`` is falsy; otherwise returns
        ``result`` (or the whole dict if there is no ``result`` key)."""
        if not resp.get("success", True):
            raise AixonError(resp.get("error") or f"tool failed: {resp.get('tool')}")
        return resp.get("result", resp)
