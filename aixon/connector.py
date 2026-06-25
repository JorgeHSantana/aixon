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

import os
from typing import Any

from aixon.exceptions import NamingError

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

        Raises:
            httpx.HTTPStatusError: on non-2xx response.

        Returns:
            Parsed JSON dict.
        """
        httpx = self._httpx()
        url = self.base_url + path
        response = httpx.get(
            url,
            headers=self._headers(),
            timeout=self.timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()

    def post(self, path: str, json: dict | None = None, **kwargs: Any) -> dict:
        """Issue a POST request to ``base_url + path`` with a JSON body.

        Raises:
            httpx.HTTPStatusError: on non-2xx response.

        Returns:
            Parsed JSON dict.
        """
        httpx = self._httpx()
        url = self.base_url + path
        response = httpx.post(
            url,
            json=json,
            headers=self._headers(),
            timeout=self.timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()

    async def aget(self, path: str, **kwargs: Any) -> dict:
        """Async GET via ``httpx.AsyncClient`` (does not block the event loop).
        Same contract as ``get``. Use from an async tool / ``ainvoke`` path."""
        httpx = self._httpx()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                self.base_url + path, headers=self._headers(), **kwargs
            )
        response.raise_for_status()
        return response.json()

    async def apost(self, path: str, json: dict | None = None, **kwargs: Any) -> dict:
        """Async POST via ``httpx.AsyncClient``. Same contract as ``post``."""
        httpx = self._httpx()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.base_url + path, json=json, headers=self._headers(), **kwargs
            )
        response.raise_for_status()
        return response.json()
