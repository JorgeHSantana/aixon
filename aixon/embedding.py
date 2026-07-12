"""Embedding ABC for aixon. Concrete subclasses provide vector representations
of texts. Backends (OpenAI, etc.) live in separate modules and are imported
lazily so the core stays dependency-free.

No LangChain type is imported here — this ABC is neutral. OpenAIEmbedding
delegates to langchain_openai.OpenAIEmbeddings lazily (Task 2)."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional


class Embedding(ABC):
    """Base class for embedding providers.

    Subclasses must implement ``embed_documents`` and ``embed_query``.
    Clients are created lazily inside those methods — never at import time
    or class definition, so importing ``aixon`` never requires a provider
    SDK to be installed.
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one float vector per text."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Return a float vector for a single query string."""


class OpenAIEmbedding(Embedding):
    """Embedding via OpenAI. The client is created lazily on the first
    ``embed_query`` / ``embed_documents`` call — importing ``aixon`` never
    requires ``langchain_openai`` to be installed.

    Args:
        model:       OpenAI embedding model name (e.g. "text-embedding-3-large").
        api_key_env: Environment variable holding the API key
                     (default: ``"OPENAI_API_KEY"``).

    Example::

        class LibraryRetriever(Retriever):
            embedding = OpenAIEmbedding("text-embedding-3-large")
    """

    def __init__(self, model: str, *, api_key_env: str = "OPENAI_API_KEY") -> None:
        self.model = model
        self.api_key_env = api_key_env
        self._client: Optional[object] = None

    def _get_client(self) -> object:
        if self._client is None:
            from langchain_openai import OpenAIEmbeddings  # lazy import
            from pydantic import SecretStr

            # OpenAIEmbeddings' `api_key` field is SecretStr | Callable | None,
            # not a bare str (pydantic coerces at runtime, but the static
            # field type rejects it). None IS accepted and falls back to the
            # SDK's own OPENAI_API_KEY env read, so only wrap when present.
            raw_key = os.getenv(self.api_key_env)
            self._client = OpenAIEmbeddings(
                model=self.model,
                api_key=SecretStr(raw_key) if raw_key else None,
            )
        return self._client

    # _get_client() is typed `-> object` (langchain_openai stays a lazy import,
    # so its real type never appears in this module's signatures) — every
    # call site necessarily hits attr-defined on `object`, not union-attr.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get_client().embed_documents(texts)  # type: ignore[attr-defined]

    def embed_query(self, text: str) -> list[float]:
        return self._get_client().embed_query(text)  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        return f"OpenAIEmbedding(model={self.model!r})"
