"""Embedding ABC for aixon. Concrete subclasses provide vector representations
of texts. Backends (OpenAI, etc.) live in separate modules and are imported
lazily so the core stays dependency-free.

No LangChain type is imported here — this ABC is neutral. OpenAIEmbedding
delegates to langchain_openai.OpenAIEmbeddings lazily (Task 2)."""

from __future__ import annotations

from abc import ABC, abstractmethod


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
