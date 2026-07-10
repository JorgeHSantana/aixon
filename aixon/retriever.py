"""Retriever ABC for context search (RAG, web search, hybrid).

Concrete subclasses must end with 'Retriever' — validated in
``__init_subclass__`` (same pattern as ``Agent``). Retriever subclasses are
NOT auto-registered as agents; they are tools consumed by ToolAgent via
``as_tool()`` (Task 4).

``write()`` is a default (non-abstract) method that raises if ``type_access``
is READ-only, so read-only subclasses need not override it."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from aixon.agent import AgentTool
from aixon.exceptions import AixonError, NamingError


class TypeAccess(Enum):
    """Controls which operations a Retriever exposes."""

    READ = "read"
    WRITE = "write"
    ALL = "all"


class Retriever(ABC):
    """Abstract base for context retrievers.

    Declarative attributes:
        description:  Human-readable description (used by ``as_tool()``).
        type_access:  READ (default) | WRITE | ALL — governs ``write()``.

    Concrete subclasses must end with ``'Retriever'`` (raises ``NamingError``
    at class definition time if not). Abstract intermediate classes pass
    ``abstract=True`` to opt out.

    Subclasses are NOT agents — they do not auto-register in the Registry.
    """

    description: str = ""
    type_access: TypeAccess = TypeAccess.READ

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if abstract:
            cls._abstract = True  # type: ignore[attr-defined]
            return
        if not cls.__name__.endswith("Retriever"):
            raise NamingError(
                f"Retriever subclass '{cls.__name__}' must end with 'Retriever' "
                f"(rename to '{cls.__name__}Retriever')."
            )

    @abstractmethod
    def search(self, query: str, *, k: int | None = None) -> list[dict]:
        """Search for relevant context.

        Returns:
            A list of dicts with at least ``{'text': str, 'metadata': dict}``.
        """

    async def asearch(self, query: str, *, k: int | None = None) -> list[dict]:
        """Async search. The default bridges to the sync ``search`` in a worker
        thread, so every retriever gets a working ``asearch`` for free and the
        event loop is not blocked. Vendor retrievers backed by an async SDK
        (Weaviate/Ragie/Tavily) should OVERRIDE this for true non-blocking I/O —
        ``as_tool`` then exposes it as the tool's async path."""
        import asyncio

        return await asyncio.to_thread(self.search, query, k=k)

    def write(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> list[str]:
        """Index texts into the retriever's backing store.

        Raises ``AixonError`` if ``type_access`` is ``TypeAccess.READ``.
        Subclasses with ``type_access = TypeAccess.ALL`` should override this.

        Returns:
            List of document IDs.
        """
        if self.type_access == TypeAccess.READ:
            raise AixonError(
                f"'{type(self).__name__}' is configured as read-only "
                f"(type_access=TypeAccess.READ). Set type_access=TypeAccess.ALL "
                f"and override write() to enable indexing."
            )
        raise NotImplementedError(
            f"'{type(self).__name__}' declares type_access={self.type_access!r} "
            f"but does not implement write(). Override write() in your subclass."
        )

    async def awrite(self, texts: list[str],
                      metadatas: list[dict] | None = None) -> list[str]:
        """Async write. Default bridges the sync ``write`` to a worker thread
        (mirrors ``asearch``); vendor retrievers with an async SDK may override."""
        import asyncio

        return await asyncio.to_thread(self.write, texts, metadatas)

    def as_tool(
        self,
        name: str | None = None,
        description: str | None = None,
        k: int | None = None,
    ) -> AgentTool:
        """Expose this retriever as a neutral AgentTool.

        The returned ``AgentTool`` is the same dataclass as ``Agent.as_tool()``
        returns, so ``coerce_tools`` (Plan 3, ``aixon._interop.tools``) handles
        both uniformly.

        Args:
            name:        Tool name (default: lowercased class name).
            description: Tool description (default: ``self.description``).
            k:           Max results forwarded to the agent. ``None`` = no cap.

        Returns:
            ``AgentTool(name, description, func)`` where ``func(query) -> str``
            calls ``self.search(query, k=k)`` and formats results as text.
        """
        _k = k
        _retriever = self

        def _format(docs: list[dict], query: str) -> str:
            if not docs:
                return f"No results found for query: {query!r}"
            parts = []
            for doc in docs:
                text = doc.get("text", "")
                meta = doc.get("metadata", {})
                if meta:
                    parts.append(f"{text} [metadata: {meta}]")
                else:
                    parts.append(text)
            return "\n".join(parts)

        def _run(query: str) -> str:
            return _format(_retriever.search(query, k=_k), query)

        async def _arun(query: str) -> str:
            return _format(await _retriever.asearch(query, k=_k), query)

        return AgentTool(
            name=name or type(self).__name__.lower(),
            description=description or self.description,
            func=_run,
            coroutine=_arun,
        )
