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
