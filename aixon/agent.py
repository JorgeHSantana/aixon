"""The Agent base class. Every executable unit in aixon is an Agent and speaks
only neutral types (``Message[]`` in, ``Message``/``Chunk`` out). Concrete
subclasses self-register at definition time; abstract subtypes
(``LLMAgent``/``ToolAgent``/``Orchestrator``, defined in later plans) pass
``abstract=True`` to opt out of validation and registration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from aixon.exceptions import NamingError
from aixon.message import Chunk, Message
from aixon.registry import get_registry


class Agent(ABC):
    # Declarative metadata (override in subclasses).
    name: str = ""
    description: str = ""
    aliases: list[str] = []
    hidden: bool = False
    owned_by: str = "aixon"

    # Required class-name suffix; abstract subtypes may override (e.g. "Orchestrator").
    _suffix: str = "Agent"
    # Set True on a class to mark it an abstract subtype (no validation/registration).
    _abstract: bool = True  # the base itself is abstract

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        if abstract:
            cls._abstract = True
            return
        cls._abstract = False
        if not cls.__name__.endswith(cls._suffix):
            raise NamingError(
                f"Agent subclass '{cls.__name__}' must end with '{cls._suffix}' "
                f"(rename to '{cls.__name__}{cls._suffix}')."
            )
        # Auto-instantiate: running __init__ registers the agent.
        # ABCMeta sets __abstractmethods__ after __init_subclass__ returns, so we
        # must compute unimplemented abstract methods ourselves to surface TypeError.
        abstracts = {
            name
            for name in dir(cls)
            if getattr(getattr(cls, name, None), "__isabstractmethod__", False)
        }
        if abstracts:
            raise TypeError(
                f"Can't instantiate abstract class {cls.__name__} without an "
                f"implementation for abstract method(s) {sorted(abstracts)!r}"
            )
        cls()

    def __init__(self) -> None:
        if not self.name:
            self.name = type(self).__name__.lower()
        get_registry().register(self)

    @abstractmethod
    def invoke(self, messages: list[Message]) -> Message:
        """Run the agent to completion and return one neutral Message."""

    @abstractmethod
    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        """Run the agent, yielding neutral Chunks as they are produced."""
