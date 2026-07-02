"""The Agent base class. Every executable unit in aixon is an Agent and speaks
only neutral types (``Message[]`` in, ``Message``/``Chunk`` out). Concrete
subclasses self-register at definition time; abstract subtypes
(``LLMAgent``/``ToolAgent``/``Orchestrator``, defined in later plans) pass
``abstract=True`` to opt out of validation and registration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Iterator

from aixon.exceptions import NamingError
from aixon.message import Chunk, Message
from aixon.registry import get_registry


@dataclass
class AgentTool:
    """Neutral descriptor of an Agent exposed as a callable tool. Later plans
    adapt this to a LangChain StructuredTool for tool-calling agents."""

    name: str
    description: str
    func: Callable[[str], str]
    # Optional async variant. When set, coerce_tools registers the LangChain tool
    # with both a sync `func` and this `coroutine`, so the tool runs on BOTH the
    # sync (`invoke`) and async (`ainvoke`) agent paths — the async path awaits
    # the coroutine for true non-blocking I/O.
    coroutine: Callable[[str], Awaitable[str]] | None = None


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
    # Set True after first successful registration; prevents RegistrationError on re-instantiation.
    _registered: bool = False

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        if abstract:
            cls._abstract = True
            return
        cls._abstract = False
        # Declarative metadata is per-class, never inherited from a concrete
        # parent: a subclass gets its own aliases, its own default name and its
        # own registration (an inherited name/_registered would silently skip
        # registration or collide in the registry).
        cls._registered = False
        if "aliases" not in vars(cls):
            cls.aliases = []
        if not vars(cls).get("name"):
            cls.name = cls.__name__.lower()
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
        # Subtype validation runs BEFORE registration so a failed check never
        # leaves a half-built agent in the registry.
        cls._validate_subclass()
        cls()

    @classmethod
    def _validate_subclass(cls) -> None:
        """Hook for abstract subtypes (LLMAgent/ToolAgent/Orchestrator) to
        validate a concrete subclass before it is registered. Override to raise
        on invalid configuration (e.g. a missing required attribute). The base
        implementation is a no-op."""

    def __init__(self) -> None:
        # Default name is set on the CLASS (before the register-once
        # short-circuit) so every instance carries it, not just the first.
        if not self.name:
            type(self).name = type(self).__name__.lower()
        if type(self)._registered:
            return
        get_registry().register(self)
        type(self)._registered = True

    @abstractmethod
    def invoke(self, messages: list[Message]) -> Message:
        """Run the agent to completion and return one neutral Message."""

    @abstractmethod
    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        """Run the agent, yielding neutral Chunks as they are produced."""

    async def ainvoke(self, messages: list[Message]) -> Message:
        """Async variant of ``invoke``. Default: run the sync ``invoke`` in a
        worker thread so it does not block the event loop. Async-native subtypes
        (LLMAgent/ToolAgent/Orchestrator) override this to use LangGraph's native
        async path. A purely sync custom Agent gets a working ``ainvoke`` for
        free via this bridge."""
        import asyncio

        return await asyncio.to_thread(self.invoke, messages)

    async def astream(self, messages: list[Message]) -> "AsyncIterator[Chunk]":
        """Async variant of ``stream``. Default: drive the sync generator in a
        background thread and forward its chunks, so it does not block the loop.
        Async-native subtypes override this.

        A producer exception is re-raised to the consumer (via ``await fut``)
        after any chunks produced before it. If the consumer stops early
        (``break``), a stop event makes the producer abandon the sync stream
        within at most one further chunk and, if it is a generator, ``close()``
        it (running its ``finally`` blocks) instead of draining it to
        completion. ``stream()``'s contract is ``Iterator[Chunk]``, not
        ``Generator``, so plain iterators (no ``close()``) are accepted too;
        the done sentinel is always enqueued, even if ``close()`` raises or
        ``stream()`` itself raises before returning."""
        import asyncio
        import contextvars
        import threading

        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue" = asyncio.Queue()
        done = object()
        stop = threading.Event()

        def _producer() -> None:
            try:
                gen = self.stream(messages)
                for chunk in gen:
                    if stop.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            finally:
                # stream() returns Iterator[Chunk], not necessarily a generator:
                # only generators have close(). Guard it, and guarantee the done
                # sentinel is enqueued even if close() raises, so the consumer
                # never deadlocks waiting for a sentinel that never comes.
                try:
                    close = getattr(gen, "close", None) if "gen" in dir() else None
                    if callable(close):
                        close()
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, done)

        # run_in_executor does not propagate contextvars (unlike asyncio.to_thread),
        # so run the producer inside a copy of the caller's context — otherwise
        # generation_params()/reasoning_channel() set around astream() are
        # invisible to the bridged sync stream().
        ctx = contextvars.copy_context()
        fut = loop.run_in_executor(None, ctx.run, _producer)
        try:
            while True:
                item = await queue.get()
                if item is done:
                    break
                yield item
        finally:
            stop.set()
            await fut

    def as_tool(
        self, name: str | None = None, description: str | None = None
    ) -> "AgentTool":
        """Expose this agent as a tool. Each call runs with a fresh message
        list, so the wrapped agent's state never leaks across invocations."""

        def _run(text: str) -> str:
            result = self.invoke([Message(role="user", content=text)])
            return result.content

        async def _arun(text: str) -> str:
            result = await self.ainvoke([Message(role="user", content=text)])
            return result.content

        return AgentTool(
            name=name or self.name,
            description=description or self.description,
            func=_run,
            coroutine=_arun,
        )
