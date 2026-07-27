"""The Agent base class. Every executable unit in aixon is an Agent and speaks
only neutral types (``Message[]`` in, ``Message``/``Chunk`` out). Concrete
subclasses self-register at definition time; abstract subtypes
(``LLMAgent``/``ToolAgent``/``Orchestrator``, defined in later plans) pass
``abstract=True`` to opt out of validation and registration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Iterator

from aixon.exceptions import AixonError, NamingError
from aixon.logging import Logger
from aixon.message import Chunk, Message
from aixon.reasoning import current_channel
from aixon.registry import get_registry

_log = Logger("aixon.agent")

# Subagent frame (#15): appended to the user text when as_tool(audience="agent")
# — the callee must return dense facts for ANOTHER AGENT, not prose for a human.
# A user-text suffix, NOT a leading system message: a leading system would
# override the subagent's own prompt (see ToolAgent._build_agent contract).
_AGENT_AUDIENCE_SUFFIX = (
    "\n\n[Contexto: sua resposta será consumida por OUTRO AGENTE, não por um "
    "humano. Devolva apenas os fatos relevantes à pergunta, densos e "
    "estruturados, sem saudação, cortesia ou oferta de ajuda.]"
)


def _client_tool_leak_text(name: str, tool_calls: list[dict]) -> str:
    """A worker wrapped by ``as_tool`` that surfaced CLIENT tool_calls (#18c)
    with no text content has nothing an ``as_tool`` caller can use — the
    proxy sentinel never had a real result, and ``as_tool`` has no channel to
    relay ``tool_calls`` back to whatever called the tool (a parent agent's
    tool loop only understands a string result). Silently returning "" would
    read as the subagent producing nothing; log a warning (so the gap is
    visible in telemetry) and hand back an explanatory string instead, so the
    caller's own model sees why the subagent came back empty-handed."""
    names = ", ".join(sorted({
        (c.get("name") if isinstance(c, dict) else getattr(c, "name", None)) or "?"
        for c in tool_calls
    }))
    _log.warning(
        f"as_tool('{name}'): worker surfaced client tool call(s) ({names}) "
        f"with empty content — client tool calls do not cross as_tool"
    )
    return (
        f"[subagente '{name}' solicitou ferramenta do cliente ({names}) — "
        f"chamadas de client tools não atravessam as_tool; trate diretamente "
        f"com o agente]"
    )


def _default_render_input(kwargs: dict) -> str:
    """Default ``render_input`` for ``as_tool(args_schema=...)`` (#23): one
    ``"FIELD: value"`` line per kwarg, key upper-cased, in insertion order.
    Fields whose value is ``None``, ``""`` or ``False`` are dropped — an
    optional schema field the caller left empty/unset should not show up as
    an empty line in the subagent's briefing."""
    return "\n".join(
        f"{k.upper()}: {v}" for k, v in kwargs.items()
        if v not in (None, "", False)
    )


@dataclass
class AgentTool:
    """Neutral descriptor of an Agent exposed as a callable tool. Later plans
    adapt this to a LangChain StructuredTool for tool-calling agents."""

    name: str
    description: str
    # Signature contract: without `args_schema`, a single free-text argument
    # (str -> str); with it, the schema's fields arrive as **kwargs.
    func: Callable[..., str]
    # Optional async variant. When set, coerce_tools registers the LangChain tool
    # with both a sync `func` and this `coroutine`, so the tool runs on BOTH the
    # sync (`invoke`) and async (`ainvoke`) agent paths — the async path awaits
    # the coroutine for true non-blocking I/O.
    coroutine: Callable[..., Awaitable[str]] | None = None
    # Optional JSON Schema (neutral dict) describing the tool's arguments. When
    # set, `func`/`coroutine` receive the schema's fields as **kwargs instead of
    # a single free-text argument — the schema, not the signature, is what the
    # LLM sees. Used by schema-carrying sources (e.g. MCPConnector.as_tools()).
    args_schema: dict | None = None
    # Request-scoped memoization (aixon.toolcache): identical (name, args)
    # calls within one active cache return the first result instead of
    # re-executing. Set False for intentionally non-deterministic tools
    # (e.g. "current time", dice) via `.as_tool(..., memoize=False)`.
    memoize: bool = True


class Agent(ABC):
    # Declarative metadata (override in subclasses).
    name: str = ""
    description: str = ""
    aliases: list[str] = []
    hidden: bool = False
    owned_by: str = "aixon"
    # Per-agent default for the wire thought mode ("custom" | "content" |
    # "hidden"). None inherits the adapter's default. Precedence at the server
    # boundary: request ``thought_stream_mode`` > agent ``thought_mode`` >
    # adapter ``default_thought_mode``. Lets programmatic agents (protocol
    # parsers, plugins) pin "hidden" without changing the server-wide default
    # that chat UIs rely on.
    thought_mode: str | None = None

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
        self, name: str | None = None, description: str | None = None,
        memoize: bool = True, audience: str = "human",
        args_schema: dict | None = None,
        render_input: Callable[[dict], str] | None = None,
    ) -> "AgentTool":
        """Expose this agent as a tool. Each call runs with a fresh message
        list, so the wrapped agent's state never leaks across invocations.
        ``memoize=False`` opts this tool out of the request-scoped tool-call
        cache (see ``aixon.toolcache``).
        ``audience="agent"`` (#15) appends the subagent frame to each call's
        text, instructing the callee to answer with dense facts for another
        agent instead of human-facing prose.

        #22: the wrapped agent is driven through ``stream()``/``astream()``
        (never ``invoke``/``ainvoke``) so its reasoning — tool-call labels,
        judge lines, retries, whatever a nested ``emit_reasoning`` produces —
        flows LIVE into the CALLER's channel instead of surfacing only after
        the whole subagent run completes. That is what lets the #20 drain
        actually have something to drain while a slow nested tool is still
        running: without this, a subagent buried a few levels deep left the
        UI stuck on "running a specialist..." for its entire wall-clock time.
        The final string returned is unchanged (all yielded ``Chunk.content``
        joined) and the tool_calls-without-content defensive path (a worker
        surfacing a CLIENT tool call, #18c) is preserved, now read off
        ``Chunk.tool_calls`` instead of ``Message.tool_calls``.

        #23 — structured briefing. By default (``args_schema=None``, unchanged)
        the tool exposes a single free-text argument and ``func``/``coroutine``
        receive it positionally (``text: str``). Pass ``args_schema`` (a
        neutral JSON-Schema dict, same shape ``AgentTool.args_schema`` already
        accepts from MCP) to expose STRUCTURED fields to the calling model
        instead (e.g. ``objetivo``/``contexto``/``restricoes``) — the schema,
        not a single string, is what the caller's LLM sees and fills in.
        ``func``/``coroutine`` then receive the schema's fields as ``**kwargs``,
        which are turned into the text the subagent actually reads via
        ``render_input(kwargs) -> str``. Default ``render_input`` (used when
        the argument is omitted): one ``"FIELD: value"`` line per kwarg, key
        upper-cased, in insertion order, skipping any field whose value is
        ``None``/``""``/``False`` (see ``_default_render_input``). The
        ``audience="agent"`` suffix, when set, is appended AFTER rendering —
        it frames the rendered briefing, not a raw field.
        """
        if audience not in ("human", "agent"):
            raise AixonError(
                f"as_tool(audience={audience!r}): use 'human' (default) ou "
                f"'agent' (anexa a moldura de subagente ao texto da chamada)."
            )
        suffix = _AGENT_AUDIENCE_SUFFIX if audience == "agent" else ""
        render = render_input or _default_render_input

        def _drive_sync(text: str) -> str:
            # The parent's channel MUST be captured BEFORE iterating
            # self.stream(): stream() opens its OWN `with reasoning_channel()`
            # for the lifetime of its generator, which — because a plain
            # generator shares the caller's execution context across `yield`
            # (no isolation, unlike a Task) — SHADOWS the ContextVar for as
            # long as this loop is pulling chunks from it. Reading
            # `current_channel()` (or calling the free `emit_reasoning()`)
            # INSIDE the loop body would therefore see the CHILD's own
            # channel, not the parent's, and any "re-emit" would silently
            # loop back into the (already-drained) child channel instead of
            # reaching the parent — never surfacing, and on a long-running
            # nested agent, compounding as the child's own next drain tick
            # picks the re-appended line back up. Capturing the reference
            # once, up front, and calling `.emit()` on that object directly
            # sidesteps the shadow entirely.
            parent_channel = current_channel()
            parts: list[str] = []
            tool_calls: list[dict] = []
            for ch in self.stream([Message(role="user", content=text + suffix)]):
                if ch.reasoning and parent_channel is not None:
                    parent_channel.emit(ch.reasoning.rstrip("\n"))
                if ch.content:
                    parts.append(ch.content)
                if ch.tool_calls:
                    tool_calls.extend(ch.tool_calls)
            content = "".join(parts)
            if tool_calls and not content:
                return _client_tool_leak_text(self.name, tool_calls)
            return content

        async def _drive_async(text: str) -> str:
            # Same capture-before-iterating rule as _drive_sync — see its comment.
            parent_channel = current_channel()
            parts: list[str] = []
            tool_calls: list[dict] = []
            async for ch in self.astream([Message(role="user", content=text + suffix)]):
                if ch.reasoning and parent_channel is not None:
                    parent_channel.emit(ch.reasoning.rstrip("\n"))
                if ch.content:
                    parts.append(ch.content)
                if ch.tool_calls:
                    tool_calls.extend(ch.tool_calls)
            content = "".join(parts)
            if tool_calls and not content:
                return _client_tool_leak_text(self.name, tool_calls)
            return content

        # Two DISTINCT closures per sync/async path (not one `*args`-sniffing
        # func): StructuredTool.from_function infers the LLM-facing signature
        # from `func`'s own `inspect.signature` whenever no explicit
        # `args_schema` overrides it, so the no-schema path MUST keep
        # accepting a single positional `text` — collapsing both shapes into
        # one function (e.g. via `**kwargs` with a manual "did the caller
        # pass args_schema" branch) would break that inference for every
        # existing free-text tool.
        if args_schema is None:
            def _run(text: str) -> str:
                return _drive_sync(text)

            async def _arun(text: str) -> str:
                return await _drive_async(text)
        else:
            def _run(**kwargs: object) -> str:
                return _drive_sync(render(kwargs))

            async def _arun(**kwargs: object) -> str:
                return await _drive_async(render(kwargs))

        return AgentTool(
            name=name or self.name,
            description=description or self.description,
            func=_run,
            coroutine=_arun,
            memoize=memoize,
            args_schema=args_schema,
        )
