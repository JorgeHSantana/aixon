# aixon/_interop/tools.py
"""Coerce neutral tool entries into LangChain BaseTools for the tool-calling
loop. This is the ONLY place neutral AgentTool -> LangChain conversion happens
(the neutral boundary, contract §2.3/§2.4): Agent.as_tool stays neutral and
returns an AgentTool; coercion to a LangChain tool occurs here, inside the
ToolAgent runtime. langchain is imported lazily so importing ``aixon`` never
requires it."""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable

from aixon.agent import AgentTool
from aixon.exceptions import AixonError
from aixon.logging import Logger
from aixon.toolcache import ToolCallCache, current_tool_cache

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.tools import BaseTool

_log = Logger("aixon.tools")


def _tool_error_text(name: str, exc: Exception) -> str:
    """Readable tool-failure result for the LLM. ``str(e) or repr(e)`` because
    some exceptions (httpx.ReadTimeout) have an EMPTY str() — the repr at
    least names the class, so neither the model nor the logs go blind."""
    detail = str(exc).strip() or repr(exc)
    return (
        f"TOOL ERROR — a ferramenta '{name}' falhou: {detail}. "
        f"O serviço pode estar indisponível; informe o usuário sobre a "
        f"indisponibilidade e/ou prossiga com o que você já tem."
    )


def _guard(name: str, fn: Callable[..., Any], *, memoize: bool,
           shield: bool, is_async: bool) -> Callable[..., Any]:
    """Wrap a tool function with the two request-scoped behaviors:

    - memoization (#5): with an active ``aixon.toolcache`` cache and
      ``memoize=True``, identical (name, args) calls return the first result.
      Errors are NEVER cached — a later round may succeed.
    - error shield (#9): with ``shield=True``, ANY exception becomes a readable
      error string returned as the tool result, so one failing tool reports
      instead of killing the whole run/stream.

    ``functools.wraps`` preserves ``__wrapped__``, so ``inspect.signature``
    (used by StructuredTool.from_function to infer the args schema) still sees
    the ORIGINAL signature, not ``(*args, **kwargs)``."""

    def _lookup(args: tuple, kwargs: dict):
        cache = current_tool_cache() if memoize else None
        key = ToolCallCache.key(name, args, kwargs) if cache is not None else None
        return cache, key

    def _handle(exc: Exception) -> str:
        if not shield:
            raise exc
        _log.warning(f"tool '{name}' failed: {str(exc).strip() or repr(exc)}")
        return _tool_error_text(name, exc)

    if is_async:
        @functools.wraps(fn)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            cache, key = _lookup(args, kwargs)
            if cache is not None and key is not None and cache.has(key):
                return cache.get(key)
            try:
                result = await fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — shield converte qualquer falha
                return _handle(exc)
            if cache is not None and key is not None:
                cache.set(key, result)
            return result
        return awrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cache, key = _lookup(args, kwargs)
        if cache is not None and key is not None and cache.has(key):
            return cache.get(key)
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — shield converte qualquer falha
            return _handle(exc)
        if cache is not None and key is not None:
            cache.set(key, result)
        return result
    return wrapper


def coerce_tools(tools: list, *, shield_errors: bool = True) -> list["BaseTool"]:
    """Convert each entry of ``tools`` to a LangChain ``BaseTool``.

    Accepted entry forms:
      * ``AgentTool`` (from ``Agent.as_tool()`` / ``Retriever.as_tool()``) ->
        wrapped with ``StructuredTool.from_function``.
      * a LangChain ``BaseTool`` (incl. ``@tool``-decorated functions) ->
        passed through unchanged.
      * a plain callable -> wrapped with ``StructuredTool.from_function``. An
        async callable is registered via ``coroutine=`` and therefore requires
        an async agent path (``ainvoke``/``astream``); calling it from sync
        ``invoke`` raises ``NotImplementedError`` rather than silently skipping.
      * any entry with a ``resolve_tools()`` method (duck-typed — no import of
        the concrete type needed here, e.g. ``aixon.mcp.MCPToolset`` from
        ``MCPConnector.toolset()``) -> expanded in place: ``resolve_tools()``
        runs (lazily, e.g. MCP catalog discovery — the reason this exists:
        keeping I/O out of the agent class body / import time) and each
        resulting ``AgentTool`` is coerced same as above. This first flattens
        the whole list so name-collision detection (below) sees the expanded
        tools too, not just the un-expanded marker.

    AgentTool and plain-callable entries pass through ``_guard`` — the error
    shield (#9, disabled via ``shield_errors=False``) and the request-scoped
    memoization (#5, per-tool opt-out via ``AgentTool.memoize`` /
    ``aixon_memoize`` attribute). A raw ``BaseTool`` entry is passed through
    UNCHANGED — no shield, no memoization — because rebuilding an arbitrary
    BaseTool (return_direct, injected state, custom schema) is lossy; wrap
    your function as a plain callable/AgentTool if you want the guard.

    Raises ``AixonError`` for any other type.
    """
    from langchain_core.tools import BaseTool, StructuredTool

    expanded: list = []
    for entry in tools:
        resolver = getattr(entry, "resolve_tools", None)
        if callable(resolver):
            expanded.extend(resolver())
        else:
            expanded.append(entry)

    coerced: list[BaseTool] = []
    seen_names: set[str] = set()
    dups: set[str] = set()
    for entry in expanded:
        if isinstance(entry, BaseTool):
            coerced.append(entry)
        elif isinstance(entry, AgentTool):
            # When the AgentTool carries an async `coroutine`, register it too so
            # the tool runs on both paths: sync `invoke` uses `func`, async
            # `ainvoke` awaits `coroutine` (true non-blocking). Retriever/Agent
            # as_tool() set both; a func-only AgentTool stays sync. `coroutine`
            # defaults to None on from_function, so passing it unconditionally
            # (rather than **kwargs-ing it in only when set) is equivalent and
            # lets mypy check each argument against its own declared type
            # instead of joining them all into one `**dict[str, object]`.
            # `args_schema` (a neutral JSON-Schema dict, e.g. an MCP tool's
            # inputSchema) defines the LLM-facing argument surface; without it,
            # from_function infers a single free-text arg from the signature.
            # langchain-core accepts the dict form directly.
            # func/coroutine pass through _guard (shield #9 + memo #5).
            memo = getattr(entry, "memoize", True)
            coerced.append(
                StructuredTool.from_function(
                    func=_guard(entry.name, entry.func, memoize=memo,
                                shield=shield_errors, is_async=False),
                    name=entry.name,
                    description=entry.description,
                    coroutine=(
                        _guard(entry.name, entry.coroutine, memoize=memo,
                               shield=shield_errors, is_async=True)
                        if entry.coroutine is not None else None
                    ),
                    args_schema=entry.args_schema,
                )
            )
        elif callable(entry):
            # An async callable MUST be registered via `coroutine=`, not as the
            # positional sync `func`. Passing a coroutine function as `func`
            # makes StructuredTool call it synchronously, producing an un-awaited
            # coroutine that is silently dropped — the tool never runs (in either
            # invoke or ainvoke). With `coroutine=`, the async path (arun) awaits
            # it; the sync path raises NotImplementedError instead of silently
            # skipping, so async tools require an async agent path (ainvoke/astream).
            # Plain callables also pass through _guard; opt-out of memoization
            # via a `aixon_memoize = False` attribute on the function.
            fn_name = entry.__name__
            memo = getattr(entry, "aixon_memoize", True)
            if inspect.iscoroutinefunction(entry):
                coerced.append(
                    StructuredTool.from_function(
                        coroutine=_guard(fn_name, entry, memoize=memo,
                                         shield=shield_errors, is_async=True),
                        name=fn_name,
                        description=(entry.__doc__ or fn_name),
                    )
                )
            else:
                coerced.append(
                    StructuredTool.from_function(
                        _guard(fn_name, entry, memoize=memo,
                               shield=shield_errors, is_async=False),
                        name=fn_name,
                        description=(entry.__doc__ or fn_name),
                    )
                )
        else:
            raise AixonError(
                f"Tool entry {entry!r} (type {type(entry).__name__}) cannot be "
                f"used as a tool. Provide an AgentTool (agent.as_tool() / "
                f"retriever.as_tool()), a LangChain BaseTool / @tool function, "
                f"or a plain callable."
            )
        added = coerced[-1]
        if added.name in seen_names:
            dups.add(added.name)
        seen_names.add(added.name)
    if dups:
        raise AixonError(
            f"Duplicate tool name(s): {sorted(dups)}. Pass as_tool(name=...) "
            f"to disambiguate."
        )
    return coerced
