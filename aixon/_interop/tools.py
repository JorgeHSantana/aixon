# aixon/_interop/tools.py
"""Coerce neutral tool entries into LangChain BaseTools for the tool-calling
loop. This is the ONLY place neutral AgentTool -> LangChain conversion happens
(the neutral boundary, contract §2.3/§2.4): Agent.as_tool stays neutral and
returns an AgentTool; coercion to a LangChain tool occurs here, inside the
ToolAgent runtime. langchain is imported lazily so importing ``aixon`` never
requires it."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from aixon.agent import AgentTool
from aixon.exceptions import AixonError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.tools import BaseTool


def coerce_tools(tools: list) -> list["BaseTool"]:
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

    Raises ``AixonError`` for any other type.
    """
    from langchain_core.tools import BaseTool, StructuredTool

    coerced: list[BaseTool] = []
    for entry in tools:
        if isinstance(entry, BaseTool):
            coerced.append(entry)
        elif isinstance(entry, AgentTool):
            coerced.append(
                StructuredTool.from_function(
                    func=entry.func,
                    name=entry.name,
                    description=entry.description,
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
            if inspect.iscoroutinefunction(entry):
                coerced.append(
                    StructuredTool.from_function(
                        coroutine=entry,
                        name=entry.__name__,
                        description=(entry.__doc__ or entry.__name__),
                    )
                )
            else:
                coerced.append(StructuredTool.from_function(entry))
        else:
            raise AixonError(
                f"Tool entry {entry!r} (type {type(entry).__name__}) cannot be "
                f"used as a tool. Provide an AgentTool (agent.as_tool() / "
                f"retriever.as_tool()), a LangChain BaseTool / @tool function, "
                f"or a plain callable."
            )
    return coerced
