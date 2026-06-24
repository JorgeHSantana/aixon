"""Interop: the boundary where aixon's neutral types meet LangChain.

INTERNAL to aixon (the leading underscore marks the whole package private —
nothing here is part of the public API). Public code speaks only
``Message``/``Chunk`` and ``AgentTool``; everything LangChain-specific is
confined to this package:

    messages.py  — ``to_langchain`` / ``from_langchain``
                   (neutral Message <-> LangChain message objects)
    tools.py     — ``coerce_tools``
                   (AgentTool / plain callable / BaseTool -> LangChain BaseTool)

This ``__init__`` deliberately does NOT re-export those symbols. ``messages``
imports ``langchain_core`` at module scope, so an eager re-export here would
force LangChain at ``import aixon`` time and break the neutral boundary
(``ToolAgent`` is exported unguarded). Import from the exact submodule instead,
e.g. ``from aixon._interop.tools import coerce_tools``.
"""
