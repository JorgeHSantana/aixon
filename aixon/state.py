"""Default LangGraph state for aixon Orchestrators. Carries the neutral
conversation (``messages``) and accumulated ``reasoning``. Users subclass
``GraphState`` to add fields (declared as ``class State(GraphState): ...``
inside their Orchestrator).

This module is the ONE place outside ``agents/orchestrator.py`` that touches
LangGraph, and only to re-export the ``END`` sentinel so concrete orchestrators
never import ``langgraph`` directly."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import END as END  # re-export; aixon.END is the public name

from aixon.message import Message


def add_messages_neutral(
    existing: list[Message] | None,
    new: list[Message] | Message | None,
) -> list[Message]:
    """LangGraph reducer for the neutral ``messages`` channel.

    Appends neutral ``Message`` objects without mutating ``existing``. LangGraph
    passes a node's return value (``state["messages"]`` update) as ``new``; that
    value may be a single ``Message`` or a list. ``None`` on either side is
    treated as empty so partial state updates are safe.
    """
    base: list[Message] = list(existing) if existing else []
    if new is None:
        return base
    if isinstance(new, Message):
        return base + [new]
    return base + list(new)


class GraphState(TypedDict, total=False):
    """Default orchestrator state. ``total=False`` makes every key optional, so
    nodes may return partial updates and subclasses may add fields freely."""

    messages: Annotated[list[Message], add_messages_neutral]
    reasoning: list[str]
