"""Hermetic fake agents for CLI and other tests. No network, no LLM.

Usage in tests:
    from tests._cli_fakes import make_cli_echo_agent
    Cls = make_cli_echo_agent("MyAgent007")
    agent = get_registry().resolve("myagent007")

Module-level EchoAgent and HiddenAgent are NOT defined here to avoid
import-order registration surprises. Use make_cli_echo_agent() in each test.
"""
from __future__ import annotations

from typing import Iterator

from aixon.agent import Agent
from aixon.message import Chunk, Message


def make_cli_echo_agent(
    cls_name: str,
    *,
    description: str = "",
    hidden: bool = False,
    reasoning: str = "",
    chunks: list[Chunk] | None = None,
) -> type:
    """Return and register a concrete Agent subclass named *cls_name*.

    The class name must already end with 'Agent'. Streams canned chunks:
    if *chunks* is provided, yields them then a final Chunk(done=True);
    otherwise yields one Chunk(content="echo: <last user message>",
    reasoning=reasoning) then Chunk(done=True).
    """
    if not cls_name.endswith("Agent"):
        raise ValueError(f"cls_name must end with 'Agent', got {cls_name!r}")

    _reasoning = reasoning
    _chunks = chunks

    def _invoke(self, messages: list[Message]) -> Message:
        last = messages[-1].content if messages else ""
        return Message(role="assistant", content=f"echo: {last}")

    def _stream(self, messages: list[Message]) -> Iterator[Chunk]:
        if _chunks is not None:
            yield from _chunks
            yield Chunk(done=True)
            return
        last = messages[-1].content if messages else ""
        yield Chunk(content=f"echo: {last}", reasoning=_reasoning)
        yield Chunk(done=True)

    cls = type(
        cls_name,
        (Agent,),
        {
            "description": description,
            "hidden": hidden,
            "invoke": _invoke,
            "stream": _stream,
        },
    )
    return cls
