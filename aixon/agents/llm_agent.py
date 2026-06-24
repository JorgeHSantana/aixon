"""LLMAgent — abstract subtype for direct LLM access (no tool-calling loop).

Pure-LLM: it does NOT build a langgraph graph and has no tools. It prepends an
optional system prompt and delegates to its LLM. (Tool-calling lives in Plan 3's
ToolAgent, which uses langchain.agents.create_agent.)

Consumer usage:
    class Athena(LLMAgent):
        llm = LLM("gpt-5.4", temperature=0.2)
        prompt = "You are a strategic planner."
        description = "Strategic planning assistant"

Athena auto-registers, gets suffix-validated, and is ready to be routed by name.
"""
from __future__ import annotations

from typing import Iterator

from aixon.agent import Agent
from aixon.exceptions import AixonError
from aixon.llm import LLM
from aixon.message import Chunk, Message


class LLMAgent(Agent, abstract=True):
    """Abstract subtype for agents that delegate directly to an LLM.

    Required class attribute:
        llm: LLM   — e.g. LLM("gpt-5.4", temperature=0.2)
    Optional class attribute:
        prompt: str   — system prompt prepended to every invocation.
    """

    _suffix: str = "Agent"
    llm: LLM         # declared; absence on a concrete subclass is an error
    prompt: str = ""

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs: object) -> None:
        # Let Agent handle suffix validation + auto-registration first.
        super().__init_subclass__(abstract=abstract, **kwargs)
        if abstract:
            return
        llm_val = cls.__dict__.get("llm") or getattr(cls, "llm", None)
        if not isinstance(llm_val, LLM):
            raise AixonError(
                f"'{cls.__name__}' must declare a class-level 'llm' attribute "
                f"of type LLM (e.g. llm = LLM('gpt-5.4')). Got: {llm_val!r}."
            )

    def invoke(self, messages: list[Message]) -> Message:
        """Prepend system prompt (if any) and delegate to self.llm.complete."""
        return self.llm.complete(self._with_prompt(messages))

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        """Prepend system prompt (if any) and delegate to self.llm.stream."""
        yield from self.llm.stream(self._with_prompt(messages))

    def _with_prompt(self, messages: list[Message]) -> list[Message]:
        """Return a new list with the system prompt prepended if set.

        Never mutates the caller's list.
        """
        if self.prompt:
            return [Message(role="system", content=self.prompt), *messages]
        return list(messages)
