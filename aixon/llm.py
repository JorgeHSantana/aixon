"""Declarative LLM handle.

Usage on an agent class body:
    class MyAgent(LLMAgent):
        llm = LLM("gpt-5.4", temperature=0.2)

The LLM handle is lazy: it does not build the underlying LangChain model
until the first access to .chat_model (or .complete / .stream). Declaring an
LLM therefore needs neither an API key nor an installed vendor SDK.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

from aixon._interop.messages import from_langchain, to_langchain
from aixon.message import Chunk, Message

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class LLM:
    """Declarative handle for a LangChain chat model behind a neutral boundary."""

    def __init__(self, model: str, *, provider: str | None = None, **params: Any):
        self.model = model
        self.params = params
        self._provider_name = provider          # None → inferred from model name
        self._chat_model: "BaseChatModel | None" = None  # lazy

    @property
    def chat_model(self) -> "BaseChatModel":
        """Lazily build and cache the LangChain model.

        Used directly by ToolAgent and Orchestrator (Plan 3+). The provider
        must already be registered (via importing its module or a custom
        register_provider() call).
        """
        if self._chat_model is None:
            from aixon.providers.base import (
                get_provider,
                resolve_provider_for_model,
            )

            if self._provider_name is not None:
                provider = get_provider(self._provider_name)
            else:
                provider = resolve_provider_for_model(self.model)
            self._chat_model = provider.build(self.model, **self.params)
        return self._chat_model

    def complete(self, messages: list[Message]) -> Message:
        """Single-shot neutral completion. Used by LLMAgent.invoke."""
        lc_result = self.chat_model.invoke(to_langchain(messages))
        return from_langchain(lc_result)

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        """Neutral streaming. Used by LLMAgent.stream.

        Yields Chunk(content=delta) per non-empty delta, then Chunk(done=True).
        Works whether the model yields AIMessageChunk deltas (real providers)
        or a single AIMessage (the fake, which has no _stream).
        """
        for lc_chunk in self.chat_model.stream(to_langchain(messages)):
            content = getattr(lc_chunk, "content", "")
            if isinstance(content, str) and content:
                yield Chunk(content=content)
        yield Chunk(done=True)
