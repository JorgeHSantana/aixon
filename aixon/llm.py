"""Declarative LLM handle.

Usage on an agent class body:
    class MyAgent(LLMAgent):
        llm = LLM("gpt-4o-mini", temperature=0.2)

The LLM handle is lazy: it does not build the underlying LangChain model
until the first access to .chat_model (or .complete / .stream). Declaring an
LLM therefore needs neither an API key nor an installed vendor SDK.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator

from aixon._interop.messages import _flatten_content, from_langchain, to_langchain
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

    def _provider(self):
        """Resolve this LLM's provider: explicit name, or inferred from the
        model name. Shared by ``chat_model`` and ``request_chat_model`` so both
        use the exact same resolution rule."""
        from aixon.providers.base import get_provider, resolve_provider_for_model

        if self._provider_name is not None:
            return get_provider(self._provider_name)
        return resolve_provider_for_model(self.model)

    @property
    def chat_model(self) -> "BaseChatModel":
        """Lazily build and cache the LangChain model.

        Used directly by ToolAgent and Orchestrator (Plan 3+). The provider
        must already be registered (via importing its module or a custom
        register_provider() call).
        """
        if self._chat_model is None:
            self._chat_model = self._provider().build(self.model, **self.params)
        return self._chat_model

    def request_chat_model(self) -> "BaseChatModel":
        """chat_model with the current request's generation params applied.

        Builds a FRESH provider model with the params merged in as constructor
        kwargs — ``.bind()`` would return a RunnableBinding, which
        ``langchain.agents.create_agent`` does not accept as a model. Used by
        ToolAgent (and any other agent that builds a langgraph agent from
        ``self.llm``) instead of the bare ``chat_model`` so per-request
        generation params (temperature, max_tokens, ...) reach the provider.

        No active params -> the cached ``chat_model`` (cache preserved, no
        rebuild)."""
        from aixon.runtime import current_generation_params

        params = current_generation_params()
        if not params:
            return self.chat_model
        return self._provider().build(self.model, **{**self.params, **params})

    def _bound_model(self) -> "BaseChatModel":
        """Chat model with the current request's generation params bound on top
        of the class-level defaults. No params active → the bare model."""
        from aixon.runtime import current_generation_params

        params = current_generation_params()
        return self.chat_model.bind(**params) if params else self.chat_model

    def complete(self, messages: list[Message]) -> Message:
        """Single-shot neutral completion. Used by LLMAgent.invoke."""
        lc_result = self._bound_model().invoke(to_langchain(messages))
        return from_langchain(lc_result)

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        """Neutral streaming. Used by LLMAgent.stream.

        Yields Chunk(content=delta) per non-empty delta, then Chunk(done=True).
        Works whether the model yields AIMessageChunk deltas (real providers)
        or a single AIMessage (the fake, which has no _stream).
        """
        for lc_chunk in self._bound_model().stream(to_langchain(messages)):
            # Some providers stream list-of-blocks deltas; flatten to text.
            content = _flatten_content(getattr(lc_chunk, "content", ""))
            if content:
                yield Chunk(content=content)
        yield Chunk(done=True)

    async def acomplete(self, messages: list[Message]) -> Message:
        """Async single-shot completion. Used by LLMAgent.ainvoke. Delegates to
        the LangChain model's native ``ainvoke`` (does not block the loop)."""
        lc_result = await self._bound_model().ainvoke(to_langchain(messages))
        return from_langchain(lc_result)

    async def astream(self, messages: list[Message]) -> AsyncIterator[Chunk]:
        """Async neutral streaming. Used by LLMAgent.astream. Mirrors stream()
        over the model's native ``astream``."""
        async for lc_chunk in self._bound_model().astream(to_langchain(messages)):
            # Some providers stream list-of-blocks deltas; flatten to text.
            content = _flatten_content(getattr(lc_chunk, "content", ""))
            if content:
                yield Chunk(content=content)
        yield Chunk(done=True)
