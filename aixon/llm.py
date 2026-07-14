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

from aixon._interop.messages import (
    _flatten_content,
    from_langchain,
    reasoning_from_chunk,
    to_langchain,
)
from aixon.logging import Logger
from aixon.message import Chunk, Message

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

_log = Logger("aixon.llm")

# Bound on LLM._request_model_cache: a per-request model is a full provider
# SDK client (HTTP connection pool and all), so the cache must not grow
# unbounded across arbitrarily many distinct param combinations. Eviction is
# oldest-inserted-first (a plain dict preserves insertion order).
_REQUEST_MODEL_CACHE_MAX = 8


class LLM:
    """Declarative handle for a LangChain chat model behind a neutral boundary."""

    def __init__(
        self,
        model: str,
        *,
        provider: str | None = None,
        reasoning: bool | dict[str, Any] | None = None,
        **params: Any,
    ):
        self.model = model
        self.params = params
        self.reasoning = reasoning               # None/False off, True/dict on
        self._provider_name = provider          # None → inferred from model name
        self._chat_model: "BaseChatModel | None" = None  # lazy
        self._request_model_cache: dict[tuple, "BaseChatModel"] = {}

    def _provider(self):
        """Resolve this LLM's provider: explicit name, or inferred from the
        model name. Shared by ``chat_model`` and ``request_chat_model`` so both
        use the exact same resolution rule."""
        from aixon.providers.base import get_provider, resolve_provider_for_model

        if self._provider_name is not None:
            return get_provider(self._provider_name)
        return resolve_provider_for_model(self.model)

    def _build(self, provider: Any, **params: Any) -> "BaseChatModel":
        """Call ``provider.build`` with the reasoning knob applied safely.

        ``params["reasoning"]`` is only injected when the provider declares
        ``supports_reasoning`` (all shipped providers do, and they pop it
        before touching the vendor constructor). A provider WITHOUT support —
        e.g. a custom one that blindly forwards **params to a pydantic-strict
        vendor class — never sees the stray key: the knob is ignored with a
        warning instead of breaking the build.

        The per-request ``reasoning_effort`` override (a GENERATION_PARAMS
        key, arriving here already merged into ``params`` by
        ``request_chat_model``) gets the identical treatment: it is a
        SEPARATE key from the ``reasoning`` knob above, and a provider
        without support never pops it itself (only ``resolve_reasoning_spec``,
        called from inside ``supports_reasoning`` providers' own ``build()``,
        does that). Left unpopped here, it would reach a pydantic-strict
        vendor constructor unguarded — one key over from the exact gap rule 5
        already closes for ``reasoning``."""
        if getattr(provider, "supports_reasoning", False):
            return provider.build(self.model, reasoning=self.reasoning, **params)
        if self.reasoning is not None and self.reasoning is not False:
            _log.warning(
                "provider '%s' does not support reasoning — ignored",
                getattr(provider, "name", type(provider).__name__),
            )
        if "reasoning_effort" in params:
            _log.warning(
                "provider '%s' does not support reasoning — 'reasoning_effort' ignored",
                getattr(provider, "name", type(provider).__name__),
            )
            params = {k: v for k, v in params.items() if k != "reasoning_effort"}
        return provider.build(self.model, **params)

    @property
    def chat_model(self) -> "BaseChatModel":
        """Lazily build and cache the LangChain model.

        Used directly when no per-request generation params are active
        (LLMAgent's ``_bound_model``, and ``request_chat_model`` itself as its
        no-params fast path). ToolAgent goes through ``request_chat_model()``
        instead, which applies per-request params via a fresh/cached provider
        model rather than this bare cached one. The provider must already be
        registered (via importing its module or a custom register_provider()
        call).
        """
        if self._chat_model is None:
            self._chat_model = self._build(self._provider(), **self.params)
        return self._chat_model

    def request_chat_model(self) -> "BaseChatModel":
        """chat_model with the current request's generation params applied.

        Builds a provider model with the params merged in as constructor
        kwargs — ``.bind()`` would return a RunnableBinding, which
        ``langchain.agents.create_agent`` does not accept as a model. Used by
        ToolAgent (and any other agent that builds a langgraph agent from
        ``self.llm``) instead of the bare ``chat_model`` so per-request
        generation params (temperature, max_tokens, ...) reach the provider.

        No active params -> the cached ``chat_model`` (cache preserved, no
        rebuild). Otherwise, models are cached by the exact param combination
        (bounded, oldest-evicted-first) so repeated requests with the SAME
        params reuse one provider model (and its HTTP connection pool)
        instead of rebuilding a fresh SDK client every time."""
        from aixon.runtime import current_generation_params

        params = current_generation_params()
        if not params:
            return self.chat_model

        # `stop` arrives as a list (unhashable) — normalize to a tuple so the
        # cache key is hashable. Other generation params are already scalars.
        key = tuple(sorted(
            (k, tuple(v) if isinstance(v, list) else v) for k, v in params.items()
        ))
        cached = self._request_model_cache.get(key)
        if cached is not None:
            return cached

        model = self._build(self._provider(), **{**self.params, **params})
        if len(self._request_model_cache) >= _REQUEST_MODEL_CACHE_MAX:
            oldest_key = next(iter(self._request_model_cache))
            del self._request_model_cache[oldest_key]
        self._request_model_cache[key] = model
        return model

    def _bound_model(self) -> "BaseChatModel":
        """Chat model with the current request's generation params applied.
        Used by LLMAgent (complete/stream/acomplete/astream).

        Delegates straight to ``request_chat_model()`` — the SAME merge/
        translate/cache path ToolAgent already uses. No params active -> the
        bare cached model, unchanged.

        This used to call ``.bind(**params)`` on the bare ``chat_model``
        instead, attaching params at INVOKE time on top of an already-built
        model. That bypassed ``Provider.build()`` (and therefore
        ``resolve_reasoning_spec``) entirely for every per-request param:
        a client ``reasoning_effort`` reached the vendor SDK as a raw,
        untranslated invoke-time kwarg (the Anthropic/OpenAI SDKs reject it,
        a 500), and a client ``temperature`` bound at invoke time could
        override the constructor-forced ``temperature=1`` Anthropic's
        extended-thinking API requires (a 400 from Anthropic). Routing
        through ``request_chat_model()`` merges params in as constructor
        kwargs BEFORE ``build()`` runs, so translation and the temperature
        force both apply the same way they already do for ToolAgent."""
        return self.request_chat_model()

    def complete(self, messages: list[Message]) -> Message:
        """Single-shot neutral completion. Used by LLMAgent.invoke."""
        lc_result = self._bound_model().invoke(to_langchain(messages))
        return from_langchain(lc_result)

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        """Neutral streaming. Used by LLMAgent.stream.

        Yields Chunk(reasoning=delta) then Chunk(content=delta) per chunk (in
        that order, when a single underlying chunk carries both — e.g. a
        Claude thinking block followed by text in the same delta), skipping
        whichever is empty, then Chunk(done=True). Works whether the model
        yields AIMessageChunk deltas (real providers) or a single AIMessage
        (the fake, which has no _stream). No reasoning present -> unchanged
        byte-for-byte from before reasoning extraction existed.
        """
        for lc_chunk in self._bound_model().stream(to_langchain(messages)):
            reasoning = reasoning_from_chunk(lc_chunk)
            if reasoning:
                yield Chunk(reasoning=reasoning)
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
        over the model's native ``astream`` (same reasoning-before-content
        ordering per chunk; see stream())."""
        async for lc_chunk in self._bound_model().astream(to_langchain(messages)):
            reasoning = reasoning_from_chunk(lc_chunk)
            if reasoning:
                yield Chunk(reasoning=reasoning)
            # Some providers stream list-of-blocks deltas; flatten to text.
            content = _flatten_content(getattr(lc_chunk, "content", ""))
            if content:
                yield Chunk(content=content)
        yield Chunk(done=True)
