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


def _drain_chunk(lc_chunk: Any, agg: Any, *, tools: list[dict] | None) -> tuple[list[Chunk], Any]:
    """Per-delta drain shared by ``LLM.stream``/``LLM.astream`` (#18a): the
    neutral ``Chunk``(s) this ONE underlying provider chunk produces
    (reasoning before content, matching a single delta that carries both —
    e.g. a Claude thinking block followed by text), plus the updated
    tool-call aggregator.

    ``agg`` accumulates chunks via LangChain's summable ``AIMessageChunk``
    (``+``) only when ``tools`` is active — real providers stream deltas that
    sum into the full tool_calls; the fake test model streams one
    already-complete message, so the "sum" is a no-op. No ``tools`` -> `agg`
    is returned unchanged (always ``None``), preserving the exact no-tools
    byte-for-byte behavior.

    Pure function, no yielding: `stream`/`astream` differ only in their
    surrounding for/async-for loop, so the actual Chunk delivery stays in
    each of them; this only says WHAT to emit for one chunk."""
    chunks: list[Chunk] = []
    reasoning = reasoning_from_chunk(lc_chunk)
    if reasoning:
        chunks.append(Chunk(reasoning=reasoning))
    # Some providers stream list-of-blocks deltas; flatten to text.
    content = _flatten_content(getattr(lc_chunk, "content", ""))
    if content:
        chunks.append(Chunk(content=content))
    if tools:
        agg = lc_chunk if agg is None else agg + lc_chunk
    return chunks, agg


class LLM:
    """Declarative handle for a LangChain chat model behind a neutral boundary."""

    def __init__(
        self,
        model: str,
        *,
        provider: str | None = None,
        reasoning: bool | dict[str, Any] | None = None,
        cache: bool = False,
        **params: Any,
    ):
        self.model = model
        self.params = params
        self.reasoning = reasoning               # None/False off, True/dict on
        # Prompt-cache knob (#4): on providers with explicit cache breakpoints
        # (Anthropic's cache_control), True marks the system message and the
        # LAST message of every call — marking the last message each call
        # yields INCREMENTAL caching across retry rounds (round N's breakpoint
        # becomes round N+1's cached prefix). Providers without the flag
        # ignore it (OpenAI caches long prefixes automatically).
        self.cache = cache
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

    def _to_wire(self, messages: list[Message]) -> list:
        """``to_langchain`` + the prompt-cache breakpoints when ``cache=True``
        and the provider declares ``supports_prompt_cache`` (Anthropic). The
        FIRST system message and the LAST message get a ``cache_control``
        block; everything in between stays untouched, so the prefix remains
        byte-stable across retry rounds."""
        lc_messages = to_langchain(messages)
        if not self.cache or not lc_messages:
            return lc_messages
        if not getattr(self._provider(), "supports_prompt_cache", False):
            return lc_messages

        def mark(lc_message) -> None:
            content = lc_message.content
            if isinstance(content, str):
                lc_message.content = [{
                    "type": "text", "text": content,
                    "cache_control": {"type": "ephemeral"},
                }]
            elif isinstance(content, list):
                for block in reversed(content):
                    if isinstance(block, dict) and block.get("type") == "text":
                        block["cache_control"] = {"type": "ephemeral"}
                        return

        if getattr(lc_messages[0], "type", "") == "system":
            mark(lc_messages[0])
        if len(lc_messages) > 1 or getattr(lc_messages[0], "type", "") != "system":
            mark(lc_messages[-1])
        return lc_messages

    def _bind_tools(self, model: "BaseChatModel", tools: list[dict] | None,
                     tool_choice: Any = None) -> Any:
        """Bind client-declared tools (#18a) onto ``model`` for one call.

        ``tools`` is the OpenAI wire shape (``{"type": "function", "function":
        {...}}``) already normalized by ``ParsedRequest.tools`` — passed
        straight to LangChain's ``bind_tools``, which every chat model
        implements (real providers translate to their own dialect; the fake
        test model ignores it and returns itself unchanged). ``tool_choice``,
        when given, is forwarded as an extra ``bind_tools`` kwarg — omitted
        entirely (not even ``None``) when absent, since some providers reject
        an explicit ``tool_choice=None``. No ``tools`` -> the model is
        returned untouched, so the no-tools path stays byte-identical.

        Return type is ``Any`` (not ``BaseChatModel``): ``bind_tools`` returns
        a ``Runnable`` (a ``RunnableBinding`` wrapping the model), which is
        not a ``BaseChatModel`` subtype but supports the same ``invoke``/
        ``stream``/``ainvoke``/``astream`` surface used below."""
        if not tools:
            return model
        extra = {"tool_choice": tool_choice} if tool_choice is not None else {}
        return model.bind_tools(tools, **extra)

    def _invoke_kwargs(self) -> dict:
        """Per-invocation kwargs derived from runtime context (#6): when a
        predicted output is active (ReflectiveAgent retry) AND the provider
        declares ``supports_prediction`` (OpenAI's Predicted Outputs), attach
        it; any other provider gets nothing — silent no-op, same precedent as
        the ``reasoning`` knob."""
        from aixon.runtime import current_prediction

        prediction = current_prediction()
        if prediction and getattr(self._provider(), "supports_prediction", False):
            return {"prediction": {"type": "content", "content": prediction}}
        return {}

    def complete(
        self, messages: list[Message], *,
        tools: list[dict] | None = None, tool_choice: Any = None,
    ) -> Message:
        """Single-shot neutral completion. Used by LLMAgent.invoke.

        ``tools`` (#18a), when given, is the OpenAI wire shape
        (``{"type": "function", "function": {...}}``) — the same normalized
        shape ``ParsedRequest.tools`` already produces. It is bound onto the
        chat model for this one call via ``bind_tools`` (see ``_bind_tools``);
        ``tool_choice``, if given, is forwarded alongside it. The returned
        ``Message.tool_calls`` (populated by ``from_langchain``) carries
        whatever the model chose to call, ready to relay to the client as-is.
        Omitting ``tools`` leaves this call byte-identical to before #18a."""
        model = self._bind_tools(self._bound_model(), tools, tool_choice)
        lc_result = model.invoke(
            self._to_wire(messages), **self._invoke_kwargs()
        )
        return from_langchain(lc_result)

    def stream(
        self, messages: list[Message], *,
        tools: list[dict] | None = None, tool_choice: Any = None,
    ) -> Iterator[Chunk]:
        """Neutral streaming. Used by LLMAgent.stream.

        Yields Chunk(reasoning=delta) then Chunk(content=delta) per chunk (in
        that order, when a single underlying chunk carries both — e.g. a
        Claude thinking block followed by text in the same delta), skipping
        whichever is empty, then Chunk(done=True). Works whether the model
        yields AIMessageChunk deltas (real providers) or a single AIMessage
        (the fake, which has no _stream). No reasoning present -> unchanged
        byte-for-byte from before reasoning extraction existed.

        ``tools``/``tool_choice`` (#18a): same wire shape and binding as
        ``complete``. When the model calls a tool, its tool_calls are
        accumulated across chunks (LangChain's summable ``AIMessageChunk``,
        when the provider streams deltas; the fake test model streams a
        single already-complete message, so the accumulator is a no-op) and
        surfaced as one ``Chunk(tool_calls=[...])`` right before the final
        ``Chunk(done=True)``. No ``tools`` -> unchanged (no accumulation, no
        extra chunk).
        """
        model = self._bind_tools(self._bound_model(), tools, tool_choice)
        agg = None
        for lc_chunk in model.stream(
                self._to_wire(messages), **self._invoke_kwargs()):
            chunks, agg = _drain_chunk(lc_chunk, agg, tools=tools)
            for c in chunks:
                yield c
        tool_calls = getattr(agg, "tool_calls", None) if agg is not None else None
        if tool_calls:
            yield Chunk(tool_calls=[dict(tc) for tc in tool_calls])
        yield Chunk(done=True)

    async def acomplete(
        self, messages: list[Message], *,
        tools: list[dict] | None = None, tool_choice: Any = None,
    ) -> Message:
        """Async single-shot completion. Used by LLMAgent.ainvoke. Delegates to
        the LangChain model's native ``ainvoke`` (does not block the loop).

        ``tools``/``tool_choice`` (#18a): see ``complete`` — identical
        semantics, async transport."""
        model = self._bind_tools(self._bound_model(), tools, tool_choice)
        lc_result = await model.ainvoke(
            self._to_wire(messages), **self._invoke_kwargs())
        return from_langchain(lc_result)

    async def astream(
        self, messages: list[Message], *,
        tools: list[dict] | None = None, tool_choice: Any = None,
    ) -> AsyncIterator[Chunk]:
        """Async neutral streaming. Used by LLMAgent.astream. Mirrors stream()
        over the model's native ``astream`` (same reasoning-before-content
        ordering per chunk; see stream()).

        ``tools``/``tool_choice`` (#18a): see ``stream`` — identical
        semantics (tool-call accumulation and the pre-done Chunk), async
        transport."""
        model = self._bind_tools(self._bound_model(), tools, tool_choice)
        agg = None
        async for lc_chunk in model.astream(
                self._to_wire(messages), **self._invoke_kwargs()):
            chunks, agg = _drain_chunk(lc_chunk, agg, tools=tools)
            for c in chunks:
                yield c
        tool_calls = getattr(agg, "tool_calls", None) if agg is not None else None
        if tool_calls:
            yield Chunk(tool_calls=[dict(tc) for tc in tool_calls])
        yield Chunk(done=True)
