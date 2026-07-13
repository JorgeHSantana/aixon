# tests/test_reasoning_extraction.py
"""R2: normalized reasoning extraction, non-stream and streaming.

Verified vendor shapes (langchain-anthropic 1.4.7, installed in ./.venv):
- Non-stream: ``ChatAnthropic._format_output`` dumps the raw Anthropic API
  response content list almost verbatim into ``AIMessage.content``. A
  thinking block looks like ``{"type": "thinking", "thinking": "...",
  "signature": "..."}``.
- Streaming: ``_make_message_chunk_from_anthropic_event`` turns each
  ``thinking_delta``/``signature_delta`` SSE event into its own
  ``AIMessageChunk`` whose ``.content`` is a *single-item* list:
  ``[{"type": "thinking", "thinking": "<delta text>", "index": N}]`` for a
  ``thinking_delta``, or ``[{"type": "thinking", "signature": "...",
  "index": N}]`` (no ``"thinking"`` key) for the trailing ``signature_delta``
  — the latter carries no visible reasoning delta.

zai/GLM's ``reasoning_content`` lands in ``additional_kwargs`` (OpenAI SDK
pydantic models use ``extra="allow"``, so unknown response fields survive
``model_dump()``); this test file fakes that shape directly since it is a
provider convention, not something langchain-openai itself guarantees.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from aixon._interop.messages import (
    from_langchain,
    reasoning_from_chunk,
    reasoning_from_message,
)
from aixon.message import Chunk, Message
from aixon.providers.base import Provider, register_provider

# ── reasoning_from_message / reasoning_from_chunk (unit) ─────────────────────


def test_reasoning_from_message_none_when_absent():
    assert reasoning_from_message(AIMessage(content="hi")) is None


def test_reasoning_from_message_thinking_blocks_in_content_list():
    msg = AIMessage(
        content=[
            {"type": "thinking", "thinking": "Let me think...", "signature": "sig"},
            {"type": "text", "text": "Here's the answer."},
        ]
    )
    assert reasoning_from_message(msg) == "Let me think..."


def test_reasoning_from_message_concatenates_multiple_thinking_blocks_in_order():
    msg = AIMessage(
        content=[
            {"type": "thinking", "thinking": "First. "},
            {"type": "text", "text": "ignored"},
            {"type": "thinking", "thinking": "Second."},
        ]
    )
    assert reasoning_from_message(msg) == "First. Second."


def test_reasoning_from_message_reasoning_content_convention():
    msg = AIMessage(
        content="answer", additional_kwargs={"reasoning_content": "GLM thought."}
    )
    assert reasoning_from_message(msg) == "GLM thought."


def test_reasoning_from_message_thinking_blocks_before_reasoning_content():
    msg = AIMessage(
        content=[{"type": "thinking", "thinking": "Claude-style. "}],
        additional_kwargs={"reasoning_content": "GLM-style."},
    )
    assert reasoning_from_message(msg) == "Claude-style. GLM-style."


def test_reasoning_from_message_ignores_signature_only_thinking_block():
    # Trailing signature_delta block carries no "thinking" text key.
    msg = AIMessage(content=[{"type": "thinking", "signature": "sig-only"}])
    assert reasoning_from_message(msg) is None


def test_reasoning_from_chunk_is_reasoning_from_message():
    # Same extraction logic: a streamed chunk's content/additional_kwargs is
    # already a per-chunk delta, exactly like the non-stream case operates on
    # one message's full content.
    assert reasoning_from_chunk is reasoning_from_message


# ── from_langchain (non-stream, full integration through the neutral Message) ─


def test_from_langchain_extracts_thinking_blocks_into_reasoning():
    msg = AIMessage(
        content=[
            {"type": "thinking", "thinking": "Reasoning here."},
            {"type": "text", "text": "Final answer."},
        ]
    )
    m = from_langchain(msg)
    assert m.reasoning == "Reasoning here."
    assert m.content == "Final answer."  # thinking excluded from flattened content


def test_from_langchain_combines_thinking_and_reasoning_content():
    msg = AIMessage(
        content=[
            {"type": "thinking", "thinking": "Claude thought. "},
            {"type": "text", "text": "Answer."},
        ],
        additional_kwargs={"reasoning_content": "GLM thought."},
    )
    m = from_langchain(msg)
    assert m.reasoning == "Claude thought. GLM thought."
    assert m.content == "Answer."


def test_from_langchain_reasoning_content_only_unchanged():
    # Pre-existing behavior (additional_kwargs only) must still work.
    msg = AIMessage(
        content="answer", additional_kwargs={"reasoning_content": "I thought."}
    )
    m = from_langchain(msg)
    assert m.reasoning == "I thought."


def test_from_langchain_no_reasoning_present():
    m = from_langchain(AIMessage(content="plain answer"))
    assert m.reasoning is None


# ── LLM.stream / LLM.astream (streaming, through the fake provider) ──────────


class _ReasoningStreamChatModel(BaseChatModel):
    """Fake model with a REAL ``_stream``/``_astream`` (unlike the shared
    FakeChatModel in tests/_fakes.py, which has none and falls back to a
    single chunk from ``_generate``). Scripted per-test via ``chunks``:
    a list of ``AIMessageChunk`` fed verbatim to the LLM.stream/astream loop.
    """

    chunks: list = []

    @property
    def _llm_type(self) -> str:
        return "fake-reasoning-stream"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in self.chunks:
            yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self.chunks:
            yield ChatGenerationChunk(message=chunk)


class _ReasoningStreamProvider(Provider):
    name = "fake-reasoning-stream"
    env_key = "FAKE_REASONING_STREAM_API_KEY"

    def build(self, model: str, **params: Any) -> _ReasoningStreamChatModel:
        return _ReasoningStreamChatModel()


def _make_reasoning_stream_llm(chunks: list) -> Any:
    register_provider(_ReasoningStreamProvider())
    from aixon.llm import LLM

    llm = LLM("fake-1", provider="fake-reasoning-stream")
    llm.chat_model.chunks = chunks
    return llm


def test_stream_emits_no_reasoning_chunk_when_absent():
    llm = _make_reasoning_stream_llm([AIMessageChunk(content="hello")])
    chunks = list(llm.stream([Message(role="user", content="hi")]))
    assert chunks == [Chunk(content="hello"), Chunk(done=True)]


def test_stream_emits_reasoning_chunk_from_thinking_delta():
    llm = _make_reasoning_stream_llm(
        [AIMessageChunk(content=[{"type": "thinking", "thinking": "pondering"}])]
    )
    chunks = list(llm.stream([Message(role="user", content="hi")]))
    assert chunks == [Chunk(reasoning="pondering"), Chunk(done=True)]


def test_stream_emits_reasoning_before_content_in_same_chunk():
    lc_chunk = AIMessageChunk(
        content=[
            {"type": "thinking", "thinking": "pondering"},
            {"type": "text", "text": "the answer"},
        ]
    )
    llm = _make_reasoning_stream_llm([lc_chunk])
    chunks = list(llm.stream([Message(role="user", content="hi")]))
    assert chunks == [
        Chunk(reasoning="pondering"),
        Chunk(content="the answer"),
        Chunk(done=True),
    ]


def test_stream_signature_only_thinking_delta_yields_no_reasoning_chunk():
    # Trailing signature_delta: content-list block with no "thinking" text.
    lc_chunk = AIMessageChunk(content=[{"type": "thinking", "signature": "sig"}])
    llm = _make_reasoning_stream_llm([lc_chunk])
    chunks = list(llm.stream([Message(role="user", content="hi")]))
    assert chunks == [Chunk(done=True)]


def test_stream_extracts_reasoning_content_from_additional_kwargs():
    lc_chunk = AIMessageChunk(
        content="", additional_kwargs={"reasoning_content": "glm delta"}
    )
    llm = _make_reasoning_stream_llm([lc_chunk])
    chunks = list(llm.stream([Message(role="user", content="hi")]))
    assert chunks == [Chunk(reasoning="glm delta"), Chunk(done=True)]


def test_stream_multiple_chunks_reasoning_then_content_over_time():
    llm = _make_reasoning_stream_llm(
        [
            AIMessageChunk(content=[{"type": "thinking", "thinking": "step 1. "}]),
            AIMessageChunk(content=[{"type": "thinking", "thinking": "step 2."}]),
            AIMessageChunk(content=[{"type": "text", "text": "answer"}]),
        ]
    )
    chunks = list(llm.stream([Message(role="user", content="hi")]))
    assert chunks == [
        Chunk(reasoning="step 1. "),
        Chunk(reasoning="step 2."),
        Chunk(content="answer"),
        Chunk(done=True),
    ]


def test_astream_emits_reasoning_before_content_in_same_chunk():
    import asyncio

    lc_chunk = AIMessageChunk(
        content=[
            {"type": "thinking", "thinking": "pondering"},
            {"type": "text", "text": "the answer"},
        ]
    )
    llm = _make_reasoning_stream_llm([lc_chunk])

    async def collect():
        return [c async for c in llm.astream([Message(role="user", content="hi")])]

    chunks = asyncio.run(collect())
    assert chunks == [
        Chunk(reasoning="pondering"),
        Chunk(content="the answer"),
        Chunk(done=True),
    ]


def test_astream_emits_no_reasoning_chunk_when_absent():
    import asyncio

    llm = _make_reasoning_stream_llm([AIMessageChunk(content="hello")])

    async def collect():
        return [c async for c in llm.astream([Message(role="user", content="hi")])]

    chunks = asyncio.run(collect())
    assert chunks == [Chunk(content="hello"), Chunk(done=True)]
