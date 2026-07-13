"""Conversion helpers between neutral Message/Chunk and LangChain types.

INTERNAL to aixon. Public code speaks only Message/Chunk. LLM, LLMAgent,
ToolAgent, and Orchestrator call these helpers at the boundary where they
must interact with LangChain internals. Validated for LangChain 1.x.
"""
from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from aixon.message import Message, Role


def _flatten_content(content: object) -> str:
    """Flatten LangChain message content to plain visible text.

    Some providers (e.g. Gemini 2.5) return ``content`` as a list of content
    blocks (``[{"type": "text", "text": ...}, ...]``) rather than a plain
    string. The neutral Message carries plain text, so join the text-bearing
    blocks and drop non-text blocks (thinking/reasoning/tool_use). A plain
    string is returned unchanged; any other shape falls back to ``str()``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def reasoning_from_message(msg: object) -> str | None:
    """Extract the reasoning text carried by a LangChain AIMessage (non-stream)
    or AIMessageChunk (one streaming delta) — same two sources, in this order:

    1. Anthropic-style ``thinking`` blocks in a list ``.content``: entries
       shaped ``{"type": "thinking", "thinking": "...", ...}``. This is the
       wire shape verbatim — langchain-anthropic 1.4.7's ``_format_output``
       (non-stream) dumps the raw Anthropic API response content list with
       only unrelated keys stripped, and its streaming event handler
       (``_make_message_chunk_from_anthropic_event``) emits one such block per
       ``thinking_delta`` SSE event (a trailing ``signature_delta`` block has
       no ``"thinking"`` key and contributes nothing here).
    2. The ``reasoning_content`` convention some OpenAI-compatible providers
       (e.g. zai/GLM) place in ``additional_kwargs``.

    Blocks/values are concatenated in the order above, thinking-blocks first.
    Returns ``None`` when neither source is present — this is what keeps
    no-reasoning behavior unchanged: nothing is added, nothing is dropped.
    """
    parts: list[str] = []
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                thinking = block.get("thinking")
                if isinstance(thinking, str) and thinking:
                    parts.append(thinking)
    kwargs = getattr(msg, "additional_kwargs", None)
    if kwargs:
        reasoning_content = kwargs.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content:
            parts.append(reasoning_content)
    return "".join(parts) or None


# Streaming chunks (AIMessageChunk) carry the exact same two shapes as a full
# AIMessage, except each chunk's .content/.additional_kwargs already IS the
# per-chunk delta (not the accumulated total) — mirroring how `_flatten_content`
# is applied to individual stream chunks in aixon.llm. The extraction logic is
# identical; this alias just names the call site (LLM.stream/astream) that
# uses it, distinct from from_langchain's non-stream call site.
reasoning_from_chunk = reasoning_from_message


def to_langchain(messages: list[Message]) -> list[BaseMessage]:
    """Convert neutral Message[] to LangChain message objects.

    Mapping:
        system    → SystemMessage
        user      → HumanMessage
        assistant → AIMessage (tool_calls forwarded if present)
        tool      → ToolMessage (requires tool_call_id)

    Does NOT reconstruct Anthropic thinking blocks (nor any other reasoning
    shape) on the way back in: a neutral Message's `.reasoning` is a plain
    string with no signature/block structure to round-trip, and clients don't
    need to resend it — ToolAgent's internal LangGraph loop keeps native
    LangChain messages (with any Claude thinking-block signatures) in its own
    graph state across turns, so those survive there independently of this
    conversion. Reasoning coming back from a neutral history (e.g. a client
    replaying `Message.reasoning` from a prior response) is simply dropped.
    """
    result: list[BaseMessage] = []
    for msg in messages:
        role = msg.role
        if role in ("system", "developer"):
            # OpenAI's modern spec adds "developer" as a system-role alias
            # (superseding "system" for newer models); aixon has no separate
            # LangChain concept for it, so it collapses to SystemMessage.
            result.append(SystemMessage(content=msg.content))
        elif role == "user":
            result.append(HumanMessage(content=msg.content))
        elif role == "assistant":
            kwargs: dict = {"content": msg.content}
            if msg.tool_calls:
                kwargs["tool_calls"] = msg.tool_calls
            result.append(AIMessage(**kwargs))
        elif role == "tool":
            result.append(
                ToolMessage(
                    content=msg.content,
                    tool_call_id=msg.tool_call_id or "",
                    name=msg.name,
                )
            )
        else:
            raise ValueError(
                f"Unknown message role '{role}'. "
                f"Expected one of: system, user, assistant, tool."
            )
    return result


def from_langchain(msg: BaseMessage) -> Message:
    """Convert a LangChain BaseMessage to a neutral Message.

    - Role inferred from the LangChain type.
    - tool_calls: forwarded from AIMessage.tool_calls (list of dicts).
    - reasoning: thinking-blocks from a list `.content` (if any) followed by
      additional_kwargs['reasoning_content'] (if any) — see
      `reasoning_from_message`.
    - usage: converted from AIMessage.usage_metadata (provider-real counts,
      LangChain naming input/output_tokens) to the neutral OpenAI shape
      (prompt/completion/total_tokens); None when the provider reported none.
    """
    # Explicitly annotated: without it, mypy widens the multi-branch literal
    # assignments below to plain `str` instead of keeping them narrowed to
    # `Role`, and the `Message(role=role, ...)` call below would fail typing.
    role: Role
    if isinstance(msg, AIMessage):
        role = "assistant"
    elif isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, SystemMessage):
        role = "system"
    elif isinstance(msg, ToolMessage):
        role = "tool"
    else:
        role = "assistant"  # safe fallback for unknown LangChain types

    content = _flatten_content(msg.content)

    tool_calls: list[dict] = []
    if isinstance(msg, AIMessage) and msg.tool_calls:
        tool_calls = [dict(tc) for tc in msg.tool_calls]

    reasoning = reasoning_from_message(msg)

    # Preserve tool-routing fields so a Message -> LangChain -> Message round-trip
    # of a tool message keeps its tool_call_id/name. Without this, to_langchain
    # would rebuild a ToolMessage with an empty tool_call_id on the next turn.
    tool_call_id = getattr(msg, "tool_call_id", None)
    name = getattr(msg, "name", None)

    usage = usage_from_metadata(getattr(msg, "usage_metadata", None)) \
        if isinstance(msg, AIMessage) else None

    return Message(
        role=role,
        content=content,
        name=name,
        tool_call_id=tool_call_id,
        tool_calls=tool_calls,
        reasoning=reasoning or None,
        usage=usage,
    )


def usage_from_metadata(usage_metadata: object) -> dict[str, int] | None:
    """LangChain ``usage_metadata`` (input/output/total_tokens) -> neutral
    OpenAI-shaped usage (prompt/completion/total_tokens). ``None``/empty/
    non-dict metadata -> None (no usage reported by the provider)."""
    if not isinstance(usage_metadata, dict) or not usage_metadata:
        return None
    prompt = int(usage_metadata.get("input_tokens", 0) or 0)
    completion = int(usage_metadata.get("output_tokens", 0) or 0)
    total = int(usage_metadata.get("total_tokens", 0) or 0) or (prompt + completion)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
