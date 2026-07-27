"""#19 — an agent must never die mute when max_execution_time breaks a
stream/astream run with NO accumulated content: a minimal, honest timeout
message is emitted as CONTENT (the reasoning "(stopped: ...)" line, hidden by
chat UIs' collapsed thinking block, is not enough). With PARTIAL content
already produced, today's behavior is unchanged: deliver it, no timeout
message added.

Idiom: FakeChatModel (tests/_fakes.py) + a slow async tool (as in
tests/test_astream_live_drain.py) for the "tool trava" cases, and a
monkeypatched aixon.agents.tool_agent.time.monotonic (as in
tests/test_tool_agent_deadline.py) for the boundary case where content is
already captured before the deadline is detected — real timing can't hit
that window deterministically because the graph naturally ends right after a
no-tool_calls answer."""
from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage

from aixon.agents.tool_agent import ToolAgent
from aixon.message import Message
from aixon.registry import get_registry
from tests._fakes import make_llm


def test_astream_timeout_zero_content_emits_message():
    # A tool that hangs well past the deadline, with the model never getting
    # a chance to answer: final_content stays empty, so astream must emit
    # the timeout message as content instead of closing mute. The reasoning
    # "(stopped: exceeded max_execution_time ...)" line must still be there.
    async def lenta(x: str) -> str:
        "tool que trava além do deadline"
        await asyncio.sleep(1.0)
        return "nunca chega"

    llm = make_llm()
    llm.chat_model.script = [
        AIMessage(content="", tool_calls=[{"name": "lenta", "args": {"x": "1"}, "id": "c1"}]),
        AIMessage(content="fim"),
    ]
    cls = type(
        "Timeout19ZeroAsyncAgent", (ToolAgent,),
        {"name": "timeout19zeroasync", "llm": llm, "tools": [lenta], "max_execution_time": 0.2},
    )
    agent = get_registry().resolve("timeout19zeroasync")

    async def run():
        return [c async for c in agent.astream([Message(role="user", content="vai")])]

    chunks = asyncio.run(run())

    reasoning = "".join(c.reasoning for c in chunks if c.reasoning)
    assert "(stopped: exceeded max_execution_time" in reasoning

    contents = [c.content for c in chunks if c.content]
    assert contents == [agent.timeout_message.format(seconds=agent.max_execution_time)]
    assert chunks[-1].done is True


def test_astream_timeout_with_partial_content_keeps_current_behavior(monkeypatch):
    # Content was already produced (a completed no-tool_calls answer) BEFORE
    # the deadline broke the polling loop. Real timing can't land the
    # deadline check exactly between "content captured" and "next poll" — the
    # graph naturally ends right there via StopAsyncIteration, and
    # langgraph's internal step count (many more astream() ticks than the 3
    # "logical" graph updates) makes a fixed call-count fake_monotonic
    # brittle. Instead, the clock is monkeypatched (same technique as
    # test_tool_agent_deadline.py) to stay within budget UNTIL
    # ``_flatten_content`` — the exact call that captures ``final_content`` —
    # has fired, then jump past the deadline: this lands timed_out=True on
    # the very next remaining-check, always AFTER final_content is set,
    # regardless of how many internal ticks that takes.
    import aixon._interop.messages as messages_mod

    async def rapida(x: str) -> str:
        "tool rapida, sem sleep"
        return "ok"

    llm = make_llm()
    llm.chat_model.script = [
        AIMessage(content="", tool_calls=[{"name": "rapida", "args": {"x": "1"}, "id": "c1"}]),
        AIMessage(content="resposta parcial"),
    ]
    cls = type(
        "Timeout19PartialAgent", (ToolAgent,),
        {"name": "timeout19partial", "llm": llm, "tools": [rapida], "max_execution_time": 10},
    )
    agent = get_registry().resolve("timeout19partial")

    captured = {"done": False}
    orig_flatten = messages_mod._flatten_content

    def flatten_and_flag(x):
        text = orig_flatten(x)
        if text:
            captured["done"] = True
        return text

    monkeypatch.setattr(messages_mod, "_flatten_content", flatten_and_flag)

    def fake_monotonic():
        # Before content is captured: a fixed in-budget time. Once captured:
        # a time far past the deadline, from the very next call on.
        return 1.0 if not captured["done"] else 999.0

    monkeypatch.setattr("aixon.agents.tool_agent.time.monotonic", fake_monotonic)

    async def run():
        return [c async for c in agent.astream([Message(role="user", content="vai")])]

    chunks = asyncio.run(run())

    contents = [c.content for c in chunks if c.content]
    assert contents == ["resposta parcial"], (
        f"partial content must pass through unchanged (current behavior): {contents}")
    assert not any("tempo limite" in c for c in contents), (
        "no timeout message must be added once content was already produced")
    assert chunks[-1].done is True


def test_sync_stream_timeout_zero_content_emits_message():
    # Sync counterpart: stream()'s deadline check runs between updates (no
    # cancellation, no poll loop). A tool that blocks past the deadline
    # forces the break before the model ever produces a final answer.
    import time as _time

    def lenta(x: str) -> str:
        "tool sincrona que trava além do deadline"
        _time.sleep(1.0)
        return "nunca chega"

    llm = make_llm()
    llm.chat_model.script = [
        AIMessage(content="", tool_calls=[{"name": "lenta", "args": {"x": "1"}, "id": "c1"}]),
        AIMessage(content="fim"),
    ]
    cls = type(
        "Timeout19ZeroSyncAgent", (ToolAgent,),
        {"name": "timeout19zerosync", "llm": llm, "tools": [lenta], "max_execution_time": 0.2},
    )
    agent = get_registry().resolve("timeout19zerosync")

    chunks = list(agent.stream([Message(role="user", content="vai")]))

    reasoning = "".join(c.reasoning for c in chunks if c.reasoning)
    assert "(stopped: exceeded max_execution_time" in reasoning

    contents = [c.content for c in chunks if c.content]
    assert contents == [agent.timeout_message.format(seconds=agent.max_execution_time)]
    assert chunks[-1].done is True
