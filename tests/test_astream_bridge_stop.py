# tests/test_astream_bridge_stop.py
"""The default astream bridge must abandon the sync stream when the consumer
stops early — not drain it to completion in the executor thread."""
from __future__ import annotations

import asyncio
import time
from typing import Iterator

import pytest

from aixon.agent import Agent
from aixon.message import Chunk, Message


def test_astream_bridge_stops_producer_on_consumer_break():
    events: list = []

    class DripStreamAgent(Agent):
        def invoke(self, messages: list[Message]) -> Message:
            return Message(role="assistant", content="x")

        def stream(self, messages: list[Message]) -> Iterator[Chunk]:
            try:
                for i in range(200):
                    events.append(i)
                    yield Chunk(content=str(i))
                    time.sleep(0.005)
            finally:
                events.append("closed")

    async def consume() -> int:
        agent = DripStreamAgent()
        got = 0
        async for _ in agent.astream([Message(role="user", content="hi")]):
            got += 1
            if got == 3:
                break
        return got

    assert asyncio.run(consume()) == 3
    produced = [e for e in events if isinstance(e, int)]
    # Pre-fix the producer drained all 200 chunks after the consumer left;
    # with the stop event it abandons within a few chunks.
    assert len(produced) < 100
    # gen.close() ran the generator's finally block, and only after the last
    # produced chunk.
    assert events[-1] == "closed"


def test_astream_bridge_full_consumption_unchanged():
    class TinyStreamAgent(Agent):
        def invoke(self, messages: list[Message]) -> Message:
            return Message(role="assistant", content="x")

        def stream(self, messages: list[Message]) -> Iterator[Chunk]:
            yield Chunk(content="a")
            yield Chunk(content="b")
            yield Chunk(done=True)

    async def consume() -> list[Chunk]:
        agent = TinyStreamAgent()
        return [c async for c in agent.astream([Message(role="user", content="hi")])]

    chunks = asyncio.run(consume())
    assert "".join(c.content for c in chunks) == "ab"
    assert chunks[-1].done is True


def test_astream_bridge_still_propagates_producer_exception():
    class BoomStreamAgent(Agent):
        def invoke(self, messages: list[Message]) -> Message:
            return Message(role="assistant", content="x")

        def stream(self, messages: list[Message]) -> Iterator[Chunk]:
            yield Chunk(content="a")
            raise RuntimeError("boom")

    async def consume() -> None:
        agent = BoomStreamAgent()
        async for _ in agent.astream([Message(role="user", content="hi")]):
            pass

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(consume())
