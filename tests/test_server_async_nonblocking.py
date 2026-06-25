# tests/test_server_async_nonblocking.py
"""The server no longer blocks the event loop on agent work: it awaits
agent.ainvoke (async-native or a threaded bridge). Two concurrent slow requests
therefore overlap instead of serializing — the headline win of going async.

Uses httpx.AsyncClient + ASGITransport for *real* concurrency (TestClient is
sync and would serialize regardless)."""
from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from aixon.agent import Agent
from aixon.message import Chunk, Message
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.server import Server

_SLEEP = 0.3


@pytest.fixture(autouse=True)
def _reset_server():
    Server._reset()
    yield
    Server._reset()


def _register_slow_agent():
    class SlowEchoAgent(Agent):
        name = "slowecho"

        def invoke(self, messages):
            time.sleep(_SLEEP)  # blocking work — must run off the event loop
            return Message(role="assistant", content="ok")

        def stream(self, messages):
            yield Chunk(content="ok")
            yield Chunk(done=True)


def test_two_concurrent_requests_do_not_serialize():
    _register_slow_agent()
    app = Server(adapters=[OpenAIAdapter()]).app

    async def _hit(client):
        return await client.post(
            "/v1/chat/completions",
            json={"model": "slowecho", "messages": [{"role": "user", "content": "hi"}]},
        )

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            t0 = time.monotonic()
            r1, r2 = await asyncio.gather(_hit(client), _hit(client))
            return r1, r2, time.monotonic() - t0

    r1, r2, elapsed = asyncio.run(_run())
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["choices"][0]["message"]["content"] == "ok"
    # If the loop blocked, two 0.3s requests would take ~0.6s. Overlapped, ~0.3s.
    assert elapsed < _SLEEP * 1.8, f"requests serialized (took {elapsed:.2f}s)"
