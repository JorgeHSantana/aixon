"""Async usage of the same agents — runnable, offline.

Every agent exposes async `ainvoke`/`astream` alongside the sync `invoke`/`stream`
(sync stays the default; async is additive). This script shows:

  1. await orchestrator.ainvoke(...)              — one async call
  2. async for chunk in orchestrator.astream(...) — async streaming
  3. asyncio.gather(...) of several requests      — they run concurrently,
     not serialized, because ainvoke never blocks the event loop

Run it:
    cd examples/support_assistant
    python async_demo.py
"""

from __future__ import annotations

import asyncio
import os
import time

os.environ.pop("OPENAI_API_KEY", None)  # force the offline demo provider

from aixon import Message, autodiscover, get_registry  # noqa: E402

autodiscover("agents")
support = get_registry().resolve("support")


async def main() -> None:
    # 1) single async call
    reply = await support.ainvoke([Message(role="user", content="where is my order 1002?")])
    print("ainvoke :", reply.content.replace("\n", " ")[:80])

    # 2) async streaming
    print("astream :", end=" ")
    async for chunk in support.astream([Message(role="user", content="how do I reset my password?")]):
        if chunk.content:
            print(chunk.content.replace("\n", " ")[:80], end="")
    print()

    # 3) concurrency — three requests at once, overlapped not serialized
    questions = ["order 1001 status?", "how do I enable SSO?", "refund order 1003"]
    t0 = time.monotonic()
    replies = await asyncio.gather(
        *(support.ainvoke([Message(role="user", content=q)]) for q in questions)
    )
    print(f"gather  : {len(replies)} replies in {time.monotonic() - t0:.3f}s (concurrent)")


if __name__ == "__main__":
    asyncio.run(main())
