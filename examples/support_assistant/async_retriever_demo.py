"""Native-async retriever — shows `asearch()` + the dual-tool concurrency win.

The retriever has BOTH a sync `search()` (used by `aixon chat` / sync `invoke`)
and a native async `asearch()` (used by the server / `ainvoke`). The
`time.sleep`/`asyncio.sleep` here **simulate** a slow vector-DB / network
round-trip — swap them for a real async SDK (Weaviate/Ragie/Tavily) in
production. What's demonstrated is the *behavior*, and it's measurable:

  1. dual path — the same retriever works sync AND async (one impl each);
  2. concurrency — native `asearch` lets N retrievals OVERLAP (~1x latency)
     instead of serializing (~Nx), with no change to how agents are written.

Run:
    cd examples/support_assistant
    python async_retriever_demo.py
"""

from __future__ import annotations

import asyncio
import os
import time

os.environ.pop("OPENAI_API_KEY", None)  # offline

from aixon import LLM, Message, Retriever, ToolAgent, TypeAccess  # noqa: E402

import providers.demo  # noqa: E402,F401  (registers the offline 'demo' provider)
from knowledge.corpus import FAQ  # noqa: E402

LATENCY = 0.2  # simulated backend round-trip (stand-in for a real async SDK call)


class SlowFaqRetriever(Retriever):
    """FAQ search over a 'slow backend'. Sync and native-async impls return the
    same results; only how they wait differs."""

    description = "Search the FAQ (simulated slow backend)."
    type_access = TypeAccess.READ

    def _rank(self, query: str) -> list[dict]:
        words = query.lower().split()
        hits = [d for d in FAQ if any(w in d["text"].lower() for w in words)]
        return (hits or FAQ[:1])[:3]

    def search(self, query: str, *, k=None) -> list[dict]:
        time.sleep(LATENCY)            # blocking call (sync path: aixon chat / invoke)
        return self._rank(query)

    async def asearch(self, query: str, *, k=None) -> list[dict]:
        await asyncio.sleep(LATENCY)   # non-blocking call (async path: server / ainvoke)
        return self._rank(query)


QUERIES = ["reset password", "enable SSO", "export data", "mobile app", "upgrade plan"]


async def main() -> None:
    r = SlowFaqRetriever()
    tool = r.as_tool(name="faq", description="search faq")

    # 1) Dual tool — same retriever, both paths.
    print("dual tool: func?", tool.func is not None, "| coroutine?", tool.coroutine is not None)
    print("  sync  func('reset password') ->", tool.func("reset password").split(".")[0][:48])
    print("  async coroutine(...)          ->", (await tool.coroutine("reset password")).split(".")[0][:48])

    # 2) Concurrency — 5 retrievals.
    t = time.monotonic()
    for q in QUERIES:                       # sequential sync
        r.search(q)
    seq = time.monotonic() - t

    t = time.monotonic()
    await asyncio.gather(*(r.asearch(q) for q in QUERIES))  # concurrent async
    conc = time.monotonic() - t

    print(f"\n{len(QUERIES)} retrievals @ {LATENCY}s each:")
    print(f"  sequential sync : {seq:.2f}s")
    print(f"  concurrent async: {conc:.2f}s   <- native asearch overlaps, doesn't block")

    # 3) Through an agent — the demo provider calls the tool; ainvoke uses asearch.
    agent = type("FaqAgent", (ToolAgent,),
                 {"name": "faq-agent", "llm": LLM("demo-1", provider="demo"), "tools": [tool]})()
    reply = await agent.ainvoke([Message(role="user", content="how do I reset my password?")])
    print("\nagent.ainvoke ->", reply.content.replace("\n", " ")[:70])


if __name__ == "__main__":
    asyncio.run(main())
