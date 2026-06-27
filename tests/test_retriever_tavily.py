# tests/test_retriever_tavily.py
"""TavilyRetriever: web search -> list[dict], sync + true-async, read-only.
Clients injected (client=/aclient=) — no network, no Tavily SDK needed."""
from __future__ import annotations

import asyncio

import pytest

from aixon.exceptions import AixonError
from aixon.retrievers.tavily import TavilyRetriever

_RESPONSE = {
    "answer": "the answer",
    "results": [
        {"title": "T1", "url": "http://a", "content": "body one"},
        {"title": "T2", "url": "http://b", "content": ""},  # skipped (no content)
    ],
}


class _FakeSync:
    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return _RESPONSE


class _FakeAsync:
    def __init__(self):
        self.calls = []

    async def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return _RESPONSE


def test_search_maps_answer_and_results():
    fake = _FakeSync()
    docs = TavilyRetriever(client=fake).search("q", k=3)
    assert docs[0] == {"text": "Resumo AI: the answer",
                       "metadata": {"source": "tavily_answer", "query": "q"}}
    assert docs[1]["text"] == "Title: T1\nURL: http://a\nContent: body one"
    assert docs[1]["metadata"] == {"source": "tavily", "url": "http://a",
                                   "title": "T1", "query": "q"}
    assert len(docs) == 2  # empty-content result skipped
    assert fake.calls[0][1]["max_results"] == 3
    assert fake.calls[0][1]["include_answer"] is True
    assert fake.calls[0][1]["search_depth"] == "basic"


def test_asearch_uses_async_client():
    fake = _FakeAsync()
    docs = asyncio.run(TavilyRetriever(aclient=fake).asearch("q"))
    assert docs[0]["metadata"]["source"] == "tavily_answer"
    assert fake.calls[0][1]["max_results"] == 5  # default max_web_results


def test_read_only_write_raises():
    with pytest.raises(AixonError):
        TavilyRetriever(client=_FakeSync()).write(["x"])


def test_missing_api_key_is_lazy_raises_on_use(monkeypatch):
    # Instantiating must NOT require the key (class-body / autodiscover safe);
    # validated lazily on first use.
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    r = TavilyRetriever()  # no raise
    with pytest.raises(AixonError):
        r.search("q")       # raises here (no key)


def test_as_tool_dual():
    tool = TavilyRetriever(client=_FakeSync()).as_tool()
    assert tool.func is not None and tool.coroutine is not None
