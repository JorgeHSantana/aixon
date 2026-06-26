# tests/test_retriever_weaviate.py
"""WeaviateRetriever: lazy connect, Document->list[dict], filters passthrough,
write com chunking, rerank flashrank opcional. VectorStore mockado via
monkeypatch (sem rede). Requer os extras instalados (importorskip)."""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("weaviate")
pytest.importorskip("langchain_weaviate")
pytest.importorskip("langchain_text_splitters")

from aixon.embedding import Embedding
from aixon.exceptions import AixonError
from aixon.retriever import TypeAccess
from aixon.retrievers.weaviate import WeaviateRetriever


class _FakeEmbedding(Embedding):
    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0]


class _Doc:
    def __init__(self, text, meta):
        self.page_content = text
        self.metadata = meta


class _FakeVS:
    last = None

    def __init__(self, **kwargs):
        _FakeVS.last = self
        self.kwargs = kwargs
        self.added = []
        self.search_kw = None

    def similarity_search(self, query, **kw):
        self.search_kw = kw
        return [_Doc("hello", {"m": 1}), _Doc("world", {"m": 2})]

    def add_texts(self, texts, metadatas=None, ids=None):
        self.added.append((list(texts), metadatas, ids))
        return [f"id-{i}" for i in range(len(list(texts)))]


class _LibRetriever(WeaviateRetriever):
    collection_name = "C"
    embedding = _FakeEmbedding()


class _WritableRetriever(WeaviateRetriever):
    collection_name = "C"
    embedding = _FakeEmbedding()
    type_access = TypeAccess.ALL


@pytest.fixture
def patch_vs(monkeypatch):
    monkeypatch.setattr("langchain_weaviate.WeaviateVectorStore", _FakeVS)
    return _FakeVS


def test_search_maps_documents(patch_vs):
    docs = _LibRetriever(client=object()).search("q")
    assert docs == [{"text": "hello", "metadata": {"m": 1}},
                    {"text": "world", "metadata": {"m": 2}}]
    assert _FakeVS.last.search_kw["k"] == 5


def test_init_does_not_connect(monkeypatch):
    import weaviate
    called = {"n": 0}
    monkeypatch.setattr(
        weaviate, "connect_to_local",
        lambda **k: called.__setitem__("n", called["n"] + 1) or object())
    _LibRetriever()  # lazy: __init__ must not connect
    assert called["n"] == 0


def test_asearch_inherits_threadbridge(patch_vs):
    docs = asyncio.run(_LibRetriever(client=object()).asearch("q"))
    assert docs[0]["text"] == "hello"


def test_filters_forwarded(patch_vs):
    sentinel = object()
    _LibRetriever(client=object()).search("q", filters=sentinel)
    assert _FakeVS.last.search_kw["filters"] is sentinel


def test_read_only_write_raises(patch_vs):
    with pytest.raises(AixonError):
        _LibRetriever(client=object()).write(["x"])


def test_write_chunks_and_adds(patch_vs):
    ids = _WritableRetriever(client=object()).write(["short text"], [{"a": 1}])
    assert _FakeVS.last.added  # add_texts called
    assert ids == ["id-0"]


def test_rerank(monkeypatch, patch_vs):
    import sys
    import types

    class _Ranker:
        def rerank(self, req):
            return [{"text": "world", "score": 0.9, "meta": {"m": 2}},
                    {"text": "hello", "score": 0.5, "meta": {"m": 1}}]

    class _RerankRequest:
        def __init__(self, query=None, passages=None):
            pass

    fake = types.ModuleType("flashrank")
    fake.Ranker = _Ranker
    fake.RerankRequest = _RerankRequest
    monkeypatch.setitem(sys.modules, "flashrank", fake)

    class _RerankedRetriever(WeaviateRetriever):
        collection_name = "C"
        embedding = _FakeEmbedding()
        rerank = True
        rerank_top_k = 1

    docs = _RerankedRetriever(client=object()).search("q")
    assert len(docs) == 1
    assert docs[0]["text"] == "world"
    assert docs[0]["metadata"]["_rerank_score"] == 0.9


def test_missing_collection_raises():
    class _BadRetriever(WeaviateRetriever):
        embedding = _FakeEmbedding()

    with pytest.raises(AixonError):
        _BadRetriever()


def test_missing_embedding_raises():
    class _Bad2Retriever(WeaviateRetriever):
        collection_name = "C"

    with pytest.raises(AixonError):
        _Bad2Retriever()
