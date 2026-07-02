# tests/test_audit_retrievers.py
"""Audit fixes: WeaviateRetriever (rerank k, Ranker cache, metadatas mismatch,
close reset), RagieRetriever (metadatas mismatch) e OpenAIAdapter (list
content, reasoning deltas em content mode). Tudo mockado — sem rede."""
from __future__ import annotations

import sys
import types

import pytest

from aixon.embedding import Embedding
from aixon.retriever import TypeAccess
from aixon.retrievers.ragie import RagieRetriever
from aixon.retrievers.weaviate import WeaviateRetriever


# --- fakes: weaviate ------------------------------------------------------

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
        return [_Doc(f"t{i}", {"i": i}) for i in range(kw["k"])]

    def add_texts(self, texts, metadatas=None, ids=None):
        self.added.append((list(texts), metadatas, ids))
        return [f"id-{i}" for i in range(len(list(texts)))]


class _FakeRanker:
    instances = 0

    def __init__(self, *args, **kwargs):
        type(self).instances += 1

    def rerank(self, req):
        return [{"text": p["text"], "score": 1.0 - i * 0.01, "meta": p["meta"]}
                for i, p in enumerate(req.passages)]


class _FakeRerankRequest:
    def __init__(self, query=None, passages=None):
        self.query = query
        self.passages = passages


class _WeavRetriever(WeaviateRetriever):
    collection_name = "C"
    embedding = _FakeEmbedding()


class _WeavRerankRetriever(WeaviateRetriever):
    collection_name = "C"
    embedding = _FakeEmbedding()
    rerank = True


class _WeavWritableRetriever(WeaviateRetriever):
    collection_name = "C"
    embedding = _FakeEmbedding()
    type_access = TypeAccess.ALL


@pytest.fixture
def patch_vs(monkeypatch):
    pytest.importorskip("langchain_weaviate")
    pytest.importorskip("langchain_text_splitters")
    _FakeVS.last = None
    monkeypatch.setattr("langchain_weaviate.WeaviateVectorStore", _FakeVS)
    return _FakeVS


@pytest.fixture
def patch_flashrank(monkeypatch):
    fake = types.ModuleType("flashrank")
    fake.Ranker = _FakeRanker
    fake.RerankRequest = _FakeRerankRequest
    _FakeRanker.instances = 0
    monkeypatch.setitem(sys.modules, "flashrank", fake)
    return _FakeRanker


# --- Finding 1: rerank must respect the effective k -----------------------

def test_rerank_respects_per_call_k(patch_vs, patch_flashrank):
    docs = _WeavRerankRetriever(client=object()).search("q", k=1)
    assert len(docs) == 1
    # rerank still fetches the wide candidate pool for quality
    assert _FakeVS.last.search_kw["k"] == _WeavRerankRetriever.rerank_fetch_k


def test_rerank_respects_constructor_k(patch_vs, patch_flashrank):
    docs = _WeavRerankRetriever(client=object(), k=2).search("q")
    assert len(docs) == 2


def test_rerank_top_k_still_caps_large_k(patch_vs, patch_flashrank):
    docs = _WeavRerankRetriever(client=object()).search("q", k=50)
    assert len(docs) == _WeavRerankRetriever.rerank_top_k  # min(k, top_k)


def test_rerank_default_k_uses_max_query_results(patch_vs, patch_flashrank):
    docs = _WeavRerankRetriever(client=object()).search("q")
    assert len(docs) == _WeavRerankRetriever.max_query_results


# --- Finding 2: OpenAI list-form content must flatten to plain text -------

def test_parse_request_flattens_list_content():
    from aixon.server.adapters.openai import OpenAIAdapter

    body = {"model": "m", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "http://x"}},
        {"type": "text", "text": " world"},
    ]}]}
    parsed = OpenAIAdapter().parse_request(body, path="/v1/chat/completions")
    assert parsed.messages[0].content == "hello world"


def test_parse_request_string_and_none_content():
    from aixon.server.adapters.openai import OpenAIAdapter

    body = {"model": "m", "messages": [
        {"role": "user", "content": "plain"},
        {"role": "assistant", "content": None},
    ]}
    parsed = OpenAIAdapter().parse_request(body, path="/v1/chat/completions")
    assert parsed.messages[0].content == "plain"
    assert parsed.messages[1].content == ""


# --- Finding 3: content mode must not join reasoning deltas with "\n" ------

def test_content_mode_multi_reasoning_deltas_are_raw():
    import json

    from aixon.message import Chunk
    from aixon.server.adapters.openai import OpenAIAdapter
    from aixon.server.protocol import ParsedRequest

    adapter = OpenAIAdapter()
    request = ParsedRequest(model="m", messages=[], params={}, stream=True)
    session = adapter.open_stream(model="m", request=request)
    raw = session.chunk(Chunk(reasoning="thin"))
    raw += session.chunk(Chunk(reasoning="king"))
    raw += session.chunk(Chunk(content="ans"))
    raw += session.chunk(Chunk(done=True))
    parts = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block.startswith("data: "):
            continue
        payload = json.loads(block[len("data: "):])
        for choice in payload["choices"]:
            parts.append(choice["delta"].get("content", ""))
    content = "".join(parts)
    assert content == "<think>\nthinking</think>\nans"


# --- Finding 4: flashrank Ranker cached per instance -----------------------

def test_rerank_ranker_instantiated_once(patch_vs, patch_flashrank):
    r = _WeavRerankRetriever(client=object())
    r.search("q1")
    r.search("q2")
    assert patch_flashrank.instances == 1


# --- fakes: ragie ----------------------------------------------------------

class _RagieDocs:
    def __init__(self):
        self.created = []

    def create_raw(self, *, request):
        self.created.append(request)
        return type("D", (), {"id": f"doc-{len(self.created)}"})()


class _FakeRagie:
    def __init__(self):
        self.documents = _RagieDocs()


class _RagieWritableRetriever(RagieRetriever):
    partition = "p1"
    type_access = TypeAccess.ALL


# --- Finding 5: metadatas length mismatch must raise, not truncate ---------

def test_weaviate_write_metadatas_mismatch_raises(patch_vs):
    r = _WeavWritableRetriever(client=object())
    with pytest.raises(ValueError, match="metadatas"):
        r.write(["a", "b", "c"], metadatas=[{"x": 1}])
    # nothing was silently indexed (validation happens before _ensure)
    assert _FakeVS.last is None or not _FakeVS.last.added


def test_weaviate_write_matching_metadatas_still_works(patch_vs):
    ids = _WeavWritableRetriever(client=object()).write(
        ["a", "b"], metadatas=[{"x": 1}, {"x": 2}])
    assert ids == ["id-0", "id-1"]


def test_ragie_write_metadatas_mismatch_raises():
    fake = _FakeRagie()
    with pytest.raises(ValueError, match="metadatas"):
        _RagieWritableRetriever(client=fake).write(
            ["a", "b", "c"], metadatas=[{"x": 1}])
    assert fake.documents.created == []  # nothing was silently indexed


def test_ragie_write_matching_metadatas_still_works():
    fake = _FakeRagie()
    ids = _RagieWritableRetriever(client=fake).write(
        ["a", "b"], metadatas=[{"x": 1}, {"x": 2}])
    assert ids == ["doc-1", "doc-2"]


# --- Finding 6: close() must reset state so the next call reconnects -------

def test_close_resets_state_and_search_reconnects(monkeypatch, patch_vs):
    weaviate = pytest.importorskip("weaviate")

    class _FakeClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    connects: list[_FakeClient] = []

    def _connect(**kwargs):
        c = _FakeClient()
        connects.append(c)
        return c

    monkeypatch.setattr(weaviate, "connect_to_local", _connect)
    r = _WeavRetriever()  # owns the client (none injected)
    assert r.search("q")
    assert len(connects) == 1
    r.close()
    assert connects[0].closed
    assert r.search("q")  # must reconnect, not reuse the closed client
    assert len(connects) == 2
