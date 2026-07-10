# tests/test_bug_sweep_vendors.py
"""Bug-sweep regressions for providers/retrievers/connector (P1, R1-R6, C1-C2).

Each test targets exactly one finding from the bug-sweep plan (Task 3):
  P1 - zai.py silently leaking OPENAI_API_KEY to the z.AI endpoint.
  R1 - WeaviateRetriever._ensure() race creating duplicate clients.
  R2 - WeaviateRetriever._rerank() race creating duplicate Ranker instances
       (covered indirectly: same lock as R1, exercised via _ensure test).
  R3 - stale tail chunks left behind when a re-written document shrinks, and
       silently colliding source_ids within a single write() call.
  R4 - Ragie computed fields (score/document_id/...) clobbered by user
       metadata instead of the other way around.
  R5 - WeaviateRetriever.metadata_fields mutable class-level default.
  R6 - Retriever.awrite() bridging the sync write() to a worker thread.
  C1 - Connector pooled AsyncClient (no more one-socket-per-call).
  C2 - Connector.get/post/aget/apost rejecting caller-supplied headers/timeout
       with a TypeError because they were also passed positionally.
"""
from __future__ import annotations

import asyncio
import threading
import uuid

import httpx
import pytest

pytest.importorskip("weaviate")
pytest.importorskip("langchain_weaviate")
pytest.importorskip("langchain_text_splitters")

from aixon.connector import Connector
from aixon.embedding import Embedding
from aixon.exceptions import AixonError
from aixon.retriever import Retriever, TypeAccess
from aixon.retrievers.ragie import RagieRetriever
from aixon.retrievers.weaviate import WeaviateRetriever


# ─────────────────────────── P1: zai.py key guard ───────────────────────────

def test_zai_build_refuses_missing_key(monkeypatch):
    from aixon.providers.zai import ZAIProvider

    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leak")
    with pytest.raises(AixonError, match="ZAI_API_KEY"):
        ZAIProvider().build("glm-4")


# ───────────────────── shared Weaviate fakes/helpers ───────────────────────

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

    def similarity_search(self, query, **kw):
        return [_Doc("hello", {"m": 1})]

    def add_texts(self, texts, metadatas=None, ids=None):
        self.added.append((list(texts), metadatas, ids))
        return [f"id-{i}" for i in range(len(list(texts)))]


# ───────────────────── R1/R2: lazy init is thread-safe ─────────────────────

def test_weaviate_lazy_init_is_thread_safe(monkeypatch):
    import weaviate

    monkeypatch.setattr("langchain_weaviate.WeaviateVectorStore", _FakeVS)

    calls = {"n": 0}
    call_lock = threading.Lock()

    def fake_connect_to_local(**kwargs):
        # Widen the race window so concurrent callers actually overlap.
        import time

        time.sleep(0.02)
        with call_lock:
            calls["n"] += 1
        return object()

    monkeypatch.setattr(weaviate, "connect_to_local", fake_connect_to_local)

    class _LazyInitRetriever(WeaviateRetriever):
        collection_name = "C"
        embedding = _FakeEmbedding()

    r = _LazyInitRetriever()
    threads = [threading.Thread(target=lambda: r.search("q")) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls["n"] == 1


# ───────────────────────── R3: stale-chunk purge ───────────────────────────

class _FakeCollectionData:
    def __init__(self, seeded_ids: set[str]):
        self._seeded = set(seeded_ids)
        self.deleted: list[str] = []

    def delete_by_id(self, id_) -> bool:
        sid = str(id_)
        self.deleted.append(sid)
        if sid in self._seeded:
            self._seeded.discard(sid)
            return True
        return False


class _FakeCollection:
    def __init__(self, data: _FakeCollectionData):
        self.data = data


class _FakeCollections:
    def __init__(self, data: _FakeCollectionData):
        self._collection = _FakeCollection(data)

    def get(self, name):
        return self._collection


class _FakeWeaviateClient:
    def __init__(self, data: _FakeCollectionData):
        self.collections = _FakeCollections(data)


class _WritableRetriever(WeaviateRetriever):
    collection_name = "C"
    embedding = _FakeEmbedding()
    type_access = TypeAccess.ALL


def test_weaviate_rewrite_purges_stale_tail_chunks(monkeypatch):
    monkeypatch.setattr("langchain_weaviate.WeaviateVectorStore", _FakeVS)

    src = "doc1"
    ns = uuid.uuid5(uuid.NAMESPACE_DNS, src)
    # Pre-seed 2 "old" tail chunks (as if a previous, longer write had 3
    # chunks: ci=0,1,2) that must be purged once the doc shrinks to 1 chunk.
    seeded = {str(uuid.uuid5(ns, str(ci))) for ci in (1, 2)}
    data = _FakeCollectionData(seeded)
    client = _FakeWeaviateClient(data)

    r = _WritableRetriever(client=client)
    # Short text -> exactly 1 chunk with the default splitter settings.
    r.write(["hi"], [{"a": 1}], source_ids=[src])

    stale_id_1 = str(uuid.uuid5(ns, "1"))
    stale_id_2 = str(uuid.uuid5(ns, "2"))
    stale_id_3 = str(uuid.uuid5(ns, "3"))
    # Both seeded stale ids were deleted, and the loop stopped at the first
    # id that did not exist (ci=3), without deleting it.
    assert data.deleted == [stale_id_1, stale_id_2, stale_id_3]
    assert not data._seeded  # both stale entries consumed


def test_weaviate_write_rejects_duplicate_source_ids(monkeypatch):
    monkeypatch.setattr("langchain_weaviate.WeaviateVectorStore", _FakeVS)

    r = _WritableRetriever(client=object())
    with pytest.raises(ValueError, match="duplicate source_id"):
        r.write(["a", "b"], source_ids=["dup", "dup"])


def test_weaviate_write_empty_source_ids_are_not_duplicates(monkeypatch):
    # "" means "no source" in the namespace-assignment path (`if src:`), so
    # the duplicate check must use the same predicate and let ["", ""] pass.
    monkeypatch.setattr("langchain_weaviate.WeaviateVectorStore", _FakeVS)

    r = _WritableRetriever(client=object())
    ids = r.write(["a", "b"], source_ids=["", ""])
    assert ids == ["id-0", "id-1"]
    # No deterministic ids were assigned (no source), so no purge either.
    assert _FakeVS.last.added[0][2] is None


def test_weaviate_write_survives_purge_failure(monkeypatch):
    # add_texts has already committed when the purge runs; a raising
    # delete_by_id (transient connection/query error — distinct from the
    # boolean not-found return) must NOT turn the successful write into a
    # hard failure: write() logs a warning and returns the new ids.
    monkeypatch.setattr("langchain_weaviate.WeaviateVectorStore", _FakeVS)

    class _RaisingData:
        def delete_by_id(self, id_):
            raise RuntimeError("weaviate connection dropped")

    client = _FakeWeaviateClient(_RaisingData())
    r = _WritableRetriever(client=client)
    ids = r.write(["hi"], source_ids=["doc1"])  # must not raise
    assert ids == ["id-0"]


# ─────────────────────── R4: ragie computed fields win ─────────────────────

class _Chunk:
    def __init__(self, text, did, score, name=None, meta=None, dmeta=None):
        self.text = text
        self.document_id = did
        self.score = score
        self.document_name = name
        self.metadata = meta or {}
        self.document_metadata = dmeta or {}


class _Resp:
    def __init__(self, chunks):
        self.scored_chunks = chunks


class _Retrievals:
    def __init__(self, resp):
        self._resp = resp

    def retrieve(self, *, request):
        return self._resp


class _FakeRagie:
    def __init__(self, resp):
        self.retrievals = _Retrievals(resp)


class _RagLibRetriever(RagieRetriever):
    partition = "p1"


def test_ragie_computed_fields_win_over_user_metadata():
    # A malicious/careless caller stashed a fake "score" in document_metadata;
    # the real, numeric chunk.score must win in the returned metadata.
    chunk = _Chunk("hello", "d1", 0.42, dmeta={"score": "A+", "document_id": "spoofed"})
    fake = _FakeRagie(_Resp([chunk]))
    docs = _RagLibRetriever(client=fake).search("q")
    assert docs[0]["metadata"]["score"] == 0.42
    assert docs[0]["metadata"]["document_id"] == "d1"


# ─────────────────── R5: metadata_fields default not a shared list ─────────

def test_weaviate_metadata_fields_default_not_shared():
    class _R1Retriever(WeaviateRetriever):
        collection_name = "C1"
        embedding = _FakeEmbedding()

    class _R2Retriever(WeaviateRetriever):
        collection_name = "C2"
        embedding = _FakeEmbedding()

    # The default must be immutable (a tuple), so even sharing the same
    # object across subclasses can never leak a mutation between them.
    assert isinstance(_R1Retriever.metadata_fields, tuple)
    assert isinstance(_R2Retriever.metadata_fields, tuple)


# ───────────────────────── R6: Retriever.awrite ────────────────────────────

class _AllRetriever(Retriever):
    type_access = TypeAccess.ALL

    def search(self, query, *, k=None):
        return []

    def write(self, texts, metadatas=None):
        return [f"id-{i}" for i in range(len(texts))]


def test_retriever_awrite_bridges_sync_write():
    ids = asyncio.run(_AllRetriever().awrite(["t"]))
    assert ids == ["id-0"]


# ───────────────────────── C1: pooled async client ─────────────────────────

class _ApiConnector(Connector):
    base_url_env = "X_URL"
    auth_token_env = "X_TOKEN"


def _fake_async_client_cls(created: dict, closed: dict):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    class _Client:
        def __init__(self, *a, **k):
            created["n"] += 1
            self.is_closed = False

        async def get(self, url, **kwargs):
            return _Resp()

        async def post(self, url, json=None, **kwargs):
            return _Resp()

        async def aclose(self):
            closed["n"] += 1
            self.is_closed = True

    return _Client


def test_connector_async_client_is_pooled(monkeypatch):
    created = {"n": 0}
    closed = {"n": 0}
    monkeypatch.setattr(httpx, "AsyncClient", _fake_async_client_cls(created, closed))

    conn = _ApiConnector(base_url="http://x.com")
    asyncio.run(conn.aget("/a"))
    asyncio.run(conn.aget("/b"))
    assert created["n"] == 1  # one instance reused across both calls

    asyncio.run(conn.aclose())
    assert closed["n"] == 1
    assert conn._aclient is None


# ──────────────────── C2: headers/timeout kwargs merge ─────────────────────

def test_connector_get_allows_header_and_timeout_override(monkeypatch):
    captured: dict = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    def mock_get(url, **kwargs):
        captured.update(kwargs)
        return _Resp()

    monkeypatch.setattr(httpx, "get", mock_get)

    conn = _ApiConnector(base_url="http://x.com", auth_token="tok")
    result = conn.get("/p", headers={"X-Extra": "1"}, timeout=5)

    assert result == {"ok": True}
    assert captured["headers"]["X-Extra"] == "1"
    assert captured["headers"]["Authorization"] == "Bearer tok"
    assert captured["timeout"] == 5
