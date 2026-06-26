# tests/test_retriever_ragie.py
"""RagieRetriever: scored_chunks -> list[dict], sync/async via o mesmo client,
write via create_raw. Client injetado — sem rede, sem SDK Ragie."""
from __future__ import annotations

import asyncio

import pytest

from aixon.exceptions import AixonError
from aixon.retriever import TypeAccess
from aixon.retrievers.ragie import RagieRetriever


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
        self.calls = []

    def retrieve(self, *, request):
        self.calls.append(request)
        return self._resp

    async def retrieve_async(self, *, request):
        self.calls.append(request)
        return self._resp


class _Docs:
    def __init__(self):
        self.created = []

    def create_raw(self, *, request):
        self.created.append(request)
        return type("D", (), {"id": f"doc-{len(self.created)}"})()


class _FakeRagie:
    def __init__(self, resp):
        self.retrievals = _Retrievals(resp)
        self.documents = _Docs()


def _resp():
    return _Resp([_Chunk("hello", "d1", 0.9, name="N",
                         meta={"page": 2}, dmeta={"src": "x"})])


class _LibRetriever(RagieRetriever):
    partition = "p1"


class _WritableRetriever(RagieRetriever):
    partition = "p1"
    type_access = TypeAccess.ALL


def test_search_maps_chunks_and_request():
    fake = _FakeRagie(_resp())
    docs = _LibRetriever(client=fake).search("q", k=7)
    assert docs[0]["text"] == "hello"
    assert docs[0]["metadata"]["document_id"] == "d1"
    assert docs[0]["metadata"]["score"] == 0.9
    assert docs[0]["metadata"]["src"] == "x"   # document_metadata merged
    assert docs[0]["metadata"]["page"] == 2    # chunk metadata merged
    assert fake.retrievals.calls[0] == {"query": "q", "top_k": 7, "partition": "p1"}


def test_asearch_uses_retrieve_async():
    fake = _FakeRagie(_resp())
    docs = asyncio.run(_LibRetriever(client=fake).asearch("q"))
    assert docs[0]["text"] == "hello"
    assert fake.retrievals.calls[0]["top_k"] == 5  # default max_query_results


def test_rerank_flag_in_request():
    class _RerankRetriever(RagieRetriever):
        partition = "p1"
        rerank = True

    fake = _FakeRagie(_resp())
    _RerankRetriever(client=fake).search("q")
    assert fake.retrievals.calls[0]["rerank"] is True


def test_read_only_write_raises():
    with pytest.raises(AixonError):
        _LibRetriever(client=_FakeRagie(_resp())).write(["x"])


def test_write_creates_raw_with_external_id():
    fake = _FakeRagie(_resp())
    ids = _WritableRetriever(client=fake).write(
        ["doc text"], [{"a": 1}], source_ids=["ext1"])
    assert ids == ["doc-1"]
    req = fake.documents.created[0]
    assert req["content"] == "doc text"
    assert req["partition"] == "p1"
    assert req["metadata"] == {"a": 1}
    assert req["external_id"] == "ext1"


def test_missing_partition_raises(monkeypatch):
    monkeypatch.setenv("RAGIE_API_KEY", "k")
    with pytest.raises(AixonError):
        RagieRetriever()


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("RAGIE_API_KEY", raising=False)

    class _NoKeyRetriever(RagieRetriever):
        partition = "p"

    with pytest.raises(AixonError):
        _NoKeyRetriever()
