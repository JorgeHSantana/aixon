"""Tests for OpenAIEmbedding. Skipped if langchain_openai is not installed.
No real API key is used — the client is monkeypatched."""

import pytest

langchain_openai = pytest.importorskip(
    "langchain_openai",
    reason="langchain_openai not installed; skipping OpenAIEmbedding tests",
)

from aixon.embedding import OpenAIEmbedding


class _FakeOpenAIEmbeddings:
    """Minimal stand-in for langchain_openai.OpenAIEmbeddings."""

    def __init__(self, model, api_key):
        self.model = model
        self.openai_api_key = api_key

    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text):
        return [0.3, 0.4]


@pytest.fixture
def patched_openai(monkeypatch):
    """Patch langchain_openai.OpenAIEmbeddings with the fake."""
    monkeypatch.setattr(
        "langchain_openai.OpenAIEmbeddings", _FakeOpenAIEmbeddings
    )
    yield _FakeOpenAIEmbeddings


def test_openai_embedding_is_lazy(patched_openai):
    emb = OpenAIEmbedding("text-embedding-3-small")
    assert emb._client is None


def test_openai_embedding_client_created_on_first_call(patched_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    emb = OpenAIEmbedding("text-embedding-3-small")
    result = emb.embed_query("test")
    assert emb._client is not None
    assert result == [0.3, 0.4]


def test_openai_embedding_embed_documents(patched_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    emb = OpenAIEmbedding("text-embedding-3-small")
    vecs = emb.embed_documents(["a", "b"])
    assert len(vecs) == 2
    assert vecs[0] == [0.1, 0.2]


def test_openai_embedding_custom_api_key_env(patched_openai, monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-custom")
    emb = OpenAIEmbedding("text-embedding-3-small", api_key_env="MY_KEY")
    emb.embed_query("x")
    assert emb._client.openai_api_key.get_secret_value() == "sk-custom"


def test_openai_embedding_client_cached(patched_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    emb = OpenAIEmbedding("text-embedding-3-small")
    emb.embed_query("first")
    client_1 = emb._client
    emb.embed_query("second")
    assert emb._client is client_1


def test_openai_embedding_repr():
    emb = OpenAIEmbedding("text-embedding-3-large")
    assert "text-embedding-3-large" in repr(emb)
