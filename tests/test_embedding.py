import pytest
from aixon.embedding import Embedding


class FakeEmbedding(Embedding):
    """Deterministic in-memory embedding for testing.
    Maps each word to a fixed position in a 4-dim vector."""

    DIM = 4

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)

    def _encode(self, text: str) -> list[float]:
        # Stable hash-based encoding — always same float for same string.
        h = hash(text) % (10 ** 6)
        base = float(h) / (10 ** 6)
        return [base + i * 0.01 for i in range(self.DIM)]


def test_embed_query_returns_float_vector():
    emb = FakeEmbedding()
    vec = emb.embed_query("hello")
    assert isinstance(vec, list)
    assert all(isinstance(v, float) for v in vec)
    assert len(vec) == FakeEmbedding.DIM


def test_embed_documents_returns_list_of_vectors():
    emb = FakeEmbedding()
    vecs = emb.embed_documents(["foo", "bar", "baz"])
    assert len(vecs) == 3
    for vec in vecs:
        assert isinstance(vec, list)
        assert len(vec) == FakeEmbedding.DIM


def test_embed_query_is_consistent():
    emb = FakeEmbedding()
    assert emb.embed_query("hello") == emb.embed_query("hello")


def test_embed_documents_differs_for_different_texts():
    emb = FakeEmbedding()
    vecs = emb.embed_documents(["hello", "world"])
    assert vecs[0] != vecs[1]


def test_cannot_instantiate_embedding_abc_directly():
    with pytest.raises(TypeError):
        Embedding()  # type: ignore[abstract]
