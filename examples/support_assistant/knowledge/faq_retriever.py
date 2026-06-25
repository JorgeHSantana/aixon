"""KnowledgeRetriever — semantic FAQ search behind the neutral `Retriever` API.

Demonstrates:
* `Retriever` + `TypeAccess.READ` (read-only; `write()` raises by default).
* `Embedding`: the corpus and the query are embedded, then ranked by cosine
  similarity. Offline it uses `DemoEmbedding`; set ``OPENAI_API_KEY`` and it
  swaps to `OpenAIEmbedding` with no other change.
* `as_tool()`: the agents consume this as a tool — see ``agents/``.

The embeddings are computed lazily on the first search and cached, so importing
the module is cheap and needs no key.
"""

from __future__ import annotations

import math

from aixon import OpenAIEmbedding, Retriever, TypeAccess

from knowledge.corpus import FAQ
from llm_config import using_real_llm
from providers.demo import DemoEmbedding


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class KnowledgeRetriever(Retriever):
    """Searches the FAQ by embedding similarity. Read-only."""

    description = "Search Acme's help-center articles and FAQ."
    type_access = TypeAccess.READ

    def __init__(self) -> None:
        # OpenAIEmbedding when a key is present; DemoEmbedding offline.
        self.embedding = (
            OpenAIEmbedding("text-embedding-3-small")
            if using_real_llm()
            else DemoEmbedding()
        )
        self._docs = FAQ
        self._doc_vectors: list[list[float]] | None = None  # lazy

    def _vectors(self) -> list[list[float]]:
        if self._doc_vectors is None:
            self._doc_vectors = self.embedding.embed_documents(
                [d["text"] for d in self._docs]
            )
        return self._doc_vectors

    def search(self, query: str, *, k: int | None = None) -> list[dict]:
        qv = self.embedding.embed_query(query)
        scored = [
            (_cosine(qv, dv), doc)
            for dv, doc in zip(self._vectors(), self._docs)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = [doc for score, doc in scored if score > 0.0]
        if k is not None:
            top = top[:k]
        return top
