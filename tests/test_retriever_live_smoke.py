# tests/test_retriever_live_smoke.py
"""Smoke live dos vendor retrievers. Opt-in: rode com RUN_LIVE_RETRIEVAL=1.
Usa chaves/URLs do ambiente. Tavily/Ragie por chave; Weaviate pula sem servidor."""
from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_RETRIEVAL") != "1",
    reason="smoke live desligado (defina RUN_LIVE_RETRIEVAL=1 para rodar)",
)


def test_tavily_search_live():
    if not os.getenv("TAVILY_API_KEY"):
        pytest.skip("TAVILY_API_KEY não definida")
    from aixon import TavilyRetriever

    class WebRetriever(TavilyRetriever):
        description = "web"

    docs = WebRetriever().search("what is weaviate", k=3)
    assert isinstance(docs, list) and docs and "text" in docs[0]


def test_ragie_search_live():
    if not os.getenv("RAGIE_API_KEY"):
        pytest.skip("RAGIE_API_KEY não definida")
    if not os.getenv("RAGIE_PARTITION"):
        pytest.skip("RAGIE_PARTITION não definida")
    from aixon import RagieRetriever

    class KbRetriever(RagieRetriever):
        description = "kb"
        partition = os.environ["RAGIE_PARTITION"]

    docs = asyncio.run(KbRetriever().asearch("test"))
    assert isinstance(docs, list)


def test_weaviate_search_live():
    if not os.getenv("WEAVIATE_HOST") and not os.getenv("WEAVIATE_PORT"):
        pytest.skip("Weaviate não configurado (WEAVIATE_HOST/PORT)")
    if not os.getenv("WEAVIATE_COLLECTION"):
        pytest.skip("WEAVIATE_COLLECTION não definida")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY não definida (embedding)")
    from aixon import OpenAIEmbedding, WeaviateRetriever

    class LibRetriever(WeaviateRetriever):
        description = "lib"
        collection_name = os.environ["WEAVIATE_COLLECTION"]
        embedding = OpenAIEmbedding("text-embedding-3-large")

    try:
        docs = LibRetriever().search("test", k=2)
    except Exception as exc:  # servidor indisponível -> não bloqueia
        pytest.skip(f"Weaviate indisponível: {exc}")
    assert isinstance(docs, list)
