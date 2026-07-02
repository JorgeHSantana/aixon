"""WeaviateRetriever — retrieval por vector store como Retriever neutro do aixon.

weaviate-client + langchain-weaviate + langchain-text-splitters importados lazy;
`pip install aixon[weaviate]`. Rerank flashrank opcional via `aixon[rerank]`. A
conexão é lazy (montada no primeiro search/write) — importar/instanciar nunca
abre socket (seguro pra autodiscover/offline). O embedding continua um
aixon.Embedding neutro, embrulhado no langchain via _LangchainEmbeddings."""

from __future__ import annotations

import os
import uuid
from typing import Any

from langchain_core.embeddings import Embeddings

from aixon.embedding import Embedding
from aixon.exceptions import AixonError
from aixon.retriever import Retriever, TypeAccess


class _LangchainEmbeddings(Embeddings):
    """Embrulha um aixon.Embedding na interface langchain_core.Embeddings (pro
    WeaviateVectorStore). langchain-core é dep core do aixon → import no topo."""

    def __init__(self, embedding: Embedding) -> None:
        self._e = embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._e.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._e.embed_query(text)


class WeaviateRetriever(Retriever):
    """Retriever por vector store via Weaviate (v4 + langchain-weaviate).

    Atributos declarativos (subclasse define):
        collection_name:  índice/coleção Weaviate (obrigatório).
        embedding:        aixon.Embedding (obrigatório).
        text_key:         campo de texto (default "content").
        metadata_fields:  atributos de metadata a retornar.
        max_query_results: teto default (sobrescrevível por chamada via k).
        chunk_size/chunk_overlap: chunking no write.
        type_access:      READ (default) | ALL — governa write().
        rerank:           liga rerank flashrank (extra aixon[rerank]).
        rerank_fetch_k/rerank_top_k: busca ampla e corte do rerank.

    Conexão por param/env (host/port; WEAVIATE_HOST/WEAVIATE_PORT) ou `client`
    injetável. A conexão é LAZY (primeiro search/write).
    """

    collection_name: str = ""
    embedding: Embedding | None = None
    text_key: str = "content"
    metadata_fields: list[str] = []
    max_query_results: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    type_access: TypeAccess = TypeAccess.READ
    rerank: bool = False
    rerank_fetch_k: int = 25
    rerank_top_k: int = 10

    def __init__(self, *, host: str | None = None, port: int | None = None,
                 skip_init_checks: bool = True, client: Any = None,
                 default_filter: Any = None, k: int | None = None) -> None:
        if not self.collection_name:
            raise AixonError(
                f"{type(self).__name__} requires a 'collection_name' class attr."
            )
        if self.embedding is None:
            raise AixonError(
                f"{type(self).__name__} requires an 'embedding' class attr."
            )
        self._host = host or os.getenv("WEAVIATE_HOST", "localhost")
        self._port = port if port is not None else int(
            os.getenv("WEAVIATE_PORT", "8080"))
        self._skip_init_checks = skip_init_checks
        self._default_filter = default_filter
        if k is not None:
            self.max_query_results = k
        self._client = client
        self._owns_client = client is None
        self._vectorstore: Any = None
        self._splitter: Any = None
        self._ranker: Any = None

    def _ensure(self) -> None:
        """Monta client + vectorstore + splitter uma vez (lazy)."""
        if self._vectorstore is not None:
            return
        if self._client is None:
            try:
                import weaviate
            except ImportError as exc:
                raise ImportError(
                    "WeaviateRetriever needs weaviate-client. Install with "
                    "'pip install aixon[weaviate]'."
                ) from exc
            self._client = weaviate.connect_to_local(
                host=self._host, port=self._port,
                skip_init_checks=self._skip_init_checks)
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_weaviate import WeaviateVectorStore
        except ImportError as exc:
            raise ImportError(
                "WeaviateRetriever needs langchain-weaviate and "
                "langchain-text-splitters. Install with "
                "'pip install aixon[weaviate]'."
            ) from exc
        self._vectorstore = WeaviateVectorStore(
            client=self._client, index_name=self.collection_name,
            text_key=self.text_key,
            embedding=_LangchainEmbeddings(self.embedding),
            attributes=self.metadata_fields)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def _search_kwargs(self, k: int | None, filters: Any) -> dict:
        kwargs: dict[str, Any] = {"k": k or self.max_query_results}
        combined = filters
        if self._default_filter is not None and filters is not None:
            combined = self._default_filter & filters
        elif self._default_filter is not None:
            combined = self._default_filter
        if combined is not None:
            kwargs["filters"] = combined
        return kwargs

    @staticmethod
    def _to_dict(doc: Any) -> dict:
        return {"text": doc.page_content, "metadata": doc.metadata}

    def search(self, query: str, *, k: int | None = None,
               filters: Any = None) -> list[dict]:
        self._ensure()
        if self.rerank:
            docs = self._vectorstore.similarity_search(
                query, **self._search_kwargs(self.rerank_fetch_k, filters))
            # Rerank fetches rerank_fetch_k candidates for quality, but the
            # result must respect the effective k (contract: k = max results).
            return self._rerank(query, docs)[: k or self.max_query_results]
        docs = self._vectorstore.similarity_search(
            query, **self._search_kwargs(k, filters))
        return [self._to_dict(d) for d in docs]

    def _rerank(self, query: str, docs: list) -> list[dict]:
        try:
            from flashrank import Ranker, RerankRequest
        except ImportError as exc:
            raise ImportError(
                "rerank=True needs flashrank. Install with "
                "'pip install aixon[rerank]'."
            ) from exc
        if not docs:
            return []
        if self._ranker is None:  # lazy, cached (ONNX model load is expensive)
            self._ranker = Ranker()
        ranker = self._ranker
        passages = [{"id": str(i), "text": d.page_content, "meta": d.metadata}
                    for i, d in enumerate(docs)]
        ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
        out: list[dict] = []
        for res in ranked[:self.rerank_top_k]:
            meta = dict(res["meta"])
            meta["_rerank_score"] = res["score"]
            out.append({"text": res["text"], "metadata": meta})
        return out

    def write(self, texts: list[str], metadatas: list[dict] | None = None,
              source_ids: list[str] | None = None) -> list[str]:
        if self.type_access == TypeAccess.READ:
            return super().write(texts, metadatas)  # raises AixonError
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError(
                f"metadatas length ({len(metadatas)}) must match texts length "
                f"({len(texts)}).")
        self._ensure()
        metadatas = metadatas or [{} for _ in texts]
        all_texts: list[str] = []
        all_metas: list[dict] = []
        all_ids: list[str | None] = []
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            chunks = self._splitter.create_documents([text], [meta])
            src = source_ids[i] if source_ids and i < len(source_ids) else None
            for ci, chunk in enumerate(chunks):
                all_texts.append(chunk.page_content)
                all_metas.append(chunk.metadata)
                if src:
                    try:
                        ns = uuid.UUID(src)
                    except ValueError:
                        ns = uuid.uuid5(uuid.NAMESPACE_DNS, src)
                    all_ids.append(str(uuid.uuid5(ns, str(ci))))
                else:
                    all_ids.append(None)
        if any(all_ids):
            return self._vectorstore.add_texts(
                texts=all_texts, metadatas=all_metas, ids=all_ids)
        return self._vectorstore.add_texts(texts=all_texts, metadatas=all_metas)

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None
        # Reset lazy state so the next search/write reconnects via _ensure()
        # instead of dying on a closed client.
        self._vectorstore = None
