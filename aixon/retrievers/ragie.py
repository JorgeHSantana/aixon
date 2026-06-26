"""RagieRetriever — RAG gerenciado como Retriever neutro do aixon.

O Ragie cuida de chunking/embedding/indexação. SDK importado lazy;
`pip install aixon[ragie]`. O mesmo client expõe sync (`retrieve`) e async
(`retrieve_async`). Rerank nativo do Ragie via `rerank=True`."""

from __future__ import annotations

import os
from typing import Any

from aixon.exceptions import AixonError
from aixon.retriever import Retriever, TypeAccess


class RagieRetriever(Retriever):
    """Retriever gerenciado via Ragie.

    Atributos declarativos (subclasse define):
        partition:          partição Ragie (obrigatório).
        type_access:        READ (default) | ALL — governa write().
        max_query_results:  teto default (sobrescrevível por chamada via k).
        rerank:             liga o rerank nativo do Ragie.

    Conexão: `api_key` (arg) ou env RAGIE_API_KEY; `client` injetável (testes).
    """

    partition: str = ""
    type_access: TypeAccess = TypeAccess.READ
    max_query_results: int = 5
    rerank: bool = False

    def __init__(self, *, api_key: str | None = None, k: int | None = None,
                 client: Any = None) -> None:
        if not self.partition:
            raise AixonError(
                f"{type(self).__name__} requires a 'partition' class attribute."
            )
        self._api_key = api_key or os.getenv("RAGIE_API_KEY")
        if client is None and not self._api_key:
            raise AixonError(
                "RagieRetriever requires an API key (pass api_key= or set "
                "RAGIE_API_KEY)."
            )
        if k is not None:
            self.max_query_results = k
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from ragie import Ragie
            except ImportError as exc:
                raise ImportError(
                    "RagieRetriever needs the Ragie SDK. Install with "
                    "'pip install aixon[ragie]'."
                ) from exc
            self._client = Ragie(auth=self._api_key)
        return self._client

    def _request(self, query: str, k: int | None) -> dict:
        request: dict[str, Any] = {
            "query": query,
            "top_k": k or self.max_query_results,
            "partition": self.partition,
        }
        if self.rerank:
            request["rerank"] = True
        return request

    @staticmethod
    def _to_docs(response: Any) -> list[dict]:
        docs: list[dict] = []
        for chunk in response.scored_chunks:
            chunk_meta = getattr(chunk, "metadata", None) or {}
            metadata = {
                "document_id": chunk.document_id,
                "document_name": getattr(chunk, "document_name", None),
                "score": chunk.score,
                **(getattr(chunk, "document_metadata", None) or {}),
                **chunk_meta,
            }
            docs.append({"text": chunk.text, "metadata": metadata})
        return docs

    def search(self, query: str, *, k: int | None = None) -> list[dict]:
        response = self._get_client().retrievals.retrieve(
            request=self._request(query, k))
        return self._to_docs(response)

    async def asearch(self, query: str, *, k: int | None = None) -> list[dict]:
        response = await self._get_client().retrievals.retrieve_async(
            request=self._request(query, k))
        return self._to_docs(response)

    def write(self, texts: list[str], metadatas: list[dict] | None = None,
              source_ids: list[str] | None = None) -> list[str]:
        if self.type_access == TypeAccess.READ:
            return super().write(texts, metadatas)  # raises AixonError
        metadatas = metadatas or [{} for _ in texts]
        client = self._get_client()
        ids: list[str] = []
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            params: dict[str, Any] = {
                "content": text, "partition": self.partition, "metadata": meta,
            }
            if source_ids and i < len(source_ids):
                params["external_id"] = source_ids[i]
            doc = client.documents.create_raw(request=params)
            ids.append(doc.id)
        return ids
