"""TavilyRetriever — web search como Retriever neutro do aixon.

Busca web pura (sem cache; write-through é domínio do consumidor). SDK Tavily
importado lazy; `pip install aixon[tavily]`. Read-only. Async real via
AsyncTavilyClient."""

from __future__ import annotations

import os
from typing import Any

from aixon.exceptions import AixonError
from aixon.retriever import Retriever, TypeAccess


class TavilyRetriever(Retriever):
    """Retriever de busca web via Tavily.

    Atributos declarativos (subclasse define):
        description:     descrição da tool.
        max_web_results: teto default de resultados (sobrescrevível por chamada via k).

    Conexão: `api_key` (arg) ou env TAVILY_API_KEY; `client`/`aclient` injetáveis
    (testes). Read-only — `write` herda o default read-only da ABC.
    """

    type_access: TypeAccess = TypeAccess.READ
    max_web_results: int = 5

    def __init__(self, *, api_key: str | None = None, k: int | None = None,
                 client: Any = None, aclient: Any = None) -> None:
        self._api_key = api_key or os.getenv("TAVILY_API_KEY")
        if client is None and aclient is None and not self._api_key:
            raise AixonError(
                "TavilyRetriever requires an API key (pass api_key= or set "
                "TAVILY_API_KEY)."
            )
        if k is not None:
            self.max_web_results = k
        self._client = client
        self._aclient = aclient

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from tavily import TavilyClient
            except ImportError as exc:
                raise ImportError(
                    "TavilyRetriever needs the Tavily SDK. Install with "
                    "'pip install aixon[tavily]'."
                ) from exc
            self._client = TavilyClient(api_key=self._api_key)
        return self._client

    def _get_aclient(self) -> Any:
        if self._aclient is None:
            try:
                from tavily import AsyncTavilyClient
            except ImportError as exc:
                raise ImportError(
                    "TavilyRetriever needs the Tavily SDK. Install with "
                    "'pip install aixon[tavily]'."
                ) from exc
            self._aclient = AsyncTavilyClient(api_key=self._api_key)
        return self._aclient

    @staticmethod
    def _to_docs(response: dict, query: str) -> list[dict]:
        docs: list[dict] = []
        answer = response.get("answer")
        if answer:
            docs.append({
                "text": f"Resumo AI: {answer}",
                "metadata": {"source": "tavily_answer", "query": query},
            })
        for res in response.get("results", []):
            content = res.get("content", "")
            if not content:
                continue
            url = res.get("url", "")
            title = res.get("title", "")
            docs.append({
                "text": f"Title: {title}\nURL: {url}\nContent: {content}",
                "metadata": {"source": "tavily", "url": url, "title": title,
                             "query": query},
            })
        return docs

    def search(self, query: str, *, k: int | None = None) -> list[dict]:
        response = self._get_client().search(
            query, search_depth="basic",
            max_results=k or self.max_web_results, include_answer=True,
        )
        return self._to_docs(response, query)

    async def asearch(self, query: str, *, k: int | None = None) -> list[dict]:
        response = await self._get_aclient().search(
            query, search_depth="basic",
            max_results=k or self.max_web_results, include_answer=True,
        )
        return self._to_docs(response, query)
