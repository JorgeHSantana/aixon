# Vendor Retrievers

`aixon` ships three vendor-backed `Retriever` bases. Each is generic (config by
class attributes / env), lazy (the vendor SDK is imported only on use), and
hidden behind an optional extra. Subclass one, set the declarative attributes,
and expose it as a tool with `as_tool()` — the same neutral `AgentTool` every
agent consumes.

| Retriever | Extra | Backend | Async |
|---|---|---|---|
| `TavilyRetriever` | `aixon[tavily]` | Tavily web search | true (`AsyncTavilyClient`) |
| `RagieRetriever` | `aixon[ragie]` | Ragie managed RAG | true (`retrieve_async`) |
| `WeaviateRetriever` | `aixon[weaviate]` (+ `aixon[rerank]`) | Weaviate vector store | thread-bridge |

Importing `aixon` (or any of these classes) never requires the vendor SDK;
instantiating/using without it raises a clear `ImportError` naming the extra.

## TavilyRetriever

```python
from aixon import TavilyRetriever

class WebRetriever(TavilyRetriever):
    description = "Search the web for fresh information."
    max_web_results = 5
```

Read-only web search. `api_key` arg or `TAVILY_API_KEY`. Returns one doc for the
AI answer (when present) plus one per result.

## RagieRetriever

```python
from aixon import RagieRetriever

class KbRetriever(RagieRetriever):
    description = "Search the managed knowledge base."
    partition = "knowledge-base"
```

Managed RAG — Ragie handles chunking/embedding/indexing. `api_key` arg or
`RAGIE_API_KEY`. Set `rerank = True` for Ragie's native reranking. Set
`type_access = TypeAccess.ALL` to enable `write()` (ingests via `create_raw`).

## WeaviateRetriever

```python
from aixon import OpenAIEmbedding, WeaviateRetriever
from aixon import TypeAccess

class LibraryRetriever(WeaviateRetriever):
    description = "Search the technical library."
    collection_name = "Library"
    embedding = OpenAIEmbedding("text-embedding-3-large")
    type_access = TypeAccess.READ
```

Vector store retrieval. The neutral `aixon.Embedding` is bridged to LangChain
internally. The connection is lazy (opened on first `search`/`write`), so
instantiation is safe at import/autodiscover time. Connection via `host`/`port`
args or `WEAVIATE_HOST`/`WEAVIATE_PORT`. Optional flashrank reranking:

```python
class DeepLibraryRetriever(WeaviateRetriever):
    description = "Library with reranking."
    collection_name = "Library"
    embedding = OpenAIEmbedding("text-embedding-3-large")
    rerank = True          # needs: pip install aixon[rerank]
    rerank_fetch_k = 25    # fetch wide
    rerank_top_k = 10      # keep top N after rerank
```

`write()` (when `type_access` allows) chunks via `RecursiveCharacterTextSplitter`
and supports deterministic IDs through `source_ids`.
