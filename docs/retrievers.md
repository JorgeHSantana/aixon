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
instantiation is safe at import/autodiscover time — first init is
thread-safe (double-checked locking), so concurrent first callers race into
the lock but only one actually builds the client/vectorstore. Connection via
`host`/`port` args or `WEAVIATE_HOST`/`WEAVIATE_PORT`. Optional flashrank
reranking:

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
and supports deterministic IDs through `source_ids`:

```python
retriever.write(
    texts=["Updated manual page 1...", "Updated manual page 2..."],
    source_ids=["doc-42", "doc-43"],
)
```

**Upsert semantics.** Passing `source_ids` makes `write()` an upsert: chunk IDs
are derived deterministically from each `source_id` (a UUID namespace), so
re-writing the same `source_id` with fewer chunks than before **purges the
now-obsolete tail chunks** left over from the previous write — otherwise stale
chunks from a longer prior version would linger in the vector store
alongside the new content. The purge runs *after* the new content is
committed and is best-effort: a transient delete failure is logged, not
raised (the write already succeeded; worst case is a lingering stale tail
until the next rewrite). `source_ids` must be unique within a single
`write()` call — a duplicate raises `ValueError` before anything is written
(colliding IDs would otherwise silently overwrite each other's chunks).
