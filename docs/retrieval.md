# Retrieval and Connectors

---

## Retriever — context search

`Retriever` is the base class for context search — vector stores, web search,
hybrid retrievers, or any source that returns ranked text chunks.

```python
from aixon import Retriever, TypeAccess

class LibraryRetriever(Retriever):
    description = "Searches the internal knowledge base"
    type_access = TypeAccess.READ

    def search(self, query: str, *, k: int | None = None) -> list[dict]:
        # Return [{"text": str, "metadata": dict}, ...]
        results = self._vector_store.similarity_search(query, k=k or 5)
        return [{"text": r.page_content, "metadata": r.metadata} for r in results]
```

**Suffix rule:** every concrete `Retriever` subclass name must end with
`"Retriever"`. Violating it raises `NamingError` at import time.

**No auto-registration:** unlike `Agent` subclasses, `Retriever` subclasses are
not added to the agent Registry. They are tools consumed by agents via
`as_tool()`.

**Install:**

```bash
pip install 'aixon[retrieval]'
```

### Retriever API

```python
class Retriever(ABC):
    description: str = ""
    type_access: TypeAccess = TypeAccess.READ

    @abstractmethod
    def search(self, query: str, *, k: int | None = None) -> list[dict]:
        """Return [{"text": str, "metadata": dict}, ...]."""

    async def asearch(self, query: str, *, k: int | None = None) -> list[dict]:
        """Async search. Default bridges to search() in a thread; override for a
        native async SDK (true non-blocking I/O)."""

    def write(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        """Store documents. Raises if type_access is READ."""

    def as_tool(self, name=None, description=None, k=None) -> AgentTool:
        """Expose as a neutral AgentTool (same shape as Agent.as_tool())."""
```

**Sync and async.** `search` is required; `asearch` is optional — its default
runs `search` in a worker thread, so every retriever works on the async agent
path without change, and the event loop is never blocked. A vendor retriever
backed by an async SDK (Weaviate/Ragie/Tavily) **overrides `asearch`** for real
non-blocking I/O. `as_tool()` returns a **dual** `AgentTool` (sync `func` →
`search`, async `coroutine` → `asearch`), so the retriever tool runs on **both**
`invoke` and `ainvoke`; under `ainvoke` it awaits `asearch`.

### TypeAccess

```python
from aixon import TypeAccess

class TypeAccess(Enum):
    READ  = "read"   # search only; write() raises
    WRITE = "write"  # write only (indexing pipeline)
    ALL   = "all"    # both search and write
```

### Using a Retriever as a tool

`Retriever.as_tool()` returns the same `AgentTool` type that `Agent.as_tool()`
returns. `ToolAgent.tools` handles both uniformly:

```python
from aixon import LLM, ToolAgent

class ResearchAgent(ToolAgent):
    llm   = LLM("gpt-4o-mini")
    tools = [
        LibraryRetriever().as_tool(k=10),
        PlannerAgent().as_tool(),
    ]
```

When the retriever returns no results, the tool yields:
`"No results found for query: '<query>'"`.

### Writing to a Retriever

Set `type_access = TypeAccess.ALL` and override `write()`:

```python
from aixon import Retriever, TypeAccess

class IndexRetriever(Retriever):
    type_access = TypeAccess.ALL

    def search(self, query: str, *, k: int | None = None) -> list[dict]:
        ...

    def write(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        ids = self._vector_store.add_texts(texts, metadatas=metadatas or [{}] * len(texts))
        return ids
```

```python
retriever = IndexRetriever()
ids = retriever.write(
    texts=["The battery should be replaced every 2 years."],
    metadatas=[{"source": "manual", "page": 42}],
)
```

Calling `write()` on a `TypeAccess.READ` retriever raises `AixonError`.

---

## Embedding — text embeddings

`Embedding` is the abstract base for vector embedding providers. All network
calls are lazy — no connection is made until `embed_*` is first invoked.

```python
from aixon import Embedding

class Embedding(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...
```

### OpenAIEmbedding

`OpenAIEmbedding` is the built-in concrete implementation. It delegates to
`langchain_openai.OpenAIEmbeddings` lazily:

```python
class OpenAIEmbedding(Embedding):
    def __init__(self, model: str, *, api_key_env: str = "OPENAI_API_KEY"): ...
```

**Install:**

```bash
pip install 'aixon[openai-embedding]'
```

**Usage:**

```python
from aixon import OpenAIEmbedding

embedding = OpenAIEmbedding("text-embedding-3-small")
vectors   = embedding.embed_documents(["First document.", "Second document."])
query_vec = embedding.embed_query("What is the battery life?")
```

You can override which environment variable holds the API key:

```python
embedding = OpenAIEmbedding(
    "text-embedding-3-large",
    api_key_env="MY_OPENAI_KEY",
)
```

### Custom embedding

Subclass `Embedding` and implement both abstract methods:

```python
from aixon import Embedding

class LocalEmbedding(Embedding):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)

    def _encode(self, text: str) -> list[float]:
        ...
```

---

## Connector — external microservice client

`Connector` is a base HTTP client for calling external microservices. It reads
its base URL and auth token from environment variables and exposes typed `get`
and `post` methods.

```python
from aixon import Connector

class InventoryConnector(Connector):
    base_url_env   = "INVENTORY_API_URL"
    auth_token_env = "INVENTORY_API_KEY"

    def get_stock(self, product_id: str) -> dict:
        return self.get(f"/products/{product_id}/stock")

    def reserve(self, product_id: str, qty: int) -> dict:
        return self.post("/reservations", json={"product_id": product_id, "qty": qty})
```

**Async:** `Connector` also provides `aget`/`apost`, which use
`httpx.AsyncClient` so they don't block the event loop. Use them from an async
tool or an agent's `ainvoke` path:

```python
class InventoryConnector(Connector):
    base_url_env = "INVENTORY_URL"

    async def get_stock(self, product_id: str) -> dict:
        return await self.aget(f"/products/{product_id}/stock")
```

**Suffix rule:** concrete `Connector` subclasses must end with `"Connector"`.
Violating it raises `NamingError` at import time.

**Install:** `Connector` uses `httpx` for transport, included in the retrieval
extra:

```bash
pip install 'aixon[retrieval]'
```

The httpx import is lazy — defining a `Connector` subclass works on a bare
install; the error only surfaces when `get()` or `post()` is called.

### Connector API

```python
class Connector:
    base_url_env:   str = ""    # env var holding the service base URL
    auth_token_env: str = ""    # env var holding the auth token (Bearer)

    def __init__(self, *, base_url=None, auth_token=None, timeout=30.0): ...

    def get(self, path: str, **kw) -> dict: ...
    def post(self, path: str, json=None, **kw) -> dict: ...
```

`get` and `post` return parsed JSON dicts. Non-2xx responses raise
`httpx.HTTPStatusError`.

### Overriding base URL at runtime (useful for tests)

Constructor kwargs take precedence over environment variables:

```python
connector = InventoryConnector(
    base_url="http://localhost:8080",
    auth_token="test-token",
)
stock = connector.get_stock("prod-123")
```

### Using a Connector inside a ToolAgent

Pass the connector's methods directly as tools — they are plain callables and
are coerced to `StructuredTool` by `ToolAgent`:

```python
from aixon import LLM, ToolAgent

class InventoryAgent(ToolAgent):
    llm   = LLM("gpt-4o-mini")
    tools = [
        InventoryConnector().get_stock,
        InventoryConnector().reserve,
    ]
```

---

## Related

- [agents.md](agents.md) — `ToolAgent` and `Agent.as_tool()`
- [architecture.md](architecture.md) — system overview
