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

    async def awrite(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        """Async write. Default bridges to write() in a thread; override for a
        native async SDK."""

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

**Async write.** `awrite(texts, metadatas=None)` mirrors `asearch`: the default
bridges the sync `write` to a worker thread (so every retriever gets a working
`awrite` for free), and a vendor retriever backed by an async SDK may override
it for true non-blocking indexing.

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

**Async:** `Connector` also provides `aget`/`apost`, which use a **pooled**
`httpx.AsyncClient` (kept alive across calls, for connection reuse) so they
don't block the event loop. Use them from an async tool or an agent's
`ainvoke` path:

```python
class InventoryConnector(Connector):
    base_url_env = "INVENTORY_URL"

    async def get_stock(self, product_id: str) -> dict:
        return await self.aget(f"/products/{product_id}/stock")
```

**Loop affinity.** `httpx.AsyncClient`'s connection pool is bound to the event
loop that created it. Caching one instance forever would eventually break
across separate `asyncio.run()` calls (each with its own loop) with an
"Event loop is closed" error, so the pooled client is rebuilt automatically
whenever the currently running loop differs from (or has closed since) the one
that built it — safe to call `aget`/`apost` from a fresh loop without managing
this yourself.

**Closing.** Call `await connector.aclose()` to close the pooled async client
explicitly (idempotent — safe to call even if no async call was ever made).
The sync `get`/`post` methods don't pool a client, so they need no closing.

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

## MCPConnector — MCP servers

`MCPConnector` plugs an [MCP](https://modelcontextprotocol.io) server into an
agent: the server publishes its tool catalog (`tools/list`) and `as_tools()`
turns every entry into a neutral `AgentTool` whose `args_schema` is the tool's
published JSON Schema — the LLM sees the server's own contract, with no
hand-written wrapper per tool.

It is the complement of `HttpToolConnector`, not its replacement:

| | Flow decided by | Use when |
|---|---|---|
| `HttpToolConnector` | **your code** — each typed method is a curated tool | you own the contract and want to shape each tool (signature, normalization, encoding) |
| `MCPConnector` | **the LLM** — the model works from the published catalog | plugging a server (often third-party) where writing method-per-tool makes no sense |

```python
from aixon import LLM, MCPConnector, ToolAgent

class MetabaseMCPConnector(MCPConnector):
    base_url_env   = "MCP_METABASE_URL"     # streamable-HTTP endpoint URL
    auth_token_env = "MCP_METABASE_TOKEN"   # optional Bearer token

class AnalystAgent(ToolAgent):
    llm   = LLM("gpt-4o-mini")
    tools = [MetabaseMCPConnector().toolset(exclude=["delete_card"])]
```

Transport is MCP **streamable HTTP** at `base_url` (the full endpoint URL,
e.g. `https://host/mcp`). Env-var resolution, constructor overrides and the
`*Connector` suffix rule are inherited from `Connector`.

**Install:** the `mcp` SDK ships behind the `mcp` extra — `pip install
"aixon[mcp]"`. The import is lazy: defining a subclass works on a bare
install; the error only surfaces when the server is actually contacted.

### `toolset()` vs `as_tools()` — where discovery runs

A `ToolAgent` subclass body executes at `autodiscover()` time (server boot).
`as_tools()` runs discovery (`tools/list`) **immediately** via `asyncio.run` —
put it in a class body and an unreachable MCP server kills the whole server's
boot, not just that one agent. Use **`toolset()`** in class bodies instead:
it does **no I/O at all** at construction, just records
`(connector, include, exclude)`. Discovery happens lazily the first time the
agent actually runs (`coerce_tools` expands it into real tools right before
the LangGraph agent is built), so a bad server only breaks requests to *that*
agent — never import, never boot. The resolved catalog is then cached on the
toolset for subsequent invokes.

`as_tools()`/`aas_tools()` stay the right tool for runtime/script code (e.g.
`examples/mcp_tools/main.py`) that wants the catalog immediately and isn't
sitting in a class body.

### MCPConnector / MCPToolset API

```python
class MCPConnector(Connector):
    def toolset(self, include=None, exclude=None) -> "MCPToolset": ...  # deferred, zero I/O — use in class bodies
    def as_tools(self, include=None, exclude=None) -> list[AgentTool]: ...       # eager — runtime/script code only
    async def aas_tools(self, include=None, exclude=None) -> list[AgentTool]: ...  # eager, async-safe counterpart
    def list_tools(self) -> list[dict]: ...          # cached per instance
    async def alist_tools(self) -> list[dict]: ...
    def call(self, name: str, **params) -> str: ...
    async def acall(self, name: str, **params) -> str: ...

class MCPToolset:                       # returned by MCPConnector.toolset()
    def resolve_tools(self) -> list[AgentTool]: ...   # lazy discovery, cached on success; called by coerce_tools
```

- `as_tools()`/`aas_tools()`/`toolset()`'s eventual discovery all share the
  same `include`/`exclude` contract: `include=...` raises `AixonError` for a
  name the server does not expose — a typo must not silently shrink an
  agent's toolbox; `exclude` ignores unknown names.
- Discovery (`list_tools`/`alist_tools`) is cached per instance (guarded by a
  `threading.Lock`, double-checked — concurrent first-use from multiple
  threads triggers exactly one discovery session, not one per thread);
  execution opens one fresh session per call (stateless — no event-loop
  affinity to manage).
- A tool result with `isError` raises `AixonError`; text content is joined for
  the LLM, with `structuredContent` as JSON fallback.
- The sync paths (`list_tools`/`call`/`as_tools`) run the async ones via
  `asyncio.run`, so they must **not** be called from a running event loop —
  use `alist_tools`/`acall`/`aas_tools` there (the async agent path already
  does). Calling `as_tools()` from a running event loop raises a clear
  `AixonError` (not a bare `RuntimeError` from `asyncio.run`) pointing at
  `aas_tools()`/`toolset()`.
- `MCPToolset.resolve_tools()` (what `coerce_tools` calls to expand a
  `toolset()` entry) works from both a sync call site and one made
  synchronously from inside a running event loop (the shape of
  `ToolAgent._build_agent`, which calls `coerce_tools` without awaiting, even
  on the async invoke path): with no running loop it uses `asyncio.run`
  directly; with one running, discovery runs on a worker thread via
  `concurrent.futures.ThreadPoolExecutor` (a bounded block on the calling
  thread — the same class of trade-off as any sync tool called from an async
  loop). A discovery failure (e.g. server unreachable) raises `AixonError`
  scoped to that one call and is **not** cached, so a later invoke can retry.
- **Known limitation (not addressed):** there is no session reuse across
  calls — every `list_tools`/`call` pays a full MCP handshake (transport
  connect + `initialize`). A chatty tool loop against the same server pays
  that cost per call.

A fully offline, runnable demo lives in
[examples/mcp_tools](../examples/mcp_tools/README.md).

---

## Related

- [agents.md](agents.md) — `ToolAgent` and `Agent.as_tool()`
- [architecture.md](architecture.md) — system overview
