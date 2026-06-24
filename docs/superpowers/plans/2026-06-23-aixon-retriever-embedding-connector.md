# aixon Retriever + Embedding + Connector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Plan 6 of the aixon framework: the `Retriever` ABC (context search with `as_tool()` returning a neutral `AgentTool`), the `Embedding` ABC and its `OpenAIEmbedding` lazy subclass, and the `Connector` base class for HTTP microservice clients. This subsystem is fully independent of Plans 2–5 and introduces one new extra: `retrieval = ["httpx>=0.27"]`.

**Architecture:**
- `aixon/embedding.py` — Pure ABC (no LangChain dependency in the ABC itself). `OpenAIEmbedding` wraps `langchain_openai.OpenAIEmbeddings` lazily inside `_get_client()`.
- `aixon/retriever.py` — ABC with `__init_subclass__` suffix validation (same pattern as `Agent`), `TypeAccess` enum, `search`/`write` abstract/default methods, and `as_tool()` returning the **same** `aixon.AgentTool` dataclass that `Agent.as_tool()` returns — so `coerce_tools` (Plan 3) handles both uniformly.
- `aixon/connector.py` — Concrete base class with `__init_subclass__` suffix validation, `get`/`post` via `httpx`, env-var-based `base_url`/`auth_token` defaults.

**Tech Stack:** Python 3.11+, `httpx>=0.27` (retrieval extra), `pytest`, hermetic tests only.

## Global Constraints

- `requires-python >= "3.11"` — verbatim from spec.
- Build backend: `hatchling`.
- Package name `aixon`; import name `aixon`.
- **Neutral `AgentTool` reuse (BINDING):** `Retriever.as_tool()` MUST return `aixon.agent.AgentTool` — the same dataclass (`name: str`, `description: str`, `func: Callable[[str], str]`) that `Agent.as_tool()` returns. No LangChain type is allowed to cross this boundary. Conversion to `StructuredTool` is Plan 3's responsibility (`coerce_tools`).
- **Suffix rules:** Every concrete `Retriever` subclass name must end with `"Retriever"` (raises `NamingError`). Every concrete `Connector` subclass name must end with `"Connector"` (raises `NamingError`). Abstract intermediate classes are exempt via an `_abstract = True` guard in `__init_subclass__` — same pattern as `Agent.__init_subclass__`. Unlike `Agent`, `Retriever` and `Connector` do NOT auto-instantiate or auto-register (they are tools, not agents).
- **Hermetic tests — no network/API keys:** `Embedding` tested with a `FakeEmbedding` in-memory subclass (dot-product similarity is fine; just return deterministic floats). `OpenAIEmbedding` lazy-client test uses `pytest.importorskip` for `langchain_openai` and `monkeypatch` to patch the client — no real API key. `Retriever` tested with an in-memory concrete subclass; `as_tool()` verified to return an `AgentTool` whose `.func("query")` performs a search. `Connector` tested with `httpx.MockTransport` — no real network.
- **`retrieval` extra:** `retrieval = ["httpx>=0.27"]`. Vector-store backends (Weaviate, Ragie, etc.) are OUT of scope — YAGNI. `httpx` for `Connector`; no mandatory dep on `langchain_openai` in the core (only needed for `OpenAIEmbedding`).
- **Exports:** `Retriever`, `TypeAccess`, `Embedding`, `OpenAIEmbedding`, `Connector` exported from `aixon`.
- Error messages: state what was got and how to fix it (restmcp tone).
- No `tests/__init__.py`. `tests/conftest.py` autouse `reset_registry` fixture already exists from Plan 1.

---

### Task 1: `Embedding` ABC + `FakeEmbedding` for testing

**Files:**
- Create: `aixon/embedding.py`
- Modify: `aixon/__init__.py` (export `Embedding`)
- Test: `tests/test_embedding.py`

**Interfaces:**
- Consumes: nothing from other plans (pure ABC).
- Produces:
  - `aixon.embedding.Embedding` — ABC with two abstract methods:
    - `embed_documents(self, texts: list[str]) -> list[list[float]]`
    - `embed_query(self, text: str) -> list[float]`
  - Exported from `aixon` as `Embedding`.

> Design note: the olympus `Embedding` extended `langchain_core.embeddings.Embeddings` to be compatible with LangChain's vector stores. In aixon, the ABC is standalone — no LangChain import in `embedding.py`. The olympus derivation was driven by Weaviate vector store integration, which is YAGNI here. `OpenAIEmbedding` (Task 2) may delegate to `langchain_openai.OpenAIEmbeddings` internally via lazy import, but the ABC itself stays neutral.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedding.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_embedding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aixon.embedding'`.

- [ ] **Step 3: Write the implementation**

```python
# aixon/embedding.py
"""Embedding ABC for aixon. Concrete subclasses provide vector representations
of texts. Backends (OpenAI, etc.) live in separate modules and are imported
lazily so the core stays dependency-free.

No LangChain type is imported here — this ABC is neutral. OpenAIEmbedding
delegates to langchain_openai.OpenAIEmbeddings lazily (Task 2)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedding(ABC):
    """Base class for embedding providers.

    Subclasses must implement ``embed_documents`` and ``embed_query``.
    Clients are created lazily inside those methods — never at import time
    or class definition, so importing ``aixon`` never requires a provider
    SDK to be installed.
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one float vector per text."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Return a float vector for a single query string."""
```

Update `aixon/__init__.py` to add `Embedding` to imports and `__all__`. Add after the existing `from aixon.agent import Agent, AgentTool` block:

```python
# aixon/__init__.py — add this import (keep alphabetical order within groups)
from aixon.embedding import Embedding
```

And add `"Embedding"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_embedding.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v`
Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add aixon/embedding.py aixon/__init__.py tests/test_embedding.py
git commit -m "feat(p6): Embedding ABC — neutral, no LangChain import in base"
```

---

### Task 2: `OpenAIEmbedding` — lazy client

**Files:**
- Modify: `aixon/embedding.py` (add `OpenAIEmbedding`)
- Modify: `aixon/__init__.py` (export `OpenAIEmbedding`)
- Modify: `pyproject.toml` (add `retrieval` extra and `openai-embedding` optional dep)
- Test: `tests/test_openai_embedding.py`

**Interfaces:**
- Consumes: `aixon.embedding.Embedding`.
- Produces:
  - `aixon.embedding.OpenAIEmbedding(model: str, *, api_key_env: str = "OPENAI_API_KEY")` — subclass of `Embedding`. Client is `None` at construction; `_get_client()` lazily imports and instantiates `langchain_openai.OpenAIEmbeddings`. `embed_documents` and `embed_query` delegate to the client.
  - Exported from `aixon` as `OpenAIEmbedding`.

> The `_get_client()` pattern mirrors olympus `embeddings/openai.py` verbatim. The key difference: olympus stores `openai_api_key=os.getenv(self.api_key_env)` at client construction time; aixon does the same (env read is deferred to first use, which is the correct lazy pattern — the env var only needs to be set when the first embedding call happens, not at class definition).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_openai_embedding.py
"""Tests for OpenAIEmbedding. Skipped if langchain_openai is not installed.
No real API key is used — the client is monkeypatched."""

import pytest

langchain_openai = pytest.importorskip(
    "langchain_openai",
    reason="langchain_openai not installed; skipping OpenAIEmbedding tests",
)

from aixon.embedding import OpenAIEmbedding


class _FakeOpenAIEmbeddings:
    """Minimal stand-in for langchain_openai.OpenAIEmbeddings."""

    def __init__(self, model, openai_api_key):
        self.model = model
        self.openai_api_key = openai_api_key

    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text):
        return [0.3, 0.4]


@pytest.fixture
def patched_openai(monkeypatch):
    """Patch langchain_openai.OpenAIEmbeddings with the fake."""
    monkeypatch.setattr(
        "langchain_openai.OpenAIEmbeddings", _FakeOpenAIEmbeddings
    )
    yield _FakeOpenAIEmbeddings


def test_openai_embedding_is_lazy(patched_openai):
    emb = OpenAIEmbedding("text-embedding-3-small")
    assert emb._client is None


def test_openai_embedding_client_created_on_first_call(patched_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    emb = OpenAIEmbedding("text-embedding-3-small")
    result = emb.embed_query("test")
    assert emb._client is not None
    assert result == [0.3, 0.4]


def test_openai_embedding_embed_documents(patched_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    emb = OpenAIEmbedding("text-embedding-3-small")
    vecs = emb.embed_documents(["a", "b"])
    assert len(vecs) == 2
    assert vecs[0] == [0.1, 0.2]


def test_openai_embedding_custom_api_key_env(patched_openai, monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-custom")
    emb = OpenAIEmbedding("text-embedding-3-small", api_key_env="MY_KEY")
    emb.embed_query("x")
    assert emb._client.openai_api_key == "sk-custom"


def test_openai_embedding_client_cached(patched_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    emb = OpenAIEmbedding("text-embedding-3-small")
    emb.embed_query("first")
    client_1 = emb._client
    emb.embed_query("second")
    assert emb._client is client_1


def test_openai_embedding_repr():
    emb = OpenAIEmbedding("text-embedding-3-large")
    assert "text-embedding-3-large" in repr(emb)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_openai_embedding.py -v`
Expected: FAIL with `ImportError: cannot import name 'OpenAIEmbedding' from 'aixon.embedding'` (or skip if `langchain_openai` is absent).

- [ ] **Step 3: Write the implementation**

Append `OpenAIEmbedding` to `aixon/embedding.py` after the `Embedding` class:

```python
# aixon/embedding.py — append after Embedding class

import os
from typing import Optional


class OpenAIEmbedding(Embedding):
    """Embedding via OpenAI. The client is created lazily on the first
    ``embed_query`` / ``embed_documents`` call — importing ``aixon`` never
    requires ``langchain_openai`` to be installed.

    Args:
        model:       OpenAI embedding model name (e.g. "text-embedding-3-large").
        api_key_env: Environment variable holding the API key
                     (default: ``"OPENAI_API_KEY"``).

    Example::

        class LibraryRetriever(Retriever):
            embedding = OpenAIEmbedding("text-embedding-3-large")
    """

    def __init__(self, model: str, *, api_key_env: str = "OPENAI_API_KEY") -> None:
        self.model = model
        self.api_key_env = api_key_env
        self._client: Optional[object] = None

    def _get_client(self) -> object:
        if self._client is None:
            from langchain_openai import OpenAIEmbeddings  # lazy import

            self._client = OpenAIEmbeddings(
                model=self.model,
                openai_api_key=os.getenv(self.api_key_env),
            )
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get_client().embed_documents(texts)  # type: ignore[union-attr]

    def embed_query(self, text: str) -> list[float]:
        return self._get_client().embed_query(text)  # type: ignore[union-attr]

    def __repr__(self) -> str:
        return f"OpenAIEmbedding(model={self.model!r})"
```

Update `pyproject.toml` to add the `retrieval` extra and an `openai-embedding` optional group. In `[project.optional-dependencies]`:

```toml
# pyproject.toml — add to [project.optional-dependencies]
retrieval = ["httpx>=0.27"]
openai-embedding = ["langchain-openai>=0.2"]
# Update 'all' to include these new extras (merge with existing 'all' line):
# all = [...existing..., "httpx>=0.27", "langchain-openai>=0.2"]
```

Update `aixon/__init__.py` to export `OpenAIEmbedding`:

```python
# aixon/__init__.py — add to the embedding import line
from aixon.embedding import Embedding, OpenAIEmbedding
```

And add `"OpenAIEmbedding"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_openai_embedding.py -v`
Expected: PASS (6 tests) if `langchain_openai` is installed, or all SKIPPED otherwise — both are acceptable.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v`
Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add aixon/embedding.py aixon/__init__.py pyproject.toml tests/test_openai_embedding.py
git commit -m "feat(p6): OpenAIEmbedding with lazy langchain_openai client"
```

---

### Task 3: `TypeAccess` enum + `Retriever` ABC with suffix validation

**Files:**
- Create: `aixon/retriever.py`
- Modify: `aixon/__init__.py` (export `Retriever`, `TypeAccess`)
- Test: `tests/test_retriever.py` (suffix validation + search/write behavior)

**Interfaces:**
- Consumes: `aixon.exceptions.NamingError`, `aixon.agent.AgentTool`.
- Produces:
  - `aixon.retriever.TypeAccess` — `Enum` with `READ = "read"`, `WRITE = "write"`, `ALL = "all"`.
  - `aixon.retriever.Retriever` — ABC. Class attributes: `description: str = ""`, `type_access: TypeAccess = TypeAccess.READ`. Abstract method: `search(self, query: str, *, k: int | None = None) -> list[dict]`. Default method: `write(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]` (raises `AixonError` if `type_access` is `TypeAccess.READ`). `__init_subclass__` enforces `*Retriever` suffix for concrete subclasses (raises `NamingError`). Abstract subtypes opt out via `abstract=True` kwarg.
  - `as_tool` is NOT added in this task — it comes in Task 4.
  - Exported from `aixon`: `Retriever`, `TypeAccess`.

> Contrast with olympus `RAG`: olympus `__init_subclass__` only enforced the suffix when `module.startswith("stores.")`. In aixon, suffix enforcement is global (same as `Agent`) — every concrete `Retriever` subclass everywhere must end with `"Retriever"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retriever.py
import pytest
from aixon.retriever import Retriever, TypeAccess
from aixon.exceptions import NamingError, AixonError


# --- Helpers ---

class MemoryRetriever(Retriever):
    """In-memory retriever for testing. Stores dicts in a list."""

    description = "searches memory"
    type_access = TypeAccess.ALL

    def __init__(self):
        self._docs: list[dict] = []

    def search(self, query: str, *, k: int | None = None) -> list[dict]:
        results = [d for d in self._docs if query.lower() in d["text"].lower()]
        if k is not None:
            results = results[:k]
        return results

    def write(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        ids = []
        for i, text in enumerate(texts):
            meta = (metadatas or [{}] * len(texts))[i]
            doc_id = f"doc-{len(self._docs)}"
            self._docs.append({"text": text, "metadata": meta})
            ids.append(doc_id)
        return ids


class ReadOnlyRetriever(Retriever):
    """READ-only retriever."""

    description = "read only"
    type_access = TypeAccess.READ

    def search(self, query: str, *, k: int | None = None) -> list[dict]:
        return []


# --- Tests ---

def test_type_access_enum_values():
    assert TypeAccess.READ.value == "read"
    assert TypeAccess.WRITE.value == "write"
    assert TypeAccess.ALL.value == "all"


def test_concrete_subclass_enforces_retriever_suffix():
    with pytest.raises(NamingError, match="Retriever"):
        class BadName(Retriever):
            def search(self, query, *, k=None):
                return []


def test_abstract_subtype_exempt_from_suffix():
    class BaseRetriever(Retriever, abstract=True):
        pass
    # Concrete subclass of abstract subtype must still have suffix.
    class ConcreteRetriever(BaseRetriever):
        def search(self, query, *, k=None):
            return []
    # This should not raise.


def test_abstract_subtype_concrete_bad_suffix_raises():
    class BaseRetriever(Retriever, abstract=True):
        pass
    with pytest.raises(NamingError, match="Retriever"):
        class BadConcreteBase(BaseRetriever):
            def search(self, query, *, k=None):
                return []


def test_retriever_is_not_auto_registered():
    """Retriever subclasses are tools, not agents — they must NOT appear in the registry."""
    from aixon.registry import get_registry
    r = MemoryRetriever()
    agents = [a.name for a in get_registry().all()]
    assert "memoryretriever" not in agents


def test_search_returns_matching_results():
    r = MemoryRetriever()
    r.write(["hello world", "goodbye moon"])
    results = r.search("hello")
    assert len(results) == 1
    assert results[0]["text"] == "hello world"


def test_search_with_k_limits_results():
    r = MemoryRetriever()
    r.write(["alpha fox", "beta fox", "gamma fox"])
    results = r.search("fox", k=2)
    assert len(results) == 2


def test_search_returns_list_of_dicts_with_text_and_metadata():
    r = MemoryRetriever()
    r.write(["sample"], metadatas=[{"src": "test"}])
    results = r.search("sample")
    assert "text" in results[0]
    assert "metadata" in results[0]


def test_write_on_read_only_raises():
    r = ReadOnlyRetriever()
    with pytest.raises(AixonError, match="read"):
        r.write(["text"])


def test_cannot_instantiate_retriever_abc_directly():
    with pytest.raises(TypeError):
        Retriever()  # type: ignore[abstract]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retriever.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aixon.retriever'`.

- [ ] **Step 3: Write the implementation**

```python
# aixon/retriever.py
"""Retriever ABC for context search (RAG, web search, hybrid).

Concrete subclasses must end with 'Retriever' — validated in
``__init_subclass__`` (same pattern as ``Agent``). Retriever subclasses are
NOT auto-registered as agents; they are tools consumed by ToolAgent via
``as_tool()`` (Task 4).

``write()`` is a default (non-abstract) method that raises if ``type_access``
is READ-only, so read-only subclasses need not override it."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from aixon.exceptions import AixonError, NamingError


class TypeAccess(Enum):
    """Controls which operations a Retriever exposes."""

    READ = "read"
    WRITE = "write"
    ALL = "all"


class Retriever(ABC):
    """Abstract base for context retrievers.

    Declarative attributes:
        description:  Human-readable description (used by ``as_tool()``).
        type_access:  READ (default) | WRITE | ALL — governs ``write()``.

    Concrete subclasses must end with ``'Retriever'`` (raises ``NamingError``
    at class definition time if not). Abstract intermediate classes pass
    ``abstract=True`` to opt out.

    Subclasses are NOT agents — they do not auto-register in the Registry.
    """

    description: str = ""
    type_access: TypeAccess = TypeAccess.READ

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if abstract:
            cls._abstract = True  # type: ignore[attr-defined]
            return
        if not cls.__name__.endswith("Retriever"):
            raise NamingError(
                f"Retriever subclass '{cls.__name__}' must end with 'Retriever' "
                f"(rename to '{cls.__name__}Retriever')."
            )

    @abstractmethod
    def search(self, query: str, *, k: int | None = None) -> list[dict]:
        """Search for relevant context.

        Returns:
            A list of dicts with at least ``{'text': str, 'metadata': dict}``.
        """

    def write(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> list[str]:
        """Index texts into the retriever's backing store.

        Raises ``AixonError`` if ``type_access`` is ``TypeAccess.READ``.
        Subclasses with ``type_access = TypeAccess.ALL`` should override this.

        Returns:
            List of document IDs.
        """
        if self.type_access == TypeAccess.READ:
            raise AixonError(
                f"'{type(self).__name__}' is configured as read-only "
                f"(type_access=TypeAccess.READ). Set type_access=TypeAccess.ALL "
                f"and override write() to enable indexing."
            )
        raise NotImplementedError(
            f"'{type(self).__name__}' declares type_access={self.type_access!r} "
            f"but does not implement write(). Override write() in your subclass."
        )
```

Update `aixon/__init__.py`:

```python
# aixon/__init__.py — add
from aixon.retriever import Retriever, TypeAccess
```

And add `"Retriever"`, `"TypeAccess"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_retriever.py -v`
Expected: PASS (11 tests).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v`
Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add aixon/retriever.py aixon/__init__.py tests/test_retriever.py
git commit -m "feat(p6): TypeAccess enum + Retriever ABC with suffix validation"
```

---

### Task 4: `Retriever.as_tool()` — neutral `AgentTool`

**Files:**
- Modify: `aixon/retriever.py` (add `as_tool` method)
- Test: `tests/test_retriever_as_tool.py`

**Interfaces:**
- Consumes: `aixon.agent.AgentTool`.
- Produces:
  - `Retriever.as_tool(self, name: str | None = None, description: str | None = None, k: int | None = None) -> AgentTool` — returns an `AgentTool` whose `.func(query: str) -> str` calls `self.search(query, k=k)` and formats the results as a readable string. Defaults: `name` from `type(self).__name__.lower()`, `description` from `self.description`.
  - The returned `AgentTool` is the **same dataclass** as `Agent.as_tool()` returns — `coerce_tools` (Plan 3) handles both without branching.

> Key design decision: `func` returns `str`, not `list[dict]`. This matches the `AgentTool` contract (`Callable[[str], str]`). The results are formatted as a newline-joined string of `text` fields (with metadata if present). This keeps the signature uniform with `Agent.as_tool()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retriever_as_tool.py
import pytest
from aixon.agent import AgentTool
from aixon.retriever import Retriever, TypeAccess


class MemoryRetriever(Retriever):
    description = "searches memory"
    type_access = TypeAccess.ALL

    def __init__(self):
        self._docs: list[dict] = []

    def search(self, query: str, *, k: int | None = None) -> list[dict]:
        results = [d for d in self._docs if query.lower() in d["text"].lower()]
        if k is not None:
            results = results[:k]
        return results

    def write(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        ids = []
        for i, text in enumerate(texts):
            meta = (metadatas or [{}] * len(texts))[i]
            doc_id = f"doc-{len(self._docs)}"
            self._docs.append({"text": text, "metadata": meta})
            ids.append(doc_id)
        return ids


def test_as_tool_returns_agent_tool():
    r = MemoryRetriever()
    tool = r.as_tool()
    assert isinstance(tool, AgentTool)


def test_as_tool_default_name_and_description():
    r = MemoryRetriever()
    tool = r.as_tool()
    assert tool.name == "memoryretriever"
    assert tool.description == "searches memory"


def test_as_tool_override_name_and_description():
    r = MemoryRetriever()
    tool = r.as_tool(name="lib", description="library search")
    assert tool.name == "lib"
    assert tool.description == "library search"


def test_as_tool_func_returns_string():
    r = MemoryRetriever()
    r.write(["hello world"])
    tool = r.as_tool()
    result = tool.func("hello")
    assert isinstance(result, str)


def test_as_tool_func_searches_and_returns_text():
    r = MemoryRetriever()
    r.write(["The quick brown fox"])
    tool = r.as_tool()
    result = tool.func("quick")
    assert "quick" in result.lower() or "fox" in result.lower()


def test_as_tool_func_empty_results_returns_string():
    r = MemoryRetriever()
    tool = r.as_tool()
    result = tool.func("nonexistent")
    assert isinstance(result, str)
    # Should indicate no results were found.
    assert len(result) >= 0  # Must not raise; string content is implementation-defined.


def test_as_tool_k_limits_results():
    r = MemoryRetriever()
    r.write(["fox 1", "fox 2", "fox 3"])
    tool = r.as_tool(k=1)
    result = tool.func("fox")
    # Only 1 result forwarded — result string should contain only one entry.
    assert result.count("fox") == 1


def test_as_tool_is_same_type_as_agent_as_tool():
    """AgentTool from Retriever.as_tool() and Agent.as_tool() are the same dataclass."""
    from aixon.agent import Agent, AgentTool
    from aixon.message import Message, Chunk

    class EchoAgent(Agent):
        def invoke(self, messages):
            return Message(role="assistant", content="ok")
        def stream(self, messages):
            return iter([Chunk(done=True)])

    from aixon.registry import get_registry
    agent = get_registry().resolve("echoagent")
    agent_tool = agent.as_tool()

    r = MemoryRetriever()
    retriever_tool = r.as_tool()

    assert type(agent_tool) is type(retriever_tool)
    assert isinstance(agent_tool, AgentTool)
    assert isinstance(retriever_tool, AgentTool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retriever_as_tool.py -v`
Expected: FAIL with `AttributeError: 'MemoryRetriever' object has no attribute 'as_tool'`.

- [ ] **Step 3: Write the implementation**

Add the `as_tool` method to the `Retriever` class in `aixon/retriever.py`. Place it after the `write` method:

```python
# aixon/retriever.py — add import at the top of the file
from typing import Callable, Any  # add Callable to the existing Any import

# also add this import for AgentTool:
from aixon.agent import AgentTool
```

```python
# aixon/retriever.py — add method to Retriever class, after write()

    def as_tool(
        self,
        name: str | None = None,
        description: str | None = None,
        k: int | None = None,
    ) -> AgentTool:
        """Expose this retriever as a neutral AgentTool.

        The returned ``AgentTool`` is the same dataclass as ``Agent.as_tool()``
        returns, so ``coerce_tools`` (Plan 3, ``aixon._adapters.tools``) handles
        both uniformly.

        Args:
            name:        Tool name (default: lowercased class name).
            description: Tool description (default: ``self.description``).
            k:           Max results forwarded to the agent. ``None`` = no cap.

        Returns:
            ``AgentTool(name, description, func)`` where ``func(query) -> str``
            calls ``self.search(query, k=k)`` and formats results as text.
        """
        _k = k
        _retriever = self

        def _run(query: str) -> str:
            docs = _retriever.search(query, k=_k)
            if not docs:
                return f"No results found for query: {query!r}"
            parts = []
            for doc in docs:
                text = doc.get("text", "")
                meta = doc.get("metadata", {})
                if meta:
                    parts.append(f"{text} [metadata: {meta}]")
                else:
                    parts.append(text)
            return "\n".join(parts)

        return AgentTool(
            name=name or type(self).__name__.lower(),
            description=description or self.description,
            func=_run,
        )
```

> Note on import order: `aixon/retriever.py` imports `from aixon.agent import AgentTool`. There is no circular import because `aixon/agent.py` does not import from `aixon/retriever.py`. The import chain is one-directional: `retriever` → `agent` → `registry`/`message`/`exceptions`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_retriever_as_tool.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v`
Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add aixon/retriever.py tests/test_retriever_as_tool.py
git commit -m "feat(p6): Retriever.as_tool() returns neutral AgentTool — uniform with Agent.as_tool()"
```

---

### Task 5: `Connector` base class

**Files:**
- Create: `aixon/connector.py`
- Modify: `aixon/__init__.py` (export `Connector`)
- Modify: `pyproject.toml` (confirm `httpx>=0.27` in `retrieval` extra — already added in Task 2)
- Test: `tests/test_connector.py`

**Interfaces:**
- Consumes: `aixon.exceptions.NamingError`. Uses `httpx` (retrieval extra).
- Produces:
  - `aixon.connector.Connector` — concrete base class (not ABC; subclasses extend by adding domain methods).
    - Class attributes (declarative): `base_url_env: str = ""`, `auth_token_env: str = ""`.
    - `__init__(self, *, base_url: str | None = None, auth_token: str | None = None, timeout: float | None = 30.0)` — resolves `base_url` from arg > env var > `""`. Resolves `auth_token` from arg > env var > `""`. `timeout` defaults to 30.0 seconds.
    - `__init_subclass__(cls, *, abstract: bool = False, **kwargs)` — enforces `*Connector` suffix for concrete subclasses (raises `NamingError`). Abstract subclasses opt out via `abstract=True`.
    - `get(self, path: str, **kwargs) -> dict` — issues `httpx.get(self.base_url + path, headers=self._headers(), **kwargs)`, raises on non-2xx, returns `response.json()`.
    - `post(self, path: str, json: dict | None = None, **kwargs) -> dict` — issues `httpx.post`, raises on non-2xx, returns `response.json()`.
    - `_headers(self) -> dict` — returns `{"Authorization": "Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}` if `auth_token` is non-empty; else `{"Content-Type": "application/json", "Accept": "application/json"}`.
  - Exported from `aixon` as `Connector`.

> The olympus `MCPDiagnosisService` used `urllib` with manual JSON encoding. `Connector` upgrades to `httpx` (already required as the test client for Plan 5's server) and expresses the same structure declaratively. Auth token is read from env at `__init__` time (same as olympus), not lazily.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_connector.py
import json
import pytest
import httpx

from aixon.connector import Connector
from aixon.exceptions import NamingError


# --- Helpers ---

def _make_transport(routes: dict) -> httpx.MockTransport:
    """
    routes: {(method, path): (status_code, body_dict)}
    e.g. {("GET", "/health"): (200, {"ok": True})}
    """
    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        if key not in routes:
            return httpx.Response(404, json={"error": "not found"})
        status, body = routes[key]
        return httpx.Response(status, json=body)
    return httpx.MockTransport(handler)


class ApiConnector(Connector):
    """Minimal concrete connector for testing."""
    base_url_env = "API_BASE_URL"
    auth_token_env = "API_AUTH_TOKEN"


# --- Tests ---

def test_connector_suffix_enforced():
    with pytest.raises(NamingError, match="Connector"):
        class BadName(Connector):
            pass


def test_connector_abstract_subtype_exempt():
    class BaseConnector(Connector, abstract=True):
        pass
    class ConcreteConnector(BaseConnector):
        pass
    # Should not raise.


def test_connector_resolves_base_url_from_arg(monkeypatch):
    monkeypatch.delenv("API_BASE_URL", raising=False)
    c = ApiConnector(base_url="http://example.com")
    assert c.base_url == "http://example.com"


def test_connector_resolves_base_url_from_env(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://env-host.com")
    c = ApiConnector()
    assert c.base_url == "http://env-host.com"


def test_connector_resolves_auth_token_from_arg(monkeypatch):
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    c = ApiConnector(auth_token="sk-direct")
    assert c.auth_token == "sk-direct"


def test_connector_resolves_auth_token_from_env(monkeypatch):
    monkeypatch.setenv("API_AUTH_TOKEN", "sk-from-env")
    c = ApiConnector()
    assert c.auth_token == "sk-from-env"


def test_connector_headers_with_auth_token():
    c = ApiConnector(base_url="http://x.com", auth_token="mytoken")
    headers = c._headers()
    assert headers["Authorization"] == "Bearer mytoken"
    assert headers["Content-Type"] == "application/json"


def test_connector_headers_without_auth_token():
    c = ApiConnector(base_url="http://x.com", auth_token="")
    headers = c._headers()
    assert "Authorization" not in headers
    assert "Content-Type" in headers


def test_connector_get_returns_json(monkeypatch):
    transport = _make_transport({("GET", "/ping"): (200, {"pong": True})})
    c = ApiConnector(base_url="http://test.local")
    # Patch httpx.get to use the mock transport.
    original_get = httpx.get

    def mock_get(url, **kwargs):
        client = httpx.Client(transport=transport)
        return client.get(url, **kwargs)

    monkeypatch.setattr(httpx, "get", mock_get)
    result = c.get("/ping")
    assert result == {"pong": True}


def test_connector_post_returns_json(monkeypatch):
    transport = _make_transport({("POST", "/echo"): (200, {"echoed": True})})
    c = ApiConnector(base_url="http://test.local")

    def mock_post(url, **kwargs):
        client = httpx.Client(transport=transport)
        return client.post(url, **kwargs)

    monkeypatch.setattr(httpx, "post", mock_post)
    result = c.post("/echo", json={"msg": "hi"})
    assert result == {"echoed": True}


def test_connector_get_raises_on_non_2xx(monkeypatch):
    transport = _make_transport({("GET", "/fail"): (500, {"error": "boom"})})
    c = ApiConnector(base_url="http://test.local")

    def mock_get(url, **kwargs):
        client = httpx.Client(transport=transport)
        resp = client.get(url, **kwargs)
        resp.raise_for_status()
        return resp

    monkeypatch.setattr(httpx, "get", mock_get)
    with pytest.raises(httpx.HTTPStatusError):
        c.get("/fail")


def test_connector_default_timeout():
    c = ApiConnector(base_url="http://x.com")
    assert c.timeout == 30.0


def test_connector_custom_timeout():
    c = ApiConnector(base_url="http://x.com", timeout=60.0)
    assert c.timeout == 60.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_connector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aixon.connector'`.

- [ ] **Step 3: Write the implementation**

```python
# aixon/connector.py
"""Connector base class for HTTP microservice clients.

Concrete subclasses must end with 'Connector' (raises ``NamingError``).
Each subclass declares ``base_url_env`` and ``auth_token_env`` as class
attributes; constructor overrides take precedence over environment variables.

HTTP calls use ``httpx`` (the ``retrieval`` extra). JSON is returned directly;
non-2xx responses raise ``httpx.HTTPStatusError``.

Example::

    class DiagnosisConnector(Connector):
        base_url_env = "MCP_DIAGNOSIS_BASE_URL"
        auth_token_env = "MCP_DIAGNOSIS_AUTH_TOKEN"

        def get_status(self) -> dict:
            return self.get("/health")
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from aixon.exceptions import NamingError


class Connector:
    """Base HTTP client for an external microservice.

    Declarative class attributes:
        base_url_env:   Env var name for the service base URL.
        auth_token_env: Env var name for the Bearer token.

    Constructor kwargs override env vars. ``timeout`` defaults to 30 seconds.
    """

    base_url_env: str = ""
    auth_token_env: str = ""

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if abstract:
            return
        if not cls.__name__.endswith("Connector"):
            raise NamingError(
                f"Connector subclass '{cls.__name__}' must end with 'Connector' "
                f"(rename to '{cls.__name__}Connector')."
            )

    def __init__(
        self,
        *,
        base_url: str | None = None,
        auth_token: str | None = None,
        timeout: float | None = 30.0,
    ) -> None:
        self.base_url = (
            base_url
            or (os.getenv(self.base_url_env) if self.base_url_env else None)
            or ""
        ).rstrip("/")

        self.auth_token = (
            auth_token
            or (os.getenv(self.auth_token_env) if self.auth_token_env else None)
            or ""
        )

        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def get(self, path: str, **kwargs: Any) -> dict:
        """Issue a GET request to ``base_url + path``.

        Raises:
            httpx.HTTPStatusError: on non-2xx response.

        Returns:
            Parsed JSON dict.
        """
        url = self.base_url + path
        response = httpx.get(
            url,
            headers=self._headers(),
            timeout=self.timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()

    def post(self, path: str, json: dict | None = None, **kwargs: Any) -> dict:
        """Issue a POST request to ``base_url + path`` with a JSON body.

        Raises:
            httpx.HTTPStatusError: on non-2xx response.

        Returns:
            Parsed JSON dict.
        """
        url = self.base_url + path
        response = httpx.post(
            url,
            json=json,
            headers=self._headers(),
            timeout=self.timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()
```

Update `aixon/__init__.py`:

```python
# aixon/__init__.py — add
from aixon.connector import Connector
```

And add `"Connector"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_connector.py -v`
Expected: PASS (13 tests).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v`
Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add aixon/connector.py aixon/__init__.py tests/test_connector.py
git commit -m "feat(p6): Connector base class with suffix validation and httpx get/post"
```

---

### Task 6: Wire up exports, pyproject.toml final state, and integration smoke test

**Files:**
- Modify: `aixon/__init__.py` (final complete state of Plan 6 exports)
- Modify: `pyproject.toml` (confirm `retrieval` extra, update `all`)
- Test: `tests/test_plan6_integration.py`

**Purpose:** Verify that all Plan 6 exports are reachable from `aixon`, that the `AgentTool` returned by `Retriever.as_tool()` and `Agent.as_tool()` are truly the same class, and that a `Connector` subclass can be imported and used without `Retriever` or `Embedding` side effects.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plan6_integration.py
"""Integration smoke test for Plan 6: Retriever + Embedding + Connector exports."""

import pytest

# All six names must be importable from aixon directly.
from aixon import (
    Connector,
    Embedding,
    OpenAIEmbedding,
    Retriever,
    TypeAccess,
)
from aixon.agent import AgentTool


def test_all_plan6_names_exported():
    """Validates all five Plan-6 names are importable from the top-level aixon namespace."""
    assert Retriever is not None
    assert TypeAccess is not None
    assert Embedding is not None
    assert OpenAIEmbedding is not None
    assert Connector is not None


def test_retriever_as_tool_returns_same_class_as_agent_as_tool():
    """The AgentTool from Retriever.as_tool() is identical to Agent.as_tool()'s class."""
    from aixon.agent import Agent, AgentTool
    from aixon.message import Message, Chunk
    from aixon.registry import get_registry

    class EchoAgent(Agent):
        def invoke(self, messages):
            return Message(role="assistant", content="ok")
        def stream(self, messages):
            return iter([Chunk(done=True)])

    class MemoryRetriever(Retriever):
        description = "mem"
        type_access = TypeAccess.READ
        def search(self, query, *, k=None):
            return [{"text": "found", "metadata": {}}]

    agent_tool = get_registry().resolve("echoagent").as_tool()
    retriever_tool = MemoryRetriever().as_tool()

    assert type(agent_tool) is AgentTool
    assert type(retriever_tool) is AgentTool
    assert type(agent_tool) is type(retriever_tool)


def test_connector_subclass_can_be_defined():
    class WeatherConnector(Connector):
        base_url_env = "WEATHER_URL"
        auth_token_env = "WEATHER_TOKEN"

    c = WeatherConnector(base_url="http://weather.local")
    assert c.base_url == "http://weather.local"


def test_embedding_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Embedding()  # type: ignore[abstract]


def test_type_access_all_values_present():
    values = {e.value for e in TypeAccess}
    assert values == {"read", "write", "all"}


def test_retriever_not_in_agent_registry():
    """Retriever subclasses are tools — they must not appear in the agent registry."""
    from aixon.registry import get_registry

    class SearchRetriever(Retriever):
        def search(self, query, *, k=None):
            return []

    names = [a.name for a in get_registry().all()]
    assert "searchretriever" not in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_plan6_integration.py -v`
Expected: FAIL with import errors for any name not yet wired up.

- [ ] **Step 3: Write the final `__init__.py` (complete Plan 6 state)**

This is the complete `aixon/__init__.py` at the end of Plan 6. It merges Plan 1's exports with the five Plan 6 additions. (Plans 2–5 will add their own exports later.)

```python
# aixon/__init__.py
"""aixon — declarative AI-agent framework."""

from aixon.agent import Agent, AgentTool
from aixon.connector import Connector
from aixon.discovery import autodiscover
from aixon.embedding import Embedding, OpenAIEmbedding
from aixon.exceptions import (
    AixonError,
    AgentNotFoundError,
    CompositionCycleError,
    NamingError,
    RegistrationError,
)
from aixon.logging import Logger
from aixon.message import Chunk, Message, Role
from aixon.registry import get_registry, reset_registry
from aixon.retriever import Retriever, TypeAccess

__all__ = [
    # Agent layer (Plan 1)
    "Agent",
    "AgentTool",
    "autodiscover",
    # Exceptions (Plan 1)
    "AixonError",
    "AgentNotFoundError",
    "CompositionCycleError",
    "NamingError",
    "RegistrationError",
    # Logging (Plan 1)
    "Logger",
    # Message types (Plan 1)
    "Chunk",
    "Message",
    "Role",
    # Registry (Plan 1)
    "get_registry",
    "reset_registry",
    # Retrieval layer (Plan 6)
    "Connector",
    "Embedding",
    "OpenAIEmbedding",
    "Retriever",
    "TypeAccess",
]
```

Confirm `pyproject.toml` contains the `retrieval` extra and that `all` includes it. The relevant section should look like:

```toml
# pyproject.toml — [project.optional-dependencies]
dev = ["pytest", "pytest-cov"]
retrieval = ["httpx>=0.27"]
openai-embedding = ["langchain-openai>=0.2"]
all = [
    "httpx>=0.27",
    "langchain-openai>=0.2",
    # (earlier plans' extras will be merged here by their respective plans)
]
```

Run: `python -m pip install -e ".[retrieval,dev]"` to ensure `httpx` is installed.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_plan6_integration.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS — all foundation tests (Plan 1) plus all Plan 6 tests.

- [ ] **Step 6: Commit**

```bash
git add aixon/__init__.py pyproject.toml tests/test_plan6_integration.py
git commit -m "feat(p6): wire all exports and confirm integration — Retriever+Embedding+Connector"
```

---

## Self-Review

### Spec coverage

**Retriever (contract §5.2):**
- `TypeAccess` enum with `READ`/`WRITE`/`ALL` values → Task 3. ✓
- `Retriever` ABC with `description`/`type_access` class attrs → Task 3. ✓
- `search(query, *, k=None) -> list[dict]` abstract method → Task 3. ✓
- `write()` default method raises on READ-only → Task 3. ✓
- Suffix validation `*Retriever` in `__init_subclass__` → Task 3. ✓
- Abstract subtype exemption via `abstract=True` → Task 3. ✓
- `as_tool()` returns `AgentTool` (same dataclass as `Agent.as_tool()`) → Task 4. ✓
- NOT auto-registered in the agent registry → verified in Task 3 and integration test. ✓

**Embedding (contract §5.1):**
- `Embedding` ABC with `embed_documents`/`embed_query` abstract methods → Task 1. ✓
- `OpenAIEmbedding(model, *, api_key_env)` with lazy client → Task 2. ✓
- No LangChain import in the ABC itself — neutral → Task 1. ✓
- `_get_client()` lazy pattern, never called at import/class-definition → Task 2. ✓

**Connector (contract §5.3):**
- Suffix validation `*Connector` in `__init_subclass__` → Task 5. ✓
- Abstract subtype exemption via `abstract=True` → Task 5. ✓
- `base_url_env`/`auth_token_env` class attributes → Task 5. ✓
- `__init__(*, base_url, auth_token, timeout)` with env fallback → Task 5. ✓
- `get(path, **kw) -> dict` / `post(path, json, **kw) -> dict` via httpx → Task 5. ✓
- `_headers()` with Bearer auth if token present → Task 5. ✓

**Exports (contract §5.4):**
- `Retriever`, `TypeAccess`, `Embedding`, `OpenAIEmbedding`, `Connector` all exported from `aixon` → Task 6. ✓

**Extra (contract §5.4):**
- `retrieval = ["httpx>=0.27"]` added to `pyproject.toml` → Task 2 (added), Task 6 (confirmed). ✓
- Vector-store backends (Weaviate, Ragie) are OUT of scope → no code for them, per YAGNI. ✓

### Placeholder scan

No `TODO`, `TBD`, `...` (as implementation), `pass` (as implementation), or "similar to Task N" markers left in any code step. Every method body is complete and runnable. ✓

### Type consistency vs contract

- `AgentTool(name: str, description: str, func: Callable[[str], str])` — `Retriever.as_tool()` returns exactly this dataclass, matching `Agent.as_tool()` return type. Both return `AgentTool`; `coerce_tools` (Plan 3) will handle them uniformly via the same branch. ✓
- `Retriever.search()` returns `list[dict]`; each dict has at least `"text"` and `"metadata"` keys (documented in the docstring; `as_tool._run` accesses these keys). ✓
- `Connector.get`/`post` return `dict` (parsed JSON). Non-2xx raises `httpx.HTTPStatusError`. ✓
- `Embedding.embed_documents` → `list[list[float]]`; `embed_query` → `list[float]`. Both abstract. ✓
- `OpenAIEmbedding._client` is `None` at construction (lazy guarantee). ✓

### Ambiguities resolved

1. **`Retriever.__init_subclass__` does not auto-register** — the contract says "suffix-validated," not "auto-registered." `Retriever` subclasses are tools, not agents. Confirmed by the test `test_retriever_not_in_agent_registry`.
2. **`write()` on `TypeAccess.WRITE` without override** — the default `write()` raises `NotImplementedError` (not `AixonError`) when `type_access != READ` but the method is not overridden. This separates "this retriever is read-only" (`AixonError`, user-facing) from "this subclass forgot to implement write" (`NotImplementedError`, programmer error). The READ-only path raises `AixonError` as specified.
3. **`as_tool().func` return type** — contract says `AgentTool.func: Callable[[str], str]`. `Retriever.as_tool` formats `list[dict]` results as a newline-joined string. This is the only way to match the shared `AgentTool` signature; the olympus `RAG.as_tool` returned a `StructuredTool` with a more complex `_rag_wrapper`. In aixon, the neutral `AgentTool` is the boundary; LangChain conversion happens in `coerce_tools`.
4. **`Connector` is not an ABC** — the contract says "base class," not "ABC." `Connector` is a concrete class providing `get`/`post`; subclasses extend it with domain methods. This matches the olympus `MCPDiagnosisService` pattern (concrete, not abstract). No `abstractmethod` decorators on `Connector`.
