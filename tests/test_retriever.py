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
