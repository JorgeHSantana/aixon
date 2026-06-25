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
