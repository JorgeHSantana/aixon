"""Integration smoke test for Plan 7: Retriever + Embedding + Connector exports."""

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


def test_all_plan7_names_exported():
    """Validates all five Plan-7 names are importable from the top-level aixon namespace."""
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
