"""Guardrail: Plan 2's FakeChatModel (tests/_fakes.py, contract §9.1) drives the
langgraph-native langchain.agents.create_agent graph offline through a tool call
then a final answer. Mirrors the validated probe. This is the linchpin pattern
the ToolAgent tests rely on."""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from tests._fakes import FakeChatModel


@tool
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"sunny in {city}"


def test_fake_chat_model_drives_create_agent_through_tool_then_answer():
    fake = FakeChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_weather", "args": {"city": "Recife"}, "id": "call_1"}
                ],
            ),
            AIMessage(content="The weather in Recife is sunny."),
        ]
    )
    agent = create_agent(fake, [get_weather], system_prompt="You are helpful.")
    result = agent.invoke({"messages": [("user", "weather in Recife?")]})

    final = result["messages"][-1]
    assert "sunny" in final.content.lower()
    # The graph really ran a tool step: Human, AI(tool_calls), Tool, AI(final).
    types = [type(m).__name__ for m in result["messages"]]
    assert "ToolMessage" in types
    assert types[-1] == "AIMessage"


def test_fake_chat_model_streams_updates_offline():
    fake = FakeChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_weather", "args": {"city": "Recife"}, "id": "c1"}
                ],
            ),
            AIMessage(content="It is sunny in Recife."),
        ]
    )
    agent = create_agent(fake, [get_weather], system_prompt="You are helpful.")

    updates = list(
        agent.stream(
            {"messages": [("user", "weather?")]}, stream_mode="updates"
        )
    )
    # Each update is {node_name: {"messages": [...]}}; collect every message.
    seen = []
    for upd in updates:
        for node_state in upd.values():
            for m in node_state.get("messages", []):
                seen.append(m)
    contents = [getattr(m, "content", "") for m in seen]
    assert any("sunny" in c.lower() for c in contents)
