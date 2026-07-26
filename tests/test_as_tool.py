import asyncio
import logging

from aixon.agent import Agent, AgentTool
from aixon.message import Message, Chunk


def _concrete(name_cls, reply, **attrs):
    return type(
        name_cls,
        (Agent,),
        {
            "invoke": lambda self, messages: Message(
                role="assistant", content=reply + ":" + messages[-1].content
            ),
            "stream": lambda self, m: iter([Chunk(done=True)]),
            **attrs,
        },
    )


def test_as_tool_returns_descriptor_with_defaults():
    _concrete("HelperAgent", "h", description="a helper")
    from aixon.registry import get_registry

    agent = get_registry().resolve("helperagent")
    tool = agent.as_tool()
    assert isinstance(tool, AgentTool)
    assert tool.name == "helperagent"
    assert tool.description == "a helper"


def test_as_tool_func_invokes_agent_with_user_message():
    _concrete("HelperAgent", "h")
    from aixon.registry import get_registry

    tool = get_registry().resolve("helperagent").as_tool()
    assert tool.func("ping") == "h:ping"


def test_as_tool_overrides():
    _concrete("HelperAgent", "h")
    from aixon.registry import get_registry

    tool = get_registry().resolve("helperagent").as_tool(
        name="custom", description="custom desc"
    )
    assert tool.name == "custom"
    assert tool.description == "custom desc"


def _leaky_client_tool_agent(name: str):
    """Worker that surfaced a CLIENT tool_calls answer (#18c): content=""
    with tool_calls set — the shape a ToolAgent(client_tools="merge") worker
    returns when the model calls a client tool. as_tool() has no channel to
    relay Message.tool_calls back to whatever called the tool."""
    calls = [{"name": "inserir_no_documento", "args": {"texto": "x"}, "id": "c1"}]

    def invoke(self, messages):
        return Message(role="assistant", content="", tool_calls=calls)

    async def ainvoke(self, messages):
        return Message(role="assistant", content="", tool_calls=calls)

    def stream(self, messages):
        yield Chunk(tool_calls=calls)
        yield Chunk(done=True)

    return type(
        name.capitalize() + "Agent", (Agent,),
        {"name": name, "invoke": invoke, "ainvoke": ainvoke, "stream": stream},
    )


def test_as_tool_worker_client_tool_calls_sem_content_avisa_e_nao_vaza_vazio(caplog):
    _leaky_client_tool_agent("leaky1")
    from aixon.registry import get_registry

    tool = get_registry().resolve("leaky1").as_tool()
    with caplog.at_level(logging.WARNING, logger="aixon.agent"):
        out = tool.func("faça algo no documento")

    assert out != ""
    assert "inserir_no_documento" in out
    assert "as_tool" in out
    assert "leaky1" in out
    assert any("leaky1" in m for m in caplog.messages)


def test_as_tool_arun_worker_client_tool_calls_sem_content_tambem_avisa():
    _leaky_client_tool_agent("leaky2")
    from aixon.registry import get_registry

    tool = get_registry().resolve("leaky2").as_tool()
    out = asyncio.run(tool.coroutine("faça algo no documento"))

    assert out != ""
    assert "inserir_no_documento" in out
