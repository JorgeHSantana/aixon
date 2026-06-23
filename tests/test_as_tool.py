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
