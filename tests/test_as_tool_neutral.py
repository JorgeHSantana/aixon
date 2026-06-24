"""Guard: Agent.as_tool() returns the NEUTRAL AgentTool (contract §2.4).
LangChain conversion is exclusively coerce_tools' job."""

from langchain_core.tools import BaseTool

from aixon.agent import Agent, AgentTool
from aixon.message import Chunk, Message
from aixon.registry import get_registry
from aixon._interop.tools import coerce_tools


def _concrete(name_cls, reply):
    return type(
        name_cls,
        (Agent,),
        {
            "invoke": lambda self, messages: Message(
                role="assistant", content=reply + ":" + messages[-1].content
            ),
            "stream": lambda self, m: iter([Chunk(done=True)]),
        },
    )


def test_as_tool_returns_neutral_agenttool_not_langchain():
    _concrete("PlainAgent", "p")
    tool = get_registry().resolve("plainagent").as_tool()
    assert isinstance(tool, AgentTool)
    assert not isinstance(tool, BaseTool)


def test_as_tool_output_is_consumable_by_coerce_tools():
    _concrete("PlainAgent", "p")
    tool = get_registry().resolve("plainagent").as_tool()
    [lc_tool] = coerce_tools([tool])
    assert isinstance(lc_tool, BaseTool)
    # Round-trip: the LangChain tool runs the neutral agent.
    assert lc_tool.invoke({"text": "ping"}) == "p:ping"
