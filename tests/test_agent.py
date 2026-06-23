import pytest
from typing import Iterator

from aixon.agent import Agent
from aixon.message import Message, Chunk
from aixon.exceptions import NamingError
from aixon.registry import get_registry


def _make_concrete(name_cls="EchoAgent", **attrs):
    """Define a concrete Agent subclass at call time (so suffix errors raise here)."""
    body = {
        "invoke": lambda self, messages: Message(role="assistant", content="ok"),
        "stream": lambda self, messages: iter([Chunk(content="ok", done=True)]),
        **attrs,
    }
    return type(name_cls, (Agent,), body)


def test_concrete_subclass_registers_itself():
    _make_concrete("EchoAgent")
    agent = get_registry().resolve("echoagent")
    assert agent.invoke([]).content == "ok"


def test_explicit_name_attribute_wins():
    _make_concrete("EchoAgent", name="echo")
    assert get_registry().resolve("echo").name == "echo"


def test_bad_suffix_raises_naming_error():
    with pytest.raises(NamingError, match="Agent"):
        _make_concrete("Echo")  # missing 'Agent' suffix


def test_abstract_subtype_is_exempt_and_unregistered():
    # Simulate how LLMAgent/ToolAgent will be declared in later plans.
    class FakeSubtype(Agent, abstract=True):
        _suffix = "Agent"

    assert get_registry().all() == []
    # A concrete subclass of the abstract subtype validates against _suffix.
    type(
        "GreeterAgent",
        (FakeSubtype,),
        {
            "invoke": lambda self, m: Message(role="assistant"),
            "stream": lambda self, m: iter([Chunk(done=True)]),
        },
    )
    assert get_registry().resolve("greeteragent").name == "greeteragent"


def test_custom_suffix_on_abstract_subtype():
    class FakeOrchestrator(Agent, abstract=True):
        _suffix = "Orchestrator"

    with pytest.raises(NamingError, match="Orchestrator"):
        type(
            "BadName",
            (FakeOrchestrator,),
            {
                "invoke": lambda self, m: Message(role="assistant"),
                "stream": lambda self, m: iter([Chunk(done=True)]),
            },
        )


def test_missing_abstract_methods_raise_type_error():
    with pytest.raises(TypeError):
        type("IncompleteAgent", (Agent,), {})  # no invoke/stream -> ABC error on cls()
