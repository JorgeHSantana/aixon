# tests/_fakes.py
"""Single owner of hermetic test doubles for aixon (contract §9.1).

Imported by Plan 2 tests and by Plans 3, 4, 5, 7. DO NOT redefine these
elsewhere. Everything here is offline: no API key, no network.

FakeChatModel is copied VERBATIM from the interface contract §9.1 and is
validated against langchain 1.3 / langchain-core 1.4 / langgraph 1.2 — it
drives langchain.agents.create_agent through a tool call then a final answer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aixon.agent import Agent
from aixon.message import Chunk, Message
from aixon.providers.base import Provider, register_provider

if TYPE_CHECKING:
    from aixon.llm import LLM


FAKE_MODEL = "fake-1"
FAKE_PROVIDER = "fake"


# ── FakeChatModel — VERBATIM from contract §9.1 (do not edit) ────────────────

class FakeChatModel(BaseChatModel):
    """Scriptable offline chat model. `script` is a list of AIMessages returned
    one per LLM call (set tool_calls on an AIMessage to drive a tool step)."""

    script: list = []
    _idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeChatModel":
        return self  # tools ignored; script drives calls

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        i = self._idx
        msg = self.script[i] if i < len(self.script) else AIMessage(content="(done)")
        object.__setattr__(self, "_idx", i + 1)
        return ChatResult(generations=[ChatGeneration(message=msg)])


# Example script for a tool-calling test (used by Plan 3):
#   FakeChatModel(script=[
#       AIMessage(content="", tool_calls=[{"name":"get_weather","args":{"city":"Recife"},"id":"call_1"}]),
#       AIMessage(content="The weather in Recife is sunny."),
#   ])


# ── Fake provider ────────────────────────────────────────────────────────────

class FakeProvider(Provider):
    """Provider named 'fake' whose build() returns a FakeChatModel.

    The returned model's `script` can be overridden per test, e.g.:
        from tests._fakes import make_llm
        llm = make_llm()
        llm.chat_model.script = [AIMessage(content="hi")]
    A bare FakeChatModel() with an empty script echoes "(done)" for each call,
    which is enough for the LLM.complete / LLM.stream smoke tests below.
    """

    name = FAKE_PROVIDER
    env_key = "FAKE_API_KEY"

    def build(self, model: str, **params: Any) -> FakeChatModel:
        return FakeChatModel()


def register_fake_provider() -> None:
    """Register the 'fake' provider. Idempotent — safe to call repeatedly."""
    register_provider(FakeProvider())


# Register at import time so `LLM("fake-1", provider="fake")` works for any
# importer without an explicit call.
register_fake_provider()


# ── Convenience factories (used by Plans 3/4/5/7) ────────────────────────────

def make_llm(**params: Any) -> "LLM":
    """Return an LLM bound to the fake provider/model."""
    register_fake_provider()
    from aixon.llm import LLM  # local import: aixon.llm depends on providers

    return LLM(FAKE_MODEL, provider=FAKE_PROVIDER, **params)


def make_echo_agent(name: str = "echo", *, hidden: bool = False):
    """Define + register a concrete Agent that echoes the last message.

    invoke([... , Message(content="x")]) -> Message(role="assistant", content="x")
    stream(...) yields one content Chunk then Chunk(done=True).
    Returns the registered agent instance. Used by server/CLI/orchestrator
    tests that need an Agent but not a real LLM.
    """
    from typing import Iterator

    def invoke(self, messages: list[Message]) -> Message:
        last = messages[-1].content if messages else ""
        return Message(role="assistant", content=last)

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        last = messages[-1].content if messages else ""
        yield Chunk(content=last)
        yield Chunk(done=True)

    cls_name = f"{name.capitalize()}Agent"
    cls = type(
        cls_name,
        (Agent,),
        {"invoke": invoke, "stream": stream, "name": name, "hidden": hidden},
    )
    # Agent.__init_subclass__ already instantiated + registered it; fetch it.
    from aixon.registry import get_registry

    return get_registry().resolve(name)
