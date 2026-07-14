# tests/test_audit_core.py
"""Audit-fix regression tests (llm.py, agent.py, runtime.py, registry.py)."""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Iterator, Optional

import pytest
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk

from aixon.llm import LLM
from aixon.message import Chunk, Message
from tests._fakes import FakeChatModel  # registers fake provider


# ── Finding 1: stream/astream must flatten list-of-blocks content deltas ─────

class BlockStreamChatModel(FakeChatModel):
    """Streams AIMessageChunk deltas whose content is a list of blocks, the
    shape Gemini 2.5 / Anthropic-with-thinking produce."""

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(
            message=AIMessageChunk(content=[{"type": "text", "text": "Hel"}])
        )
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=[
                    {"type": "thinking", "thinking": "hidden"},
                    {"type": "text", "text": "lo"},
                ]
            )
        )


def _block_llm() -> LLM:
    llm = LLM("fake-1", provider="fake")
    llm._chat_model = BlockStreamChatModel()  # inject; skip provider.build
    return llm


def test_stream_flattens_list_of_blocks_content():
    chunks = list(_block_llm().stream([Message(role="user", content="hi")]))
    assert "".join(c.content for c in chunks) == "Hello"
    assert chunks[-1].done is True


def test_astream_flattens_list_of_blocks_content():
    async def run() -> list[Chunk]:
        llm = _block_llm()
        return [c async for c in llm.astream([Message(role="user", content="hi")])]

    chunks = asyncio.run(run())
    assert "".join(c.content for c in chunks) == "Hello"
    assert chunks[-1].done is True


# ── Finding 2: default astream bridge must propagate contextvars ─────────────

def test_astream_bridge_propagates_generation_params_and_reasoning():
    from aixon.agent import Agent
    from aixon.reasoning import emit_reasoning, reasoning_channel
    from aixon.runtime import current_generation_params, generation_params

    class CtxProbeAgent(Agent):
        def invoke(self, messages: list[Message]) -> Message:
            return Message(role="assistant", content="")

        def stream(self, messages: list[Message]) -> Iterator[Chunk]:
            params = current_generation_params()
            emit_reasoning("probe-line")
            yield Chunk(content=f"temp={params.get('temperature')}")
            yield Chunk(done=True)

    agent = CtxProbeAgent()

    async def run() -> tuple[list[Chunk], list[str]]:
        with generation_params({"temperature": 0.7}):
            with reasoning_channel() as channel:
                chunks = [
                    c async for c in agent.astream([Message(role="user", content="hi")])
                ]
                return chunks, channel.lines

    chunks, reasoning_lines = asyncio.run(run())
    assert chunks[0].content == "temp=0.7"
    assert reasoning_lines == ["probe-line"]


# ── Finding 3: re-instantiation must keep the default name ───────────────────

def test_reinstantiated_agent_has_default_name():
    from aixon.agent import Agent
    from aixon.registry import get_registry

    class NameProbeAgent(Agent):
        def invoke(self, messages: list[Message]) -> Message:
            return Message(role="assistant", content="")

        def stream(self, messages: list[Message]) -> Iterator[Chunk]:
            yield Chunk(done=True)

    # First instance (auto-created at class definition) got the default name.
    assert get_registry().resolve("nameprobeagent").name == "nameprobeagent"
    # A second instance must get the same default name, not "".
    second = NameProbeAgent()
    assert second.name == "nameprobeagent"
    assert second.as_tool().name == "nameprobeagent"
    # Register-once semantics preserved: no duplicate registration happened.
    assert len(get_registry().all()) == 1


def test_subclass_of_concrete_agent_gets_own_name_and_registration():
    from aixon.agent import Agent
    from aixon.registry import get_registry

    class ParentProbeAgent(Agent):
        def invoke(self, messages: list[Message]) -> Message:
            return Message(role="assistant", content="parent")

        def stream(self, messages: list[Message]) -> Iterator[Chunk]:
            yield Chunk(done=True)

    class ChildProbeAgent(ParentProbeAgent):
        def invoke(self, messages: list[Message]) -> Message:
            return Message(role="assistant", content="child")

    # The child must not inherit the parent's name nor its _registered flag:
    # it gets its own default name and its own registry entry.
    assert ChildProbeAgent().name == "childprobeagent"
    assert get_registry().resolve("childprobeagent").invoke([]).content == "child"
    assert get_registry().resolve("parentprobeagent").invoke([]).content == "parent"


# ── Finding 4: normalize provider-dialect generation params ──────────────────

def test_max_completion_tokens_normalized_to_max_tokens():
    from aixon.runtime import current_generation_params, generation_params

    with generation_params({"max_completion_tokens": 128}):
        assert current_generation_params() == {"max_tokens": 128}


def test_stop_sequences_normalized_to_stop():
    from aixon.runtime import current_generation_params, generation_params

    with generation_params({"stop_sequences": ["END"]}):
        assert current_generation_params() == {"stop": ["END"]}


def test_canonical_params_win_over_dialect_aliases():
    from aixon.runtime import current_generation_params, generation_params

    with generation_params(
        {"max_tokens": 64, "max_completion_tokens": 128,
         "stop": ["A"], "stop_sequences": ["B"]}
    ):
        assert current_generation_params() == {"max_tokens": 64, "stop": ["A"]}


def test_normalized_params_reach_the_model(monkeypatch):
    """UPDATED (final-review bind-path unification): per-request params now
    reach the model via ``Provider.build()`` constructor kwargs
    (``request_chat_model()``), not via ``.bind()`` onto a manually-injected
    ``_chat_model`` — see the matching update in test_sp1_genparams.py for
    the full rationale."""
    from aixon.runtime import generation_params
    from tests._fakes import FakeProvider

    captured: list[dict] = []
    original_build = FakeProvider.build

    def recording_build(self, model, **params):
        captured.append(params)
        return original_build(self, model, **params)

    monkeypatch.setattr(FakeProvider, "build", recording_build)

    llm = LLM("fake-1", provider="fake")
    with generation_params({"max_completion_tokens": 42}):
        llm.complete([Message(role="user", content="hi")])
    assert captured[-1].get("max_tokens") == 42
    assert "max_completion_tokens" not in captured[-1]


# ── Finding 5: Registry.clear() must also reset _registered flags ────────────

def test_clear_allows_reregistration():
    from aixon.agent import Agent
    from aixon.registry import get_registry

    class ClearProbeAgent(Agent):
        def invoke(self, messages: list[Message]) -> Message:
            return Message(role="assistant", content="")

        def stream(self, messages: list[Message]) -> Iterator[Chunk]:
            yield Chunk(done=True)

    registry = get_registry()
    registry.clear()
    assert registry.all() == []
    assert ClearProbeAgent._registered is False
    # Re-instantiation must re-register into the (same) cleared registry.
    ClearProbeAgent()
    assert registry.resolve("clearprobeagent").name == "clearprobeagent"


# ── Finding 6: get_registry()/register() need synchronization ────────────────

def test_get_registry_concurrent_first_call_builds_one_instance(monkeypatch):
    import time

    import aixon.registry as registry_mod

    constructions: list[int] = []

    class SlowRegistry(registry_mod.Registry):
        def __init__(self) -> None:
            constructions.append(1)
            time.sleep(0.05)  # widen the check-then-set race window
            super().__init__()

    monkeypatch.setattr(registry_mod, "Registry", SlowRegistry)
    monkeypatch.setattr(registry_mod, "_registry", None)

    barrier = threading.Barrier(4)
    results: list[object] = []

    def worker() -> None:
        barrier.wait()
        results.append(registry_mod.get_registry())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(constructions) == 1
    assert all(r is results[0] for r in results)


def test_register_concurrent_duplicate_name_admits_exactly_one():
    import time

    from aixon.exceptions import RegistrationError
    from aixon.registry import Registry

    class SlowAliases:
        """Iterating sleeps, widening register()'s check-then-insert window."""

        def __iter__(self):
            time.sleep(0.05)
            return iter(())

    class FakeAgent:
        name = "dupe"
        aliases = SlowAliases()
        hidden = False

    registry = Registry()
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def worker() -> None:
        barrier.wait()
        try:
            registry.register(FakeAgent())
            outcomes.append("ok")
        except RegistrationError:
            outcomes.append("rejected")

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(outcomes) == ["ok", "rejected"]
    assert len(registry.all()) == 1


# ── Finding 7: mutating the returned params dict must not pollute state ──────

def test_mutating_default_generation_params_does_not_pollute_global_state():
    from aixon.runtime import current_generation_params

    current_generation_params()["temperature"] = 9.9
    assert current_generation_params() == {}


def test_mutating_active_generation_params_does_not_leak_between_reads():
    from aixon.runtime import current_generation_params, generation_params

    with generation_params({"temperature": 0.3}):
        current_generation_params()["injected"] = True
        assert current_generation_params() == {"temperature": 0.3}
