# tests/test_agent_impose_model.py
"""0.1.28 — `Agent.impose_model`: an agent imposes its own model on its tree.

An agent declared with `impose_model = True` runs under
`imposed_model_scope(<its llm's model>, <its llm's provider>)`, so every LLM of
every agent beneath it (tools, reflective workers, judges) builds THAT model,
whatever it was declared with. A scope that is already in force wins.

The scope is a ContextVar and the entry points are generators and coroutines,
so half of these tests are about where the scope must NOT be: a generator that
holds it across a `yield` hands it to whoever consumes the generator (and, if
the consumer walks away, leaves it there for good). The wrapper therefore opens
the scope per step, only while the inner iterator runs, and closes it before
each chunk reaches the consumer.

Everything is offline. The imposed model is `tests/_fakes.RecordingProvider`
("rec"); the providers the tree is declared with are booby-trapped by the
`explodiu` fixture. The tree tests run on a ToolAgent coordinator whose tools
are a ToolAgent and a ReflectiveAgent (worker plus judge), the shape of a real
assistant that fans out to specialists."""
from __future__ import annotations

import asyncio
import contextvars
from abc import abstractmethod

import pytest
from langchain_core.messages import AIMessage

from aixon import Agent, LLMAgent, ReflectiveAgent, ToolAgent
from aixon.exceptions import AgentNotFoundError, AixonError
from aixon.llm import LLM
from aixon.message import Chunk, Message
from aixon.providers.openai import OpenAIProvider
from aixon.registry import get_registry
from aixon.runtime import _imposed_model, current_imposed_model, imposed_model_scope
from tests._fakes import RecordingChatModel, make_recording_provider

# Every test finds a fresh "rec" provider registered, whatever ran before it:
# an agent that imposes ("tier-x", "rec") resolves it as it starts. The tests
# that need `rec` to answer register one with a script of their own.
pytestmark = pytest.mark.usefixtures("rec")

ENTRIES = ["invoke", "ainvoke", "stream", "astream"]
ASK = [Message(role="user", content="q")]
PAIR = ("tier-x", "rec")  # what the coordinator and the probe agents impose


@pytest.fixture(autouse=True)
def _no_scope_left_behind():
    """Every test of this module must leave the context it ran in without an
    imposed model. pytest runs them all in one context, so a leak would also
    fail every test after it: report it on the test that left it, and clear it."""
    yield
    left = current_imposed_model()
    if left is not None:
        _imposed_model.set(None)
        pytest.fail(f"the test left the imposed model {left} set in its context")


def script() -> list[AIMessage]:
    """What the model answers, in the order the tree makes its calls."""
    return [
        AIMessage(content="", tool_calls=[{"name": "especialista", "args": {"text": "q"}, "id": "c1"}]),
        AIMessage(content="dado interno"),
        AIMessage(content="", tool_calls=[{"name": "revisor", "args": {"text": "q"}, "id": "c2"}]),
        AIMessage(content="rascunho revisado"),
        AIMessage(content="APROVADO"),
        AIMessage(content="resposta final"),
    ]


def tree() -> Agent:
    """Define the coordinator tree and return the coordinator. Each caller gets
    fresh classes, so fresh LLMs with empty model caches."""

    class EspecialistaAgent(ToolAgent):
        name = "especialista"
        description = "Dados internos."
        llm = LLM("gpt-5.4", provider="openai")

    class RevisorTrabalhadorAgent(ToolAgent):
        name = "revisor_trabalhador"
        description = "Redige o rascunho."
        llm = LLM("claude-sonnet-5", provider="anthropic")

    class RevisorAgent(ReflectiveAgent):
        name = "revisor"
        description = "Redige e revisa."
        agent = RevisorTrabalhadorAgent
        judge_llm = LLM("gpt-5.4-mini", provider="openai")
        judge_rubric = "A resposta deve ser objetiva."

    registry = get_registry()

    class CoordenadorAgent(ToolAgent):
        name = "coordenador"
        description = "Coordena os especialistas."
        llm = LLM("tier-x", provider="rec")
        impose_model = True
        tools = [
            registry.resolve("especialista").as_tool(),
            registry.resolve("revisor").as_tool(),
        ]

    return registry.resolve("coordenador")


def run(agent: Agent, entry: str, messages: list[Message]) -> str:
    """The final content `agent` answers through `entry`: the Message's content
    for invoke/ainvoke, the chunks joined for stream/astream. The async entries
    run on an event loop of their own."""
    if entry == "invoke":
        return agent.invoke(messages).content
    if entry == "stream":
        return "".join(chunk.content for chunk in agent.stream(messages))

    async def go() -> str:
        if entry == "ainvoke":
            return (await agent.ainvoke(messages)).content
        return "".join([chunk.content async for chunk in agent.astream(messages)])

    return asyncio.run(go())


def _no_answer(self, messages):
    return Message(role="assistant", content="")


def _no_chunks(self, messages):
    yield Chunk(done=True)


# ── the whole tree ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("entry", ENTRIES)
def test_whole_tree_runs_on_imposed_model(entry, explodiu):
    rec = make_recording_provider("rec", script())
    coordenador = tree()

    assert run(coordenador, entry, ASK) == "resposta final"

    # coordinator, specialist, reflective worker and judge: one build each, all
    # of the imposed model, none of the providers they were declared with
    assert {model for model, _ in rec.builds} == {"tier-x"}
    assert len(rec.builds) == 4
    assert explodiu == []


@pytest.mark.parametrize("entry", ENTRIES)
def test_outer_scope_wins(entry, explodiu):
    rec = make_recording_provider("rec", script())
    coordenador = tree()

    with imposed_model_scope("tier-outer", "rec"):
        assert run(coordenador, entry, ASK) == "resposta final"

    assert {model for model, _ in rec.builds} == {"tier-outer"}
    assert explodiu == []


# ── declaration ──────────────────────────────────────────────────────────────

def test_default_is_off():
    assert Agent.impose_model is False and ToolAgent.impose_model is False
    assert LLMAgent.impose_model is False


@pytest.mark.parametrize("llm", [None, "gpt-5.4"], ids=["no-llm", "llm-is-not-an-LLM"])
def test_impose_model_requires_llm(llm):
    body = {"impose_model": True, "invoke": _no_answer, "stream": _no_chunks}
    if llm is not None:
        body["llm"] = llm

    with pytest.raises(AixonError, match="impose_model"):
        type("SemLlmAgent", (Agent,), body)

    with pytest.raises(AgentNotFoundError):  # refused before it was registered
        get_registry().resolve("semllmagent")


@pytest.mark.parametrize("value", ["yes", 1])
def test_impose_model_must_be_bool(value):
    body = {"llm": LLM("tier-x", provider="rec"), "impose_model": value}

    with pytest.raises(AixonError, match="impose_model"):
        type("TalvezAgent", (ToolAgent,), body)

    with pytest.raises(AgentNotFoundError):
        get_registry().resolve("talvezagent")


def test_abstract_subtype_leaves_the_llm_to_its_concrete_subclasses():
    class BaseDeCoordenadorAgent(ToolAgent, abstract=True):
        impose_model = True  # no `llm` here: only a concrete class needs one

    with pytest.raises(AixonError, match="impose_model"):
        type("SemLlmAgent", (BaseDeCoordenadorAgent,), {})  # inherits True, has no llm

    type("ComLlmAgent", (BaseDeCoordenadorAgent,), {"llm": LLM("tier-x", provider="rec")})
    assert get_registry().resolve("comllmagent").impose_model is True


# ── the scope does not leak through generators ───────────────────────────────

@pytest.mark.parametrize("entry", ["stream", "astream"])
def test_consumer_never_sees_scope_between_chunks(entry, explodiu):
    rec = make_recording_provider("rec", script())
    coordenador = tree()
    between: list = []

    if entry == "stream":
        for _ in coordenador.stream(ASK):
            between.append(current_imposed_model())
    else:
        async def go() -> None:
            async for _ in coordenador.astream(ASK):
                between.append(current_imposed_model())

        asyncio.run(go())

    assert len(between) > 1  # there were chunks to look between
    assert between == [None] * len(between)
    assert {model for model, _ in rec.builds} == {"tier-x"}  # and the tree did run imposed
    assert explodiu == []


def test_abandoned_astream_does_not_leak_scope(explodiu):
    rec = make_recording_provider("rec", script())
    coordenador = tree()

    async def go():
        async for _ in coordenador.astream(ASK):
            break  # abandoned: no aclose()

        right_after = current_imposed_model()
        # the dropped generator is finalized by the loop, in a task of its own
        await asyncio.sleep(0.05)
        return (
            right_after,
            current_imposed_model(),
            LLM("fake-1", provider="fake").request_chat_model(),
        )

    right_after, later, model = asyncio.run(go())

    assert [m for m, _ in rec.builds] == ["tier-x"]  # the stream did impose, before it was dropped
    assert right_after is None and later is None
    assert model is not rec.model  # the next call in this context uses its own model
    assert explodiu == []


# ── LLMAgent ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("entry", ["astream", "stream", "ainvoke", "invoke"])
def test_llm_agent_imposes_and_streams(entry):
    rec = make_recording_provider("rec", [AIMessage("oi")])

    class EcoAgent(LLMAgent):
        llm = LLM("tier-x", provider="rec", seed=7)
        impose_model = True

    # LLM.stream/astream resolve the model on their first step, not when they
    # are created: the scope has to be open for the step, not just the call.
    assert run(get_registry().resolve("ecoagent"), entry, ASK) == "oi"
    # `seed` is not portable: only the scope keeps it out of the build
    assert rec.builds == [("tier-x", {})]


# ── the wrapper ──────────────────────────────────────────────────────────────

def test_off_returns_exactly_what_the_method_returns():
    returned = {entry: object() for entry in ENTRIES}

    class PuroAgent(Agent):
        def invoke(self, messages):
            return returned["invoke"]

        def ainvoke(self, messages):
            return returned["ainvoke"]

        def stream(self, messages):
            return returned["stream"]

        def astream(self, messages):
            return returned["astream"]

    agent = get_registry().resolve("puroagent")

    for entry in ENTRIES:  # the very object, not one of ours wrapped around it
        assert getattr(agent, entry)(ASK) is returned[entry]


def test_entry_points_are_wrapped_once():
    class UmAgent(Agent):
        def invoke(self, messages):
            return Message(role="assistant", content="")

        def stream(self, messages):
            yield Chunk(done=True)

    for name in ("invoke", "stream"):
        wrapper = vars(UmAgent)[name]
        assert wrapper.__wrapped__.__qualname__.endswith(f"UmAgent.{name}")  # what the body wrote
        assert not hasattr(wrapper.__wrapped__, "__wrapped__")  # under one layer, not two

    class DoisAgent(UmAgent):
        invoke = UmAgent.invoke  # a function that is already wrapped

    assert vars(DoisAgent)["invoke"] is vars(UmAgent)["invoke"]  # not wrapped again


def test_abstract_subtypes_are_wrapped_and_their_subclasses_inherit_it():
    class BaseAgent(Agent, abstract=True):
        def invoke(self, messages):
            return Message(role="assistant", content="")

        def stream(self, messages):
            yield Chunk(done=True)

        async def ainvoke(self, messages):
            return Message(role="assistant", content="")

        async def astream(self, messages):
            yield Chunk(done=True)

    class FilhoAgent(BaseAgent):
        pass

    for name in ENTRIES:
        assert hasattr(vars(BaseAgent)[name], "__wrapped__")
        assert name not in vars(FilhoAgent)  # nothing of its own to wrap


def test_wrapping_keeps_abstract_methods_abstract():
    class BaseAgent(Agent, abstract=True):
        @abstractmethod
        def invoke(self, messages): ...

        @abstractmethod
        def stream(self, messages): ...

    assert vars(BaseAgent)["invoke"].__isabstractmethod__ is True

    with pytest.raises(TypeError, match="abstract"):
        class MeioAgent(BaseAgent):
            def invoke(self, messages):
                return Message(role="assistant", content="")


def probe(seen: list) -> Agent:
    """A plain Agent that imposes `PAIR` and logs into `seen`, as `(where,
    scope)`, the scope in force at every moment its own code runs. `stream` and
    `astream` are plain functions that return the iterator, so a `create` entry
    shows the scope while the iterator is made, apart from the steps."""

    class SondaAgent(Agent):
        llm = LLM("tier-x", provider="rec")
        impose_model = True

        def invoke(self, messages):
            seen.append(("invoke", current_imposed_model()))
            return Message(role="assistant", content="ok")

        async def ainvoke(self, messages):
            seen.append(("ainvoke", current_imposed_model()))
            return Message(role="assistant", content="ok")

        def stream(self, messages):
            seen.append(("stream:create", current_imposed_model()))
            return self._chunks()

        def _chunks(self):
            try:
                for n in (1, 2):
                    seen.append((f"stream:step{n}", current_imposed_model()))
                    yield Chunk(content=str(n))
            finally:
                seen.append(("stream:close", current_imposed_model()))

        def astream(self, messages):
            seen.append(("astream:create", current_imposed_model()))
            return self._achunks()

        async def _achunks(self):
            try:
                for n in (1, 2):
                    seen.append((f"astream:step{n}", current_imposed_model()))
                    yield Chunk(content=str(n))
            finally:
                seen.append(("astream:close", current_imposed_model()))

    return get_registry().resolve("sondaagent")


def test_invoke_and_ainvoke_run_under_the_scope():
    seen: list = []
    agent = probe(seen)

    agent.invoke(ASK)
    asyncio.run(agent.ainvoke(ASK))

    assert seen == [("invoke", PAIR), ("ainvoke", PAIR)]
    assert current_imposed_model() is None


def test_stream_scope_is_on_only_while_the_inner_iterator_runs():
    seen: list = []
    agent = probe(seen)

    between = [current_imposed_model() for _ in agent.stream(ASK)]

    assert seen == [
        ("stream:create", PAIR), ("stream:step1", PAIR), ("stream:step2", PAIR),
        ("stream:close", PAIR),
    ]
    assert between == [None, None]


def test_astream_scope_is_on_only_while_the_inner_iterator_runs():
    seen: list = []
    agent = probe(seen)

    async def go() -> list:
        return [current_imposed_model() async for _ in agent.astream(ASK)]

    assert asyncio.run(go()) == [None, None]
    assert seen == [
        ("astream:create", PAIR), ("astream:step1", PAIR), ("astream:step2", PAIR),
        ("astream:close", PAIR),
    ]


def test_closing_an_unfinished_stream_closes_the_inner_one_under_the_scope():
    seen: list = []
    stream = probe(seen).stream(ASK)

    next(stream)
    assert current_imposed_model() is None
    stream.close()

    assert seen[-1] == ("stream:close", PAIR)
    assert current_imposed_model() is None


def test_stream_advanced_from_a_different_context_at_each_step_still_imposes():
    # Starlette advances a sync iterator with one `next()` per worker thread,
    # each in a copy of the context of its own. A scope held across the steps
    # could not even be reset there; one opened per step does not care.
    seen: list = []
    stream = probe(seen).stream(ASK)
    chunks: list = []

    while True:
        try:
            chunks.append(contextvars.copy_context().run(next, stream))
        except StopIteration:
            break

    assert [chunk.content for chunk in chunks] == ["1", "2"]
    assert [scope for _, scope in seen] == [PAIR] * 4  # create, step1, step2, close
    assert current_imposed_model() is None


def test_closing_an_unfinished_astream_closes_the_inner_one_under_the_scope():
    seen: list = []
    agent = probe(seen)

    async def go() -> tuple:
        stream = agent.astream(ASK)
        await anext(stream)
        suspended = current_imposed_model()
        await stream.aclose()
        return suspended, current_imposed_model()

    assert asyncio.run(go()) == (None, None)
    assert seen[-1] == ("astream:close", PAIR)


def test_abandoned_astream_is_closed_under_the_scope_without_leaking_it():
    seen: list = []
    agent = probe(seen)

    async def go():
        async for _ in agent.astream(ASK):
            break  # abandoned: no aclose()

        # the loop finalizes the dropped generator, in a task of its own
        await asyncio.sleep(0.05)
        return current_imposed_model()

    assert asyncio.run(go()) is None
    assert seen[-1] == ("astream:close", PAIR)  # it did run, and the inner one closed under the scope


class Boom(Exception):
    pass


@pytest.mark.parametrize("entry", ENTRIES)
def test_failing_entry_point_leaves_no_scope_behind(entry):
    class FalhaAgent(Agent):
        llm = LLM("tier-x", provider="rec")
        impose_model = True

        def invoke(self, messages):
            raise Boom()

        async def ainvoke(self, messages):
            raise Boom()

        def stream(self, messages):
            yield Chunk(content="1")
            raise Boom()

        async def astream(self, messages):
            yield Chunk(content="1")
            raise Boom()

    agent = get_registry().resolve("falhaagent")

    def fail_in_sync() -> tuple | None:
        with pytest.raises(Boom):
            agent.invoke(ASK) if entry == "invoke" else list(agent.stream(ASK))
        return current_imposed_model()

    async def fail_in_async() -> tuple | None:
        with pytest.raises(Boom):
            if entry == "ainvoke":
                await agent.ainvoke(ASK)
            else:
                [chunk async for chunk in agent.astream(ASK)]
        return current_imposed_model()  # in the very task that consumed it

    left = fail_in_sync() if entry in ("invoke", "stream") else asyncio.run(fail_in_async())
    assert left is None


@pytest.mark.parametrize("entry", ["ainvoke", "stream", "astream"])
def test_outer_scope_is_looked_up_when_the_run_starts_not_when_it_is_made(entry):
    seen: list = []
    agent = probe(seen)

    def make():
        return getattr(agent, entry)(ASK)

    def drive(made) -> None:
        if entry == "stream":
            list(made)
        elif entry == "ainvoke":
            asyncio.run(made)
        else:
            async def drain() -> None:
                [chunk async for chunk in made]

            asyncio.run(drain())

    with imposed_model_scope("tier-outer", "rec"):
        made = make()
    drive(made)  # made under an outer scope, started with none in force: it imposes its own
    assert {scope for _, scope in seen} == {PAIR}

    seen.clear()
    made = make()
    with imposed_model_scope("tier-outer", "rec"):
        drive(made)  # made with none in force, started under an outer one: that one wins
    assert {scope for _, scope in seen} == {("tier-outer", "rec")}


def test_provider_is_the_one_the_llm_resolves(monkeypatch):
    # `LLM("gpt-5.4")` names no provider: it is inferred from the model name.
    builds: list = []
    model = RecordingChatModel(script=[AIMessage("oi")])

    def build(self, name, **params):
        builds.append((name, dict(params)))
        return model

    monkeypatch.setattr(OpenAIProvider, "build", build)

    class InferidoAgent(LLMAgent):
        llm = LLM("gpt-5.4", temperature=0.2, seed=7)
        impose_model = True

    assert get_registry().resolve("inferidoagent").invoke(ASK).content == "oi"
    assert builds == [("gpt-5.4", {"temperature": 0.2})]  # imposed on "openai"; `seed` stays behind


def test_sync_only_agent_is_bridged_to_async_under_the_scope():
    seen: list = []

    class LegadoAgent(Agent):
        llm = LLM("tier-x", provider="rec")
        impose_model = True

        def invoke(self, messages):
            seen.append(("invoke", current_imposed_model()))
            return Message(role="assistant", content="ok")

        def stream(self, messages):
            seen.append(("stream", current_imposed_model()))
            yield Chunk(content="1")
            yield Chunk(done=True)

    agent = get_registry().resolve("legadoagent")

    async def go() -> list:
        await agent.ainvoke(ASK)  # Agent.ainvoke: self.invoke in a worker thread
        return [current_imposed_model() async for _ in agent.astream(ASK)]  # self.stream in one

    assert asyncio.run(go()) == [None, None]
    assert seen == [("invoke", PAIR), ("stream", PAIR)]
