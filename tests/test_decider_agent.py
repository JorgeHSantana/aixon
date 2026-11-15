"""Doubt and Check, alone and declared on a ToolAgent. The agent tests run every
scenario through the four entry points (invoke, ainvoke, stream, astream): a
gate that covers only one of them leaves the others unguarded."""
import asyncio

import pytest
from langchain_core.messages import AIMessage

from aixon import AixonError, Check, Decider, Doubt
from aixon.agents.tool_agent import ToolAgent
from aixon.decider import apply_doubts, failed_checks, redo_messages, resolve_doubts
from aixon.deciders import FakeDeciderProvider
from aixon.llm import LLM
from aixon.message import Message
from aixon.registry import get_registry
from tests._fakes import RecordingChatModel

STORES = ["acerola", "abadia"]


def find_store_word(message):
    """Pure and cheap: is a store name loose in the message?"""
    word = next((w for w in message.lower().replace("?", "").split() if w in STORES), None)
    return {"word": word} if word else None


def store_decider(script):
    class StoreOrProductDecider(Decider):
        provider = FakeDeciderProvider({"store_or_product": script})
        instruction = "What does the word in 'word' refer to?"
        options = {"store": "a branch of the chain", "product": "an item for sale"}

    return StoreOrProductDecider


def store_doubt(script):
    return Doubt(
        store_decider(script),
        applies=find_store_word,
        on_sure=lambda d, case: f"[{case['word']} is a {d.choice}]",
        on_unsure=lambda d, case: f"[ask the user: store or product '{case['word']}'?]",
    )


def sense_check(script, **kw):
    class MakesSenseDecider(Decider):
        provider = FakeDeciderProvider(script)
        instruction = "Does the answer address what the user asked?"
        options = {"yes": "it does", "no": "it is about something else"}

    return Check(MakesSenseDecider, redo=lambda d, answer: "Your answer missed the question. Redo it.", **kw)


def user(text):
    return [Message(role="user", content=text)]


# ---- Doubt -------------------------------------------------------------------

def test_doubt_sure_appends_the_directive_to_the_last_user_message():
    messages = [Message(role="user", content="hi"), Message(role="assistant", content="hello"),
                Message(role="user", content="how much did acerola sell?")]
    out = apply_doubts(messages, [store_doubt("product")])
    assert out[-1].content == "how much did acerola sell?\n\n[acerola is a product]"
    assert out[:2] == messages[:2] and messages[-1].content == "how much did acerola sell?"


def test_doubt_unsure_takes_the_fallback():
    out = apply_doubts(user("how much did acerola sell?"), [store_doubt(("store", 0.6))])
    assert out[-1].content.endswith("[ask the user: store or product 'acerola'?]")


def test_doubt_not_in_the_message_does_not_call_the_engine():
    doubt = store_doubt("store")
    messages = user("total revenue yesterday?")
    assert apply_doubts(messages, [doubt]) is messages
    assert doubt.decider.provider.calls == []


def test_doubt_without_a_decision_leaves_the_message_alone():
    class DownDecider(Decider):
        provider = FakeDeciderProvider(error=TimeoutError())
        instruction = "x"
        options = ["store", "product"]

    messages = user("acerola?")
    assert apply_doubts(messages, [Doubt(DownDecider, on_sure=lambda d, c: "never")]) is messages


def test_doubt_hooks_that_raise_or_return_none_add_nothing():
    def boom(decision, case):
        raise RuntimeError("hook bug")

    messages = user("acerola?")
    decider = store_decider("store")
    assert apply_doubts(messages, [Doubt(decider, on_sure=boom)]) is messages
    assert apply_doubts(messages, [Doubt(decider, on_sure=lambda d, c: None)]) is messages
    assert apply_doubts(messages, [Doubt(decider, applies=boom)]) is messages


def test_doubt_default_context_and_custom_context():
    doubt = store_doubt("store")
    resolve_doubts("did acerola sell?", [doubt])
    assert doubt.decider.provider.calls[0][0] == {"message": "did acerola sell?", "word": "acerola"}
    custom = Doubt(store_decider("store"), context=lambda message, case: {"text": message.upper()})
    resolve_doubts("hi", [custom])
    assert custom.decider.provider.calls[0][0] == {"text": "HI"}


def test_doubts_with_the_same_context_share_one_request():
    provider = FakeDeciderProvider({"tone": "upset", "irony": "yes"})

    class ToneDecider(Decider):
        instruction = "Tone?"
        options = ["neutral", "upset"]

    class IronyDecider(Decider):
        instruction = "Ironic?"
        options = ["yes", "no"]

    ToneDecider.provider = IronyDecider.provider = provider
    outcomes = resolve_doubts("great, wrong again", [
        Doubt(ToneDecider, on_sure=lambda d, c: "[the user is upset]"),
        Doubt(IronyDecider, on_sure=lambda d, c: "[the message is ironic]"),
    ])
    assert [o.directive for o in outcomes] == ["[the user is upset]", "[the message is ironic]"]
    assert [o.outcome for o in outcomes] == ["sure", "sure"]
    assert len(provider.calls) == 1


def test_doubts_only_read_a_user_turn():
    resumed = [Message(role="user", content="acerola?"),
               Message(role="tool", content="42", tool_call_id="c1")]
    assert apply_doubts(resumed, [store_doubt("store")]) is resumed


# ---- Check -------------------------------------------------------------------

def test_check_fails_when_the_chance_of_yes_is_low():
    check = sense_check({"makes_sense": ("no", 0.9)})
    failures = failed_checks("revenue in august?", "September was 10.", [check])
    assert [f.instruction for f in failures] == ["Your answer missed the question. Redo it."]
    assert check.decider.provider.calls[0][0] == {"message": "revenue in august?", "answer": "September was 10."}


def test_check_passes_on_doubt_by_default():
    # "no" at 0.6 leaves "yes" at 0.4: above the 0.3 line, so the answer stands.
    assert failed_checks("q", "a", [sense_check({"makes_sense": ("no", 0.6)})]) == []
    assert failed_checks("q", "a", [sense_check({"makes_sense": ("no", 0.6)}, fails_below=0.5)])


def test_check_without_a_decision_or_without_an_answer_passes():
    class DownDecider(Decider):
        provider = FakeDeciderProvider(error=TimeoutError())
        instruction = "ok?"
        options = ["yes", "no"]

    assert failed_checks("q", "a", [Check(DownDecider, redo=lambda d, a: "redo")]) == []
    check = sense_check({"makes_sense": ("no", 0.99)})
    assert failed_checks("q", "   ", [check]) == [] and check.decider.provider.calls == []


def test_check_custom_failure_rule_and_context():
    class PeriodDecider(Decider):
        provider = FakeDeciderProvider({"period": "other"})
        instruction = "Which period does the answer cover?"
        options = ["asked", "other"]

    check = Check(PeriodDecider, fails=lambda d: d.sure and d.choice == "other",
                  redo=lambda d, answer: "Wrong period.",
                  context=lambda message, answer: {"asked": message, "said": answer})
    assert [f.instruction for f in failed_checks("august", "september", [check])] == ["Wrong period."]
    assert PeriodDecider.provider.calls[0][0] == {"asked": "august", "said": "september"}


def test_check_default_rule_needs_a_yes_no_decider():
    with pytest.raises(AixonError, match="yes/no decider"):
        Check(store_decider("store"), redo=lambda d, a: "redo")


def test_redo_messages():
    failures = failed_checks("q", "bad", [sense_check({"makes_sense": ("no", 0.9)})])
    out = redo_messages(user("q"), "bad", failures)
    assert [(m.role, m.content) for m in out] == [
        ("user", "q"), ("assistant", "bad"), ("user", "Your answer missed the question. Redo it.")]


# ---- on a ToolAgent, through the four entry points -----------------------------

PATHS = ["invoke", "ainvoke", "stream", "astream"]
_count = iter(range(10_000))


def build(monkeypatch, script, *, doubts=(), checks=(), **attrs):
    """A registered ToolAgent with a recording fake model."""
    name = f"decided{next(_count)}"
    cls = type(name.capitalize() + "Agent", (ToolAgent,), {
        "name": name, "llm": LLM("fake-1", provider="fake"), "tools": [],
        "doubts": list(doubts), "checks": list(checks), **attrs})
    agent = get_registry().resolve(name)
    model = RecordingChatModel(script=[AIMessage(content=text) for text in script])
    model.calls = []
    monkeypatch.setattr(cls.llm.__class__, "chat_model", property(lambda self: model))
    return agent, model


def run(agent, path, messages):
    """(content, reasoning) of one run through *path*."""
    async def collect():
        return [chunk async for chunk in agent.astream(messages)]

    if path == "invoke":
        message = agent.invoke(messages)
        return message.content, message.reasoning or ""
    if path == "ainvoke":
        message = asyncio.run(agent.ainvoke(messages))
        return message.content, message.reasoning or ""
    chunks = list(agent.stream(messages)) if path == "stream" else asyncio.run(collect())
    assert chunks[-1].done and sum(c.done for c in chunks) == 1
    return "".join(c.content for c in chunks), "".join(c.reasoning for c in chunks)


def sent_user_texts(model, call):
    return [m.content for m in model.calls[call] if m.type == "human"]


@pytest.mark.parametrize("path", PATHS)
def test_agent_without_deciders_is_untouched(monkeypatch, path):
    agent, model = build(monkeypatch, ["The total is 4."])
    assert run(agent, path, user("add 2 and 2"))[0] == "The total is 4."
    assert sent_user_texts(model, 0) == ["add 2 and 2"]


@pytest.mark.parametrize("path", PATHS)
def test_agent_doubt_reaches_the_model_as_a_directive(monkeypatch, path):
    agent, model = build(monkeypatch, ["Which Acerola store?"], doubts=[store_doubt("store")])
    content, _ = run(agent, path, user("how much did acerola sell?"))
    assert content == "Which Acerola store?"
    assert sent_user_texts(model, 0) == ["how much did acerola sell?\n\n[acerola is a store]"]


@pytest.mark.parametrize("path", PATHS)
def test_agent_check_passes_and_the_answer_goes_out_once(monkeypatch, path):
    agent, model = build(monkeypatch, ["August was 10."],
                         checks=[sense_check({"makes_sense": ("yes", 0.95)})])
    content, reasoning = run(agent, path, user("revenue in august?"))
    assert content == "August was 10." and len(model.calls) == 1
    assert "Reviewing" not in reasoning


@pytest.mark.parametrize("path", PATHS)
def test_agent_failed_check_redoes_once_and_only_the_second_answer_goes_out(monkeypatch, path):
    check = sense_check(lambda context, name, q: ("no", 0.95) if "September" in context["answer"] else "yes")
    agent, model = build(monkeypatch, ["September was 12.", "August was 10."], checks=[check])
    content, reasoning = run(agent, path, user("revenue in august?"))
    assert content == "August was 10."
    assert "Reviewing the answer..." in reasoning
    assert len(model.calls) == 2
    assert sent_user_texts(model, 1) == ["revenue in august?", "Your answer missed the question. Redo it."]
    assert [m.content for m in model.calls[1] if m.type == "ai"] == ["September was 12."]
    assert len(check.decider.provider.calls) == 1  # the second answer is not checked again


@pytest.mark.parametrize("path", PATHS)
def test_agent_second_attempt_that_fails_keeps_the_first_answer(monkeypatch, path):
    check = sense_check({"makes_sense": ("no", 0.95)})
    agent, _ = build(monkeypatch, ["September was 12.", ""], checks=[check])
    assert run(agent, path, user("revenue in august?"))[0] == "September was 12."


@pytest.mark.parametrize("path", PATHS)
def test_agent_doubt_and_check_together(monkeypatch, path):
    check = sense_check({"makes_sense": ("no", 0.95)})
    agent, model = build(monkeypatch, ["bad", "good"], doubts=[store_doubt("product")], checks=[check])
    assert run(agent, path, user("did acerola sell?"))[0] == "good"
    # the check judges the user's own words, without the directive...
    assert check.decider.provider.calls[0][0]["message"] == "did acerola sell?"
    # ...and the second attempt keeps the directive the doubt added.
    assert sent_user_texts(model, 1)[0] == "did acerola sell?\n\n[acerola is a product]"


@pytest.mark.parametrize("path", PATHS)
def test_agent_engine_down_behaves_as_if_there_were_no_deciders(monkeypatch, path):
    class DownDecider(Decider):
        provider = FakeDeciderProvider(error=TimeoutError())
        instruction = "ok?"
        options = ["yes", "no"]

    agent, model = build(monkeypatch, ["The total is 4."],
                         doubts=[Doubt(DownDecider, on_sure=lambda d, c: "[never]")],
                         checks=[Check(DownDecider, redo=lambda d, a: "redo")])
    assert run(agent, path, user("add 2 and 2"))[0] == "The total is 4."
    assert sent_user_texts(model, 0) == ["add 2 and 2"] and len(model.calls) == 1


def test_agent_redo_label_can_be_silenced(monkeypatch):
    agent, _ = build(monkeypatch, ["bad", "good"], check_redo_label="",
                     checks=[sense_check({"makes_sense": ("no", 0.95)})])
    content, reasoning = run(agent, "stream", user("q"))
    assert content == "good" and reasoning == ""


def test_agent_usage_covers_both_attempts(monkeypatch):
    agent, model = build(monkeypatch, [], checks=[sense_check({"makes_sense": ("no", 0.95)})])
    usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    model.script = [AIMessage(content="bad", usage_metadata=usage),
                    AIMessage(content="good", usage_metadata=usage)]
    assert agent.invoke(user("q")).usage == {
        "prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}


@pytest.mark.parametrize("attr, value", [("doubts", [store_decider("store")]), ("checks", ["sense"])])
def test_agent_rejects_wrong_entries_at_definition(attr, value):
    with pytest.raises(AixonError, match=f"{attr} takes"):
        type("BrokenAgent", (ToolAgent,), {"name": f"broken_{attr}", "tools": [],
                                           "llm": LLM("fake-1", provider="fake"), attr: value})


class _SecondCallFails(RecordingChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if len(self.calls) >= 1:
            self.calls.append(list(messages))
            raise RuntimeError("provider down on the second attempt")
        return super()._generate(messages, stop, run_manager, **kwargs)


@pytest.mark.parametrize("path", PATHS)
def test_agent_second_attempt_that_raises_keeps_the_first_answer(monkeypatch, path):
    agent, _ = build(monkeypatch, [], checks=[sense_check({"makes_sense": ("no", 0.95)})])
    model = _SecondCallFails(script=[AIMessage(content="September was 12.")])
    model.calls = []
    monkeypatch.setattr(type(agent).llm.__class__, "chat_model", property(lambda self: model))
    assert run(agent, path, user("revenue in august?"))[0] == "September was 12."
    assert len(model.calls) == 2


def test_async_helpers_match_the_sync_ones():
    from aixon.decider import aapply_doubts, afailed_checks, aresolve_doubts

    async def go():
        doubt, check = store_doubt("store"), sense_check({"makes_sense": ("no", 0.9)})
        applied = await aapply_doubts(user("acerola?"), [doubt])
        untouched = await aapply_doubts([], [doubt])
        outcomes = await aresolve_doubts("acerola?", [doubt])
        failures = await afailed_checks("q", "a", [check])
        return applied, untouched, outcomes, failures

    applied, untouched, outcomes, failures = asyncio.run(go())
    assert applied[-1].content.endswith("[acerola is a store]") and untouched == []
    assert outcomes[0].outcome == "sure" and len(failures) == 1


def test_check_hooks_that_raise_do_not_fail_the_answer():
    def boom(*args):
        raise RuntimeError("hook bug")

    base = sense_check({"makes_sense": ("no", 0.95)})
    assert failed_checks("q", "a", [Check(base.decider, redo=boom)]) == []
    assert failed_checks("q", "a", [Check(base.decider, redo=base.redo, context=boom)]) == []
