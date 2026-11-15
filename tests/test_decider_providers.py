"""Decision providers: the System One wire format (Jev and Laya), the LLM
provider and the registry. No network, no checkpoint, no torch."""
import json
import sys

import pytest
from langchain_core.messages import AIMessage

from aixon import AixonError, Decider, get_decider_provider, register_decider_provider
from aixon.deciders import (
    DeciderProvider,
    FakeDeciderProvider,
    JevProvider,
    LayaProvider,
    LLMDeciderProvider,
    Question,
)
from aixon.deciders.systemone import from_wire, to_wire, usage_from_wire
from tests._fakes import FakeChatModel, make_llm

CHOICE = Question("Tone?", options={"neutral": "plain", "upset": "complaint"})
YES_NO = Question("Upset?", options={"yes": "complaint or irony", "no": "plain request"})
SCALE = Question("How upset?", scale=("not at all", "slightly", "clearly", "furious"))

# What the real Jev API answered for these three questions (probe of 2026-11-15).
JEV_REPLY = {
    "model": "jev-1.14.0",
    "answers": {
        "tone": {"type": "choice", "choice": "upset", "confidence": 1.0,
                 "probabilities": {"neutral": 0.0, "upset": 1.0}},
        "upset": {"type": "noul", "noul": 0.91},
        "level": {"type": "score", "score": 1.76, "confidence": 0.73,
                  "legend": {"0": "not at all", "1": "slightly", "2": "clearly", "3": "furious"},
                  "probabilities": {"0": 0.01, "1": 0.24, "2": 0.74, "3": 0.01}},
    },
    "usage": {"input_tokens": 415, "output_tokens": 64},
}
QUESTIONS = {"tone": CHOICE, "upset": YES_NO, "level": SCALE}


# ---- System One wire format -------------------------------------------------

def test_to_wire_for_each_kind():
    assert to_wire(CHOICE) == {"type": "choice", "instructions": "Tone?",
                               "criteria": {"neutral": "plain", "upset": "complaint"}}
    assert to_wire(YES_NO) == {"type": "noul", "instructions": "Upset?",
                               "criteria": {"true": "complaint or irony", "false": "plain request"}}
    assert to_wire(SCALE) == {"type": "score", "instructions": "How upset?",
                              "criteria": ["not at all", "slightly", "clearly", "furious"]}


def test_yes_no_without_criteria_sends_none():
    assert to_wire(Question("Upset?", options={"yes": None, "no": None})) == {
        "type": "noul", "instructions": "Upset?"}


def test_from_wire_for_each_kind():
    answers = JEV_REPLY["answers"]
    tone = from_wire(answers["tone"], CHOICE)
    assert (tone.choice, tone.certainty, tone.chances) == ("upset", 1.0, {"neutral": 0.0, "upset": 1.0})
    upset = from_wire(answers["upset"], YES_NO)
    assert upset.choice == "yes" and upset.chances == {"yes": 0.91, "no": pytest.approx(0.09)}
    assert upset.certainty is None  # the Decider falls back to the chance of the choice
    level = from_wire(answers["level"], SCALE)
    assert (level.choice, level.score, level.certainty) == ("clearly", 1.76, 0.73)
    assert level.chances == {"not at all": 0.01, "slightly": 0.24, "clearly": 0.74, "furious": 0.01}


def test_from_wire_no_below_half():
    assert from_wire({"noul": 0.2}, YES_NO).choice == "no"


def test_from_wire_scale_without_probabilities_rounds_the_score():
    assert from_wire({"score": 2.6}, SCALE).choice == "furious"


def test_from_wire_refuses_a_choice_that_is_not_an_option():
    with pytest.raises(ValueError, match="not an option"):
        from_wire({"choice": "angry"}, CHOICE)


def test_usage_from_wire():
    assert usage_from_wire({"input_tokens": 415, "output_tokens": 64}) == {
        "prompt_tokens": 415, "completion_tokens": 64, "total_tokens": 479}
    assert usage_from_wire(None) is None


# ---- Jev ---------------------------------------------------------------------

def _jev(reply=JEV_REPLY):
    sent = []

    def transport(url, headers, body, timeout):
        sent.append({"url": url, "headers": headers, "body": json.loads(body), "timeout": timeout})
        if isinstance(reply, Exception):
            raise reply
        return reply

    return JevProvider(transport), sent


def test_jev_request_and_reply(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k-test")
    provider, sent = _jev()
    reply = provider.decide({"message": "wrong again"}, QUESTIONS, model=None, timeout=3.0)
    call = sent[0]
    assert call["url"] == "https://api.typesafe.ai/v1/systemone"
    assert call["headers"]["Authorization"] == "Bearer k-test" and call["timeout"] == 3.0
    assert call["body"]["model"] == "jev-latest"
    assert call["body"]["state"] == {"message": "wrong again"}
    assert set(call["body"]["questions"]) == {"tone", "upset", "level"}
    assert reply.model == "jev-1.14.0" and reply.usage["total_tokens"] == 479
    assert reply.answers["tone"].choice == "upset"


def test_jev_model_and_base_url_from_the_environment(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setenv("JEV_MODEL", "jev-1.14.0")
    monkeypatch.setenv("JEV_BASE_URL", "https://jev.internal/v1/")
    provider, sent = _jev()
    provider.decide({}, {"tone": CHOICE}, model=None, timeout=1)
    assert sent[0]["url"] == "https://jev.internal/v1/systemone"
    assert sent[0]["body"]["model"] == "jev-1.14.0"
    provider.decide({}, {"tone": CHOICE}, model="jev-pinned", timeout=1)
    assert sent[1]["body"]["model"] == "jev-pinned"  # the Decider's model wins


def test_jev_without_a_key_is_unavailable_and_sends_nothing(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    provider, sent = _jev()
    assert provider.unavailable() == "no_key"

    class ToneDecider(Decider):
        instruction = "Tone?"
        options = ["neutral", "upset"]
        redact = None

    ToneDecider.provider = provider
    assert ToneDecider.decide({"message": "x"}) is None and sent == []


@pytest.mark.parametrize("failure", [TimeoutError(), OSError("HTTP 429"), {"answers": {}},
                                     {"error": "quota"}])
def test_jev_failures_become_no_decision(monkeypatch, failure):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    provider, _ = _jev(failure)

    class ToneDecider(Decider):
        instruction = "Tone?"
        options = ["neutral", "upset"]
        redact = None

    ToneDecider.provider = provider
    assert ToneDecider.decide({"message": "x"}) is None


def test_jev_is_hosted_and_limits_options():
    provider = get_decider_provider("jev")
    assert provider.hosted and provider.max_options == 255
    with pytest.raises(AixonError, match="hosted"):
        type("OpenDecider", (Decider,), {"provider": "jev", "instruction": "x", "options": ["a", "b"]})
    with pytest.raises(AixonError, match="at most 255"):
        type("WideDecider", (Decider,), {"provider": "jev", "instruction": "x", "redact": None,
                                         "options": [f"o{i}" for i in range(256)]})


# ---- Laya --------------------------------------------------------------------

class _Engine:
    def __init__(self):
        self.calls = []

    def system_one(self, state, questions, max_len=None):
        self.calls.append((state, questions, max_len))
        return {"answers": {name: JEV_REPLY["answers"][name] for name in questions},
                "usage": {"input_tokens": 120, "output_tokens": 0}}


def test_laya_loads_each_checkpoint_once_and_speaks_system_one(monkeypatch):
    loads, engine = [], _Engine()

    def loader(model, device):
        loads.append((model, device))
        return engine

    provider = LayaProvider(loader)
    monkeypatch.setenv("LAYA_DEVICE", "cpu")
    for _ in range(3):
        reply = provider.decide({"message": "x"}, QUESTIONS, model="/models/judge-v3", timeout=3)
    assert loads == [("/models/judge-v3", "cpu")]
    assert engine.calls[0][1]["upset"]["type"] == "noul" and engine.calls[0][2] == 1024
    assert reply.answers["level"].choice == "clearly" and reply.model == "/models/judge-v3"
    provider.decide({"message": "x"}, {"tone": CHOICE}, model=None, timeout=3)
    assert loads[-1] == (None, "cpu")  # another checkpoint: loaded on its own


def test_laya_model_from_the_environment(monkeypatch):
    loads = []
    provider = LayaProvider(lambda model, device: loads.append((model, device)) or _Engine())
    monkeypatch.setenv("LAYA_MODEL", "org/laya-pt")
    monkeypatch.setenv("LAYA_DEVICE", "mps")
    monkeypatch.setenv("LAYA_MAX_LEN", "512")
    provider.decide({}, {"tone": CHOICE}, model=None, timeout=3)
    assert loads == [("org/laya-pt", "mps")]


def test_laya_not_installed_is_unavailable_and_importing_aixon_never_imports_it(monkeypatch):
    assert "laya" not in sys.modules and "torch" not in sys.modules
    monkeypatch.setitem(sys.modules, "laya", None)  # find_spec -> not importable
    provider = get_decider_provider("laya")
    assert not provider.hosted
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert provider.unavailable() == "not_installed"

    class LocalToneDecider(Decider):
        provider = "laya"          # local: no `redact` needed
        instruction = "Tone?"
        options = ["neutral", "upset"]

    assert LocalToneDecider.decide({"message": "x"}) is None


# ---- LLM as a decision engine -------------------------------------------------

def _llm_provider(monkeypatch, text):
    llm = make_llm()
    fake = FakeChatModel(script=[AIMessage(content=text)])
    monkeypatch.setattr(type(llm), "chat_model", property(lambda self: fake))
    return LLMDeciderProvider(llm)


def test_llm_provider_parses_json_and_normalises_chances(monkeypatch):
    provider = _llm_provider(monkeypatch, 'Here you go:\n{"tone": {"choice": "upset", '
                                          '"chances": {"neutral": 1, "upset": 3}}}')
    reply = provider.decide({"message": "x"}, {"tone": CHOICE}, model=None, timeout=3)
    assert reply.answers["tone"].choice == "upset"
    assert reply.answers["tone"].chances == {"neutral": 0.25, "upset": 0.75}
    assert reply.model == "fake-1"


def test_llm_provider_scale_gets_a_score(monkeypatch):
    provider = _llm_provider(monkeypatch, json.dumps({"level": {"choice": "clearly", "chances": {
        "not at all": 0, "slightly": 0.5, "clearly": 0.5, "furious": 0}}}))
    assert provider.decide({}, {"level": SCALE}, model=None, timeout=3).answers["level"].score == 1.5


def test_llm_provider_without_chances_is_all_in_on_the_choice(monkeypatch):
    provider = _llm_provider(monkeypatch, '{"tone": {"choice": "neutral"}}')
    assert provider.decide({}, {"tone": CHOICE}, model=None, timeout=3).answers["tone"].chances == {
        "neutral": 1.0, "upset": 0.0}


@pytest.mark.parametrize("text", ["I think the user is upset.", '{"tone": {"choice": "angry"}}',
                                  '{"other": {}}'])
def test_llm_provider_bad_output_becomes_no_decision(monkeypatch, text):
    class ToneDecider(Decider):
        instruction = "Tone?"
        options = {"neutral": "plain", "upset": "complaint"}
        redact = None

    ToneDecider.provider = _llm_provider(monkeypatch, text)
    assert ToneDecider.decide({"message": "x"}) is None


def test_llm_provider_is_hosted():
    with pytest.raises(AixonError, match="hosted"):
        type("OpenDecider", (Decider,), {"provider": LLMDeciderProvider(make_llm()),
                                         "instruction": "x", "options": ["a", "b"]})


# ---- registry -----------------------------------------------------------------

def test_registry_roundtrip_and_unknown_name():
    class MineProvider(FakeDeciderProvider):
        name = "mine"

    mine = MineProvider()
    mine.name = "mine"
    register_decider_provider(mine)
    assert get_decider_provider("mine") is mine
    with pytest.raises(AixonError, match="No decider provider registered as 'nope'"):
        get_decider_provider("nope")


def test_a_provider_needs_a_name():
    class Nameless(DeciderProvider):
        def decide(self, context, questions, *, model, timeout):
            raise NotImplementedError

    with pytest.raises(AixonError, match="non-empty `name`"):
        register_decider_provider(Nameless())
