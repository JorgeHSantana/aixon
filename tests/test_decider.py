"""Decider core: the declaration, the three outcomes (sure / unsure / no
decision), batching, redaction and the never-raises contract."""
import asyncio

import pytest

from aixon import AixonError, Decider, Decision, NamingError, decide_many
from aixon.decider import adecide_many
from aixon.deciders import FakeDeciderProvider, RawAnswer, register_decider_provider


def make(provider, **attrs):
    """A fresh ToneDecider on *provider* (class attributes overridable)."""
    body = {
        "provider": provider,
        "instruction": "What is the tone of the user's message?",
        "options": {"neutral": "a plain request", "upset": "a complaint or irony"},
        **attrs,
    }
    return type("ToneDecider", (Decider,), body)


# ---- the three outcomes ----------------------------------------------------

def test_sure_decision_carries_choice_certainty_and_every_chance():
    Tone = make(FakeDeciderProvider({"tone": ("upset", 0.93)}))
    d = Tone.decide({"message": "wrong again, great job"})
    assert isinstance(d, Decision)
    assert (d.choice, d.certainty, d.sure) == ("upset", 0.93, True)
    assert d.chances == {"neutral": 0.07, "upset": 0.93}
    assert d.chance("neutral") == 0.07 and d.chance("unknown") == 0.0
    assert (d.decider, d.provider) == ("tone", "fake")


def test_below_the_threshold_the_decision_is_unsure():
    Tone = make(FakeDeciderProvider({"tone": ("upset", 0.6)}))
    d = Tone.decide({"message": "hm"})
    assert d.choice == "upset" and d.sure is False


def test_threshold_is_per_decider():
    Tone = make(FakeDeciderProvider({"tone": ("upset", 0.6)}), certainty=0.55)
    assert Tone.decide({"message": "hm"}).sure is True


def test_certainty_falls_back_to_the_chance_of_the_choice():
    raw = RawAnswer(choice="upset", chances={"neutral": 0.2, "upset": 0.8})  # no certainty
    Tone = make(FakeDeciderProvider({"tone": raw}))
    assert Tone.decide({"message": "x"}).certainty == 0.8


# ---- no decision: never raises ---------------------------------------------

@pytest.mark.parametrize("error", [TimeoutError(), ConnectionError("down"), KeyError("answers"),
                                   ValueError("malformed")])
def test_an_engine_failure_is_no_decision_not_an_exception(error):
    provider = FakeDeciderProvider(error=error)
    assert make(provider).decide({"message": "x"}) is None
    assert len(provider.calls) == 1


def test_unavailable_provider_is_not_even_called():
    provider = FakeDeciderProvider(unavailable="no_key")
    assert make(provider).decide({"message": "x"}) is None
    assert provider.calls == []


def test_switch_turns_one_decider_off(monkeypatch):
    provider = FakeDeciderProvider()
    Tone = make(provider, switch="TONE")
    assert Tone.enabled() and Tone.decide({"message": "x"}) is not None
    for off in ("0", "off", "OFF", "false", "no"):
        monkeypatch.setenv("TONE", off)
        assert not Tone.enabled() and Tone.decide({"message": "x"}) is None
    monkeypatch.setenv("TONE", "1")
    assert Tone.decide({"message": "x"}) is not None
    assert len(provider.calls) == 2


def test_a_redact_that_raises_is_no_decision():
    def broken(context):
        raise RuntimeError("mask failed")

    provider = FakeDeciderProvider()
    assert make(provider, redact=broken).decide({"message": "x"}) is None
    assert provider.calls == []  # nothing left the process


def test_provider_from_the_environment(monkeypatch):
    register_decider_provider(_named("envfake", {"tone": "upset"}))
    Tone = make(None)
    assert Tone.decide({"message": "x"}) is None  # no provider declared, none in the env
    monkeypatch.setenv("AIXON_DECIDER_PROVIDER", "envfake")
    assert Tone.decide({"message": "x"}).choice == "upset"
    monkeypatch.setenv("AIXON_DECIDER_PROVIDER", "does-not-exist")
    assert Tone.decide({"message": "x"}) is None  # a typo in the env never raises


def _named(name, script=None, **kw):
    provider = FakeDeciderProvider(script, **kw)
    provider.name = name
    return provider


# ---- redaction -------------------------------------------------------------

def test_redact_runs_before_the_context_leaves_and_does_not_touch_the_original():
    provider = FakeDeciderProvider(hosted=True)
    Tone = make(provider, redact=lambda ctx: {**ctx, "answer": "R$ #"})
    original = {"message": "x", "answer": "R$ 1.234,56"}
    Tone.decide(original)
    assert provider.calls[0][0] == {"message": "x", "answer": "R$ #"}
    assert original["answer"] == "R$ 1.234,56"


def test_hosted_provider_requires_redact_to_be_declared():
    with pytest.raises(AixonError, match="hosted"):
        make(FakeDeciderProvider(hosted=True))
    make(FakeDeciderProvider(hosted=True), redact=None)  # explicit "nothing to mask" is accepted


def test_hosted_provider_from_the_env_without_redact_is_no_decision(monkeypatch):
    provider = _named("hostedenv", hosted=True)
    register_decider_provider(provider)
    monkeypatch.setenv("AIXON_DECIDER_PROVIDER", "hostedenv")
    assert make(None).decide({"message": "x"}) is None
    assert provider.calls == []


# ---- question kinds --------------------------------------------------------

def test_yes_no_decider():
    class MakesSenseDecider(Decider):
        provider = FakeDeciderProvider({"makes_sense": ("no", 0.9)})
        instruction = "Does the answer address what the user asked?"
        options = {"yes": "it does", "no": "it talks about something else"}

    assert MakesSenseDecider.question().kind == "yes_no"
    d = MakesSenseDecider.decide({"message": "a", "answer": "b"})
    assert d.choice == "no" and d.chance("yes") == pytest.approx(0.1)


def test_scale_decider():
    class UpsetLevelDecider(Decider):
        provider = FakeDeciderProvider({"upset_level": RawAnswer(
            choice="clearly", chances={"not at all": 0.01, "slightly": 0.24, "clearly": 0.75},
            certainty=0.73, score=1.74)})
        instruction = "How upset is the user?"
        scale = ["not at all", "slightly", "clearly"]

    assert UpsetLevelDecider.question().kind == "scale"
    d = UpsetLevelDecider.decide({"message": "x"})
    assert (d.choice, d.score, d.certainty, d.sure) == ("clearly", 1.74, 0.73, False)


def test_options_may_be_just_names():
    Tone = make(FakeDeciderProvider(), options=["neutral", "upset"])
    assert Tone.options == {"neutral": None, "upset": None}


# ---- several decisions, one request ----------------------------------------

def _pair(provider):
    class ToneDecider(Decider):
        instruction = "Tone?"
        options = ["neutral", "upset"]

    class IronyDecider(Decider):
        instruction = "Is the message ironic?"
        options = ["yes", "no"]

    ToneDecider.provider = IronyDecider.provider = provider
    return ToneDecider, IronyDecider


def test_decide_many_sends_one_request_for_deciders_that_share_a_provider():
    provider = FakeDeciderProvider({"tone": "upset", "irony": ("yes", 0.7)})
    Tone, Irony = _pair(provider)
    out = decide_many({"message": "great, wrong again"}, [Tone, Irony])
    assert out[Tone].choice == "upset" and out[Irony].choice == "yes" and not out[Irony].sure
    assert provider.calls == [({"message": "great, wrong again"}, ("tone", "irony"))]


def test_a_provider_without_batch_gets_one_request_per_question():
    provider = FakeDeciderProvider({"tone": "upset"}, batch=False)
    Tone, Irony = _pair(provider)
    out = decide_many({"message": "x"}, [Tone, Irony])
    assert out[Tone] is not None and out[Irony] is not None
    assert [names for _, names in provider.calls] == [("tone",), ("irony",)]


def test_one_group_failing_does_not_take_the_other_down():
    Tone = make(FakeDeciderProvider(error=TimeoutError()))

    class IronyDecider(Decider):
        provider = FakeDeciderProvider({"irony": "yes"})
        instruction = "Ironic?"
        options = ["yes", "no"]

    out = decide_many({"message": "x"}, [Tone, IronyDecider])
    assert out[Tone] is None and out[IronyDecider].choice == "yes"


def test_decide_many_with_nothing_to_ask():
    assert decide_many({"message": "x"}, []) == {}


def test_async_twins():
    provider = FakeDeciderProvider({"tone": "upset"})
    Tone, Irony = _pair(provider)

    async def go():
        one = await Tone.adecide({"message": "x"})
        many = await adecide_many({"message": "x"}, [Tone, Irony])
        return one, many

    one, many = asyncio.run(go())
    assert one.choice == "upset" and set(many) == {Tone, Irony}


# ---- declaration mistakes fail when the class is defined --------------------

def test_name_must_end_with_decider():
    with pytest.raises(NamingError, match="must end with 'Decider'"):
        type("Tone", (Decider,), {"instruction": "x", "options": ["a", "b"]})


@pytest.mark.parametrize("attrs, message", [
    ({"instruction": ""}, "`instruction` is required"),
    ({"options": None}, "exactly one of `options` or `scale`"),
    ({"scale": ["a", "b"]}, "exactly one of `options` or `scale`"),
    ({"options": ["only"]}, "at least two options"),
    ({"options": "ab"}, "must be a dict"),
    ({"options": None, "scale": ["one"]}, "at least two levels"),
    ({"certainty": 0}, "`certainty` must be in"),
    ({"certainty": 1.5}, "`certainty` must be in"),
    ({"timeout": 0}, "`timeout` must be positive"),
    ({"redact": "mask"}, "`redact` must be a function"),
    ({"provider": "does-not-exist"}, "No decider provider registered"),
])
def test_declaration_mistakes(attrs, message):
    with pytest.raises(AixonError, match=message):
        make(attrs.pop("provider", FakeDeciderProvider()), **attrs)


def test_too_many_options_for_the_provider():
    provider = FakeDeciderProvider()
    provider.max_options = 2
    with pytest.raises(AixonError, match="at most 2 options"):
        make(provider, options=["a", "b", "c"])


def test_question_kind_the_provider_does_not_answer():
    provider = FakeDeciderProvider()
    provider.capabilities = frozenset({"choice"})
    with pytest.raises(AixonError, match="does not answer 'yes_no'"):
        make(provider, options=["yes", "no"])


def test_a_decider_is_not_instantiated():
    with pytest.raises(AixonError, match="declarative"):
        make(FakeDeciderProvider())()


def test_abstract_base_skips_validation_and_children_inherit():
    class HouseDecider(Decider, abstract=True):
        provider = FakeDeciderProvider({"tone": "upset"})
        certainty = 0.9
        redact = None

    class ToneDecider(HouseDecider):
        instruction = "Tone?"
        options = ["neutral", "upset"]

    assert ToneDecider.certainty == 0.9 and ToneDecider.decide({"message": "x"}).choice == "upset"


def test_name_defaults_to_snake_case_and_can_be_set():
    class StoreOrProductDecider(Decider):
        provider = FakeDeciderProvider()
        instruction = "x"
        options = ["store", "product"]

    assert StoreOrProductDecider.name == "store_or_product"
    assert make(FakeDeciderProvider(), name="custom").name == "custom"
