"""Every decision leaves one span; tracing can never break a decision."""
import aixon.observability as obs
from aixon import Decider
from aixon.deciders import FakeDeciderProvider


def _decider(provider):
    class ToneDecider(Decider):
        instruction = "Tone?"
        options = ["neutral", "upset"]
        redact = staticmethod(lambda ctx: {**ctx, "secret": "#"})

    ToneDecider.provider = provider
    return ToneDecider


def _capture(monkeypatch):
    spans = []
    monkeypatch.setattr(obs, "record_decision",
                        lambda name, **kw: spans.append({"name": name, **kw}))
    return spans


def test_a_decision_records_the_redacted_input_and_the_outcome(monkeypatch):
    spans = _capture(monkeypatch)
    _decider(FakeDeciderProvider({"tone": ("upset", 0.9)})).decide({"message": "x", "secret": "42"})
    (span,) = spans
    assert span["name"] == "tone"
    assert span["input"] == {"message": "x", "secret": "#"}  # what left the process, not the raw context
    assert span["output"] == {"choice": "upset", "certainty": 0.9, "sure": True,
                              "chances": {"neutral": 0.1, "upset": 0.9}}
    assert span["metadata"]["provider"] == "fake" and span["metadata"]["threshold"] == 0.8


def test_a_failure_records_a_failed_span_with_the_reason(monkeypatch):
    spans = _capture(monkeypatch)
    _decider(FakeDeciderProvider(error=TimeoutError())).decide({"message": "x"})
    (span,) = spans
    assert span["failed"] is True and span["output"] == {"no_decision": "TimeoutError"}


def test_off_and_keyless_deciders_leave_no_span(monkeypatch):
    spans = _capture(monkeypatch)
    _decider(FakeDeciderProvider(unavailable="no_key")).decide({"message": "x"})
    assert spans == []


def test_record_decision_is_a_noop_without_langfuse():
    obs.record_decision("tone", input={}, output={})  # conftest clears the LANGFUSE_* env


def test_record_decision_swallows_tracing_failures(monkeypatch):
    import sys
    import types

    def boom():
        raise RuntimeError("langfuse down")

    monkeypatch.setattr(obs, "_ensure_handler", lambda: object())
    monkeypatch.setitem(sys.modules, "langfuse", types.SimpleNamespace(get_client=boom))
    obs.record_decision("tone", input={}, output={})  # must not raise


def test_record_decision_opens_one_span(monkeypatch):
    import contextlib
    import sys
    import types

    opened = []

    class Client:
        @contextlib.contextmanager
        def start_as_current_observation(self, **kw):
            opened.append(kw)
            yield

    monkeypatch.setattr(obs, "_ensure_handler", lambda: object())
    monkeypatch.setitem(sys.modules, "langfuse", types.SimpleNamespace(get_client=Client))
    obs.record_decision("tone", input={"m": 1}, output={"choice": "upset"}, metadata={"latency_ms": 3})
    obs.record_decision("tone", input={}, output={"no_decision": "TimeoutError"}, failed=True)
    assert opened[0] == {"name": "decider:tone", "as_type": "span", "input": {"m": 1},
                         "output": {"choice": "upset"}, "metadata": {"latency_ms": 3}, "level": None}
    assert opened[1]["level"] == "WARNING"


def test_a_decision_with_tokens_is_a_generation_with_usage_and_cost(monkeypatch):
    import contextlib
    import sys
    import types

    opened = []

    class Client:
        @contextlib.contextmanager
        def start_as_current_observation(self, **kw):
            opened.append(kw)
            yield

    monkeypatch.setattr(obs, "_ensure_handler", lambda: object())
    monkeypatch.setitem(sys.modules, "langfuse", types.SimpleNamespace(get_client=Client))
    usage = {"prompt_tokens": 415, "completion_tokens": 64, "total_tokens": 479}
    obs.record_decision("tone", input={}, output={}, model="jev-1.14.0", usage=usage, cost=0.0000201)
    obs.record_decision("tone", input={}, output={}, model="jev-1.14.0", usage=usage)   # price unknown
    priced, unpriced = opened
    assert priced["as_type"] == "generation" and priced["model"] == "jev-1.14.0"
    assert priced["usage_details"] == {"input": 415, "output": 64, "total": 479}
    assert priced["cost_details"] == {"total": 0.0000201}
    assert "cost_details" not in unpriced   # Langfuse prices it from its own model table, if any


def test_the_decider_hands_model_usage_and_cost_to_the_trace(monkeypatch):
    from aixon.deciders import ProviderReply

    class Priced(FakeDeciderProvider):
        def price_per_mtok(self):
            return (1.0, 1.0)

        def decide(self, context, questions, *, model, timeout):
            reply = super().decide(context, questions, model=model, timeout=timeout)
            return ProviderReply(reply.answers, {"prompt_tokens": 90, "completion_tokens": 10,
                                                 "total_tokens": 100}, "priced-1")

    spans = _capture(monkeypatch)
    _decider(Priced()).decide({"message": "x"})
    (span,) = spans
    assert (span["model"], span["usage"]["total_tokens"], span["cost"]) == ("priced-1", 100, 0.0001)
    assert span["metadata"]["cost_usd"] == 0.0001
