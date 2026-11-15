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
