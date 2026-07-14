from __future__ import annotations

from aixon.llm import LLM
from aixon.message import Message
from aixon.runtime import (
    GENERATION_PARAMS,
    current_generation_params,
    generation_params,
)


def test_allowlist_filters_non_generation_keys():
    with generation_params({"temperature": 0.1, "thought_stream_mode": "content",
                            "stream_options": {"include_usage": True}}):
        cur = current_generation_params()
    assert cur == {"temperature": 0.1}
    assert "temperature" in GENERATION_PARAMS
    assert "thought_stream_mode" not in GENERATION_PARAMS


def test_contextvar_resets_after_block():
    with generation_params({"temperature": 0.5}):
        assert current_generation_params() == {"temperature": 0.5}
    assert current_generation_params() == {}


def test_llm_forwards_generation_params_to_model(monkeypatch):
    """UPDATED (final-review bind-path unification): LLMAgent's
    ``_bound_model()`` (used by ``complete``/``stream``) now merges
    per-request params into ``Provider.build()`` as constructor kwargs via
    ``request_chat_model()`` — the same path ``ToolAgent`` already used —
    instead of ``.bind()``-ing them at invoke time onto an already-built
    model. Assert on what ``build()`` received, not on a manually-injected
    model's ``_generate()`` runtime kwargs (the old test bypassed
    ``provider.build`` entirely by setting ``llm._chat_model`` directly,
    which is no longer how per-request params reach the model)."""
    from tests._fakes import FakeProvider

    captured: list[dict] = []
    original_build = FakeProvider.build

    def recording_build(self, model, **params):
        captured.append(params)
        return original_build(self, model, **params)

    monkeypatch.setattr(FakeProvider, "build", recording_build)

    llm = LLM("fake-1", provider="fake")
    with generation_params({"temperature": 0.1, "thought_stream_mode": "content"}):
        out = llm.complete([Message(role="user", content="hi")])
    assert out.content == "(done)"
    assert captured[-1].get("temperature") == 0.1
    assert "thought_stream_mode" not in captured[-1]  # filtered by allow-list


def test_llm_no_params_does_not_break():
    llm = LLM("fake-1", provider="fake")
    out = llm.complete([Message(role="user", content="hi")])
    assert out.role == "assistant"
