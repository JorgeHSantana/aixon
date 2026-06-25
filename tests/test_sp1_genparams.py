from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aixon.llm import LLM
from aixon.message import Message
from aixon.runtime import (
    GENERATION_PARAMS,
    current_generation_params,
    generation_params,
)
from tests._fakes import FakeChatModel


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


def test_llm_forwards_generation_params_to_model():
    captured: dict = {}

    class Rec(FakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            captured.update(kwargs)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    llm = LLM("fake-1", provider="fake")
    llm._chat_model = Rec()  # inject; skip provider.build
    with generation_params({"temperature": 0.1, "thought_stream_mode": "content"}):
        out = llm.complete([Message(role="user", content="hi")])
    assert out.content == "ok"
    assert captured.get("temperature") == 0.1
    assert "thought_stream_mode" not in captured  # filtered by allow-list


def test_llm_no_params_does_not_break():
    llm = LLM("fake-1", provider="fake")
    out = llm.complete([Message(role="user", content="hi")])
    assert out.role == "assistant"
