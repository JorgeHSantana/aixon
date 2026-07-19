# tests/test_predicted_outputs.py
"""#6 — Predicted Outputs (OpenAI) no retry do ReflectiveAgent.

No retry, ~o texto todo da tentativa anterior tende a se repetir; passar
``answer.content`` como ``prediction`` corta a latência dos trechos inalterados
(decodificação especulativa). O valor viaja num ContextVar (``aixon.runtime``)
ativado pelo ReflectiveAgent SÓ em volta da chamada de retry do worker; o LLM
o anexa como kwarg de invocação apenas quando o provider declara
``supports_prediction`` (OpenAI) — demais providers: no-op silencioso."""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aixon.message import Message
from aixon.runtime import current_prediction, prediction_scope
from tests._fakes import FakeChatModel, FakeProvider, make_llm
from tests.test_reflective import USER, make_reflective


# ── runtime ──────────────────────────────────────────────────────────────────

def test_prediction_scope_roundtrip():
    assert current_prediction() is None
    with prediction_scope("prev answer"):
        assert current_prediction() == "prev answer"
        with prediction_scope(None):            # None = no-op explícito
            assert current_prediction() is None
    assert current_prediction() is None


# ── LLM: kwarg só com suporte do provider ────────────────────────────────────

class RecordingModel(FakeChatModel):
    calls: list = []

    def _generate(self, messages: list[BaseMessage],
                  stop: Optional[list[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        type(self).calls.append(kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


def test_llm_passes_prediction_when_provider_supports(monkeypatch):
    monkeypatch.setattr(FakeProvider, "supports_prediction", True, raising=False)
    llm = make_llm()
    llm._chat_model = RecordingModel()
    RecordingModel.calls = []
    with prediction_scope("texto anterior"):
        llm.complete([Message(role="user", content="q")])
    assert RecordingModel.calls[-1].get("prediction") == {
        "type": "content", "content": "texto anterior"}


def test_llm_omits_prediction_without_provider_support():
    llm = make_llm()                             # fake: sem supports_prediction
    llm._chat_model = RecordingModel()
    RecordingModel.calls = []
    with prediction_scope("texto anterior"):
        llm.complete([Message(role="user", content="q")])
    assert "prediction" not in RecordingModel.calls[-1]


def test_openai_provider_declares_support():
    from aixon.providers.openai import OpenAIProvider
    assert getattr(OpenAIProvider, "supports_prediction", False) is True


# ── ReflectiveAgent: prediction ativa SÓ nos retries ─────────────────────────

def test_reflective_sets_prediction_on_retry_rounds():
    from aixon.agent import Agent
    from aixon.message import Chunk
    from aixon.registry import get_registry

    seen: list = []

    def invoke(self, messages):
        seen.append(current_prediction())
        return Message(role="assistant", content=f"v{len(seen)}")

    def stream(self, messages):
        yield Chunk(content=invoke(self, messages).content)
        yield Chunk(done=True)

    type("PredworkerAgent", (Agent,),
         {"name": "predworker", "invoke": invoke, "stream": stream})
    worker = get_registry().resolve("predworker")
    r = make_reflective("predref", worker, ["1. Ruim.", "APROVADO"])
    out = r.invoke(USER)
    assert out.content == "v2"
    # rodada 1: sem prediction; rodada 2 (retry): a resposta anterior inteira
    assert seen == [None, "v1"]
