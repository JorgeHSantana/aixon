"""Reflective Review — the `ReflectiveAgent` evaluator-optimizer loop, offline.

Wraps a toy "writer" `Agent` with a `ReflectiveAgent`: a judge LLM checks
every answer against an objective rubric ("does it cite a source?") and, if
it doesn't, sends the critique back to the writer for another attempt — up
to `max_rounds` times. Both the writer and the judge are scripted here so the
whole example runs with **no API key and no network call**:

    cd examples/reflective_review
    python main.py

Expected: the judge REJECTS the writer's first answer ("missing the
source"), the writer retries citing one, and the judge APPROVES — you will
see both rounds on the reasoning channel, then the final, approved answer.
See README.md for the full expected output.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aixon import LLM, ReflectiveAgent
from aixon.agent import Agent
from aixon.message import Chunk, Message
from aixon.providers.base import Provider, register_provider

# ── scripted judge model (offline, deterministic) ───────────────────────────
# Shape mirrors the BaseChatModel doubles aixon's own hermetic tests use: a
# `script` list replayed one reply per call (see docs/agents.md, "LLM —
# declaring a language model", for the real `register_provider` extension
# point this borrows).


class ScriptedChatModel(BaseChatModel):
    """Offline chat model that replays `script`, one reply per call."""

    script: list = []
    _idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        i = self._idx
        content = self.script[i] if i < len(self.script) else self.script[-1]
        object.__setattr__(self, "_idx", i + 1)
        msg = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=msg)])


class ScriptedProvider(Provider):
    """Provider named ``scripted`` whose ``build()`` returns a fresh
    `ScriptedChatModel`. `LLM.chat_model` builds it lazily on first access and
    caches it, so a caller can grab `llm.chat_model` right after constructing
    the `LLM` and set `.script` on it — exactly like aixon's own tests do."""

    name = "scripted"
    env_key = ""  # no API key needed

    def build(self, model: str, **params: Any) -> ScriptedChatModel:
        return ScriptedChatModel()


register_provider(ScriptedProvider())


def scripted_llm(script: list[str], **params: Any) -> LLM:
    """An `LLM` backed by the offline ``scripted`` provider, pre-loaded with
    `script` (one reply per call — here, the judge's successive verdicts)."""
    llm = LLM("scripted-1", provider="scripted", **params)
    llm.chat_model.script = list(script)
    return llm


# ── scripted writer (the "generator" half of the loop) ──────────────────────
# A real generator would be an LLMAgent/ToolAgent; this one is scripted so the
# example is deterministic and needs no key. `ReflectiveAgent.agent` accepts
# any Agent — class or instance (see docs/agents.md).


class DraftWriterAgent(Agent):
    """Answers a geography question — first without a source, then with one."""

    name = "draft-writer"
    hidden = True
    description = "Toy generator: drafts an answer, cites a source on retry."

    ANSWERS = [
        "Fortaleza is the capital of Ceará.",
        "Fortaleza is the capital of Ceará (source: IBGE).",
    ]
    calls: list = []

    def invoke(self, messages: list[Message]) -> Message:
        type(self).calls.append(list(messages))
        i = min(len(type(self).calls) - 1, len(self.ANSWERS) - 1)
        return Message(role="assistant", content=self.ANSWERS[i])

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        yield Chunk(content=self.invoke(messages).content)
        yield Chunk(done=True)


# ── the ReflectiveAgent itself ───────────────────────────────────────────────


class ReviewedWriterAgent(ReflectiveAgent):
    """Reviews `DraftWriterAgent`'s answer against an objective rubric before
    returning it — the evaluator-optimizer loop documented in docs/agents.md."""

    name = "reviewed-writer"
    description = "Drafts an answer and reviews it against a citation rubric."
    agent = DraftWriterAgent
    judge_llm = scripted_llm(
        [
            "1. The answer does not cite a source for the fact stated.",
            "APROVADO",
        ]
    )
    judge_rubric = "The answer must cite the source of the fact it states."
    max_rounds = 3


def main() -> None:
    question = [Message(role="user", content="What is the capital of Ceará?")]
    print(f"> {question[0].content}\n")

    content = ""
    for chunk in ReviewedWriterAgent().stream(question):
        if chunk.reasoning:
            print(f"[reasoning] {chunk.reasoning}", end="")
        if chunk.content:
            content += chunk.content

    print(f"\nFinal answer: {content}")
    print(
        f"\nDraftWriterAgent was called {len(DraftWriterAgent.calls)} time(s) — "
        "the judge rejected round 1 (no source) and approved round 2."
    )


if __name__ == "__main__":
    main()
