"""Decider — closed questions with a measured certainty, offline.

    cd examples/decider
    PYTHONPATH=../.. python main.py

Four parts:

1. One decision, three outcomes: sure, unsure, no decision.
2. Several decisions about one context in a single request.
3. A ``ToolAgent`` with ``doubts`` (before the model) and ``checks`` (after it).
4. Calibrating the certainty threshold with ``aixon decide eval``.

With no environment set, an OFFLINE stand-in plays the decision engine (a few
keyword rules) and a scripted model plays the LLM: no key, no network. To put
a real engine behind the very same deciders:

    AIXON_DECIDER_PROVIDER=jev TYPESAFE_API_KEY=... PYTHONPATH=../.. python main.py
    AIXON_DECIDER_PROVIDER=laya PYTHONPATH=../.. python main.py   # pip install aixon[laya]

The LLM stays scripted either way, so the example never needs a chat-model key.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aixon import LLM, ToolAgent, decide_many, register_decider_provider
from aixon.deciders import FakeDeciderProvider, Question
from aixon.message import Message
from aixon.providers.base import Provider, register_provider

# ── offline stand-in for the decision engine ────────────────────────────────
# A real engine reads the language; this one matches keywords, just enough to
# show the three outcomes. It answers (option, certainty).


def offline_engine(context: Mapping[str, Any], name: str, question: Question):
    message = str(context.get("message", "")).lower()
    if name == "store_or_product":
        if any(clue in message for clue in ("manager", "open", "revenue at", "make")):
            return "store", 0.96
        if any(clue in message for clue in ("jars", "oil", "stock", "price", "200g")):
            return "product", 0.94
        return "store", 0.55  # could be either: below the 0.8 threshold
    if name == "upset":
        return ("yes", 0.97) if any(clue in message for clue in ("wrong", "again", "great job")) \
            else ("no", 0.98)
    if name == "answers_the_question":
        asked_august = "august" in message
        said_september = "september" in str(context.get("answer", "")).lower()
        return ("no", 0.95) if asked_august and said_september else ("yes", 0.95)
    if name == "urgency":
        return ("right now", 0.9) if "now" in message else ("can wait", 0.7)
    return question.labels[0], 1.0


offline = FakeDeciderProvider(offline_engine)
offline.name = "offline"
register_decider_provider(offline)
os.environ.setdefault("AIXON_DECIDER_PROVIDER", "offline")

from deciders import (  # noqa: E402  (after the provider is chosen)
    CHECKS,
    DOUBTS,
    AnswersTheQuestionDecider,
    StoreOrProductDecider,
    UpsetDecider,
    UrgencyDecider,
)

# ── scripted chat model (offline, deterministic) ────────────────────────────


class ScriptedChatModel(BaseChatModel):
    """Plays the LLM: answers by reading the directives it was sent. It gets
    the period wrong on purpose the first time, so the check has work to do."""

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        last = str(messages[-1].content)
        print(f"      model received: {last!r}")
        if "Reply with one short question" in last:
            text = "Do you mean the Olive Street store or the product olive?"
        elif last.startswith("Your answer does not address"):
            text = "Sorry about that. In August the Harbor store made $ 41,200."
        elif "August" in last:
            text = "In September the Harbor store made $ 38,900."
        else:
            text = "Olive oil has a 31% margin."
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class ScriptedProvider(Provider):
    name = "scripted"
    env_key = "SCRIPTED_API_KEY"

    def build(self, model: str, **params: Any) -> ScriptedChatModel:
        return ScriptedChatModel()


register_provider(ScriptedProvider())


class GroceryAgent(ToolAgent):
    """Answers a grocery chain's managers."""

    name = "grocery"
    llm = LLM("scripted-1", provider="scripted")
    prompt = "You answer the managers of a grocery chain."
    tools: list = []
    doubts = DOUBTS      # before the model: read the user's message
    checks = CHECKS      # after the model: does the answer make sense? redo once


def show(decision) -> str:
    if decision is None:
        return "no decision (engine off, no key, timeout): carry on without it"
    state = "sure" if decision.sure else "UNSURE"
    return f"{decision.choice}  certainty {decision.certainty:.2f}  {state}  chances {decision.chances}"


def main() -> None:
    engine = os.environ["AIXON_DECIDER_PROVIDER"]
    print(f"decision engine: {engine}\n")

    print("1. One decision, three outcomes")
    for message in ("Who is the manager at Olive?",
                    "Do we have olive in stock?",
                    "How much did olive sell in August?"):
        decision = StoreOrProductDecider.decide({"message": message, "word": "olive"})
        print(f"   {message!r}\n      -> {show(decision)}")
    os.environ["STORE_OR_PRODUCT"] = "0"  # the decider's own switch
    print(f"   switched off -> {show(StoreOrProductDecider.decide({'message': 'x', 'word': 'olive'}))}")
    del os.environ["STORE_OR_PRODUCT"]

    print("\n2. Several decisions, one request")
    context = {"message": "Great job, the Harbor report is wrong again. I need it now."}
    for decider, decision in decide_many(context, [UpsetDecider, UrgencyDecider]).items():
        print(f"   {decider.name}: {show(decision)}")

    print("\n3. An agent with doubts and checks")
    agent = GroceryAgent()
    for question in ("How much did olive sell in August?",
                     "What is the margin on olive oil?",
                     "Great job, wrong again. How much did the Harbor store make in August?"):
        print(f"   user: {question!r}")
        answer = agent.invoke([Message(role="user", content=question)])
        if answer.reasoning:
            print(f"      reasoning: {answer.reasoning!r}")
        print(f"   assistant: {answer.content!r}\n")

    print("   What the hosted engine would have received for the check (amounts masked):")
    sent = AnswersTheQuestionDecider.redact({"message": "Harbor in August?",
                                             "answer": "In August the Harbor store made $ 41,200."})
    print(f"      {sent}")

    print("\n4. Calibrate the threshold before turning a decider on:")
    print("   PYTHONPATH=../.. aixon decide eval deciders:StoreOrProductDecider examples.jsonl")


if __name__ == "__main__":
    main()
