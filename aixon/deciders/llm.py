"""Any aixon ``LLM`` as a decision engine.

    from aixon import LLM
    from aixon.deciders import LLMDeciderProvider

    class ToneDecider(Decider):
        provider = LLMDeciderProvider(LLM("gpt-4o-mini", temperature=0))
        ...

It exists for two things: comparing a decision engine against a chat model on
the same questions (``aixon decide eval``), and as a stand-in where no engine
is available. It is NOT the recommended engine: it is slower, costs a full
completion, and — the important part — its chances are the model's own
ESTIMATE written into JSON, not a calibrated probability. Treat ``certainty``
from this provider as a rough signal and calibrate the threshold with
``aixon decide eval`` before trusting ``Decision.sure``.

Not registered under a name: it needs an ``LLM`` instance, so it is always
passed as ``provider = LLMDeciderProvider(...)``. ``timeout`` is the LLM's own
(set it on the ``LLM``); the ``Decider``'s ``timeout`` is not applied here.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Mapping

from aixon.deciders.base import DeciderProvider, ProviderReply, Question, RawAnswer
from aixon.message import Message

if TYPE_CHECKING:
    from aixon.llm import LLM

_SYSTEM = (
    "You are a decision engine. You answer closed questions about a context. "
    "You never explain. Reply with ONE JSON object and nothing else: for each "
    "question id, an object {\"choice\": <one of the allowed answers, copied "
    "exactly>, \"chances\": {<every allowed answer>: <probability 0..1>}}. "
    "The chances of one question sum to 1."
)

_JSON = re.compile(r"\{.*\}", re.S)


def _render(context: Mapping[str, Any], questions: Mapping[str, Question]) -> str:
    lines = ["CONTEXT (JSON):", json.dumps(dict(context), ensure_ascii=False), "", "QUESTIONS:"]
    for name, q in questions.items():
        lines.append(f"- id: {name}")
        lines.append(f"  question: {q.instruction}")
        if q.kind == "scale":
            lines.append("  allowed answers (an ordered scale, lowest first):")
            lines.extend(f"    - {label}" for label in q.labels)
        else:
            lines.append("  allowed answers:")
            for label, criterion in (q.options or {}).items():
                lines.append(f"    - {label}" + (f": {criterion}" if criterion else ""))
    return "\n".join(lines)


def _parse(text: str, questions: Mapping[str, Question]) -> dict[str, RawAnswer]:
    found = _JSON.search(text or "")
    if not found:
        raise ValueError("the model did not return JSON")
    data = json.loads(found.group(0))
    answers: dict[str, RawAnswer] = {}
    for name, q in questions.items():
        raw = data[name]
        choice = str(raw["choice"])
        if choice not in q.labels:
            raise ValueError(f"the model chose '{choice}', which is not an allowed answer")
        chances = {label: max(0.0, float((raw.get("chances") or {}).get(label, 0.0)))
                   for label in q.labels}
        total = sum(chances.values())
        if total <= 0:
            chances = {label: (1.0 if label == choice else 0.0) for label in q.labels}
        else:
            chances = {label: value / total for label, value in chances.items()}
        score = None
        if q.kind == "scale":
            score = sum(i * chances[label] for i, label in enumerate(q.labels))
        answers[name] = RawAnswer(choice=choice, chances=chances, score=score)
    return answers


class LLMDeciderProvider(DeciderProvider):
    name = "llm"
    hosted = True
    capabilities = frozenset({"choice", "yes_no", "scale", "batch"})

    def __init__(self, llm: "LLM") -> None:
        self.llm = llm

    def decide(
        self,
        context: Mapping[str, Any],
        questions: Mapping[str, Question],
        *,
        model: str | None,
        timeout: float,
    ) -> ProviderReply:
        reply = self.llm.complete([
            Message(role="system", content=_SYSTEM),
            Message(role="user", content=_render(context, questions)),
        ])
        return ProviderReply(answers=_parse(reply.content, questions),
                             usage=reply.usage, model=self.llm.model)
