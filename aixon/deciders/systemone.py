"""The "System One" wire format, shared by the Jev API and the ``laya`` package.

Both engines take ``{question_id: {"type", "instructions", "criteria"}}`` and
answer ``{question_id: {...}}`` with three shapes:

- ``choice``: ``{"choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}}``
- ``noul`` (yes/no): ``{"noul": 0.91}`` — always P(true)
- ``score`` (scale): ``{"score": 1.76, "confidence": 0.73, "probabilities": {"0": 0.01, "1": 0.24, ...}}``
  with probabilities keyed by the level's INDEX.

These two functions are the whole translation; they are pure so both providers
are tested without a network or a checkpoint.
"""

from __future__ import annotations

from typing import Any, Mapping

from aixon.deciders.base import NO, YES, Question, RawAnswer


def to_wire(question: Question) -> dict[str, Any]:
    """Neutral ``Question`` -> one System One question definition."""
    if question.kind == "scale":
        return {"type": "score", "instructions": question.instruction,
                "criteria": list(question.scale or ())}
    options = question.options or {}
    if question.kind == "yes_no":
        wire: dict[str, Any] = {"type": "noul", "instructions": question.instruction}
        criteria = {"true": options.get(YES), "false": options.get(NO)}
        if all(criteria.values()):
            wire["criteria"] = criteria
        return wire
    return {"type": "choice", "instructions": question.instruction,
            "criteria": dict(options)}


def from_wire(answer: Mapping[str, Any], question: Question) -> RawAnswer:
    """One System One answer -> neutral ``RawAnswer``.

    Raises ``KeyError``/``ValueError`` on a malformed answer; the caller turns
    that into "no decision"."""
    if question.kind == "yes_no":
        p = min(1.0, max(0.0, float(answer["noul"])))
        return RawAnswer(choice=YES if p >= 0.5 else NO,
                         chances={YES: p, NO: 1.0 - p})
    if question.kind == "scale":
        levels = question.labels
        by_index = answer.get("probabilities") or {}
        chances = {label: float(by_index.get(str(i), 0.0)) for i, label in enumerate(levels)}
        score = float(answer["score"])
        if any(chances.values()):
            choice = max(levels, key=lambda label: chances[label])
        else:
            choice = levels[min(len(levels) - 1, max(0, round(score)))]
        return RawAnswer(choice=choice, chances=chances,
                         certainty=_optional_float(answer.get("confidence")), score=score)
    choice = str(answer["choice"])
    if choice not in question.labels:
        raise ValueError(f"engine chose '{choice}', which is not an option")
    chances = {k: float(v) for k, v in (answer.get("probabilities") or {}).items()
               if k in question.labels}
    return RawAnswer(choice=choice, chances=chances,
                     certainty=_optional_float(answer.get("confidence")))


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def usage_from_wire(usage: Mapping[str, Any] | None) -> dict[str, int] | None:
    """System One usage (``input_tokens``/``output_tokens``) -> the OpenAI
    shape the rest of aixon uses."""
    if not usage:
        return None
    prompt = int(usage.get("input_tokens", 0))
    completion = int(usage.get("output_tokens", 0))
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion}
