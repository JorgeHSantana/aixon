"""The deciders of the example: four closed questions about language.

Nothing here names an engine. Every decider leaves ``provider`` unset, so the
engine comes from the ``AIXON_DECIDER_PROVIDER`` environment variable:

    AIXON_DECIDER_PROVIDER=jev   TYPESAFE_API_KEY=...   hosted, no training
    AIXON_DECIDER_PROVIDER=laya  (pip install aixon[laya])  local, open weights
    (unset)                      main.py registers an offline stand-in

Swapping engines is that one variable; no class below changes.
"""

from __future__ import annotations

import re

from aixon import Check, Decider, Decision, Doubt

# The chain's stores. "olive" and "harbor" are also ordinary words — and
# "olive" is a product on the shelves. That is the doubt.
STORES = {"olive": "Olive Street store", "harbor": "Harbor store"}


def mask_values(context: dict) -> dict:
    """What leaves the process when the engine is hosted: every amount becomes
    ``#``. Dates and store names stay — the engine needs them to judge whether
    the answer is about the right thing; it never needs the figures."""
    masked = dict(context)
    if "answer" in masked:
        masked["answer"] = re.sub(r"\$ ?[\d.,]+|\b\d[\d.,]*\b(?!\s*(?:st|nd|rd|th)\b)", "#", masked["answer"])
    return masked


class HouseDecider(Decider, abstract=True):
    """Defaults shared by the example's deciders."""

    certainty = 0.8
    timeout = 3.0
    redact = None  # user messages here carry nothing to mask


class StoreOrProductDecider(HouseDecider):
    """Is the word a store of the chain, or something on the shelves?"""

    instruction = (
        "The user of a grocery chain's assistant wrote 'message'. Decide what "
        "the word in 'word' refers to in that message."
    )
    options = {
        "store": "a store of the chain: a place, 'the' + name, the store's revenue or staff",
        "product": "an item for sale, a category or an ingredient",
    }
    switch = "STORE_OR_PRODUCT"  # STORE_OR_PRODUCT=0 turns just this one off


class UpsetDecider(HouseDecider):
    """Is the user complaining (plainly or with irony)?"""

    instruction = "Is the user upset with the assistant in 'message'?"
    options = {
        "yes": "a complaint, an accusation of error, or irony ('great job, wrong again')",
        "no": "a plain request or a neutral follow-up",
    }
    certainty = 0.6


class AnswersTheQuestionDecider(HouseDecider):
    """Does the answer talk about what was asked? (Never: is the number right.)"""

    instruction = (
        "'message' is what the user asked and 'answer' is the assistant's reply, "
        "with every amount replaced by '#'. Does the answer address what was "
        "asked: same period, same store, same subject?"
    )
    options = {
        "yes": "same period, store and subject as the question, or an honest 'I could not find it'",
        "no": "another period, another store, another subject, or no answer at all",
    }
    redact = staticmethod(mask_values)


class UrgencyDecider(HouseDecider):
    """An ordinal grade instead of a choice."""

    instruction = "How urgent is the user's request in 'message'?"
    scale = ["can wait", "today", "right now"]


# ---- how the agent uses them -------------------------------------------------

def find_store_word(message: str) -> dict | None:
    """Pure, cheap, no engine: is a store's name loose in the message, without
    the word 'store' next to it? If not, there is no doubt to settle."""
    lowered = message.lower()
    for word, store in STORES.items():
        if re.search(rf"\b{word}\b", lowered) and not re.search(rf"\b{word}\b\s+(street\s+)?store", lowered):
            return {"word": word, "store": store}
    return None


def store_is_meant(decision: Decision, case: dict) -> str | None:
    if decision.choice != "store":
        return None  # a product: nothing to add, the model reads it as written
    return f"[The user means the {case['store']}. Filter by that store.]"


def ask_store_or_product(decision: Decision, case: dict) -> str:
    return (f"[The word '{case['word']}' may be the {case['store']} or the product. Do not look "
            f"anything up yet. Reply with one short question asking which one the user means.]")


def acknowledge(decision: Decision, case: dict) -> str | None:
    if decision.choice != "yes":
        return None
    return "[The user is upset. Open with one short sentence acknowledging it, then answer.]"


def redo(decision: Decision, answer: str) -> str:
    return ("Your answer does not address what was asked (check the period and the store). "
            "Answer again, about exactly what was asked.")


DOUBTS = [
    Doubt(StoreOrProductDecider, applies=find_store_word,
          on_sure=store_is_meant, on_unsure=ask_store_or_product),
    Doubt(UpsetDecider, on_sure=acknowledge),
]

CHECKS = [Check(AnswersTheQuestionDecider, redo=redo)]
