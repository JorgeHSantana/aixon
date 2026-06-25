"""One place to choose the model.

`make_llm()` returns a real `LLM` handle:

* If ``OPENAI_API_KEY`` is set  -> ``gpt-4o-mini`` (real inference).
* Otherwise                     -> the offline ``demo`` provider, so the whole
                                   example runs with no keys and no network.

The agents never branch on this — they just call ``make_llm()`` in their class
body. Swap the model here and every agent follows.
"""

from __future__ import annotations

import os

from aixon import LLM


def using_real_llm() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def make_llm(*, temperature: float = 0.2) -> LLM:
    if using_real_llm():
        return LLM("gpt-4o-mini", temperature=temperature)
    # Importing the module registers the 'demo' provider (idempotent).
    import providers.demo  # noqa: F401

    return LLM("demo-1", provider="demo", temperature=temperature)
