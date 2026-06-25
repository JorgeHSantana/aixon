"""Offline demo backends — so the example runs with **zero** API keys.

This module registers a custom LLM `Provider` named ``demo`` and ships a
deterministic `Embedding`. Both are real aixon extension points:

* `DemoProvider` / `register_provider` — plug any vendor (or a fake) into the
  framework. When ``OPENAI_API_KEY`` is set, the agents use ``gpt-4o-mini``
  instead (see ``llm_config.py``); when it is not, they fall back to ``demo``
  and everything still runs.
* `DemoEmbedding` — a hashing-based `Embedding` so `KnowledgeRetriever` can do
  real vector search offline. Swap in `OpenAIEmbedding` for production.

The demo chat model is intentionally simple and deterministic:

* With **no tools bound** (an `LLMAgent` like triage) it classifies the user's
  message into one word: ``orders`` or ``knowledge``.
* With **tools bound** (a `ToolAgent`) it calls the first tool with the user's
  message, then turns the tool's result into a final answer — so the retriever
  and connector are genuinely exercised end to end, offline.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aixon import Embedding
from aixon.providers.base import Provider, register_provider

DEMO_MODEL = "demo-1"
DEMO_PROVIDER = "demo"

# Words that route a request to the orders specialist; everything else is a
# knowledge-base question.
_ORDER_WORDS = (
    "order", "orders", "refund", "ship", "shipping", "deliver", "delivery",
    "track", "tracking", "cancel", "invoice", "payment", "charge", "package",
)


def _last_human(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(messages):
        if getattr(m, "type", "") == "human":
            return str(getattr(m, "content", "") or "")
    # Fall back to the last message of any kind.
    return str(getattr(messages[-1], "content", "") or "") if messages else ""


def _last_tool_result(messages: Sequence[BaseMessage]) -> Optional[str]:
    for m in reversed(messages):
        if getattr(m, "type", "") == "tool":
            return str(getattr(m, "content", "") or "")
    return None


def _classify(text: str) -> str:
    low = text.lower()
    return "orders" if any(w in low for w in _ORDER_WORDS) else "knowledge"


class DemoChatModel(BaseChatModel):
    """Deterministic, offline chat model (a real LangChain ``BaseChatModel``).

    ``tools_meta`` is a list of ``(tool_name, first_arg_name)`` captured in
    ``bind_tools``; an empty list means "no tools" (the LLMAgent path)."""

    tools_meta: list = []

    @property
    def _llm_type(self) -> str:
        return "demo"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "DemoChatModel":
        meta: list = []
        for t in tools:
            name = getattr(t, "name", None) or getattr(t, "__name__", "tool")
            try:
                arg = next(iter(t.args.keys()))
            except Exception:
                arg = "query"
            meta.append((name, arg))
        # Return a bound copy (the LangChain convention) rather than mutating.
        return self.model_copy(update={"tools_meta": meta})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        human = _last_human(messages)

        if self.tools_meta:
            tool_out = _last_tool_result(messages)
            if tool_out is not None:
                msg = AIMessage(content=f"Here's what I found:\n\n{tool_out}")
            else:
                name, arg = self.tools_meta[0]
                msg = AIMessage(
                    content="",
                    tool_calls=[{"name": name, "args": {arg: human}, "id": "call_1"}],
                )
        else:
            msg = AIMessage(content=_classify(human))

        return ChatResult(generations=[ChatGeneration(message=msg)])


class DemoProvider(Provider):
    """Provider named ``demo`` whose ``build()`` returns a `DemoChatModel`."""

    name = DEMO_PROVIDER
    env_key = ""  # no API key needed

    def build(self, model: str, **params: Any) -> BaseChatModel:
        return DemoChatModel()  # params (e.g. temperature) intentionally ignored


# Self-register on import, exactly like the built-in providers.
register_provider(DemoProvider())


class DemoEmbedding(Embedding):
    """Deterministic hashing embedding — no network, no API key.

    Each token is hashed into one of ``DIM`` buckets; the vector is the bucket
    histogram. Good enough for the example's keyword-overlap ranking, and a
    drop-in for `OpenAIEmbedding` (same `Embedding` interface)."""

    DIM = 256

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM
            v[bucket] += 1.0
        return v

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def __repr__(self) -> str:
        return "DemoEmbedding(offline)"
