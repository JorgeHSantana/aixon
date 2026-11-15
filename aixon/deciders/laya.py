"""Laya: local, open-weights decision engine (``pip install aixon[laya]``).

Runs in-process on CPU or GPU; nothing leaves the machine, so no ``redact`` is
required. The ``laya`` package (and torch behind it) is imported on the FIRST
decision, never at ``import aixon``.

- Model: the ``Decider``'s ``model`` — a Hugging Face id or a local checkpoint
  directory — else ``LAYA_MODEL``, else the package default. A fine-tuned
  checkpoint is just a path here; training is done with Laya's own tooling.
- Device: ``LAYA_DEVICE`` (default ``cpu``).
- Each (model, device) is loaded once per process and reused; loading and
  inference are serialized by a lock because a checkpoint is not safe to run
  from two threads at once.

``timeout`` is not enforced: a local forward pass cannot be interrupted. Keep
the context short instead (the engine truncates at ``LAYA_MAX_LEN`` tokens,
default 1024).
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, Mapping

from aixon.deciders.base import DeciderProvider, ProviderReply, Question, register_decider_provider
from aixon.deciders.systemone import from_wire, to_wire, usage_from_wire

# (model, device) -> engine with ``system_one(state, questions, **kw)``. Swapped in tests.
Loader = Callable[[str | None, str], Any]


def _load(model: str | None, device: str) -> Any:
    import laya

    return laya.load(model, device=device) if model else laya.load(device=device)


class LayaProvider(DeciderProvider):
    """``provider = "laya"``."""

    name = "laya"
    hosted = False
    capabilities = frozenset({"choice", "yes_no", "scale", "batch"})
    default_model = None

    def __init__(self, loader: Loader | None = None) -> None:
        self._loader = loader
        self._engines: dict[tuple[str | None, str], Any] = {}
        self._lock = threading.Lock()

    def unavailable(self) -> str | None:
        if self._loader is not None:
            return None
        import importlib.util

        return None if importlib.util.find_spec("laya") else "not_installed"

    def _engine(self, model: str | None, device: str) -> Any:
        key = (model, device)
        if key not in self._engines:
            self._engines[key] = (self._loader or _load)(model, device)
        return self._engines[key]

    def decide(
        self,
        context: Mapping[str, Any],
        questions: Mapping[str, Question],
        *,
        model: str | None,
        timeout: float,
    ) -> ProviderReply:
        model = model or os.getenv("LAYA_MODEL") or self.default_model
        device = os.getenv("LAYA_DEVICE") or "cpu"
        max_len = int(os.getenv("LAYA_MAX_LEN") or 1024)
        wire = {name: to_wire(q) for name, q in questions.items()}
        with self._lock:
            reply = self._engine(model, device).system_one(dict(context), wire, max_len=max_len)
        answers = reply["answers"]
        return ProviderReply(
            answers={name: from_wire(answers[name], q) for name, q in questions.items()},
            usage=usage_from_wire(reply.get("usage")),
            model=model,
        )


register_decider_provider(LayaProvider())
