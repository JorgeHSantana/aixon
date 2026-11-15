"""Jev (TypeSafe AI): hosted, zero-shot decision engine.

One HTTP request answers N typed questions about one context in ~250 ms. No
SDK: the call is plain ``urllib`` so this provider adds no dependency.

- Key: ``TYPESAFE_API_KEY``. Without it the provider reports ``"no_key"`` and
  every ``Decider`` on it returns ``None`` — nothing is sent, nothing raises.
- Endpoint: ``JEV_BASE_URL`` (default ``https://api.typesafe.ai/v1``).
- Model: the ``Decider``'s ``model``, else ``JEV_MODEL``, else ``jev-latest``.

Hosted: the context leaves your process, so a ``Decider`` here must declare
``redact`` (see ``aixon.decider``).
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Mapping

from aixon.deciders.base import DeciderProvider, ProviderReply, Question, register_decider_provider
from aixon.deciders.systemone import from_wire, to_wire, usage_from_wire

DEFAULT_BASE_URL = "https://api.typesafe.ai/v1"

# (url, headers, body, timeout) -> parsed JSON. Swapped in tests.
Transport = Callable[[str, dict[str, str], bytes, float], Mapping[str, Any]]


def _http_post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> Mapping[str, Any]:
    import urllib.request

    request = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class JevProvider(DeciderProvider):
    """``provider = "jev"``."""

    name = "jev"
    hosted = True
    capabilities = frozenset({"choice", "yes_no", "scale", "batch"})
    max_options = 255
    default_model = "jev-latest"

    def __init__(self, transport: Transport | None = None) -> None:
        self._transport: Transport = transport or _http_post

    def unavailable(self) -> str | None:
        return None if os.getenv("TYPESAFE_API_KEY") else "no_key"

    def decide(
        self,
        context: Mapping[str, Any],
        questions: Mapping[str, Question],
        *,
        model: str | None,
        timeout: float,
    ) -> ProviderReply:
        base = (os.getenv("JEV_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        body = json.dumps({
            "model": model or os.getenv("JEV_MODEL") or self.default_model,
            "state": dict(context),
            "questions": {name: to_wire(q) for name, q in questions.items()},
        }).encode()
        headers = {
            "Authorization": "Bearer " + os.environ["TYPESAFE_API_KEY"],
            "Content-Type": "application/json",
        }
        reply = self._transport(base + "/systemone", headers, body, timeout)
        answers = reply["answers"]
        return ProviderReply(
            answers={name: from_wire(answers[name], q) for name, q in questions.items()},
            usage=usage_from_wire(reply.get("usage")),
            model=reply.get("model"),
        )


register_decider_provider(JevProvider())
