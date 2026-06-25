"""TriageAgent — a pure `LLMAgent` (no tools) that classifies the request.

It answers with a single word — ``orders`` or ``knowledge`` — which the
`SupportOrchestrator` reads to route to the right specialist. Being an
`LLMAgent`, it talks straight to the model (no tool-calling loop).

It is ``hidden`` so it doesn't show up in the public agent list / server model
list — only the orchestrator is a public entry point.
"""

from __future__ import annotations

from aixon import LLMAgent

from llm_config import make_llm


class TriageAgent(LLMAgent):
    name = "triage"
    hidden = True
    description = "Classifies a support request as 'orders' or 'knowledge'."
    llm = make_llm(temperature=0.0)
    prompt = (
        "You are a support request router. Read the user's message and reply "
        "with exactly ONE word, lowercase, no punctuation: 'orders' if it is "
        "about an order, shipping, refund, payment or invoice; otherwise "
        "'knowledge'."
    )
