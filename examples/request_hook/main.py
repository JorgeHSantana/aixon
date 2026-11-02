"""request_hook (0.1.29) — a aplicação decide, por request, o modelo imposto,
a identidade e a recusa, a partir dos cabeçalhos.

Ciclo de cada request: parse → resolve (agente) → hook → escopos → agente.

    cd examples/request_hook
    OPENAI_API_KEY=... PYTHONPATH=../.. python main.py
"""

from __future__ import annotations

from aixon import LLM, LLMAgent
from aixon.exceptions import RequestRejected
from aixon.server.context import RequestContext, RequestDecision
from aixon.server.server import Server

# x-tier → (modelo, provedor) imposto à árvore inteira do agente.
TIERS: dict[str, tuple[str, str]] = {
    "fast": ("gpt-5.4-mini", "openai"),
    "deep": ("gpt-5.4", "openai"),
}


class HelperAgent(LLMAgent):
    """Responde perguntas curtas."""

    name = "helper"
    llm = LLM("gpt-5.4-mini")
    prompt = "You are a concise helper."


class AnalystAgent(LLMAgent):
    """Analisa com mais cuidado."""

    name = "analyst"
    llm = LLM("gpt-5.4-mini")
    prompt = "You are a careful analyst."


def tier_hook(ctx: RequestContext) -> RequestDecision:
    """Sem ``x-user`` a request é recusada com 401; ``x-tier`` escolhe o
    modelo imposto (ausente ou desconhecido: vale o modelo da classe)."""
    user = ctx.headers.get("x-user")
    if not user:
        raise RequestRejected("x-user header is required", status=401, code="no_user")
    return RequestDecision(
        impose=TIERS.get(ctx.headers.get("x-tier", "")),
        user_email=user,
        trace_user_id=user,
    )


server = Server(request_hook=tier_hook)

if __name__ == "__main__":
    server.serve()
