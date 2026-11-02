"""request_hook (0.1.30) — a aplicação decide, por request, o modelo imposto,
a identidade e a recusa, a partir dos cabeçalhos. O hook é uma classe:
``HeaderIdentityHook`` lê a identidade; ``check`` recusa; ``imposition`` escolhe
o modelo.

Ciclo de cada request: parse → resolve (agente) → hook → escopos → agente.

    cd examples/request_hook
    OPENAI_API_KEY=... PYTHONPATH=../.. python main.py
"""

from __future__ import annotations

from aixon import LLM, LLMAgent
from aixon.exceptions import RequestRejected
from aixon.server.context import HeaderIdentityHook, RequestContext
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


class TierHook(HeaderIdentityHook):
    """``x-user`` é a identidade (e-mail e usuário do trace); sem ele a request
    é recusada com 401. ``x-tier`` escolhe o modelo imposto (ausente ou
    desconhecido: vale o modelo da classe)."""

    def __init__(self) -> None:
        super().__init__(email_header="x-user")

    def check(self, ctx: RequestContext) -> None:
        if not ctx.headers.get("x-user", "").strip():
            raise RequestRejected("x-user header is required", status=401, code="no_user")

    def imposition(self, ctx: RequestContext) -> tuple[str, str] | None:
        return TIERS.get(ctx.headers.get("x-tier", ""))


server = Server(request_hook=TierHook())

if __name__ == "__main__":
    server.serve()
