"""Langfuse (#26) — integração de primeira classe, runnable dos dois jeitos.

SEM as envs do Langfuse: roda offline com um handler de demonstração no
MESMO ponto de acoplamento (configure hook do langchain-core) e imprime os
eventos capturados — prova de que o handler alcança grafo + modelos sem
mudar call-site nenhum.

COM LANGFUSE_PUBLIC_KEY/SECRET_KEY (e opcionalmente LANGFUSE_HOST): o mesmo
run vira um trace real "Suporte" no dashboard, com user_id/session_id e uma
generation por turno de modelo (modelo real + usage).

    cd examples/langfuse
    PYTHONPATH=../.. python main.py
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aixon import LLM, ToolAgent
from aixon.message import Message
from aixon.observability import _handler_scope, langfuse_enabled, observe_request
from aixon.providers.base import Provider, register_provider

# ── modelo scriptado (offline) ───────────────────────────────────────────────


class ScriptedChatModel(BaseChatModel):
    script: list = []
    _idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(self, messages: list[BaseMessage],
                  stop: Optional[list[str]] = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        i = self._idx
        msg = self.script[i] if i < len(self.script) else AIMessage(content="(fim)")
        object.__setattr__(self, "_idx", i + 1)
        return ChatResult(generations=[ChatGeneration(message=msg)])


class ScriptedProvider(Provider):
    name = "scripted"
    env_key = "SCRIPTED_API_KEY"

    def build(self, model: str, **params: Any) -> ScriptedChatModel:
        return ScriptedChatModel(script=[
            AIMessage(content="", tool_calls=[
                {"name": "status_pedido", "args": {"pedido": "42"}, "id": "c1"}]),
            AIMessage(content="O pedido 42 está em transporte."),
        ])


register_provider(ScriptedProvider())


def status_pedido(pedido: str) -> str:
    """Status de um pedido pelo número."""
    return f"pedido {pedido}: em transporte"


class SuporteAgent(ToolAgent):
    name = "Suporte"
    description = "Consulta pedidos."
    llm = LLM("scripted-1", provider="scripted")
    prompt = "Você atende clientes; use as tools."
    tools = [status_pedido]


class DemoHandler(BaseCallbackHandler):
    """No lugar do CallbackHandler do Langfuse — mesmo acoplamento."""

    def on_chain_start(self, serialized, inputs, **kwargs):
        print("  [handler] chain start")

    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("  [handler] model turn")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"  [handler] tool start: {input_str}")


def main() -> None:
    agent = SuporteAgent()
    pergunta = [Message(role="user", content="Cadê meu pedido 42?")]

    if langfuse_enabled():
        print("Langfuse configurado — enviando trace real 'Suporte'...")
        # No deploy real o Server faz isto por request, com a identidade
        # vinda dos headers do Open WebUI.
        with observe_request("Suporte", user_id="cliente@exemplo.com",
                             session_id="chat-demo") as active:
            answer = agent.invoke(pergunta)
        print(f"ativo={active} | resposta: {answer.content}")
        print("Veja o trace no dashboard (nome 'Suporte', user cliente@exemplo.com).")
    else:
        print("Sem envs do Langfuse — demonstrando o mecanismo offline:")
        with _handler_scope(DemoHandler()):
            answer = agent.invoke(pergunta)
        print(f"resposta: {answer.content}")
        print("\nOs eventos acima chegaram ao handler SEM nenhum call-site "
              "anexar callbacks — é o mesmo caminho que o Langfuse usa.")


if __name__ == "__main__":
    main()
