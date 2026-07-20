"""Tracing — veja TUDO que um tracer captura de um agente aixon, offline.

O aixon fala LangChain/LangGraph por baixo, então o tracing do ecossistema
engancha sem mudar nada no framework. Este exemplo usa a rota mais barata —
o debug tracer de console (`set_debug`) — com um modelo scriptado: **sem API
key, sem rede, sem conta**. As outras rotas (LangSmith via env vars; Langfuse
self-hosted via OTel) capturam exatamente a mesma árvore — ver
docs/tracing.md.

    cd examples/tracing
    PYTHONPATH=../.. python main.py
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.globals import set_debug
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aixon import LLM, ToolAgent
from aixon.message import Message
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
                  stop: Optional[list[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        i = self._idx
        msg = self.script[i] if i < len(self.script) else AIMessage(content="(done)")
        object.__setattr__(self, "_idx", i + 1)
        return ChatResult(generations=[ChatGeneration(message=msg)])


class ScriptedProvider(Provider):
    name = "scripted-tracing"
    env_key = ""

    def build(self, model: str, **params: Any) -> ScriptedChatModel:
        return ScriptedChatModel()


register_provider(ScriptedProvider())


# ── um agente com tool, para a árvore ter graça ─────────────────────────────

def get_weather(city: str) -> str:
    """Fake weather lookup."""
    return f"Sunny in {city}, 28°C"


llm = LLM("scripted-tracing-1", provider="scripted-tracing")
llm.chat_model.script = [
    AIMessage(content="", tool_calls=[
        {"name": "get_weather", "args": {"city": "Recife"}, "id": "c1"}]),
    AIMessage(content="It is sunny in Recife (28°C)."),
]


class WeatherAgent(ToolAgent):
    name = "weather-traced"
    hidden = True
    description = "Toy agent to demonstrate tracing."
    llm = llm
    tools = [get_weather]


def main() -> None:
    # A ÚNICA linha de "tracing": tudo que rolar abaixo (chains, chamadas de
    # modelo, tool calls, tokens) é despejado no console. Trocar esta linha
    # por env vars do LangSmith ou pela instrumentação OTel do Langfuse dá a
    # MESMA árvore numa UI — ver docs/tracing.md.
    set_debug(True)

    out = WeatherAgent().invoke([Message(role="user", content="Weather in Recife?")])
    set_debug(False)
    print(f"\nFinal answer: {out.content}")
    print("\n^ Repare acima: [chain:LangGraph] > [chain:model] > [tool:get_weather]")
    print("  — a árvore inteira, com inputs/outputs por nó. É isso que um")
    print("  tracer real (LangSmith/Langfuse) recebe, sem mudar o aixon.")


if __name__ == "__main__":
    main()
