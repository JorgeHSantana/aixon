"""Client tools, first-class merge mode (#18c) — offline, no server.

``ToolAgent(client_tools="merge")`` puts the CLIENT's declared tools in the
SAME tool-calling loop as the agent's own (internal) tools:

  - a call to an INTERNAL tool executes server-side, same as always — the
    loop keeps going;
  - a call to a CLIENT tool ends the turn immediately: the run returns
    ``Message(role="assistant", content="", tool_calls=[...])`` instead of
    trying to run it — the CLIENT, not aixon, must execute it.

This mirrors ``tests/test_tool_agent_client_tools.py::test_retomada_com_tool_
result_do_cliente`` end to end, but calls ``ToolAgent.invoke`` directly (no
``Server``/HTTP) so the two "requests" are just two Python calls:

  1. request 1 — the model calls the internal tool (``buscar_orcamento``,
     runs here) THEN the client tool (``inserir_no_documento``) — the run
     stops there and returns the client's tool_calls.
  2. the "editor" executes ``inserir_no_documento`` itself and appends the
     result as a ``role="tool"`` message.
  3. request 2 — same history + the tool result; the model concludes in text.

    cd examples/client_tools
    PYTHONPATH=../.. python merge_demo.py

See ``examples/client_tools/main.py`` for the raw-passthrough alternative
(``LLMAgent(client_tools="passthrough")``, #18a) and the README's "Modo merge (#18c)"
section for the request/response diagram.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aixon import LLM, ToolAgent
from aixon.message import Message
from aixon.providers.base import Provider, register_provider
from aixon.runtime import client_tools_scope

# ── scripted driver model (offline) ──────────────────────────────────────────


class ScriptedChatModel(BaseChatModel):
    """Replays `script` (AIMessages) one per call; tool_calls drive the loop."""

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
    name = "scripted-merge"
    env_key = ""

    def build(self, model: str, **params: Any) -> ScriptedChatModel:
        return ScriptedChatModel()


register_provider(ScriptedProvider())


# ── the client's tool (what the "editor" declares on the request) ───────────

DOC_TOOL = [{
    "type": "function",
    "function": {
        "name": "inserir_no_documento",
        "description": "Insere texto no documento aberto no editor.",
        "parameters": {
            "type": "object",
            "properties": {"texto": {"type": "string"}},
            "required": ["texto"],
        },
    },
}]


# ── the agent's own (internal) tool ──────────────────────────────────────────

def buscar_orcamento(item: str) -> str:
    """Consulta o orçamento (server-side) do item pedido."""
    return "R$ 4.200,00"


# ── request 1: the model calls the internal tool, then the client's ─────────

llm = LLM("scripted-merge-1", provider="scripted-merge")
llm.chat_model.script = [
    AIMessage(content="", tool_calls=[
        {"name": "buscar_orcamento", "args": {"item": "licenças"}, "id": "c1"}]),
    AIMessage(content="", tool_calls=[
        {"name": "inserir_no_documento",
         "args": {"texto": "Orçamento de licenças: R$ 4.200,00"}, "id": "c2"}]),
    # Request 2 (below): the model sees the client's tool result and concludes.
    AIMessage(content="Inserido com sucesso no documento."),
]


class RedatorAgent(ToolAgent):
    name = "redator-merge-demo"
    hidden = True
    description = "Toy agent demonstrating client_tools='merge' (#18c)."
    llm = llm
    tools = [buscar_orcamento]
    client_tools = "merge"  # the client's tool defs join the loop above


def main() -> None:
    pergunta = [Message(
        role="user",
        content="Busque o orçamento de licenças e insira no documento.",
    )]
    print(f"> {pergunta[0].content}\n")

    print("== request 1: editor -> agent (com tools=[inserir_no_documento]) ==")
    with client_tools_scope(DOC_TOOL):
        resposta1 = RedatorAgent().invoke(pergunta)
    print("role:", resposta1.role)
    print("tool_calls:", resposta1.tool_calls)
    assert resposta1.tool_calls, "a call do cliente deveria ter borbulhado"
    call = resposta1.tool_calls[0]

    print("\n== o editor executa a call localmente ==")
    resultado = "ok — texto inserido"
    print("resultado:", resultado)

    print("\n== request 2: editor -> agent (histórico + role=tool) ==")
    historico = [
        *pergunta,
        Message(role="assistant", content="", tool_calls=[call]),
        Message(role="tool", content=resultado, tool_call_id=call["id"]),
    ]
    with client_tools_scope(DOC_TOOL):
        resposta2 = RedatorAgent().invoke(historico)
    print("role:", resposta2.role)
    print("content:", resposta2.content)


if __name__ == "__main__":
    main()
