"""O cancel scope do anyio cancela a árvore do ToolAgent.astream.

Em produção (uvicorn, ASGI spec_version 2.3) o StreamingResponse do Starlette consome o stream
dentro de um task group do anyio e, quando o cliente desconecta, cancela pelo cancel scope, que
REENTREGA o cancelamento a cada volta do loop. O `finally` do `ToolAgent.astream` cancela o
`next_task` e o espera: com `await next_task`, cada cancelamento reentregue passa, pelo
`_fut_waiter`, ao next_task e dali ao `exit_task` do `AsyncPregelLoop.__aexit__`; a limpeza do
LangGraph morre antes de cancelar as tarefas dos nós e a chamada de LLM do especialista em voo
segue sozinha até o fim (e, no fim, abre a seguinte). Com `await asyncio.wait({next_task})` a
espera não repassa o cancelamento e a limpeza termina."""
import asyncio
from typing import Any

import anyio
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aixon.agents.tool_agent import ToolAgent
from aixon.llm import LLM
from aixon.message import Message
from aixon.providers.base import Provider, register_provider
from aixon.registry import get_registry
from tests._fakes import make_llm

_EVENTOS: list[str] = []
ESPECIALISTAS = 2


class _ModeloLento(BaseChatModel):
    """O modelo do especialista: a chamada demora 0.5 s e diz como terminou."""

    @property
    def _llm_type(self) -> str:
        return "lento"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ModeloLento":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **k) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="x"))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **k) -> ChatResult:
        _EVENTOS.append("chamada-comecou")
        try:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            _EVENTOS.append("chamada-cancelada")
            raise
        _EVENTOS.append("chamada-terminou")  # o que uma chamada órfã faria
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="x"))])


class _ProvedorLento(Provider):
    name = "lento-cancel-scope"
    env_key = ""

    def build(self, model: str, **p: Any) -> BaseChatModel:
        return _ModeloLento()


register_provider(_ProvedorLento())


def test_cancel_scope_do_anyio_cancela_a_chamada_de_llm_do_no_em_voo():
    _EVENTOS.clear()
    type("EspecialistaLentoAgent", (ToolAgent,),
         {"name": "especialistalento", "llm": LLM("m", provider="lento-cancel-scope"),
          "hidden": True})
    especialista = get_registry().resolve("especialistalento")

    coordenador_llm = make_llm()
    coordenador_llm.chat_model.script = [
        AIMessage(content="", tool_calls=[
            {"name": f"esp{i}", "args": {"text": "vai"}, "id": f"c{i}"} for i in range(ESPECIALISTAS)]),
        AIMessage(content="fim"),
    ]
    type("CoordenadorCancelScopeAgent", (ToolAgent,),
         {"name": "coordenadorcs", "llm": coordenador_llm,
          "tools": [especialista.as_tool(name=f"esp{i}", description="especialista lento")
                    for i in range(ESPECIALISTAS)]})
    coordenador = get_registry().resolve("coordenadorcs")

    async def run():
        async with anyio.create_task_group() as tg:
            async def consome():
                async for _ in coordenador.astream([Message(role="user", content="vai")]):
                    pass

            tg.start_soon(consome)
            while _EVENTOS.count("chamada-comecou") < ESPECIALISTAS:
                await asyncio.sleep(0.005)
            tg.cancel_scope.cancel()  # o "parar": o cliente desconectou
        await asyncio.sleep(1.0)  # o que ficou rodando de órfão termina (ou continua) aqui

    asyncio.run(run())

    assert sorted(_EVENTOS) == ["chamada-cancelada"] * ESPECIALISTAS + ["chamada-comecou"] * ESPECIALISTAS, (
        "a chamada de LLM do nó em voo tinha de ser cancelada, e nenhuma outra podia começar")
