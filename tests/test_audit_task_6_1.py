"""Testes dos achados da Task 6.1 (varredura 0.1.29): runtime, llm e agent."""

from __future__ import annotations

import asyncio
import contextvars
import threading

import pytest

from aixon import runtime
from aixon.agent import Agent
from aixon.llm import LLM
from aixon.message import Chunk, Message
from tests._fakes import make_llm

# ----- runtime-1 / llm-1 / agent-4: normalização de aliases num lugar só ----

def test_runtime_1_normalize_param_aliases():
    n = runtime.normalize_param_aliases
    assert n({"max_completion_tokens": 5}) == {"max_tokens": 5}
    assert n({"max_completion_tokens": 5, "max_tokens": 7}) == {"max_tokens": 7}
    assert n({"max_completion_tokens": 5, "max_tokens": None}) == {"max_tokens": 5}
    assert n({"stop_sequences": ["x"], "seed": 1}) == {"stop": ["x"], "seed": 1}
    assert n({"max_completion_tokens": None}) == {}
    src = {"stop_sequences": ["x"]}
    n(src)
    assert src == {"stop_sequences": ["x"]}  # não muta a entrada


def test_runtime_1_generation_params_igual_ao_antigo():
    with runtime.generation_params(
        {"max_completion_tokens": 9, "stop_sequences": ["a"], "temperature": 0.2, "user": "u"}
    ):
        assert runtime.current_generation_params() == {
            "max_tokens": 9, "stop": ["a"], "temperature": 0.2,
        }
    with runtime.generation_params({"max_completion_tokens": 9, "max_tokens": 3}):
        assert runtime.current_generation_params() == {"max_tokens": 3}


def test_llm_1_modulos_nao_importam_o_dict_privado():
    import inspect

    import aixon.agent as agent_mod
    import aixon.llm as llm_mod

    assert "_PARAM_ALIASES" not in inspect.getsource(llm_mod)
    assert "_PARAM_ALIASES" not in inspect.getsource(agent_mod)


def test_agent_4_settings_lost_under_imposition():
    llm = LLM("gpt-x", provider="openai", reasoning={"effort": "high"},
              max_completion_tokens=5, seed=1, timeout=3, temperature=0.1, top_p=None)
    assert llm.settings_lost_under_imposition() == ["reasoning", "seed", "timeout"]
    assert LLM("gpt-x", provider="openai", pinned=True, seed=1).settings_lost_under_imposition() == []
    assert LLM("gpt-x", provider="openai", reasoning=False).settings_lost_under_imposition() == []


# ----- runtime-2: reset em outro Context não levanta -------------------------

@pytest.mark.parametrize("make", [
    lambda: runtime.generation_params({"temperature": 0.1}),
    lambda: runtime.client_tools_scope([{"type": "function"}]),
    lambda: runtime.tool_choice_scope("auto"),
    lambda: runtime.prediction_scope("x"),
    lambda: runtime.user_email_scope("a@b.c"),
    lambda: runtime.imposed_model_scope("m", "p"),
])
def test_runtime_2_reset_em_outro_context_nao_levanta(make):
    cm = make()
    contextvars.copy_context().run(cm.__enter__)  # entra num Context descartável
    contextvars.Context().run(cm.__exit__, None, None, None)


def test_runtime_2_reasoning_channel_usa_o_mesmo_helper():
    from aixon.reasoning import reasoning_channel

    cm = reasoning_channel()
    contextvars.copy_context().run(cm.__enter__)
    contextvars.Context().run(cm.__exit__, None, None, None)


def test_runtime_2_reset_quietly_restaura_no_mesmo_context():
    with runtime.prediction_scope("x"):
        with runtime.prediction_scope("y"):
            assert runtime.current_prediction() == "y"
        assert runtime.current_prediction() == "x"
    assert runtime.current_prediction() is None


# ----- runtime-3: o valor entregue ao `with` não é o estado guardado ----------

def test_runtime_3_mutar_o_yield_nao_polui_o_estado():
    with runtime.generation_params({"temperature": 0.1}) as p:
        p["temperature"] = 9
        assert runtime.current_generation_params() == {"temperature": 0.1}
    with runtime.client_tools_scope([{"type": "function", "function": {"name": "a"}}]) as t:
        t[0]["function"]["name"] = "zz"
        t.append({"x": 1})
        assert runtime.current_client_tools() == [{"type": "function", "function": {"name": "a"}}]


# ----- llm-2: chat_model preguiçoso sob concorrência -------------------------

def test_llm_2_chat_model_concorrente_devolve_um_so_cliente(monkeypatch):
    llm = LLM("gpt-x", provider="openai")
    barrier = threading.Barrier(2)

    def fake_build(self, provider, **kw):
        barrier.wait(timeout=5)
        return object()

    monkeypatch.setattr(LLM, "_build", fake_build)
    got: list = []
    threads = [threading.Thread(target=lambda: got.append(llm.chat_model)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(got) == 2 and got[0] is got[1] is llm.chat_model


# ----- agent-3: provedor resolvido pelo LLM ----------------------------------

def test_agent_3_llm_provider_name_publico():
    assert LLM("gpt-x", provider="openai").provider_name == "openai"
    assert LLM("gpt-x", provider="openai").imposition() == ("gpt-x", "openai")


def test_agent_3_step_scope_nao_usa_o_privado():
    import inspect

    from aixon.agent import _step_scope

    assert "_provider()" not in inspect.getsource(_step_scope)


# ----- agent-2 / reflective-7: timeout_texts ---------------------------------

def test_agent_2_timeout_texts():
    from aixon.agents.reflective import ReflectiveAgent
    from aixon.agents.tool_agent import ToolAgent

    class BareAgent(Agent):
        def invoke(self, messages):
            return Message(role="assistant", content="")

        def stream(self, messages):
            yield from ()

    assert BareAgent().timeout_texts() == set()

    class TAgent(ToolAgent):
        name = "t"
        llm = make_llm()
        max_execution_time = 7
        timeout_message = "estourou {seconds}s"

    assert TAgent().timeout_texts() == {"estourou 7s"}

    class RAgent(ReflectiveAgent):
        name = "r"
        agent = TAgent
        judge_llm = make_llm()
        judge_rubric = "responde"

    r = RAgent()
    assert r.timeout_texts() == {"estourou 7s"}
    assert r._is_worker_timeout_answer(Message(role="assistant", content="estourou 7s"))
    assert not r._is_worker_timeout_answer(Message(role="assistant", content="ok"))


# ----- agent-1: pós-run do as_tool num lugar só ------------------------------

def _agent_streaming(chunks):
    class SAgent(Agent):
        name = "s"

        def invoke(self, messages):
            return Message(role="assistant", content="")

        def stream(self, messages):
            yield from chunks

        async def astream(self, messages):
            for c in chunks:
                yield c

    return SAgent()


@pytest.mark.parametrize("chunks,expected", [
    ([Chunk(content="a"), Chunk(content="b")], "ab"),
    ([Chunk(content="")], ""),
])
def test_agent_1_sync_e_async_iguais(chunks, expected):
    tool = _agent_streaming(chunks).as_tool()
    assert tool.func("x") == expected
    assert asyncio.run(tool.coroutine("x")) == expected


def test_agent_1_tool_calls_sem_conteudo_iguais():
    tc = [{"id": "1", "name": "f", "args": {}}]
    tool = _agent_streaming([Chunk(tool_calls=tc)]).as_tool()
    assert tool.func("x") == asyncio.run(tool.coroutine("x"))
    assert "f" in tool.func("x")
