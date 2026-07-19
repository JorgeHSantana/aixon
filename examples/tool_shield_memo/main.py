"""Tool shield + memoization — the 0.1.19 tool-robustness features, offline.

Two behaviors every `ToolAgent` now gets for free:

1. **Error shield** (`shield_tool_errors`, default True): a tool that raises —
   here, a fake "database" whose connection times out — hands the model a
   readable ``TOOL ERROR (...)`` result instead of killing the whole run. The
   agent answers explaining the outage.
2. **Request-scoped memoization** (`aixon.toolcache`): inside one activated
   cache scope (one served request; one ReflectiveAgent run), a tool called
   again with the SAME arguments returns the first result without
   re-executing — watch the call counter.

Both the driving model and the tools are scripted/deterministic: **no API key,
no network call**:

    cd examples/tool_shield_memo
    PYTHONPATH=../.. python main.py
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aixon import LLM, ToolAgent
from aixon.message import Message
from aixon.providers.base import Provider, register_provider
from aixon.toolcache import tool_call_cache

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
    name = "scripted-tools"
    env_key = ""

    def build(self, model: str, **params: Any) -> ScriptedChatModel:
        return ScriptedChatModel()


register_provider(ScriptedProvider())


# ── the tools ────────────────────────────────────────────────────────────────

CALLS = {"weather": 0}


def query_database(sql: str) -> str:
    """Fake DB whose connection always times out (infra outage)."""
    raise TimeoutError("connection to db.example.com:5432 timed out")


def get_weather(city: str) -> str:
    """Fake weather lookup with a call counter (to show memoization)."""
    CALLS["weather"] += 1
    return f"Sunny in {city} (lookup #{CALLS['weather']})"


# ── the agent ────────────────────────────────────────────────────────────────
# Script: call the failing DB tool, then call get_weather TWICE with the same
# args, then answer. The shield turns the DB exception into a TOOL ERROR
# result; the memoization makes the second get_weather call free.

llm = LLM("scripted-tools-1", provider="scripted-tools")
llm.chat_model.script = [
    AIMessage(content="", tool_calls=[
        {"name": "query_database", "args": {"sql": "SELECT 1"}, "id": "c1"}]),
    AIMessage(content="", tool_calls=[
        {"name": "get_weather", "args": {"city": "Recife"}, "id": "c2"}]),
    AIMessage(content="", tool_calls=[
        {"name": "get_weather", "args": {"city": "Recife"}, "id": "c3"}]),
    AIMessage(content=(
        "The database is unavailable right now (connection timeout), but the "
        "weather in Recife is sunny."
    )),
]


class FieldAssistantAgent(ToolAgent):
    name = "field-assistant"
    hidden = True
    description = "Toy agent demonstrating tool shield + memoization."
    llm = llm
    tools = [query_database, get_weather]
    # shield_tool_errors = True is the DEFAULT — shown here for the reader:
    shield_tool_errors = True


def main() -> None:
    question = [Message(role="user", content="DB status and Recife weather?")]
    print(f"> {question[0].content}\n")

    # Activate one memoization scope, as the aixon Server does per request
    # (and the ReflectiveAgent does per run). Without an active scope, tools
    # always execute.
    with tool_call_cache():
        answer = FieldAssistantAgent().invoke(question)

    print(f"Final answer: {answer.content}\n")
    print(f"query_database raised TimeoutError — and the run SURVIVED (shield).")
    print(f"get_weather was asked twice with the same args but executed "
          f"{CALLS['weather']} time(s) — the second call was memoized.")


if __name__ == "__main__":
    main()
