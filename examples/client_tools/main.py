"""Client Tools — an agentic client brings its own tools; the agent calls back.

Editors and IDEs (e.g. ONLYOFFICE's AI agent) POST ``tools`` on the request
and execute the ``tool_calls`` the model returns — the OpenAI function-calling
handshake, with execution on the CLIENT side. This example runs that whole
loop in-process, with **no API key and no network call**, using the
first-class opt-in for this pattern: ``LLMAgent(client_tools="passthrough")`` (#18a,
see ``aixon/agents/llm_agent.py``). The agent does NOT read
``current_client_tools()`` itself anymore — it just sets the class attribute
and the base class binds whatever the client declared onto the (here,
scripted) LLM call for that turn, returning the model's raw ``tool_calls`` to
the wire:

  1. the "editor" sends ``tools=[open_file]`` plus a user request;
  2. ``LLMAgent.invoke`` (via ``_client_bind``) forwards those tools to the
     LLM call; the (scripted) model answers with a ``tool_calls`` turn
     (``finish_reason="tool_calls"``);
  3. the editor "executes" the call, appends the ``role="tool"`` result to the
     history and POSTs again; the model, scripted to see the result, answers
     in text.

    cd examples/client_tools
    python main.py

Expected output: the two wire exchanges printed step by step — first the
tool_calls turn, then the final text answer. See README.md.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aixon import LLM, LLMAgent
from aixon.providers.base import Provider, register_provider
from aixon.server.server import Server

# ── scripted driver model (offline) ──────────────────────────────────────────
# A real deployment would put a real provider (OpenAI, Anthropic, ...) behind
# this LLM; what this example is actually about is LLMAgent(client_tools="passthrough")
# itself — the routing logic in aixon/agents/llm_agent.py is what matters.


class ScriptedChatModel(BaseChatModel):
    """Replays `script` (AIMessages) one per call; tool_calls drive the turn."""

    script: list = []
    _idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self  # tools ignored; script drives the answer

    def _generate(self, messages: list[BaseMessage],
                  stop: Optional[list[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        i = self._idx
        msg = self.script[i] if i < len(self.script) else AIMessage(content="(done)")
        object.__setattr__(self, "_idx", i + 1)
        return ChatResult(generations=[ChatGeneration(message=msg)])


class ScriptedProvider(Provider):
    name = "scripted-client-tools"
    env_key = ""

    def build(self, model: str, **params: Any) -> ScriptedChatModel:
        return ScriptedChatModel()


register_provider(ScriptedProvider())

llm = LLM("scripted-client-tools-1", provider="scripted-client-tools")
llm.chat_model.script = [
    AIMessage(content="", tool_calls=[
        {"name": "open_file", "args": {"path": "/home/user/report.docx"},
         "id": "call_1"}]),
    AIMessage(content="Done — the client reported the file was opened."),
]


class FileButlerAgent(LLMAgent):
    name = "FileButler"
    description = "Opens the file the user asks for using the client's own tools."
    llm = llm
    client_tools = "passthrough"  # #18a: bind current_client_tools()/current_tool_choice() raw


# ── the "editor" (client) side ───────────────────────────────────────────────

OPEN_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "open_file",
        "description": "Opens a file in the editor. Input: an absolute path.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}


def main() -> None:
    client = TestClient(Server().app)
    messages: list[dict] = [
        {"role": "user", "content": "Open /home/user/report.docx for me, please."}
    ]

    print("== turn 1: editor -> server (with tools) ==")
    body = {"model": "FileButler", "messages": messages, "tools": [OPEN_FILE_TOOL]}
    choice = client.post("/v1/chat/completions", json=body).json()["choices"][0]
    print("finish_reason:", choice["finish_reason"])
    tool_call = choice["message"]["tool_calls"][0]
    print("tool_call:", tool_call["function"]["name"], tool_call["function"]["arguments"])

    print("\n== the editor executes the call locally ==")
    args = json.loads(tool_call["function"]["arguments"])
    result = f'{{"status": "success", "opened": "{args["path"]}"}}'
    print("result:", result)

    print("\n== turn 2: editor -> server (with the tool result) ==")
    messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
    messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})
    body = {"model": "FileButler", "messages": messages, "tools": [OPEN_FILE_TOOL]}
    choice = client.post("/v1/chat/completions", json=body).json()["choices"][0]
    print("finish_reason:", choice["finish_reason"])
    print("answer:", choice["message"]["content"])


if __name__ == "__main__":
    main()
