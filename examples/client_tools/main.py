"""Client Tools — an agentic client brings its own tools; the agent calls back.

Editors and IDEs (e.g. ONLYOFFICE's AI agent) POST ``tools`` on the request
and execute the ``tool_calls`` the model returns — the OpenAI function-calling
handshake, with execution on the CLIENT side. This example runs that whole
loop in-process, with **no API key and no network call**:

  1. the "editor" sends ``tools=[open_file]`` plus a user request;
  2. the agent reads them via ``current_client_tools()`` and answers with a
     ``tool_calls`` turn (``finish_reason="tool_calls"``);
  3. the editor "executes" the call, appends the ``role="tool"`` result to the
     history and POSTs again; the agent, seeing the result, answers in text.

    cd examples/client_tools
    python main.py

Expected output: the two wire exchanges printed step by step — first the
tool_calls turn, then the final text answer. See README.md.
"""

from __future__ import annotations

import json
import re
from typing import AsyncIterator, Iterator

from fastapi.testclient import TestClient

from aixon.agent import Agent
from aixon.message import Chunk, Message
from aixon.runtime import current_client_tools
from aixon.server.server import Server

# ── the agent: scripted, offline ─────────────────────────────────────────────
# A real deployment would put an LLM here (ToolAgent/LLMAgent); the routing
# logic below is what matters — read the CLIENT's tools, answer with
# tool_calls, and finish in text once the tool result comes back.


class FileButlerAgent(Agent):
    name = "FileButler"
    description = "Opens the file the user asks for using the client's own tools."

    def _decide(self, messages: list[Message]) -> Message:
        # Turn 2: the client already executed our call and posted the result.
        tool_results = [m for m in messages if m.role == "tool"]
        if tool_results:
            return Message(
                role="assistant",
                content=f"Done — the client reported: {tool_results[-1].content}",
            )

        # Turn 1: pick the client-declared tool and call it.
        tools = current_client_tools()
        names = [t.get("function", {}).get("name") for t in tools]
        if "open_file" not in names:
            return Message(
                role="assistant",
                content="This client declared no open_file tool; nothing to call.",
            )
        user_text = next((m.content for m in reversed(messages) if m.role == "user"), "")
        match = re.search(r"(/\S+)", user_text)
        path = match.group(1) if match else "/tmp/notes.txt"
        return Message(
            role="assistant",
            content="",
            tool_calls=[{"name": "open_file", "args": {"path": path}, "id": "call_1"}],
        )

    # Neutral boundary: sync + async, invoke + stream.
    def invoke(self, messages: list[Message]) -> Message:
        return self._decide(messages)

    async def ainvoke(self, messages: list[Message]) -> Message:
        return self._decide(messages)

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        final = self._decide(messages)
        if final.content:
            yield Chunk(content=final.content)
        if final.tool_calls:
            yield Chunk(tool_calls=final.tool_calls)
        yield Chunk(done=True)

    async def astream(self, messages: list[Message]) -> AsyncIterator[Chunk]:
        for chunk in self.stream(messages):
            yield chunk


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
