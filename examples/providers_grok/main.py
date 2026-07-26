"""Provider xAI (Grok) — exemplo mínimo (0.1.20).

Requer XAI_API_KEY no ambiente. O nome `grok-*` resolve o provider sozinho;
`reasoning=True` vira `reasoning_effort` no endpoint OpenAI-compatível da xAI.
Rode: XAI_API_KEY=... python main.py "explique RAG em uma frase"
"""
import sys

from aixon import LLM, LLMAgent


class GrokAgent(LLMAgent):
    name = "grok-demo"
    description = "Demo do provider xAI"
    llm = LLM("grok-4.5", temperature=0.2, reasoning=True)
    prompt = "Você é um assistente direto: responda em uma frase."


if __name__ == "__main__":
    from aixon.message import Message
    from aixon.registry import get_registry

    question = " ".join(sys.argv[1:]) or "explique RAG em uma frase"
    agent = get_registry().resolve("grok-demo")
    out = agent.invoke([Message(role="user", content=question)])
    print(out.content)
