"""KnowledgeAgent — a `ToolAgent` that answers from the FAQ.

It is given the `KnowledgeRetriever` as a tool via ``as_tool()``. The
tool-calling loop (LangGraph under the hood) searches the FAQ and the model
turns the hits into an answer. ``hidden`` — reached through the orchestrator.
"""

from __future__ import annotations

from aixon import ToolAgent

from knowledge.faq_retriever import KnowledgeRetriever
from llm_config import make_llm


class KnowledgeAgent(ToolAgent):
    name = "knowledge"
    hidden = True
    description = "Answers product, billing and security questions from the FAQ."
    llm = make_llm()
    prompt = (
        "You are Acme's support assistant. Use the faq_search tool to find "
        "relevant help-center articles, then answer the user concisely. If the "
        "FAQ has no answer, say so."
    )
    tools = [
        KnowledgeRetriever().as_tool(
            name="faq_search",
            description="Search Acme's help-center articles and FAQ.",
            k=3,
        )
    ]
