"""aixon — declarative AI-agent framework."""

from aixon.agent import Agent, AgentTool
from aixon.agents.orchestrator import Orchestrator
from aixon.agents.tool_agent import ToolAgent
from aixon.connector import Connector
from aixon.discovery import autodiscover
from aixon.embedding import Embedding, OpenAIEmbedding
from aixon.exceptions import (
    AixonError,
    AgentNotFoundError,
    CompositionCycleError,
    NamingError,
    RegistrationError,
)
from aixon.logging import Logger
from aixon.message import Chunk, Message, Role
from aixon.providers.base import Provider, get_provider, register_provider
from aixon.reasoning import emit_reasoning, reasoning_channel
from aixon.registry import Registry, get_registry, reset_registry
from aixon.retriever import Retriever, TypeAccess
from aixon.state import END, GraphState

__all__ = [
    # Plan 1 — foundation
    "Agent",
    "AgentTool",
    "autodiscover",
    "AixonError",
    "AgentNotFoundError",
    "CompositionCycleError",
    "NamingError",
    "RegistrationError",
    "Logger",
    "Message",
    "Chunk",
    "Role",
    "Registry",
    "get_registry",
    "reset_registry",
    # Plan 2 — providers (no langchain dep at import)
    "Provider",
    "get_provider",
    "register_provider",
    # Plan 3 — reasoning channel (stdlib-only, always available)
    "emit_reasoning",
    "reasoning_channel",
    # Plan 4 — orchestrator foundation (LangGraph state)
    "GraphState",
    "END",
    # Plan 7 — retriever + embedding + connector
    "Connector",
    "Embedding",
    "OpenAIEmbedding",
    "Retriever",
    "TypeAccess",
]

# Plan 2 — LLM + LLMAgent. These pull in langchain_core (the `llm` extra).
# Guard so `import aixon` still works on a bare install without that extra
# (contract §9.4).
try:
    from aixon.llm import LLM
    from aixon.agents.llm_agent import LLMAgent

    __all__ += ["LLM", "LLMAgent"]
except ImportError:  # pragma: no cover - bare install without [llm]
    pass

__all__ += ["ToolAgent", "Orchestrator"]

# Plan 5 — server surface (optional; requires aixon[server]). Guard so
# `import aixon` still works on a bare install without FastAPI/uvicorn
# (mirrors the LLM guard above; contract §9.4).
try:
    from aixon.server.adapters.anthropic import AnthropicAdapter
    from aixon.server.adapters.openai import OpenAIAdapter
    from aixon.server.protocol import ParsedRequest, ProtocolAdapter
    from aixon.server.server import Server

    __all__ += ["Server", "ProtocolAdapter", "OpenAIAdapter", "AnthropicAdapter", "ParsedRequest"]
except ImportError:  # pragma: no cover - bare install without [server]
    pass
