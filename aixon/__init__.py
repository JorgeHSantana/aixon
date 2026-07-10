"""aixon — declarative AI-agent framework."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("aixon")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout)
    __version__ = "0.0.0"

from aixon.agent import Agent, AgentTool
from aixon.agents.llm_agent import LLMAgent
from aixon.agents.orchestrator import Orchestrator
from aixon.agents.reflective import ReflectiveAgent
from aixon.agents.tool_agent import ToolAgent
from aixon.connector import Connector, HttpToolConnector
from aixon.discovery import autodiscover
from aixon.embedding import Embedding, OpenAIEmbedding
from aixon.exceptions import (
    AixonError,
    AgentNotFoundError,
    CompositionCycleError,
    NamingError,
    RegistrationError,
)
from aixon.llm import LLM
from aixon.logging import Logger
from aixon.message import Chunk, Message, Role
from aixon.providers.base import Provider, get_provider, register_provider
from aixon.reasoning import emit_reasoning, reasoning_channel
from aixon.registry import Registry, get_registry, reset_registry
from aixon.retriever import Retriever, TypeAccess
from aixon.retrievers.ragie import RagieRetriever
from aixon.retrievers.tavily import TavilyRetriever
from aixon.retrievers.weaviate import WeaviateRetriever
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
    # Plan 2 — LLM + providers (langchain is a core dependency)
    "LLM",
    "LLMAgent",
    "Provider",
    "get_provider",
    "register_provider",
    # Plan 3 — reasoning channel + tool-calling agent
    "emit_reasoning",
    "reasoning_channel",
    "ToolAgent",
    # Plan 4 — orchestrator + graph state
    "Orchestrator",
    "ReflectiveAgent",
    "GraphState",
    "END",
    # Plan 7 — retriever + embedding + connector
    "Connector",
    "HttpToolConnector",
    "Embedding",
    "OpenAIEmbedding",
    "Retriever",
    "TypeAccess",
    "TavilyRetriever",
    "RagieRetriever",
    "WeaviateRetriever",
]

# Plan 5 — server surface. aixon.server.server, protocol, and both adapters
# import only stdlib + aixon at module level (fastapi/uvicorn are lazy, used
# only inside methods), so this import never fails on a bare install — no
# guard needed (was dead code; see bug-sweep I5).
from aixon.server.adapters.anthropic import AnthropicAdapter
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.protocol import ParsedRequest, ProtocolAdapter
from aixon.server.server import Server

__all__ += ["Server", "ProtocolAdapter", "OpenAIAdapter", "AnthropicAdapter", "ParsedRequest"]
