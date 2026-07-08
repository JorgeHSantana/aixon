"""Neutral message types. The agent runtime speaks ONLY these — protocol
adapters (Plan 5) translate wire formats to and from them. Nothing here may
import a provider or protocol module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """A single neutral message. ``tool_calls`` carries provider-agnostic
    tool-call dicts; ``reasoning`` carries model reasoning when present."""

    role: Role
    content: str = ""
    name: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    reasoning: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict, omitting empty optional fields."""
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            data["name"] = self.name
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.reasoning is not None:
            data["reasoning"] = self.reasoning
        return data


@dataclass
class Chunk:
    """A streamed delta from an Agent. ``content`` and ``reasoning`` are
    additive text deltas; ``done`` marks the final chunk of a stream.

    ``tool_calls`` carries neutral tool-call dicts (``{"name", "args", "id"}``,
    the same shape as ``Message.tool_calls``) that the agent wants surfaced to
    the CLIENT for execution — adapters translate them to the wire dialect
    (e.g. OpenAI ``delta.tool_calls``)."""

    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
