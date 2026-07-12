"""Process-global registry of agents. Agents self-register on definition
(see ``Agent.__init_subclass__``); the server and CLI read this registry to
route requests and build menus."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

from aixon.exceptions import AgentNotFoundError, RegistrationError
from aixon.logging import Logger

if TYPE_CHECKING:
    # Type-only: aixon.agent imports get_registry from this module at runtime,
    # so a real top-level import here would be circular. Under
    # `from __future__ import annotations` every annotation below is a
    # deferred string, so this guarded import is enough for mypy without
    # ever executing at import time.
    from aixon.agent import Agent

_log = Logger("aixon.registry")


class Registry:
    def __init__(self) -> None:
        self._agents: dict[str, "Agent"] = {}   # name -> agent
        self._aliases: dict[str, str] = {}      # alias -> name
        self._order: list[str] = []             # registration order of names
        # Serializes register()'s check-then-insert so concurrent registrations
        # cannot both pass the uniqueness check.
        self._lock = threading.Lock()

    def register(self, agent: "Agent") -> None:
        with self._lock:
            name = agent.name
            if name in self._agents or name in self._aliases:
                raise RegistrationError(
                    f"Agent name '{name}' is already registered. Names and aliases "
                    f"must be unique across the registry."
                )
            for alias in agent.aliases:
                if alias in self._agents or alias in self._aliases:
                    raise RegistrationError(
                        f"Alias '{alias}' (on agent '{name}') collides with an "
                        f"existing name or alias."
                    )
            self._agents[name] = agent
            self._order.append(name)
            for alias in agent.aliases:
                self._aliases[alias] = name
        hidden = " (hidden)" if agent.hidden else ""
        _log.info(f"registered agent '{name}'{hidden} aliases={agent.aliases}")

    def resolve(self, name: str) -> "Agent":
        if name in self._agents:
            return self._agents[name]
        if name in self._aliases:
            return self._agents[self._aliases[name]]
        # Convenience: an empty/missing model on a single-agent registry resolves
        # to that lone agent (the client need not know its name). A NON-empty
        # unknown name always raises — we never silently mask a typo'd model.
        if not name and len(self._agents) == 1:
            return next(iter(self._agents.values()))
        raise AgentNotFoundError(
            f"No agent registered as '{name}'. "
            f"Known agents: {sorted(self._agents)}."
        )

    def public(self) -> list["Agent"]:
        return [self._agents[n] for n in self._order if not self._agents[n].hidden]

    def all(self) -> list["Agent"]:
        return [self._agents[n] for n in self._order]

    def clear(self) -> None:
        self._agents.clear()
        self._aliases.clear()
        self._order.clear()
        # Keep the class-side flags in sync with the emptied registry, otherwise
        # Agent.__init__ would short-circuit forever and nothing could ever
        # re-register (the same desync reset_registry guards against).
        _reset_registered_flags()


def _reset_registered_flags() -> None:
    """Clear ``_registered`` on Agent and every subclass. Imported lazily to
    avoid a registry<->agent import cycle."""
    from aixon.agent import Agent

    stack = [Agent]
    while stack:
        cls = stack.pop()
        cls._registered = False
        stack.extend(cls.__subclasses__())


_registry: Optional[Registry] = None
# Guards the lazy creation in get_registry() against concurrent first calls.
_registry_lock = threading.Lock()


def get_registry() -> Registry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = Registry()
        return _registry


def reset_registry() -> None:
    """Replace the global registry with a fresh, empty one AND clear the
    ``_registered`` flag on every Agent subclass.

    Without the flag reset, ``Agent.__init__`` would short-circuit on the stale
    ``cls._registered = True`` and never re-register into the new registry — the
    registry would stay empty while the classes still believe they are
    registered (a desync between the two reset paths). Imported lazily to avoid a
    registry<->agent import cycle."""
    global _registry
    _registry = Registry()
    _reset_registered_flags()
