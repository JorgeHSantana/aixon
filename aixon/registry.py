"""Process-global registry of agents. Agents self-register on definition
(see ``Agent.__init_subclass__``); the server and CLI read this registry to
route requests and build menus."""

from __future__ import annotations

from typing import Optional

from aixon.exceptions import AgentNotFoundError, RegistrationError
from aixon.logging import Logger

_log = Logger("aixon.registry")


class Registry:
    def __init__(self) -> None:
        self._agents: dict[str, object] = {}   # name -> agent
        self._aliases: dict[str, str] = {}      # alias -> name
        self._order: list[str] = []             # registration order of names

    def register(self, agent: object) -> None:
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

    def resolve(self, name: str) -> object:
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

    def public(self) -> list:
        return [self._agents[n] for n in self._order if not self._agents[n].hidden]

    def all(self) -> list:
        return [self._agents[n] for n in self._order]

    def clear(self) -> None:
        self._agents.clear()
        self._aliases.clear()
        self._order.clear()


_registry: Optional[Registry] = None


def get_registry() -> Registry:
    global _registry
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
    from aixon.agent import Agent

    stack = [Agent]
    while stack:
        cls = stack.pop()
        cls._registered = False
        stack.extend(cls.__subclasses__())
