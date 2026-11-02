"""Contexto por request entregue ao ``request_hook`` do Server.

Tipos puros: nenhum import de fastapi/starlette, para que a aplicação possa
escrever e testar seu hook sem o extra ``server`` instalado."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class RequestContext:
    """O que o hook enxerga de uma request: modelo pedido, agente resolvido,
    cabeçalhos (chaves minúsculas, somente leitura), parâmetros, adapter e
    se a resposta é em stream."""

    model: str
    agent_name: str
    headers: Mapping[str, str]
    params: Mapping[str, Any]
    adapter: str
    stream: bool


@dataclass(frozen=True)
class RequestDecision:
    """A decisão do hook. Todo campo ``None`` significa "não mexer"."""

    impose: tuple[str, str] | None = None
    user_email: str | None = None
    trace_user_id: str | None = None
    trace_session_id: str | None = None


RequestHook = Callable[[RequestContext], "RequestDecision | None"]
"""Hook síncrono: devolve uma decisão, ``None``, ou levanta ``RequestRejected``."""


def make_context(
    *,
    model: str,
    agent_name: str,
    raw_headers: Iterable[tuple[str, str]],
    params: Mapping[str, Any],
    adapter: str,
    stream: bool,
) -> RequestContext:
    """Monta o contexto normalizando as chaves dos cabeçalhos para minúsculas;
    em duplicata, o PRIMEIRO valor vence (como ``starlette.Headers.get``)."""
    headers: dict[str, str] = {}
    for k, v in raw_headers:
        headers.setdefault(k.lower(), v)
    return RequestContext(
        model=model,
        agent_name=agent_name,
        headers=MappingProxyType(headers),
        params=params,
        adapter=adapter,
        stream=stream,
    )
