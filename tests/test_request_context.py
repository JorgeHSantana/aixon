"""Tipos de contexto por request e a exceção RequestRejected."""

import dataclasses
import subprocess
import sys

import pytest

from aixon.exceptions import AixonError, RequestRejected
from aixon.server.context import RequestDecision, make_context


def _ctx():
    return make_context(
        model="m",
        agent_name="a",
        raw_headers=[("X-Tier", "free"), ("x-tier", "pro")],
        params={},
        adapter="openai",
        stream=False,
    )


def test_context_e_imutavel():
    ctx = _ctx()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.model = "outro"  # type: ignore[misc]
    with pytest.raises(TypeError):
        ctx.headers["x"] = 1  # type: ignore[index]


def test_make_context_normaliza_chaves():
    ctx = make_context(
        model="m", agent_name="a", raw_headers=[("X-Tier", "pro")],
        params={}, adapter="openai", stream=True,
    )
    assert ctx.headers["x-tier"] == "pro"
    # último valor vence em duplicata
    assert _ctx().headers["x-tier"] == "pro"


def test_decision_vazia_tem_tudo_none():
    d = RequestDecision()
    assert d.impose is None
    assert d.user_email is None
    assert d.trace_user_id is None
    assert d.trace_session_id is None


def test_request_rejected_padroes():
    exc = RequestRejected("negado")
    assert isinstance(exc, AixonError)
    assert exc.status == 403
    assert exc.code == "request_rejected"
    assert str(exc) == "negado"


def test_context_nao_importa_fastapi():
    code = (
        "import sys, aixon.server.context, aixon.server; "
        "assert 'fastapi' not in sys.modules; "
        "assert 'starlette' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
