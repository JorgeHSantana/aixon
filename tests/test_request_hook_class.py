# tests/test_request_hook_class.py
"""0.1.30 — `RequestHook` vira classe com fases (check → identity →
imposition), `Identity` e o `HeaderIdentityHook` genérico."""
from __future__ import annotations

import pytest

from aixon.exceptions import AixonError, RequestRejected
from aixon.server.context import (
    HeaderIdentityHook,
    Identity,
    RequestDecision,
    RequestHook,
    RequestHookLike,
    make_context,
)


def _ctx(headers=()):
    return make_context(
        model="m",
        agent_name="a",
        raw_headers=list(headers),
        params={},
        adapter="openai",
        stream=False,
    )


def test_base_sem_opiniao():
    assert RequestHook()(_ctx()) == RequestDecision()


def test_ordem_check_identity_imposition():
    calls: list[str] = []

    class H(RequestHook):
        def check(self, ctx):
            calls.append("check")

        def identity(self, ctx):
            calls.append("identity")
            return Identity()

        def imposition(self, ctx):
            calls.append("imposition")
            return None

    H()(_ctx())
    assert calls == ["check", "identity", "imposition"]


def test_check_recusa_antes_de_identity():
    called = {"identity": False}

    class H(RequestHook):
        def check(self, ctx):
            raise RequestRejected("x", status=401, code="no_email")

        def identity(self, ctx):
            called["identity"] = True
            return Identity()

    with pytest.raises(RequestRejected) as ei:
        H()(_ctx())
    assert called["identity"] is False
    assert ei.value.status == 401
    assert ei.value.code == "no_email"


def test_rejected_dentro_de_imposition_propaga():
    class H(RequestHook):
        def imposition(self, ctx):
            raise RequestRejected("y", status=429, code="quota")

    with pytest.raises(RequestRejected) as ei:
        H()(_ctx())
    assert ei.value.status == 429
    assert ei.value.code == "quota"


def test_so_imposition_sobrescrita():
    class H(RequestHook):
        def imposition(self, ctx):
            return ("gpt-5.4", "gateway")

    d = H()(_ctx())
    assert d.impose == ("gpt-5.4", "gateway")
    assert d.user_email is None
    assert d.trace_user_id is None
    assert d.trace_session_id is None


def test_funcao_solta_continua_valida():
    fn: RequestHookLike = lambda ctx: RequestDecision(user_email="a@b.c")  # noqa: E731
    assert fn(_ctx()) == RequestDecision(user_email="a@b.c")


def test_header_identity_le_os_tres():
    hook = HeaderIdentityHook(
        email_header="x-e", user_id_header="x-u", session_header="x-s"
    )
    ctx = _ctx([("x-e", "a@b.c"), ("x-u", "u1"), ("x-s", "s1")])
    assert hook.identity(ctx) == Identity("a@b.c", "a@b.c", "s1")
    d = hook(ctx)
    assert (d.user_email, d.trace_user_id, d.trace_session_id) == (
        "a@b.c",
        "a@b.c",
        "s1",
    )


def test_header_identity_email_vazio_cai_no_user_id():
    hook = HeaderIdentityHook(
        email_header="x-e", user_id_header="x-u", session_header="x-s"
    )
    ctx = _ctx([("x-e", ""), ("x-u", "u1")])
    assert hook.identity(ctx) == Identity("", "u1", None)


def test_header_identity_so_email_obrigatorio():
    hook = HeaderIdentityHook(email_header="x-e")
    ctx = _ctx(
        [("x-e", "a@b.c"), ("x-openwebui-user-id", "u1"), ("x-openwebui-chat-id", "c1")]
    )
    ident = hook.identity(ctx)
    assert ident.trace_user_id == "a@b.c"
    assert ident.trace_session_id is None


def test_header_identity_normaliza_caixa():
    hook = HeaderIdentityHook(email_header="X-My-Email")
    ctx = _ctx([("X-MY-EMAIL", "a@b.c")])
    assert hook.identity(ctx).user_email == "a@b.c"


def test_header_identity_email_header_vazio_falha():
    with pytest.raises(AixonError):
        HeaderIdentityHook(email_header="  ")


def test_exports():
    from aixon.server import (  # noqa: F401
        HeaderIdentityHook,
        Identity,
        RequestHook,
        RequestHookLike,
    )


# --- ChainRequestHook ---


def test_chain_vazio_sem_opiniao():
    from aixon.server.context import ChainRequestHook

    assert ChainRequestHook([])(_ctx()) == RequestDecision()


def test_chain_primeiro_nao_none_vence():
    from aixon.server.context import ChainRequestHook

    a = lambda ctx: RequestDecision(user_email="a", impose=None)  # noqa: E731
    b = lambda ctx: RequestDecision(user_email="b", impose=("m", "p"))  # noqa: E731
    d = ChainRequestHook([a, b])(_ctx())
    assert d.user_email == "a"
    assert d.impose == ("m", "p")


def test_chain_recusa_para_a_cadeia():
    from aixon.server.context import ChainRequestHook

    flag = {"chamado": False}

    def h1(ctx):
        raise RequestRejected("não")

    def h2(ctx):
        flag["chamado"] = True

    with pytest.raises(RequestRejected):
        ChainRequestHook([h1, h2])(_ctx())
    assert flag["chamado"] is False


def test_chain_aceita_funcao_e_classe():
    from aixon.server.context import ChainRequestHook

    def fn(ctx):
        return None

    def sess(ctx):
        return RequestDecision(trace_session_id="s0", impose=("m", "p"))

    chain = ChainRequestHook([fn, HeaderIdentityHook(email_header="x-e"), sess])
    d = chain(_ctx([("x-e", "a@b.c")]))
    assert d == RequestDecision(
        impose=("m", "p"), user_email="a@b.c", trace_user_id="a@b.c",
        trace_session_id="s0",
    )
