# tests/test_server_compat_hook.py
"""0.1.29 — hook padrão de compatibilidade: o que a 0.1.28 lia dos headers."""

from __future__ import annotations

import pytest

from aixon.server.compat import (
    OPEN_WEBUI_CHAT_ID_HEADER,
    OPEN_WEBUI_EMAIL_HEADER,
    OPEN_WEBUI_USER_ID_HEADER,
    open_webui_request_hook,
)
from aixon.server.context import RequestDecision, make_context


def _ctx(headers):
    return make_context(model="m", agent_name="a", raw_headers=list(headers.items()),
                        params={}, adapter="openai", stream=False)


def test_nomes_dos_headers():
    assert OPEN_WEBUI_EMAIL_HEADER == "x-openwebui-user-email"
    assert OPEN_WEBUI_USER_ID_HEADER == "x-openwebui-user-id"
    assert OPEN_WEBUI_CHAT_ID_HEADER == "x-openwebui-chat-id"


@pytest.mark.parametrize("headers, esperado", [
    ({}, RequestDecision()),
    ({"x-openwebui-user-email": "a@b.c"},
     RequestDecision(user_email="a@b.c", trace_user_id="a@b.c")),
    ({"x-openwebui-user-id": "u1"}, RequestDecision(trace_user_id="u1")),
    ({"x-openwebui-chat-id": "c1"}, RequestDecision(trace_session_id="c1")),
    ({"x-openwebui-user-email": "a@b.c", "x-openwebui-user-id": "u1",
      "x-openwebui-chat-id": "c1"},
     RequestDecision(user_email="a@b.c", trace_user_id="a@b.c", trace_session_id="c1")),
    ({"x-openwebui-user-email": "", "x-openwebui-user-id": "u1"},
     RequestDecision(user_email="", trace_user_id="u1")),
], ids=["nenhum", "so-email", "so-user-id", "so-chat", "todos", "email-vazio"])
def test_padrao_reproduz_0128(headers, esperado):
    assert open_webui_request_hook(_ctx(headers)) == esperado


def test_padrao_le_header_sem_distinguir_caixa():
    d = open_webui_request_hook(_ctx({"X-OpenWebUI-User-Email": "a@b.c",
                                      "X-OpenWebUI-Chat-Id": "c1"}))
    assert d == RequestDecision(user_email="a@b.c", trace_user_id="a@b.c",
                                trace_session_id="c1")


def test_padrao_preserva_valor_cru():
    d = open_webui_request_hook(_ctx({"x-openwebui-user-email": "  a@b.c "}))
    assert d.user_email == "  a@b.c "
    assert d.trace_user_id == "  a@b.c "


def test_request_identity_delegada_e_avisa():
    from aixon.observability import request_identity

    headers = {"x-openwebui-user-email": "a@b.c", "x-openwebui-chat-id": "c1"}
    with pytest.warns(DeprecationWarning, match="request_hook"):
        assert request_identity(headers) == {"user_id": "a@b.c", "session_id": "c1"}
    with pytest.warns(DeprecationWarning):
        assert request_identity({"x-openwebui-user-id": "u1"}) == {
            "user_id": "u1", "session_id": None}
    with pytest.warns(DeprecationWarning):
        assert request_identity(object()) == {"user_id": None, "session_id": None}


# --- 0.1.30: OpenWebUIRequestHook (classe) ---


def test_padrao_e_instancia_da_classe():
    from aixon.server.compat import OpenWebUIRequestHook
    from aixon.server.context import HeaderIdentityHook

    assert isinstance(open_webui_request_hook, OpenWebUIRequestHook)
    assert isinstance(open_webui_request_hook, HeaderIdentityHook)


def test_subclasse_herda_identidade_e_sobrescreve_imposicao():
    from aixon.server.compat import OpenWebUIRequestHook

    class Mine(OpenWebUIRequestHook):
        def imposition(self, ctx):
            return (ctx.model.partition("-")[0].lower(), "gateway")

    ctx = make_context(model="Pro-Gerente", agent_name="a",
                       raw_headers=[(OPEN_WEBUI_EMAIL_HEADER, "a@b.c")],
                       params={}, adapter="openai", stream=False)
    assert Mine()(ctx) == RequestDecision(
        impose=("pro", "gateway"), user_email="a@b.c", trace_user_id="a@b.c"
    )


def test_subclasse_recusa_em_check():
    from aixon.exceptions import RequestRejected
    from aixon.server.compat import OpenWebUIRequestHook

    class Mine(OpenWebUIRequestHook):
        def check(self, ctx):
            if not ctx.headers.get(OPEN_WEBUI_EMAIL_HEADER):
                raise RequestRejected("sem e-mail")

    with pytest.raises(RequestRejected):
        Mine()(_ctx({}))
    assert Mine()(_ctx({OPEN_WEBUI_EMAIL_HEADER: "a@b.c"})).user_email == "a@b.c"
