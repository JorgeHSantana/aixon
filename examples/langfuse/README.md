# Langfuse — runnable example (#26)

Mostra a integração de primeira classe do Server com o Langfuse: **por
agente** (nome do trace), **por usuário** (headers `X-OpenWebUI-User-*`),
**por modelo** (cada generation registra o modelo real do provider) — sem
nenhuma mudança de código no agente.

## Sem Langfuse (offline — demonstra o mecanismo)

```bash
cd examples/langfuse
PYTHONPATH=../.. python main.py
```

Sem as envs, o exemplo roda um agente com um handler de demonstração no
MESMO ponto de acoplamento do Langfuse (o *configure hook* do
langchain-core) e imprime os eventos capturados — a prova de que o handler
alcança grafo e modelos sem tocar nos call-sites.

## Com Langfuse de verdade

```bash
pip install "aixon[langfuse]"
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
export LANGFUSE_HOST="https://seu-langfuse"   # omitido = Langfuse Cloud

PYTHONPATH=../.. python main.py
```

Com as envs, o mesmo run sai como um trace `Suporte` no dashboard: span raiz
com `user_id`/`session_id`, generations por turno com o modelo e o usage de
cada um.

Num deploy real nada disso é chamado à mão: o **Server** faz o
`observe_request` por request (com a identidade vinda dos headers do Open
WebUI quando `ENABLE_FORWARD_USER_INFO_HEADERS=true`). Guia completo:
[docs/tracing.md](../../docs/tracing.md).
