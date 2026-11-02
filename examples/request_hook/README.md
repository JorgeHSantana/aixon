# Contexto por request (`request_hook`)

A aplicação decide, a cada request, **qual modelo a árvore do agente usa**,
**quem é o usuário** e **se a request entra** — a partir dos cabeçalhos, sem
mexer nas classes dos agentes.

## O que o hook decide

`tier_hook(ctx)` recebe um `RequestContext` e devolve um `RequestDecision`:

| Cabeçalho | Efeito |
|---|---|
| `x-tier: fast` | impõe `("gpt-5.4-mini", "openai")` |
| `x-tier: deep` | impõe `("gpt-5.4", "openai")` |
| `x-tier` ausente | nada imposto: vale o `llm` da classe |
| `x-user` | vira `user_email` e `trace_user_id` |
| `x-user` ausente | `RequestRejected("x-user header is required", status=401, code="no_user")` |

## Ciclo de vida

```
parse → resolve (agente) → hook → escopos → agente
```

O Server interpreta o corpo, resolve o agente pelo `model`, chama o hook uma
vez, abre os escopos da decisão (modelo imposto, e-mail, identidade do trace)
e só então roda o agente.

## Como rodar

```bash
pip install -r requirements.txt
OPENAI_API_KEY=sk-... PYTHONPATH=../.. python main.py
```

## Os três casos

```bash
# fast → gpt-5.4-mini
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -H 'x-user: ana@example.com' -H 'x-tier: fast' \
  -d '{"model":"helper","messages":[{"role":"user","content":"oi"}]}'

# deep → gpt-5.4
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -H 'x-user: ana@example.com' -H 'x-tier: deep' \
  -d '{"model":"analyst","messages":[{"role":"user","content":"oi"}]}'

# sem usuário → 401, code "no_user"
curl -s -i localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"helper","messages":[{"role":"user","content":"oi"}]}'
```

O teste `tests/test_example_request_hook.py` roda esses três casos contra um
provedor fake (sem rede, sem chave).
