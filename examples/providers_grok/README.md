# Provider xAI (Grok)

Exemplo mínimo de uso do provider xAI com o Grok na aixon.

## Requisitos

- `XAI_API_KEY`: chave de API da xAI (obrigatória)
- `XAI_BASE_URL` (opcional): URL base customizada do endpoint da xAI

## Como rodar

```bash
pip install -r requirements.txt
XAI_API_KEY=sk-... python main.py "sua pergunta aqui"
```

Sem argumentos, usa a pergunta padrão:

```bash
XAI_API_KEY=sk-... python main.py
```

## O que faz

Define um `LLMAgent` simples (`GrokAgent`) que:
- Usa o modelo `grok-4.5`
- Ativa `reasoning=True` (traduzido para `reasoning_effort` no endpoint OpenAI-compatível da xAI)
- Responde em uma frase

O nome `grok-*` é resolvido automaticamente para o provider xAI.
