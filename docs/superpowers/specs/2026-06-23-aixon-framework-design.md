# aixon — Design do Framework

**Data:** 2026-06-23
**Status:** Aprovado para implementação (pendente revisão final do spec)

## Contexto e objetivo

`aixon` é um framework de agentes de IA extraído do **olympus-ai-server**, seguindo o mesmo padrão que produziu o **restmcp** a partir do **mcp-financial-server** (e que depois foi usado para construir o mcp-diagnosis-server). O olympus continuará existindo como *consumidor* do `aixon`, assim como o mcp-diagnosis-server consome o restmcp.

O framework generaliza a camada reutilizável do olympus: servidor com protocolo desacoplado via `ProtocolAdapter` (OpenAI-compatible é apenas o primeiro adapter, não algo hardwired), agentes declarativos, orquestração multi-agente, RAG/busca, embeddings e clientes de microserviços.

### Objetivos declarados (requisitos)

1. **Suporte a LangGraph** como orquestrador de agentes.
2. **Tudo o mais declarativo possível** — subclasse + atributos de classe, zero fiação manual.
3. **Obrigatoriedade de sufixo nos nomes** — validada no `__init_subclass__`, falha antes do server subir (igual restmcp).
4. **Máximo desacoplamento** — em especial o protocolo "openai-like" deve ser trocável por outros estilos no futuro.
5. **CLI** com menu interativo de agentes.
6. **Documentação no nível do restmcp.**

## Nome

**`aixon`** — trocadilho *AI + axon* (o axônio conecta neurônios e transmite sinais → metáfora de orquestrar/conectar agentes). Namespace 100% livre no PyPI, import casado (`import aixon`), sem risco de colisão. Mantém a linha neural do nome anterior (`cortex`, que estava ocupado).

## Decisão arquitetural central: colapsar `Agent` + `Model`

O olympus tem **dois** conceitos quase 1:1 que são a maior fonte de ambiguidade ("model" sobrecarregado):

- `Agent` (pasta `agents/`) — identidade pública: registro no server, aliases, `hidden`, `owned_by`, API `chat()`, resolve um Model.
- `Model` (pasta `models/`) — motor de execução: `AgentExecutor`, `llm`, `prompt`, `tools`, agentes aninhados, `invoke()`, `as_tool()`.

A separação real que importa **não** é Agent-vs-Model, é **"unidade executável" vs "está publicada no server"** — e *publicado* é metadado/registro, não uma classe à parte.

**Resolução:** um único conceito **`Agent`** = unidade executável e componível (`invoke`/`stream`/`as_tool`, aninhável). Publicar no server vira registro/flag, não uma segunda classe. O que o olympus chama de `Model` vira `Agent`; o que ele chama de `Agent` se dissolve em metadado + camada de registro.

## Vocabulário (renomeações)

| Conceito | olympus | Problema | aixon |
|---|---|---|---|
| Unidade executável | `Model` | "model" sobrecarregado | **`Agent`** |
| Identidade pública | `Agent` | duplica Model | dissolve em registro + metadado |
| LLM puro exposto | `passthrough=True` | termo obscuro | **`LLMAgent`** (subtipo) |
| Tool-calling | `Model` (AgentExecutor) | colide | **`ToolAgent`** (subtipo) |
| Orquestrador grafo | `Model.agents` aninhado | implícito | **`Orchestrator`** (subtipo) |
| Declaração de LLM/provider | `BaseLLM`/`LLM()` | colidia com "Model" | **`LLM`** (mantido) |
| Campo `model` do request | `model` | é nome de agente | mapeia → nome do `Agent` |
| Cliente de microserviço externo | `Service` | colide com `Service` do restmcp | **`Connector`** |
| Busca de contexto (RAG/Store/search) | `RAG`+`Store`+`storage` | três termos | **`Retriever`** |
| Cache de busca | `storage` | parece storage genérico | **`cache`** |
| Streaming de raciocínio | `thought` | formato indefinido | **`reasoning`** |
| Provider de embeddings | `Embedding` | ok | **`Embedding`** (mantido) |

### Por que `LLM` e não `Model`

Nome de framework cai em código alheio cheio de "Model" (Pydantic `BaseModel`, ORM, data models). `from aixon import LLM` é inconfundível em qualquer codebase; `from aixon import Model` colidiria. Embedding/reranker/visão já são conceitos separados, então `LLM` não limita a expansão.

## Os três subtipos de `Agent`

Todos cumprem a **mesma interface** (`invoke`/`stream`/`as_tool`), então o server, o registry e a camada de protocolo nunca precisam saber qual é qual.

- **`LLMAgent`** — LLM direto, sem loop de tool-calling, sem grafo. Substitui o `passthrough`. Caminho limpo e sem overhead para expor uma LLM pura.
- **`ToolAgent`** — loop de tool-calling (o `AgentExecutor` atual).
- **`Orchestrator`** — grafo LangGraph (ver abaixo).

Regra mental: LLM puro → `LLMAgent`; LLM + ferramentas → `ToolAgent`; vários agentes coordenados → `Orchestrator`.

### Exemplos declarativos

```python
class Athena(LLMAgent):
    llm = LLM("gpt-5.4", temperature=0.2)
    description = "Planejador estratégico"

class Diagnosis(ToolAgent):
    llm = LLM("gpt-5.4", temperature=0.1)
    prompt = "..."
    tools = [LibraryRetriever, check_battery]
```

## Orchestrator — API declarativa em 3 camadas

Estratégia para **não ficar complexo de usar**: a camada simples resolve ~80% dos casos; a explícita é escape para grafos de verdade; o callback é o teto (LangGraph cru). Todas declarativas, todas subclasse de `Orchestrator`, sufixo `*Orchestrator`.

**Ordem de detecção:** `build_graph` sobrescrito → Camada 3 · senão `nodes` declarado → Camada 2 · senão `supervisor`+`agents` → Camada 1.

### Camada 1 — Supervisor (padrão)

```python
class SupportOrchestrator(Orchestrator):
    description = "Atendimento que roteia entre especialistas"
    supervisor  = LLM("gpt-5.4")                       # o roteador
    agents      = [BillingAgent, TechAgent, Athena]    # workers (qualquer Agent)
    # supervisor decide quem atende cada turno e faz loop até concluir.
```

### Camada 2 — Grafo explícito

```python
class TriageOrchestrator(Orchestrator):
    description = "Triagem com roteamento condicional"

    nodes = {                       # nome -> qualquer Agent
        "triage":   TriageAgent,
        "diagnose": Diagnosis,
        "respond":  Athena,
    }
    entry = "triage"
    edges = [                       # arestas FIXAS (incondicionais)
        ("diagnose", "respond"),
        ("respond",  END),
    ]

    def route_triage(self, state) -> str:     # aresta condicional: convenção route_<nó>
        return "diagnose" if state.needs_diagnosis else "respond"
```

**Mecânica de execução (a documentar com destaque):**
- `edges` **não é uma sequência nem uma etapa** — é só a fiação (regras "depois de X vai pra Y"). A ordem textual da lista é irrelevante.
- Quem roda primeiro é o **`entry`**; a ordem de execução vem do `entry` + topologia, não da ordem em `edges`.
- Cada nó tem **uma** forma de saída: ou uma aresta fixa em `edges`, ou um `route_<nó>`. Declarar ambos para o mesmo nó → erro no `__init_subclass__` (saída ambígua). Nenhum dos dois → nó terminal.

**Dois tipos de bifurcação (via `route_<nó>`):**
1. **Condicional (escolhe um caminho):** retorna o nome do próximo nó.
2. **Fan-out paralelo (vários ao mesmo tempo):** retorna uma *lista* de nós; rodam em paralelo e o grafo junta depois.

### Camada 3 — Callback / LangGraph cru (escape hatch)

```python
class WeirdOrchestrator(Orchestrator):
    description = "..."
    def build_graph(self):              # LangGraph puro; o framework só roda o que voltar
        g = StateGraph(self.State)
        g.add_node(...); g.add_conditional_edges(...)
        return g.compile()
```

### Estado

Declarativo e opcional. Default carrega `messages` + `reasoning` (usuário não toca). Estado extra:

```python
class TriageOrchestrator(Orchestrator):
    class State(GraphState):
        needs_diagnosis: bool = False
```

### Orchestrator como `as_tool()`

Como `Orchestrator` cumpre a interface `Agent`, `as_tool()` é uniforme (subgrafo como ferramenta; supervisor-de-supervisor sai de graça). Três pontos tratados:

- **Isolamento de estado** — de graça: cada `invoke()` roda com seu próprio `State`/histórico.
- **Propagação de `reasoning`** — o raciocínio do subgrafo borbulha pro stream do pai pelo mesmo mecanismo (canal de reasoning) dos agentes aninhados.
- **Guarda de recursão** — ver abaixo.

### Proteção contra recursão (dois tipos distintos)

**A) Ciclo de composição (estrutural)** — `A` usa `B` as_tool e `B` usa `A` as_tool, ou um orquestrador que se inclui. A expansão da árvore de tools nunca termina. **Proteção sempre ligada, não desabilitável:** no `__init_subclass__`/build, caminhar o grafo de composição; revisitar uma classe já no caminho atual → erro claro no startup. (Ciclo *dentro* de um grafo — um nó que volta — é legítimo e continua permitido; quem o limita é o item B.)

**B) Profundidade/loop em runtime** — atributo declarativo:

```python
class FooOrchestrator(Orchestrator):
    recursion_limit = 50    # supersteps do grafo. Default: 25 (igual LangGraph). None = sem teto.
    timeout         = 600   # backstop wall-clock (segundos). Rede de segurança final.
```

"Sem limites" = `recursion_limit=None`, ainda limitado por `timeout` (custo/tempo nunca explodem). Literalmente nada = `recursion_limit=None` + `timeout=None` (não recomendado).

## Desacoplamento de protocolo

Separar o **runtime do agente** (fala só tipos neutros — `Message[]` entra, `Message`/`Chunk` sai) da **camada de protocolo** (traduz formato de fio ↔ tipos neutros). OpenAI-compatible vira o primeiro `ProtocolAdapter`.

**Decisão:** construir a costura (`ProtocolAdapter` + tipos neutros) **e dois adapters**:
- **`OpenAIAdapter`** — completo.
- **`AnthropicAdapter`** — fino, **prova** de que a abstração aguenta um estilo estruturalmente diferente (system fora do array, blocos de content tipados, envelope `content[]`/`stop_reason`, eventos de streaming nomeados). Garante que os "tipos neutros" são realmente neutros e não OpenAI disfarçado.

Adicionar outro estilo depois = nova classe `*Adapter`.

## CLI

```
aixon chat            # menu interativo de agentes
aixon new my-server   # scaffold de projeto consumidor
aixon serve           # sobe o server (atalho uvicorn)
aixon list            # lista agentes registrados
```

### Fluxo do `aixon chat`

1. `autodiscover()` no projeto consumidor, lê o registry.
2. **Menu** — só agentes não-`hidden`, com tipo e descrição.
3. Escolhe → **chat** com stream de `reasoning` (esmaecido) + `content`.
4. Comandos no chat: `/menu` (volta pro menu), `/exit` (sai), `Ctrl+C` (interrompe a geração atual; de novo no prompt vazio → volta pro menu).

UX `/menu` + `Ctrl+C` escolhida em vez de hotkeys raros (Esc/setas) por estes exigirem terminal em raw-mode e serem frágeis entre OSes.

### Dois modos (mesmo comando)

- **In-process** (default): invoca agentes direto, sem server rodando. Para dev.
- **Remoto**: `aixon chat --url http://host:porta` — cliente OpenAI contra server `aixon` no ar (reaproveita o `OpenAIAdapter`).

## Layout do pacote

```
aixon/
├── aixon/
│   ├── __init__.py          # API pública
│   ├── agent.py             # Agent — base + interface invoke/stream/as_tool
│   ├── message.py           # Message / Chunk / Role — tipos neutros
│   ├── agents/
│   │   ├── llm_agent.py     # LLMAgent
│   │   ├── tool_agent.py    # ToolAgent
│   │   └── orchestrator.py  # Orchestrator (3 tiers)
│   ├── llm.py               # LLM — declaração + factory + registry de modelos
│   ├── providers/           # adapters de provider (openai/anthropic/google)
│   ├── _interop/            # fronteira LangChain privada (messages.py, tools.py)
│   ├── retriever.py         # Retriever — base de busca de contexto
│   ├── embedding.py         # Embedding — base + OpenAIEmbedding (lazy)
│   ├── connector.py         # Connector — cliente de microserviço externo
│   ├── server/
│   │   ├── server.py        # singleton + registry + app ASGI + auth
│   │   ├── protocol.py      # ProtocolAdapter (interface) + tipos neutros
│   │   └── adapters/
│   │       ├── openai.py    # OpenAIAdapter (completo)
│   │       └── anthropic.py # AnthropicAdapter (fino, prova)
│   ├── discovery.py         # autodiscover()
│   ├── registry.py          # registro de agentes
│   ├── reasoning.py         # canal de reasoning (stream, propaga no aninhamento)
│   ├── state.py             # GraphState / END
│   ├── logging.py           # Logger
│   ├── exceptions.py
│   └── cli.py               # click: list / chat / new / serve
├── docs/
├── pyproject.toml
└── tests/
```

## Mapa de abstrações e sufixos

| Camada | Conceito | Responsabilidade | Sufixo |
|---|---|---|---|
| Execução | `Agent` | unidade invocável/componível | `*Agent` |
| ↳ subtipo | `LLMAgent` | LLM direto | `*Agent` |
| ↳ subtipo | `ToolAgent` | tool-calling | `*Agent` |
| ↳ subtipo | `Orchestrator` | grafo LangGraph | `*Orchestrator` |
| Modelo | `LLM` | declara modelo/provider | — |
| Contexto | `Retriever` | busca (vetor/web/híbrido), `as_tool()` | `*Retriever` |
| Contexto | `Embedding` | provider de embeddings | `*Embedding` |
| Integração | `Connector` | cliente de microserviço externo | `*Connector` |
| Borda | `ProtocolAdapter` | traduz fio ↔ tipos neutros | `*Adapter` |
| Borda | `Server` | transporte ASGI + registry + auth | — |

## Fluxo de um request

```
HTTP → Transport(ASGI) → ProtocolAdapter (OpenAI|Anthropic ↔ Message[]/Chunk neutros)
     → Server resolve Agent pelo nome (campo "model" do request)
     → Agent.invoke/stream  (fala SÓ tipos neutros)
          ├ LLMAgent     → LLM
          ├ ToolAgent    → executor → tools: Retriever.as_tool · @tool · Agent.as_tool · Connector.métodos
          └ Orchestrator → grafo → nós (Agents) → ... (recursion_limit + timeout)
     → canal de reasoning borbulha de volta pelo aninhamento → adapter formata o stream
```

## Garantias de desacoplamento

1. **Protocolo trocável** — nenhum tipo OpenAI cruza pra dentro do `Agent`; novo estilo = nova classe `*Adapter`. Provado pelo `AnthropicAdapter`.
2. **Runtime trocável** — `LLMAgent`/`ToolAgent`/`Orchestrator` intercambiáveis sob a mesma interface.
3. **Provider trocável** — `LLM` esconde openai/anthropic/google atrás de `providers/`.
4. **Tudo declarativo** — subclasse + atributos; `__init_subclass__` valida sufixo e detecta ciclo de composição antes do server subir.

## Documentação (entregável)

README + `docs/`, nível restmcp: filosofia, arquitetura em camadas, modelo `Agent`/subtipos, API declarativa, regras de sufixo, os 3 tiers do Orchestrator (incluindo a mecânica `entry`/topologia vs. ordem textual de `edges`, e os dois tipos de branching), `ProtocolAdapter`/desacoplamento, `Retriever`/`Connector`/`LLM`/`Embedding`, CLI e quickstart de projeto consumidor.

## Empacotamento

`pyproject.toml` no padrão restmcp (hatch): nome `aixon`, `requires-python >=3.11`, `[project.scripts] aixon = "aixon.cli:app"`. Dependências core (obrigatórias): `langchain`, `langchain-core`, `langgraph` — o framework não funciona sem elas. Todo o resto é extra opcional: `server` (`fastapi`, `uvicorn[standard]`, `pydantic`, `httpx`), `cli` (`click`, `openai`), `openai`/`anthropic`/`google` (bindings de provider — `langchain-openai`/`langchain-anthropic`/`langchain-google-genai`, carregados lazy), `retrieval` (`httpx>=0.27` para `Connector`), `openai-embedding` (`langchain-openai` para `OpenAIEmbedding`), `dev`, e `all` (agrega os extras). Backends de vector store (Weaviate/Ragie/Tavily) estão fora de escopo — YAGNI.

## Fora de escopo (YAGNI)

- Adapters de protocolo além de OpenAI (completo) e Anthropic (fino de prova).
- Implementações concretas específicas do domínio olympus (prompts, tools de diagnóstico, collections Weaviate) — vão para o consumidor.
- Persistência/checkpointing do LangGraph além do default (namespacing só quando habilitado).
