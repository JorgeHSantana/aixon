# Decider

A minimal, **fully offline** demonstration of deciders, documented in
[docs/deciders.md](../../docs/deciders.md): closed questions about language,
answered by a decision engine with a measured certainty.

| File | What it holds |
|---|---|
| `deciders.py` | four deciders (choice, yes/no, yes/no with `redact`, scale) and the `Doubt` / `Check` entries an agent uses |
| `main.py` | the walk-through below, with an offline stand-in for the engine and a scripted model for the LLM |
| `examples.jsonl` | 12 labelled cases for `aixon decide eval` |

## Run it

```bash
cd examples/decider
PYTHONPATH=../.. python main.py
```

No key, no network. To put a real engine behind the same deciders — nothing in
`deciders.py` changes, the engine comes from the environment:

```bash
AIXON_DECIDER_PROVIDER=jev TYPESAFE_API_KEY=... PYTHONPATH=../.. python main.py
AIXON_DECIDER_PROVIDER=laya PYTHONPATH=../.. python main.py     # pip install aixon[laya]
```

The LLM stays scripted either way.

## What you see

**1. One decision, three outcomes.** A chain has a store called Olive Street,
and sells olives.

```
'Who is the manager at Olive?'
   -> store  certainty 0.96  sure  chances {'store': 0.96, 'product': 0.04}
'Do we have olive in stock?'
   -> product  certainty 0.94  sure  chances {'store': 0.06, 'product': 0.94}
'How much did olive sell in August?'
   -> store  certainty 0.55  UNSURE  chances {'store': 0.55, 'product': 0.45}
switched off -> no decision (engine off, no key, timeout): carry on without it
```

**2. Several decisions, one request.** `decide_many` asks "is the user upset?"
and "how urgent is it?" about the same message in a single call.

**3. An agent with `doubts` and `checks`.** Three questions to `GroceryAgent`:

- *"How much did olive sell in August?"* The engine is unsure, so the doubt
  tells the model to ask back. The agent answers
  `Do you mean the Olive Street store or the product olive?`
- *"What is the margin on olive oil?"* The engine is sure it is a product; the
  message reaches the model untouched.
- *"Great job, wrong again. How much did the Harbor store make in August?"*
  The tone doubt adds "acknowledge it first". The scripted model answers about
  **September** on purpose; the check catches it, the agent redoes the answer
  once, and only the second one goes out:
  `Sorry about that. In August the Harbor store made $ 41,200.`

`main.py` prints what the model received on each call, so you can see the
directives the doubts appended and the redo instruction of the check. It also
prints what a hosted engine would receive for the check: the answer with the
amount masked by `redact` (`... the Harbor store made #`).

**4. Calibration.**

```bash
PYTHONPATH=../.. aixon decide eval deciders:StoreOrProductDecider examples.jsonl
```

Offline this prints the table for the stand-in engine. Against a real engine
it is the table that decides the decider's `certainty` — see
[Calibrating the threshold](../../docs/deciders.md#calibrating-the-threshold-aixon-decide-eval).

The test `tests/test_example_decider.py` runs all of the above with the offline
engine.
