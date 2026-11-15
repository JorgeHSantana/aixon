# Deciders

A **decider** answers a closed question about a context and tells you how sure
it is. It does not generate text and it does not call tools. It is the piece
you reach for when an agent has to *choose* — which of these two things the
user means, whether the user is upset, whether an answer talks about what was
asked — and you want that choice made in ~250 ms by an engine built for it,
with a number you can put a threshold on.

```python
from aixon import Decider

class StoreOrProductDecider(Decider):
    provider = "jev"
    instruction = "Decide what the word in 'word' refers to in the user's 'message'."
    options = {
        "store":   "a store of the chain: a place, 'the' + name, the store's revenue",
        "product": "an item for sale, a category or an ingredient",
    }
    redact = None      # nothing to mask in this context (see "Data that leaves the process")

decision = StoreOrProductDecider.decide({"message": text, "word": "olive"})
```

Runnable, offline: [examples/decider](../examples/decider).

- [The three outcomes](#the-three-outcomes)
- [Declaring a decider](#declaring-a-decider)
- [Asking](#asking)
- [What a decider is for, and what it is not](#what-a-decider-is-for-and-what-it-is-not)
- [In a ToolAgent: `doubts` and `checks`](#in-a-toolagent-doubts-and-checks)
- [Engines](#engines)
- [Data that leaves the process: `redact`](#data-that-leaves-the-process-redact)
- [Calibrating the threshold: `aixon decide eval`](#calibrating-the-threshold-aixon-decide-eval)
- [Tracing and logs](#tracing-and-logs)
- [Testing your deciders](#testing-your-deciders)
- [Writing a provider](#writing-a-provider)
- [Environment variables](#environment-variables)
- [Reference](#reference)

---

## The three outcomes

Every call ends in exactly one of three states, and your code handles the three:

```python
decision = StoreOrProductDecider.decide({"message": text, "word": "olive"})

if decision is None:
    ...   # NO DECISION: engine off, no key, timeout, bad reply. Carry on as if
          # the decider did not exist.
elif decision.sure:
    ...   # SURE: decision.choice == "store", decision.certainty == 0.98. Act on it.
else:
    ...   # UNSURE: the engine would be guessing. Take the fallback — usually,
          # ask the user which one they mean.
```

| Outcome | When | What you do |
|---|---|---|
| `None` | the decider is switched off, the provider has no key or is not installed, the call timed out or failed, the reply was malformed | nothing: the request goes on without the decision |
| `decision.sure` | `decision.certainty >= <Decider>.certainty` | act on `decision.choice` |
| not `sure` | below the threshold | the fallback you declared |

`decide` **never raises at call time**. A decision is an improvement on top of
an agent that already works; it must never be the reason a request fails.
Mistakes in the *declaration* (no instruction, one option, a hosted engine
without `redact`) do raise — `AixonError`, when the class is defined, like the
rest of aixon.

`Decision`:

| Field | Meaning |
|---|---|
| `choice` | the option (or scale level) chosen |
| `certainty` | 0..1. The engine's calibrated confidence when it reports one, otherwise the chance of `choice` |
| `chances` | chance of every option, e.g. `{"store": 0.99, "product": 0.01}` |
| `sure` | `certainty >= <Decider>.certainty` |
| `chance(option)` | chance of one option; `0.0` if the engine did not report it |
| `score` | fractional position on a scale (`0` = first level); scale deciders only |
| `decider`, `provider`, `model` | who answered |
| `latency_ms`, `usage` | what it cost (`usage` in the OpenAI shape, when the engine reports tokens) |

`certainty` is not always the chance of the choice. An engine that reports its
own confidence may return `choice="product"`, `chances={"store": 0.38,
"product": 0.62}` and `certainty=0.24`: it leans to "product" but says it would
be guessing. Threshold on `certainty` (that is what `sure` does), not on
`chances`.

---

## Declaring a decider

Subclass `Decider`, set class attributes, never instantiate. The class name
must end with `Decider`.

| Attribute | Required | Default | Meaning |
|---|---|---|---|
| `instruction` | yes | — | the question the engine answers. Say what each key of the context holds |
| `options` | one of the two | — | `{option: criterion}`; the criterion tells the engine when the option applies (may be `None`). A list of names also works. `{"yes", "no"}` makes a yes/no question |
| `scale` | one of the two | — | ordered level labels, lowest first, for an ordinal grade |
| `provider` | no | `None` | `"jev"`, `"laya"`, a `DeciderProvider` instance, or `None` to read `AIXON_DECIDER_PROVIDER` at call time |
| `model` | no | provider default | engine model, or a checkpoint path for a local engine |
| `certainty` | no | `0.8` | threshold of `Decision.sure` |
| `timeout` | no | `3.0` | seconds a hosted engine may take |
| `switch` | no | `None` | name of an environment variable; `0` / `off` / `false` / `no` turns this decider off |
| `redact` | on hosted engines | — | `(context) -> context`, applied before the context leaves the process; `None` = declared as not needed |
| `name` | no | class name in snake_case, without the suffix | identifier in logs and traces |

Three kinds of question:

```python
class StoreOrProductDecider(Decider):          # choice
    instruction = "..."
    options = {"store": "...", "product": "..."}

class UpsetDecider(Decider):                   # yes / no
    instruction = "Is the user upset with the assistant in 'message'?"
    options = {
        "yes": "a complaint, an accusation of error, or irony",
        "no":  "a plain request or a neutral follow-up",
    }

class UrgencyDecider(Decider):                 # scale
    instruction = "How urgent is the user's request in 'message'?"
    scale = ["can wait", "today", "right now"]
```

Shared defaults go in an abstract base:

```python
class HouseDecider(Decider, abstract=True):
    provider = "jev"
    certainty = 0.8
    redact = None

class UpsetDecider(HouseDecider):
    instruction = "..."
    options = ["yes", "no"]
```

**Write the criteria.** The single biggest lever on quality is the text next to
each option. `"store": "a store of the chain: a place, 'the' + name, the
store's revenue or staff"` beats `"store": None` by a wide margin, because the
engine has no other way to know what "store" means in your domain.

---

## Asking

```python
decision = UpsetDecider.decide({"message": text})          # sync
decision = await UpsetDecider.adecide({"message": text})   # async (runs in a worker thread)
```

The context is a plain JSON-able `dict`. Name the keys in the `instruction`.

Several decisions about the **same context** cost the time of one — deciders
that share provider, model and `redact` travel in a single request:

```python
from aixon import decide_many

out = decide_many({"message": text}, [UpsetDecider, UrgencyDecider])
out[UpsetDecider]      # Decision | None
out[UrgencyDecider]    # Decision | None
```

`aixon.decider.adecide_many` is the async twin. One group failing does not
affect another.

---

## What a decider is for, and what it is not

Measured on a production agent (~160 benchmark questions, ~2,000 real answers):

| Task | Result |
|---|---|
| "Is this assistant message a question back to the user?" | 34 of 34, where a regex got 13 |
| Irony in user messages (public PT-BR dataset) | 30 of 30, 2 false alarms in 60 |
| "Does the answer address what was asked?" (amounts masked) | 0.25% false alarms at a 0.3 failure line |
| Choosing which tool answers a question | 98.8% |
| "Store or product?" with the options on the table | no confident error on clear cases |
| Judging whether a **number** in an answer is right | 69% — do not use it for this |
| "Is this message ambiguous?" (open, no options) | flagged 32 of 39 clear messages — do not ask this |

So:

- **Use it for language**: tone, intent, which-of-these, does-this-follow.
- **Never for numbers or arithmetic.** It reads; it does not compute.
- **Ask concrete questions.** Put the candidates in `options`. "Is this
  ambiguous?" has no usable answer; "store or product?" does.
- **Find the case in code first.** A cheap, pure function that detects "there
  is a store name loose in this message" decides whether to call the engine at
  all. That is what `Doubt.applies` is for.
- **Measure before turning one on.** A doubt that fires on clear messages makes
  the agent ask back when it should answer. See
  [`aixon decide eval`](#calibrating-the-threshold-aixon-decide-eval).

---

## In a ToolAgent: `doubts` and `checks`

Two declarative lists on `ToolAgent`. Both are empty by default, and an agent
that leaves them empty runs exactly as it did before.

```python
from aixon import Check, Doubt, LLM, ToolAgent

class GroceryAgent(ToolAgent):
    llm = LLM("gpt-4o-mini")
    tools = [sales, stock]
    doubts = [                                   # BEFORE the model
        Doubt(StoreOrProductDecider, applies=find_store_word,
              on_sure=store_is_meant, on_unsure=ask_store_or_product),
        Doubt(UpsetDecider, on_sure=acknowledge),
    ]
    checks = [                                   # AFTER the model
        Check(AnswersTheQuestionDecider, redo=redo),
    ]
```

Both cover `invoke`, `ainvoke`, `stream` and `astream`. Adding a case is adding
a line.

### `Doubt` — read the user's message

A doubt is something in the user's last message that could be read two ways.

```python
def find_store_word(message: str) -> dict | None:
    """Pure and cheap. Returns the concrete case, or None when the doubt is
    not in this message — and then the engine is not called."""
    for word, store in STORES.items():
        if word in message.lower().split():
            return {"word": word, "store": store}
    return None

def store_is_meant(decision, case) -> str | None:
    if decision.choice != "store":
        return None                               # nothing to add
    return f"[The user means the {case['store']}. Filter by that store.]"

def ask_store_or_product(decision, case) -> str:
    return (f"[The word '{case['word']}' may be the {case['store']} or the product. "
            f"Do not look anything up yet. Reply with one short question asking which one.]")
```

| Field | Meaning |
|---|---|
| `decider` | who settles it |
| `applies` | `(message) -> case \| None`. `None` / `False` = not in this message, engine not called. Omit to ask on every message (a tone decider) |
| `context` | `(message, case) -> context`. Default: `{"message": message, **case}` |
| `on_sure` | `(decision, case) -> directive \| None` when the engine is sure |
| `on_unsure` | `(decision, case) -> directive \| None` when it is not |

The directive is **appended to the user's last message** for the model
(`"<message>\n\n<directive>"`); the conversation you passed in is not mutated.
Doubts whose contexts are identical share one request. A doubt with no
decision adds nothing. Doubts are only read when the conversation ends on a
user turn.

What the model receives for `"How much did olive sell in August?"` when the
engine is unsure:

```
How much did olive sell in August?

[The word 'olive' may be the Olive Street store or the product. Do not look
anything up yet. Reply with one short question asking which one the user means.]
```

### `Check` — read the answer

A check is a test the answer must pass. Failing it triggers **one** second
attempt: the model gets the conversation, its own failed answer, and your
instruction.

```python
class AnswersTheQuestionDecider(Decider):
    provider = "jev"
    instruction = ("'message' is what the user asked and 'answer' the assistant's reply, "
                   "with every amount replaced by '#'. Does the answer address what was "
                   "asked: same period, same store, same subject?")
    options = {"yes": "same period, store and subject, or an honest 'I could not find it'",
               "no":  "another period, store or subject, or no answer at all"}
    redact = staticmethod(mask_values)

def redo(decision, answer) -> str:
    return ("Your answer does not address what was asked (check the period and the store). "
            "Answer again, about exactly what was asked.")

checks = [Check(AnswersTheQuestionDecider, redo=redo)]
```

| Field | Meaning |
|---|---|
| `decider` | who judges. With a yes/no decider, "yes" means the answer is fine |
| `redo` | `(decision, answer) -> instruction` for the second attempt |
| `fails` | `(decision) -> bool`. Default, yes/no deciders only: the chance of "yes" is below `fails_below` |
| `fails_below` | default `0.3` |
| `context` | `(message, answer) -> context`. Default: `{"message": message, "answer": answer}` |

Rules of the second attempt:

- At most **one**. The second answer is not checked again.
- Only a real answer is checked: not a client tool call, not empty, not the
  timeout text.
- If the second attempt raises or comes back empty, **the first answer
  stands**. A check never makes things worse.
- `Message.usage` covers both attempts. A reasoning line (`check_redo_label`,
  default `"Reviewing the answer..."`, `""` to silence) marks the redo.
- In `stream` / `astream`, reasoning chunks flow live; the answer is held until
  the checks pass, so a rejected answer is never shown. (A `ToolAgent` only
  emits its answer at the end of the run anyway, so nothing is delayed.)
- The check judges the user's **own words**, without the directives the doubts
  added; the second attempt keeps those directives.

**Why `fails_below` is 0.3 and not 0.5.** A check that fires on doubt rewrites
good answers. On ~2,000 real answers, failing below 0.3 gave 0.25% false alarms
and failing below 0.5 gave 1.9%. A check should catch the answer that is
plainly about something else, and stay out of the way otherwise.

A decider with more than two options needs an explicit rule:

```python
Check(PeriodDecider, fails=lambda d: d.sure and d.choice == "other", redo=...)
```

### Using the pieces without a ToolAgent

The same machinery is available as functions (`aixon.decider`):

```python
from aixon.decider import apply_doubts, failed_checks, redo_messages, resolve_doubts

outcomes = resolve_doubts(text, DOUBTS)              # [DoubtOutcome(decision, case, outcome, directive)]
messages = apply_doubts(messages, DOUBTS)            # conversation with directives appended
failures = failed_checks(text, answer, CHECKS)       # [CheckFailure(check, decision, instruction)]
retry    = redo_messages(messages, answer, failures) # conversation for the second attempt
```

Async twins: `aresolve_doubts`, `aapply_doubts`, `afailed_checks`.

---

## Engines

| Provider | Where it runs | Training | Key / install | Notes |
|---|---|---|---|---|
| `"jev"` | hosted (TypeSafe AI), one HTTP call | none: zero-shot | `TYPESAFE_API_KEY`; no SDK, no extra | ~250 ms, up to 255 options, reports a calibrated confidence. Needs `redact` |
| `"laya"` | local, in-process, CPU or GPU | fine-tunable (the base checkpoint is weak zero-shot on domain questions) | `pip install aixon[laya]` | ~40 ms after the first call, which loads the checkpoint. Nothing leaves the machine |
| `LLMDeciderProvider(llm)` | wherever the `LLM` runs | — | the LLM's own | for comparison and as a stand-in; its chances are the model's own estimate, not a calibrated probability |
| `FakeDeciderProvider(...)` | memory | — | — | tests and offline examples |

Switching engine is one attribute — or no code at all, if the decider leaves
`provider` unset and the environment chooses:

```python
class UpsetDecider(Decider):      # provider not declared
    instruction = "..."
    options = ["yes", "no"]
    redact = None
```

```bash
AIXON_DECIDER_PROVIDER=jev   TYPESAFE_API_KEY=...   # hosted
AIXON_DECIDER_PROVIDER=laya  LAYA_MODEL=/models/my-checkpoint   # local
```

### Jev

```python
class UpsetDecider(Decider):
    provider = "jev"
    model = "jev-latest"          # default; or pin a version
    ...
```

Without `TYPESAFE_API_KEY` the provider reports `no_key`: nothing is sent and
every decider on it returns `None`. That is how you ship the code before the
key exists in an environment.

### Laya

```python
class JudgeDecider(Decider):
    provider = "laya"
    model = "/models/judge-v3"    # a local checkpoint directory, or a Hugging Face id
    ...
```

The `laya` package (and torch) is imported on the first decision, never at
`import aixon`. Each `(model, device)` is loaded once per process;
inference is serialized by a lock. `timeout` is not enforced on a local forward
pass. Training a checkpoint is done with Laya's own tooling; aixon only loads
the result.

### An LLM as the engine

```python
from aixon import LLM
from aixon.deciders import LLMDeciderProvider

class UpsetDecider(Decider):
    provider = LLMDeciderProvider(LLM("gpt-4o-mini", temperature=0))
    ...
    redact = None
```

Useful to compare a chat model against a decision engine on the same examples
(`aixon decide eval`), or where no engine is available. It is slower, costs a
completion, and the certainty is self-reported: calibrate before trusting
`sure`.

---

## Data that leaves the process: `redact`

A hosted engine receives your context. A decider on a hosted provider must
therefore say what it does about that — otherwise the class does not define:

```
AixonError: Decider 'AnswersTheQuestionDecider': provider 'jev' is hosted: the
context leaves this process. Declare `redact = <function>` to mask it first, or
`redact = None` to state that nothing in it needs masking.
```

```python
import re

def mask_values(context: dict) -> dict:
    """Every amount becomes '#'. Dates and store names stay: the engine needs
    them to judge the answer; it never needs the figures."""
    masked = dict(context)
    if "answer" in masked:
        masked["answer"] = re.sub(r"\$ ?[\d.,]+|\b\d[\d.,]*\b", "#", masked["answer"])
    return masked

class AnswersTheQuestionDecider(Decider):
    provider = "jev"
    redact = staticmethod(mask_values)
    ...
```

- `redact` receives a copy; your context is not modified.
- If `redact` raises, there is **no decision** and nothing is sent.
- The trace span records the context **after** redaction — what actually left.
- A local provider (`laya`) does not require it.
- When the provider comes from `AIXON_DECIDER_PROVIDER` and turns out to be
  hosted on a decider that never declared `redact`, the decider gives no
  decision (and logs why) instead of sending the context.

---

## Calibrating the threshold: `aixon decide eval`

A decider is worth turning on when, at its threshold, it is almost never
**wrong with certainty** and rarely in doubt on cases that were clear. Find out
with labelled examples before production does it for you.

`examples.jsonl` — one object per line; `expected` is the right option, or
`null` when the right behaviour is to be **unsure**:

```json
{"context": {"message": "Who is the manager at Olive?", "word": "olive"}, "expected": "store"}
{"context": {"message": "Do we have olive in stock?", "word": "olive"}, "expected": "product"}
{"context": {"message": "How much did olive sell in August?", "word": "olive"}, "expected": null}
```

```bash
aixon decide eval deciders:StoreOrProductDecider examples.jsonl
aixon decide eval deciders:StoreOrProductDecider examples.jsonl -t 0.7,0.8,0.9
```

The first argument is `package.module:ClassName`, imported from the current
directory. Output of the example's 12 cases against the real Jev engine:

```
decider: store_or_product    provider: jev
examples: 12    answered: 12    no decision: 0
choice accuracy (ignoring certainty): 100.0%
median latency: 220 ms

certainty   right  WRONG, sure  right doubt  needless doubt
     0.50       8            3            1               0
     0.60       8            3            1               0
     0.70       8            3            1               0
     0.80       8            2            2               0  <- current
     0.90       8            2            2               0
     0.95       8            1            3               0
```

| Column | Meaning |
|---|---|
| `right` | sure and correct |
| `WRONG, sure` | sure and wrong — or sure where doubt was expected. Nothing downstream catches these: keep it at zero or near it |
| `right doubt` | unsure where doubt was expected |
| `needless doubt` | unsure on a case that had a right answer: a question back the user did not need |

Read it like this: all 8 clear cases are right at every threshold, and none
falls into needless doubt, so the threshold costs nothing on clear cases. Of the
4 cases labelled ambiguous, the engine stays sure about 2 even at 0.9 — worth
reading those two before deciding whether the label or the criteria is what
needs fixing. Raise `certainty` only as far as `WRONG, sure` needs.

The command exits with status 1 when no example was answered (missing key,
decider switched off), so it can gate a CI job. The same report is available in
code: `aixon.deciders.evaluation.evaluate(decider, examples)`.

---

## Tracing and logs

With the [Langfuse integration](tracing.md) on, every decision becomes one span
named `decider:<name>` under the request's trace:

- **input**: the context after `redact`;
- **output**: `choice`, `certainty`, `sure`, `chances` — or `{"no_decision": "<reason>"}`;
- **metadata**: provider, model, `latency_ms`, `usage`, the decider's threshold.

A failed call (timeout, HTTP error, malformed reply) is a span at `WARNING`
level. A decider that is switched off or has no key leaves no span: that is
configuration, not an event.

Log lines (`aixon.decider`):

```
INFO    decider 'upset' -> yes (certainty 0.95, sure) via jev in 231 ms
WARNING decider 'upset' -> no decision (TimeoutError)
INFO    agent 'grocery': answer failed check(s) [answers_the_question]; second attempt
```

Decision tokens are reported on `Decision.usage` and in the span. They are not
added to `Message.usage`, which stays the chat model's bill.

---

## Testing your deciders

`FakeDeciderProvider` scripts the engine, so your doubts, checks and hooks are
tested without a key:

```python
from aixon.deciders import FakeDeciderProvider

def test_agent_asks_back_when_unsure(monkeypatch):
    monkeypatch.setattr(StoreOrProductDecider, "provider",
                        FakeDeciderProvider({"store_or_product": ("store", 0.55)}))
    ...
```

| Script | Engine behaviour |
|---|---|
| `{"upset": "yes"}` | answers "yes", certainty 1.0 |
| `{"upset": ("yes", 0.55)}` | answers "yes", unsure |
| `{"level": RawAnswer(choice="today", chances={...}, score=1.2)}` | a full answer |
| `lambda context, name, question: ...` | computed per call |
| `FakeDeciderProvider(error=TimeoutError())` | the engine is down |
| `FakeDeciderProvider(unavailable="no_key")` | no key in the environment |
| `FakeDeciderProvider(hosted=True)` | behaves as a hosted engine (`redact` required) |

`provider.calls` records `(context, question_names)` per request: assert what
was sent, and that nothing was sent when it should not be.

---

## Writing a provider

A provider is one method. Translate the neutral `Question`s to your engine,
call it, translate back:

```python
from aixon.deciders import DeciderProvider, ProviderReply, RawAnswer, register_decider_provider

class MyEngineProvider(DeciderProvider):
    name = "myengine"
    hosted = True                                   # the context leaves the process
    capabilities = frozenset({"choice", "yes_no"})  # no "scale", no "batch"
    max_options = 20
    default_model = "v1"

    def unavailable(self) -> str | None:
        return None if os.getenv("MYENGINE_KEY") else "no_key"

    def decide(self, context, questions, *, model, timeout) -> ProviderReply:
        answers = {}
        for name, question in questions.items():
            label, chances = my_engine(context, question.instruction, question.options, timeout)
            answers[name] = RawAnswer(choice=label, chances=chances)
        return ProviderReply(answers=answers, model=model)

register_decider_provider(MyEngineProvider())
```

- `decide` may raise; the `Decider` turns any failure into "no decision".
- Without `"batch"` in `capabilities`, you receive one question per call.
- A kind missing from `capabilities`, or more options than `max_options`, is
  refused when a decider on your provider is defined.
- Leave `RawAnswer.certainty` as `None` unless your engine reports a calibrated
  confidence; the chance of the choice is used instead.
- Import your SDK inside `decide`, never at module level.

Engines that speak the "System One" format (`choice` / `noul` / `score`
questions) can reuse `aixon.deciders.systemone.to_wire` and `from_wire`, as the
Jev and Laya providers do.

---

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `AIXON_DECIDER_PROVIDER` | deciders with no `provider` | provider name |
| `TYPESAFE_API_KEY` | `jev` | API key; absent = provider unavailable, deciders return `None` |
| `JEV_MODEL` | `jev` | model when the decider declares none (default `jev-latest`) |
| `JEV_BASE_URL` | `jev` | endpoint base (default `https://api.typesafe.ai/v1`) |
| `LAYA_MODEL` | `laya` | Hugging Face id or checkpoint path when the decider declares none |
| `LAYA_DEVICE` | `laya` | `cpu` (default), `cuda`, `mps` |
| `LAYA_MAX_LEN` | `laya` | context truncation in tokens (default `1024`) |
| *your* `switch` name | one decider | `0` / `off` / `false` / `no` turns it off |

---

## Reference

```python
from aixon import Check, Decider, DeciderProvider, Decision, Doubt, decide_many
from aixon import get_decider_provider, register_decider_provider

from aixon.decider import (
    adecide_many, apply_doubts, aapply_doubts, resolve_doubts, aresolve_doubts,
    failed_checks, afailed_checks, redo_messages, DoubtOutcome, CheckFailure,
)
from aixon.deciders import (
    FakeDeciderProvider, JevProvider, LayaProvider, LLMDeciderProvider,
    ProviderReply, Question, RawAnswer,
)
from aixon.deciders.evaluation import Example, Report, evaluate, load_examples, render
```

`ToolAgent` attributes: `doubts: list[Doubt]`, `checks: list[Check]`,
`check_redo_label: str`.
