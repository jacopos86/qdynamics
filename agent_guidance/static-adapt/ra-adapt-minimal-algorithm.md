# RA-ADAPT — the algorithm, minimally

Author's design target, 2026-08-24. Paper I defines the method in one page; the
implementation scatters it across 93k lines. This file is what the code should
look like. It is deliberately short: if it grows past a few screens, the design
has failed.

**Three files. Nothing else is the algorithm.**

| file | holds | must not hold |
|---|---|---|
| `algorithm.py` | the loop, the phases, admission, acceptance | cost models, pool construction, telemetry |
| `generators.py` | generator support, records, scoring, cost terms | control flow, extensions |
| `extensions.py` | batch, prune, beam | anything the default path executes |

Metadata, receipts, and tests live outside all three.

---

## 1. The loop

The whole method. Everything else is a detail of one of these calls.

```python
def run(problem, request) -> Result:
    state = initial_state(problem)
    while not request.stop.done(state):
        records   = generators(problem, state, request.representation)
        scored    = score(records, state, request.cost, request.phases)
        shortlist = take(scored, request.shortlist)
        decision  = admit(shortlist, request.admission)
        state     = accept(state, decision, request.refit)
        state     = maintain(state, request.extensions)
    return Result(state)
```

`maintain` is a no-op unless an extension is enabled. There is no branch in the
loop for batching, pruning, or beam.

## 2. Phases 0–III are one function

They are not four code paths. They are the same evaluation at increasing
response order, each applied to what the previous one left.

```python
def score(records, state, cost, phases) -> list[Scored]:
    for phase in phases:
        records = phase.restrict(records, state)
        records = [evaluate(r, state, cost, phase.order) for r in records]
    return records
```

A phase is a pair: which records it keeps, and how much response information it
uses.

```python
Phase = namedtuple("Phase", "order restrict")
#  order 0 : raw gradient pilot
#  order 1 : g          (trust-region gain)
#  order 2 : g, H       (geometry)
#  order 3 : g, H, G    (Fubini-Study support, retained)
```

**Only the records fed in change between representations.** Macro-to-singleton
(the Bundle-5 route) mutates `generators(...)`, not `score(...)`.

## 3. One cost term, two implementations

Both are available at phases 1–3. They share one normalization.

```python
class CostTerm(Protocol):
    def __call__(self, record, state) -> Cost: ...

qiskit_cost: CostTerm   # compiled: backend transpile, integer N_2q / D_2q / D_c
proxy_cost:  CostTerm   # proxy: logical ladder span

def normalized(cost: Cost, stats: Stats) -> float:
    """Shared statistical normalization. Identical for both sources."""
```

Selection is a runtime argument — `request.cost` — not a profile, not a family,
not an ablation identity. Both may be evaluated in one run; the receipt records
which one ranked.

Today these exist as the strings `backend_transpile_v1` and
`proxy_logical_ladder_span_v1` with no shared interface and no shared
normalization. That is the gap.

## 4. Shortlisting is a function of the score

Fixed and adaptive are not modes. They are two rules over the same scored list.

```python
def take(scored, rule) -> list[Scored]:
    return rule(scored)      # e.g. top_k(12)  |  above_relative(0.1)
```

No enum, no mode normalization, no per-mode validation.

## 5. Extensions come after, and are optional

Batch, prune, and beam are defined once, after the algorithm, and compose over
the same records and state. They are absent from the default path.

```python
def maintain(state, extensions) -> State:
    for ext in extensions:      # empty by default
        state = ext(state)
    return state
```

- **batch** — changes `admit` from singleton to a constructed set.
- **prune** — proposes a deletion; deletion is accepted only after the
  remove-and-refit energy check.
- **beam** — forks accepted continuations and terminates the loser locally.

## 6. Metadata attaches to generators, not to everything

A record carries what it is and where it came from. A scored record carries what
it got and whether it was admitted. That is the whole provenance model.

```python
@dataclass(frozen=True)
class Record:
    generator: Generator
    position:  int
    source:    StateId        # which accepted state it was formed from

@dataclass(frozen=True)
class Scored:
    record:   Record
    value:    float
    cost:     Cost
    admitted: bool
```

Everything a receipt needs is derivable from a list of `Scored`. Receipts are
built outside the algorithm, from that list.

---

## What this removes

| today | after |
|---|---|
| `adapt_pipeline.py` — 72,528 lines, one 41,210-line function with 348 parameters | `algorithm.py` — the loop above |
| `hh_continuation_scoring.py` — 19,420 lines, 203 functions, 45 phase-specific | `score()` plus the phase table |
| `phase_shortlists.py` — 1,928 lines | `take(scored, rule)` |
| 31 settings dicts across 5 disconnected roots | `request` — one object, base plus diff |
| 1,878 `raise` statements, one per 46 lines | the guards that survive removal of duplicate state |
| route family / route profile / compatibility route / controlled ablation | one `request`, recorded in the receipt |

## Rules

1. **No arbitrary constraints.** A check earns its place only if it catches
   something that cannot be made structurally impossible.
2. **No mode enums where a function will do.** Shortlist rules, cost terms, and
   extensions are values, not strings to be normalized and validated.
3. **No setting that cannot take effect.** Every key in `request` reaches the
   algorithm or does not exist. Today at least one canonical profile key
   (`phase1_prune_small_theta_abs`) is read by nothing.
4. **Metadata is an output, not a parameter.** Nothing in the loop takes a
   telemetry argument.
5. **One representation of each fact.** Where two exist, a guard is needed to
   keep them agreeing; that guard is the signal to collapse them.
