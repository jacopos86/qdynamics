# Paper-I refactor — shared worklog

**Purpose:** one coordination surface for two high-level agents (Claude and
Codex) working the `adapt_pipeline.py` decomposition in parallel. Claude holds
architecture and diagnosis; Codex executes with repo access. Both write here.

Conventions come from `agent_guidance/shared/agent-handoff-contract.md`. This
file does not restate it.

---

## Anchors

| field | value |
|---|---|
| Checkout | `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3` (NOT the `~/Documents/Holstein_implementation` iCloud mirror) |
| Branch / commit | `paper-ii-exchange-selector` @ `00a5f098` |
| Governing contract | `/Users/jakestrobel/local_repos/ADAPT---Paper-I/PAPER_I_REFACTOR_BEHAVIORAL_CONTRACT.md` — separate repo, Overleaf-synced, never merge into this checkout |
| Domain glossary | `agent_guidance/static-adapt/CONTEXT.md` |
| Architecture decision | `docs/adr/0001-sr-snake-deep-module-seam.md` |
| Golden data | `agent_guidance/static-adapt/golden/` on branch `golden-rescue-20260824` — see "Evidence state" |
| RAM ceiling | 10 GB aggregate, all agents. `agent_guidance/shared/memory-budget.md`. Wrap heavy work in `pipelines/shell/ram_guard.py --limit-mb 8000` |

### Test baseline — READ THIS BEFORE QUOTING A NUMBER

```bash
# Collection only. This is what everyone has been quoting.
python3 -m pytest test --collect-only -q
#   -> 5624 collected, 55 errors, "Interrupted: 55 errors during collection"

# The suite does NOT run without this flag. Plain `pytest test` executes ZERO tests.
python3 pipelines/shell/ram_guard.py --limit-mb 8000 -- \
  python3 -m pytest test -q --tb=no -rf --continue-on-collection-errors
```

The 55 collection errors abort the session before any test executes. That is
why no failure count existed. Measured with `--continue-on-collection-errors`:
**~16% of executed tests fail** (see "Live findings" for the settled number).

---

# THE DESIGN TARGET

Author's specification, 2026-08-24. This is what the code should look like.
Everything below this heading, down to "Coordination protocol", is the target;
the rest of the file is current state and evidence.

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


## 6b. Geometry is one Gram; phases are restrictions of it

Author's formulation, 2026-08-24. There is **one** geometry Gram over
generators and ansatz. Phases II, I and 0 are successive restrictions of it,
coupled to the phase scoring. The complication is that each restriction must
carry **recorded amounts for estimator-count reporting**.

### The nesting

Let `G` be the Gram at the current accepted state, `c` a candidate index, `W`
the Phase-II geometry window, `R` the Phase-III retained/refit support, with
`{c} subset W union {c} subset R union {c}`. Each phase reads a principal
submatrix:

| phase | restriction | today's scattered form |
|---|---|---|
| 0 | no Gram; gradient only | `phase0_cost_lambdas`, raw pilot |
| I | `G[c, c]` — one diagonal entry | `F_metric`, `metric_proxy`, `cheap_metric_proxy` |
| II | `G[W u {c}, W u {c}]` | `Q_window`, `q_window`, `phase2_geometry_window_indices`, `phase2_raw_overlap_max`, `phase2_span_projection_z` |
| III | `G[R u {c}, R u {c}]` | `phase3_geometry_refit_window_indices`, `phase3_geometry_active_post_indices`, `phase3_geometry_window_size` |

Confirmed in source: the window fields are integer index lists
(`phase2_geometry_window_indices=[int(i) for i in ...]`,
`phase3_geometry_refit_window_indices=[int(i) for i in ...]`), and `F_metric` is
a scalar taken from `F_raw` or `metric_proxy`. They are already index sets and a
diagonal element — that is, already restrictions, just materialized separately
and passed as unrelated arguments.

### Estimator accounting falls out, it does not need adding

`estimator_call_ledger.py` already keys a primitive by
`projective_state_fingerprint` plus `canonical_symmetric_pair(left, right)`,
content-addressed through `_digest`. **That key is a Gram entry**: an unordered
operand pair at a state.

So:

- measuring `G[i, j]` is exactly one estimator primitive, keyed `(state, {i,j})`;
- a phase's estimator cost is the number of pairs in its restriction **not
  already in the ledger**;
- because the restrictions nest, Phase II reuses Phase I's diagonal and Phase
  III reuses Phase II's block automatically — the ledger's dedup *is* the reuse
  policy;
- `CONTEXT.md`'s **Estimator primitive** ("counted once regardless of repeated
  consumption") is satisfied structurally rather than by bookkeeping.

The restriction therefore carries its own recorded amount: `|pairs(restriction)|`
requested, and `|new pairs|` charged. Both are derivable from the index set and
the ledger, and neither needs a field on the candidate.

### What this deletes

Of the 26 geometry summaries on `CandidateFeatures` (249 fields total,
`hh_continuation_types.py:79-334`), the map above replaces the data-carrying
ones with submatrix selections. Four are not data at all but policy smuggled
onto every candidate — `phase2_curvature_policy`,
`phase2_geometry_window_policy`, `phase3_geometry_window_policy`,
`selector_geometry_mode` — and belong in `request`, resolved once.

It also removes the dead threading: `phase2_raw_geometry_score` currently takes
`q_window, Q_window` and runs `del q_window, Q_window` on the first line, keeping
the Gram in its signature purely because the same geometry is reused elsewhere.
With one Gram there is nothing to thread.

## 6c. `evaluate()` and the Gram accessor — the implementable signature

> **SUPERSEDED by 6g.** The `g()/H()/G()` three-accessor shape below is the
> generic framing, not Paper I's. The support table and the estimator-charge
> reasoning still hold; the return shape does not. Implement 6g.


Written against the real call sites, 2026-08-24. This is what Codex implements.

```python
# ---------- one Gram, read by restriction ----------

@dataclass(frozen=True)
class EstimatorCharge:
    requested: int                 # |pairs(restriction)|
    charged:   int                 # pairs not already in the ledger
    keys:      tuple[str, ...]     # EstimatorCallKey digests, for the receipt

class Geometry(Protocol):
    """One Fubini-Study Gram at one accepted state. Phases read submatrices."""
    def restrict(self, support: Sequence[int]) -> np.ndarray: ...
    def charge(self, support: Sequence[int]) -> EstimatorCharge: ...

# ---------- one cost term, two implementations ----------

@dataclass(frozen=True)
class Cost:
    n_2q: int
    d_2q: int
    d_c:  int
    source: str          # "qiskit" | "proxy" -- recorded, NEVER substituted

class CostTerm(Protocol):
    def __call__(self, record: Record, state: State) -> Cost: ...

def normalized(cost: Cost, stats: Stats) -> float:
    """Shared normalization, identical for both sources. Returns the full 1+K."""

# ---------- the one scorer ----------

def evaluate(record, state, geometry, cost_term, order, stats) -> Scored:
    support = _support(order, state, record)
    G       = geometry.restrict(support)
    charge  = geometry.charge(support)
    dE      = _descent(order, record, state, G)
    K       = normalized(cost_term(record, state), stats)
    return Scored(record=record, value=dE / K, charge=charge, admitted=False)
```

`_support` is the whole phase table:

| order | support | meaning |
|---|---|---|
| 0 | `()` | no Gram; gradient pilot only |
| 1 | `(c,)` | the diagonal entry `G[c,c]` |
| 2 | `W + (c,)` | geometry window block |
| 3 | `R + (c,)` | retained/refit support block |

`_descent` is the existing phase formula at that order, and nothing more:

| order | today |
|---|---|
| 0 | `phase0_raw_gradient_pilot_components:3004` |
| 1 | `phase1_trust_region_gain:2614` — `trust_region_drop(g, lambda_F*F, F, rho)`, `F = G[c,c]` |
| 2 | `phase2_raw_geometry_score:3168` — adds measured curvature from the window block |
| 3 | `phase3_canonical_score_components:3831` — `DeltaE_TR / (1 + K3)` |

### What already exists and must be reused, not rewritten

| target | existing |
|---|---|
| `normalized()` | `_hardware_cost_denominator_payload:550` already returns the full `1 + K3`; `_hardware_cost_normalization_mode:862` selects it, default `HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1` |
| `Geometry.charge()` | `estimator_call_ledger.py` — `projective_state_fingerprint` + `canonical_symmetric_pair`, digested. One Gram entry = one primitive |
| `Cost.source` | today's `backend_transpile_v1` / `proxy_logical_ladder_span_v1` |

### Invariants

1. `evaluate` takes no telemetry argument and returns no receipt. Receipts are
   built outside, from a list of `Scored`.
2. `cost_term` is chosen by the caller. **A failing `qiskit_cost` raises.** It
   never returns a proxy number (rule 7).
3. `geometry.charge()` is the only place estimator work is counted, and because
   supports nest, reuse across phases is the ledger dedup rather than a policy.
4. `_descent` may read only `G` and the record. If it needs a scalar summary not
   derivable from `G`, that is a defect in the restriction, not a reason for a
   new `CandidateFeatures` field.

## 6d. CORRECTION to 6c — the accessor is over the response, not the Gram alone

> **SUPERSEDED by 6g.** The `g()/H()/G()` three-accessor shape below is the
> generic framing, not Paper I's. The support table and the estimator-charge
> reasoning still hold; the return shape does not. Implement 6g.


Committed 6c said `_descent` "may read only `G` and the record". **That is
wrong**, verified 2026-08-24:

- `_selector_gradient_lcb` returns `g_hw_lcb` from
  `_selector_gradient_resolution(feat, cfg)` — the **gradient**, not the Gram.
- `h_raw` is measured **curvature**, passed separately into
  `phase2_raw_geometry_score`.

The method acquires three response objects, exactly as the author states it:
progressively richer **g, H and G**. The accessor must be over all three.

```python
class Response(Protocol):
    """One response set at one accepted state, read by restriction."""
    def g(self, support: Sequence[int]) -> np.ndarray: ...   # coordinate gradient
    def H(self, support: Sequence[int]) -> np.ndarray: ...   # ordered-coordinate energy Hessian
    def G(self, support: Sequence[int]) -> np.ndarray: ...   # Fubini-Study Gram
    def charge(self, order: int, support: Sequence[int]) -> EstimatorCharge: ...

def evaluate(record, state, response, cost_term, order, stats) -> Scored:
    support = _support(order, state, record)
    dE      = _descent(order, record, response, support)
    K       = normalized(cost_term(record, state), stats)
    return Scored(record, value=dE / K,
                  charge=response.charge(order, support), admitted=False)
```

Phase table, corrected:

| order | reads | support |
|---|---|---|
| 0 | `g` | candidate only |
| 1 | `g`, `G[c,c]` | `(c,)` |
| 2 | `g`, `H(W)`, `G[W u c]` | `W + (c,)` |
| 3 | `g`, `H(R)`, `G[R u c]` | `R + (c,)` |

Phase II's own formula is `DeltaE_TR_raw / (1 + K2)` and Phase III's is
`DeltaE_TR / (1 + K3)` (`PHASE2_CANONICAL_RAW_SCORE_FORMULA:275`,
`PHASE3_CANONICAL_SCORE_FORMULA:281`) — same shape, different order. The
collapse still holds; only the accessor was under-specified.

### The estimator charge is uniform, and reuse survives

`EstimatorCallKey` already carries either **one unary operand** or **one
symmetric pair**, never both, plus a `primitive_kind`. So:

| response | key form | `primitive_kind` in use |
|---|---|---|
| `g` | unary `operand_identity` (schema v2) | `coordinate_gradient` |
| `G` | `symmetric_pair` | `metric_element` |
| `H` | pair | **no dedicated kind exists** |

**`precision_contract` is a run-level constant**
(`complex128_float64_deterministic_v1`, or `analytic_exact_float64_v1` on the
exact backend), not per-phase. Therefore a `G[i,j]` measured at order 1 has an
identical key at order 3, dedup applies, and **nesting reuse holds in practice**,
not just in principle.

**Open, for the author:** curvature has no `primitive_kind` of its own. The kinds
in use are `hamiltonian_expectation`, `coordinate_gradient`, `metric_element`,
`state_overlap`. Either `H` is correctly composed from `hamiltonian_expectation`
primitives and needs no kind, or curvature work is not being separately charged.
This bears directly on the reported estimator totals.

## 6e. Curvature is a restriction of the Hessian — one support, three responses

> **SUPERSEDED by 6g.** The `g()/H()/G()` three-accessor shape below is the
> generic framing, not Paper I's. The support table and the estimator-charge
> reasoning still hold; the return shape does not. Implement 6g.


Author's correction, 2026-08-24, verified in source. 6d still treated `h_raw` as
a separate scalar parameter. It is not: **curvature is `H[c,c]`**, the exact
analogue of `F = G[c,c]`.

Evidence:

```
hh_continuation_scoring.py:5546   h_raw = _energy_hessian_entry(...)
hh_continuation_scoring.py:6507   "H_BB": float(phase2_h_raw)
```

`_energy_hessian_entry` is an entry of the ordered-coordinate energy Hessian, and
the code already labels it `H_BB` — the candidate's own diagonal block. It is
charged as `hessian_element` with observable identity `energy_hessian_v2`
(`estimator_call_ledger.py:67`, `adapt_pipeline.py:20393` and others).

### The design collapses further: one support, applied to three responses

| response | order 0 | order 1 | order 2 | order 3 |
|---|---|---|---|---|
| `g` | `g[c]` | `g[c]` | `g[W u c]` | `g[R u c]` |
| `H` | — | — | `H[c,c]` = `h_raw` = `H_BB` | `H[R u c]` |
| `G` | — | `G[c,c]` = `F` | `G[W u c]` | `G[R u c]` |

There is **one** `_support(order)`. Each response object is restricted to it.
Nothing is passed as a pre-extracted scalar.

```python
def evaluate(record, state, response, cost_term, order, stats) -> Scored:
    support = _support(order, state, record)
    dE      = _descent(order, record,
                       response.g(support),
                       response.H(support),
                       response.G(support))
    K       = normalized(cost_term(record, state), stats)
    return Scored(record, value=dE / K,
                  charge=response.charge(order, support), admitted=False)
```

### What this deletes

`F_metric`, `metric_proxy`, `cheap_metric_proxy`, `h_raw`, `phase2_h_raw`,
`H_BB`, `h_hat` are all diagonal entries of `G` or `H` that today travel as
independent scalars on `CandidateFeatures` or as keyword arguments. Under one
support they are `response.G(support)[c,c]` and `response.H(support)[c,c]`.

This is the same defect as F3, F4 and the test pollution, in a fourth place: one
fact — an entry of a response object — materialized in several places, each copy
then needing a guard that it still agrees with the others.

## 6f. HOOK INVENTORY — where each target piece already lives

Division of labour (author, 2026-08-24): Claude plans and locates the hooks;
Codex implements. This section exists so Codex never has to search. Every line
number measured on `77aeeba8` or later — re-verify before editing, the tree moves.

### HOOK A — `Response` already exists as a 303-line class

**`_DefaultNoPruneEstimatorService`, `adapt_pipeline.py:56603-56905`** — 303
lines, 13 methods. This is the response accessor from 6d/6e, already written:

| line | method | target role |
|---|---|---|
| 56676 | `_record_estimator_primitive` (18) | the ledger charge |
| 56695 | `_active_physical_tangent` (2) | ansatz index -> coordinate identity |
| 56698 | `_candidate_physical_tangent` (3) | candidate -> coordinate identity |
| 56720 | `_record_candidate_self_metric_primitive` (4) | `G[c,c]` — `symmetric_pair=(coord, coord)` |
| 56725 | `_record_scaffold_geometry_primitives` (18) | **`G[S]` and `H[S]` over a support** |
| 56744 | `_record_candidate_geometry_primitives` (36) | candidate-active cross block |

`_record_scaffold_geometry_primitives` is `Response.charge(support)` in embryo:
it takes `refit_window_indices` (the support), iterates the upper triangle of
coordinate pairs, and records `('metric_element','fubini_study_metric_v2')` and
`('hessian_element','energy_hessian_v2')` **for the same pair**, gated by
`record_metric` / `record_hessian`. That is 6e's claim — one support, three
responses — already implemented.

`_DefaultNoPruneNumericalSession:60123-64580` (4,458 lines, 37 methods) wraps
these with 8-line delegators at 60530-60598. Extract the service, not the session.

**Codex: this is extraction, not invention.**

### HOOK B — the support is already a named parameter

`refit_window_indices` is the support throughout. Construction sites:
`adapt_pipeline.py:8048, 8058, 8230, 20776, 23596, 23609, 30518`, and
`_prune_refit_window_indices_live:24393`. Phase-II window:
`phase2_geometry_window_indices`, built at `30519, 31327, 40567, 42738`.

`_support(order)` replaces these; the values are already integer index lists.

### HOOK C — the gradient

`primitive_kind="coordinate_gradient"` issued at `adapt_pipeline.py:20141, 20326`.
Unary key via `operand_identity` (schema v2). 14,325 of them in a real run — 10x
all pair primitives combined.

### HOOK D — the four phase scorers

`hh_continuation_scoring.py`: `phase0_raw_gradient_pilot_components:3004`,
`phase1_trust_region_gain:2614`, `phase2_raw_geometry_score:3168`,
`phase3_canonical_score_components:3831`. These become `_descent(order, ...)`.

Formulas already agree: `PHASE2_CANONICAL_RAW_SCORE_FORMULA:275` is
`DeltaE_TR_raw / (1 + K2)`, `PHASE3_CANONICAL_SCORE_FORMULA:281` is
`DeltaE_TR / (1 + K3)`.

### HOOK E — `CandidateFeatures` construction

13 sites. Principal: `adapt_pipeline.py:3625, 22066, 31376, 40620` and
`selector_candidate_metadata.py:36` (which rebuilds by `__dict__` merge).
Definition `hh_continuation_types.py:79-334`, 249 fields, 26 geometry summaries.

### HOOK F — cost

`hh_backend_compile_oracle.py`: `BackendCompileConfig:83` (note `mode="proxy"`
default and `allow_preferred_fallback=True`, both against rule 7),
`backend_compile_scope_uses_qiskit_for_stage:60`, bare
`except Exception: pass` at `:350`.

Shared normalization already exists:
`hh_continuation_scoring.py:_hardware_cost_denominator_payload:550` returns the
full `1 + K`, selected by `_hardware_cost_normalization_mode:862`, default
`HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1`.

### HOOK G — the legacy seam

| site | role |
|---|---|
| `sr_snake/_context.py:885` | `adapt_pipeline._run_hardcoded_adapt_vqe(**executor_kwargs)` — the real invocation |
| `sr_snake/_context.py:249` | `legacy_executor_kwargs()` builds the payload |
| `adapt_pipeline.py:72076-72090` | the 348-name reflective filter |
| `cli_config.py:3736` | the CLI adapter |
| `exact_bench/hh_static_ground_state_benchmark.py:975` | second product caller |

## 6g. The response interface, in Paper I's own form — IMPLEMENT THIS

Supersedes the return shape of 6c/6d/6e. Settled by Q13: the partitioned block
form is the manuscript's structure, not an implementation artifact.

Paper I, Eq. `hessian_block_def`:

```
H = [ H_aa        H_a-theta     ]      three distinct blocks: candidate curvature
    [ H_theta-a   H_theta-theta ]      (aa), candidate--ansatz coupling (a-theta),
                                       active-ansatz response (theta-theta)
```

with **G** the Fubini--Study metric defining the trust region `z^T G z <= ...`,
and Phase III isolating the candidate contribution by subtracting the active-only
response (see Q16 — selectable).

### Interface

```python
@dataclass(frozen=True)
class ResponseBlocks:
    """One response set restricted to one support, in the paper's partition."""
    G_AB: np.ndarray     # active-candidate cross block
    G_BB: np.ndarray     # candidate-candidate block
    b_B:  np.ndarray     # candidate-direction residual

class Response(Protocol):
    def restrict(self, support: Sequence[int]) -> ResponseBlocks: ...
    def charge(self, order: int, support: Sequence[int]) -> EstimatorCharge: ...

def evaluate(record, state, response, cost_term, order, stats) -> Scored:
    support = _support(order, state, record)
    blocks  = response.restrict(support)
    dE      = _descent(order, record, blocks)
    K       = normalized(cost_term(record, state), stats)
    return Scored(record, value=dE / K,
                  charge=response.charge(order, support), admitted=False)
```

`_support(order)` is unchanged from 6e: `()`, `(c,)`, `W + (c,)`, `R + (c,)`.

### Verified vs still to locate

| piece | status |
|---|---|
| `(G_AB, G_BB, b_B)` at a support | **verified** — `selector_query_closure.py:795`, `QueryClosedPopulationWorkspace.subset_geometry`, returns `G_AC[:, idx]`, `G_CC[ix_(idx,idx)]`, `b_C[idx]` |
| the charge half | **verified** — `_DefaultNoPruneEstimatorService`, `adapt_pipeline.py:56603-56905` |
| the Hessian partition `H_aa / H_a-theta / H_theta-theta` | **NOT located.** `h_raw` comes from `_energy_hessian_entry` (`hh_continuation_scoring.py:5546`) and is labelled `H_BB` at `:6507`, so the `aa` block exists as a scalar path. The full partition returned as blocks has not been found. |

**Codex: do not invent the Hessian partition.** If `_descent` at order 2 or 3
needs a block that no existing symbol returns, stop and report the gap rather
than assembling one — an assembled Hessian block would change the numbers.

## 6h. The value half is located — 6g's gap is closed

> **PARTLY WRONG — see 6j.** `FormalAdmissionCurvatureReceipt` has **zero
> production callers**; it is reached only from
> `test/test_static_adapt_selector_query_closure.py`. The live partition exists,
> but elsewhere and under different names. Implement 6j.


Found by Codex, 2026-08-24. `build_formal_admission_curvature_receipt`
(`selector_query_closure.py:2458`) returns `FormalAdmissionCurvatureReceipt`
(`:2207`), which carries **actual arrays**:

```
G_AA, G_AB, G_BB          Fubini-Study metric partition
H_AA, H_AB, H_BB          energy Hessian partition, Paper I Eq. hessian_block_def
active / candidate gradients
```

That is the complete Paper-I partition, Hessian blocks included. 6g listed the
Hessian partition as "NOT located"; it is located. **Codex must no longer treat
that as a stop condition** — nothing needs assembling.

Caveat recorded by Codex: it is shaped as a **Phase-III handoff**, not as a
`restrict(support)` interface. `subset_geometry:795` is the restriction-shaped
one but returns only `(G_AB, G_BB, b_B)`. So the two halves exist in different
shapes and the work is reconciling them, not building either.

## 6i. RULE: every increment states a net-line target

Prompted by the author's observation on the `ResponseAccounting` extraction:
*"all it did was change and actually added more code than what we had before."*

Measured, and correct: `adapt_pipeline.py` -465, `response_accounting.py` +505,
**net +40**. The extraction was verified safe (ledger SHA256 identical, 18
occurrences, 13 unique primitives, error diff empty) but condensed nothing.

**Moving code between files cannot reduce total lines.** It reduces one file. The
goal is condensation, so:

> Every increment from here declares a **net lines removed** target in its
> claim row, and reports the measured net in its execution-log entry. An
> increment with a target of zero must say why it is worth doing anyway —
> "it unblocks X" is a valid reason, "it is cleaner" is not.

Extraction increments are permitted only as a *prerequisite* to a named deletion
increment that follows them.

### Deletion targets now available, sized

| target | size | status |
|---|---|---|
| Phase-0 cost path | **124** references in `adapt_pipeline.py` | approved (Q14, Q17), verified numerical no-op |
| 348-name reflective filter and its parameter surface | **~326** parameter lines + 15 filter lines | blocked on the typed payload replacing it |
| 26 geometry scalar summaries on `CandidateFeatures` | 26 fields x construction + read sites, in a 249-field dataclass | unblocked by 6h — the arrays now have a located source |
| `del q_window, Q_window` dead threading | signature + call sites | unblocked by 6h |

## 6j. The live response partition — reconciled, correcting 6h and 6b

Ratified against source after 6h was recorded on Codex's find without checking
callers.

### Correction 1 — the full-partition receipt is test-only

`build_formal_admission_curvature_receipt` / `FormalAdmissionCurvatureReceipt`
have **no production callers**. Every reference outside their own module is in
`test/test_static_adapt_selector_query_closure.py`. Same for
`FormalGrowthGeometryReceipt`. They are a validated freeze-frame for an FM
handoff, not the live value half.

### Correction 2 — `Q_window` is the Hessian, not the Gram

6b called `phase2_raw_geometry_score`'s discarded `q_window, Q_window` "the Gram
blocks". **Wrong.** The live naming is:

| symbol | object |
|---|---|
| `G_AA`, `G_AC`, `G_CC` | Fubini--Study **metric** |
| `Q_AA`, `Q_AC`, `Q_CC` | ordinary coordinate **Hessian** (Paper I's `H`) |
| `b_A`, `b_C` | descent **gradient** |

So `del q_window, Q_window` discards **Hessian** blocks. Stronger than 6b said,
not weaker.

### Where each live piece is

| piece | class | fields | site |
|---|---|---|---|
| A side (active), per accepted state | `SelectorGeometryAnchor` | `active_coordinate_indices`, `G_AA`, `b_A`, state/hamiltonian fingerprints | `selector_query_closure.py` |
| C side (candidate population) | `QueryClosedPopulationWorkspace` | `G_AC`, `G_CC`, `b_C` | same |
| restriction to a candidate subset | `subset_geometry(indices)` | `(G_AB, G_BB, b_B)` | `:795` |
| Hessian partition | `Phase2OrdinaryHessianBlocks` | `Q_AA`, `Q_AC`, `Q_CC`, `source_query_receipts` | `:1155` |

The anchor already carries `state_fingerprint` and `hamiltonian_fingerprint`, so
charge and value share one identity.

### The reconciled interface

`_support(order)` in 6e conflated two index sets. Paper I partitions into active
`theta` and candidate `alpha`, so there are two, and **the restriction is over
candidates** while the active side is the anchor:

```python
class Response(Protocol):
    """A = active coordinates (anchor, fixed per accepted state).
       C = candidate population; B = a restriction of C."""
    def restrict(self, candidates: Sequence[int]) -> ResponseBlocks: ...
    def charge(self, candidates: Sequence[int]) -> EstimatorCharge: ...

@dataclass(frozen=True)
class ResponseBlocks:
    G_AB: np.ndarray; G_BB: np.ndarray      # metric
    Q_AB: np.ndarray; Q_BB: np.ndarray      # Hessian (Paper I H)
    b_B:  np.ndarray                        # gradient
```

Order selects which blocks `_descent` reads, not a different index set:

| order | reads |
|---|---|
| 0 | `b_B` |
| 1 | `b_B`, `diag(G_BB)` |
| 2 | `b_B`, `G_BB`, `Q_BB` |
| 3 | `b_B`, `G_AB`, `G_BB`, `Q_AB`, `Q_BB` |

Smaller than 6e implied.

### Remaining gap — do not assume

`Phase2OrdinaryHessianBlocks` is referenced only within
`selector_query_closure.py`; no other `pipelines/` module names it. Whether the
live Phase-II path constructs one, or computes curvature only through the
`_energy_hessian_entry` scalar route (`hh_continuation_scoring.py:5546`), is
**not established**. Codex must not assume the block form is populated.

## 6k. The 6j gap is closed — the Hessian partition is live, just unpackaged

Ratified 2026-08-24. 6j said it was "not established" whether the live Phase-II
path constructs `Phase2OrdinaryHessianBlocks`. It does not — and that does not
matter.

**The dataclass is dead.** `Phase2OrdinaryHessianBlocks(` is constructed at 6
sites: 4 in `test/test_static_adapt_selector_query_closure.py`, and 2 in
`chtc/paper_iv_h2o_sr_source_locked_resume_d12_20260716/runtime_source/pipelines/static_adapt/adapt_pipeline.py`
— a **frozen source overlay**, an archived copy of an older `adapt_pipeline.py`.
The live tree once built it and no longer does. Deletion candidate.

**The partition itself is live, as bare arrays:**

| block | live site | shape / note |
|---|---|---|
| `H_AB` | `adapt_pipeline.py:1417` | `_matrix("H_AB_raw", (window_count, 1))` — candidate vs active window |
| `H_AB` | `hh_continuation_scoring.py:5959` | `np.zeros(active_count)` |
| `H_AB` | `hh_continuation_scoring.py:6165` | `np.asarray(source.get("H_AB", ()))` |
| entries | `_energy_hessian_entry:4481` | fills `hess[row, col]` at `:5184`; mixed block at `:5584`; active-candidate at `:5964` |

So all three response objects have live sources:

| object | live source |
|---|---|
| metric `G` | `SelectorGeometryAnchor.G_AA`, `QueryClosedPopulationWorkspace.G_AC/G_CC`, `subset_geometry:795` |
| Hessian `Q`/`H` | `H_AB_raw` and `_energy_hessian_entry`, unpackaged |
| gradient `b` | `b_A`, `b_C`, `b_B` |

**Consequence: the 26-scalar deletion is unblocked.** Every scalar summary on
`CandidateFeatures` now has a located array source, so each can be replaced by an
indexing expression rather than a stored field. The work is packaging the
existing arrays behind `ResponseBlocks`, not computing anything new — which keeps
it inside the "no numerical change" envelope.

## 6l. `metric_proxy` fails the Paper-I test, and is a rule-7 fallback

Author's test applied 2026-08-24: *if it is not defined in Paper I, it is
probably archaic.*

**Result: there is no metric proxy in Paper I.** All 8 occurrences of "proxy" in
`Paper_I_author_revision.tex` are **cost** proxies — `eq:pauli_2count_proxy`,
`eq:pauli_2depth_proxy`, `eq:pauli_1q_proxy`, `eq:shot_cost_proxy`. The metric is
the Fubini--Study pullback metric with no proxy variant.

**It is also a silent substitution.** `phase1_trust_region_gain:2614`:

```python
F_measured = max(0.0, float(feat.metric_proxy), float(feat.F_metric))
```

Whichever is larger wins. When `metric_proxy > F_metric`, Phase I's
trust-region gain is computed from a quantity the manuscript does not define,
with no signal that it happened. That is rule 7 (no fallbacks) in the scoring
path, not just in the cost oracle.

The default energy model is `PHASE1_ENERGY_MODEL_LEGACY_LAMBDA_F_QUADRATIC_V1` —
"legacy" again.

Surface: 6 sites set `metric_proxy` (`adapt_pipeline.py:30362, 31170, 40410,
42581, 60612, 61212`); 35 references to `cheap_metric_proxy` /
`phase1_lambda_f_proxy_applied` / `phase2_lambda_f_proxy_applied`.

**Deletion is numerically safe only if the proxy never won.** `max()` means
removing it changes results in exactly those rounds where
`metric_proxy > F_metric`. Both values are carried on `CandidateFeatures`, so
this is checkable from completed-run receipts. **Check before deleting** —
otherwise this is a silent scientific change, not a cleanup.

## 6m. HOOKS for the `metric_proxy` unification (Q20)

### What it actually is

Not a proxy. `candidate_metric_proxy` is assigned straight from
`exact_first_order_geometry["fubini_study_metric"]` at `adapt_pipeline.py:40298,
40375, 42471, 64663`, and `F_metric` is populated *from it* at
`hh_continuation_scoring.py:19013` (`F_metric = max(0.0, metric_proxy)`). Two
field names, one quantity, reconciled by `max()` because nothing guarantees which
path wrote.

### The one branch that is a real substitution

`_base_metric_for_candidate`, `adapt_pipeline.py:39614`:

```python
gradient_abs = float(abs(float(gradient_signed)))
if not phase3_enabled:
    return float(gradient_abs)      # |g| standing in for F
```

`phase3_enabled = bool(continuation_mode == "phase3_v1" and staged_problem_enabled)`
(`adapt_pipeline.py:18340`).

### Hooks

| piece | site |
|---|---|
| the `max()` to collapse | `hh_continuation_scoring.py:2625-2630` in `phase1_trust_region_gain` |
| `metric_proxy` setters (6) | `adapt_pipeline.py:30362, 31170, 40410, 42581, 60612, 61212` |
| `F_metric` setters (3) | `adapt_pipeline.py:65072` (`=1.0`), `hh_continuation_scoring.py:6586, 19013` |
| the `|g|` branch | `adapt_pipeline.py:39614` |
| `cheap_metric_proxy` / `*_lambda_f_proxy_applied` | 35 references |
| field definitions | `hh_continuation_types.py`, `CandidateFeatures` |

### MANDATORY CHECK BEFORE DELETING

The `|g|` branch changes numbers **only when Phase III is off**. Canonical
`adapt_continuation_mode: "phase3_v1"` is declared at
`sr_snake_route_profile.py:467`, inside `CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS`
— **the 116-key root that the 20-profile active family does not inherit** (F3).
So source alone does not establish that Phase III was on in the Bundle runs.

Verify from a completed-run receipt that `phase3_enabled` was true, or that
`metric_proxy` never exceeded `F_metric`, before removing the branch. If it ever
fired, this is a scientific change and stops being a cleanup.

## 7. No fallbacks

**Author's rule, 2026-08-24: no fallbacks.** A fallback silently substitutes a
different computation for the one that was asked for, and the result is
indistinguishable from success. Compiled cost degrading to proxy means the run
reports a compiled-cost number it never measured.

If the requested thing cannot be done, the run stops and says so. There is no
second-choice path.

Measured today:

| module | occurrences of `fallback` |
|---|---|
| `adapt_pipeline.py` | 323 |
| `hh_continuation_scoring.py` | 113 |
| `phase_shortlists.py` | 43 |
| `hh_backend_compile_oracle.py` | 13 |
| **all of `pipelines/`** | **1,622** |

Concrete instances in the cost path:

- `hh_backend_compile_oracle.py:350` — `except Exception: pass`, a bare swallow
  around backend loading.
- `BackendCompileConfig.mode` defaults to **`"proxy"`**, so the proxy is the
  default cost and the compiled cost is opt-in — inverted, given compiled cost
  is the manuscript's headline axis.
- `BackendCompileConfig.allow_preferred_fallback` defaults to `True`.

In the target design the two cost terms are peers selected by `request.cost`.
Neither substitutes for the other, ever. `qiskit_cost` failing is a stopped run,
not a proxy number.

---

## DECISIONS — author's answers, poll this section

**Codex: re-read this section before each increment.** It is the running record
of decisions the author has made. Every row is binding unless a later dated row
supersedes it. Questions are appended by Claude; answers are the author's.

An unanswered question is **not** permission to choose. Stop and report instead.

| # | question | author's answer | date |
|---|---|---|---|
| Q1 | What makes two runs "the same run"? | **Not** exact sequence reproduction. Ideally the same results, but initial sensitivity exists, so what matters is that new results are **at least as good** as old. Bug fixes that alter trajectories are unavoidable and acceptable. | 2026-08-24 |
| Q2 | Keep the route family vs route profile distinction? | **No.** Semantic differentiation that hurts more than helps. "Realization" settings can change trajectories too, which disproves the split. Drop the taxonomy. | 2026-08-24 |
| Q3 | Candidate gain: total joint or marginal? | **Try both.** Full matrix, not one representative cell. | 2026-08-24 |
| Q4 | Was `pipelines/hardcoded/` retirement deliberate? | **Yes, definitely deliberate.** | 2026-08-24 |
| Q5 | Re-run Bundle 3 to prove Gate 1? | **No.** Reverting via git covers recovery. Withdrawn by Claude; only an archive readability check was warranted, and it passed. | 2026-08-24 |
| Q6 | What does "at least as good" measure? | **DeltaE, the qiskit costs, and the estimator count** — the three Paper-I axes. | 2026-08-24 |
| Q13 | Is `(G_AB, G_BB, b_B)` archaic? | **No — it is Paper I's own structure**, Eq. `hessian_block_def`. Test applied: if it is not defined in Paper I it is probably archaic. It is defined there. | 2026-08-24 |
| Q14 | Delete the Phase-0 cost path? | **Yes, delete.** Verified a numerical no-op first: `phase0_K0` is 1.0 in all 30 recorded instances. | 2026-08-24 |
| Q15 | Should Claude move to its own worktree? | Author: not a technical call. **Claude decided yes** — `claude/paper-i-20260824`. Measurements from the shared checkout were contaminated by another agent's uncommitted files. | 2026-08-24 |
| Q17 | Drop `phase0_K0` and the two hardware-cost fields from the checkpoint payload, or keep as constants? | **Drop them outright.** Accept that resume/replay of old checkpoints may surface a real bug; find it rather than paper over it. | 2026-08-24 |
| Q16 | Is Bundle 9's legacy total-joint path inconsistent with the published Phase III? | **No — marginal vs non-marginal is an option. Test both.** Consistent with Q3. Bundle 9 is not disqualified; it exercised one option. | 2026-08-24 |
| Q19 | Does the marginal-vs-total gain campaign run before or after the refactor? | **After the refactor.** Running it first would compare two policies across an implementation about to change underneath them. | 2026-08-24 |
| Q20 | `metric_proxy` — delete? | **Yes, delete.** It is the Fubini--Study metric under a second name, not a distinct object: unify `metric_proxy` and `F_metric` into one `F` field, and delete the `not phase3_enabled` branch that substitutes `abs(gradient)` for the metric. | 2026-08-24 |

### Standing rules from the author

| rule | source |
|---|---|
| **No fallbacks.** A failing qiskit cost stops the run; it never degrades to proxy. | 2026-08-24, section 7 |
| **A profile is a run plus a diff.** One base with complete effective settings; every other profile a named delta. | 2026-08-24, F3 |
| **Separate the mathematical algorithm from the ansatz and generators**, so the algorithm calls those objects. | 2026-08-24, Codex session |
| **No arbitrary constraints.** A check earns its place only if it catches something that cannot be made structurally impossible. | 2026-08-24 |
| **One file for coordination.** Do not create new `.md` files for this effort. | 2026-08-24 |
| **GitNexus stays scoped to Paper I.** No repo-wide re-index. | 2026-08-24, Codex session |

### OPEN — awaiting the author

| # | question | why it matters |
|---|---|---|
| Q18 | Q16 makes marginal vs non-marginal an **option**. Paper I currently states it as the definition — line 442 defines Phase III as isolating the candidate contribution "by subtracting the corresponding active-only response", with no alternative offered. **Does the manuscript text need to change to present it as a selectable policy?** | Not a re-litigation of Q16. If both are legitimate options but the paper defines only one, then evidence produced under the other cannot be described as implementing the stated Phase III. Affects Gate 5 manuscript synchronization. |

---

## Coordination protocol — two writers, one file

1. **Claim before you work.** Add a row to the Claims table below, with your
   agent name and the date. Do not start an item another agent holds.
2. **Never rewrite another agent's section.** Append a new dated entry under
   "Live findings" instead. Corrections to someone else's finding go in a new
   entry that names the entry it corrects.
3. **Sections have single owners.** "Verified findings" is Claude-owned.
   "Execution log" is Codex-owned. "Open decisions" is the author's.
   Anyone may append to "Live findings" and "Claims".
4. **Stage explicit paths only.** Never `git add -A` here — a broad add sweeps
   the other agent's uncommitted work into your commit (incident `ba7f2ac9`).
5. **Commit this file on its own** when you update it, so a conflict is a
   one-file conflict.
6. **An index proposes; source ratifies.** Every claim below carries the
   command that produced it. Re-run it rather than trusting the number.

### Claims

| item | agent | date | state |
|---|---|---|---|
| Increment 0 — golden data rescue | Claude | 2026-08-24 | done, `5e6fcb17` on `golden-rescue-20260824` |
| Full-suite failure census | Claude | 2026-08-24 | in progress |
| Worklog consistency and reproducibility audit | Codex | 2026-08-24 | done; documentation-only findings below |
| Algorithm/ansatz/generator boundary and cost-source audit | Codex | 2026-08-24 | done; author clarification and measured correction below |

| Delete orphaned `pipelines/hardcoded` tests | Codex | 2026-08-24 | stopped; an implicated test imports an existing module, as recorded below |
| Extract default no-prune response accounting into `response_accounting.py` | Codex | 2026-08-24 | done in `4b33e682` on `codex/paper-i-worklog-audit-20260824` |
| Delete the Phase-0 cost path | Codex | 2026-08-24 | stopped; preserved Bundle-9 checkpoint fails in the untouched SR route-profile validator before replay; no compatibility shim or implementation commit |
| Q20 — unify `metric_proxy` / `F_metric` and delete the `abs(gradient)` substitution | Codex | 2026-08-24 | in progress; target at least 15 net lines removed, with the 35-reference proxy cluster removed only if reachability proves it dead |
| _(add yours)_ | | | |

---

## CORRECTIONS to the committed Codex handoff

`agent_guidance/static-adapt/HANDOFF_ADAPT_PIPELINE_DECOMPOSITION_20260824.md`
(commits `a00cf1fa`, `64e42db0`) is still broadly sound on the diagnosis, but
**three of its numbers are wrong and one of its increments should not be run as
written.** Measured on `00a5f098`.

| claim | handoff | measured | impact |
|---|---|---|---|
| Increment 1: restoring `route_identity.py` fixes 34 of 55 collection errors | 34, "55 → 21" | **2**, "55 → 53" | **Increment 1 is not worth running as specified** |
| Production code imports `route_identity` | implied load-bearing | **zero importers**; only two `.md` files mention it | restoring it adds an unused module |
| Profile inheritance depth | 7 levels | **5** | cosmetic |
| Descendants of the 116-key root | 27 | **6** | changes the drift story — see below |
| CLI flags | 467 | **450** (409 in `cli_config.py`) | cosmetic |
| The `run_profile(...)` seam | to be designed | **already exists** as `run_ra_adapt`, 3 params | changes the work from design to migration |

### Why Increment 1 is wrong

The prior session counted files *containing the string* `route_identity`. There
are 18 in `test/`, but 17 use it inside test names and assertion strings
(`test_summary_rejects_..._noncanonical_route_identity`), and the one real
import is `historical_route_identity`, which **exists** at
`pipelines/static_adapt/historical_route_identity.py`.

```bash
# 18 files mention it
grep -rl "route_identity" test/ | wc -l
# 0 files import the missing module
grep -rl '^\s*\(from\|import\).*[^_]route_identity' test/ | wc -l
# only 2 collection errors are actually caused by it
```

This is exactly the failure mode the handoff's own §4 warns about.

### What the 55 collection errors actually are

~30 distinct causes, dominated by a missing package surface, not by the beam
refactor:

| cause | count |
|---|---|
| `pipelines.hardcoded.*` — ~12 modules (`hubbard_pipeline`, `hh_pareto_tracking`, `hh_staged_cli_args`, `hh_time_dynamics_spectra`, …) | ~17 |
| `docs.reports.report_labels` | 5 |
| `chtc.*` and `pipelines.exact_bench.*` | ~8 |
| `plots`, `pipelines.error_protected.contracts`, `pipelines.excited_dynamics.*`, misc | ~8 |
| `pipelines.static_adapt.route_identity` | **2** |
| `FileNotFoundError` (missing fixture files) | 3 |

Reproduce:

```bash
python3 -m pytest test --collect-only --continue-on-collection-errors --tb=short
```

That pattern reads like a package that was moved or removed without updating
its tests. **Open question for the author:** was `pipelines/hardcoded/`
deliberately retired? If so these tests should be deleted or ported, not
repaired.

---

## Verified findings _(Claude-owned)_

Every number here was measured from source on `00a5f098`, not inherited.

### F1 — The prescribed seam already exists

`run_ra_adapt(problem, request, operational_controls)` —
`pipelines/static_adapt/ra_adapt/engine.py:5625–6388`, **3 parameters**.
This is the contract's `run_profile(...)` seam and ADR-0001's "one run
operation". **Do not design a competing architecture; the decision is made.**

The work is migration, not design: the interface is deep, but the
implementation still lives in the monolith.

### F2 — The 348-name reflective seam (largest hazard)

```
adapt_pipeline.py:72076   _CANONICAL_SR_SNAKE_LEGACY_EXECUTOR_PARAMETER_NAMES
                          = frozenset(inspect.signature(_run_hardcoded_adapt_vqe).parameters)
adapt_pipeline.py:72081   _canonical_sr_snake_legacy_executor_kwargs(...)
                          -> keeps only keys matching those 348 names
```

- `_run_hardcoded_adapt_vqe` — `adapt_pipeline.py:14876–56085`, **41,210
  lines, 348 keyword-only parameters, no `**kwargs`**.
- **Keys that do not match a parameter name are dropped silently.** Removing a
  parameter loses values instead of raising.
- Two adapters cross this seam: `cli_config.py:3736` (CLI) and
  `sr_snake/_context.py:249` (typed). Two adapters make the seam real; its
  placement at a 348-name signature makes it shallow.

Reproduce:

```bash
python3 - <<'PY'
import ast
t=ast.parse(open('pipelines/static_adapt/adapt_pipeline.py').read())
for n in ast.walk(t):
    if isinstance(n,ast.FunctionDef) and n.name=='_run_hardcoded_adapt_vqe':
        a=n.args
        print(n.lineno, n.end_lineno, len(a.posonlyargs+a.args+a.kwonlyargs), a.kwarg)
PY
```

### F3 — Settings drift: five disconnected roots, no shared base

31 settings dicts in `sr_snake_route_profile.py` (4,588 lines), dispatched by
`normalize_sr_route_profile_namespace:3904–4233` (330 lines, 37 branches).

| root | own keys | descendants |
|---|---|---|
| `CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1` | 18 | **20** — the active family |
| `CANONICAL_SR_SNAKE_V1` | 116 | 6 — the legacy family |
| 3 × `HISTORICAL_SR_SNAKE_*` | 1–4 | 0, isolated |

**The family carrying 20 of the profiles does not inherit the 116-key canonical
root.** So most keys were never pinned by the profile at all — they fall through
to parser defaults. That, not a deep chain, is why a minimal change moves many
things at once.

The author's model, adopted: **a profile is a run plus a diff.** One base holds
the complete effective settings; every other profile is base + a named delta.
`CONTEXT.md` already defines **Controlled ablation** this way; the code never
implemented it.

### F4 — Guard sprawl is a symptom, not the disease

**1,878 `raise` statements across 87k lines — one every 46 lines.** In
`sr_snake/_selection.py` and `sr_snake/_transition.py`, **one every 9 lines**.

Of 1,867 that carry a message:

| kind | count |
|---|---|
| agreement between two representations | **190** |
| required-but-missing | 212 |
| value range / finiteness | 88 |
| type / shape assertion | 58 |
| forbidden combination | 49 |
| unknown enum value | 37 |
| uncategorised | 1,233 |

Samples of the 190:

> "Fresh Phase-III Gram candidate position differs from the admitted singleton position."
> "Default runtime sidecar identity set disagrees with the immutable admission decision."
> "Resolved SR-SNAKE profile does not match its required legacy controller contract."

Each is only possible because the same fact is stored twice — once in the typed
decision, once in the flat runtime dict. **Do not delete these guards.** They
are catching real drift. They become unreachable once F2 and F3 remove the
duplication, and can go then.

---

## Recommended order

Numbered by dependency, not priority.

1. **Close the reflective seam (F2).** Replace the `inspect.signature` filter
   with an explicit typed payload; unknown key raises. Implementation does not
   move. Smallest change, and it makes the rest safe.
2. **Materialize the profile as base + delta (F3).** Must come *after* 1:
   today the filter would silently truncate a newly complete mapping to the 348
   names it recognises, and the result would look like it worked.
3. **Move execution behind `run_ra_adapt` (F1).** Use the `sr_snake` package
   that already holds selection, transition and resume (~528 KB). The mega
   function becomes a **Compatibility route** in the `CONTEXT.md` sense.

Guard sprawl (F4) is retired as a consequence of 1 and 2, not as its own task.

---

## Evidence state

Increment 0 is **done** — `5e6fcb17` on branch `golden-rescue-20260824`,
preserved under `agent_guidance/static-adapt/golden/` with `MANIFEST.sha256`
passing. That branch is not merged; merge it before relying on the data.

**Blocking evidence problem.** `bundle3_final_results_manifest.json` is a
pointer file recording 24 inputs with hashes. **20 are already gone** — they
lived in a `/private/tmp` scratchpad of another session under the iCloud
checkout. The 4 survivors all match their recorded hashes and are preserved.

Among the missing is `kstar_tables.json` @ `629c8c13…`, the version that
generated the Bundle-3 PDF; no surviving copy matches. **Bundle 3 cannot
currently be re-derived from its recorded inputs, so contract Gate 1 is not
satisfiable for it.** The contract forbids filling those cells from another
bundle. This is an author decision, not an agent decision.

The two run-archive trees (11 GB + 2.5 GB) are hash-recorded in place and
**still have no off-repo backup**.

---

## Open decisions — author's, not an agent's

1. **Bundle 3.** Gate 1 is unsatisfiable for it. Accept, re-run, or re-scope?
2. **`pipelines/hardcoded/`** — deliberately retired? Decides whether ~17
   collection errors are repairs or deletions.
3. **`support_frontier.py`** (Paper II) — delete and correct four documents, or
   re-wire? Affects a scientific claim. Commit `657239a3` does not say which was
   intended.
4. **Canonical realtime runner** — root `README.md` calls
   `runners/hh_from_adapt_artifact.py` the "realtime anchor";
   `pipelines/time_dynamics/README.md` calls it legacy awaiting migration.
5. **H2O / Paper-IV material in the Paper-I lane** — relocate, or document an
   exception to root `AGENTS.md`?
6. **Bundle-9 candidate-gain item** — `phase3_candidate_gain_policy=joint_total_gain_v1`
   versus the marginal joint-minus-active-only score. Contract requires
   resolution before numerical lock.

---

## Traps

- **`pytest test` runs zero tests.** The 55 collection errors abort the
  session. Always pass `--continue-on-collection-errors`.
- **Removing a parameter from `_run_hardcoded_adapt_vqe` silently drops
  values.** It does not raise. Until F2 is fixed, a green suite does not mean
  settings survived.
- **`output/` and `prompt-exports/` are gitignored** (`.gitignore:49`).
- **Thirteen worktrees exist** and several Codex branches were committed to on
  2026-08-24. Stage explicit paths only.
- **AST/word-boundary analysis has produced two wrong verdicts in this effort
  already** (three live cost symbols marked dead via closures; `route_identity`
  credited with 34 errors it does not cause). Grep, read the call site, run the
  test.
- **The contract's "Deliberately out of scope" list binds the refactor:** do
  not change optimizer tolerances, estimator accounting, forced-admission
  behavior, or cost normalization under the label of refactoring.

---

## Live findings _(append-only; anyone may add)_

Add dated entries. Name the entry you are correcting, if any.

### 2026-08-24 — Claude — full-suite failure census (SETTLED)

```
789 failed, 4789 passed, 18 skipped, 88 errors in 1233.23s (20:33)
peak RSS 1168 MB
```

88 errors = the 55 collection errors + 33 runtime errors.

**Most of those 789 failures are not defects.** Re-running the worst offenders
in isolation:

| file | in full suite | run alone | pollution |
|---|---|---|---|
| `test_ra_adapt_semantic_closure_routes.py` | 77 | **2** | 75 |
| `test_ra_adapt_facade.py` + `test_static_adapt_sr_snake_resolved_context.py` | 70 | **7** | 63 |
| `test_generic_static_benchmark.py` | 104 | **41** | 63 |

**251 reported, 50 real — ~80% is cross-test interference.** Tests share mutable
module-level state, so outcomes depend on execution order.

Corrects the earlier partial entry in this log, which reported "~16% of tests
fail". That figure is inflated and should not be quoted.

The suite is also slow: `test_ra_adapt_semantic_closure_routes.py` alone takes
8m16s for 92 tests. Slow plus order-dependent is why nobody runs it, which is
why no failure count existed.

**Consequence for the refactor:** the suite cannot serve as the acceptance
signal until isolation is fixed. Before trusting any test result, re-run the
affected files alone and compare. A file's failure count in a full run is not
evidence about that file.

### 2026-08-24 — Claude — the gates are a scaffold, not a standing rule

Correcting an overstatement I made earlier in this effort. I described the
behavioral contract as *forbidding* changes such as adopting Qiskit Nature for
operator construction. It does not. It says such a change "requires its own
named profile, rerun, and manuscript review." Those are different claims and I
collapsed them into a prohibition.

Framing to carry forward:

- "The accepted generator sequence must not move" is valuable **while code is
  being moved and no numerical change is intended** — it is the cheap way to
  distinguish "I refactored" from "I changed the science and did not notice",
  which is the author's original drift symptom.
- It has near-zero value as a **permanent** rule, and as a standing ban on
  adopting better libraries it costs more than it returns.
- Scope the gates to the refactor window. Do not treat them as an acceptance
  criterion for all future work, and do not let them block a change the author
  has decided to make deliberately.

### 2026-08-24 — Claude — one root cause behind three symptoms

The guard sprawl (F4), the settings drift (F3), and the ~80% test pollution
above are the same defect in three places: **the same fact is stored in several
places, and each copy needs a check that it still agrees with the others.**

- 190 guards asserting two representations agree
- 31 settings dicts across 5 roots, so no single place states what a profile is
- tests sharing mutable module state, so order changes outcomes

Adding checks does not fix this, and deleting checks is unsafe because they are
catching real drift. Collapsing the duplicate representations fixes it, and the
checks then have nothing left to disagree about.

### 2026-08-24 — Codex — worklog consistency and reproducibility audit

This entry corrects the **Anchors**, **Evidence state**, F1, the
`route_identity` correction row, and the worktree-count trap without rewriting
their owners' text.

#### Current integration state

The active coordination branch advanced after the measurements above:
`paper-ii-exchange-selector` is now at `c223b419`, which includes the
golden-data rescue merged by `e0cb8175`. Commit `5e6fcb17` is an ancestor of
that HEAD, and the tracked golden
manifest passes. Therefore, the statement in **Evidence state** that the branch
is not merged no longer applies. Keep `00a5f098` as the explicit measurement
baseline for the 55-error census and F1–F4 unless a later entry remeasures them;
do not treat it as the current integration HEAD.

Reproduce:

```bash
git -C /Users/jakestrobel/local_repos/Holstein_test_fullclone_3 \
  branch --show-current
# -> paper-ii-exchange-selector
git -C /Users/jakestrobel/local_repos/Holstein_test_fullclone_3 \
  rev-parse --short HEAD
# -> c223b419
git -C /Users/jakestrobel/local_repos/Holstein_test_fullclone_3 \
  merge-base --is-ancestor 5e6fcb17 HEAD
# -> exit 0
(cd agent_guidance/static-adapt/golden && \
  shasum -a 256 -c MANIFEST.sha256)
# -> ten entries report OK
```

#### F1 is an existing implementation seam, not yet a proved contract seam

The implementation has two positional parameters, `problem` and `request`,
plus the keyword-only `operational_controls`. The Paper-I lane contracts define
the ordinary public interface exactly as `run_ra_adapt(problem, request=None)`;
the behavioral contract's `run_profile(profile_id, problem_id, arm, horizon)`
is explicitly conceptual. Read F1 as evidence that migration has a destination,
not as proof that the current facade already satisfies profile selection,
receipt completeness, or Gates 1–4. `operational_controls` must remain limited
to the validated-protocol use described by its docstring.

The same guard currently contains explicit exceptional admissions for HH
`L=3`, a named pure-Hubbard application, and the H2O family even though its
error text and the ordinary lane contract say the facade is locked to HH
`L=2`. This is pre-existing admissibility/compatibility debt related to open
decision 5, not permission to remove or generalize those paths during the seam
migration.

Reproduce:

```bash
python3 - <<'PY'
import ast
from pathlib import Path

tree = ast.parse(Path("pipelines/static_adapt/ra_adapt/engine.py").read_text())
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "run_ra_adapt":
        args = node.args
        print("positional", [arg.arg for arg in args.posonlyargs + args.args])
        print("keyword_only", [arg.arg for arg in args.kwonlyargs])
PY
sed -n '5625,5675p' pipelines/static_adapt/ra_adapt/engine.py
```

#### Three measurement corrections

- The test-baseline paragraph's ~16% figure is superseded by Claude's settled
  census at `c223b419` and must not be quoted. The Claims row still says the
  census is in progress; use the settled Live finding for its status and
  isolation warning.
- At `00a5f098`, **11** Markdown files mention `route_identity`, not two. The
  load-bearing conclusion remains narrower and supported: no Python file under
  the active `pipelines/`, `src/`, or `test/` surfaces imports the missing
  `pipelines.static_adapt.route_identity` module. Archived/CHTC snapshots do
  contain imports and are outside that conclusion. Documentation mentions must
  not be used as an importer count.
- Worktree counts are volatile. There were **16** registered worktrees during
  this audit, not 13. Always run `git worktree list`; never use the recorded
  count as a cleanup target. In addition, `output/` is ignored at
  `.gitignore:49`, while `prompt-exports/*` is ignored at `.gitignore:37`.
  `git check-ignore -v` is the durable check.

Reproduce:

```bash
git grep -l route_identity 00a5f098 -- '*.md' | wc -l
# -> 11
git grep -l -E \
  '^[[:space:]]*from[[:space:]]+(\.+|pipelines\.static_adapt\.)route_identity[[:space:]]+import|^[[:space:]]*import[[:space:]]+pipelines\.static_adapt\.route_identity' \
  00a5f098 -- 'pipelines/**/*.py' 'src/**/*.py' 'test/**/*.py' | wc -l
# -> 0
git worktree list --porcelain | awk '$1 == "worktree" { count++ } END { print count }'
# -> 16 at audit time; refresh rather than preserve this count
git check-ignore -v output/example prompt-exports/example
# -> .gitignore:49 and .gitignore:37, respectively
```

#### F3 needs a provenance boundary

"Base + delta" may be an implementation strategy for materializing complete
effective settings, but it is not a scientific inheritance claim. The
behavioral contract defines H-L3, HH-B3, HH-B5, and HH-B9 as separate protected
regression profiles and explicitly forbids treating Bundles 3, 5, and 9 as a
one-factor ablation. Each resolved profile must therefore remain independently
complete and receipt-identifiable; any shared base must be internal, with every
effective difference explicit and no fallback to parser defaults.

#### GitNexus scope — author directive, 2026-08-24

GitNexus use for this refactor is Paper-I-local, not repository-wide. Limit
queries and change analysis to `pipelines/static_adapt/`,
`agent_guidance/static-adapt/`, named Paper-I tests, this worklog, and the ADR
named in **Anchors**. Do not run a repository-wide re-index or use unrelated
paper-lane symbols, test failures, or execution flows to justify a Paper-I
change. If the current index cannot isolate that surface, ratify claims with
path-limited source, AST, and Git commands until a dedicated Paper-I index is
available.

#### The recommended order is diagnostic, not executable yet

Items 1–3 do not yet meet handoff-contract §3: they lack exact verification
commands, expected results tied to the `00a5f098` baseline, and stop conditions.
Do not claim an implementation increment from this list alone. First append a
bounded increment containing those four elements and state whether Codex should
continue autonomously or pause. The broken full-suite baseline makes targeted
profile/adapter tests and change-specific failure deltas mandatory; a raw
"tests pass" statement is not an acceptance result.

---

### 2026-08-24 — Claude — CORRECTION: Bundle 3 is probably re-derivable

Corrects "Evidence state" above, which says Bundle 3 "cannot currently be
re-derived from its recorded inputs, so contract Gate 1 is not satisfiable for
it." **That was over-claimed.** What is gone are *derived intermediates* from a
dead scratchpad, not the upstream evidence.

Measured:

- The 18 Bundle-3 run archives are intact and hash-recorded. Each carries
  `run/checkpoints/current.json`, `run/result/estimator_ledger.json`, **and a
  full `source/` snapshot of the pipeline that produced it**.
- The renderer named in the manifest, `results_tooling/build_variant_pages_v7.py`,
  **exists at the exact recorded tooling commit `551a9fe0`**:

```bash
git show 551a9fe0:chtc/paper_i_ra_adapt_repair_20260727/results_tooling/build_variant_pages_v7.py
```

So the path archives → renderer → `kstar_tables.json` appears intact.
`kstar_tables.json` is an aggregate reporting table (keys: `kstar`, `aavqe`,
`b1`, `b3`, `live_star`, `spot_check`, by regime), i.e. a *derived* artifact.

**Status: plausibly recoverable, not confirmed.** One experiment settles it —
regenerate with the recorded renderer and compare against
`bundle3_final_results_manifest.json`. Do not describe Bundle 3 as lost
evidence until that experiment has been run and failed.

### 2026-08-24 — Claude — CORRECTION: `pipelines/hardcoded/` was not deleted

Corrects "What the 55 collection errors actually are" above, which reads as if
the package was removed. **The package still exists** with 8 modules
(`adapt_circuit_execution.py`, `hh_continuation_pruning.py`,
`hh_staged_noise*.py`, and others). What is missing are ~12 specific modules
the tests import (`hubbard_pipeline`, `hh_pareto_tracking`, `hh_staged_cli_args`,
`hh_time_dynamics_spectra`, …). Last touched by `6442fbb5`
"Replace Snake snapshot with RA". This reads as a partial deliberate retirement
that left its tests behind — still an author question, but a narrower one.

---

### 2026-08-24 — Claude — author's design target, and two measured facts behind it

**Design target:** see "THE DESIGN TARGET" at the top of this file.
Author's specification, not an agent proposal. Three files: the loop, the
generators/score/cost, the optional extensions. Phases 0-III are one function at
increasing response order, not four code paths. One cost-term interface with a
qiskit and a proxy implementation sharing one normalization, selected at
runtime. Batch/prune/beam defined after, absent from the default path. Fixed vs
adaptive shortlisting is a rule over the score, not a mode.

**Fact 1 — candidate-gain semantics differ by bundle.** Census over all 13
Bundle-9 archives and 5 Bundle-3 archives, from the run checkpoints:

| bundle | `joint_gain_semantics` | `active_only_baseline` | receipts |
|---|---|---|---|
| Bundle 9 (13/13 cells) | `full_joint_trust_gain_legacy_v1` | `None` in all | 2,040 |
| Bundle 3 (5 cells) | `incremental_candidate_gain_v1` | populated in all | 1,746 |

Unanimous, no mixing. Bundle 9 ranked candidates by legacy total joint gain;
Bundle 3 by the incremental gain. **Both gain policies have been run, but on
different bundles**, so they are confounded with every other B3/B9 difference.
A clean comparison requires both within one bundle — which is why the author's
answer to "which policy" is "try both, full".

The counterfactual cannot be recovered from existing receipts: Bundle 9 never
computed the active-only baseline, so there is no second ranking to extract.

Reproduce: extract `./run/checkpoints/current.json` from an archive and collect
objects with `schema == "phase3_candidate_gain_receipt_v1"`.

**Fact 2 — a profile's keys have three different fates, and nothing says
which.** Instrumenting the 348-name filter on the typed path, 22 keys are
dropped per call. They are not one thing:

- **7** reach the executor under an unprefixed twin with the identical value
  (`adapt_seed=7` → `seed=7`). Harmless duplication.
- **several** are consumed through the typed contracts instead, not the legacy
  kwargs (`phase_live_hysteresis_enabled` is read by `sr_snake/contracts.py` and
  `ra_adapt/runtime.py`). Dropping them here is correct.
- **at least one is read by nothing at all**: `phase1_prune_small_theta_abs` is
  declared in `CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS` and appears in no other
  `.py` file.

20 of the 22 are declared by a canonical profile;
`phase_live_hysteresis_enabled` by seven of them.

**This corrects an earlier over-claim of mine in this session** that the filter
was silently losing science. It is not, mostly. The real defect is that reading
a profile does not tell you which of its keys take effect — some reach the
executor, some reach a different channel, some reach nothing. That is the
author's drift symptom, and it is rule 3 of the design target.

Reproduce: wrap `adapt_pipeline._canonical_sr_snake_legacy_executor_kwargs` and
record keys absent from `_CANONICAL_SR_SNAKE_LEGACY_EXECUTOR_PARAMETER_NAMES`.

---

### 2026-08-24 — Claude — the phase collapse is verified, and the Gram formulation holds

The design target rests on phases 0-III being one function. Checked against
`pipelines/scaffold/hh_continuation_scoring.py`. **It holds.** All three phases
compute the same quantity — predicted descent over cost — at increasing fidelity
of the descent estimate.

| phase | what it computes | source |
|---|---|---|
| I | `trust_region_drop(g_hw_lcb, lambda_F*F, F, rho)` | `phase1_trust_region_gain:2614` |
| II | same, with measured curvature `h_raw` | `phase2_raw_geometry_score:3168` |
| III | `S3_primary = DeltaE_TR / (1 + K3)` | `phase3_canonical_score_components:3831` |

Phase III's docstring states the canonical form outright: *"Canonical static
Phase3 scoring is physics-first: S3_primary = DeltaE_TR / (1 + K3)"*. That is
`evaluate(record, state, cost, order)` — descent estimate at `order`, divided by
the cost term. The collapse is real.

**The author's Gram formulation is also confirmed, negatively.** Geometry should
be one Gram over generators and ansatz, with phases II, I and 0 as successive
restrictions, coupled to the phase scoring. Today it is not:

1. `phase2_raw_geometry_score` accepts the Gram blocks `q_window, Q_window` and
   **immediately discards them**:

   ```python
   del q_window, Q_window
   ```

   Its own docstring: *"The Gram cross-block inputs remain in the call contract
   because the same exact geometry is reused by insertion, supported response,
   and the deferred all-models-infeasible fallback. They are not ranking
   inputs."* The geometry is threaded through the scorer and dropped, to keep a
   call contract alive.

2. `CandidateFeatures` (`hh_continuation_types.py:79-334`) has **249 fields**, of
   which **26 are separate summaries of that one geometric object**:
   `F_metric`, `metric_proxy`, `cheap_metric_proxy`, `phase2_raw_overlap_max`,
   `phase2_span_projection_z`, `phase2_geometry_window_indices`,
   `phase3_geometry_window_size`, `phase3_geometry_refit_window_indices`,
   `phase3_geometry_active_post_indices`, and more.

   Under one Gram these are submatrix selections, not stored fields. This is the
   same defect as F3 and F4 — one fact, many stored copies, each needing a guard.

3. Several of the 26 are not data at all but **policy smuggled into the
   per-candidate record**: `phase2_curvature_policy`,
   `phase2_geometry_window_policy`, `phase3_geometry_window_policy`,
   `selector_geometry_mode`. Policy belongs in `request`, resolved once, not
   carried on every candidate.

**Consequence for the target:** section 2 needs no revision. `score()` and the
phase table are sound, and `evaluate` is `descent(order) / cost`. The work is to
build the one Gram and express phases 0-II as restrictions of it, which deletes
most of the 26 fields and the call-contract threading with them.

---

### 2026-08-24 — Claude — scoped GitNexus index: useful for inventory, unsafe for reachability

Built a second index scoped to the Paper-I lane, including the mega file:

```bash
cd pipelines/static_adapt
GITNEXUS_MAX_FILE_SIZE=8192 node ../../.gitnexus/run.cjs analyze pipelines/static_adapt \
  --skip-git --skip-agents-md --skip-skills --force
# 26.7s, peak 1812 MB, 122 MB index
# 13,097 nodes | 20,971 edges | 358 clusters
```

Notes:

- `pipelines/static_adapt` is not itself a git repo, so `--skip-git` is
  required. **Consequence: no `detect_changes` on this index** — the root index
  keeps commit tracking but is stale at `ade04b3`.
- `adapt_pipeline.py` is 3.3 MB and is skipped by the default 512 KB cap. With
  the cap raised it contributes ~1,890 nodes and ~3,483 edges, so the mega
  function does decompose into symbols rather than collapsing to one node.
- All three indexes are named `Holstein_test`, so every CLI call needs
  `--repo <absolute path>`.
- `--skip-agents-md --skip-skills` matter here: `AGENTS.md` and `CLAUDE.md` have
  uncommitted edits in the working tree.

**Do not trust its reachability verdicts.** `impact _run_hardcoded_adapt_vqe
--direction upstream` returns `impactedCount: 0, risk: LOW` for a 41,210-line
function. Source shows real callers:

| caller | note |
|---|---|
| `sr_snake/_context.py:885` | `adapt_pipeline._run_hardcoded_adapt_vqe(**executor_kwargs)` — the typed path's real invocation, **inside the indexed subtree** |
| `exact_bench/hh_static_ground_state_benchmark.py:975` | second product caller, outside the subtree |
| `test/test_static_adapt_full_reopt_duplicate_guard.py`, `test/test_paper_i_hh_route_a_repair.py` | tests |

The in-subtree miss is a module-attribute call through a **function-local
deferred import**. Static call graphs do not follow that, and this codebase uses
the pattern heavily.

**This is the third wrong index verdict in this effort**, after the three live
cost symbols marked dead through closures and `route_identity` credited with 34
collection errors it does not cause. Use the index for inventory, clustering and
navigation. Ratify every reachability or deletion claim against source.

---

### 2026-08-24 — Claude — measured estimator ledger: all three response kinds are charged

Answers the open item in 6d, and **corrects it**. Curvature *does* have its own
primitive kind. My earlier claim that it did not came from grepping string
literals in source, which missed it — the fourth time in this effort an index or
grep proposed something source contradicted.

From a completed Bundle-9 cell
(`run/checkpoints/current.estimator_call_ledger_checkpoint.*.json`,
schema `paper_i_estimator_call_ledger_checkpoint_sidecar_v2`):

| primitive_kind | count | key form | response object |
|---|---|---|---|
| `coordinate_gradient` | 14,325 | unary operand | **g** |
| `hamiltonian_expectation` | 3,888 | observable identity only | energies |
| `metric_element` | 1,436 | symmetric pair | **G** |
| `hessian_element` | 1,300 | symmetric pair | **H** |

Pair keys total 2,736 = 1,436 + 1,300 exactly; unary keys 14,325 = gradients
exactly. **All three response objects already charge as distinct kinds with the
right key shapes**, so `Response.charge(order, support)` in 6d maps directly onto
existing machinery. No new accounting is required.

### Correction: nesting reuse is real but small

6b said the ledger's dedup "is the reuse policy" and implied it as a benefit.
Measured:

```
raw_occurrence_count    = 21,605
unique_primitive_count  = 20,949
reuse factor            = 1.03x   (656 repeated consumptions charged once)
```

Only 3% of charges are dedup hits. The reason is structural: the key includes
`projective_state_fingerprint`, and the accepted state changes every controller
round, so keys cannot collide across rounds. Reuse is available only *within* a
round — roughly 3-10 hits per round here.

**So do not sell the one-Gram design as a measurement saving.** Its accounting
value is correct *attribution* — one entry, one primitive, one kind — not fewer
measurements. That value stands; the saving does not.

### A fact worth carrying into the manuscript

Gradients are **10x** every other primitive combined (14,325 vs 1,436 + 1,300).
The estimator economy is dominated by `coordinate_gradient`, not by the
expensive-looking high-order geometry. If screening cost is discussed as a
motivation for staged phases, this is the number that supports or undercuts it.


### 2026-08-24 — Codex — author clarification: the algorithm calls domain objects

This corrects the **"Three files"** target above. The mathematical algorithm
must not construct or encode the ansatz or the generator population. It calls
those domain objects through small interfaces.

| module | owns |
|---|---|
| `algorithm.py` | only the controller loop, Phase 0--III restriction/evaluation order, shortlist, admission, and acceptance orchestration |
| `ansatz.py` | the ordered accepted generators and parameters, state construction, insertion, and refit/update operations |
| `generators.py` | generator identity and support, population construction, records, macro-to-singleton exposure, scoring, and the two cost terms |
| `extensions.py` | optional batch, prune, and beam composition after the singleton route |

Only `algorithm.py` is the algorithm. Its functions receive an `Ansatz` and a
generator population and return their next immutable values. Concrete pool
construction, operator algebra, circuit construction, telemetry, and receipts
do not enter that file. The public command may still be
`run_ra_adapt(problem, request=None)`; its adapter constructs the domain
objects before calling the mathematical route.

The existing `CandidateRepresentationAdapter` already points toward the
generator boundary: it owns `parent_inventory`, `executable_pool`, and
`expose_children`. The Bundle-5 macro-to-singleton step should become one
population transformation after Phase I, while `score` remains unchanged.

### 2026-08-24 — Codex — correction: cost normalization is partly shared already

This corrects the design-target sentence saying the current cost sources have
"no shared normalization." They lack one small `CostTerm` interface, but the
normalization math is already partially centralized in
`pipelines/scaffold/hh_continuation_scoring.py`:

- `hardware_cost_family_normalization(...)` dispatches the population
  statistics;
- `apply_symmetric_hardware_cost_normalization(...)` applies the common
  bounded transform; and
- `rescore_hardware_cost_family(...)` applies the resulting factor to Phase
  I, II, and III score fields. Targeted tests cover both the proxy-family and
  signed-Qiskit policies in all three phases.

For component $a\in\{2q,d,1q,\theta,\mathrm{shot}\}$, both current signed
policies use

\[
u_a=\frac{2}{\pi}\arctan\!\left(\frac{c_a-\mu_a}{s_a}\right),\qquad
I=\frac{\sum_a\lambda_a u_a}{\sum_a\lambda_a},\qquad
f=\operatorname{clip}(1-\tfrac12 I,0.5,1.5).
\]

They differ only in the statistics supplied to that transform:

- proxy-family `family_robust_symmetric_arctan_v1` uses the population median
  for \(\mu_a\) and `max(scale_floor, MAD)` for \(s_a\);
- signed-Qiskit `zero_centered_signed_arctan_v1` uses \(\mu_a=0\) and
  `max(scale_floor, median(nonzero(abs(delta))))` for \(s_a\).

The cost-source vocabulary is currently three-layered, not two-layered:
`proxy_logical_ladder_span_v1` is the structural baseline,
`marrakesh_graph_span_v1` is the graph-aware no-transpile proxy, and
`backend_transpile_v1` is the compiled full-base/full-trial marginal. The
simplest source-faithful two-implementation target is therefore:

- `ProxyCost`: the Marrakesh graph-span estimate, using the logical-ladder
  helper only for the coordinates it supplies; and
- `QiskitCost`: signed deltas of compiled \(N_{2q}\), \(D_{2q}\), and
  \(N_{1q}\), with zero theta and shot coordinates.

Both return the same typed `Cost`; the shared normalizer receives that cost and
the selected statistics policy. There is no string alias dispatch and no
substitution between implementations. In particular, the current branch in
`normalize_hardware_cost_feature_family(...)` that changes a requested signed
policy to `family_robust_v1` when signed telemetry is absent violates the
author's no-fallback rule and must become an error at the extracted seam.

One semantic mismatch must remain explicit: the target says cost is available
in Phases I--III and Phase 0 is the raw-gradient pilot, while current semantic
Phase-0 proxy routes divide by a graph-proxy denominator. Removing Phase-0 cost
would change rankings; do not hide that change inside the structural
extraction. Make it a separately named semantic change with its own evidence.

GitNexus was used only for exact Paper-I symbols
`CandidateRepresentationAdapter` and `select_adaptive_phase_shortlist`. Its
index is stale at `ade04b3`, so the newer cost functions were ratified directly
from the Paper-I source and targeted tests rather than triggering a repo-wide
re-index.

Reproduce:

```bash
rg -n 'class Phase1CompileCostOracle|class MarrakeshGraphSpanCostOracle' \
  pipelines/scaffold/hh_continuation_scoring.py \
  pipelines/static_adapt/hh_backend_compile_oracle.py
sed -n '1050,1210p' pipelines/scaffold/hh_continuation_scoring.py
sed -n '1359,1445p' pipelines/scaffold/hh_continuation_scoring.py
sed -n '430,525p' pipelines/static_adapt/hh_backend_compile_oracle.py
python3 -m pytest -q \
  test/test_hh_continuation_scoring.py::test_symmetric_hardware_cost_factor_applies_to_phase1_phase2_and_phase3 \
  'test/test_ra_adapt_phase123_qiskit_scope.py::test_signed_qiskit_cost_can_reverse_raw_benefit_order_in_every_phase'
# -> 4 passed in 2.85s
```

---

### 2026-08-24 — Claude — three corrections from the first Codex round

**1. The extraction target is `ResponseAccounting`, not `Response`. Codex is
right; 6f was wrong.**

6f said "the `Response` accessor already exists as
`_DefaultNoPruneEstimatorService`". Verified against every method's return type:

| method | returns |
|---|---|
| `_record_estimator_primitive` | `Any` (a receipt) |
| `_record_candidate_self_metric_primitive` | `str | None` |
| `_record_scaffold_geometry_primitives` | `tuple[str, ...]` |
| `_record_candidate_geometry_primitives` | `tuple[str, ...]` |
| `_record_active_gradient_primitives` | `dict[str, Any]` |
| `_record_candidate_pair_geometry_primitives` | `dict[str, Any]` |

**Not one returns a Gram, Hessian or gradient array.** They return primitive ids
and receipts. The class implements `Response.charge(order, support)` and nothing
else — the accounting half. The `g()/H()/G()` value half lives elsewhere and is
still to be located. Naming the extracted module `Response` would assert an
interface it does not provide.

Codex's second point also stands: extraction must carry the tangent-identity
helpers (`_active_physical_tangent`, `_candidate_physical_tangent`, and
`PhysicalTangentOperandIdentity`) or the new module imports back into
`adapt_pipeline.py`.

**2. Collection baselines in the main checkout are contaminated.**

I gave Codex a baseline of 5598 collected / 54 errors. Codex measured 5573 / 54
and was right to trust its own. Cause: **14 test files are modified-uncommitted
in the shared working tree** (`test/test_ap_mclachlan_*.py`,
`test_paper_ii_runs.py`, `test_time_dynamics_*.py`) — another agent's in-progress
Paper-II work. My count includes their uncommitted edits; a clean worktree at a
committed SHA does not.

Collection is deterministic within a tree (5598 twice, back to back). It is not
comparable across trees.

**Rule: quote a baseline with its commit SHA *and* whether the tree was clean.**
A number measured in the shared checkout is not a shared anchor.

**3. The shared surface forked into three copies, and the protocol caused it.**

This file's protocol says "commit this file on its own so a conflict is a
one-file conflict". It did not account for the repository's worktree-isolation
invariant, which *requires* Codex to work on its own branch — so its worklog
edits were invisible here for the whole session (`f39f4638`, `bcc169cb`,
`c510624d`, `8133d01c`), and mine were invisible to it.

Merged at `2c999dcf` and `a467a897`, both resolved by keeping both sides, since
Live findings and Execution log are append-only.

**Added to the protocol:** before appending, `git fetch` and merge the other
agent's branch into yours; after committing, say which branch the commit is on.
A worklog entry on an unmerged branch has not been communicated.

---

### 2026-08-24 — Claude — Paper I defines the partitioned form; Phase-0 cost deletion verified safe

**Q13 settled from the manuscript.** `subset_geometry` returning
`(G_AB, G_BB, b_B)` is not archaic — it is Paper I's own structure.
`Paper_I_author_revision.tex`, Eq. `hessian_block_def`:

```
H = [ H_aa      H_a-theta     ]
    [ H_theta-a H_theta-theta ]
```

"three distinct blocks: candidate curvature (alpha-alpha), candidate--ansatz
coupling (alpha-theta), and active-ansatz response (theta-theta)", with **G** the
Fubini--Study metric defining the trust region. Line 231 likewise: "RA uses
candidate--active Gram blocks to define the trust regions of its local response
models".

**Therefore 6d/6e are wrong in their interface shape.** `g()/H()/G()` as three
independent accessors is the generic framing, not this method's.
`Response.restrict(support) -> (cross_block, own_block, rhs)` is the faithful one,
and `selector_query_closure.py:795` already implements it on
`QueryClosedPopulationWorkspace`. The `_support(order)` table in 6e stands; only
the return shape changes.

**Phase-0 cost deletion is a no-op on recorded evidence.** Author approved
deletion 2026-08-24. Verified before any edit:

- `phase0_cost_lambdas = {}` and `phase0_cost_lambda_source = "unresolved"` in
  **5 of 5** cells sampled across Bundle 3 and Bundle 9.
- `phase0_K0` is recorded 30 times in one Bundle-9 cell and is **exactly 1.0
  every time** — one distinct value.

The Phase-0 score is `DeltaE0_upper * N0 / K0` (`adapt_pipeline.py:22346`) with
`denominator = cost_row["hardware_cost_denominator"]` (`:22341`). Dividing by a
denominator that is always 1.0 means removing it cannot change any recorded
ranking.

**Open concern for whoever implements it:** `phase0_K0`,
`phase0_hardware_cost_denominator` and `phase0_hardware_cost_excess_sum` are
written into the checkpoint payload. Removing them changes the checkpoint schema,
which resume and replay read. The numerical no-op does not imply a schema no-op.

**Clean baseline, this worktree, `claude/paper-i-20260824` @ `43213503`:**
`5573 collected, 54 errors` — identical to Codex's independently measured figure,
confirming that the earlier 5598 came from uncommitted Paper-II test edits in the
shared checkout.

---

## Execution log _(Codex-owned)_

Append one entry per increment: goal, commands run, measured result, and
whether the stop condition triggered.

### 2026-08-24 — orphaned `pipelines/hardcoded` tests — STOPPED

**Goal:** delete only tests orphaned by deliberately retired
`pipelines/hardcoded` modules, then prove the collection-error list changed by
removals only.

**Measured baseline:**

```bash
python3 pipelines/shell/ram_guard.py --limit-mb 8000 -- \
  python3 -m pytest test --collect-only -q \
    --continue-on-collection-errors
# -> 5573 tests collected, 54 errors in 23.70s
# -> peak RSS 432 MB
```

The error count matches the supplied baseline; the collected-test count is 25
lower than the supplied 5598 and is reported rather than normalized away.

**Source ratification and stop condition:**

```bash
rg -n 'pipelines\.hardcoded\.hh_staged_noise|hh_staged_cli_args' \
  test/test_hh_staged_noise_workflow.py \
  pipelines/hardcoded/hh_staged_noise.py
test -f pipelines/hardcoded/hh_staged_noise.py
```

`test/test_hh_staged_noise_workflow.py:16-19` directly imports
`pipelines.hardcoded.hh_staged_noise` and
`pipelines.hardcoded.hh_staged_noise_workflow`. Both modules still exist on
disk. Collection fails because the existing `hh_staged_noise.py:14` then
imports the retired `pipelines.hardcoded.hh_staged_cli_args` module.

The handoff explicitly requires stopping if an orphaned test imports a module
that still exists. The stop condition therefore triggered before deletion.
No test file was deleted, no after-collection run was made, and no ERROR-list
diff exists. No implementation or scoring file was touched.

### 2026-08-24 — `ResponseAccounting` extraction — COMPLETE

**Goal:** move the default no-prune estimator-charge service and its tangent
identity helpers out of `adapt_pipeline.py`, without changing a ledger key,
primitive id, receipt, cache replay, or ordering.

**Coordination and clean baseline:** the local `paper-ii-exchange-selector`
branch resolved to `432135036d3f770e874c0f7556a771b19d9c290f`; it was merged before
the claim was appended. The clean pre-extraction baseline, including the
claim-only commit, was
`aa76c086283ce36509b4de39c85d88b5ef8d0c4b`.

```bash
python3 pipelines/shell/ram_guard.py --limit-mb 8000 -- \
  python3 -m pytest test --collect-only -q \
    --continue-on-collection-errors
# -> 5573 tests collected, 54 errors in 21.33s
# -> peak RSS 435 MB
```

**Impact and extraction:** Paper-I-scoped GitNexus reported CRITICAL upstream
risk for `_physical_generator_block_payload` (36 upstream symbols, seven
processes, 12 modules), `_physical_tangent_operand_identity` (three direct
callers, eight processes, 16 modules), and
`_candidate_physical_tangent_operand_identity` (two direct callers, eight
processes, 11 modules). The scoped class lookup under-reported
`_DefaultNoPruneEstimatorService` as zero upstream callers, so source
ratification governed: there is one construction site and the helpers also
serve legacy closures and direct test imports. To preserve that surface,
`adapt_pipeline.py` imports the same private names from the new module; no call
site changed.

Commit `4b33e682` adds
`pipelines/static_adapt/response_accounting.py` and moves, without rewriting,
these definitions:

- `_DefaultNoPruneEstimatorService`;
- `_physical_generator_block_payload`;
- `_physical_tangent_operand_identity`;
- `_candidate_physical_tangent_operand_identity`; and
- the candidate-cache estimator replay schema and field constants.

`PhysicalTangentOperandIdentity` remains owned by
`estimator_call_ledger.py`. Phase scorers, cost terms, legacy closures, and the
g/H/G value interface were not edited.

The dedicated Paper-I GitNexus index is `--skip-git` scoped. A pre-commit
`detect-changes` invocation against this worktree therefore failed closed with
"Repository ... not found" instead of silently using the repo-wide index.
Path-limited Git diff/checks, import identity, focused tests, collection, and
the ledger lock below supplied the change evidence.

**Receipt-identity lock:** the same bounded, cache-disabled completed
Hubbard--Holstein run was executed before and after extraction (`L=2`,
`n_ph_max=1`, full-Hamiltonian pool, one admitted operator, no
prune/batch/rescue/Phase-0). Both runs stopped at `max_depth`, recorded 18 raw
occurrences and 13 unique primitives, and serialized the same full ledger.

```bash
shasum -a 256 ledger-before.json ledger-after.json
# -> 5e1f2ff05251262d30d9eff6dbc09993d77b6d443d9dbb6949cf3033027c34de  (both)
cmp -s ledger-before.json ledger-after.json
# -> exit 0
```

The byte identity covers primitive ids, call keys, entries, occurrence order,
consumer receipts, and the ledger fingerprint. The stop condition did not
trigger.

**Regression evidence:**

```bash
python3 pipelines/shell/ram_guard.py --limit-mb 8000 -- \
  python3 -m pytest -q \
    test/test_static_adapt_estimator_call_ledger.py \
    test/test_adapt_candidate_record_cache.py \
    test/test_static_adapt_sr_v4_runtime.py::test_v4_disabled_finite_angle_switch_skips_flat_gradient_guard \
    test/test_ra_adapt_refactor_parity.py
# -> 61 passed, 2 warnings in 18.72s; peak RSS 476 MB

python3 pipelines/shell/ram_guard.py --limit-mb 8000 -- \
  python3 -m pytest test --collect-only -q \
    --continue-on-collection-errors
# -> 5573 tests collected, 54 errors in 19.04s; peak RSS 429 MB

diff -u errors-before.txt errors-after.txt
# -> empty diff
```

Thus the collection-error list is exactly unchanged and no focused test that
already collected began failing.

**Located, not extracted: the g/H/G value half.** The newly merged Paper-I
interface correction led to the typed full-block carrier that the first search
missed:
`selector_query_closure.py:FormalAdmissionCurvatureReceipt`. The function
`build_formal_admission_curvature_receipt` returns that object with actual
arrays `G_AA`, `G_AB`, `G_BB`, `H_AA`, `H_AB`, `H_BB`,
`descent_gradient_A`, and `descent_gradient_B`. It validates and freezes a
historical Phase-III summary; it is not yet a `restrict(support)` response
interface. In the manuscript-faithful partitioned form,
`QueryClosedPopulationWorkspace.subset_geometry` already returns actual
`(G_AB, G_BB, b_B)` arrays for a candidate subset.

The upstream full materializations feeding related paths are
`hh_continuation_scoring.py:_build_phase2_joint_geometry_cache`, which returns
`_Phase2JointGeometryCache` arrays `g_A`, `g_B`, `G_AA`, `G_AB`,
`G_BB_diagonal`, `H_AA`, `H_AB`, and `H_BB_diagonal`, and
`hh_continuation_scoring.py:_build_batch_full_geometry_workspace`, which
returns `_BatchFullGeometryWorkspace` arrays `g_A`, `g_B`, `G_AA`, `G_AB`,
`G_BB`, `H_AA`, `H_AB`, and `H_BB`.

Other partial producers are
`engine_support.py:evaluate_exact_gradient_surface` (gradient arrays),
`accepted_refit.py:_fubini_study_gram` (a Gram array),
`hh_continuation_scoring.py:_selector_scaffold_context` and
`OrderedInsertionGeometryOracle.prepare_scaffold_context` (`Q_window` and
`H_window_hessian`), and
`exact_geometry_backend.py:CompiledExactManifoldAdapter._state_gradient_tangents`
(a gradient and tangent arrays). None was changed in this increment.

### 2026-08-24 — Phase-0 cost-path deletion — STOPPED

**Goal:** delete the numerically inert Phase-0 cost denominator and its
checkpoint fields under Q14/Q17, with at least 80 net lines removed, while
preserving estimator receipt identity and resuming an existing Bundle-9
checkpoint that still carries `phase0_K0`.

**Coordination and clean baseline:** `paper-ii-exchange-selector` at
`bfd62d8d` was fast-forward merged before the claim was appended. The clean
baseline, including the claim-only commit, is
`eeaa858e02a44022c30d75e4a529ab8122b9fc54`.

```bash
python3 pipelines/shell/ram_guard.py --limit-mb 8000 -- \
  python3 -m pytest test --collect-only -q \
    --continue-on-collection-errors
# -> 5573 tests collected, 54 errors in 23.77s
# -> peak RSS 368 MB
```

The uncommitted implementation draft removes 189 lines and adds 22, for 167
net lines removed across 11 files. It deletes the nine Phase-0 cost fields
from `CandidateFeatures`, removes cost estimation and normalization from the
Phase-0 pilot, ranks directly by `DeltaE0_upper * N0`, and removes the retired
cost source from route accounting. Threshold, record/operator caps, `alpha0`,
and shortlist-unit metadata remain. No compatibility shim was added.

**Receipt-identity lock:** before reaching the stop condition, the same
cache-disabled completed Hubbard--Holstein run was repeated on the draft. Both
runs stopped at `max_depth`, had ansatz depth 5, recorded 244 raw occurrences
and 202 unique primitives, and retained ledger fingerprint
`98f1aea9c0dc4dff91748b159da2b71ab1247cf614068972037dfdbcc509f839`.
The canonical serialized ledgers were byte-identical:

```bash
shasum -a 256 ledger-before.json ledger-after.json
# -> 6ac9108e809e7142797bf124f637247b81affef641cdf6c0f8504ad474875f40  (both)
cmp -s ledger-before.json ledger-after.json
# -> exit 0
```

Nine directly affected tests passed under the 8 GB RAM guard. Wider focused
runs also exposed two pre-existing, unrelated failures: a missing molecular
vibronic fixture and a stale semantic-route expected set. Neither surface was
edited.

**Stop condition:** the existing archive
`b9mr__b_depth_append_ra__strong_weak_u8__nph3__9675590__2.tar.gz` contains a
113,343,339-byte `current.json` with SHA-256
`b8c330923f083bb0b3221bd289717174cd734968d13054a5578941522dce788f`
and exactly 30 objects carrying `phase0_K0`. Its authenticated estimator-ledger
and verified-singleton sidecars were staged beside it.

`load_static_resume_source(current.json)` fails before strict replay or
checkpoint extraction. The untouched resume validator rejects
`settings.sr_route_profile_request` because the checkpoint stores
`paper_i_ra__global_singleton__phase0_measured_residual__phase123_adaptive__qiskit_signed__forced_admission_k50__semantic_closure_v1__insertion-append_only__cost-explicit_depth_qiskit_cost_v1`,
which is an RA algorithm identity rather than a recognized SR profile. The
same loader source is byte-identical at the clean baseline, and neither
`resume_scaffold.py` nor `sr_snake_route_profile.py` is in the draft diff.

Per Q17's explicit stop rule, no override or compatibility shim was attempted.
The after-collection run, ERROR-list diff, and implementation commit were not
performed. Paper-I-scoped `detect-changes` was attempted, but this isolated
worktree is not registered in the scoped index; it failed closed rather than
using the repo-wide index. The 11-file draft remains uncommitted for author
inspection.
