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

## 6n. The 26 geometry fields, classified by the author's Paper-I test

Screen run 2026-08-24. **This is a screen, not a verdict** — "in Paper I" means
the concept word appears in `Paper_I_author_revision.tex`. Ratify each against
source before deleting, per the standing rule.

Only **4 of 26 are actually values.** The rest are policy, index sets,
telemetry, or proxies.

| field | in Paper I | kind |
|---|---|---|
| `F_metric` | yes | **VALUE** |
| `hardware_cost_lambdas` | yes | **VALUE** |
| `phase2_raw_overlap_max` | yes | **VALUE** |
| `phase2_span_projection_z` | yes | **VALUE** |
| `phase0_cost_lambdas` | yes | VALUE — but deleted by Q14 |
| `metric_proxy` | **NO** | proxy/fallback → Q20 |
| `cheap_metric_proxy` | **NO** | proxy/fallback → Q20 |
| `phase1_lambda_f_proxy_applied` | **NO** | proxy/fallback |
| `phase2_lambda_f_proxy_applied` | **NO** | proxy/fallback |
| `phase2_cheap_curvature_proxy_policy` | **NO** | proxy policy |
| `phase2_missing_curvature_fallback_used` | **NO** | **fallback flag — rule 7** |
| `selector_geometry_mode` | **NO** | policy |
| `curvature_mode` | yes | policy → `request` |
| `phase2_curvature_policy` | yes | policy → `request` |
| `phase2_geometry_window_policy` | yes | policy → `request` |
| `phase3_geometry_window_policy` | yes | policy → `request` |
| `hardware_cost_lambda_source` | yes | policy → `request` |
| `phase0_cost_lambda_source` | yes | policy → deleted by Q14 |
| `phase2_geometry_window_indices` | yes | support (index set) |
| `phase3_geometry_refit_window_indices` | yes | support |
| `phase3_geometry_active_post_indices` | yes | support |
| `phase3_geometry_nested_refit_window` | yes | support |
| `phase3_geometry_window_size` | yes | support (derivable) |
| `phase3_geometry_window_accounting` | yes | support/telemetry |
| `phase2_joint_geometry_reuse` | yes | telemetry (an output) |
| `phase2_curvature_receipt` | yes | telemetry (an output) |

### What each class becomes

| class | count | destination |
|---|---|---|
| VALUE | 4 | `ResponseBlocks` entries — indexing expressions, not fields |
| policy | 7 | `request`, resolved once per run |
| support (index set) | 6 | the `candidates` argument to `restrict()` |
| telemetry | 2 | built outside `evaluate()` from the `Scored` list |
| proxy/fallback | 7 | **delete** — 6 fail the Paper-I test outright |

**Seven fields fail the Paper-I test.** Six are proxy or fallback machinery;
`phase2_missing_curvature_fallback_used` is a flag recording that a fallback
fired, which rule 7 says should not exist.

That is the shape of the 26-scalar deletion: almost none of it is data the method
needs. It is policy on the wrong object, index sets passed as fields, telemetry
mixed into inputs, and undefined proxies.

## 6o. The six remaining Paper-I test failures, ratified against source

6n screened by keyword. These are ratified. `metric_proxy` is already covered by
Q20 and 6m.

| field | ratified finding |
|---|---|
| `phase2_missing_curvature_fallback_used` | **Never set `True` anywhere.** Three `False` literals (`hh_continuation_scoring.py:3268, 3890, 6617`) and ~7 `.get(..., False)` reads forwarded into payloads (`adapt_pipeline.py:21044, 37863, 48701, 50732, 60232`). The flag can only ever be `False`. Dead. |
| `phase1_lambda_f_proxy_applied` | Pass-through telemetry. Read with `.get(..., False)` and forwarded into a payload at `adapt_pipeline.py:37853, 48691, 50722`. Carries no decision. |
| `phase2_lambda_f_proxy_applied` | Same pattern, `:37858, 48696, 50727`. |
| `cheap_metric_proxy` | Dataclass default `0.0` (`hh_continuation_types.py:112`), set to `1.0` once (`adapt_pipeline.py:65089`), otherwise forwarded (`:51058-51061`). Effectively a constant. |
| `selector_geometry_mode` | 56 references, default `"reduced"`, living in `historical_route_identity.py:62,130` and `selector_debug.py`. Route-identity and debug policy, not a value. |
| `phase2_cheap_curvature_proxy_policy` | **The important one — see below.** |

### The curvature proxy: rule 7 is already canonical policy

The proxy substitutes `lambda_F * F` for measured curvature — the Fubini--Study
**metric** standing in for the energy **Hessian**, two objects Paper I defines
separately (`hh_continuation_scoring.py:2656, 5518`).

Two profiles set the policy, and they disagree:

| profile | `phase2_curvature_policy` | `phase2_cheap_curvature_proxy_policy` |
|---|---|---|
| `CANONICAL_SR_SNAKE_V4_EXECUTION_SETTINGS` (:659) | `MEASURED_REQUIRED_FAIL_CLOSED_V1` | **`OFF`** |
| `HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS` (:614) | `LEGACY_OPTIONAL_V1` | `LEGACY_LAMBDA_F_RATIO_V1` |

V4 carries an explicit comment: *"Keep it explicitly disabled so neither a
generic parser default nor a source-locked command can silently activate the
guard."*

**So the canonical modern profile already implements rule 7 for curvature:
measured required, fail closed, proxy off.** The substitution survives only on
the historical phase-1/2 energy model, reached through the conventional v3.1
compatibility route.

That changes how the no-fallbacks rule should be argued in the refactor. It is
not a new constraint being imposed — it is an existing canonical decision being
generalized, with a precedent and a rationale already written in the source.

### Deletion summary

Six of the seven are dead flags, constants, pass-through telemetry, or
compatibility-route policy. None carries a live decision in a canonical run. The
seventh (`metric_proxy`) is Q20.

## 6p. Deleting the `lambda_F * F` substitution (Q21) — scope and consequence

The substitution uses the Fubini--Study **metric** as a surrogate for the energy
**Hessian**, two objects Paper I defines separately. It appears in two places:

| site | what it does |
|---|---|
| `hh_continuation_scoring.py:2653-2658` | Phase-I legacy model: `trust_region_drop(g, lambda_F * F_legacy, F_legacy, rho)` |
| `hh_continuation_scoring.py:5518` | Phase-II cheap curvature proxy: `lambda_F * legacy_metric` |

### It is in the NUMERATOR, not only the trust region

Author's question, and the answer is yes. `trust_region_drop(g_lcb, h_eff, F, rho)`
(`:2790`):

```python
alpha_max = rho / math.sqrt(F)                   # F sets the trust-region RADIUS
if h_eff_pos > 0.0:
    alpha_newton = g_lcb / h_eff_pos
    if alpha_newton <= alpha_max:
        return 0.5 * g_lcb**2 / h_eff_pos        # h_eff in the returned drop
return g_lcb * alpha - 0.5 * h_eff_pos * alpha**2  # and here
```

`h_eff` enters the **returned energy drop** in both branches, and the legacy call
passes `h_eff = lambda_F * F_legacy`. So the metric does double duty: once
legitimately as the radius via `rho/sqrt(F)`, and again, scaled by `lambda_F`, as
the curvature in the quadratic producing `DeltaE`.

The surviving branch proves the contrast — `rho * g / sqrt(F)` is exactly the
`h_eff -> 0` limit, with `F` used **only** for the radius. Deleting the
substitution therefore removes a metric-for-Hessian surrogate from the score
numerator while leaving the legitimate trust-region use of `F` untouched.

### No canonical evidence used it

| profile | `phase1_energy_model` |
|---|---|
| `HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS` (:614) | `LEGACY_LAMBDA_F_QUADRATIC_V1` — the substitution |
| `CANONICAL_SR_SNAKE_V4_EXECUTION_SETTINGS` (:659) | `FIRST_ORDER_FS_TRUST_V1` |
| `CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS` (:721) | `FIRST_ORDER_FS_TRUST_V1` |

The third is the **active family root with 20 descendants** (F3). Both canonical
families compute the surviving branch instead:

```python
rho * g_hw_lcb / math.sqrt(F_measured)
```

A first-order Fubini--Study trust step. No `lambda_F`, no Hessian surrogate.
Combined with V4 already setting `phase2_cheap_curvature_proxy_policy: OFF` and
`phase2_curvature_policy: MEASURED_REQUIRED_FAIL_CLOSED_V1` (6o), **the canonical
route has already abandoned this proxy on both phases.** Deleting it removes
compatibility-route behaviour only.

### Deletion scope

- the `LEGACY_LAMBDA_F_QUADRATIC_V1` branch of `phase1_trust_region_gain`, and
  with it `F_legacy`, `metric_floor`, and `cfg.lambda_F` where they exist only to
  serve it
- `PHASE2_CHEAP_CURVATURE_PROXY_POLICY_LEGACY_LAMBDA_F_RATIO_V1` and `:5518`
- the two `*_lambda_f_proxy_applied` telemetry flags (6o — pass-through, no
  decision)
- `HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS` loses 2 of its 4 keys

### Consequence to settle first

Those historical settings are spread into `SR_ROUTE_PROFILE_CONVENTIONAL_V3_1`.
Deleting the models they select **retires that compatibility route** — it would
have no Phase-I energy model to resolve. Under `CONTEXT.md` a **Compatibility
route** is "reachable only through an explicit versioned identity", so retiring
one is a deliberate act, not a side effect. See Q22.

## 6q. The 348 executor parameters, screened

Keyword screen over `_run_hardcoded_adapt_vqe`'s signature. **A screen, not a
verdict** — ratify before deleting.

| bucket | count |
|---|---|
| phase / score / method | 130 |
| unclassified | 65 |
| policy / mode / flag | 58 |
| cost / resource | 36 |
| plumbing / runtime | 33 |
| dead / legacy / historical | 10 |
| proxy / fallback | 9 |
| telemetry / observation | 7 |

Named legacy candidates: `historical_singleton_coordinate_solve_policy`,
`historical_singleton_coordinate_solve_scope`,
`historical_singleton_trust_region_update_policy`,
`phase3_shadow_legacy_geometry_mode`, `phase3_shadow_legacy_max_depth`, and the
five `phase2_compat_*_weight`.

Named proxy/fallback candidates: `finite_angle_fallback`,
`phase1_compile_cx_proxy_weight`, `phase1_compile_sq_proxy_weight`,
`phase2_compile_cx_proxy_weight`, `phase2_compile_sq_proxy_weight`,
`phase2_cheap_curvature_proxy_policy`, `deferred_gram_fallback_ridge`,
`phase2_remaining_evaluations_proxy_mode`, `phase3_enable_rescue`.

### Ratified: the five `phase2_compat_*_weight`

Not dead, and not core either — they are **batch-extension parameters on the core
signature**.

- Defaults are **non-zero** (`0.4`, `0.2`) at `adapt_pipeline.py:15006` and
  `cli_config.py:2333, 2345`.
- **No profile sets any of them** — 0 occurrences in `sr_snake_route_profile.py`.
  They depend entirely on parser defaults.
- They are live in the scorer: `_compatibility_penalty_components:6935-7002`
  computes a weighted sum of support overlap, non-commutation, cross curvature,
  schedule and measurement mismatch.
- The consumer is batch admission — the result lands in
  `phase2_last_batch_penalty_total` (`adapt_pipeline.py:38279, 51309, 53995`).

So they only affect a run when **batching is enabled**, and
`phase3_enable_batching` is `False` canonically.

**Two conclusions.** First, under the design target batch is an *optional
extension defined after* the algorithm, so its five weights do not belong on the
core executor signature at all — they move to `extensions.py` with the batch
policy. Second, they are a live instance of the F3 drift mechanism in its purest
form: non-zero scoring weights that no profile pins, so their values come from
argparse and are invisible to profile review. If batching is ever enabled, the
score depends on numbers no profile records.

## 6r. Batch admission: two nested defaults (Q24)

> **REFINED by Q28.** The author's preference is deletion of the legacy path, or
> a true extrication that makes it unreachable from the working code. A
> `legacy_batch` mode that merely defaults off still leaves it reachable, so it
> is the weaker option.


> **CORRECTED by 6u.** Q24 called the additivity-defect gate "canonical Paper-I
> batch". Paper I's criterion is different: feasibility is *existence of the
> supported joint Phase-III solve*, and admission is
> `argmax_B dE_3(B)/K_3(B)`. `batch_additivity_tol` is a third criterion, not the
> manuscript's. The nested-defaults structure of Q24 still stands.


```
batch                     default OFF
  └─ if batch = true:
       paper_i_batch      DEFAULT     Schur additivity defect
       legacy_batch       default OFF five-weight heuristic
```

**Canonical (Paper I).** A block **B** is feasible when the joint drop is nearly
additive:

    defect = 1 - dE_joint / sum_i dE_i          (hh_continuation_scoring.py:7302)
    feasible iff defect <= batch_additivity_tol  (:7309)

This follows from the Schur-reduced quadratic, as the manuscript states.

**Legacy.** `_compatibility_penalty_components:6935-7002`, a weighted sum of
support overlap, non-commutation, cross curvature, schedule and measurement
mismatch, with the five `phase2_compat_*_weight` constants.

Both live under batch in `extensions.py`; neither belongs on the core executor
signature. Reaching the legacy path requires two explicit opt-ins, so no parser
default can select it.

**The legacy weights have no defaults (Q25).** Enabling `legacy_batch` asks for
the five weights. This is the **conditional policy interview** already defined in
`CONTEXT.md` — silent while the policy is disabled, revealing only that policy's
required choices when enabled. Consequences: nothing can drift, because there is
no default to drift from; the weights appear in every legacy-batch receipt by
construction; and the five `phase2_compat_*_weight` parameters leave the 348-name
signature entirely rather than moving to a new default.

## 6s. Extension parameters on the core signature (Q26)

**57 of the 348 executor parameters belong to the three optional extensions** —
16% of the signature, for behaviour that is off by default.

| extension | params | notable defaults |
|---|---|---|
| batch | 11 | `phase2_enable_batching=True`, `batch_target_size=2`, `batch_additivity_tol=0.25` |
| prune | 35 | `phase1_prune_enabled=True`, `prune_fraction=0.25`, `retained_gain_ratio=0.5`, 6 tolerance coefficients |
| beam | 11 | `tie_beam_max_branches=1`, `beam_live_branches=1`, `beam_lambda=0.0` |

Under Q24/Q26 these move to `extensions.py`, and their constants come from the
conditional policy interview rather than parameter defaults.

**Correction — the `=True` signature defaults are not the operative values.**
Verified rather than left as a flag:

- argparse sets `phase2_enable_batching=None` and `phase3_enable_batching=None`
  (`cli_config.py:2203`), so the signature default only applies when nothing
  supplies a value.
- Inside the executor the flag is **overwritten** at `adapt_pipeline.py:15565`,
  `phase2_enable_batching = bool(selector_config.batch_size_cap > 1)`, whenever
  the funnel mode is in `ROUTE_A_FUNNEL_CHILD_12_MODES`.
- `phase1_prune_enabled: False` **is** pinned by the active family root
  (`sr_snake_route_profile.py:760`, `CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1`).
- `phase2/3_enable_batching: False` are pinned only in
  `CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS` (`:515-516`) — the 116-key root the
  active family does **not** inherit (F3). `batch_size_cap` is pinned by no
  profile at all.

So batching is not silently on, but its effective value comes from a funnel
cardinality (`batch_size_cap > 1`) rather than from any profile.

## 6t. Extension surface to relocate (Q27)

| surface | batch | prune | beam | total | of |
|---|---|---|---|---|---|
| executor parameters | 11 | 35 | 11 | **57** | 348 |
| CLI flags | 20 | 36 | 11 | **67** | 409 |
| lines mentioning, `adapt_pipeline.py` | 1105 | 2293 | 814 | **~4212** | 72063 |
| lines mentioning, `hh_continuation_scoring.py` | 611 | 4 | 1 | ~616 | 19420 |

Mentions are not all movable lines — extension logic is interleaved with the
controller rather than blocked — so treat the line figures as the surface to
work through, not the deletion target.

**Destination.** `extensions.py`, per the design target: batch, prune and beam
are optional extensions defined *after* the algorithm, absent from the default
path. `maintain(state, extensions)` iterates an empty list unless one is enabled.

**Consequences that make this more than a move.** Their choices come from the
conditional policy interview (Q25/Q26), so the 57 parameters and 67 flags do not
reappear as defaults elsewhere — they cease to exist as a default surface. That
is what makes this a deletion under rule 6i rather than a relocation like the
`ResponseAccounting` extraction.

`prune` is the largest by far — 35 parameters, 36 flags, 2293 mentions — and the
active family root already pins `phase1_prune_enabled: False`
(`sr_snake_route_profile.py:760`), so canonical evidence does not exercise it.

## 6u. What Paper I actually specifies for batching

Settled from the manuscript, `Paper_I_author_revision.tex:1051-1086`.

Let `W_k` be a score-ordered window of the Phase-II population, and
`B_k(W_k)` its **bounded feasible batches**: the subsets of records carrying
**distinct generators**, **through the batch-size cap**, for which the
**supported joint Phase-III solve exists**.

For `B = {r_1..r_s}` the coordinate and gradient vectors preserve the
candidate--active ordering of the Phase-III model:

    z_B      = (alpha_B, delta_theta)^T
    g_joint  = (g_alpha_B, g_theta)^T

The corresponding extensions of `H` and `G` give the joint predicted decrease
`dE_3(B)`, and `K_3(B)` combines the member records' resource contributions.
Admission is

    B_k* in argmax_{B in B_k(W_k)}  dE_3(B) / K_3(B)        (eq:batch_argmax)

"The bounded route enumerates every feasible subset, while a score-ordered greedy
construction provides an optional approximation when enumeration is impractical."

For the `L=3` controls, batch admission "enumerates subsets of at most three"
(`:1499`), and the manuscript's own pseudocode contains
"**if** batching is enabled **then**" (`:1654`).

### Consequences

1. **The batch-size cap is part of the method.** It bounds `B_k`. An earlier open
   question asked whether `|B| > 1` should be an outcome rather than a
   precondition — the manuscript settles it: the cap is a precondition, and
   `batch_size_cap` corresponds to it directly.
2. **`enable_batching` corresponds to the manuscript** — the pseudocode branches
   on it.
3. **Feasibility is not an additivity tolerance.** Paper I's test is whether the
   supported joint Phase-III solve *exists*. The code's
   `additivity_defect = 1 - dE_joint/sum(dE_i)` gated by `batch_additivity_tol`
   (`hh_continuation_scoring.py:7302, 7309`) is a separate criterion with no
   counterpart in the manuscript.
4. **Admission is the same energy-per-cost ranking as singleton**, extended to
   blocks: `dE_3(B)/K_3(B)`. This is the batch case of `evaluate`'s
   `descent / cost`, not a different rule.
5. Both the **bounded enumeration** and the **greedy approximation** are in the
   method; greedy is explicitly optional.

So under Q24's structure, canonical batch is `argmax dE_3(B)/K_3(B)` over
cap-bounded subsets with an existing joint solve. The additivity tolerance and
the five-weight compatibility penalty are both outside the manuscript.

## 6v. The published batching route does not use the extra gates

Question: were `batch_additivity_tol` and the five-weight compatibility penalty
the route that produced Paper I's batching result? **No.**

**The arm.** `Paper_I_author_revision.tex:1406` records the source label:

```
l3_batching_h15   CombinatorialBatchAdmission, maximum_size=3, search_window_size=6
```

matching the text at `:1499`, "batch admission enumerates subsets of at most
three", and `maximum_size=3` = the batch-size cap bounding `B_k`,
`search_window_size=6` = the score-ordered window `W_k`.

**The route.**

```
CombinatorialBatchAdmission            sr_snake/contracts.py:175
  -> run_combinatorial_batch_proposals adapt_pipeline.py:68311-68324
       "Enumerate generator-distinct subsets of one fixed ranked prefix."
  -> _run_batch_proposals              adapt_pipeline.py:68062-68294 (233 lines)
```

`_run_batch_proposals` contains **none** of `additivity_defect`,
`batch_additivity_tol`, `compatibility_penalty`, or `compat_*`. Neither does any
module under `sr_snake/`.

Its docstring restates the manuscript's definition — generator-distinct subsets
of a score-ordered prefix — which is exactly `B_k(W_k)`.

**Conclusion.** The additivity tolerance
(`hh_continuation_scoring.py:7302, 7309`) and the five-weight compatibility
penalty (`:6935-7002`) sit on a different Phase-II batch path that the published
`l3_batching_h15` result never crossed. They are not the manuscript's criterion
and were not the route used for the published evidence.

Under Q24 they belong to the legacy `batch_mode`, default off, reached only by
opting in twice — and their five weights come from the conditional policy
interview (Q25) rather than from defaults.

## 6w. Standing preference: delete, do not demote (Q28)

When a path is found to be superseded — not the manuscript's, and not the route
that produced the evidence — the order of preference is:

1. **Delete it.**
2. Failing that, **extricate it completely** so it cannot be reached from the
   working part of the tree.
3. An off-by-default flag is **not** sufficient. It leaves the code present,
   reachable, and one setting away from firing.

Rationale: every defect catalogued in this file — the settings drift, the 1,878
guards, the 1,622 fallbacks, the five unpinned compat weights — is something that
was left reachable "just in case". Reachable dead paths are what the guards
exist to defend against.

Applies to the Phase-II batch path (`_compatibility_penalty_components`,
`additivity_defect`/`batch_additivity_tol`) confirmed off the published route in
6v, and to anything later found in the same position.

## 6y. Backend fallbacks — there are two, and the second is worse (Q30)

**1. `allow_aer_fallback` — delete (Q30).** 28 occurrences. Hard-coded `True` at
`oracle_lifecycle.py:178, 402` and `ra_adapt/pure_hubbard_noise_page12.py:171`,
and configurable in `hardcoded/hh_staged_noise_workflow.py`. It sits on the
**noise-oracle execution** path, not the compile path.

**2. The preferred-fakes chain — DELETE (Q31).**
`hh_backend_compile_oracle.py`:

```python
_DEFAULT_PREFERRED_FAKES = ("FakeMarrakesh", "FakeNighthawk", "FakeFez")   # :34
allow_preferred_fallback: bool = True                                       # BackendCompileConfig
```

This one is on the **compiled-cost** path. Paper I names the cost target
explicitly — `Paper_I_author_revision.tex:1400`:

```
% COST  FakeMarrakesh, optimization level 1, seed_transpiler 7, full
%       circuit including reference-state preparation, no initial layout,
%       weights (w_2q, w_D2q, w_D) = (0.30, 0.30, 0.25), 1q and size 0
```

`N_2q`, `D_2q` and `D_c` are properties of a specific coupling graph. A silent
fall through to `FakeNighthawk` or `FakeFez` would report compiled resources for
a device the manuscript does not name, in the paper's headline cost axis. That is
the qiskit-to-proxy pattern (rule 7) applied to the backend itself.

Related, already recorded in section 7: `hh_backend_compile_oracle.py:350` is a
bare `except Exception: pass` around backend loading.

## 6x. The fallback-reporting surface (Q29)

**222 distinct fallback-related keys** across `pipelines/`. They are not one
thing; sort before deleting:

| kind | examples | disposition |
|---|---|---|
| **reporting** — records that a fallback fired | `fallback_used` (36), `fallback_reason` (16), `legacy_fallback_used` (12), `fallback_detail` (9), `phase2_missing_curvature_fallback_used` (10), `phase1/2_lambda_f_proxy_applied` (18) | **delete with the fallback** (Q29) |
| **policy** — permits a fallback | `allow_aer_fallback` (28), `fallback_policy` (10), `fallback_mode` (15), `actual_fallback_mode` (11), `fallback_family` (25) | delete the permission and the path (Q28) |
| **mechanism** — the fallback itself | `fallback_to_incumbent` (25), `deferred_gram_fallback_ridge` (9), `qngd_fallback_optimizer` (8), `phase3_joint_rescue_v1` (8), `adapt_finite_angle_fallback` (10) | delete per rule 7, after checking whether it ever fired |

`phase2_missing_curvature_fallback_used` is the clearest case: 10 occurrences,
**never set `True` anywhere**, so it reports an event that cannot occur.

**Rule.** A field whose only job is to record that a fallback fired has no
purpose once the fallback is gone. Keeping it as always-`False` is not evidence
that nothing went wrong; it is a receipt field describing an impossible event,
and it keeps the concept alive in every payload and schema that carries it.

**Caveat.** "Fallback" in a name does not prove a fallback. Some of the 222 are
legitimate — a *nominated* alternative that is then tested, rather than a silent
substitution. Ratify each against source, as with everything else here.


### What deleting the chain also removes (Q31)

With a single fixed target, the backend-selection machinery around it loses its
subject:

| symbol | why it goes |
|---|---|
| `_DEFAULT_PREFERRED_FAKES` (`:34`) | a three-element chain becomes one constant |
| `allow_preferred_fallback` | nothing to fall back to |
| `requested_backend_shortlist` | no shortlist to reduce |
| `shortlist_reduction_mode="best_backend_in_shortlist_v1"` | selects among a shortlist of one |
| `except Exception: pass` (`:350`) | a swallowed backend-load failure must now stop the run |

`requested_backend_name` stays — it names the target, and Paper I's COST block
fixes it to `FakeMarrakesh` alongside `optimization_level=1` and
`seed_transpiler=7`. Those three are the compilation contract and are the thing
that makes `N_2q`, `D_2q`, `D_c` reproducible.

## 6z. Cost default inversion (Q32)

`BackendCompileConfig.mode` currently defaults to `"proxy"`
(`hh_backend_compile_oracle.py:84`), so a run that does not ask produces the
non-headline quantity.

**This is a default change, not an evidence change.** Canonical runs already
recorded `hardware_cost_source = "backend_transpile_v1"` and
`compile_cost_source = "backend_transpile_v1"` (verified in a completed Bundle-9
checkpoint), so they were explicitly requesting qiskit compiled cost. Inverting
the default matches what the evidence already did; it removes the possibility of
a run silently producing proxy numbers because nobody set the flag.

Combined with Q31, the compile path becomes: one target (`FakeMarrakesh`), one
optimization level, one transpiler seed, and compiled cost by default — the
contract Paper I's COST block states.

The proxy remains a peer implementation behind the single `CostTerm` interface
(the author's design target, section 3), selected explicitly. It is never
substituted for a failed qiskit cost (rule 7).

## 6aa. Collapsing the per-phase cost encoding (Q33)

**Three scope constants exist only to encode which phase uses which cost source**
(`hh_backend_compile_oracle.py:45-53`):

```
phase_i_phase_ii_marrakesh_graph_span_phase_iii_qiskit_transpile_v1
phase_i_proxy_phase_ii_phase_iii_qiskit_transpile_v1
phase0_proxy_or_off_phase_i_phase_ii_phase_iii_qiskit_transpile_v1
```

Under a single run-level choice these collapse to `request.cost`, one of
`qiskit` or `proxy`, applied across Phases I-III.

**23 per-phase cost/compile parameters** sit on the executor signature. Five are
duplicated verbatim between Phase I and Phase II and become one each:

| pair | becomes |
|---|---|
| `phase1/phase2_compile_cx_proxy_weight` | `compile_cx_proxy_weight` |
| `phase1/phase2_compile_sq_proxy_weight` | `compile_sq_proxy_weight` |
| `phase1/phase2_compile_rotation_step_weight` | `compile_rotation_step_weight` |
| `phase1/phase2_compile_position_shift_weight` | `compile_position_shift_weight` |
| `phase1/phase2_compile_refit_active_weight` | `compile_refit_active_weight` |
| `phase1/phase2_opt_dim_cost_scale` | `opt_dim_cost_scale` |
| `phase1/phase2_family_repeat_cost_scale` | `family_repeat_cost_scale` |

Seven duplicated pairs, so 14 parameters become 7. Combined with the three scope
constants and the Phase-0 cost deletion (Q14), the cost surface reduces to:

- `request.cost` — `qiskit` or `proxy`
- the compilation contract — `FakeMarrakesh`, optimization level 1,
  `seed_transpiler` 7 (Q31)
- one shared normalization (`_hardware_cost_denominator_payload`, already exists)
- the proxy weight set, once rather than per phase

## 6ab. Per-phase parameter duplication — 60 into 28, but not all of it

**28 parameter stems repeat across phases; 60 parameters carry them.** Collapsing
every one would remove 32 — but some per-phase splits are the method, not
duplication.

**The method (keep per-phase):**

| stem | why |
|---|---|
| `shortlist_size` | the funnel narrows: `macro_phase1_cap` -> `macro_phase2_cap` -> `child_phase3_cap` (`adapt_pipeline.py:15559-15562`). Distinct caps per phase are what staged screening *is*. |

**Duplication (collapse):**

| stem | why |
|---|---|
| `lambda_F`, `lambda_2q`, `lambda_d`, `lambda_1q`, `lambda_theta`, `lambda_shot` | Paper I's COST block states one weight set for the run: `(w_2q, w_D2q, w_D) = (0.30, 0.30, 0.25)`. Not per-phase. |
| the 7 `compile_*` / `*_cost_scale` pairs | already collapsed by Q33 |
| `backend_transpile_seed` | run-level by Q34 |
| `maturity_cap_min` / `_max` / `maturity_shot_cap` | disposition not yet determined |

**Test to apply per stem:** does Paper I specify a *different* value at each
phase, or one value for the run? The funnel caps differ by construction; the cost
weights do not.

## 6ac. `lambda_F` is not a cost weight — it is the substitution's coefficient

Q35 makes the lambda weights run-level. **`lambda_F` is not among them.** All 7
of its references in `hh_continuation_scoring.py` are the metric-as-curvature
substitution Q21 deletes:

| site | use |
|---|---|
| `:113`, `:175` | dataclass defaults, `1.0` |
| `:2656` | Phase-I legacy: `trust_region_drop(g, lambda_F * F_legacy, ...)` |
| `:5518` | Phase-II cheap curvature proxy: `lambda_F * legacy_metric` |
| `:3416-3418` | `phase3_cheap_ratio_v1`: `g^2 / (2 * lambda_F_eff * cheap_metric_proxy)` — the same Newton step `0.5*g^2/h` with `h = lambda_F * metric_proxy`, and it uses `cheap_metric_proxy`, deleted by Q20 |
| `:19026` | the label string `"lambda_F_metric_proxy_only"` |

**There is no use of `lambda_F` that is not a substitution of the Fubini--Study
metric for the energy Hessian.** Paper I's cost weights are
`(w_2q, w_D2q, w_D) = (0.30, 0.30, 0.25)`, which map to `lambda_2q`, `lambda_d`
and their siblings. `lambda_F` scales a *metric*, not a resource.

So `lambda_F` is deleted entirely by Q21 rather than collapsed to run-level by
Q35. **Codex: do not preserve it as a run-level parameter.**

`phase3_cheap_ratio_v1` (`:3378-3430`) is a third site of the substitution,
beyond the two named in 6p, and goes with them. **Verified dead**: it has no
callers anywhere in `pipelines/` — the only reference outside its own definition
is the version string in its own return payload (`:3400`). It never ranked a
Phase-III candidate.

## 6ad. The maturity controller (Q36) — delete

> **SCOPE CORRECTION — read 6ae before deleting anything.** Verified against a
> locked bundle's sidecar: the 12 `maturity_*` **parameters** are inert, but the
> snapshot *named* `phase123_controller_maturity_v2` is the **phase controller**
> and is live. Deleting "the maturity controller" by that name would take the
> funnel caps with it.


An adaptive per-candidate **shot-budget controller**: as a candidate accumulates
measurements it "matures", and the caps bound its shots at each phase.

| evidence | value |
|---|---|
| occurrences of "maturity" in `Paper_I_author_revision.tex` | **0** |
| parameters on the executor signature | **12** |
| read sites in `hh_continuation_scoring.py` | **0** |

Every cap defaults off: `phase{1,2,3}_maturity_cap_min/max = None`,
`phase{1,2,3}_maturity_shot_cap = 0`, `phase_maturity_shot_min/max = 1`.
`phase1_prune_maturity_threshold = 0.5` goes with prune (Q27).

**Resume coupling — the one complication.** The controller serializes state into
`current.json` as `phase123_controller_maturity_v2`
(`resume_scaffold.py:291-292, 3292, 3354-3355, 5770`), and resume *validates the
snapshot version*, raising if it is unsupported. Deleting the controller changes
the checkpoint schema, so existing Bundle checkpoints carry a snapshot the new
code will not write.

**CORRECTION.** I recorded Q37 on the premise that "completed Bundle runs do not
need restarting". **That is wrong.** The author: *a finished run can be continued
to longer iteration horizons.* Continuation of a locked bundle is a live use
case, so old checkpoints must stay readable.

What that code actually does (`resume_scaffold.py:3288-3296`): legacy
`current.json` did **not** serialize the maturity-controller state, so
continuation loads it from a **compact signed-prefix sidecar**, "source-result
authenticated by the locked bundle", which carries the unique final-round
controller snapshot.

Its validation at `:3348-3362` is **two things entangled**:

| check | serves |
|---|---|
| snapshot is typed; `snapshot_version == "phase123_controller_maturity_v2"` | the maturity controller — goes with it |
| `step_index`, `depth_local`, `depth_left` identify the unique pre-round state | **continuation correctness generally** — proves you resume from the right round |

The second is not maturity machinery and is load-bearing for extending a bundle's
horizon. Deleting the controller must not take it.

**Open:** whether the maturity state carries anything that affects a continued
trajectory. Every cap defaults off (`None`/`0`), so an inert controller's snapshot
should carry no trajectory-relevant information — but that is inference from
defaults, not from a locked bundle's sidecar.

## 6ae. What the "maturity" snapshot actually holds — verified from a bundle

Read from a locked Bundle-9 sidecar,
`run/checkpoints/current.verified_singleton_resume.*.json` (4,429 bytes). Its
`controller_snapshot` carries **38 fields**. Exactly one is maturity:

```
phase_shots_maturity_floor = {phase1: 1, phase2: 1, phase3: 1}   # inert
```

matching the `phase_maturity_shot_min/max = 1` defaults. **The other 37 are the
phase controller**, and several are load-bearing:

| field | value in this bundle | what it is |
|---|---|---|
| `phase_caps` | `{phase1: 24, phase2: 12, phase3: 12}` | **the funnel shortlist caps** — the method (6ab) |
| `phase_caps_scheduled` | same | the scheduled caps |
| `H_t`, `gamma_t`, `m_t`, `s_t`, `rho_t` | non-trivial | controller scheduling state |
| `useful_horizon`, `frontier_ratio`, `runway_ratio`, `n_rem_*` | non-trivial | horizon estimation |
| `step_index`, `depth_local`, `depth_left` | `14`, `14`, `36` | round identity, needed for continuation |
| `phase_shots*`, `phase_signal`, `phase_thresholds` | all `1` / `0` / floors | shot machinery, inert here |
| `phase_null_reasons` | `phase_live_retired_non_authoritative` | ties to `phase_live_hysteresis_enabled` |

**Conclusion.** "The maturity controller" is a misnomer taken from the snapshot
version string. The 12 `maturity_*` parameters are dead and Q36 deletes them
correctly. The **controller** is not dead — it schedules the funnel caps and the
horizon, and a locked bundle's continuation depends on restoring its state.

**Codex: delete the 12 `maturity_*` parameters and the inert
`phase_shots_maturity_floor`. Do not delete the controller or its snapshot.** The
version string may need renaming since "maturity" no longer describes it, but
renaming it breaks reading existing sidecars — leave the string alone.

This is why the author's rule applies: *always check things you do not know.*
The name said inert knob; the bundle said phase controller.

## 6af. Snapshot version rename (Q38) — and what it costs

The snapshot is versioned `phase123_controller_maturity_v2`, but 6ae established
that 37 of its 38 fields are phase-controller state and only
`phase_shots_maturity_floor` is maturity. The name is wrong.

**Decision: rename, drop the old string.** An alias that reads the old version
while writing the new was offered and declined.

**Consequence, accepted by the author.** `resume_scaffold.py:3354-3355` validates
the version and raises on anything it does not recognise. After the rename,
**existing locked-bundle sidecars cannot be continued to longer horizons** — the
Bundle-3 and Bundle-9 archives carry the old string.

This is the one decision in this file that costs a capability rather than
removing dead weight. It is recorded here so nobody later reads the resulting
failure as a bug. Continuation of *new* runs is unaffected.

**Codex:** rename the constant and the written value together, and delete the
maturity fields (Q36) in the same change so the new version string describes what
the snapshot actually holds.

## 6ag. PAOP (Q39) — verified unused by both producer lanes

> **6ag IS WRONG — DO NOT DELETE PAOP. See 6al.** I checked
> `primitive_pools._build_full_meta_pool` and found no PAOP. **The Hubbard--Holstein
> path uses a different builder**, `hh_pool_presets._build_hh_full_meta_pool`,
> which lists `paop_full`, `paop_lf_full` and `_HH_FULL_META_EXTRA_PAOP_KEYS` as
> required components. Codex materialized the canonical L=2 HH parent pool: **89
> records, 40 of them PAOP**.


An alternative operator-pool family from `src.quantum.operator_pools.hh_paop`,
reached through its own pool keys (`uccsd_otimes_paop_lf_std`,
`uccsd_otimes_paop_lf2_std`, `paop_bond_disp_std`, and siblings —
`primitive_pools.py:84-99`).

| check | result |
|---|---|
| occurrences in `Paper_I_author_revision.tex` | **0** |
| selected by any canonical profile | **no** — profiles select `full_meta` / `full_meta_derivative_resolved_v2` |
| inside `full_meta` construction | **no** — `_build_full_meta_pool` (`primitive_pools.py:895-1214`, 320 lines) contains no `paop`, no `make_paop_pool`, no motif families. It assembles CSE, family-HVA, full-Hamiltonian(+flow), Hamiltonian-blocks, Hamiltonian-quadratures, UCCSD and molecular-UCCSD pools |
| used by Paper IV | **no** — the Paper-IV H2O run receipt records `adapt_pool = "full_meta"`; its four `paop_*` entries are echoed defaults (`normalization "none"`, `prune_eps 0.0`, `r 1`, `split_paulis false`) |
| references in code | 812 |
| executor parameters | 4 — `paop_r`, `paop_split_paulis`, `paop_prune_eps`, `paop_normalization` |

The 2026-08-17 inventory already flagged `paop_r` and `paop_split_paulis` as
`unknown_needs_evidence` (`adapt_pipeline_inventory_20260817.md:353-354`). This
is that evidence.

**One contact point with the canonical path.**
`_build_selected_logical_pool_match_report` (`pool_resolution.py:321-505`) takes
`paop_split_paulis` and uses it only to choose a report label,
`"deliberate_split"` vs `"preserve"` (`:423`). It does not affect pool
construction, and goes with the rest.

**Note for Codex:** `pipelines/scaffold/hh_vqe_from_adapt_family.py` carries 131
`paop` references and `builders/hh_pool_presets.py` 223 — the largest
concentrations. Check whether either has non-PAOP responsibilities before
removing wholesale.

## 6ah. `paop_prune_eps` — reconciling Codex's Q27 call with Q39

Codex, working Q27, kept `paop_prune_eps` in core rather than moving it to
`extensions.py`, on the grounds that it is **pool/generator truncation, not
accepted-coordinate pruning**. That call is right, and it does not conflict with
Q39.

The two decisions answer different questions:

| | question | answer |
|---|---|---|
| Q27 (Codex) | does it belong in `extensions.py` with the prune extension? | **No** — it truncates pool polynomial coefficients at construction, nothing to do with deleting accepted generators |
| Q39 (author) | does the PAOP family get deleted? | **Yes** — and `paop_prune_eps` is one of its 4 parameters |

Verified so the later deletion is safe:

- The **mechanism** is generic: `_clean_real_pool_polynomial(..., prune_eps)`
  drops terms with `abs(coeff) <= prune_eps` (`primitive_pools.py:1558-1583`).
  It stays.
- The **parameter** `paop_prune_eps` reaches it only through
  `hh_pool_presets.py` (`:223, 361, 376, 392, 410, 461, 485`) — the PAOP presets.
- `_build_full_meta_pool` calls `_clean_real_pool_polynomial` but **passes no
  `prune_eps`**, so the canonical pool uses the `0.0` default and truncates
  nothing.

**So deleting `paop_prune_eps` under Q39 does not change `full_meta`.** Keep
`_clean_real_pool_polynomial` and its `prune_eps` argument; delete only the
PAOP-level parameter that feeds it.

## 6ai. The policy/mode/flag block — surveyed

Executor signature is now **313 parameters** (Codex's prune cut removed 34).
55 are policy/mode/flag: 30 `_mode`, 17 `_policy`, 3 enable, 2 `_source`,
1 `allow`, 2 other.

### The one that matters: the estimator ledger defaults OFF

```python
adapt_estimator_call_ledger_enabled: bool = False          # :14753
...
EstimatorCallLedger() if bool(adapt_estimator_call_ledger_enabled) else None   # :19496
```

When false there is **no ledger object at all** — no estimator accounting. The
estimator count is one of the three axes the author named for "at least as good"
(Q6).

**Canonical is safe**: `_build_canonical_sr_snake_runtime_kwargs`
(`:70853-71002`) hardcodes `"adapt_estimator_call_ledger_enabled": True`
(`:70954`), which is why Bundle-9 archives carry 20,949 unique primitives. No
profile sets it; the canonical runtime builder does.

But off the canonical path — a direct executor call, or the CLI without the flag
— a run produces no estimator accounting and nothing says so. **Same shape as
Q32**, where the cost defaulted to proxy and produced the non-headline quantity
unless asked. See Q40.

### The rest

| group | count | note |
|---|---|---|
| `_policy` + `_mode` | 47 | |
| ...defaulting `off` / `none` / `disabled` | **16** | features present but inert by default |
| ...legacy-valued or legacy-named | 3 | `historical_singleton_coordinate_solve_policy = archival_reduced_scalar_v1`, `historical_singleton_trust_region_update_policy`, `phase3_shadow_legacy_geometry_mode = off` |

The 16 default-off modes are the Q28 population: features reachable by flag but
inert in every canonical run — `adapt_child_pool_expansion_mode`,
`adapt_schur_warm_start_mode`, `phase2_gram_novelty_policy`,
`phase3_gram_novelty_policy`, `phase3_runtime_split_mode`,
`phase3_shadow_damping_policy`, `adapt_noise_floor_stop_policy`, and others.
Under Q28 the preference is deletion rather than demotion; each needs the usual
source ratification first.

The 3 legacy-named ones join the 10 already listed in 6q.

## 6aj. Deferred-Gram audit (Q41) — three questions, answered from evidence

> **Q41 IS WRONG — DO NOT ACT ON IT. See 6ak.** The claim "the rescue never ran"
> came from grepping checkpoints for `deferred_gram_fallback_enabled`, which is a
> **derived** value that is never written to a checkpoint. It is computed from the
> gram-novelty policy keys, and **both canonical V4 and the active family root set
> them to `fallback_only_v1`**, which enables it. The rescue was enabled in
> canonical runs.


The author's test for removing a method not in Paper I: **is it in the paper, was
it on the route that produced the results, and did it fire?**

### 1. In Paper I? No.

Zero occurrences in `Paper_I_author_revision.tex` of *deferred*, *metric
collapse*, *novelty*, *geometry expansion*, or *collapse*. (Also confirms the
retired novelty multipliers were never manuscript material.)

### 2. On the route? The rescue, no. The detector, yes.

`deferred_gram_fallback_enabled` is **absent from every checkpoint** across all
58 archives and defaults `False`. The geometry-expansion rescue never ran.

The collapse branch is inside `phase3_canonical_score_components:3933` — the
canonical Phase-III scorer — and returns early with
`block_reason: "reduced_metric_collapse"` when
`phase3_window_relaxation_mode` is not disabled. Profiles set `"reduced"`
(`sr_snake_route_profile.py:558`) and the runs record `"reduced"`, so the branch
was live.

### 3. Did it fire? Yes — 25 distinct events.

Audited all 58 local bundle archives (`b3` 18, `b6` 18, `b7` 3, `b8` 6, `b9` 13).
Counts are `history` + `history_tail` double-records, so halve them:

| bundle | archives | collapse records | distinct events | payloads emitted |
|---|---|---|---|---|
| b3 | 18 | 4 | 2 | 2250 |
| b6 | 18 | 2 | 1 | 2250 |
| b7 | 3 | 4 | 2 | 300 |
| b8 | 6 | 4 | 2 | 370 |
| b9 | 13 | 36 | 18 | 1114 |
| **total** | **58** | **50** | **25** | **6284** |

### Every occurrence is at machine-level error

B3 `always_insertion__weak_weak`, round 42 of 50: `F_red = 2.17e-11`,
`dE = -1.32e-10`, `max_grad = 1.85e-06`, rank 34 of 43. Candidate admitted.

B9 `costexp_plateau_position__weak_weak` (`9679133__10`), rounds 34-49 of 50 —
14 consecutive events. From round 39 the **same generator**
`guarded_singleton::eeyzeeee` is admitted at the same position eleven times:

```
round 39  dE=-5.551e-16   max_grad=1.010e-07
round 42  dE= 0.000e+00   max_grad=1.008e-07
round 49  dE= 0.000e+00   max_grad=1.009e-07
```

**Author's judgement: this is machine-level error, so delete.** The branch only
ever acted where `dE` is between `0.0` and `-6e-11`. Removing it may alter
accepted trajectories at that magnitude, and that is accepted.

### Unresolved, recorded rather than guessed

The round-42 candidate carried `append_exact_metric_collapse_v1` and was still
**selected**, and no `block_reason` string appears anywhere in the checkpoint.
Whether the blocked payload still ranks through `**base`, or the mode is stamped
after selection, was not traced. It does not change the decision.

### Separate observation, not part of Q41

Rounds 39-49 of that B9 cell re-admit an identical null generator eleven times
with the gradient frozen. That is ~22% of the run's controller budget and its
estimator work, inside the reported `k` range, and no mechanism stops it. The
collapse detector was the only record of it.

## 6ak. RETRACTION of Q41 — the deferred-Gram rescue was enabled

**My error.** 6aj concluded the rescue "never ran" because
`deferred_gram_fallback_enabled` is absent from all 58 checkpoints. It is absent
because it is **never written** — it is derived at runtime
(`adapt_pipeline.py:20486`):

```python
deferred_gram_fallback_enabled = bool(
    phase2_gram_novelty_policy_key == GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1
    and phase3_gram_novelty_policy_key == GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1
)
```

Both canonical profiles set both keys to `fallback_only_v1`:

| profile | `phase2_gram_novelty_policy` | `phase3_gram_novelty_policy` |
|---|---|---|
| `CANONICAL_SR_SNAKE_V4_EXECUTION_SETTINGS` (`:664-665`) | `fallback_only_v1` | `fallback_only_v1` |
| `CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1` (`:721-722`) | `fallback_only_v1` | `fallback_only_v1` |
| `CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS` | `ordinary_multiplier_v1` | `ordinary_multiplier_v1` |

The second is the **active family root with 20 descendants**. Its own comment:
*"Ordinary novelty multiplication is disabled. `fallback_only_v1` retains the
bounded all-energy-models-infeasible safety path and its telemetry."*

**So `deferred_gram_fallback_enabled` was `True` in canonical runs.** The rescue
was authorized throughout.

### What still stands from 6aj

- Not in Paper I — still true, 0 occurrences.
- The collapse branch fired 25 times, all at machine-level error — still true.
- The B9 stall observation — still true, and the author has ruled it a non-issue
  since the published trajectory shows it.

### What is now open

Whether the rescue **fires** — the branch runs only when all energy models are
infeasible, which is narrower than being enabled. That was never measured,
because I stopped at the enablement question and got that wrong too.

**Method lesson.** A derived value's absence from a receipt proves nothing about
whether it was set. Before concluding "never enabled", check whether the flag is
recorded at all, and if not, resolve it from the profiles that feed it.

## 6al. RETRACTION of Q39 — PAOP is inside canonical HH `full_meta`

**My error, caught by Codex's stop condition before any edit.**

6ag concluded PAOP was outside `full_meta` because
`primitive_pools._build_full_meta_pool` (`:895-1214`) contains no `paop`.
**There are two `full_meta` builders**, and Hubbard--Holstein uses the other one:

```
ra_adapt/pools.py:_parent_pool_spec        -> pool_key="full_meta" for ordinary L=2 HH
  _parent_records -> resolve_pool_plan     (no class or label filter)
  pool_resolution.py                        -> build_hh_pool_by_key
  hh_pool_presets.py:_build_hh_full_meta_pool (:208-548)
        lists paop_full, paop_lf_full, _HH_FULL_META_EXTRA_PAOP_KEYS
        as REQUIRED components, then builds and appends them
```

Verified independently: `_build_hh_full_meta_pool` contains `paop_full`,
`paop_lf_full` and `_HH_FULL_META_EXTRA_PAOP_KEYS`.

**Codex's direct measurement:** materializing the ordinary small HH parent pool
gives **89 candidate-position records, 40 with PAOP labels** — 45% of the
canonical pool. Its Paper-I-scoped GitNexus graph independently rates the PAOP
builder blast radius HIGH/CRITICAL across pool resolution, replay scaffolds, the
exact-benchmark audit and the executor.

**What was wrong in 6ag, item by item:**

| 6ag claim | status |
|---|---|
| 0 occurrences in Paper I | still true |
| no canonical profile selects a `paop_*` pool **key** | still true — but irrelevant, since PAOP enters as a *component* of `full_meta`, not as a pool key |
| "not inside `full_meta` construction" | **false** — I checked the non-HH builder |
| Paper-IV receipt shows `adapt_pool = "full_meta"` | true, and now means the opposite of what I concluded: it means Paper IV got PAOP too |

**Open for the author (Codex's question):** should Q39 be re-scoped so canonical
`full_meta` keeps its 40 PAOP records while only the standalone PAOP pool keys and
their 4 executor parameters are retired — or should PAOP be removed from
canonical `full_meta` as an explicit scientific pool change?

**Method lesson, second in a row.** Both retractions came from checking one
artifact and generalizing: a checkpoint that never stores a derived flag (Q41),
and one of two same-named builders (Q39). When a name resolves to more than one
implementation, enumerate them before concluding.

## 6am. PAOP is Paper-I material — keep (Q42)

Author's decision, closing the retracted Q39: **if it was used for Paper I, keep
it.** PAOP contributes 40 of the 89 candidate-position records in the canonical
Hubbard--Holstein parent pool, so it is live Paper-I material.

Consequences:

- **No pool change, no rerun, no manuscript note.** The operator pool behind
  every Bundle stays as it was.
- Work-plan item **1.1 is withdrawn**. Codex stopped before editing, so nothing
  needs reverting.
- The 4 executor parameters (`paop_r`, `paop_split_paulis`, `paop_prune_eps`,
  `paop_normalization`) **stay**. They configure a pool the canonical path
  builds.
- 6ah's reconciliation of `paop_prune_eps` still holds on its own terms — it is
  pool-polynomial truncation and belongs in the generator/pool layer, not in
  `extensions.py` — but it is no longer a deletion candidate.
- `hh_vqe_from_adapt_family.py` (131 refs) and `hh_pool_presets.py` (223) are not
  to be stripped.

**Note on the manuscript.** PAOP does not appear in
`Paper_I_author_revision.tex` under that name. That is a naming question, not an
evidence question: the pool it contributes to is the `full_meta` physical macro
pool the paper does describe. Flagged only so a future reader applying the
"is it in Paper I?" test does not repeat this deletion attempt.

## 6an. B5 evidence is not local — retrieval task for Codex

Constraint 3 names Bundles 3-9 as reproduction targets. **B5 has no local run
archives**, and it is `HH-B5` in the behavioral contract — the hierarchical
parent-to-singleton profile, one of the four named profiles. So it is a
reproduction target that cannot currently be checked.

**Where it is.**

```
chtc/paper_i_ra_adapt_repair_20260727/
  paper_i_ra_adapt_macro_then_singleton_phase123_qiskit_phase23_no_lanes_tau1em4_r50_20260807_v1_chtc/
    jobs/            6 staged jobs, one per regime
    package_manifest.json    6 job entries
    raw_outputs/     EMPTY — results were never fetched back
```

The six staged jobs match the six regimes in the bundle pages:
`weak_weak`, `intermediate_weak`, `strong_weak_u8`, `weak_strong`,
`intermediate_strong`, `strong_strong_u8` — each
`ra_macro_then_singleton_phase123_qiskit_phase23_plateau`.

**Task ownership.** Retrieval is a read, not a state change, but it needs CHTC
access and Duo. Codex has repo access and can run it; the repo has a
`chtc-direct` capability for exactly this. Claude cannot complete the
authentication.

**What is needed** to bring B5 to the same footing as b3/b6/b7/b8/b9: the run
archives for those 6 jobs, specifically each run's
`run/checkpoints/current.json` and `run/result/estimator_ledger.json`, landed
under that package's `raw_outputs/`.

**Do not** submit, resubmit or modify any job. This is a fetch of completed
results only. If the results are not on the cluster, report that rather than
re-running — a rerun is a scientific decision and is the author's.

Until this lands, constraint 3 is verifiable for five of seven bundles and
asserted for b4 and b5.

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

## PROGRESS — measured, 2026-08-24

Verified on the working branch after merging Codex's increments.

| measure | start | now | change |
|---|---|---|---|
| `_run_hardcoded_adapt_vqe` parameters | 348 | **286** | −62 |
| the mega function, lines | 41,210 | **39,742** | −1,468 |
| `adapt_pipeline.py`, lines | 72,528 | **70,235** | −2,293 |
| CLI flags in `cli_config.py` | 409 | **336** | −73 |
| collection errors | 54 | **54** | unchanged |

New: `extensions.py` (1,360 lines) and `response_accounting.py` (505).

### Increments landed

| what | commit | net lines |
|---|---|---|
| dead test-only modules (Claude) | `cf1ba3a0` | −5,827 |
| `ResponseAccounting` extraction (Codex) | `05604a36` region | +40 |
| Q20–Q23 (Codex) | `05604a36` | −698 |
| prune extraction (Codex) | `54980606` | −295 |
| batch (Codex) | `bb4541f6` | ⎫ |
| beam + default surfaces (Codex) | `17fc2d56` | ⎭ −6,517 code lines across the Q24–Q28 pass |

Codex's verification on its own branch: ledger byte-identical (289 entries, 527
ordered occurrences, fingerprint `651c0dda…109ac`), 242 focused regressions
passing, empty `ERROR test/...` diff, working tree clean.

**The `+40` line is worth keeping visible.** The `ResponseAccounting` extraction
moved code between files and added 40 lines — it condensed nothing, which is what
prompted rule 6i. It was still worth doing: it demonstrated that the estimator
ledger survives a structural change byte-identically, which every later deletion
relies on.

## GOAL BRIEF — the mega function and the Paper-I route

**Mode note (author, 2026-08-24).** Goal mode is overkill where the work plan
already names items and sites — those are tasks. Goal mode is for work whose path
cannot be enumerated in advance: the mega function and the Paper-I route. What is
specifiable there is not the steps but the **constraints**.

### The goal

`_run_hardcoded_adapt_vqe` (39,742 lines, 286 parameters) and the Paper-I route
become the shape in the design target: `algorithm.py` (the loop), `ansatz.py`
(accepted ansatz, insertion, refit), `generators.py` (generator support, scoring,
cost terms), `extensions.py` (batch, prune, beam — already exists).

### Hard constraints — reproducibility

1. **The estimator ledger is byte-identical.** Same primitive ids, same ordering,
   same fingerprint, on a completed-run replay. This has held through every
   increment so far and is the single strongest signal.
2. **Accepted generator/position sequences do not change**, except where a
   decision in DECISIONS explicitly licenses it. Q1: the standard is *at least as
   good* on DeltaE, qiskit costs and estimator count — not bit-identity — but a
   changed accepted sequence must be attributable to a licensed change, never a
   side effect.
3. **Bundles 3 through 9 reproduce, across all six regimes.** The reference
   artifact is
   `output/pdf/paper_i_ra_allphase_adaptive_20260817/paper_i_bundle_pages_20260823.pdf`
   — 7 pages, one per bundle, each plotting `|E_k - E_exact|` against accepted
   controller round `k` for `weak_weak`, `intermediate_weak`, `strong_weak_u8`,
   `weak_strong`, `intermediate_strong`, `strong_strong_u8`, with AAVQE
   comparator curves. Those curves are the reproduction target, judged by Q1's
   *at least as good* standard on DeltaE, qiskit cost and estimator count.

   **Evidence availability is not uniform** — check before assuming a bundle can
   be replayed locally:

   | bundle | local run archives |
   |---|---|
   | b3 | 18 |
   | b4 | **none** |
   | b5 | **none** — the macro-then-singleton route, CHTC only |
   | b6 | 18 |
   | b7 | 3 |
   | b8 | 6 |
   | b9 | 13 |

   b4 and b5 have no `output/local_runs` archives. B5 is the contract's HH-B5
   hierarchical parent-to-singleton profile, so it is a *named* reproduction
   target with no local evidence to replay against. Reproducing it needs the
   CHTC material under
   `chtc/paper_i_ra_adapt_repair_20260727/paper_i_ra_adapt_macro_then_singleton_*`.

4. **The compilation contract is fixed**: `FakeMarrakesh`, optimization level 1,
   `seed_transpiler` 7 (Q31, Q34). `N_2q`, `D_2q`, `D_c` are properties of that
   contract.
5. **`_canonical_route_contract_for_request` keeps its resolved contract
   byte-identical.** It is CRITICAL — 23 upstream symbols, five processes
   including `run_ra_adapt` and campaign authorization. A contract-digest change
   stops the increment.

### Hard constraints — method

6. **Paper I is the definition.** Where code and manuscript differ, the
   manuscript decides what the method *is*. Where a construct is absent from the
   manuscript, that is a question, not a licence to delete — see the retractions
   in 6ak and 6al.
7. **No fallbacks** (rule 7). A requested computation that cannot be performed
   stops the run. No silent substitution: not qiskit to proxy, not metric for
   Hessian, not one backend for another.
8. **One representation of each fact.** Where two exist, a guard is needed to
   keep them agreeing; that guard is the signal to collapse them, not to keep.
9. **No setting that cannot take effect**, and no receipt field for an event that
   cannot happen (Q29).
10. **Optional extensions are absent by default**, with required choices supplied
   by a conditional policy interview rather than defaults (Q25-Q27).

### Hard constraints — process

11. **DECISIONS is binding and changes between increments.** Re-read it each
    time. An unanswered question is not permission to choose.
12. **Report net lines removed** per increment (rule 6i). Extraction that moves
    without deleting is permitted only as a prerequisite to a named deletion.
13. **Enumerate before concluding.** Both retractions came from checking one
    artifact: a checkpoint that never stores a derived flag, and one of two
    same-named builders. When a name resolves to more than one implementation,
    or a value is derived rather than stored, resolve every path before acting.
14. **Stop and report** on: a ledger fingerprint change, a rise in collection
    errors, a route-contract digest change, or a canonical profile selecting
    something the worklog calls unused.

### Not in scope

The four evidence profiles and their numbers; optimizer tolerances; estimator
accounting semantics; forced-admission behaviour; cost normalization. Changing
any of those is a scientific decision with its own rerun and manuscript review.

## WORK PLAN — execute in this order

Derived from decisions Q1-Q39. Each item names its decision, its sites, and what
it removes. Rule 6i applies throughout: report measured net lines.

### Stage 1 — verified-dead deletions (no numerical effect)

| # | item | decision | surface | verified by |
|---|---|---|---|---|
| 1.1 | PAOP pool family | Q39 | 812 refs, 4 params | absent from Paper I; no profile selects it; not in `_build_full_meta_pool`; Paper-IV receipt shows `adapt_pool="full_meta"` (6ag) |
| 1.2 | `maturity_*` parameters | Q36 | 12 params | 0 in Paper I, 0 read sites in scorer, all caps default off (6ad) |
| 1.3 | `phase3_cheap_ratio_v1` | Q21 | `:3378-3430` | **no callers anywhere** (6ac) |
| 1.4 | Phase-0 cost path | Q14, Q17 | 46 refs / ~94 lines in `adapt_pipeline.py`, 9 in types, 6 in `ra_adapt/*`, 7 in tests | `phase0_K0 == 1.0` in all 30 recorded instances (6p) |

**Do not** delete the phase controller in 1.2 — only the `maturity_*` parameters
and `phase_shots_maturity_floor`. The snapshot named `..._maturity_v2` holds 37
fields of live controller state including the funnel caps (6ae).

### Stage 2 — substitutions and fallbacks

| # | item | decision | note |
|---|---|---|---|
| 2.1 | `lambda_F * F` metric-for-Hessian substitution | Q21 | 3 sites; it is in the score **numerator** (6p). `lambda_F` itself goes entirely — every use is the substitution (6ac) |
| 2.2 | `metric_proxy` / `F_metric` unification | Q20 | one `F` field; delete the `abs(gradient)` branch at `:39614` (6m) |
| 2.3 | fallbacks **and** fallback-reporting fields | Q29 | 222 keys, sorted into reporting / policy / mechanism in 6x. `phase2_missing_curvature_fallback_used` is never `True` |
| 2.4 | `allow_aer_fallback` | Q30 | 28 refs, noise-oracle path |
| 2.5 | preferred-fakes chain | Q31 | `_DEFAULT_PREFERRED_FAKES`, `allow_preferred_fallback`, `requested_backend_shortlist`, `shortlist_reduction_mode`, and the bare `except Exception: pass` at `:350` |

### Stage 3 — defaults and contracts

| # | item | decision |
|---|---|---|
| 3.1 | cost default `"proxy"` -> qiskit compiled | Q32 |
| 3.2 | cost source per-phase -> single run-level `request.cost` | Q33 — collapses 3 scope constants and 7 duplicated pairs |
| 3.3 | transpiler seed -> run-level compilation contract | Q34 |
| 3.4 | six `lambda_*` cost weights -> run-level | Q35 (**not** `lambda_F`) |
| 3.5 | rename the snapshot version, drop the old string | Q38 — costs continuation of existing bundles, accepted |

### Stage 4 — extension relocation

| # | item | decision | surface |
|---|---|---|---|
| 4.1 | prune -> `extensions.py` | Q27 | 35 params, 36 flags, 2293 mentions — safest first, active family pins it off |
| 4.2 | beam -> `extensions.py` | Q27 | 11 params, 11 flags, 814 mentions |
| 4.3 | batch -> `extensions.py` | Q27, Q24 | 11 params, 20 flags, 1105 mentions |
| 4.4 | delete the non-manuscript batch gates | Q28 | `batch_additivity_tol` and the 5-weight penalty — off the published route (6v) |

Extension choices come from the **conditional policy interview** (Q25, Q26), so
the 57 params and 67 flags do not reappear as defaults. Canonical batch is
`argmax_B dE_3(B)/K_3(B)` over cap-bounded generator-distinct subsets whose joint
solve exists (6u) — keep `batch_size_cap` and `enable_batching`, both are in the
manuscript.

### Stage 5 — the core refactor

| # | item | reference |
|---|---|---|
| 5.1 | `Response.restrict(candidates) -> ResponseBlocks` | 6j, 6k — restriction is over **candidates**; the active side is the anchor |
| 5.2 | `evaluate(record, state, response, cost, order)` = `descent(order) / cost` | 6g, corrected by 6j |
| 5.3 | delete the 26 `CandidateFeatures` geometry fields | 6n, 6o — only 4 are values; 7 are policy, 6 index sets, 2 telemetry, 7 proxy/fallback |
| 5.4 | phases 0-III as one `score()` over a phase table | design target section 2 |

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
| Q21 | The `lambda_F * F` substitution — metric standing in for the energy second order | **Delete it.** "Subbing in the gram part squared for the energy second order was an old proxy I want deleted." Covers both the Phase-I legacy energy model and the Phase-II cheap curvature proxy. | 2026-08-24 |
| Q22 | Retiring `SR_ROUTE_PROFILE_CONVENTIONAL_V3_1` as a consequence of Q21 | **Accepted.** "It's legacy of course." |  2026-08-24 |
| Q23 | Confirm from a run receipt that the legacy numerator never fired? | **No — not worth it.** Deletion proceeds on source evidence: `phase1_energy_model` is explicitly pinned to `FIRST_ORDER_FS_TRUST_V1` in both canonical families. Recorded so the basis is known. | 2026-08-24 |
| Q24 | The five-term compatibility penalty vs the Schur additivity defect | **Two nested defaults, both off.** `batch` defaults **off**. If `batch = true`, the default is **canonical Paper-I batch** — block feasibility from the Schur-reduced quadratic, `defect = 1 - dE_joint / sum(dE_i)` against `batch_additivity_tol`. The five-weight heuristic becomes a separate legacy `batch_mode`, **also default off**. Reaching it requires opting in twice. | 2026-08-24 |
| Q25 | Should legacy-batch weights be pinned in a profile? | **No — the if-then asks for them.** Enabling `legacy_batch` requires supplying the five weights; they have no defaults. This is `CONTEXT.md`'s **conditional policy interview**: silent while the policy is off, reveals its required choices when enabled. Nothing to drift from, and every legacy-batch run records them by construction. | 2026-08-24 |
| Q26 | Do prune and beam follow the batch shape? | **Yes.** All optional extensions are off by default, and enabling one asks for its required choices through a conditional policy interview rather than supplying defaults. | 2026-08-24 |
| Q27 | Move batch, prune and beam out of the main algorithm and the run commands? | **Yes.** They leave both `adapt_pipeline.py` and `cli_config.py` for `extensions.py`, with their choices supplied by the conditional policy interview (Q25/Q26) rather than by flags and parameter defaults. | 2026-08-24 |
| Q28 | Keep superseded paths as off-by-default options, or remove them? | **Preference is deletion**, or failing that a **true extrication** so the path cannot be reached from the working part at all. An off-by-default flag is not sufficient — it leaves the code reachable and re-enableable. Refines Q24. | 2026-08-24 |
| Q29 | Delete fallback-*reporting* fields along with the fallbacks? | **Yes.** No receipt field should exist for an event rule 7 says cannot happen. An always-`False` flag is not evidence that nothing went wrong — it is a field describing an impossible event. Fallback, and the field recording it, go together. | 2026-08-24 |
| Q30 | `allow_aer_fallback` | **Delete it.** "We have FakeMarrakesh backend we compile to; why would we need some fallback?" An unavailable backend stops the run. | 2026-08-24 |
| Q31 | The preferred-fakes chain on the compile path | **Delete it.** `FakeMarrakesh` is the single compile target, as Paper I's COST block states. No chain, no `allow_preferred_fallback`. An unavailable backend stops the run rather than compiling to a device the manuscript does not name. | 2026-08-24 |
| Q32 | `BackendCompileConfig.mode` defaults to `"proxy"` | **Invert it.** Qiskit compiled cost is the default; the proxy is used only when explicitly requested. Paper I's headline axis is the compiled resource tuple, so the default must produce it. The proxy stays as a peer implementation behind the one `CostTerm` interface, never as a substitute (rule 7). | 2026-08-24 |
| Q33 | Is cost source per-phase or run-level? | **Single run-level choice.** It used to be per-phase; simplify to one selection applying across Phases I-III. With Phase-0 cost deleted (Q14), there is no phase that differs. | 2026-08-24 |
| Q34 | `phase3_backend_transpile_seed` — per-phase or run-level? | **Run-level.** The seed joins the compilation contract with the target and optimization level, as Paper I's COST block fixes `seed_transpiler = 7` for the run. No phase may compile against a different seed. | 2026-08-24 |
| Q35 | The six lambda weights — per-phase or run-level? | **Run-level, not per-phase.** Applies to the cost weights `lambda_2q`, `lambda_d`, `lambda_1q`, `lambda_theta`, `lambda_shot`. **`lambda_F` is different — it does not survive Q21 at all**, see 6ac. | 2026-08-24 |
| Q36 | The maturity shot-budget controller | **Archaic — delete.** Adaptive per-candidate shot allocation. Absent from Paper I, no read sites in the scorer, and every cap defaults off. | 2026-08-24 |
| Q37 | The resume guard that raises on an unrecognised maturity snapshot | **Delete it with the controller.** Completed Bundle runs do not need restarting, and this is exactly the class of guard the author has said hurts rather than helps — it exists only to police a serialized section that is itself being removed. | 2026-08-24 |
| Q38 | The snapshot version string `phase123_controller_maturity_v2` | **Rename it, and drop the old string** — no alias, no dual-read. The name no longer describes what the snapshot holds (6ae). **Consequence accepted: existing locked-bundle sidecars can no longer be continued** by the new code, since resume validates the version and will not recognise the old one. | 2026-08-24 |
| Q39 | The PAOP pool family | **Delete.** Absent from Paper I, selected by no profile, and unused by Paper IV — verified from that lane's own run receipt, which records `adapt_pool = "full_meta"`. 812 references, 4 executor parameters. | 2026-08-24 |
| Q41 | The deferred-Gram rescue and the metric-collapse branch | **Delete both.** The rescue never fired in 58 archives. The collapse branch fired 25 times, but **every occurrence is at machine-level error** — `F_red` ~ 2e-11 with `dE` between `0.0` and `-6e-11` and `max_grad` ~ 1e-7 — so it only ever acted on candidates contributing floating-point noise. Author accepts that removing it may alter accepted trajectories at that level. | 2026-08-24 |
| Q41 | **RETRACTED** — see 6ak | The delete decision rested on a false premise. `deferred_gram_fallback_enabled` is derived, not stored; the active family enables it via `fallback_only_v1`. Whether the rescue *fires* is still open. | 2026-08-24 |
| Q39 | **RETRACTED** — see 6al | PAOP is inside canonical Hubbard--Holstein `full_meta`: 40 of 89 records in the materialized parent pool. Codex stopped before editing. Re-scoping question open. | 2026-08-24 |
| Q42 | Re-scope of the retracted Q39 | **Keep PAOP.** It was used for Paper I — 40 of the 89 records in the canonical HH parent pool — so it stays. No pool change, no rerun. The standalone `paop_*` pool keys and the 4 executor parameters are **not** to be deleted either while the pool depends on the same builder machinery; treat PAOP as live Paper-I material. | 2026-08-24 |

### Handoff register — author's guidance, 2026-08-24

*"I think your prompts are too aggressive; they should be like 'work into your
refactor naturally if it comes up, or afterwards.'"*

An executing agent is usually mid-task and holding context. A findings handoff
that arrives as a blocking increment, with hard gates and stop conditions, fights
that context and forces a switch that costs more than the fix.

**Write findings as things to fold in, not as interrupts:**

- Lead with "when you are next in this area" or "afterwards", not "TASK:".
- Keep the evidence and the caveats — those are the value. Drop the imperative
  mood and the mandatory sequencing.
- A contradiction is "worth raising rather than working around", not
  "STOP AND REPORT".
- State the basis for a decision so the executor can judge it, rather than
  ordering compliance with it.
- Let the executor choose the moment. Only genuine blockers — evidence at risk,
  a change that would invalidate work already done — justify an interrupt.

This applies to handoffs of *findings*. A scoped increment the author has
explicitly commissioned may still be written as a task.

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
| Q20–Q23 — unify the metric field and delete gradient/curvature substitutions plus dead telemetry | Codex | 2026-08-24 | done in `05604a36`; 698 net lines removed; conventional-v3.1 retired under Q22 |
| Q24–Q28 — condense batch, prune, and beam behind `extensions.py`; delete superseded batch gates | Codex | 2026-08-24 | done in `54980606` (prune), `bb4541f6` (batch), and `17fc2d56` (beam/default surfaces); 6,812 net lines removed |
| Work Plan Stages 1–3 — verified-dead deletions, fallbacks, and unified cost contracts | Codex | 2026-08-24 | awaiting author decisions: Q30 cross-lane scope, Q33 graph-span disposition, Q38 replacement snapshot name; all independent ratified increments complete |
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

### 2026-08-24 — Q20–Q23 metric and curvature-proxy condensation — COMPLETE

**Goal:** represent the measured Fubini--Study scalar once as `F`, delete the
non-Phase-III `abs(gradient)` substitution, remove both `lambda_F * F`
energy-curvature substitutions and their pass-through telemetry, and retire
the conventional-v3.1 compatibility route authorized by Q22.

**Coordination and baseline:** `paper-ii-exchange-selector` through
`e658d42b` was merged before the Q21–Q23 claim expansion.  The clean measured
baseline was `69d85e29cc0c85c29ffcf5e859565d647de3bff2`; the final claim-only
coordination parent was `91a76bbc7f5097fa2c93899169334734cacdce83`.

```bash
python3 pipelines/shell/ram_guard.py --limit-mb 8000 -- \
  python3 -m pytest test --collect-only -q \
    --continue-on-collection-errors
# before -> 5573 tests collected, 54 errors; peak RSS 467 MB
# after  -> 5560 tests collected, 54 errors; peak RSS 422 MB

diff -u errors-before.txt errors-after.txt
# -> empty diff
```

The 13-test collection decrease is the intentional removal of tests for the
deleted field aliases, proxy branches, telemetry, and conventional-v3.1
route.  No collection error was added or removed.

**Mandatory pre-deletion evidence:** the smallest available completed
Bundle-9 archive,
`b9mr__b_depth_append_ra__strong_weak_u8__nph3__9675590__2.tar.gz`, was
examined and its temporary extraction was removed afterwards.  Across all 30
checkpoint rows carrying both values, `metric_proxy > F_metric` occurred zero
times.  Thus the `abs(gradient)` arm had not changed the completed-run value
surface covered by the archive, and the Q20 stop condition did not trigger.

**Implementation:** commit `05604a36d4a9f464d891fb92e54b6c1edb366980`
changes 27 files, adding 214 lines and deleting 912: **698 net lines removed**.

- `CandidateFeatures.F` is the sole measured scalar; `F_metric`,
  `metric_proxy`, and `cheap_metric_proxy` are gone from construction,
  scoring, debug, serialization, and tests.
- Phase I now has only `rho * g_lcb / sqrt(F)`.  The legacy
  `lambda_F * F` quadratic numerator, its numeric CLI/executor controls, and
  its telemetry are deleted.
- Phase II retains measured directional curvature.  Its lambda-F ratio proxy,
  dead missing-curvature flag, and lambda-F telemetry are deleted.  The
  historical optional-curvature path now contributes zero when no measured
  `h_hat` exists instead of fabricating an energy Hessian from the metric.
- `SR_ROUTE_PROFILE_CONVENTIONAL_V3_1` and its aliases, contract, digest, and
  dispatch are deleted.  Unqualified `sr_snake`/`conventional`/`canonical`
  now resolve to retained conventional-v3.  Frozen historical v1–v3 contract
  payloads remain unchanged.
- Removing the now-false lambda-F semantic invariant changes active route
  digests by construction.  The current Phase-III Qiskit Page-7 parent source
  lock is therefore updated from `69af64db...` to `3f4ebed3...`; preserved
  historical artifact and reporting locks retain their historical digest.

**Paper-I-local impact audit:** a disposable GitNexus graph contained only the
118 `pipelines/static_adapt` files plus
`hh_continuation_scoring.py` and `hh_continuation_types.py` (121 source files
total).  It did not consult the repo-wide index.  Baseline impact analysis
classified the central scoring and route-contract symbols as HIGH/CRITICAL,
including 59 upstream symbols and 11 processes for the active no-prune
contract.  The pre-commit `detect-changes --scope all` mapped exactly the ten
intended production files, 52 changed symbols, and 38 Paper-I execution flows;
no unrelated production module appeared.

**Receipt-identity lock:** the same cache-disabled, completed two-round
Hubbard--Holstein run was executed at the baseline and after the change.  Each
ledger has 444 entries, 555 occurrences, and fingerprint
`28c61d6849f78ce4a07d60e90454c566d5ecb9d9723eaa0aa76f1ac412a0dde6`.
The complete canonical JSON is byte-identical:

```bash
shasum -a 256 ledger-before.json ledger-after.json
# -> 9a1e912039ef28a2437097077ac47c4161faf31eb327f9ffefa6e8030d5b547f  (both)
cmp -s ledger-before.json ledger-after.json
# -> exit 0; 619705 bytes each
```

This locks primitive ids, ledger keys, entry order, occurrence/receipt order,
consumer identities, and the ledger fingerprint.  No scientific accounting
identity changed.

**Regression evidence:** the cache-disabled changed unit surface passes
`521 passed in 48.61s` at 300 MB peak RSS.  The entire changed integration
file was then run on the baseline and after the change: baseline
`30 failed, 291 passed`; after `30 failed, 290 passed`.  The exact `FAILED`
node-id diff is empty; the one-pass decrease is the deliberately deleted
proxy test.  All 30 failures therefore pre-exist this increment.

### 2026-08-24 — Q24–Q28 extension condensation — COMPLETE

**Goal:** remove batch, pruning, and beam from the default executor/CLI
surface, require their scientific choices only when the typed extension is
enabled, and delete superseded batch and legacy beam paths rather than leave
re-enableable flags.

**Coordination and baseline:** `paper-ii-exchange-selector` through
`770f7fc1` was merged before implementation.  The clean claim/baseline commit
was `aea78d9abf01133d4cb3562d35441c98f132362e`:

```bash
python3 pipelines/shell/ram_guard.py --limit-mb 8000 -- \
  python3 -m pytest test --collect-only -q \
    --continue-on-collection-errors
# before -> 5560 tests collected, 54 errors
# after  -> 5467 tests collected, 54 errors; peak RSS 421 MB

diff -u errors-before.txt errors-final.txt
# -> empty diff
```

The 93-test collection decrease is the intentional deletion of tests for
retired extension flags, compatibility scoring, additivity gates, tie-beam,
and the unreachable legacy controller paths.  No collection error was added
or removed.

**Implementation:** three explicit-path commits add 1,942 lines and delete
8,754: **6,812 net lines removed**.

- `54980606` isolates pruning in `PruningExtension`/`PruningRuntime`, removes
  its 35 executor choices and CLI defaults, and requires the complete
  authenticated pruning contract only when pruning is enabled: **295 net
  lines removed**.
- `bb4541f6` isolates the Paper-I batch extension and retains the manuscript
  choices `maximum_size` and `search_window_size`.  The five-weight
  compatibility score, additivity-defect gate, their telemetry, and their
  superseded tests are deleted: **2,547 net lines removed**.
- `17fc2d56` moves fork-local beam execution and its four required choices to
  `extensions.py`, deletes tie-beam and the legacy controller loop, and
  removes extension fields from exact-benchmark/CHTC default policy surfaces:
  **3,970 net lines removed**.

`_run_hardcoded_adapt_vqe` now has no batch, pruning, or beam parameters, and
`cli_config.py` exposes no corresponding flags.  Disabled extensions are
absence (`None`), while enabled batch/beam/pruning values reject missing
choices.  Frozen historical hashed route dictionaries retain old fields only
as passive provenance.  Residual monolith beam receipt fields are guarded by
an immutable, non-init disabled state; no input, route, CLI flag, or executor
argument can re-enable the deleted legacy loop.  This is Q28's true
extrication, not an off-by-default policy.

**Receipt identity:** the cache-disabled completed one-round run at
`bb4541f6` and after the beam extraction produced byte-identical ledger JSON:

```bash
shasum -a 256 ledger-before.json ledger-after.json
# -> 654ecec994f3040bc37243e82e2e8511e2ac4bbe1878cc39254c4c01e43d5f0c  (both)
cmp -s ledger-before.json ledger-after.json
# -> exit 0
```

Both ledgers contain 289 entries and 527 ordered occurrences with fingerprint
`651c0dda8554920a7f7df50b71da0869ee01450487a221329f6c8e232bc109ac`.
Thus primitive ids, keys, occurrence/receipt ordering, and the fingerprint are
unchanged by the beam extraction.

**Regression evidence:** the final focused extension/controller surface has
`242 passed`.  Two additional facade assertions fail identically at
`bb4541f6`: its expected public-export set omits the already-exported
`EndpointOverlapDisplacementTrust`, and its stale fixed ledger count expects
709 where the baseline produces 585.  The final changed sources compile, and
`git diff --check` is clean.

**Paper-I-local impact audit:** an accurate disposable GitNexus repository was
built from `bb4541f6` with `pipelines/static_adapt`, the two continuation
scaffold modules, and only the touched Paper-I exact-benchmark/CHTC files.  The
large monolith was explicitly included.  Final `detect-changes --scope all`
reports 15 production files, 88 changed symbols, two affected controller
flows, and MEDIUM risk; no repo-wide index was consulted.

### 2026-08-24 — Work Plan Stages 1–3 — STOPPED BEFORE 1.1

**Coordination and baseline:** `paper-ii-exchange-selector` through
`c31f1563a38b` was merged first.  The clean claim/baseline commit is
`e995fe6c51c8`.  Collection is `5467 tests collected, 54 errors`; the baseline
ledger completed one round with 256 entries, 266 ordered occurrences, and
fingerprint
`196e8029ff0830f2abe6a6af09d0d0059225952b12abd12d2bc6494b90a71ef1`.

**Stop condition:** Q39 and 6ag say PAOP is outside canonical `full_meta`, but
the live Hubbard--Holstein resolver proves the opposite:

- `ra_adapt/pools.py:_parent_pool_spec` selects `pool_key="full_meta"` for the
  ordinary L=2 HH problem.
- `_parent_records` passes it through `resolve_pool_plan` with no class or
  label filter.
- `pool_resolution.py` dispatches HH `full_meta` to
  `build_hh_pool_by_key`.
- `hh_pool_presets.py:_build_hh_full_meta_pool` lists `paop_full`,
  `paop_lf_full`, and `_HH_FULL_META_EXTRA_PAOP_KEYS` as required components,
  then builds and appends them.

A direct materialization of the ordinary small HH parent pool measured 89
candidate-position records, **40 with PAOP labels**.  This is exactly the
commissioned stop condition: a canonical path selects something the worklog
calls unused.  The Paper-I-scoped GitNexus graph independently classifies the
PAOP builder blast radius HIGH/CRITICAL across pool resolution, replay
scaffolds, the exact-benchmark audit, and the core executor.

No PAOP source, test, default, parameter, or later Stage 1–3 item was edited.
Measured increment net is **0 lines**, collection-error delta is **0**, and no
post-edit ledger exists because implementation did not begin.

**Question for ratification:** should Q39 be withdrawn/re-scoped so canonical
`full_meta` retains its 40 PAOP records while only standalone PAOP choices are
retired, or should PAOP be removed from canonical `full_meta` as an explicit
scientific pool change?

### 2026-08-24 — Work Plan 1.4 — Phase-0 cost path deleted

**Implementation:** commit `5680f959` deletes the executable Phase-0 hardware
cost denominator, normalization, lambdas, compile/measurement acquisition, and
the corresponding candidate/checkpoint/accounting fields. Phase 0 now ranks the
same recorded quantity directly as `DeltaE0_upper * N0`. The change adds 22
lines and deletes 187: **165 net lines removed**.

The two `phase0_cost = "gradient_only_no_metric_no_resource_v1"` entries in the
locked Page-12 application-source contracts remain. Removing either invalidates
an authenticated source-contract SHA; they are passive declarations that Phase
0 performs no resource-cost computation, not the deleted executable cost path.

**Reproducibility:** before and after both produced 256 ledger entries and 266
ordered occurrences. The ledger fingerprint is byte-identical at
`196e8029ff0830f2abe6a6af09d0d0059225952b12abd12d2bc6494b90a71ef1`;
the primitive-id hash is
`a02151a2f0af8120268b32ee074a7e41690de3b04cd0a8d496b749a63a3e7aaa`,
and the complete ledger JSON hash is
`b93079dbac9ef10d3bf6057ee4be7128c8ece6e1d4b1de7819a5bb395702d2e7`.
The resolved `_canonical_route_contract_for_request` digest also remains
byte-identical at
`36a7d72c562738612fea509a795c7f7d5c4a7a93d495e0de7993a1934e8b747b`.

**Collection and tests:** collection remains `5467 tests collected, 54 errors`;
the before/after `ERROR test/...` diff is empty. Ten direct Phase-0/accounting
tests and three locked application-source-contract cases pass. Two broader
targeted assertions fail identically on the clean baseline: the position-route
matrix omits three already-present variants, and a historical route-digest
literal is stale. Neither was changed.

**Paper-I-local impact audit:** final GitNexus `detect-changes` reports four
production files, six changed symbols, one affected execution flow, and MEDIUM
risk. `CandidateFeatures` was the pre-edit HIGH-risk schema surface (23 direct
importers, 80 transitive symbols); collection, direct tests, source-lock checks,
route digest, and ledger identity close that risk.

### 2026-08-24 — Work Plan 2.3 / 3.1 — compile-cost fallback chain deleted

**Implementation:** from baseline `919488ff`, commit `c4400c36` implements Q31
and Q32. `BackendCompileConfig` now defaults to Qiskit compilation against the
single exact target `FakeMarrakesh`; proxy cost remains available only by
explicit selection. The preferred-fakes chain, alternate-backend permission,
shortlist mode/CLI/executor fields, shortlist reduction, and the receipt fields
describing those impossible fallback events are deleted. Explicitly requested
single alternate backend names remain valid experiments, but a failed target
resolution stops construction. The graph-span implementation may load the
static graph representation of the same `FakeMarrakesh` target; errors are no
longer swallowed and it cannot change devices.

The change adds 88 lines and deletes 261: **173 net lines removed**. The
standalone `hh_adapt_backend_shortlist.py` experiment driver remains because it
runs separately named, single-backend experiments rather than selecting or
falling back among targets inside one run.

**Reproducibility:** the completed one-round ledger remains byte-identical:
256 entries, 266 ordered occurrences, fingerprint
`196e8029ff0830f2abe6a6af09d0d0059225952b12abd12d2bc6494b90a71ef1`,
primitive-id hash
`a02151a2f0af8120268b32ee074a7e41690de3b04cd0a8d496b749a63a3e7aaa`,
and complete ledger JSON hash
`b93079dbac9ef10d3bf6057ee4be7128c8ece6e1d4b1de7819a5bb395702d2e7`.
The resolved route-contract digest is also unchanged at
`36a7d72c562738612fea509a795c7f7d5c4a7a93d495e0de7993a1934e8b747b`.

**Collection and tests:** collection changes from `5467 tests collected, 54
errors` to `5466 tests collected, 54 errors`; the one-test decrease is the
intentional deletion of the mutable-shortlist-default test, and the complete
`ERROR test/...` diff is empty. The direct compile-oracle and Phase-I--III
Qiskit-scope surface passes `32 passed`. Broader route-profile/digest failures
encountered during focused runs reproduce at the untouched baseline and are
stale assertions from earlier ratified route changes.

**Paper-I-local impact audit:** final GitNexus `detect-changes` reports six
production files, thirteen changed symbols, one affected controller flow, and
MEDIUM risk. The pre-edit HIGH-risk semantic-final-accounting validator lost
only the fallback receipt assertion; the unchanged ledger and route digest,
focused tests, and empty collection-error diff close that risk.

### 2026-08-24 — Work Plan 3.4 — run-level cost weights

**Implementation:** commit `2d23fb1b` implements Q35. The five surviving
resource weights are now the one run-level tuple
`cost_lambda_{2q,d,1q,theta,shot}`. Both Phase-I `SimpleScoreConfig` and
Phase-II/III `FullScoreConfig` consume that tuple. Ten phase-prefixed executor
parameters and CLI flags are replaced by five run-level parameters and flags;
the old names have no alias. `lambda_F` is not preserved because Q21 deleted
its metric-for-Hessian substitution separately.

All live canonical Phase-I/II pairs were enumerated before the edit and were
equal. The authenticated route contract carries none of these infrastructure
defaults, so its schema and digest did not change. The change adds 50 lines and
deletes 77: **27 net lines removed**.

**Reproducibility:** the completed one-round ledger remains byte-identical at
256 entries and 266 ordered occurrences. Its fingerprint is
`196e8029ff0830f2abe6a6af09d0d0059225952b12abd12d2bc6494b90a71ef1`,
primitive-id hash is
`a02151a2f0af8120268b32ee074a7e41690de3b04cd0a8d496b749a63a3e7aaa`,
and complete ledger JSON hash is
`b93079dbac9ef10d3bf6057ee4be7128c8ece6e1d4b1de7819a5bb395702d2e7`.
The resolved route-contract digest remains
`36a7d72c562738612fea509a795c7f7d5c4a7a93d495e0de7993a1934e8b747b`.

**Collection and tests:** before and after are both `5466 tests collected, 54
errors`; the `ERROR test/...` diff is empty. The direct CLI/default and score
configuration surface passes `50 passed`.

**Paper-I-local impact audit:** final GitNexus `detect-changes` reports two
production files, seven changed symbols, one affected controller flow, and
MEDIUM risk. All pre-edit symbol impacts were LOW.

### 2026-08-24 — Work Plan 3.3 — run-level transpiler seed

**Implementation:** commit `abdba188` implements Q34. The executable parameter,
CLI flag, output field, backend experiment drivers, and compile-oracle builders
now use the run-level name `backend_transpile_seed`. The authenticated route
contracts retain `phase3_backend_transpile_seed` byte-for-byte as historical
provenance; `normalize_sr_route_profile_namespace` translates that key once at
the contract boundary and does not create a runtime alias. The implementation
adds 20 lines and deletes 20: **0 net lines removed**.

**Reproducibility:** the completed one-round ledger remains byte-identical at
256 entries and 266 ordered occurrences. Its fingerprint is
`196e8029ff0830f2abe6a6af09d0d0059225952b12abd12d2bc6494b90a71ef1`,
primitive-id hash is
`a02151a2f0af8120268b32ee074a7e41690de3b04cd0a8d496b749a63a3e7aaa`,
and complete ledger JSON hash is
`b93079dbac9ef10d3bf6057ee4be7128c8ece6e1d4b1de7819a5bb395702d2e7`.
The resolved route-contract digest remains
`36a7d72c562738612fea509a795c7f7d5c4a7a93d495e0de7993a1934e8b747b`.

**Collection and tests:** before and after are both `5466 tests collected, 54
errors`; the `ERROR test/...` diff is empty. Route-profile normalization and
contract materialization pass `112 passed`. Three broader output tests fail
identically at the untouched `e5d00dbc` baseline because they still expect
prune/batch fields deleted by the completed Q24-Q28 extension increments.

**Paper-I-local impact audit:** all pre-edit symbol impacts were LOW. Final
GitNexus `detect-changes` reports six production files, fifteen changed symbols,
seven serialization/contract flows, and HIGH aggregate risk. Exact route and
ledger hashes, the complete route-profile test file, baseline failure
reproduction, and the empty collection-error diff close that risk.

### 2026-08-24 — Work Plan Stages 1-3 — remaining decision boundaries

The independently executable ratified items are complete. The remaining three
surfaces require author choices that DECISIONS does not yet supply:

- **1.2 / 3.5:** Q36 and Q38 require the maturity parameters and
  `phase_shots_maturity_floor` to be deleted in the same change that renames
  `phase123_controller_maturity_v2`. Q38 does not name the replacement version
  string. No string was invented and the phase controller was not touched.
- **2.4:** Q30 ratifies deleting `allow_aer_fallback` for Paper I, but the field
  and fallback mechanism are implemented in the shared
  `exact_bench/noise_oracle_runtime.py` and are called by active time-dynamics
  code. Paper-I call sites alone can be made strict, but deleting the shared
  mechanism changes Paper-II semantics; that cross-lane scope is not ratified.
- **3.2:** Q33 names two run-level sources, Qiskit and the logical-ladder proxy,
  while the active Paper-I family explicitly selects the third implementation
  `marrakesh_graph_span_v1`. DECISIONS does not say whether graph-span is
  deleted, mapped to `proxy`, or retained as a third run-level source. No
  scientific mapping was inferred.

The Q20/Q21 fallback-reporting fields named in 6o
(`metric_proxy`, `cheap_metric_proxy`, `phase1/2_lambda_f_proxy_applied`, and
`phase2_missing_curvature_fallback_used`) are already absent from production
code. The deferred-Gram path and all PAOP machinery remain intact under the Q41
retraction and Q42. The 16 default-off policy/mode parameters, 30 `_mode`
parameters, and three legacy-named settings excluded by the commission were not
touched.
