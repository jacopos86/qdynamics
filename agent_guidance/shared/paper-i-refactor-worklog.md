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

---

## Execution log _(Codex-owned)_

Append one entry per increment: goal, commands run, measured result, and
whether the stop condition triggered.
