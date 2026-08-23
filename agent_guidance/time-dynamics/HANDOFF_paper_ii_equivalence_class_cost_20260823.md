# Handoff — Paper II: least-cost realization over the insertion equivalence class

**Author:** Claude (planning session 2026-08-22/23) · **Executor:** Codex (repo access)
**Contract:** read `agent_guidance/shared/agent-handoff-contract.md` first. It defines
anchors, evidence standards, traps, and the report-back format. This document supplies
only what is specific to this task.

---

## 1. Anchors

| field | value |
|---|---|
| Checkout | `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3` (**not** the `~/Documents/Holstein_implementation/` iCloud mirror) |
| Branch | `paper-ii-exchange-selector` |
| Base commit | `e3852b36e21cf2306a10a05b53958d2e3271758e` |
| Lane test baseline | `PYTHONPATH=. python3 -m pytest test/test_ap_mclachlan_*.py test/test_time_dynamics_*.py test/test_paper_ii_runs.py -q -p no:randomly` → **295 passed** |
| Repo-wide collection | Has 55 pre-existing collection errors unrelated to this lane; see `agent_guidance/time-dynamics/test-baseline-20260815.md`. Do not try to fix them. |
| Run commands | `pipelines/time_dynamics/paper_ii_runs.py` is the single source. Never hand-spell a runner invocation. |

Entry command for any trajectory (copy-pasteable, fully guarded):

```bash
PYTHONPATH=. python3 -m pipelines.time_dynamics.paper_ii_runs show --arm exchange --drive fastweak --horizon t2 --output-json output/scratch/run.json
```

That prints the exact shell command; run what it prints.

---

## 2. The task

Add an **optional** Paper-I-style Qiskit backend cost compilation for Paper-II insertion
candidates, in which the cost charged to a candidate is the **minimum over its
commutation equivalence class** rather than the cost of one arbitrary representative.

### Why this is needed — the finding that motivates it

`estimate_append_atom_set_cost` in `pipelines/time_dynamics/ap_mclachlan/append_cost.py`
calls the cost oracle with `position_id=0, append_position=0` **hardcoded**. Inside
`Phase1CompileCostOracle.estimate` (`pipelines/scaffold/hh_continuation_scoring.py:2108`)
position enters only through `position_shift_span = abs(append_position - position_id)`,
so that term is **always exactly 0**.

Consequences, both of which must be preserved as true statements:

1. The current proxy is **position-blind**. Every position in a commutation equivalence
   class scores identically, so choosing the lex-least normal form already *is* a
   least-cost representative under this proxy. The new work must not claim to fix a bug
   in the current route — it adds a cost model that can *distinguish* positions.
2. Under the current proxy, positioned insertion is a **geometry/expressivity** device,
   not a cost-reduction device. Any manuscript claim that positioning reduces hardware
   cost is unsupported by the cost model actually being run. Do not add such a claim; if
   you find one in `MATH/paper_details/time_dynamics_paper_II.tex`, report it, do not
   silently edit it.

### What "equivalence class" means here

`pipelines/time_dynamics/ap_mclachlan/insertion_words.py` already owns the algebra:

- `quotient_insertion_plans(...)` enumerates insertion plans modulo the trace-monoid
  commutation quotient.
- `canonical_word(word, tokens_commute)` returns the lex-least normal form; inserted
  tokens sort before survivors and that ordering is load-bearing — do not change it.
- `tokens_commute_from_terms(...)` supplies the commutation relation.

The equivalence class of an insertion plan is the set of concrete circuit words reachable
by commuting adjacent independent tokens. Today one representative is picked and priced;
the task is to price the class by its cheapest realizable member.

---

## 3. Scope

**In scope**

- `pipelines/time_dynamics/ap_mclachlan/append_cost.py`
- a new module for the class-minimization, e.g. `pipelines/time_dynamics/ap_mclachlan/equivalence_class_cost.py`
- `pipelines/time_dynamics/ap_mclachlan/insertion_words.py` — **read-only** unless you need
  a new pure enumeration helper; if so, add one, do not alter `canonical_word` ordering
- `pipelines/time_dynamics/paper_ii_runs.py` — register the new option
- `test/test_ap_mclachlan_*.py`, `test/test_paper_ii_runs.py`

**Out of scope, with reasons**

- `pipelines/static_adapt/**` — Paper I lane; reuse `hh_backend_compile_oracle.py` and
  `qiskit_backend_tools.py` by *calling* them, do not modify them. Another agent commits
  there.
- `pipelines/excited_dynamics/**`, `pipelines/qse_spectra/**` — Paper III lane.
- The selector's scoring formula and the exchange/deletion logic. This task changes how a
  candidate's cost is *computed*, never how patches are chosen. `U_ins + U_del + w·δ`
  stays as it is.
- Manuscript edits. Report findings; do not edit the `.tex`.

**Shared-resource limits**

- Hard **10 GB** ceiling for agent-launched processes; the machine is shared and has been
  crashed before. One heavy local job at a time.
- No CHTC submission — cluster access needs the user's interactive Duo login.
- Qiskit transpilation is expensive. Default the new path **off**; when on, it must be
  memoized per `(class signature, backend target)`.

---

## 4. Increments

Proceed through all five autonomously; commit at each boundary. Stop and report only on a
stop condition.

### Increment 1 — expose the position dependence that already exists

**Goal:** `estimate_append_atom_set_cost` accepts real positions instead of hardcoded
zeros, with default behavior byte-identical to today.

Add optional `position_id` / `append_position` parameters threaded from the candidate's
actual cut. Default them to `0, 0`.

**Expected result:** lane baseline still 295 passed; `test_ap_mclachlan_route_parity.py`
unchanged. A new test asserts that with equal positions the estimate is bit-identical to
the pre-change value.

**Stop condition:** if route parity moves at all, the defaults are not neutral — stop.

### Increment 2 — enumerate the equivalence class

**Goal:** a pure function returning the realizable positions of an insertion plan within
its commutation class.

Build on `quotient_insertion_plans` / `tokens_commute_from_terms`. The function must be
deterministic and total; the class of a plan with no commuting neighbours is the singleton
containing itself.

**Expected result:** unit tests covering (a) singleton class, (b) a plan with two
independent tokens giving exactly the expected number of orderings, (c) determinism across
repeated calls, (d) the lex-least member equals `canonical_word`.

**Stop condition:** if class size can grow without bound on a real HH seed, cap it, record
the cap in telemetry, and report the observed distribution — an uncapped enumeration inside
a per-candidate scoring loop is how this lane produced hour-long checkpoints before.

### Increment 3 — least-cost realization under the proxy

**Goal:** `cost(class) = min over members` under the existing `Phase1CompileCostOracle`.

**Expected result:** on the HH snake seed the minimum differs from the canonical
representative's cost for at least some candidates, **or** you demonstrate it never does.
Either outcome is a real result — report the number of candidates where the minimum is
strictly below the representative, and the distribution of the gap. Per the contract, a
"never fires" claim needs a measurement, not an argument.

**Stop condition:** if the minimum never differs, do not proceed to Increment 4 before
reporting — it would mean the proxy is position-blind in a deeper way than the hardcoded
zeros, and the Qiskit path may be the only thing that can distinguish positions.

### Increment 4 — optional Qiskit backend compilation

**Goal:** a `--append-cost-model {proxy,qiskit_least_cost}` option that prices the class by
real transpiled gate counts.

Reuse Paper I's machinery by calling it: `BackendCompileOracle` and
`MarrakeshGraphSpanCostOracle` in `pipelines/static_adapt/hh_backend_compile_oracle.py`,
and `compile_circuit_for_backend` / `compiled_gate_stats` / `resolve_backend_targets` in
`pipelines/qiskit_backend_tools.py`. Match Paper I's backend defaults and record the
resolved backend target in run provenance.

Requirements:

- Default **off**. `proxy` remains the default cost model.
- Memoize on `(class signature, backend target)`; transpilation must not run twice for the
  same class in one trajectory.
- Record in the decision payload: cost model id, backend target snapshot, class size,
  representative cost, least cost, and the index of the chosen member.
- If Qiskit or the fake backend is unavailable, **fail closed** with a clear error. Do not
  silently fall back to the proxy — a comparison that silently changed cost models is
  exactly the class of failure the run locks exist to prevent.

**Expected result:** a short run at `--horizon smoke` completes under both cost models;
provenance shows the model actually used; wall time under the Qiskit model is reported.

**Stop condition:** if per-checkpoint wall time exceeds ~10× the proxy, stop and report
before running anything longer — this option is meant to be usable, not merely correct.

### Increment 5 — register and lock

**Goal:** the option is reachable only through the run registry, and drift is caught by a
test.

Add the cost model to `pipelines/time_dynamics/paper_ii_runs.py` as a named part, and
extend `test/test_paper_ii_runs.py` so every registered combination parses against the
runner's live parser.

**Expected result:** `test_paper_ii_runs.py` covers the new option; lane baseline ≥ 295 + new
tests, all passing.

---

## 5. Settled decisions — do not reopen

- **Append is not this paper's contribution.** The insertion condition is being aligned
  with the published AVQDS rule (Yao et al., PRX Quantum 2, 030307) on purpose. Do not
  propose a novel append heuristic.
- **The conditioning gate stays off.** Measured: no setting is useful — 5e7 never binds
  against observed κ ≈ 1e8, 3e7 costs accuracy (1.15e-3 vs 7.79e-4), 1e7 rejects all 600
  candidates and starves the ansatz. Do not re-enable it as part of cost work.
- **Candidate pool cap 128, above the 125-word deduplicated pool**, so it does not bind. A
  cap of 8 was the largest single accuracy defect found in this audit.
- **Subdivision budget 10.** At 4 a step could exhaust its budget, fail to cure a
  state-motion cap violation, and advance unsubdivided anyway; that took a measured HH
  energy error from 3.8e-3 to 2.1e-1.
- **The AVQDS comparator is uncapped.** Capping appends per checkpoint is a different
  algorithm and misrepresents it.

## 6. Traps specific to this task

- `position_shift_span` is currently identically zero. If your change makes it nonzero,
  every existing cost number moves. That is intended under the new model and **forbidden**
  under the default one — hence the byte-identical requirement in Increment 1.
- Cost enters the selector score multiplied by `--append-cost-alpha` (default 1.0) and
  normalized by `--append-cost-normalization-mode family_robust_v1`. A change in cost scale
  silently rescales selection. Report the normalization statistics before and after.
- `output/` and `prompt-exports/` are gitignored. This handoff lives in
  `agent_guidance/time-dynamics/` precisely so it is tracked.
- Another agent's broad `git add` can sweep your uncommitted work into their commit.
  Commit at every increment boundary.

## 7. Context (optional reading)

The accuracy work that led here, in case a number above needs its provenance: commits
`bc81c1c8` (subdivision budget + conditioning gate), `01e74fe1` (accumulated-drift
escalation, off by default), `c5d6bb23` (AVQDS append condition adopted as an option),
`e3852b36` (run registry). Route contract: `agent_guidance/time-dynamics/AGENTS.md`.
Mathematics: `prompt-exports/paper_ii_noiseless_conditional_exchange_implementation_spec.md`.

## 8. Report back

One block per increment in the contract's format, including commit SHAs. For Increment 3,
the measured distribution of (representative cost − least cost) is the deliverable, not a
description of it.
