# Handoff — AAVQE comparator on pure-Hubbard L=3

Run the AAVQE (Append-ADAPT) comparator on the one L=3 point the RA appendix
arms already use, under the same pool and optimizer protocol, so it can be
plotted against them.

This is a single-regime task. There are no Hubbard--Holstein regimes here, no
phonons, and no `nph` axis. If you find yourself iterating over
`weak_weak / intermediate_weak / strong_weak_u8 / ...`, you are in the wrong
campaign — those are Bundle 3's Hubbard--Holstein regimes and are out of scope.

`agent_guidance/shared/agent-handoff-contract.md` is referenced by the handoff
skill but **does not exist in this checkout**. Do not invent it. The anchors,
evidence standards, traps and report-back format you need are inlined below.

---

## 1. Anchors

| item | value |
|---|---|
| checkout | `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/appendix-demos-20260820` |
| branch | `appendix-demos-20260820` |
| commit | `cde4253949f43aeaa0c11bce6bf7982de2a0c1c3` |
| python | `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` |
| do not use | `/Users/jakestrobel/Documents/Holstein_implementation/...` (iCloud mirror) |

Work in this worktree on this branch. Other agent sessions may be active in
the shared checkout, so stage commits with **explicit paths only** — never
`git add -A`.

Test baseline before you start (these must be green; if they are not, stop and
report):

```bash
cd /Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/appendix-demos-20260820 && python3 -m pytest test/test_ra_adapt_appendix_l3_generator_pool.py test/test_ra_adapt_appendix_campaign_runner_l3_v8.py test/test_ra_adapt_prune_protection_ablation.py -q -p no:randomly
```

Known pre-existing failures elsewhere in the suite (**not yours, do not fix**):
`test_ra_adapt_bundles.py`, `test_ra_adapt_campaign.py`, `test_ra_adapt_append_facade.py`,
`test_ra_adapt_facade.py`, `test_ra_adapt_phase3_incremental_gain.py`,
`test_ra_adapt_semantic_closure_routes.py`, `test_ra_adapt_beam_metric_plateau_composition.py`
and others — 55 failures reproduce identically at the pre-existing baseline commit.

---

## 2. The physics point (one regime, fixed)

Built by `build_ra_adapt_appendix_pure_hubbard_problem(num_sites=3)` in
`pipelines/static_adapt/ra_adapt/appendix_demos.py`. Do not construct the
`ProblemRequest` by hand; the builder validates a source lock.

| parameter | value |
|---|---|
| problem_key | `hubbard` (pure Hubbard, **not** `hh`) |
| num_sites | 3 |
| t | 1.0 |
| **U/t** | **1.25** |
| dv, v_nn, t_prime | 0.0 |
| g_ep, omega0, n_ph_max | 0.0, 0.0, 0 (no phonons) |
| boundary / ordering | `open` / `blocked` |
| n_fermions | 3 |
| sector `num_particles` | `(2, 1)` |
| register | 6 qubits, all fermionic |
| E_exact (same-cutoff ED) | `-2.1414935642714075` |
| Hartree--Fock reference | `\|001011>`, E = `+1.25` |

The exact reference is **reporting only** and must never enter the controller
stopping rule.

---

## 3. The candidate pool (identical to the RA arms)

`build_ra_adapt_appendix_pure_hubbard_sector_guarded_macro_pool(problem)` in
`pipelines/static_adapt/ra_adapt/pools.py` — **24 intact multi-Pauli
generators**, `candidate_representation = macro_generator_v1`.

Why not single-Pauli: at L=3 the guarded single-Pauli exposure is
sector-degenerate. Its nine survivors are diagonal `Z` words whose
Hartree--Fock gradients and tangent self-metrics vanish identically, giving
supported response rank 0 and a hard controller crash. The macro pool is the
same construction `build_executable_macro_pool` uses for the HH macro route:
audit parent templates against the sector contract, drop what the `hard_guard`
parent filter rejects, attach a measured per-generator fixed-count sector
receipt. It has 12 genuinely off-diagonal directions and Lie closure u(9) on
the 9-dimensional (2,1) sector, so it is universal there.

**AAVQE must use this same pool.** That is the "equal pool protocol"
requirement. Pass the appendix adapter at macro representation (§5).

---

## 4. What AAVQE is here

AAVQE in this repo is **Append-ADAPT**: `run_append_adapt` with an
`AppendAdaptRequest`. Precedent to copy:
`chtc/paper_i_ra_adapt_repair_20260727/paper_i_aavqe_r50_to_r75_continuation_20260819_v1_chtc/worker.py`
(that one is the HH six-regime r75 continuation, cluster 9665004 — **use it as
a code pattern only**, not as scope).

Measured identity of the resolved append protocol
(`build_resolved_append_protocol`, `pipelines/static_adapt/ra_adapt/append.py`):

| field | value |
|---|---|
| `algorithm_id` | `paper_i_append_adapt_v1` |
| `selector_identity` | `append_adapt_largest_absolute_commutator_gradient_v1` |
| `selector_scope` | `conventional_append_no_phase3_no_trust_v1` |
| `resource_weighting_scope` | `all_phase_resource_weighting_v1` |

Note the append facade is source-locked: `_require_paper_i_problem` admits only
the Hubbard--Holstein L=2 problem plus the named L=3 Page-12 weak-sector
comparator. The pure-Hubbard L=3 point is **not** currently admitted, and
opening that seam is the first real piece of work in §7.

One measured fact about the RA side you should carry into the comparison
rather than rediscover: the RA appendix arms resolve
`algorithm_id = paper_i_ra_adapt_nonstationary_incremental_full_response_v2`,
and on that lane the Phase-III predictive cost `(1 + K3)` is exactly 1.0 in
every round. The three-term `K = 0.30 dN2q + 0.30 dD2q + 0.25 dDc` is declared
in the route contract and reported in the accepted-prefix tuples on both sides.
If you need to characterise how cost enters either selector, measure it on the
run you produce — do not infer it from a contract dataclass's fields.

## 5. Exact protocol for the AAVQE run

Match the RA arms on everything the comparator can match.

| setting | value | source |
|---|---|---|
| problem | `build_ra_adapt_appendix_pure_hubbard_problem(num_sites=3)` | §2 |
| adapter | appendix adapter at **macro** representation | §5 snippet |
| pool | sector-guarded macro, 24 generators | §3 |
| horizon | **15** controller rounds | matches v8 |
| inner optimizer | Powell, `maxiter=200` | canonical Paper-I |
| seeds | 7 | canonical |
| warm start | **off** — new amplitudes start at zero, existing inherited | fixed decision |
| resume | `FreshStart()` | fixed decision |
| compile receipts | FakeMarrakesh, `optimization_level=1`, `seed_transpiler=7` | matches v8 |
| noise | none (noiseless) | this task |

Construct the adapter exactly like this — the representation must be macro or
you will silently get the degenerate nine-word pool:

```python
from pipelines.static_adapt.ra_adapt.appendix_demos import (
    RAAdaptAppendixPureHubbardCandidateAdapter,
    build_ra_adapt_appendix_pure_hubbard_problem,
    ra_adapt_appendix_generator_representation,
)
from pipelines.static_adapt.ra_adapt.contracts import AppendAdaptRequest
from pipelines.static_adapt.sr_snake.contracts import (
    SRExecutionPolicy, SRStopPolicy, FreshStart,
)

problem = build_ra_adapt_appendix_pure_hubbard_problem(num_sites=3)
representation = ra_adapt_appendix_generator_representation(3)   # macro_generator_v1
adapter = RAAdaptAppendixPureHubbardCandidateAdapter(
    candidate_representation_id=representation
)
request = AppendAdaptRequest(
    adapter=adapter,
    execution=SRExecutionPolicy(
        stop=SRStopPolicy(maximum_controller_rounds=15),
        resume=FreshStart(),
    ),
)
```

Assert before running: `adapter.executable_pool(problem).receipt.count == 24`
and `.candidate_representation == "macro_generator_v1"`.

Expect friction: `run_append_adapt` and its validators have only been exercised
on the singleton representation and on HH problems. If a validator rejects the
macro appendix adapter, that is ordinary implementation work — repair it
test-first, keep the fix scoped to the append lane, and record what you
changed. Do **not** work around it by falling back to the single-Pauli pool;
that pool is degenerate at L=3 and the run would be scientifically void.

---

## 6. What to compare against

The six RA arms are already published and immutable at
`output/local_runs/ra_adapt_appendix_demos_l3_20260820_v8/cells/` with the
tracked report at `chtc/ra_adapt_appendix_demos_20260820/results_l3_v8/`.

Terminal values at round 15, for cross-checking your AAVQE row:

| arm | ops | err@15 | N2q | D2q | Dc | S_alg (branch-local) |
|---|---|---|---|---|---|---|
| minimal RA | 15 | 7.11e-15 | 127 | 109 | 441 | 9 547 |
| RA + prune (gated) | 9 | 8.88e-16 | 112 | 93 | 378 | 10 658 |
| RA + beam | 15 | 4.44e-16 | 128 | 103 | 413 | 9 646 |
| RA + batch | 16 | 8.44e-15 | 229 | 206 | 853 | 11 823 |

Threshold rounds for minimal RA: 1e-2 at 7, 1e-4 at 8, 1e-6 at 8. Batch reaches
all three at round 3.

**S_alg is branch local.** Do not read
`estimator_ledger_receipt.cumulative_executed_queries.S_alg` — on any branching
run that is the global all-branch counter and overstates the delivered
ansatz's cost (it reads 40 162 for the beam arm whose winning lineage actually
spent 9 645). Accumulate the per-round
`estimator_ledger_receipt.executed_query_delta.S_alg` along the lineage, and
subtract `typed_receipt.trial_s_alg` for any prune round whose status is
`rejected`. Append-ADAPT does not branch, so for AAVQE the two agree — but use
the branch-local rule anyway so the column is defined identically across arms.

---

## 7. Increments

Each increment ends with a checkable result. Proceed autonomously through 1--3;
**pause and report** before 4 if any validator required a code change.

**1. Smoke.** Run AAVQE for 2 controller rounds. Expected: completes, admits 2
generators, energy strictly decreases from +1.25, pool count 24.

**2. Full run.** Horizon 15. Expected: 15 accepted rounds, monotone
non-increasing energy, terminal error at or below ~1e-8 (minimal RA reaches
9.3e-10 by round 8; AAVQE is not expected to beat it, and a worse number is a
legitimate result — report it, do not tune toward a target).

**3. Evidence + costs.** Emit per round: `controller_round`, `accepted_energy`,
`same_cutoff_absolute_error`, `active_ansatz_depth`, admitted generator label,
and the compiled tuple `(N2q, D2q, Dc, N1q, W1q, S_alg)` under the §5 compile
settings, with S_alg branch-local per §6. Validate every accepted prefix
compiles and that stored `N2q/D2q/Dc` close against a recompilation.

**4. Figure.** One one-column PNG+PDF, minimal RA vs AAVQE, same style as
`chtc/ra_adapt_appendix_demos_20260820_v8/plot_l3_v8_appendix_convergence.py`:
log error vs accepted round, colour-matched `(N2q,D2q,Dc,N1q,W1q,S_alg)` tuples
embedded at the matched-accuracy round and the terminal round. Reuse that
script's helpers rather than writing new plotting code.

---

## 8. Out of scope (deliberately)

- **The six HH regimes / Bundle 3 / cluster 9664325.** Different campaign,
  different problem family.
- **Making K3 non-zero.** Requires a macro semantic-closure adapter and new
  route variants; the gap is already written up in
  `chtc/paper_i_ra_adapt_repair_20260727/CAMPAIGN_STATE_20260817.md` §B
  ("GAP (route work required)"). Do not attempt it inside this task.
- **v1--v7 campaigns and `results_l3_v6` / `results_l3_v7`.** Preserved
  evidence; read-only.
- **CHTC.** This is a local run. Do not submit, and do not touch retired SNAKE
  routes or the manuscript repository.
- **Re-running the RA arms.** They are done and immutable.

---

## 9. Report back

Per increment: status, exact command, observed result, how you verified it,
and any deviation from this document with the reason. Then:

- the AAVQE row in the §6 table format, with S_alg branch-local;
- first round reaching 1e-2, 1e-4, 1e-6 (or "not reached");
- every code change you made, with commit SHAs and why it was needed;
- anything that contradicts this handoff — that matters more than a clean run.

Report negative outcomes plainly. AAVQE losing to RA is the expected result and
is publishable; a suspiciously good AAVQE number is more likely a protocol
mismatch than a discovery.
