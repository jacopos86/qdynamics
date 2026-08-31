# Handoff — L=3 pure-Hubbard RA appendix on the Paper-I append-RA route

Goal-mode task. Everything below is verified against the repo unless a line
says otherwise.

---

## 0. Anchors

| item | value |
|---|---|
| checkout | `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/appendix-demos-20260820` |
| branch | `appendix-demos-20260820` |
| HEAD | `cde4253949f43aeaa0c11bce6bf7982de2a0c1c3` |
| python | `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` |
| do NOT use | `/Users/jakestrobel/Documents/Holstein_implementation/...` (iCloud mirror) |

Other agents may be active in the shared checkout: stage commits with
**explicit paths only**, never `git add -A`.

There are **4 modified files uncommitted** in the tree (see §4). They are
work-in-progress toward this task, they compile and partially run, and nothing
else in the repo is broken. Decide whether to build on them or `git checkout --`
them; do not leave them half-applied.

Run everything locally through the allowlisted localhost remote-runner
(`~/.remote-runner`, `curl -fsS -X POST http://127.0.0.1:8765/api/jobs/<id>/run`).
**No CHTC.** Do not touch retired SNAKE routes or the manuscript repo.

---

## 1. The goal

Establish, on the **Paper-I append-RA route with its 3-metric compiled cost
live**, whether prune / batch / beam beat minimal RA on the pure-Hubbard L=3
point — then extend to noise, then AAVQE.

Ordered, do not reorder:

1. **B.** Re-run all L=3 RA arms (minimal, prune, batch, beam) to **iteration
   15**, one regime, on the correct route.
2. **C.** Cost terms — all three metrics `(N2q, D2q, Dc)`:
   - **beam**: total Qiskit compiled cost of one branch vs another
     (branch-level total, not per-candidate marginal);
   - **batch**: marginal batch compiled cost (straightforward);
   - **prune**: select generators in the ansatz by the 3-metric weighted cost.
3. **D.** Escalate/tune prune, batch, beam algorithmic parameters until the
   advantage is demonstrated. Prune **must keep** the property that it does not
   pay for delete-and-reject retaliation (see §6).
4. **E.** Promote to the Paper-I appendix.
5. **F.** Same L=3 regime **under noise to iteration 20**, conventional
   append-RA as the baseline.
6. **G.** AAVQE on the same L=3 physics, **noiseless and under noise**, as the
   benchmark baseline for all appendix plots. A drafted AAVQE handoff exists at
   `chtc/ra_adapt_appendix_demos_20260820_v8/HANDOFF_AAVQE_L3_20260820.md` —
   its §4 was corrected once already; re-verify its protocol table against §2
   before using it.
7. **H. Deferred** (probably never, likely too much appendix clutter): RA
   insertion and RA insertion-plateau across all these routes.

---

## 2. The route — verified, use exactly this

Backtracked from the artifact, not inferred: `bundle3_final_results.pdf` →
`chtc/paper_i_ra_adapt_repair_20260727/refresh_paper_i_ra_comparison_20260817.sh`
→ staging campaign `paper_i_ra_allphase_adaptive_three_arm_maximum_k50_20260817_v7`
→ `chtc/paper_i_ra_adapt_repair_20260727/paper_i_ra_allphase_adaptive_three_arm_maximum_k50_20260817_v7_chtc/jobs/allphase_maxk50__append_ra__intermediate_weak__nph3.json`.

That job spec says:

```
arm_id             = append_ra
builder            = build_paper_i_ra_all_phase_adaptive_natural_terminal_request
route_constant     = PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2
route_variant      = gradient_only_adaptive_shortlist_phase123_adaptive_natural_terminal_v2
insertion_policy   = append_only
phase0_population  = append_endpoint_generators
phase123_population= append_records
fresh_start        = true
```

The materializer (`.../v7_chtc/materialize_package.py`, ~line 160) runs:

```python
problem  = build_paper_i_ra_hh_regime_problem(regime_id)
request  = builder(insertion_policy="append_only",
                   maximum_controller_rounds=TARGET_HORIZON)   # NO compile_cost_policy
protocol = materialize_paper_i_ra_semantic_protocol(problem, request)
```

then `run_ra_adapt(problem, protocol)`.

**Confirmed working**, unpatched, 3 rounds, on HH `intermediate_weak`. Use this
shape. Two traps:

- Do **not** pass `compile_cost_policy` / `ExplicitDepthQiskitCompileCost()`.
  Paper I passes none; the 3-metric cost enters through the route's Phase I–III
  Qiskit scoring. The `ra_adapt_appendix_demos` lane passes it, which is why
  that lane is not the Paper-I route.
- Do **not** hand-build `RAAdaptRequest`. Use the builder.

The route requires a **materialization authority** — it is bundle-only by
design; `run_ra_adapt(problem, request)` will refuse with *"Study policies
require a validated bundle protocol materialization authority."*
`materialize_paper_i_ra_semantic_protocol` is the in-memory seam that mints it
(no on-disk bundle, no CHTC).

---

## 3. Read this before touching the cost term

Paper I §III B (`/Users/jakestrobel/local_repos/ADAPT---Paper-I/Paper_I_overleaf_20260818/Paper_I.pdf`):

```
S^(t)(r) = ΔE_t(r) / K_t(r) ,        K_t(r) = 2 / (2 − C̄_λ(r;t)) ,   C̄_λ ∈ [−1, 1]
```

C̄ is median-centred and MAD-scaled through arctan
(`hardware_cost_normalization_mode = family_robust_symmetric_arctan_v1`), so:

- **K_t = 1 exactly at median burden** (neutral),
- K_t → 2/3 for the cheapest records (boosts score),
- K_t → 2 for the most expensive (suppresses score).
- Phase 0 is scored at unit cost, `K_0(r) = 1`, by definition.

In code, `K3 = burden − 1.0`, i.e. `predictive_cost.value = selector_burden =
phase3_burden_total = 1 + K3`.

**A previous agent (me) misread `burden = 1.0` as "the compiled cost is
inert / K3 ≡ 0" and re-routed the whole task on it. That was wrong.** 1.0 is
the neutral point of a *relative* factor. Reading the selected record's burden
tells you almost nothing; if you want to know whether cost discriminates, you
must measure the **spread of K_t across the scored population in a phase**, not
the winner's value. I did not succeed in capturing that — `scientific_receipts`
does not carry K components, and `build_full_candidate_features` was the wrong
hook. Find where `phase_score_components["K3"]` is populated
(`pipelines/scaffold/hh_continuation_scoring.py` ~line 6068) and instrument
there, or add the population statistic to a receipt.

---

## 4. What is already built (uncommitted, 4 files, +219/−11)

The singleton semantic-closure family is **singleton-only** and at L=3 its
guarded single-Pauli pool is degenerate: 9 diagonal `Z` words, zero
Hartree–Fock gradient and tangent self-metric, supported response rank 0, hard
controller crash. So L=3 needs a **macro** variant. This is the documented gap
in `chtc/paper_i_ra_adapt_repair_20260727/CAMPAIGN_STATE_20260817.md` §B
("GAP (route work required)").

What is in the tree now:

| file | change |
|---|---|
| `semantic_closure_routes.py` | `PAPER_I_RA_MACRO_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2` token + identity record (macro representation, `sector_guarded_macro_pool_v1` population); `PaperIRASemanticClosureRouteIdentity` gains `candidate_representation` / `candidate_population`; adapter defaults representation from the identity and dispatches `parent_inventory` / `executable_pool` to the appendix macro pool for `family_key == "hubbard"`; `application_family_key` property; macro variant joined to the phase-0 and variant frozensets (6 set-literal sites); pure-Hubbard regime label `pure_hubbard_l3_u1p25` admitted |
| `engine.py` | `is_paper_i_ra_macro_semantic_pure_hubbard_application` wired into **both** pure-Hubbard facade gates |
| `bundles.py` | `materialize_semantic_closure_protocol` takes the representation from the adapter instead of hard-coding singleton |
| `paper_i_run_summary.py` | semantic identity rebuild accepts macro; family and size gates accept `named_semantic_application` |

Verified reachable with these: protocol builds with
`route_profile = paper_i_ra__macro__phase0_gradient_adaptive__phase123_adaptive__qiskit_signed__natural_terminal__semantic_closure_v2`,
`phase3_backend_cost_scope = phase0_proxy_or_off_phase_i_phase_ii_phase_iii_qiskit_transpile_v1`,
`phase_iii_compile_cost_source = backend_transpile_v1`, 24-generator macro
pool, and the materialization authority mints.

**Current blocker — localized precisely:**

```
RuntimeError: Semantic Phase-0 receipt failed decision recomputation.
```

Root cause, traced (do not re-derive):

- raised from `validate_semantic_proxy_phase0_receipt`
  (`semantic_closure_routes.py` ~2466) → `build_semantic_proxy_phase0_receipt`
  (~2018) → `_normalize_semantic_phase0_rows` (~1358),
  message *"Append-endpoint scored population identity is invalid."*
- the failing condition is `not normalized`: instrumented, the normalized row
  list is **empty — n = 0**, with no duplicate indices, no duplicate generator
  ids, and no append positions.

So the append-endpoint **Phase-0 population produces zero rows for macro
records**. It is not a validator allowlist and not a receipt-format mismatch.

Ruled out already:

- the macro Phase-0 *policy* is correct and identical to singleton —
  `_phase0_policy` returns `population = same_ordered_append_endpoint_generator_population_v1`,
  `graph_proxy_cost = off`, `score = absolute_append_endpoint_generator_gradient_v1`;
- the macro variant is **not** in `PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS`
  (so it is not misrouted to the position family);
- the 24 macro pool records have unique labels and unique generator identities.

**The remaining work is to write the append-endpoint Phase-0 population
producer for macro records.** The singleton producer emits one row per guarded
single-Pauli `child:` record at the append endpoint; macro records are intact
`parent:` generators with no `child:` splits (the campaign doc notes "the bug-5
accounting simplifies" for exactly this reason), and nothing currently emits
rows for them. Find the producer that feeds
`_normalize_semantic_phase0_rows` and give it a macro path emitting
`pool_index / generator_id / pool_label / append_position /
append_gradient_signed` per intact generator at the single append endpoint.

Treat every patch listed in the table above as **provisional**. In particular,
the macro variant was added to six set literals by a scripted edit that
appended it after each occurrence of the singleton natural-terminal variant;
that was blunt, and each of those six sites should be reviewed individually.
The campaign doc asks for "receipt/validator audit … tests, then G2/H2-style
local smoke through finalization" — none of that audit has been done.

---

## 5. The physics point (one regime, fixed)

`build_ra_adapt_appendix_pure_hubbard_problem(num_sites=3)` in
`pipelines/static_adapt/ra_adapt/appendix_demos.py`. Do not hand-build the
`ProblemRequest`.

| parameter | value |
|---|---|
| problem_key | `hubbard` (pure Hubbard, **not** `hh`) |
| num_sites / t / **U/t** | 3 / 1.0 / **1.25** |
| g_ep, omega0, n_ph_max | 0.0, 0.0, 0 — **no phonons** |
| boundary / ordering / n_fermions | `open` / `blocked` / 3 |
| sector | `(2, 1)`, 6 qubits, all fermionic |
| E_exact (same-cutoff ED) | `-2.1414935642714075` |
| Hartree–Fock ref | `|001011>`, E = `+1.25` |

One regime for every arm. There is **no** `nph` axis and **no** HH regime
matrix here — `weak_weak / intermediate_weak / strong_weak_u8 / weak_strong /
intermediate_strong / strong_strong_u8` belong to Bundle 3's Hubbard–Holstein
lane and are out of scope. The exact reference is reporting-only and must never
enter the controller stopping rule.

**Pool:** `build_ra_adapt_appendix_pure_hubbard_sector_guarded_macro_pool(problem)`
— 24 intact multi-Pauli generators, `macro_generator_v1`, measured
fixed-count sector receipts with zero leakage, 12 genuinely off-diagonal
directions, Lie closure u(9) on the 9-dimensional (2,1) sector (universal
there). Same pool for every arm, AAVQE included.

---

## 6. Prune: the defect to avoid

A prune *nomination* is cheap. What follows it is not: the route forks an
immutable sibling with the generator deleted, runs a **complete Powell refit**
on the survivors, and compares. On the old lane, rounds 3–10 each paid for one
of those and rejected it — measured regressions 2.5e-3 up to 1.4, all far above
`phase1_prune_max_regression = 1e-8`. Eight full refits bought nothing.

The fix already found: screen nominees by angle magnitude **before** paying.
`MetricPruning(protect_steps=1, small_theta_abs=1e-3)`. Load-bearing generators
carry |θ| ∈ [0.14, 1.42]; the numerically null ones sit at ~1e-9; the screen
sits between. A round with no qualifying coordinate returns the existing
`not_executed` receipt via `_no_nominee`, which already records both
`prune_nomination` and `prune_verification`, so the transition kernel's fixed
operation order survives.

Also known: `phase1_prune_protect_steps` (canonical 2) is an admission-protection
window — a freshly admitted generator cannot be nominated for that many rounds.
`protect_steps=0` is **not reachable**: with no window prune nominates the only
active generator in round 1 and deleting it leaves an empty ansatz with no
supported metric subspace. 1 is the minimum runnable value.

These knobs exist on the **old lane's** `MetricPruning` (committed at
`abfb9ccb`). Re-establish the equivalent behaviour on the Paper-I route; per §1
step C, prune's generator selection should use the 3-metric weighted cost.

---

## 7. Superseded evidence — do not reuse, do not delete

`results_l3_v6`, `results_l3_v7`, `results_l3_v8` and the six-arm figures are
all on the **RA v2 lane** (`paper_i_ra_adapt_nonstationary_incremental_full_response_v2`),
not the Paper-I append-RA route. They are internally consistent controlled
comparisons but are not the requested route. Treat as superseded; keep them.

For orientation only, the v8 terminal values at round 15 (S_alg **branch-local**
— see below):

| arm | ops | err@15 | N2q | D2q | Dc | S_alg |
|---|---|---|---|---|---|---|
| minimal RA | 15 | 7.11e-15 | 127 | 109 | 441 | 9 547 |
| RA + prune (gated) | 9 | 8.88e-16 | 112 | 93 | 378 | 10 658 |
| RA + beam | 15 | 4.44e-16 | 128 | 103 | 413 | 9 646 |
| RA + batch | 16 | 8.44e-15 | 229 | 206 | 853 | 11 823 |

**S_alg is branch local.** Do not read
`estimator_ledger_receipt.cumulative_executed_queries.S_alg` — on a branching
run that is the global all-branch counter and overstates the delivered ansatz
(40 162 for a beam arm whose winning lineage spent 9 645). Accumulate per-round
`executed_query_delta.S_alg` along the lineage and subtract
`typed_receipt.trial_s_alg` for any prune round with status `rejected`. This
reproduces the recorded `winning_lineage_s_alg` exactly and is identical to the
cumulative counter on non-branching arms.

---

## 8. Reporting

Per accepted round: `controller_round`, `accepted_energy`,
`same_cutoff_absolute_error`, `active_ansatz_depth`, admitted generator labels,
and the compiled tuple `(N2q, D2q, Dc, N1q, W1q, S_alg)` — FakeMarrakesh,
`optimization_level=1`, `seed_transpiler=7`. Plus: first round reaching 1e-2 /
1e-4 / 1e-6; admitted and pruned generators; batch membership; beam
child/survivor accounting.

Figures: one one-column PNG+PDF per extension vs minimal RA, log error vs
accepted round, colour-matched tuples embedded at **both** the matched-accuracy
round and the terminal round (with only the first, prune and beam read as
identical to baseline, which is true at round 8 and false by round 15). Reuse
`chtc/ra_adapt_appendix_demos_20260820_v8/plot_l3_v8_appendix_convergence.py`.

Use a fresh immutable campaign version/job/runtime (v9+). Preserve v1–v8.

---

## 9. Test baseline

```bash
cd /Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/appendix-demos-20260820 && python3 -m pytest test/test_ra_adapt_appendix_l3_generator_pool.py test/test_ra_adapt_appendix_campaign_runner_l3_v8.py test/test_ra_adapt_prune_protection_ablation.py -q -p no:randomly
```

Green at HEAD. **55 failures elsewhere in the suite are pre-existing** —
`test_ra_adapt_bundles.py`, `test_ra_adapt_campaign.py`,
`test_ra_adapt_append_facade.py`, `test_ra_adapt_facade.py`,
`test_ra_adapt_phase3_incremental_gain.py`,
`test_ra_adapt_semantic_closure_routes.py`,
`test_ra_adapt_beam_metric_plateau_composition.py` and others — they reproduce
identically at the pre-existing baseline commit. Do not fix them. **Do** re-run
`test_ra_adapt_semantic_closure_routes.py` after §4 work and separate any new
failure from the pre-existing ones.

---

## 10. Report back

Per step: status, exact command, observed result, how verified, deviations with
reasons, commit SHAs. Report negative outcomes plainly — an extension failing
to beat minimal RA is a publishable result; a suspiciously good number is more
likely a protocol mismatch. Anything that contradicts this document matters
more than a clean run.
