# Paper II dual-selector implementation — Fable report (2026-09-02/03)

Executor: Fable (Claude Code), executing
`docs/plans/2026-09-02-gitnexus-plan-paper-ii-dual-selector.md`. Scope ends at
code, tests, local smokes, benchmark, and this report. No CHTC submission, no
evidence adoption, no manuscript edit, no PDF.

## 1. Outcome

- **Three closed route bundles exist and run end to end** on the real HH seed:
  `paper_ii_exchange_legacy_proxy_fixed_v1`, `paper_ii_exchange_q_compiled_v1`,
  `paper_ii_exchange_gamma_compiled_v1`. Each is `EXCHANGE` plus one
  `--exchange-route-bundle`; objective, cost model, kernel, and compile
  contract are fixed by `exchange_config.EXCHANGE_ROUTE_BUNDLES` and cannot be
  mixed by hand. The historical route (`paper_ii_generalized_exchange_v2`) is
  the default and untouched — its goldens pass unchanged.
- **The singleton K = 1 defect is reproduced and repaired.** On
  `hh_snake_nph1 × fastweak × smoke` the historical proxy denominator is
  identically 1.0; the repaired legacy control's insertion denominators span
  1.000–1.190 across the competing population.
- **Q and Γ share the identical prefilter population and K map**; only the
  ordinate differs (pinned: equal `population_sha256`, medians, MADs, and
  per-patch K).
- **Realized Q / L² remains final authority**: under accuracy debt a
  certified finalist whose materialized solve does not reduce L² is rejected
  inside the certification loop and the next ranked candidate is tried with
  the sealed population untouched.
- **Parallelism (user request, plan step 9).** Threads give no transpile
  speedup (0.83× on four threads, GIL-bound), so the compile stage fans out
  over a per-checkpoint process pool with deterministic fan-in. On the real
  seed the Q bundle (4 reporting points) runs **81.2 s → 51.9 s → 32.5 s at
  1/2/4 workers (1.56×, 2.50×)** with the decision digest identical across all
  three; the fixture and full-trajectory parity tests pin bitwise identity.
- **Campaign**: `paper_ii_final_matched_cells_v1.json` locks the 15 cells and
  `paper_ii_final_selector_comparison_v1.py` emits exactly 45 full t=0→20 APM
  actions, 0 AVQDS actions, `PREPARED_NOT_SUBMITTED`. The promotion-preview
  and ledger artifacts named by the plan are **absent locally** (SHA hunt
  over `chtc/`, `output/`, and the Codex context directory); every external
  hash, run-lock fingerprint, metric, and paper status is `null`, and the
  adoption statuses are declared from the plan's table, not derived.

## 2. Base, commits, files

Base: `origin/main` = `557f20ae`. The parallelization commits were
reconciled by cherry-pick: `a579bf3e` (batched logical_shared tangent pass,
clean) and `8fd67e28` (deterministic scoring workers; one append-position
conflict in `test_ap_mclachlan_route_parity.py`, resolved by keeping both
sides and de-hardcoding the golden the parity test compared against — main's
golden is `[insert, insert, delete, stay]`, final support 3).

| commit | content |
|---|---|
| `cf6a8bed` | reconcile: main + a579bf3e + 8fd67e28 (206 lane tests) |
| `26e455a4` | `paper_ii_runs.py` `__main__` guard moved after the ported SIAM block (CLI crashed on `LOCKED_STRUCTURAL_POLICY`) |
| `29a6d0ac` | closed route bundles, `SupportPatchControllerConfig` fields, runner flags, registry arms |
| `fe1fa948` | `objective_ordinates.py` (m_Q realized, m_Γ ideal), Γ memoized as fifth solve-memo slot |
| `a0c27ad2` | `population_scoring.py` (Paper-I K, shifted score, regimes, legacy repair helper) |
| `6b74443d` | `compiled_patch_cost.py` (complete-patch FakeMarrakesh compile, class placement, word cache) |
| `43539dd6` | enumeration → seal → rank wiring; realized-L² gate; route telemetry |
| `9bb1029f` | 15-cell lock + 45-action generator + tests |
| `52dde340` | process-pool compile warm with deterministic fan-in |

New files: `pipelines/time_dynamics/ap_mclachlan/{objective_ordinates,population_scoring,compiled_patch_cost}.py`,
`pipelines/time_dynamics/campaigns/paper_ii_final_matched_cells_v1.json`,
`pipelines/time_dynamics/campaigns/paper_ii_final_selector_comparison_v1.py`,
and their test files. Modified: `exchange_config.py`, `adaptive_trajectory.py`,
`exchange_structural.py`, `exchange_selector.py`, `exchange_integration.py`,
`structural_cache.py`, `paper_ii_runs.py`,
`runners/ap_append_from_adapt_artifact.py`, plus tests.

## 3. Mathematics as implemented

Ordinates, for `B = (D, A)`, `J^B = (J \ D) ∪ A`, common denominator
`‖b̄‖² + ε_norm`:

    m_Q = [Q(J^B) − Q(J)] / (‖b̄‖² + ε)      Q from solve_theta_dot under the live policy
                                             (symmetrize → ridge → eigh → rcond → damping),
                                             captured_drift against the RAW G — bitwise the
                                             quantity structural_cache already memoizes
    m_Γ = [Γ(J^B) − Γ(J)] / (‖b̄‖² + ε)      Γ = f^T G^+ f on raw symmetrized G, ridge = damping
                                             = 0, NAMED rcond 1e-12 (numpy's implicit 1e-15 was
                                             undeclared anywhere in the lane)

Ideal Γ under a relative cutoff is not nested-monotone (measured benign floor
6e-14 of ‖b‖², a constructed spectrum shift 1e-2): pure insertions fail closed
at slack < −1e-9 (`GammaMonotonicityError`) rather than clipping.

Cost: complete base and patched circuits at structural θ = 1 (angle
`2·c·θ`, the Paper-I builder), two independent transpiles on FakeMarrakesh /
optimization 1 / seed 7, signed marginals `Δx = x(W^B) − x(W)` for
`x ∈ {N_2q, D_2q, D_c}`;

    C̄_x = (2/π) atan((Δx − median Δx) / max(MAD, 1e-12)),   C̄ = (.30 C̄_N2q + .30 C̄_D2q + .25 C̄_Dc)/.85,
    K = 2/(2 − C̄)  ∈ [2/3, 2]  (singleton and uniform coordinates neutral),

sealed once per certification level over every evaluated candidate before
sign filtering, hashed, and never renormalized after a rejection. Kernel:

    S = (m + τ)/K − τ,   large miss (L² > τ_L²): τ = 0, require m > 0;
                         maintenance: τ = τ_del, stay has S = 0, require S > 0.

Placement: each candidate atom's commutation class (keys are exactly the
`singleton_insertion_cuts` representatives, classes partition the cuts) is
compiled at every member; the member minimizing the weighted raw marginal is
retained (tie-break raw tuple, then lower cut). Paper I retains the earliest
member (`adapt_pipeline.py:13341`, `Paper_I.tex` eq. commutation-reduced
positions); the minimum-cost member is the plan's Paper-II deviation and is
recorded as `paper_ii_min_weighted_raw_marginal_placement_v1`. The
materialized patch uses the placement cut, so the compiled word is the
committed word.

Legacy control: the historical score and debt ordering are unchanged; C_I and
C_D are recomputed over the competing population through `append_cost.py`'s
own `family_robust_v1` arithmetic and enter as exact rescalings of the
C = 1 utilities.

## 4. Measurements

### 4.1 Compile cost and parallel speed-up (real seed)

`hh_snake_nph1 × fastweak × smoke`, `--num-times 4`, BLAS pinned to 1, M4:

| run | wall | real transpiles | cache hits | compile s | warm s | decisions | digest |
|---|---:|---:|---:|---:|---:|---|---|
| Q bundle, 1 worker | 81.2 s | 3,862 | 4,316 | 27.5 | — | stay, stay, insert | `2e361ea8…` |
| Q bundle, 2 workers | 51.9 s | 3,862 | 8,175 | 0.7 | 17.0 | identical | `2e361ea8…` |
| Q bundle, 4 workers | 32.5 s | 3,862 | 8,175 | 0.7 | 10.5 | identical | `2e361ea8…` |
| Γ bundle, 1 worker | 85.8 s | 3,862 | 4,316 | 28.9 | — | stay, stay, insert | `0b96803f…` |
| legacy fixed, 1 worker | ~8 s | — | — | — | — | 12 checkpoints, insert-dominated | — |

A single transpile costs ≈ 7.6 ms; one large-miss checkpoint on this seed
(population 2,149) costs ≈ 3.9 k transpiles ≈ 30 s serial. The digest covers
decision kinds, per-point support, energies, sealed-level ranks, and attempts.
Threads: 4 concurrent transpiles ran at 0.83× of serial (GIL), which is why
the warm uses processes.

### 4.2 Isolated benchmark matrix (plan step 12)

`hh_snake_nph1 × fastweak × smoke`, `--num-times 4`, one job at a time, BLAS pinned
to 1 in every child, phase profiler on, subprocess-isolated (`ap_runtime_benchmark`).
`cpu` is the parent's CPU/wall; under the pool the compile work moves to worker
processes, which is why the parent's utilization drops. `p50/p90/p99` are the
per-call `patch.exchange.compile` phase (12,541 calls per compiled run, cache hits
included); serial they are the transpile itself, pooled they are cache hits.

| case | wall | cpu | parent RSS | compile calls | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| exchange w1 | 1.8 s | 0.94 | 343 MB | 0 | — ms | — ms | — ms |
| exchange w2 | 1.9 s | 1.12 | 345 MB | 0 | — ms | — ms | — ms |
| exchange w4 | 2.1 s | 1.25 | 368 MB | 0 | — ms | — ms | — ms |
| exchange_legacy_proxy_fixed w1 | 1.9 s | 0.94 | 344 MB | 0 | — ms | — ms | — ms |
| exchange_legacy_proxy_fixed w2 | 2.0 s | 1.11 | 359 MB | 0 | — ms | — ms | — ms |
| exchange_legacy_proxy_fixed w4 | 2.1 s | 1.25 | 347 MB | 0 | — ms | — ms | — ms |
| exchange_q_compiled w1 | 83.3 s | 2.12 | 572 MB | 12541 | 7.1 ms | 15.8 ms | 56.2 ms |
| exchange_q_compiled w2 | 49.9 s | 0.15 | 396 MB | 12541 | 0.1 ms | 0.2 ms | 0.7 ms |
| exchange_q_compiled w4 | 31.2 s | 0.27 | 461 MB | 12541 | 0.1 ms | 0.2 ms | 0.7 ms |
| exchange_gamma_compiled w1 | 83.2 s | 2.07 | 576 MB | 12541 | 7.1 ms | 15.8 ms | 56.2 ms |
| exchange_gamma_compiled w2 | 49.9 s | 0.15 | 368 MB | 12541 | 0.1 ms | 0.2 ms | 0.7 ms |
| exchange_gamma_compiled w4 | 31.2 s | 0.27 | 461 MB | 12541 | 0.1 ms | 0.2 ms | 0.7 ms |

Compiled bundles: **1.67× at 2 workers, 2.67× at 4** (Q and Γ compile the same
words and time identically). Historical and legacy-fixed arms are scoring-light at
this scale (~2 s) and are unaffected by the worker count.

Memory: the harness records the parent's peak RSS only (≤ 576 MB). Each pool worker
imports Qiskit and the backend (≈ 350 MB each by the parent's own baseline), so
4 workers add ≈ 1.4 GB: an estimated ≈ 2 GB process-tree peak, under the 8 GB bound.

### 4.3 Population repair evidence

Legacy-fixed smoke, large-miss checkpoints: distinct insertion denominators
`{1.0, 1.0120, 1.0220, 1.0299, 1.1102, 1.1242, 1.1805, 1.1898}` (historical:
`{1.0}`); population sizes 1,957–2,186; maintenance checkpoints had no
permitted deletion, so the delete population was empty there.

## 5. Tests

Lane set + cross-lane implicated suites: **330 passed** (`ram_guard` peak
793 MB). Files: `test_ap_mclachlan_{adaptive_trajectory, deletion_permission,
exchange_certification, exchange_config, exchange_integration,
exchange_oracle, exchange_selector, exchange_structural, performance,
route_parity, structural_cache, schur_identity, inverse, objective_ordinates,
population_scoring, compiled_patch_cost}.py`, `test_compiled_ansatz.py`,
`test_paper_ii_runs.py`, `test_paper_ii_final_selector_comparison.py`,
`test_paper_ii_staged_accuracy_campaign.py`, `test_time_dynamics_campaign.py`,
`test_ap_terminal_qiskit_cost.py`, `test_qse_compact_root_refit.py`,
`test_ra_adapt_exact_geometry_backend.py`.

What is pinned, by the plan's categories: mathematical (Γ = Q exact,
divergence under stabilization, rank/threshold under the named cutoff,
fail-closed monotonicity, tiny-violation logging); score (K = 1 identity,
τ = 0 gives m/K, m = −τ cost-independent, stay S = 0, zero-merit cheap prune
beats stay, K bounds, sha order-sensitivity, rank determinism); cost
(singleton bug reproduced and 1/10/100 discriminated, signed
saving/burden, word identity, class partition, placement tie-break, cache
counters, compiler failure raises with no fallback, backend provenance,
pool-vs-in-process triple identity); search (Q/Γ identical population hash
and K, placement cut honored at materialization, legacy repair over the
population, historical payload identical after stripping only telemetry);
concurrency (fixture and full-trajectory bitwise parity at 1/2/4 workers for
Q and Γ, pool telemetry mode); campaign (15 cells, 45 actions, 0 AVQDS, live
parser re-parse, anti-fabrication nulls, route binding fail-closed).
`paper_ii_runs list` shows the seven arms; `matrix` emits 91 commands; the
generator prints 45 actions with `code_revision` bound.

GitNexus `detect_changes` (compare vs `origin/main`): 155 symbols, 28 files,
risk low, no `partial`/`truncated`; the index is stale (`a35db0e`), so the
empty affected-process set is corroboration of the source-level caller audit,
not a standalone verdict.

## 6. Author decisions surfaced (not decided here)

1. `τ_del` defaults to `prune_patch_smoothness_eta_max` (the normalized
   Schur-loss tolerance the deletion permission already enforces); recorded
   as requested/resolved in provenance. Any other value is an author choice.
2. Population membership: every valid evaluated candidate of the level, all
   faces, enters the K population; compiler-invalid and ordinate-invalid
   candidates are recorded and excluded before sealing. The sign filter then
   applies after sealing. A pure prune with m > 0 is eligible in the
   large-miss regime (it improves accuracy); the plan's "prunes form
   maintenance" is realized by the face closure below the cut, not by
   exclusion above it.
3. Frontier child-acquisition priorities under population bundles use the
   route ordinate instead of the historical composite score (inert for every
   registered run: insertion batch size 1).
4. The compile guard counts candidates (families admitted whole), not
   transpiles; cache hits reduce actual work below the count.
5. The Γ monotonicity fail-closed check rejects only the affected candidate
   from the shared Q/Γ population and records the rejection. It does not clip
   Γ, relax the tolerance, or abort an otherwise valid selection level.
6. The current manuscript's absolute-value kernel is superseded by the
   shifted kernel here; no manuscript edit was made.

## 7. Limitations and remaining bottlenecks

- Compile cost scales with `|pool| × |cuts|` per large-miss checkpoint
  (≈ 30 s serial here at ~30 params; more with support growth). At t = 20
  with several hundred in-debt checkpoints that is hours per trajectory
  serial, or roughly a third of that at 4 workers. Placement compiles are the
  larger half; a persistent pool across checkpoints would save the per-pool
  spawn (~1–2 s) but was not needed at this scale.
- BLAS thread pinning for workers is done by setting the five `*_NUM_THREADS`
  variables in the parent before spawn (parent numerics unaffected; pins
  recorded per selection). `threadpoolctl` is not installed.
- The qiskit-ibm-runtime plugin loader emits an
  `IBMFractionalTranslationPlugin` deprecation warning on every transpile
  session; it is not ours.
- `report_refresh_required.json` still blocks further submission and any
  claim that the visible PDF is current; the lock records both blocks.
- The plan's SIAM-side facts are unchanged: the r07 engine lineage cannot be
  resumed on `main` (see `PAPER_II_FULL_APM_PARALLELIZATION_20260901.md` §7).

## 8. Later Codex phase (not started here)

Calibration (`paper_ii_benchmark_calibration --allow-unverified-response-scales`
remains degraded until the author rebaselines), submission-block resolution,
package/upload/submit/monitor/fetch/SHA/aggregate, one comprehensive PDF.
