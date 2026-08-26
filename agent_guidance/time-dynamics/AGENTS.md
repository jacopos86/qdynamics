# Time-Dynamics Lane (Paper II)

Progressive-disclosure contract for the AP-McLachlan lane. Read this before
opening lane source; load only the module the task needs. Scientific
invariants and paper policy remain governed by root `AGENTS.md`,
`MATH/AGENTS.md`, and `agent_guidance/shared/scientific-invariants.md`.

## Route identity

The single current support-patch route is **generalized exchange**
(`paper_ii_generalized_exchange_v2`). Its paper-owned language is defined in
`agent_guidance/time-dynamics/CONTEXT.md`. At each checkpoint it
enumerates measurement-free-permitted deletion-cardinality rungs and
positioned zero-angle insertions on one frozen ray, scores realized captured
drift weighted by Paper-I proxy hardware cost, and materializes/refits only
ranked finalists through hard commit gates. The former append-ladder,
prune-ladder, unified,
and legacy selectors were deleted 2026-08-18 (git history preserves them);
"checkpoint controller" remains a legacy compatibility surface, never the
route identity.

## Module map — `pipelines/time_dynamics/ap_mclachlan/`

Selector stack (bottom-up; each layer only imports below itself):

| Module | Contract |
|---|---|
| `../generalized_exchange.py` | Ansatz-independent mathematics: the patch pair `(D,I)`, admissible faces, debt ordering, and realized-L2 acceptance/fallback |
| `exchange_config.py` | AP adapter settings partitioned into score, eligibility, certification, and search-budget objects |
| `commutation.py` | Block-swap certificates and singleton insertion cuts; algebra delegated to `pipelines.static_adapt.commutation_metadata` (single source both lanes share, parity-tested against Paper I) |
| `insertion_words.py` | Immutable `(W, W_D)` plan words, deletion erasure, whole-word commutation quotient (lex-least trace normal form; inserted tokens sort before survivors — load-bearing) |
| `deletion_family.py` | Complete deletion rungs and the joint-work guard (admits whole families only; the sole computational cap) |
| `deletion_permission.py` | Measurement-free set permission: accumulated effective-angle ray upper bound plus normalized reverse-Schur deletion loss; memoized decision and run telemetry |
| `structural_cache.py` | Checkpoint-local frozen-ray geometry: positioned tangents in one batched pass, deletion-independent cross/Gram/force blocks, memoized candidate solves |
| `exchange_structural.py` | Family enumeration (singleton level, rungs, priorities, frontiers), the scalar score `U_ins + U_del + w·δ`, deterministic ordering |
| `exchange_certification.py` | Finalist materialization (delete → remapped-cut insert), optional refit hook, hard gates (ray/smoothness are deletion-containing checks; every finalist gets finite/conditioning) |
| `exchange_selector.py` | Selection loop: every guard-admitted deletion rung is scored with the d0 singletons before the first certification pass (deletions always compete on score); frontiers are escalation-gated; certify one at a time, commit atomically |
| `exchange_integration.py` | AP realization adapter: ansatz/generator atoms, cuts, costs, labels, geometry, and certification oracles → pure generalized-exchange domain and selector |

Route core:

| Module | Contract |
|---|---|
| `adaptive_trajectory.py` | Trajectory loop, route transport config, decision types, cooldown/history runtime state, ordered parallel map |
| `state.py` | AP state build/append/delete/**insert-at-cut** materialization with parity checks |
| `geometry_eval.py`, `geometry.py`, `inverse.py` | Frozen-ray geometry, realized metrics, supported solve. `McLachlanSolve.captured_drift = 2fᵀθ̇ − θ̇ᵀKθ̇` is the scoring authority; `gamma` is legacy telemetry |
| `fixed_step.py`, `integrators.py` | Propagation solve with repair, integration |
| `support_atoms.py`, `support_patch.py`, `support_frontier.py` | Atom enumeration/occurrence policy; retained score/telemetry types; frontier compatibility surface |
| `hamiltonian.py` | `H(t) = H_static + c(t)·D` with cached dense operators |
| `performance.py` | Zero-cost-when-inactive phase profiler |
| `commutation`… tests | One test file per module: `test/test_ap_mclachlan_<module>.py`; `test_ap_mclachlan_exchange_oracle.py` is the brute-force authority |

Diagnostics: `pipelines/time_dynamics/diagnostics/ap_runtime_benchmark.py`
(subprocess-isolated wall/CPU/RSS/BLAS benchmarking of the ordinary runner).
Seeds: `pipelines/time_dynamics/fixed_vqe_conditioning.py` +
`runners/build_fixed_vqe_conditioning_seed.py`.

## Runner and settings

`pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py` remains the
compatibility-named runner. Live selector settings are partitioned by
`APGeneralizedExchangeConfig` and recorded in run provenance:

- `--max-insertion-batch-size` (zero closes the insertion face)
- `--no-exchange-deletions` (insert-face ablation only; deletions are on for
  the operating route)
- `--interaction-frontier-widths` (comma ints; None → `2,4,8,…`)
- `--structural-score-floor` — τ_score; the floor, not certification, excludes
  numerical-noise candidates
- `--max-joint-patch-evaluations` — enumeration work cap; admits complete
  families only; `None` is for small systems and oracles
- `--max-certification-attempts-per-level` — certification work cap: bounds
  finalist materializations per level (`None` = unbounded). Needed whenever
  gates can reject broadly — e.g. on a near-stationary ray `||b||^2 ~ 0` makes the
  smoothness denominator vanish, every deletion-containing finalist fails,
  and an unbounded level grinds through every ranked candidate at one full
  state materialization each

Certification thresholds reuse `--prune-ray-distance-tol`,
`--prune-patch-smoothness-eta-max`, `--append-schur-max-condition-number`.

**Measurement economics**: insertion candidates require new quantum
measurements, so the Paper-II factorial campaign opens insertion faces only
while the absolute McLachlan distance satisfies (L_k^2>\tau_{L^2}). Deletion
sets are first screened using only already-paid checkpoint information: the
effective rotation angles bound finite ray motion, and the reverse-Schur
quadratic bounds frozen-support captured-drift loss. Only permitted sets enter
structural scoring; only ranked finalists reach candidate-state materialization
and the temporarily retained overlap/velocity validation gates. Within the
open patch family, a pure deletion is an empty insertion set and a pure
insertion is an empty deletion set; they are not separate algorithmic routes.

Deletion-utility hooks (all recorded in provenance):

- **Conditioning** — `--prune-condition-lambda-kappa-rel/-dam` (default 0 =
  off) weight the log10 condition-number relief/damage of the deletion branch
  versus the base support, read from solve metadata the enumeration already
  paid for (no extra eigendecompositions).
- **History** — `--prune-history-lambda` (default 0) weights a windowed-mean
  prior (`--prune-history-window`) of previously attempted deletion losses,
  recorded per stable runtime coordinate label on the prune runtime state.
- **Certification refit** — `--certification-refit` (default on) runs a
  bounded L-BFGS-B trust-region refit (`--certification-refit-trust-radius`,
  `--certification-refit-max-iterations`) of each materialized finalist's
  angles toward the frozen checkpoint ray before the hard gates; pure
  zero-angle insertions are skipped, and a refit that fails to reduce the
  Fubini–Study infidelity is discarded.

## Run commands — one source (2026-08-23)

`pipelines/time_dynamics/paper_ii_runs.py` is the single authority for Paper-II
run configuration. Compose runs from named parts (arm x insertion gate x drive
x horizon); never hand-spell a runner invocation, and never restate numerics or
guards in a script, a campaign module, or a document.

```bash
PYTHONPATH=. python3 -m pipelines.time_dynamics.paper_ii_runs list
PYTHONPATH=. python3 -m pipelines.time_dynamics.paper_ii_runs show --arm exchange --drive fastweak --horizon t10 --output-json output/x/run.json
```

`test/test_paper_ii_runs.py` parses every registered command against the
runner's live parser and asserts that campaign cells and registry runs
configure the same trajectory. This exists because the copies drifted: the
campaign module carried `max_structural_pool_size: 8` for weeks after that cap
was identified as the lane's largest accuracy defect (it discarded ~117 of the
125 deduplicated pool words), and any campaign built from it silently
reproduced the defect.

Current registry contents:

- **Numerics** (never method-dependent): the initial factorial campaign fixes
  Euler and ridge `1e-6`; the general Paper-II accuracy default remains RK4
  with ridge `1e-7` for non-factorial work.
- **Time-step controllers** (crossed with both methods): tangent-state motion
  `1e-2`, parameter motion `5e-3`, and their composition. The state/composed
  controllers use subdivision budget **10**. No realized trial-state overlap
  is prepared or measured.
- **Structure**: candidate pool cap 128 (above the 125-word deduplicated pool,
  so it does not bind), **conditioning gate off**, guards 50000 / 12 / 2, and
  insertion batch 1.
- **Algorithmic methods in the factorial campaign**: `exchange` and `avqds`.
  `append_only` remains an ablation, not a third primary method.
- **Insertion gates**: the primary campaign uses the McLachlan-
  (L^2) gate and varies its cut offline per run configuration.

### Why the two repaired guards read as they do

The **subdivision budget** was 4 until 2026-08-22. At 4 a step could exhaust its
budget, fail to cure a state-motion cap violation, and then advance
unsubdivided anyway, marked only by a counter in the summary; two such steps out
of 51 took a measured HH energy error from 3.8e-3 to 2.1e-1. At 10 there are
none, and 14 is identical to 10. The runner now warns on stderr whenever a step
advances uncured; `--fail-on-unsupported-steps` makes it exit non-zero. Note
that `solve_repair_unsupported_count` is *not* that alarm — it runs 16-27 on
accurate trajectories.

The **certification conditioning gate** is off because no setting of it is
useful. Measured at residual 1e-4: off and 5e7 are identical (7.79e-4 mean
|dE|, and kappa never reaches 5e7); 3e7 rejects 20 candidates for 1.15e-3; 1e7
rejects all 600, certifies nothing, and starves the ansatz to 28 parameters
(6.89e-3, doublon 1.75e-2). kappa is a symptom of the manifold, not a cause of
trajectory error; the step-size guard is what protects a trajectory.

## Continuation

Every `run.json` carries `resume_state` (`pipelines/time_dynamics/resume.py`):
per-coordinate Pauli word, coefficient, register width, plus the final
parameter vector and stop time. `--resume-from-run-json <prior>` rebuilds that
support and continues; the seed artifact is still required for the problem and
reference state. The grid origin follows automatically (`--t-initial` defaults
to the prior stop time) so the reporting grid and drive phase continue rather
than replay -- omitting that was a real bug, silently propagating the resumed
state from t=0 with the wrong drive phase.

Continuation is physically faithful, not decision-identical: energies agree
with an uninterrupted run to solver tolerance (2.8e-9 measured), but controller
history does not cross the boundary, so structural choices in the continued leg
may differ.

## Run locks

Every `run.json` carries `run_lock` (`pipelines/time_dynamics/run_lock.py`):
seed path and sha256, family, phonon cutoff, time grid, drive profile,
integrator, inverse policy, repair config, structural policy, guards, and the
code revision. `physics_fingerprint` hashes only the physics a comparison must
hold fixed — policy and code revision are deliberately excluded, since arms
*should* differ in policy. Call `assert_comparable(locks)` before aggregating
arms into any table or figure; it refuses runs whose physics differs. This
exists because an arm once silently ran another arm's configuration and nothing
in the artifacts flagged it.

## Campaign specification

`pipelines/time_dynamics/campaign.py` exposes a scientific campaign interface:
seed × algorithmic method × time-step controller × drive × horizon × activation
cut. Every cell resolves through `paper_ii_runs.build_run`; campaign code never
hand-spells scientific runner flags. Preparation binds seed hashes, parses all
commands with the live runner, and emits `PREPARED_NOT_SUBMITTED`.

The primary declaration is
`pipelines/time_dynamics/campaigns/paper_ii_factorial_euler_v1.py`: two methods
(`exchange`, `avqds`) × three controllers across six drives, Euler/ridge
`1e-6`, with prior-informed threshold worklists. Old frontier results choose
initial cuts only and never fill new cells. After execution,
`audit_completed_campaign` checks run locks, exact-reference reporting
separation, step-work telemetry, and hash-bound terminal FakeMarrakesh costs.
The terminology contract is
`MATH/paper_facing/paper_II_dynamics/run_configuration_contract.md`.

## Operating configuration vs configuration surface

Knob audit (`pipelines/time_dynamics/diagnostics/knob_audit.py`, 885 steps /
33 runs, 2026-08-18): of 11,499 repair candidates, 6 were applied; inverse
ladders stayed on the base rung 883/885 steps; damping never moved; the
conditioning gate never bound (peak kappa 5.1e7 vs 1e12 cap). Only
state-motion-triggered local subdivision repeatedly acts. `--solve-repair-profile
minimal` (default) keeps exactly that; `full` restores the ladder search for
diagnosis. Score hooks (conditioning relief/damage, history) are carried at
zero weight and no result depends on them. Rerun the audit before claiming a
knob matters.

## Canonical numerics (2026-08-18 sweep)

The initial two-method × three-controller factorial campaign deliberately uses
Euler with ridge `1e-6` for every configuration, isolating controller and
structural-rule effects. After the full Euler matrix is selected, repeat the
same six configurations under RK4 as a consistency check; do not mix Euler and
RK4 within one factorial comparison. Outside that declared campaign, RK4 is
the general accuracy default. Deletion gates: ray tolerance
5e-2 admits cumulative structural damage over long horizons (44 certified
deletions -> 0.19 drift); 2e-3 or tighter for trajectory work. Class-tuned
controller-era settings live in
`chtc/generic_time_dynamics_table/input/class_settings/` (restored from
CHTC 2026-08-18).

## Scale guidance

The singleton level is `|pool| × |retained cuts|` candidates, each one
supported solve. Measured: ~2.9k candidates (87 params, nph=1 seed, ~2.5 s per
checkpoint) versus ~188k candidates (616 params, nph=3 seed, ~85 s plus
~770 MB of tangent columns). Because families admit whole, a small guard on a
large pool rejects the singleton level entirely (permanent stay): bound the
candidate pool in the run configuration, not with selector-side prefilters,
which the specification forbids.

Because every admitted deletion rung is scored before certification, running
with `--max-joint-patch-evaluations None` at scale enumerates ALL rungs
(spec-mandated) — set the guard on any run with a nontrivial deletable set.

## Test baseline

`agent_guidance/time-dynamics/test-baseline-20260815.md` pins the pre-existing
failures (quarantined legacy collection errors and 37 known reds); a lane run
is clean when its only failures are listed there.
