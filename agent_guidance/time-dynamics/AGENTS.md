# Time-Dynamics Lane (Paper II)

Progressive-disclosure contract for the AP-McLachlan lane. Read this before
opening lane source; load only the module the task needs. Scientific
invariants and paper policy remain governed by root `AGENTS.md`,
`MATH/AGENTS.md`, and `agent_guidance/shared/scientific-invariants.md`.

## Route identity

The single current support-patch route is the **deletion-conditioned exchange
selector** (`paper_ii_deletion_conditioned_exchange_v1`), specified in
`prompt-exports/paper_ii_noiseless_conditional_exchange_implementation_spec.md`
(mathematics plus the appended code-level section). At each checkpoint it
enumerates complete deletion-cardinality rungs and positioned zero-angle
insertions on one frozen ray, scores realized captured drift weighted by
Paper-I proxy hardware cost, and materializes/refits only ranked finalists
through hard commit gates. The former append-ladder, prune-ladder, unified,
and legacy selectors were deleted 2026-08-18 (git history preserves them);
"checkpoint controller" remains a legacy compatibility surface, never the
route identity.

## Module map — `pipelines/time_dynamics/ap_mclachlan/`

Selector stack (bottom-up; each layer only imports below itself):

| Module | Contract |
|---|---|
| `commutation.py` | Block-swap certificates and singleton insertion cuts; algebra delegated to `pipelines.static_adapt.commutation_metadata` (single source both lanes share, parity-tested against Paper I) |
| `insertion_words.py` | Immutable `(W, W_D)` plan words, deletion erasure, whole-word commutation quotient (lex-least trace normal form; inserted tokens sort before survivors — load-bearing) |
| `deletion_family.py` | Complete deletion rungs and the joint-work guard (admits whole families only; the sole computational cap) |
| `structural_cache.py` | Checkpoint-local frozen-ray geometry: positioned tangents in one batched pass, deletion-independent cross/Gram/force blocks, memoized candidate solves |
| `exchange_structural.py` | Family enumeration (singleton level, rungs, priorities, frontiers), the scalar score `U_ins + U_del + w·δ`, deterministic ordering |
| `exchange_certification.py` | Finalist materialization (delete → remapped-cut insert), optional refit hook, hard gates (ray/smoothness are deletion-containing checks; every finalist gets finite/conditioning) |
| `exchange_selector.py` | Selection loop: every guard-admitted deletion rung is scored with the d0 singletons before the first certification pass (deletions always compete on score); frontiers are escalation-gated; certify one at a time, commit atomically |
| `exchange_integration.py` | Route adapter: atoms/cuts/costs/labels/gates from route objects → selector; `PatchDecision` payload back |

Route core:

| Module | Contract |
|---|---|
| `adaptive_trajectory.py` | Trajectory loop, typed configs, decision types, deletion feasibility gates, prune runtime state, ordered parallel map (~2.3k lines) |
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

`pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py` (84 flags
after the 2026-08-18 purge). Selector-specific settings, all typed on
`SupportPatchControllerConfig` and recorded in run provenance:

- `--max-insertion-batch-size` (None → falls back to `--max-append-batch-size`)
- `--interaction-frontier-widths` (comma ints; None → `2,4,8,…`)
- `--structural-score-floor` — τ_score; the floor, not certification, excludes
  numerical-noise candidates
- `--max-joint-patch-evaluations` — enumeration work cap; admits complete
  families only; `None` is for small systems and oracles
- `--max-certification-attempts-per-level` — certification work cap: bounds
  finalist materializations per level (`None` = unbounded). Needed whenever
  gates can reject broadly — e.g. on a converged ray `||b||^2 ~ 0` makes the
  smoothness denominator vanish, every deletion-containing finalist fails,
  and an unbounded level grinds through every ranked candidate at one full
  state materialization each

Certification thresholds reuse `--prune-ray-distance-tol`,
`--prune-patch-smoothness-eta-max`, `--append-schur-max-condition-number`.

**Measurement economics** (`--residual-ratio-threshold`): insertion
candidates require new quantum measurements, so they are enumerated only at
checkpoints whose residual ratio meets the threshold. Pure deletions are
row/column selections of the already-paid frozen-ray geometry —
measurement-free — and are considered at every checkpoint (prune-only mode,
`insertions_enabled: false` in the decision payload).

Deletion-utility hooks (all recorded in provenance):

- **Conditioning** — `--prune-condition-lambda-kappa-rel/-dam` (default 0 =
  off) weight the log10 condition-number relief/damage of the deletion branch
  versus the base support, read from solve metadata the enumeration already
  paid for (no extra eigendecompositions).
- **History** — `--prune-history-lambda` (default 1) weights a windowed-mean
  prior (`--prune-history-window`) of previously attempted deletion losses,
  recorded per stable runtime coordinate label on the prune runtime state.
- **Certification refit** — `--certification-refit` (default off) runs a
  bounded L-BFGS-B trust-region refit (`--certification-refit-trust-radius`,
  `--certification-refit-max-iterations`) of each materialized finalist's
  angles toward the frozen checkpoint ray before the hard gates; pure
  zero-angle insertions are skipped, and a refit that fails to reduce the
  Fubini–Study infidelity is discarded.

## Canonical defaults (2026-08-20)

Defaults now reproduce the reported configuration, so a flagless call runs the
paper's route rather than a diagnostic one: `rk4`, solve repair on at the
`minimal` profile, certification refit on with trust radius 0.6, insertion gate
`residual_ratio_threshold = 2e-2`, deletion ray tolerance `2e-3`, and all four
computational guards populated (joint 50000, per-level 12, per-branch 2, pool
8, insertion batch 1). Euler, loose caps, and unbounded guards remain available
as explicit opt-outs (`--integrator euler`, `--no-solve-repair`,
`--solve-repair-profile full`, `--no-certification-refit`).

Guards being unset was not merely slow: the same three-checkpoint run took over
ten minutes with `None` guards and 25 seconds with them populated.

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

`pipelines/time_dynamics/campaign.py` declares a run matrix as the product of
`SeedSpec` x `DriveSpec` x `HorizonSpec` x `PolicyArm`, resolving each cell to
a runner argv with canonical numerics, the four computational guards, and
locked seed provenance (sha256 per seed, binary-aligned nph enforced at
construction). Arms: `exchange_arm(ray_tol)`, `append_only_arm()`,
`avqds_arm(l2_cut)` (flagged `is_comparator`). `write_campaign_manifest`
records every cell. It owns no scientific defaults - physics comes from the
seed/drive, structure from the arm - and never searches artifact trees.
Golden-run parity locks in `test/test_ap_mclachlan_route_parity.py` pin the
route's decisions and energies across refactors.

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

For accurate trajectories always run `--integrator rk4 --solve-repair
--solve-repair-state-motion-l2-step-max 1.0e-2 --solve-repair-kink-eta-max
5.0e-3`. Measured on the stress seed (append-only, dt=0.04, exact
reference): Euler+default caps 1.6e-2 energy error, rk4 alone 1.1e-2,
rk4+tight caps 1.3e-3. The runner's Euler default is a fast-diagnostic
setting, not the accuracy configuration. Deletion gates: ray tolerance
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
