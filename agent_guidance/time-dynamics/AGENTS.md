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
Below `--residual-ratio-threshold` no structural family is acquired at all.

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
