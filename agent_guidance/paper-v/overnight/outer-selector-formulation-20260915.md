# outer-selector-formulation (2026-09-15)

Round 3, overnight loop.  Find a conic formulation of the one-shell outer
target stage that returns an accepted witness where the current one returns
`NumericalError`.

Round 2 measured, at the cutoff-16 correlated preparation (15 entrance keys,
`lambda_ep = 1.5`, `gamma = 0.5`, drive amplitude 1, phonon cutoff 16,
`t = 0`, one evaluation, boxed, `canonicalize=False`): the outer-aware
invariant target solve ends in `explicit_target:NumericalError` after about
`46 s` under both gate settings; the strictly convex form at frontier weight
`1e-5` reaches `AlmostSolved` with relative duality gap `1.1194e-6` and is
accepted only under a named `2e-6` acceptance tolerance; and at a forced
trigger stage's own lower moments all three SSPRK(3,3) stages return
`explicit_target:NumericalError` (`46.4`, `43.1`, `58.8 s`, peak `3.90 GB`),
which no acceptance tolerance admits.

Status: complete.

## The result, first

The negative result first: none of the six alternative formulations is what
turns the failure into an accepted witness.  The recorded program, reassembled
unchanged and run at the recorded routine's *own* tightened numerical profile
(`retry_profile=2`: conic tolerance `1e-10`, `max_iter` 500, 20 iterative
refinement passes at `1e-14`, equilibration off), is accepted at the cutoff-16
correlated preparation with relative duality gap `7.463367558152577e-07`
against the unchanged exploratory acceptance `1e-6`.  It is accepted at the
forced-trigger stage's own lower moments too, at `7.546886686991722e-07`.  So
is every formulation tried at that profile bar one, all within 10 percent of
the same gap.

The reason the recorded stage never gets there is one line:
`_solve_explicit_target_qp` retries at the tightened profile only for the
statuses `{"AlmostSolved", "InsufficientProgress", "MaxIterations"}`
(`apcm_projective_guard.py:1493`), and the first pass at both preparations
ends in `NumericalError`, which is not in that set.  Round 1 identified that
line as the reason `accept_first_provisional` does not discriminate; what was
not done is to run the tightened pass anyway.

| operand set | recorded program, recorded first pass | recorded program, tightened profile |
|---|---|---|
| cutoff-16 correlated preparation | `NumericalError`, rejected, `50.12 s` | `AlmostSolved`, accepted `provisional_independent_kkt`, gap `7.4634e-07`, `75.86 s` |
| forced-trigger SSPRK(3,3) stage-one trial point | `NumericalError`, rejected, `49.46 s` | `AlmostSolved`, accepted `provisional_independent_kkt`, gap `7.5469e-07`, `74.91 s` |
| `t = 5.086270` saved failing stage (`contain`, `moment_extension_degeneracy`) | `PrimalInfeasible`, rejected, `46.32 s` | `PrimalInfeasible`, rejected, `53.40 s` |

Both acceptances are `provisional_independent_kkt` under the unchanged
exploratory rule at its unchanged `1e-6` tolerance; no tolerance, default, or
acceptance rule was moved.  The accepted objective at the preparation is
`0.011267123150627062`, which is `1.6e-8` in relative terms from the record's
accepted outer-aware witness objective `0.01126712315`.  Its restriction to
the current 24-row cone is feasible: minimum scaled eigenvalue
`-7.466675726570977e-16`, invariant target image preserved to `0.0` maximum
departure, current envelope ratio `0.9927388271978883` against the guard's
`envelope_fallback` threshold `1 - 1e-8`.

Three further findings, each measured below.

1. **The declared frontier term is what cost round 2 its acceptance.**  At the
   tightened profile the recorded program at frontier weight `1e-5` has
   relative duality gap `1.1193523475885861e-06` -- round 2's `1.1194e-6` to
   five digits -- against `7.4634e-07` at weight `0`.  Round 2's strictly
   convex solve was already at the tightened profile, because a positive
   weight makes the first pass return `AlmostSolved`, which *is* in the retry
   set.  The frontier term is therefore not what buys a witness; it is what
   pushed the gap above `1e-6` and forced the named `2e-6` tolerance.
2. **The `t = 5.086270` stage is nonextendible, not unresolved.**  At the
   retracted lower moments the projective-face agent saved for the post-pulse
   stage failure, the outer target stage returns `PrimalInfeasible` at both
   profiles, with the same reconstructed Gram minimum scaled eigenvalue
   `-0.25537193825`.  No formulation, profile or tolerance admits a witness
   there: those lower moments do not extend to the one-shell outer Gram inside
   the box at all.
3. **Whitening buys two decades on the first pass and nothing on the
   tightened one.**  The invariant target image quadratic `R^T M R` is exactly
   the orthogonal projector onto the row space of `R`, because the target
   metric `M` is the inverse of the readout covariance `R R^T`: at the
   preparation its 238 eigenvalues are 15 ones and 223 zeros, while `M` itself
   has condition number `1613.88` and `R` singular values from `165.638` down
   to `4.123`.  Carrying the image in whitened coordinates `t = U R z`, with
   `U` the upper Cholesky factor of `M = U^T U`, gives the image map
   orthonormal rows (`(U R)(U R)^T = I`) and the identity objective
   `0.5 ||t||^2`.  On the recorded first pass that moves the status from
   `NumericalError` to `AlmostSolved`, the relative duality gap from
   `3.5740e-04` to `3.2661e-06` (a factor `109`) and the reconstructed Gram
   minimum scaled eigenvalue from `-2.590e-09` to `-3.347e-13`; it is still
   rejected, by a factor `3.3` on the gap.  At the tightened profile it is
   accepted at `7.5480e-07`, marginally behind the recorded assembly's
   `7.4634e-07`.

## Files I own this round

- `paper_5/src/paper5/stability/apcm_outer_selector_formulations.py` (new)
- `paper_5/tests/test_apcm_outer_selector_formulations.py` (new)
- this report
- run directories `output/local_runs/paper_v_outer_selector_*_20260915_v1/`,
  which also hold the two harness scripts
  (`run_outer_selector_formulations.py`, `restrict_accepted_witness.py`) and
  `run_batch.sh`

Nothing else is edited.  `apcm_projective_guard.py` and `projected_apcm.py`
are imported from, never modified; they are another agent's this round.

## Impact analysis (main checkout index, read-only; index 4 commits behind)

No existing function is edited this round, so no symbol carries an edit risk.
The symbols this module imports and wraps were checked anyway:

| symbol | verdict | callers |
|---|---|---|
| `_solve_explicit_target_qp` | LOW, 3 impacted | `select_outer_feasible_projective_guard`, `select_projective_guard`, `audit_projective_preparation` |
| `_guard_acceptance` | MEDIUM, 9 impacted | the six solve paths of `apcm_projective_guard.py` |
| `select_outer_feasible_projective_guard` | UNKNOWN -> LOW after grep | `projected_apcm.py:970`, `apcm_projective_event_guard_analysis.py:335`, `tests/test_apcm_projective_event_guard.py` |

## (a) The operands, and how they were extracted

Three operand sets, all extracted by calling the existing builders and saving
the arrays they produce.

**Set A, the cutoff-16 correlated preparation.**  The canonical correlated
preparation contracted into the selector's lower moments exactly as
`apcm_projective_event_guard_analysis._preparation` contracts it: 15 entrance
keys, `lambda_ep = 1.5`, `gamma = 0.5`, drive amplitude 1, phonon cutoff 16,
`canonical_embedding=True`, no proximal log-det completion.  The one-shell
outer extension, the invariant target image readout embedded in the literal
outer frontier, and the whitened target metric are then formed by
`projective_guard_outer_extension`, `_outer_current_image_matrix` and the
readout-covariance inverse, exactly as
`select_outer_feasible_projective_guard` forms them; the affine Gram pieces
exactly as `_solve_explicit_target_qp` forms them.  Build `0.076 s`.
Artifact: `output/local_runs/paper_v_outer_selector_operands_prep_20260915_v1/`.

**Set B, a forced-trigger stage's own lower moments.**  Round 2's artifact
`output/local_runs/paper_v_apcm_projective_forced_trigger_acceptance2em6_w1em5_20260915_v1/`
stores the 44-coordinate states and the three stage records but not the lower
moments the selector was handed.  The first stage's operands were therefore
reconstructed along the rollout's own path:
`prepare_projected_apcm_initial_state` at phonon cutoff 16, one
`FixedDictionaryProjectedAPCM.evaluate` at `t = 0` from the prepared
completion, the trial point `state + h * derivative` at `h = 0.0025` that
`_advance_projected_apcm_interval` hands to `contain_stage`, and
`state_lower_moments` of that trial point, which is what
`ProjectiveOuterEventGuard.contain_stage` contracts before calling the
selector.  Build `1.50 s`.  The reconstruction is confirmed by the recorded
formulation reproducing the failure round 2 recorded at that stage:
`explicit_target:NumericalError`, `49.74 s` against round 2's `46.42 s`.
Artifact: `output/local_runs/paper_v_outer_selector_operands_stage_20260915_v1/`.

**Set C, the projective-face agent's saved failing stage.**  Part way through
this round that agent began saving `stage_failure.npz` with the failing
stage's own retracted lower moments.  Five exist; the one used here is the
post-pulse `contain`-stage failure at `t = 5.086270` with the message
`retracted lower moments did not admit the declared selector:
moment_extension_degeneracy`, from
`output/local_runs/paper_v_apcm_projective_t4restart_default_t8_20260915_v1/`.
Its 29 `lower_moment_names` match `repr` of the 15-key extension's
`lower_keys` in order, which the harness checks before using them.  Artifact:
`output/local_runs/paper_v_outer_selector_operands_t5086_20260915_v1/`.

All three sets are the same program shape:

| quantity | value |
|---|---:|
| one-shell outer Gram rows | 60 |
| realified positive semidefinite cone side | 120 |
| outer frontier moments (standardized decision variables) | 238 |
| invariant target readout rank (image variables) | 15 |
| box rows | 476 |
| positive semidefinite cone rows in the svec packing | 7260 |

Saved arrays, per set: `scaled_readout` (`R`), `target_metric` (`M`),
`real_base` (the realified scaled Gram at the stage's lower moments with every
frontier coordinate zero), `coefficient_matrices` (the realified scaled Gram
coefficient of each standardized frontier coordinate), `base_vector` (the
Clarabel svec packing of `real_base`), the lower-moment values, the frontier
scales and the word scales, with `operands.json` naming every one of them and
listing the lower and frontier keys in order.

Conditioning of set A's operands, which is what the formulations act on:

| quantity | value |
|---|---:|
| condition number of the target metric `M` | 1613.88 |
| singular values of the embedded readout `R`, largest and smallest | 165.638 and 4.123 |
| eigenvalues of the invariant target image quadratic `R^T M R` | 15 equal to 1, 223 equal to 0 |
| spectral norm of the realified scaled base Gram, and its eigenvalue range | 3.2825, `-1.909e-1` to `3.2825` |
| Frobenius norms of the 238 Gram coefficients, smallest and largest | 2.000 and 10.472 |

## How every candidate is scored

Each formulation is a change of variables or of assembly of the *same*
mathematical problem, so every candidate frontier point is mapped back to the
recorded variable order `(z, q = R z)` and scored on the recorded program's own
`(P, q, A, b)`.  The reported quantities are therefore the recorded ones:

- **objective**, the invariant target image quadratic `0.5 q^T M q`, always
  with `q = R z` and the recorded `M`, in every formulation;
- **minimum scaled eigenvalue**, from `extension.matrix(lower, frontier)` and
  `extension.scaled_matrix`, that is on the *uninflated* Gram, which is what
  makes the psd-inflation rows comparable;
- **independent KKT residuals**, from `_independent_kkt_diagnostics` on the
  recorded program, with the positive semidefinite dual mapped back through
  the congruence `Y -> E Y E` and the box dual unchanged, and the image
  equality's multiplier `-M q` supplied from the point itself;
- **relative duality gap**, recomputed from `(x, z)` on the recorded program
  as `_original_relative_duality_gap` defines it, because a reformulated
  solver's own objective pair belongs to its own program;
- **acceptance**, from `_guard_acceptance` at its unchanged exploratory
  tolerance, with no tolerance passed.

Gate on that scoring: reassembling the recorded program in this module and
solving it reproduces the recorded routine's own numbers.  At set A,
`0.011286373848805029` against `0.01128637384343377` on the objective,
`-2.5903862151141617e-09` on the minimum scaled eigenvalue exactly,
`6.411437065053453e-05` against `6.411437065107105e-05` on stationarity, and
`3.5739542823707714e-04` against `3.573954177064206e-04` on the relative
duality gap, the last being the solver's own objective pair against this
module's recomputation from `(x, z)`.  On the small fixture the two agree
bitwise on the frontier point.

## (b) The six alternatives, measured

All rows: one-shell outer cone, 60 Gram rows, 238 outer frontier moments,
boxed, `frontier_weight = 0` unless the row says otherwise, Clarabel pinned to
one thread, one evaluation.  "profile" is `_solve_explicit_target_qp`'s own
numerical profile: `0` its first pass, `2` its tightened retry.  "admitted"
answers whether the existing exploratory acceptance at its unchanged `1e-6`
would take the result with no tolerance change.

### Set A, the cutoff-16 correlated preparation

| formulation | profile | status | admitted | objective | min scaled eigenvalue | independent feasibility | stationarity | complementarity | relative duality gap | s |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| recorded stage, as the selector calls it | ladder | `NumericalError` | no | 0.011286373848805029 | -2.5904e-09 | 2.0375e-08 | 6.4114e-05 | 1.0880e-07 | 3.5740e-04 | 48.43 |
| recorded program, this module's assembly | 0 | `NumericalError` | no | 0.011286373843433770 | -2.5904e-09 | 2.5904e-09 | 6.4114e-05 | 1.0833e-07 | 3.5740e-04 | 50.12 |
| recorded program | 2 | `AlmostSolved` | **yes** | 0.011267123150627062 | -2.1331e-14 | 2.1331e-14 | 3.8631e-08 | 9.1884e-09 | 7.4634e-07 | 75.86 |
| recorded program, frontier weight `1e-5` | 2 | `AlmostSolved` | no | 0.011271046130179874 | -4.2422e-14 | 4.2422e-14 | 5.4955e-08 | 1.4754e-08 | 1.1194e-06 | 72.14 |
| (1) equilibration, whitening absorbed | 0 | `AlmostSolved` | no | 0.011267801282793243 | -7.5039e-12 | 7.5039e-12 | 1.4625e-06 | 2.3485e-09 | 1.2748e-05 | 66.13 |
| (1) equilibration, whitening absorbed | 2 | `AlmostSolved` | **yes** | 0.011267124484936037 | -1.9616e-14 | 1.9616e-14 | 3.3903e-08 | 9.8952e-09 | 8.0852e-07 | 80.47 |
| (1a) whitening absorbed, no equilibration | 0 | `AlmostSolved` | no | 0.011267272594158413 | -3.3471e-13 | 3.3471e-13 | 3.3365e-07 | 4.3889e-09 | 3.2661e-06 | 73.05 |
| (1a) whitening absorbed, no equilibration | 2 | `AlmostSolved` | **yes** | 0.011267123370848875 | -2.1062e-14 | 2.1062e-14 | 3.9069e-08 | 9.3105e-09 | 7.5480e-07 | 75.32 |
| (1b) Gram congruence only | 0 | `NumericalError` | no | 0.011301705028921022 | -4.3231e-09 | 4.3231e-09 | 1.4258e-04 | 1.7004e-07 | 7.0634e-04 | 45.19 |
| (1c) frontier column scaling only | 0 | `NumericalError` | no | 0.011271388710256030 | -7.2450e-11 | 7.2450e-11 | 1.0000e-05 | 3.9351e-09 | 5.3115e-05 | 61.93 |
| (2) psd inflation `1e-11` | 0 | `NumericalError` | no | 0.011286375683364743 | -2.5037e-09 | 2.5037e-09 | 6.4148e-05 | 1.0186e-07 | 3.5759e-04 | 50.46 |
| (2) psd inflation `1e-10` | 0 | `NumericalError` | no | 0.011286375651423349 | -2.4889e-09 | 2.4889e-09 | 6.4149e-05 | 1.0070e-07 | 3.5761e-04 | 50.08 |
| (2) psd inflation `1e-9` | 0 | `InsufficientProgress` | no | 0.011306136841292483 | -5.3960e-09 | 5.3960e-09 | 1.5841e-04 | 3.2229e-07 | 8.8368e-04 | 46.50 |
| (2) psd inflation `1e-8` | 0 | `NumericalError` | no | 0.011306136210536223 | -5.3902e-09 | 5.3902e-09 | 1.8814e-04 | 3.6885e-07 | 1.0515e-03 | 46.02 |
| (2) psd inflation `1e-9`, with (1) | 0 | `NumericalError` | no | 0.011284053132668996 | -2.0453e-09 | 2.0453e-09 | 4.6099e-05 | 1.2025e-07 | 4.0218e-04 | 51.57 |
| (2) psd inflation `1e-8`, with (1) | 0 | `NumericalError` | no | 0.011305799229487072 | -3.8849e-09 | 3.8849e-09 | 1.0063e-04 | 3.2558e-07 | 8.7963e-04 | 46.97 |
| (3) the dual problem | 0 | `NumericalError` | no | 0.011370749911707360 | -3.9218e-05 | 3.9218e-05 | 1.1593e-05 | 4.7684e-05 | 1.7716e-02 | 50.53 |
| (4) first-order pass, then polish | 0 | `AlmostSolved` | no | 0.011267244837402147 | -1.3679e-09 | 1.3679e-09 | 4.1302e-07 | 4.0753e-08 | 2.5779e-06 | 80.74 |
| (4) first-order pass, then polish | 2 | `AlmostSolved` | **yes** | 0.011267122904161010 | -2.1077e-14 | 2.1077e-14 | 3.8164e-08 | 9.0830e-09 | 7.3731e-07 | 75.66 |
| (5) nullspace parameterization | 0 | `NumericalError` | no | 0.011276322386505010 | -1.1876e-09 | 1.1876e-09 | 1.6749e-05 | 2.1302e-07 | 1.6399e-04 | 52.67 |
| (5) nullspace parameterization | 2 | `AlmostSolved` | **yes** | 0.011267123278887885 | -2.1035e-14 | 2.1035e-14 | 3.8855e-08 | 9.2426e-09 | 7.5067e-07 | 74.74 |
| (6) homogeneous frontier scale `1e-1` | 0 | `NumericalError` | no | 0.011305573537470910 | -5.2486e-09 | 5.2486e-09 | 1.6225e-03 | 2.7919e-07 | 9.3697e-03 | 45.59 |
| (6) homogeneous frontier scale `1e+1` | 0 | `AlmostSolved` | no | 0.011267303409910268 | -2.2915e-13 | 2.2915e-13 | 4.8522e-07 | 1.1237e-08 | 2.5172e-06 | 75.90 |
| (6) homogeneous frontier scale `1e+1` | 2 | `AlmostSolved` | **yes** | 0.011267122859777570 | -2.1410e-14 | 2.1410e-14 | 3.8129e-08 | 9.0457e-09 | 7.3663e-07 | 74.69 |

### Set B, the forced-trigger stage-one trial point

| formulation | profile | status | admitted | objective | min scaled eigenvalue | stationarity | complementarity | relative duality gap | s |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| recorded stage, as the selector calls it | ladder | `NumericalError` | no | 0.011298297586012280 | -3.4445e-09 | 1.2162e-04 | 1.5164e-07 | 6.7813e-04 | 49.74 |
| recorded program, this module's assembly | 0 | `NumericalError` | no | 0.011298297580589320 | -3.4445e-09 | 1.2162e-04 | 1.5121e-07 | 6.7813e-04 | 49.46 |
| recorded program | 2 | `AlmostSolved` | **yes** | 0.011267149490469126 | -2.2104e-14 | 3.9063e-08 | 9.3047e-09 | 7.5469e-07 | 74.91 |
| (1) equilibration, whitening absorbed | 0 | `NumericalError` | no | 0.011305643015830591 | -3.8589e-09 | 9.9008e-05 | 3.3650e-07 | 8.6368e-04 | 46.29 |
| (1) equilibration, whitening absorbed | 2 | `AlmostSolved` | no | 0.011267286606556461 | -3.4318e-13 | 1.6274e-07 | 4.7673e-08 | 3.8807e-06 | 75.52 |
| (1a) whitening absorbed, no equilibration | 0 | `AlmostSolved` | no | 0.011267814220700261 | -3.5954e-12 | 1.2719e-06 | 1.0135e-08 | 1.2451e-05 | 72.74 |
| (1a) whitening absorbed, no equilibration | 2 | `AlmostSolved` | **yes** | 0.011267150070861133 | -2.2101e-14 | 3.9979e-08 | 9.5208e-09 | 7.7240e-07 | 74.01 |
| (1b) Gram congruence only | 0 | `NumericalError` | no | 0.011301654224754792 | -4.3023e-09 | 1.4332e-04 | 1.7125e-07 | 7.1004e-04 | 47.71 |
| (1c) frontier column scaling only | 0 | `AlmostSolved` | no | 0.011270936068507433 | -5.2241e-11 | 8.0793e-06 | 4.6169e-09 | 4.2912e-05 | 65.79 |
| (2) psd inflation `1e-11` | 0 | `NumericalError` | no | 0.011305980073824795 | -5.3463e-09 | 1.5850e-04 | 3.2395e-07 | 8.8396e-04 | 46.40 |
| (2) psd inflation `1e-10` | 0 | `NumericalError` | no | 0.011286347503874843 | -1.9127e-09 | 6.3932e-05 | 6.7852e-08 | 3.5641e-04 | 49.84 |
| (2) psd inflation `1e-9` | 0 | `NumericalError` | no | 0.011305979022438406 | -5.3456e-09 | 1.5849e-04 | 3.2262e-07 | 8.8409e-04 | 45.62 |
| (2) psd inflation `1e-8` | 0 | `NumericalError` | no | 0.011271328226356757 | -8.6749e-11 | 1.0594e-05 | 5.6205e-09 | 6.7193e-05 | 65.35 |
| (3) the dual problem | 0 | `NumericalError` | no | 0.011370773240158904 | -3.9306e-05 | 3.8841e-06 | 4.7718e-05 | 1.7757e-02 | 49.99 |
| (4) first-order pass, then polish | 0 | `AlmostSolved` | no | 0.011267285124391934 | -1.6804e-09 | 4.6211e-07 | 4.5600e-08 | 2.8843e-06 | 72.80 |
| (5) nullspace parameterization | 0 | `AlmostSolved` | no | 0.011267293579460427 | -3.1947e-13 | 3.2497e-07 | 4.3831e-09 | 3.1812e-06 | 71.79 |
| (5) nullspace parameterization | 2 | `AlmostSolved` | **yes** | 0.011267149379466363 | -2.2232e-14 | 3.8856e-08 | 9.2657e-09 | 7.5069e-07 | 75.19 |
| (6) homogeneous frontier scale `1e-1` | 0 | `NumericalError` | no | 0.011277128311639005 | -1.1933e-10 | 3.0082e-05 | 3.4950e-09 | 1.7213e-04 | 60.83 |
| (6) homogeneous frontier scale `1e+1` | 0 | `AlmostSolved` | no | 0.011267327702216327 | -2.2554e-13 | 4.8213e-07 | 1.1176e-08 | 2.5012e-06 | 76.06 |
| (6) homogeneous frontier scale `1e+1` | 2 | `AlmostSolved` | **yes** | 0.011267148871837981 | -2.2412e-14 | 3.8195e-08 | 9.0950e-09 | 7.3793e-07 | 74.39 |

### Set C, the `t = 5.086270` saved failing stage

| formulation | profile | status | admitted | objective | min scaled eigenvalue | independent feasibility | relative duality gap | s |
|---|---:|---|---|---:|---:|---:|---:|---:|
| recorded stage, as the selector calls it | ladder | `PrimalInfeasible` | no | 5.3526e-22 | -0.25537193822 | 0.49999999993 | infinite | 46.32 |
| recorded program | 2 | `PrimalInfeasible` | no | 1.8755e-23 | -0.25537193825 | 0.25537193825 | 0.49999999979 | 53.40 |

### What each alternative is, and what it did

1. **Row and column equilibration of the affine map, with the whitening
   absorbed into the variables.**  The row-and-column equilibration of a
   positive semidefinite block is a diagonal congruence `G -> E G E`, not a
   scaling of its svec rows, since only a congruence maps the cone onto
   itself; `E` is 20 symmetric Ruiz sweeps on the entrywise magnitude map
   `max(|base|, max_i |C_i|)` and the frontier columns are scaled by the
   inverse square root of each coefficient's Frobenius norm.  The whitening is
   absorbed by carrying `t = U R z` with the identity objective.  At set A the
   equilibration is mild -- congruence condition `1.44`, column condition
   `2.29` -- and the pair is *worse* than whitening alone at both profiles
   (`1.2748e-05` against `3.2661e-06` at profile 0).  At set B it is worse
   still: `NumericalError` at profile 0 and, uniquely among the profile-2
   rows, **not admitted** at profile 2 (`3.8807e-06`).  Split out, the Gram
   congruence alone is the single most harmful change tried
   (`7.0634e-04` at set A, worse than the recorded `3.5740e-04`) and the
   column scaling alone is a mild improvement (`5.3115e-05`).  Dropped.
2. **Declared psd inflation of the outer cone, `1e-11` to `1e-8`, acceptance
   read on the uninflated Gram.**  The constraint becomes
   `G(z) >= eps I`; the reported minimum scaled eigenvalue is still
   `extension.scaled_matrix(extension.matrix(...))`, which the test
   `test_psd_inflation_acceptance_is_read_on_the_uninflated_gram` pins.  At
   `1e-11` and `1e-10` nothing moves (`3.5759e-04`, `3.5761e-04` against
   `3.5740e-04`); at `1e-9` and `1e-8` the gap gets two to three times worse
   and the reconstructed Gram becomes *more* negative (`-5.396e-09`,
   `-5.390e-09` against `-2.590e-09`), because the solve stops early on the
   inflated cone and the point it returns is further inside the inflated one
   but no better on the true one.  Dropped.
3. **The dual problem.**  The eliminated program's Lagrangian dual,
   `max -0.5 z^T P z - b^T y` subject to `P z + A^T y = 0` and `y` in the
   product cone, handed to Clarabel as a primal in `(z, y)` with 7974
   variables and 7974 cone rows.  Because `P` has rank 15, the dual's own `z`
   is determined only modulo the nullspace of `P`, so the frontier point is
   read from the zero-cone block's multiplier; all three candidate readings
   are scored and the best-feasibility one reported (it was the negated
   multiplier at both sets).  It is the worst formulation tried: minimum
   scaled eigenvalue `-3.92e-05`, relative duality gap `1.77e-02`, at the same
   wall time.  Dropped.
4. **A first-order pass, then a Clarabel polish shifted to the point.**  The
   first-order pass is a numpy alternating-direction split of the eliminated
   program into one Cholesky solve in the 238 frontier coordinates, one
   eigenvalue projection of the 120-row realified Gram, and one box clip; 400
   iterations cost `0.380 s` and reach primal residual `3.364e-05`, dual
   residual `1.933e-04` and minimum scaled eigenvalue `-3.283e-05`.  The
   polish is the displacement program with linear term `P z0` and cone
   right-hand side `b - A z0`, which is the shift
   `_solve_explicit_target_qp` already documents for its named warm refinement
   pass.  At profile 0 it reaches `2.5779e-06`, a factor 139 better than the
   recorded first pass and still rejected; at profile 2 it is admitted at
   `7.3731e-07`, the same point every other profile-2 row reaches.  The
   first-order pass therefore contributes `0.38 s` of the `75.66 s` and
   nothing to the accepted point.  Kept only as a measured negative.
5. **Nullspace parameterization.**  The only affine equality is the image
   equality `q = R z`, whose nullspace parameterization is the substitution
   itself, leaving Clarabel the positive semidefinite cone and the box and
   `P = R^T M R`.  It removes 15 variables and 15 zero-cone rows and improves
   the first pass by a factor 2.2 on the gap at set A (`1.6399e-04`) and by a
   factor 213 at set B (`3.1812e-06`, `AlmostSolved`); admitted at profile 2
   at both sets.  Kept as a named option: it is the cheapest of the
   improvements and never hurt.
6. **Homogeneous rescaling of the frontier variables by the box widths.**  The
   recorded box is `|z_i| <= 1` in every one of the 238 standardized frontier
   coordinates -- the test
   `test_homogeneous_box_rescaling_returns_the_same_witness` pins the
   right-hand side to `ones` -- so rescaling each frontier variable by its own
   box half-width is the identity map and the only content of a homogeneous
   rescaling is one common factor.  A factor `1e-1` makes everything worse
   (`9.3697e-03`); a factor `1e+1` is the best first pass of any formulation
   tried at set A (`2.5172e-06`, `AlmostSolved`) and is admitted at profile 2
   at both sets, with the widest margin of all (`7.3663e-07` and
   `7.3793e-07`).  Kept as a named option.

Reading across the three sets: at profile 0 the formulation matters a great
deal -- the relative duality gap spans `9.3697e-03` to `2.5172e-06`, three and
a half decades -- and no formulation is admitted.  At profile 2 the
formulation matters almost not at all: all six set A rows are admitted and
their gaps lie between `7.3663e-07` and `8.0852e-07`, a 10 percent spread;
four of the five set B rows are admitted, between `7.3793e-07` and
`7.7240e-07`, a 5 percent spread, and the one rejected is the equilibrated
pair at `3.8807e-06`.  The margin to the unchanged `1e-6` acceptance is 19 to
27 percent on every admitted row, which is thin; it is the same margin at both
preparations, so it is a property of the program rather than of the state.

## (c) The callable

`apcm_outer_selector_formulations.solve_outer_target_with_tightened_retry`
takes the four positional operands `_solve_explicit_target_qp` takes
(`extension`, `lower_moments`, `scaled_readout`, `target_metric`) and returns
a `GuardConicResult` with the same fields and the same acceptance, so the
projective-face agent or a later round can wire it as a named selector option
inside `select_outer_feasible_projective_guard` without changing anything
else.

- `ladder=True`, the default, runs the recorded first pass and then, whenever
  that pass is not accepted, the recorded tightened profile.  That is the
  recorded ladder with one difference: it does not condition the retry on the
  first pass's status.
- `ladder=False` runs the tightened profile alone, at half the wall time and
  the same accepted point.
- `whiten_image=False` is the default, the recorded assembly.  `True` carries
  the image in whitened coordinates.
- `frontier_weight=0.0` is the default; the declared frontier term is what
  round 2 needed a `2e-6` tolerance for and is measured above as a factor 1.5
  worse on the gap.

Tests (`paper_5/tests/test_apcm_outer_selector_formulations.py`, 14 tests,
`1.75 s`):

| test | what it holds |
|---|---|
| `test_tightened_retry_target_stage_on_the_strict_fixture` | the callable on the strict fixture `test_apcm_projective_guard` uses -- the `MatrixDimerState` with `0.5 I` electron density, unit phonon density and zero elsewhere, 15 entrance keys -- where `select_projective_guard` returns a *certified* target stage: the callable is accepted, returns the `GuardConicResult` shape with the extension's own frontier keys and Gram dimension, and selects the same invariant target image to `1e-7` and the same objective to `1e-9` |
| `test_tightened_retry_target_stage_reaches_the_one_shell_outer_cone` | the same callable on the one-shell outer cone built from that fixture, for both `ladder` settings and both `whiten_image` settings |
| `test_recorded_kernel_reproduces_the_recorded_target_stage_bitwise` | the default formulation reproduces `_solve_explicit_target_qp` bitwise on the frontier point, and identically on status, objective, acceptance and minimum scaled eigenvalue |
| `test_default_formulation_states_the_recorded_program` | every field of `OuterTargetFormulation()` is the recorded program, listed explicitly |
| `test_recorded_program_matches_the_recorded_variable_and_row_order` | the assembled `(P, q, A, b, cones)` carry the recorded variable order `(z, q)` and row order (positive semidefinite, box, image equality) |
| `test_target_image_quadratic_is_an_orthogonal_projector` | `R^T M R` has `readout.rank` unit eigenvalues and the rest zero, and `U R` has orthonormal rows |
| `test_the_recorded_retry_ladder_does_not_fire_on_numerical_error` | the recorded retry trigger set is the three statuses and does not contain `NumericalError` |
| `test_psd_inflation_acceptance_is_read_on_the_uninflated_gram` | the reported minimum scaled eigenvalue of an inflated solve equals the eigenvalue recomputed from the returned frontier moments on the uninflated Gram |
| `test_homogeneous_box_rescaling_returns_the_same_witness` | the recorded box half-widths are all exactly one, and the rescaled solve returns the same objective |
| the remaining five | the Ruiz congruence equilibrates a symmetric magnitude, the first-order pass stays inside the box and the cone, the polish and the dual agree with the recorded objective, the operands round-trip to a saved run directory, and the from-program entry point gives the same operands as the builder entry point |

## Gates

| gate | result |
|---|---|
| the test file I own | 14 passed in `1.75 s` |
| the whole `paper_5` suite under the 5000 MB guard | 425 passed, 9 warnings in `188.23 s`, peak `1413 MB` |
| defaults unchanged | no existing file was edited; every new behaviour is a field of the frozen `OuterTargetFormulation`, whose default instance is asserted field by field to be the recorded program, or a keyword of the new callable |
| recorded-path reproduction | the recorded stage at set A reproduces round 1's and round 2's `NumericalError`, objective `0.011286373848805029`, minimum scaled eigenvalue `-2.5903862151141617e-09`, stationarity `6.411437065053453e-05` and relative duality gap `3.5739542823707714e-04` to every printed digit; at set B it reproduces round 2's stage-one `NumericalError` |
| memory | every solve ran under `ram_guard --limit-mb 5000`, one at a time; the largest reported peak was `4.10 GB` and no run was killed |

## Kept

- `apcm_outer_selector_formulations`, with the operand extractor, the recorded
  program assembly, the scoring of any candidate point on that program, the
  frozen `OuterTargetFormulation` whose default is the recorded program, and
  the three standalone formulations (dual, first-order pass plus polish, and
  the named combinations).
- `solve_outer_target_with_tightened_retry`, the drop-in callable.
- The named options `eliminate_image_equality` and
  `homogeneous_frontier_scale`, which improve the first pass and never hurt.
- The negative results above, which the record should carry: the
  equilibration, the psd inflation, the dual and the first-order pass are all
  measured and all worse than doing nothing.

## Dropped

- Every one of the six alternative formulations as *the* answer.  The measured
  answer is the recorded routine's own tightened profile, which the recorded
  retry ladder never reaches because its trigger set omits `NumericalError`.
- The Ruiz Gram congruence, which is the single most harmful change tried and
  is the only profile-2 row not admitted at set B.
- The psd inflation at every decade from `1e-11` to `1e-8`.
- The dual problem.
- The declared frontier term at the tightened profile: it makes the gap a
  factor 1.5 worse and is what made round 2's acceptance need `2e-6`.

## Blockers

1. The `t = 5.086270` post-pulse stage is `PrimalInfeasible` on the one-shell
   outer cone at both profiles, with reconstructed Gram minimum scaled
   eigenvalue `-0.25537193825`.  That is not a solver failure and no
   formulation, profile or acceptance tolerance touches it: those retracted
   lower moments do not extend to the one-shell outer Gram inside the box.
   Whatever stops the projective rollout between `t = 5` and `t = 6` has to be
   fixed upstream of the selector, in what produces the lower moments.
2. The admitted margin is 19 to 27 percent on the relative duality gap at both
   preparations, and nearly identical across formulations, so it is a property
   of the program.  Widening it needs either a change to the acceptance rule, which
   is the user's, or a genuinely better-conditioned program than any of the
   six tried here.
3. The accepted witness sits on the box face: outer envelope ratio
   `0.99946` and current envelope ratio `0.99274` at set A (`0.99946` and
   `0.99154` at set B), against `ProjectiveOuterEventGuard._guarded_completion`'s
   `envelope_fallback` threshold `1 - 1e-8`.  It passes, but the margin is
   `7.3e-03`, so a stage whose witness is a little further out will be
   refused by the envelope test rather than by the acceptance test.
4. Cost is unchanged: `75 s` for the tightened pass alone and about `125 s`
   for the full ladder, against the record's "not an executable controller" at
   `171 s` per outer solve and three stages per step.  An accepted witness per
   triggered stage is now available; an every-stage outer guard still is not.

## Comparability in a shared worktree

`apcm_projective_guard.py` (`03efd75c341235afa764d9ee5c07dfd1`),
`apcm_positive_extension.py` (`fc799d8589fb416d665e2c048d5b6498`) and
`matrix_reference.py` (`e62409317773b7378b97ef5721f7e2bb`) were unchanged
across all 47 measurements; those are the three modules every operand and
every score passes through.  `projected_apcm.py` changed four times during the
window (another agent's), and it enters only set B's operand build, through
`prepare_projected_apcm_initial_state` and `FixedDictionaryProjectedAPCM.evaluate`.
Set B's recorded row was therefore rerun at the end of the round under the
tree as it then stood
(`output/local_runs/paper_v_outer_selector_formulations_stage_recheck_20260915_v1/`):
objective `0.011298297580589320`, minimum scaled eigenvalue
`-3.444450008528313e-09`, stationarity `1.2161555618802002e-04`, relative
duality gap `6.781291143889349e-04`, every printed digit of the opening run.
All rows in this report are therefore mutually comparable and comparable with
rounds 1 and 2.

## Artifacts

| directory | contents |
|---|---|
| `output/local_runs/paper_v_outer_selector_operands_prep_20260915_v1/` | set A operands, plus the three harness scripts |
| `output/local_runs/paper_v_outer_selector_operands_stage_20260915_v1/` | set B operands |
| `output/local_runs/paper_v_outer_selector_operands_t5086_20260915_v1/` | set C operands |
| `output/local_runs/paper_v_outer_selector_formulations_prep_20260915_v1/` | `formulations.jsonl` and the frontier point of every set A row |
| `output/local_runs/paper_v_outer_selector_formulations_stage_20260915_v1/` | the same for set B |
| `output/local_runs/paper_v_outer_selector_formulations_t5086_20260915_v1/` | the same for set C |
| `output/local_runs/paper_v_outer_selector_formulations_stage_recheck_20260915_v1/` | the end-of-round comparability recheck |

No `detect_changes` run: this worktree is not in the GitNexus index (only the
main checkout is), and nothing was committed.
