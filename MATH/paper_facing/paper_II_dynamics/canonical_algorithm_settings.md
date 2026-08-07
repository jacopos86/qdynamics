# Paper II Canonical Algorithm Settings

Status: agent-facing canonical-settings specification for Paper-II/APM diagnostics.

This file is different from `runtime_algorithm_settings.md`.

- `runtime_algorithm_settings.md` inventories knobs the code can expose.
- This file records which parameter choices are currently canonical, candidate-canonical, or diagnostic-only for Paper-II APM runs.

Do not treat this file as a paper-promotion decision. It is the working contract for reproducible diagnostics while settings are tuned by run evidence.

## Status Labels

| Label | Meaning |
|---|---|
| Active canonical | Use this unless a run explicitly studies another setting. |
| Candidate canonical | Current best-supported candidate, still being tuned or stress-tested. |
| Diagnostic-only | Useful for comparison or failure isolation, not a default. |
| Retired / archaic | Kept only for old artifact interpretation or legacy tests. |

## Current Diagnostic Anchor

The current weak-weak kink diagnostic surface is:

`output/pdf/ap_mclachlan_weak_weak_snake_progress_diagnostic.pdf`

Relevant rows after the Paper-II state-space repair implementation:

| Run | Setting delta | Evidence status |
|---:|---|---|
| 38 | Current-code no-repair baseline | Diagnostic baseline for the old Euler ridge `1e-7` kink issue. |
| 39 | Paper-II state repair, no forced subdivision | Shows local kink-window detection and finite-rung response, but subdivision did not fire. |
| 40 | Paper-II state repair + kink subdivision | Shows the useful current behavior: repair remains local near the old kink window and local subdivision fires there. |
| 41 | Stress ridge `1e-8`, no repair | Deliberately less-stabilized inverse; produces a much larger kink. |
| 42 | Stress ridge `1e-8` + state repair | Strongly reduces the stress kink, but all local subdivisions used depth 1. |
| 43 | Stress ridge `1e-8` + severity-scaled repair | First successful visible cure of the deliberate Run-41/42 kink; uses deeper local subdivision for larger state-motion/kink defects. |
| 44 | Stress ridge `1e-8` + rho-free subdivision severity | Confirms the visible Run-43 cure was not caused by using `rho_num` alone as a local-subdivision request. Superseded by the Paper-II scheduler patch for future reruns. |
| 45 | Stress ridge `1e-8` + Paper-II scheduler | Current faithful scheduler: active lane set was temporal kink only, inverse breadth stayed zero, local subdivision breadth was capped at four, and the visible kink cure remains. |

Run 40 supports the current stabilization implementation as useful: it reduces the largest `theta_dot_l2` jump and state-space kink jump relative to Run 38 while acting locally near the old \(t \approx 2.3\) to \(2.5\) defect window. It does not yet prove the settings are final, because the repaired checkpoints are still marked unsupported.

## Baseline APM Diagnostic Settings

These are the current baseline choices for the weak-weak seed-1 diagnostic lane unless a run label says otherwise.

| Parameter group | Canonical setting | Status |
|---|---|---|
| Static seed | Paper-I weak-weak SNAKE seed replayed through `replay_family` from the reconstructed-state diagnostic seed artifact | Active canonical for this diagnostic lane |
| Drive | `enable_drive=True`, `drive_A=0.6`, staggered density drive | Active canonical for the current weak-weak driven diagnostic lane |
| Drive-aligned ANZATS augmentation | `drive_aligned_ansatz=True` | Active canonical for driven variational APM |
| Time grid | `t_final=3`, `num_times=601`, `dt=0.005` | Active canonical for kink diagnostics |
| Integrator | `euler` | Active canonical for isolating the known kink issue; RK4 remains a diagnostic comparison |
| Inverse baseline | `ridge_lambda=1e-7`, `pinv_rcond=1e-10`, `solve_damping=0` | Active baseline for reproducing the kink issue |
| Parameterization | `per_pauli_term` | Active canonical default |
| Support-patch policy | unified exchange-family finalist set: stay / pure append / pure prune / true exchange. Pure append and pure prune are limiting cases of the same selector. | Active canonical default |
| Append mode | combinatorial support-atom append ladder, max batch size `10` as a rung upper bound, first append allowed at `t=0.005` | Active canonical for this diagnostic lane |
| Append search trigger | `residual_ratio_threshold=1e-3` | Active canonical. This is intentionally not a machine-precision threshold; below this normalized residual ratio the controller should stop spending measurements/search on append. |
| Append granularity | Pauli-child / polynomial-term support atoms under `per_pauli_term` | Active canonical default |
| Append occurrence policy | `layer_reuse` | Active canonical default. A base pool atom may be appended again at a later ANZATS layer; one proposed batch may contain that base atom at most once. |
| Append ranking | append gain divided by Paper-I proxy cost denominator, `append_cost_alpha=1.0`, `family_robust_v1` normalization | Active canonical default |
| Append admission conditioning | Schur novelty enabled; full Schur rank required, Schur condition cap `1e12`, novelty ridge `0`; pre-commit augmented-solve confirmation always on | Active canonical default |
| Append parent-scout frontier | optional `parent_tangent_schur_gain` macro-parent scout with a finite parent cap, then expansion back to Pauli/poly-child atoms | Candidate canonical search-budget route |
| Macro-generator append | `logical_shared` / macro support atoms only when explicitly requested | Diagnostic-only until no-harm and Schur-conditioning stress tests support it |
| Prune | enabled inside the unified support-patch family for current append/prune/exchange diagnostics; commit requires current safety gates and a finite commit budget | Active canonical default for support-patch diagnostics |
| Numerical stabilization with support patching | `solve_repair=true` with local subdivision and the canonical state-space candidate policy | Active canonical default whenever append, prune, or exchange is enabled. Disable only for an explicitly requested no-repair ablation. |
| Prune nomination conditioning | conditioning-aware, history-aware prune score available in the active lane; grouped deletion loss remains deletion authority and conditioning only changes prune ranking. | Candidate canonical tuning surface |
| Results pdf cost row | final active ansatz at the row's terminal trajectory time, compiled structurally with a recorded support digest and Qiskit backend settings | Active reporting invariant |
| Exact/reference trajectory | post-run reporting only | Active invariant |

The dataclass/CLI raw defaults for prune remain off so an ordinary append smoke
does not delete support by accident. Canonical support-patch diagnostics are
defined by the profile and flags passed to the run: they enable prune scouting,
and they enable prune commit only when the run is explicitly testing deletion
or exchange. Do not infer the current canonical route from raw CLI defaults
alone.

## Canonical Support-Patch Controller Direction

The active canonical controller direction is a unified, combinatorial
support-patch family at Pauli/poly-child granularity. The finalist set contains
stay, pure append, pure prune, and true exchange. Append-only and prune-only are
limiting cases of this selector, not separate canonical routes.

| Choice | Canonical setting | Status |
|---|---|---|
| Pure support growth name | `append` | Active canonical. `insert` is legacy payload language or future internal-placement language only. |
| Data-structure placement | tail append to the ANZATS/runtime coordinate list | Active canonical. |
| Default append ladder | combinatorial rung search | Active canonical. |
| Default append atom | Pauli child / polynomial term under `per_pauli_term` | Active canonical. |
| Default append occurrence policy | `layer_reuse` | Active canonical. Repeated noncommuting product-ANZATS occurrences receive distinct runtime identities; `unique_support` is compatibility-only. |
| Default append trigger | normalized McLachlan residual ratio above `1e-3` | Active canonical. Prevents repeated append searches for residuals that are nonzero only relative to an unrealistically strict machine-zero target. |
| Default append score | `append_gain / paper_i_proxy_denominator^append_cost_alpha` | Active canonical. Cost acts as a denominator/weight, not as a subtractive penalty. |
| Default append conditioning | checkpoint-local Schur novelty guard plus pre-commit augmented-solve confirmation | Active canonical. Rejects rank-deficient candidate batches, extremely ill-conditioned Schur blocks, and non-finite augmented solve geometry. |
| Default append parent frontier | optional parent/macro scout, then child expansion | Candidate canonical search-budget route. The scout may reduce the child frontier submitted to the append ladder, but the accepted patch remains child-level. |
| Sector legality | Paper-I legal-subspace append guard enforced when metadata is available | Active canonical. |
| Prune | enabled in the support-patch family | Active canonical. Current deletion loss, persistence, cooldown, ray-distance, differential-miss, and patch-smoothness gates remain authoritative. |
| Prune conditioning | relief/history/damage weights tune nomination only | Candidate canonical tuning surface. Conditioning can make a redundant block more prune-attractive but cannot replace current deletion safety. |
| Prune patch smoothness | enabled for deletion-containing patches | Active canonical. Same-checkpoint state-space velocity jumps defer unsafe deletions rather than committing or blacklisting them. |
| Prune history transition | atom-level history and cooldowns survive append-only support changes for still-active atoms; support-dependent loss/conditioning/smoothness records clear after support mutation | Active canonical. Prevents append from resetting prune persistence while preserving current-support safety. |
| Exchange | enabled when append and prune scouts are both available | Active canonical. True exchange must pass joint patched-support utility and inherited prune-side safety. |
| Internal insertion between existing generators | disabled / not implemented as canonical route | Planned only. |
| Macro append | explicit diagnostic setting | Diagnostic-only after weak-weak A=0.8 runs showed collective macro patches can damage energy dynamics. |

Paper II's Schur novelty criterion and augmented-solve confirmation are now
active for append batches. The candidate batch is projected against the current
tangent support at the current checkpoint; accepted batches must retain the
configured Schur rank fraction and must not exceed the configured Schur
condition cap. Before commit, the augmented support solve is recomputed under
the same retained-support/ridge/damping convention used for propagation, and
its residual telemetry must be finite. The novelty ridge is `0` by default so
the propagation ridge cannot hide duplicate tangent directions. Macro-generator
batching remains diagnostic-only until this guard plus future no-harm checks
show it is stable enough to become canonical.

The new append parent-scout frontier is different from macro append. In
`per_pauli_term` mode it constructs only a temporary parent tangent for ranking
parents, then expands retained parents back into Pauli/poly-child atoms before
the normal child-level append ladder and support-patch selector run. It should
be tested next as a search-budget diagnostic, not assumed canonical. The first
test should compare the non-scout append ladder to
`parent_tangent_schur_gain` with fail-open behavior enabled and should report
parent counts, child counts before/after filtering, selected child labels,
Schur novelty, augmented-solve confirmation, and whether the score was marked
measurement-saving. `full_child_block_diagnostic` remains diagnostic-only and
must not be used as evidence that the macro layer saved measurements.

The append ladder treats `max_append_batch_size` as a maximum rung, not as a
forced batch cardinality. A run labeled `max batch <= 5` may admit singleton,
pair, triple, quadruple, or quintuple batches depending on the checkpoint-local
score and Schur guard. Runtime enumeration budgets such as
`append_rung_set_cap` and `append_prefilter_size` are implementation controls
for finite local diagnostics; they are not part of the Paper-II mathematical
admission rule and should not be used to describe the algorithmic criterion.

The canonical occurrence policy is `layer_reuse`. Selecting a Pauli/poly child
does not consume that base pool atom for the rest of the trajectory: the same
operator can be admitted later at a new tail position, where intervening
noncommuting factors generally give it a different tangent direction. Runtime
labels therefore distinguish occurrences, while candidate batches remain sets
of base atoms and cannot contain two copies of the same base atom in one patch.
The older `unique_support` policy is retained only for compatibility ablations.

Failed append-search reuse is available only as an explicit diagnostic option.
It is disabled by default. When enabled, a failed full append ladder search can
store a local geometry certificate and skip repeated full rescoring until the
certificate drift reopens the ordinary ladder. The active implementation
supports both Paper-II routes: direct threshold reopening and model-change
reopening with direct fallback when no useful secant utility scale has been
measured. Candidate-level secant waits are advisory telemetry only; they do not
trigger new measurements or suppress a reopened full search. This mechanism is
not a maximum total append limit.

Every Results-pdf row used for current diagnostic interpretation must carry the
final-time Qiskit cost of the active APM ansatz. This cost is a terminal-support
compiled-resource descriptor (`N2q`, `D2q`, `Dc`) and is separate from both the
APM controller and Qiskit-community comparator trajectories. It must never be
used for online support-patch decisions, and it must not be hand-filled without
a cost sidecar or equivalent machine-readable provenance.

For prune-enabled diagnostics, a candidate deletion batch that passes grouped
deletion loss and persistence must also pass same-checkpoint patch smoothness.
The active implementation compares the pre-prune McLachlan velocity with the
zero-transport pruned-support velocity in Hilbert space. If the normalized
patch velocity severity exceeds one, the batch is recorded as
smoothness-deferred and placed on a severity-scaled cooldown. A later retry can
occur only through the normal prune scorer after cooldown; repeated tests of the
same batch may record an opportunistic trend, but the trend does not force extra
measurements. Trust-radius reduced-state refit remains a stricter future prune
rung and is not assumed by the current zero-transport implementation.

## Numerical Solve-Repair Candidate Settings

These settings implement the current Paper-II solve-repair policy. They are candidate canonical for the numerical stabilization lane.

| Parameter | Current value | Status | Notes |
|---|---:|---|---|
| `solve_repair` | true | Active canonical for support-patch runs | Enabled whenever append, prune, or exchange is active. Disable only for an explicitly requested no-repair ablation. |
| `condition_number_max` | `1.705e7` in Run 40 | Candidate canonical | Soft kappa warning scale in scoring, not a repair trigger. |
| `condition_number_fail` | `None` | Active canonical for noiseless diagnostics | Hard kappa rejection is off outside explicit strict finite-shot validation. |
| `strict_finite_shot_validation` | false | Active canonical for noiseless diagnostics | Finite-shot mode is not the current local diagnostic setting. |
| `rho_num_max` | `1.0` in Run 40 | Candidate canonical for kink isolation | Keeps `rho_num` from globally dominating this diagnostic; `rho_num` remains a trigger when the threshold is intentionally tightened. |
| `state_motion_l2_step_max` | `5e-2` | Candidate canonical | Bounds both the tangent-linear interval motion and the realized prospective trial-state ray distance; the value is unchanged. |
| `state_space_kink_eta_max` | `1e-2` | Candidate canonical | Temporal state-space kink guard. |
| `local_subdivision_enabled` | true | Candidate canonical | Required for Run 40 behavior. |
| `max_local_subdivisions` | `4` | Candidate canonical | Current local replay cap. The Paper-II scheduler uses `q_k` to choose the first attempted depth only when the state-motion or temporal-kink lane is active. |
| `local_subdivision_factor` | `2` | Candidate canonical | Bisects intervals at each depth; severity depth is computed using this factor. |
| `min_local_dt` | `1e-6` | Candidate canonical | Safety floor. |
| `release_kink_threshold_scale` | `0.5` | Candidate canonical | Stricter return-to-base release threshold. |
| `release_patience_min` | `1` | Candidate canonical | Minimum release passes. |
| `release_patience_max` | `5` | Candidate canonical | Maximum release passes. |
| `release_kink_severity_scale` | `4` | Candidate canonical | Severity-dependent patience scale. |
| `ridge_ladder` | `1e-7,3e-8,1e-8,0,3e-7,1e-6,3e-6,1e-5` | Candidate canonical | Candidate set values, not ordered first-pass list. |
| `pinv_rcond_ladder` | `1e-10,1e-11,1e-12,1e-9,1e-8,1e-7` | Candidate canonical | Candidate set values, not ordered first-pass list. |
| `solve_damping_ladder` | `0` | Candidate canonical for Run 40 | Damping is not the first cure for the current kink defect. |

## Diagnostic-Only / Retired Settings

| Setting | Status | Reason |
|---|---|---|
| Raw `theta_dot_l2_max` repair guard | Retired / diagnostic-only | Coordinate-size guard is not the Paper-II state-space repair criterion. |
| Ordered first-passing solve-repair guard list | Retired / archaic | Paper II uses finite candidate set evaluation plus least-intervention state-space scoring. |
| Kappa as a global repair trigger | Retired for noiseless diagnostics | Kappa is a soft response-risk modifier except in explicit strict finite-shot validation. |
| Repair by artificial damping before local subdivision | Retired for kink-like defects | Paper II prefers local subdivision for temporal/state-motion defects when the solve is otherwise finite. |

## Current Interpretation

Run 40 is positive evidence that the current stabilization implementation is helpful: the repair path identifies the old kink window locally, applies local subdivision there, and improves state-space smoothness telemetry relative to the no-repair baseline.

Run 40 is not final evidence that the canonical settings are complete. The remaining unsupported checkpoint markers mean the next tuning question is whether the acceptability thresholds, candidate set, or local replay depth should be refined so the same local defect can be supported rather than merely improved.

Run 43 shows that making local subdivision depth proportional to response severity can cure the visible stress kink that remains in Runs 41 and 42. Run 45 is the current implementation check for Paper II's scheduler: `rho_num` alone widens inverse-policy candidate breadth, while local subdivision is requested only when the state-motion or temporal-kink lane is active. In Run 45, the stress defect entered through the temporal-kink lane, so inverse breadth remained zero and the correction used local subdivision only.
