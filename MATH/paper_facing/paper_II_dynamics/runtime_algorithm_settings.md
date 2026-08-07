# Paper II Runtime and Algorithmic Settings Inventory

Status: agent-facing Paper-II support document.

This file tracks the high-level runtime settings for Paper-II time dynamics.
Executable defaults remain in the code and in locked class-setting manifests.
When this document disagrees with a runner, dataclass, manifest, or test, fix
the source of truth first and then update this inventory.

Primary code surfaces:

| Surface | Path |
|---|---|
| Fixed AP-McLachlan runner | `pipelines/time_dynamics/runners/ap_fixed_from_adapt_artifact.py` |
| Append-enabled AP-McLachlan runner | `pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py` |
| AP state / ANZATS handoff | `pipelines/time_dynamics/ap_mclachlan/state.py` |
| AP geometry and solve | `pipelines/time_dynamics/ap_mclachlan/geometry_eval.py`, `fixed_step.py`, `inverse.py` |
| AP append/prune trajectory | `pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py` |
| Support atoms, frontier, and support-patch scoring | `pipelines/time_dynamics/ap_mclachlan/support_atoms.py`, `support_frontier.py`, `support_patch.py` |
| Hamiltonian and drive adapter | `pipelines/time_dynamics/ap_mclachlan/hamiltonian.py`, `pipelines/time_dynamics/adapters/drive_terms.py` |
| Static ANZATS loader | `pipelines/scaffold/runtime_loader.py` |
| Paper-II class settings lock | `chtc/generic_time_dynamics_table/input/class_settings/paper_ii_class_settings_lock_v1.json` |

## Runtime Layers

| Layer | Main settings | Current role |
|---|---|---|
| Static seed handoff | `artifact_json`, `loader_mode`, `generator_family`, `fallback_family`, candidate-pool mode | Loads the Paper-I ANZATS, state, and candidate universe for Paper II. |
| Legal subspace append guard | Paper-I `adapt_pool_legal_subspace_filter` metadata | Enforced candidate-pool guard for AP append. Dropped/leaking terms are not append candidates, and grouped-risk macro generators are not split into Pauli children. |
| Time grid | `times`, `t_final`, `num_times` | Defines recorded time points and integration intervals. |
| Variational coordinates | `parameterization_mode` | Chooses per Pauli / polynomial term coordinates or per logical / macro generator coordinates. |
| McLachlan inverse | `pinv_rcond`, `ridge_lambda`, `solve_damping`, `epsilon`, `policy_id` | Defines the retained eigenspace, ridge, and damped solve convention for `theta_dot`. |
| McLachlan solve repair | `solve_repair`, condition cap, `rho_num` cap, state-motion cap, local subdivision, release patience, ridge / `pinv_rcond` / damping ladders | Canonical for append/prune/exchange support-patch runs and optional for fixed-support diagnostics. Checkpoint repair evaluates a finite inverse-policy candidate set; interval repair can replay an advance with local substeps. |
| Integrator | `integrator` | Advances `theta` between adjacent time points. |
| Drive | `enable_drive`, drive profile fields, drive operator resolution | Builds `H(t)=H0+c(t)D`. |
| Drive-aligned ANZATS augmentation | `drive_aligned_ansatz` | Adds zero-angle `D` as an ansatz generator for driven variational methods. |
| Runtime observables | HH site occupations, spin-resolved site occupations, doublon, staggered density, primary density | Evaluated from the already prepared AP state or already propagated exact state as one bundled diagnostic observable set; not separate serial trajectory reruns. |
| Fixed AP | fixed-support runner fields | Propagates on the inherited support after optional drive-aligned augmentation. |
| Legacy singleton append/prune AP | `AppendControllerConfig` fields | Compatibility path. Supports singleton append and singleton prune when explicitly selected. |
| Canonical support-patch controller | `SupportPatchControllerConfig` exchange-family subset | Default combinatorial support-patch family. At each checkpoint the finalist set is stay / pure append / pure prune / true exchange. Append and prune scouts nominate branches; the final decision is made on the full patched support. |
| Append parent-scout frontier | `append_macro_scout_enabled`, `append_macro_scout_score_mode`, `append_macro_scout_parent_cap` | Optional parent-generator frontier layer. Retained parents are expanded back into Pauli/poly-child atoms before the actual child-level support-patch decision. It does not change final granularity to macro generators. |
| Classical support-patch scoring workers | `support_patch_scoring_workers` | Computational speed knob only. Candidate scores within one append/prune rung or exchange-pair list may be evaluated with ordered threads; time stepping, cost normalization, support mutation, safety gates, and selection remain serial. |
| Reference diagnostics | `reference_energy_json`, `reference_energy_atol` | Post-run reporting only. The reference cache can carry energy plus HH site/doublon observables from the same exact-state propagation. Never controller input. |
| Results pdf reporting | trajectory JSONs, report manifest, final-time Qiskit cost sidecar | Diagnostic review artifact for weak-weak APM progress. Each completed row must carry the final active ansatz cost at the terminal trajectory time. This is not controller input and not a Qiskit comparator trajectory. |

## Shared AP Runner Flags

These flags are shared by fixed AP and append-enabled AP unless noted.

| Flag / field | Default | Meaning |
|---|---:|---|
| `--artifact-json` | required | Static seed artifact consumed by the runtime loader. |
| `--output-json` | required | Output trajectory JSON path. |
| `--loader-mode` | `None` | Loader policy. Common values include replay-family style modes and fixed-support compatibility modes. |
| `--tag` | `None` | Loader/run tag propagated into provenance. |
| `--generator-family` | `match_adapt` | Family used to reconstruct the static ANZATS generator pool. |
| `--fallback-family` | `full_meta` | Fallback generator family used by the loader when allowed by the loader contract. |
| `--times` | `None` | Explicit comma-separated time grid. Overrides `--t-final` and `--num-times`. |
| `--t-final` | `0.2` | Final time used when `--times` is omitted. |
| `--num-times` | `3` | Number of time points used when `--times` is omitted. |
| `--integrator` | `euler` | Explicit integrator. Choices: `euler`, `rk4`. |
| `--pinv-rcond` | `1e-10` | Relative retained-eigenvalue threshold for the supported inverse. |
| `--ridge-lambda` | `1e-7` | Ridge added before eigenspace thresholding. |
| `--solve-damping` | `0.0` | Extra denominator damping on retained eigenmodes. Default zero preserves the prior supported inverse. |
| `--parameterization-mode` | `per_pauli_term` | `per_pauli_term` or `logical_shared`. |
| `--enable-drive` | false | Enables runtime drive model construction. |
| `--drive-A` | `0.0` | Gaussian-sinusoid amplitude in `c(t)`. |
| `--drive-omega` | `1.0` | Gaussian-sinusoid angular frequency in `c(t)`. |
| `--drive-tbar` | `1.0` | Gaussian envelope width in `c(t)`. |
| `--drive-phi` | `0.0` | Phase in `c(t)`. |
| `--drive-pattern` | `staggered` | Spatial weight pattern for supported density-style drives. |
| `--drive-custom-weights` | `None` | Explicit spatial weights when `drive_pattern` uses custom weights. |
| `--drive-include-identity` | false | Keeps the scalar identity part of density drives when supported. |
| `--drive-time-sampling` | `midpoint` | Sampling convention for drive/reference helper paths. |
| `--drive-t0` | `0.0` | Time origin shift for drive helpers. |
| `--drive-n-sites` | inferred | Optional drive-site override checked against the resolved problem. |
| `--drive-ordering` | inferred | Optional ordering override checked against the resolved problem. |
| `--drive-aligned-ansatz` / `--no-drive-aligned-ansatz` | true | For driven AP, appends the resolved drive operator as a zero-angle ansatz generator before propagation or patch scoring. |
| `--reference-energy-json` | `None` | Optional post-run reference-energy trajectory. |
| `--reference-energy-atol` | `1e-12` | Tolerance for post-run reference-energy comparisons. |

## Runtime Observable Bundle

For HH/AP-McLachlan diagnostic runs, the active AP runners emit a bundled
observable snapshot in each `plot_rows` entry when the resolved problem supports
HH measurement observables. The fields are `n_up_site`, `n_dn_site`,
`site_occupations`, `doublon`, `staggered`, and `primary_density`, plus matched
`*_exact` and absolute-error fields when the reference cache contains the same
observable bundle.

The bundle is computed from the state vector already available at that point:
AP uses the McLachlan geometry state already prepared for the time point, and
exact reference generation uses the exact state already produced by the dense
reference propagator. This is a reporting/diagnostic layer only; it does not
spawn one trajectory per observable and does not feed append, prune, solve
repair, integrator choice, or online tuning.

## Static Seed and Candidate-Pool Settings

| Setting | Values / fields | Current behavior |
|---|---|---|
| Runtime loader | `load_scaffold_runtime_input(...)` | Builds `psi_ref`, `psi_initial`, selected terms, layout, runtime theta, optional logical theta, and candidate pool. |
| Prepared-state parity | loader parity checks | Stored `psi_initial` and reconstructed `U(theta_runtime) psi_ref` must agree unless a deliberately loud diagnostic reconstruction path is used. |
| Candidate-pool source | `candidate_pool_source.source_kind`, `completeness`, `pool_key` | Output state records whether append candidates came from a complete resolved pool or selected terms only. |
| Complete pool override | `--replay-candidate-pool-mode`, `--diagnostic-append-pool-mode` | Append-runner diagnostic smoke can request a replay-family pool. Treat this as diagnostic unless promoted by a later run contract. |
| Incomplete pool policy | `allow_incomplete_candidate_pool`, `--require-complete-candidate-pool` | Default legacy append path allows incomplete pools; stricter settings fail or no-op when the pool is incomplete. |
| Legal subspace append guard | automatic when source metadata is present | The loader removes candidates marked dropped by the Paper-I legal-subspace filter. In `per_pauli_term` mode, a macro generator marked as group-safe but child-leaking is blocked from Pauli-child append; in `logical_shared` mode, the same macro can remain available as one grouped generator. |

This guard is deliberately not counted as append/prune logic. It is a legality
filter on the candidate universe before AP scores support patches.

### Pool And Support-Atom Accounting

The AP route reports several related counts. They are not interchangeable.

| Runtime accounting field | Meaning |
|---|---|
| `selected_seed_terms` | Active seed support loaded from serialized `adapt_vqe.parameterization` when present. Fallback label matching is legacy/repair behavior for older artifacts without serialized layout. |
| `runtime_parameter_count` | Active AP coordinate count. In `per_pauli_term` mode this is the number of active Pauli/poly-child runtime coordinates. In `logical_shared` mode this is the number of logical/macro coordinates. |
| `candidate_parent_pool_terms` | Top-level candidate terms loaded or rebuilt from the replay/full-meta candidate pool before AP support-atom expansion. Reports must state whether this count is before or after legal-subspace filtering. |
| `all_pauli_child_atoms` | Pauli/poly-child atoms obtained by expanding `candidate_parent_pool_terms` under `per_pauli_term` before removing already-active atoms. |
| `active_pauli_child_atoms` | Pauli/poly-child atoms already present in the loaded seed state. |
| `available_append_atoms` | One-use/`unique_support` accounting: AP append atoms remaining after removing active support and applying legality/no-split filters. Retained for compatibility audits. |
| `reusable_append_frontier_atoms` | Canonical `layer_reuse` frontier: one next occurrence for every legal base pool atom. An already-active base atom remains eligible at a later ANZATS layer, but appears only once in any one proposed batch. |
| `macro_scout_child_count_before`, `macro_scout_child_count_after` | Child append frontier size before and after optional parent-scout filtering. A reduced count means only the finite search frontier was narrowed; committed patch atoms remain Pauli/poly children. |
| `paper_i_sidecar_pool_labels` | Paper-I report/sidecar labels used for audit and equivalence checks. These are not AP runtime candidate objects unless normalized into the same support-atom representation. |

Raw Paper-I sidecar labels are not the same object type as AP runtime support
atoms. Equivalence must be checked only after normalizing the sidecar labels
and AP append atoms to a common Pauli/poly-child representation, and count
equality alone is not sufficient. Some Paper-I sidecars expose only Pauli-string
labels, not the parent/coefficient data needed to prove full runtime-object
identity. For example, a diagnostic may need to report parent-pool terms,
expanded child atoms, active child atoms, available append atoms, raw sidecar
labels, unique Pauli-string set digests, and Pauli-string multiset digests as
separate labeled quantities before interpreting any mismatch.

### Normalized Comparator Pool Profiles

Pool-normalized comparisons use one shared ordered set of unique,
unit-coefficient Pauli generators. The selected static seed support is never
rewritten; the profile replaces only the legal pool for future support growth.

| Shared profile | AP runtime setting | AVQDS/TETRIS setting | Meaning |
|---|---|---|---|
| hamiltonian_drive_pauli | --normalized-candidate-pool-profile hamiltonian_drive_pauli | avqds_tetris_pool_source=hamiltonian_pauli | Deduplicated union of nonidentity static-Hamiltonian and drive Pauli strings. |
| full_meta_pauli_children | --normalized-candidate-pool-profile full_meta_pauli_children | avqds_tetris_pool_source=runtime_candidate_pool | Deduplicated Pauli/poly children of the complete replay/full-meta candidate pool. |

Both routes consume paper_ii_normalized_pauli_pool_v1 and must report
ordered_unique_pauli_sha256 and ordered_atom_contract_sha256. Count equality is
insufficient: rows belong to the same normalized-pool lane only when the
profile, atom count, and ordered atom digest agree. A comparison must not place
AP on the Hamiltonian/drive profile while placing a competitor on the full-meta
profile and call that pool-normalized.

For the Paper-I POWELL A1 weak-weak SNAKE \(k=10\) seed used by the current
\(A=0.8\) diagnostic, the locked profiles contain 32 and 948 unique Pauli
atoms, respectively. These are diagnostic comparison contracts, not a change
to the canonical full-meta APM setting.

## Variational Parameterization

| Math coordinate choice | Code mode | Output label | Append/prune granularity |
|---|---|---|---|
| One coordinate per Pauli / polynomial child | `per_pauli_term` | per Pauli / polynomial term | Support atoms are Pauli children. This is the AP default. |
| One coordinate per logical / macro generator | `logical_shared` | per logical / macro generator | Support atoms are macro generators. Requires compatible logical theta from the seed. |

The AP state records `parameterization_mode`, `parameterization_label`,
`active_parameter_count`, `runtime_parameter_count`,
`runtime_pauli_parameter_count`, and `logical_parameter_count`.

Batching Pauli/poly children can recover a macro-like support update when a
batch admits several children from the same parent generator, but this is not
identical to `logical_shared` macro append. Under `per_pauli_term`, admitted
children receive independent coordinates, making the ANZATS more flexible and
potentially more expensive than one shared macro coordinate.

## McLachlan Inverse Policy

Code owner: `pipelines/time_dynamics/ap_mclachlan/inverse.py`.

| Field | Default | Meaning |
|---|---:|---|
| `policy_id` | `supported_eigh_ridge_v1` | Shared AP supported-eigenspace inverse convention. |
| `ridge_lambda` | `1e-7` | Adds `ridge_lambda I` before eigendecomposition. |
| `pinv_rcond` | `1e-10` | Retains eigenmodes with absolute eigenvalue above `pinv_rcond * max_abs_eigenvalue`. |
| `solve_damping` | `0.0` | Uses retained-mode weights `sign(s)/(abs(s)+solve_damping)`. |
| `epsilon` | `1e-14` | Floor used in condition-number telemetry. |

Output telemetry includes `inverse_rank_retained`,
`inverse_condition_number`, eigenvalues, retained mask, `solve_damping`,
`theta_dot`, and `gamma`.

## McLachlan Solve Repair Lane

Code owner: `SolveRepairConfig` in
`pipelines/time_dynamics/ap_mclachlan/fixed_step.py`; append-runner surface in
`pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py`.

Current status: the CLI flag is explicit, but canonical append/prune/exchange
support-patch runs pass `--solve-repair` unless the user explicitly requests a
no-repair ablation. Fixed AP exposes `--solve-damping` as an inverse-policy
field, but the automated repair candidate set is currently wired through the
append trajectory route.

The repair lane is numerical only. It does not append, prune, consult exact
references, skip the time point, or alter the Hamiltonian. When enabled, the
runner first solves with the requested inverse policy. Repair entry comes from
hard execution invalidity, numerical miss `rho_num`, state-space step motion, or
temporal state-space kink. The condition number is a soft response-risk term in
candidate scoring, not a global repair trigger, unless strict finite-shot
validation is explicitly enabled. Once repair is opened, the code evaluates a
finite candidate set containing stabilizing, relaxing, and balanced rungs, then
selects the least-intervention state-space score. The response scheduler follows
Paper II: the normalized severity `m_k` is the maximum of the numerical-miss,
state-motion, and temporal-kink ratios; the active lane set `M_k` decides which
response is used. If state motion or temporal kink is active, local subdivision
is requested. If `rho_num` is active, the same-checkpoint inverse-policy
candidate breadth is widened in both relaxing and stabilizing directions.
`rho_num` alone does not request local subdivision. If no finite candidate satisfies the
acceptability predicate, the
trajectory continues with the least-bad finite rung and marks that checkpoint
unsupported. Runs abort only for execution invalidity such as NaN/Inf values,
incompatible dimensions, missing required inputs, or file/write failure.

| Field / flag | Default | Current behavior |
|---|---:|---|
| `enabled` / `--solve-repair` | CLI default false; canonical support-patch profile true | Enables the Paper-II solve-repair candidate set. Append/prune/exchange runs use it unless explicitly labeled as no-repair ablations. |
| `condition_number_max` / `--solve-repair-condition-number-max` | `1e6` | Soft kappa warning scale used in the least-intervention score; not a repair trigger. |
| `condition_number_fail` / `--solve-repair-condition-number-fail` | `None` | Hard kappa rejection threshold only when strict finite-shot validation is enabled. |
| `strict_finite_shot_validation` / `--solve-repair-strict-finite-shot-validation` | false | Enables hard kappa rejection for strict finite-shot validation runs. |
| `theta_dot_l2_max` / `--solve-repair-theta-dot-l2-max` | `None` | Archaic diagnostic-only coordinate cap on `||dot theta||_2`; not an active repair guard. |
| `rho_num_max` / `--solve-repair-rho-num-max` | `1e-2` | Opens repair search when the implemented stabilized solve fails to realize otherwise available tangent-plane expressivity. It does not choose the repair direction. |
| `state_motion_l2_step_max` / `--solve-repair-state-motion-l2-step-max` | `5e-2` | State-space interval motion guard on the larger of `dt ||T dot theta||` and the realized trial-state ray distance; local subdivision is preferred for this defect. |
| `state_space_kink_eta_max` / `--solve-repair-kink-eta-max` | `1e-2` | Temporal state-space kink threshold using the previous accepted same-dimension velocity. |
| `local_subdivision_enabled` / `--solve-repair-local-subdivision`, `--no-solve-repair-local-subdivision` | true | Allows local interval replay when subdivision can cure state-motion or kink-like defects. |
| `max_local_subdivisions` / `--solve-repair-max-local-subdivisions` | `4` | Maximum subdivision depth. The starting depth uses Paper-II `q_k` only when the state-motion or temporal-kink lane is active, then is capped here. |
| `local_subdivision_factor` / `--solve-repair-local-subdivision-factor` | `2` | Branching factor per subdivision depth. |
| `min_local_dt` / `--solve-repair-min-local-dt` | `1e-6` | Stops subdivision below this local step size. |
| `release_patience_min` / `--solve-repair-release-patience-min` | `1` | Minimum healthy return-to-base passes before releasing a held repair. |
| `release_patience_max` / `--solve-repair-release-patience-max` | `5` | Maximum healthy return-to-base passes before releasing a held repair. |
| `release_kink_threshold_scale` / `--solve-repair-release-kink-threshold-scale` | `0.5` | Stricter return-to-base release threshold multiplier. |
| `release_kink_severity_scale` / `--solve-repair-release-kink-severity-scale` | `4` | Scales how original kink severity increases the release patience count. |
| `ridge_ladder` / `--solve-repair-ridge-ladder` | `1e-7,3e-8,1e-8,0,3e-7,1e-6,3e-6,1e-5` | Ridge values used to form same-checkpoint repair candidates in both stabilizing and relaxing directions. |
| `pinv_rcond_ladder` / `--solve-repair-pinv-rcond-ladder` | `1e-10,1e-11,1e-12,1e-9,1e-8,1e-7` | Retained-eigenspace thresholds used to form same-checkpoint repair candidates in both stabilizing and relaxing directions. |
| `solve_damping_ladder` / `--solve-repair-damping-ladder` | `0` | Extra retained-mode damping values used to form same-checkpoint repair candidates. |

Telemetry records the effective `pinv_rcond`, `ridge_lambda`, and
`solve_damping` at every point, plus `solve_repair_enabled`,
`solve_repair_applied`, `solve_repair_unsupported`, `solve_repair_reason`,
`solve_repair_response_lanes`, `solve_repair_response_severity`,
`solve_repair_response_breadth`, `solve_repair_inverse_policy_breadth`,
`solve_repair_local_subdivision_breadth`, and the per-candidate repair
attempts. Rows also expose `rho_real`, `rho_expr`, `rho_num`,
`state_velocity_l2`, `state_motion_l2_step`, solve-guard booleans, and local
subdivision metadata. Candidate scoring at a repaired checkpoint uses the
effective inverse policy selected for that checkpoint.

## Integration Settings

Code owner: `pipelines/time_dynamics/ap_mclachlan/integrators.py`.

| Integrator | RHS evaluations per step | Use |
|---|---:|---|
| `euler` | 1 | Default interactive and runner path. |
| `rk4` | 4 | Explicit higher-order option. |

The time grid must contain at least one finite time point and must be
monotonically nondecreasing. When solve repair local subdivision is active, an
accepted recorded interval can contain multiple local substeps; output records
`local_subdivision_applied`, subdivision depth, substep count, reason, and a
repair summary for the RHS solves used inside that interval.

## Drive Settings

The drive adapter separates the scalar profile from the operator:

| Object | Runtime field |
|---|---|
| `c(t)` | `drive_A`, `drive_omega`, `drive_tbar`, `drive_phi` |
| `D` | `drive_model.drive_poly` |
| `H(t)` | `TimeDependentHamiltonian.polynomial_at(t)` |
| zero-angle drive tangent | `drive_aligned_ansatz` augmentation |

Supported drive operator families are resolved through
`pipelines/time_dynamics/adapters/drive_terms.py`. Current adapter routes
include spin-boson imbalance, spinful lattice density, HH density legacy,
spinless lattice density, boson-chain number, harmonic-Kerr-chain displacement,
and molecular-vibronic H2 `dH/dR`.

For driven AP runners, `drive_aligned_ansatz=True` is default. This modifies the
ANZATS tangent space and does not modify `H(t)`.

## Append-Enabled AP: Legacy Controller

Code owner: `AppendControllerConfig` in
`pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py`.

This is a compatibility append runner path selected only when
`--append-ladder-mode legacy_singleton`.

| Field / flag | Default | Current behavior |
|---|---:|---|
| `max_append_candidates` / `--max-append-candidates` | `8` | Number of candidate terms scored in the legacy singleton append pass. |
| `max_prune_candidates` / `--max-prune-candidates` | `0` | Number of active terms considered in the legacy singleton prune pass. |
| `max_total_prunes` / `--max-total-prunes` | `0` | Maximum accepted singleton prune events over the trajectory. |
| `append_gain_threshold` / `--append-gain-threshold` | `1e-10` | Minimum append gain for accepting append. |
| `prune_loss_threshold` / `--prune-loss-threshold` | `0.0` | Maximum deletion loss for accepting prune. |
| `residual_ratio_threshold` / `--residual-ratio-threshold` | `1e-3` | Below this residual ratio, append is not considered. This is an append-search trigger, not an accuracy claim; the default intentionally avoids machine-zero append churn. |
| `min_logical_parameter_count` / `--min-logical-parameter-count` | `1` | Prevents pruning below this logical support size. |
| `allow_incomplete_candidate_pool` | true | Controlled by `--require-complete-candidate-pool`. |

Legacy append and prune rank by neutral McLachlan support-patch score. This path
does not implement cost weighting, prune history, prune shadow, or exchange.

## Append-Enabled AP: Canonical Append Ladder

Runner default: `--append-ladder-mode combinatorial`.

Current status: implemented for append support atoms, cost-weighted combinatorial
scoring, Schur-conditioned append batches, pre-commit augmented-solve
confirmation, and diagnostic runs. In the current Paper-II controller this
append ladder supplies the append branch of the unified stay / pure append /
pure prune / true exchange support-patch family. Append-only runs are now
diagnostic limiting cases, not the primary canonical route.

| Field / flag | Default | Current behavior |
|---|---:|---|
| `append_ladder_mode` / `--append-ladder-mode` | `combinatorial` at CLI | `combinatorial` enables the support-atom append ladder. `legacy_singleton` is compatibility-only. |
| `append_occurrence_policy` / `--append-occurrence-policy` | `layer_reuse` | `layer_reuse` permits the same base Pauli/poly atom at a later tail position and assigns each occurrence a distinct runtime identity. `unique_support` removes active base atoms and is compatibility-only. Neither policy permits two copies of one base atom in the same candidate batch. |
| `max_append_batch_size` / `--max-append-batch-size` | `10` | Largest append rung considered by the combinatorial ladder. This is an upper bound: the accepted batch may have size `1,2,\ldots,max_append_batch_size`; it is not forced to have the maximum size. |
| `append_rung_set_cap` / `--append-rung-set-cap` | `64` | Runtime enumeration budget for diagnostic/local execution. This limits how many candidate combinations are scored per rung after prefiltering; it is not a Paper-II mathematical admission rule. |
| `append_prefilter_size` / `--append-prefilter-size` | `12` | Singleton prefilter size before combinatorial rungs. |
| `append_prefilter_policy` / `--append-prefilter-policy` | `cost_weighted_singleton_rank_score_prefilter_v1` | Singleton prefilter uses the same cost-weighted rank score as final batch selection; older neutral-policy aliases normalize to this active policy. |
| `append_gain_threshold` | `1e-10` | Minimum append gain for the selected batch. |
| `append_batch_score_threshold` / `--append-batch-score-threshold` | `1e-10` | Minimum cost-weighted rank utility for accepting a batch. |
| `cost_model` | `paper_i_proxy_denominator_v1` | Active append cost model. Raw append gain is still computed geometrically; ranking utility divides that gain by the Paper-I proxy denominator. |
| `cost_normalization_mode` / `--append-cost-normalization-mode` | `family_robust_v1` | Normalizes candidate-family cost primitives by robust positive excess before applying lambdas. `raw_legacy_v1` is available for diagnostics. |
| `append_cost_alpha` / `--append-cost-alpha` | `1.0` | Exponent in `append_gain / denominator^alpha`. |
| `append_cost_lambda_2q`, `append_cost_lambda_d`, `append_cost_lambda_1q`, `append_cost_lambda_theta`, `append_cost_lambda_shot` | `0.05`, `0.05`, `0.025`, `0.0`, `0.02` | Paper-I proxy denominator weights for two-qubit, depth/span, one-qubit, new-coordinate, and measurement-cache pressure. The AP append path currently uses zero online measurement-cache pressure and records that source explicitly. |
| `append_cost_scale_floor` | `1e-12` | Robust-normalization scale floor. |
| `append_schur_guard_enabled` / `--append-schur-guard` | true | Requires the selected append batch to pass checkpoint-local Schur novelty checks. |
| `append_schur_min_rank_fraction` / `--append-schur-min-rank-fraction` | `1.0` | Required retained Schur rank fraction. The default requires full candidate-block rank. |
| `append_schur_max_condition_number` / `--append-schur-max-condition-number` | `1e12` | Maximum Schur novelty condition number. Set `0` to disable this condition cap while keeping rank checks. |
| `append_schur_novelty_ridge_lambda` / `--append-schur-novelty-ridge-lambda` | `0.0` | Ridge used in the Schur novelty check. Default zero prevents solve ridge from masking duplicate tangent directions. |
| Augmented-solve confirmation | always on | Before commit, candidate batches must produce finite `K_{J\cup R,k}^{\oplus} f_{J\cup R,k}` solve telemetry and a finite recomputed augmented residual under the same inverse convention used for propagation. |
| `failed_append_reuse_enabled` / `--failed-append-reuse` | false | Optional diagnostic reuse of a failed full append search. When enabled, a certificate can skip repeated full ladder rescoring until geometry drift reopens the search. This is not an append count cap. |
| `failed_append_reuse_reopen_mode` / `--failed-append-reuse-reopen-mode` | `direct_threshold` | Reopen route: direct `D_cert >= tau_reopen(m*)`, or model-change with direct fallback when no useful local utility scale exists. |
| `failed_append_reuse_tau_min`, `failed_append_reuse_tau_margin_scale`, `failed_append_reuse_tau_max` | `1e-4`, `1.0`, `1.0` | Direct reopen threshold schedule. Larger failed-search margin permits more certificate reuse before rescoring. |
| `failed_append_reuse_eta_reopen`, `failed_append_reuse_model_l_min` | `0.5`, `1e-12` | Model-change reopen controls. The model route is used only when the stored secant utility scale is finite and large enough. |
| `failed_append_reuse_sentinel_count` | `4` | Number of rejected candidates/batches retained as compact secant sentinels after an authorized full search. |
| `failed_append_reuse_secant_*` | see config | Candidate-level wait scheduling for sentinels. Secant due state is advisory only; it must not trigger candidate rescoring by itself. |
| `residual_ratio_threshold` / `--residual-ratio-threshold` | `1e-3` | Below this residual ratio, the ladder returns no edit. This is the canonical append-search trigger and is deliberately much looser than machine precision. |
| solve repair flags | see McLachlan Solve Repair Lane | Optional numerical repair for checkpoint solves and interval advances. |
| `cost_required_for_decisions` | false | No external cost sidecar is required for append decisions. The active Paper-I proxy denominator is computed locally from candidate Pauli structure. |
| `allow_incomplete_candidate_pool` | false in config, runner maps from pool flag | Controls whether incomplete pools can be used diagnostically. |

The mathematical ladder is the rung set `1..max_append_batch_size` with
checkpoint-local scoring and Schur conditioning. Runtime search-budget knobs
such as `append_rung_set_cap` and `append_prefilter_size` bound local
enumeration cost; they must be reported as implementation budgets, not as
physical or variational criteria.

Telemetry fields include rung diagnostics, selected rung size, candidate count,
scored count, selected appended labels, append gain, cost-weighted rank utility,
Paper-I proxy cost denominator components, the Schur guard reason,
augmented-solve confirmation, failed-search reuse route/skip/certificate
metadata, and cost-policy metadata. Legacy `inserted_*` and
`insertion_gain` aliases remain for older manifests and scripts.

Failed append-search reuse is downstream of the full ladder. A certificate is
created only after the configured append ladder completes and no append is
accepted. The certificate is invalidated by active-support identity changes,
candidate-pool identity changes, append-policy changes, or retained-rank
signature changes. If it remains valid, the controller computes the certificate
drift from naturalized active-support Gram drift, naturalized force drift,
cumulative McLachlan path length, and stored sentinel utility drift. If the
selected reopen route does not fire, the checkpoint records
`failed_append_reuse_skip` and does not rescore candidates. If it reopens, the
ordinary append ladder runs; the reuse policy does not admit candidates, tune
against exact references, or limit the number of future appends.

### Append Batch Conditioning Guard

The live append guard rejects a candidate batch whose appended tangent block is
rank-deficient after checkpoint-local Schur projection against the current
support. It also rejects batches whose retained Schur block exceeds the
configured condition cap. The selected batch must also pass the pre-commit
augmented-solve confirmation: the controller builds the augmented support
geometry at the current state, solves with the same retained-support/ridge/
damping convention used by propagation, and recomputes the augmented residual
before committing the support edit. These guards are evaluated at the current
state and do not use exact-reference tuning, observable replay, or nonlinear
future-trajectory certification. For the exchange-family route, the same Schur
novelty and augmented-solve convention is evaluated on the full patched support;
deletion-containing finalists also inherit prune persistence, cooldown,
ray-distance, differential-miss, and patch-smoothness gates.

### Append Parent-Scout Frontier

Code owners:
`pipelines/time_dynamics/ap_mclachlan/support_frontier.py` for parent-indexed
frontier filtering and
`pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py` for prepared-state
parent-tangent scoring.

The append parent scout is a search-budget layer, not a new committed support
granularity. In `per_pauli_term` mode the controller still commits only
Pauli/poly-child append atoms. The scout groups available child atoms by parent,
assigns each parent a non-authoritative score, keeps a bounded parent set, then
expands retained parents back into child atoms before the ordinary append ladder
and unified support-patch selector run.

The implemented cheap score modes are:

| Score mode | Status | Meaning |
|---|---|---|
| `off` | Live | Disables parent-scout filtering. |
| `parent_tangent_schur_gain` | Candidate diagnostic | Builds a temporary one-block parent tangent from the parent's available child atoms, computes its cross/support geometry against the active tangent support, and routes the resulting one-block geometry through the same `score_support_patch()` Schur/inverse convention used by child append scoring. The resulting parent score ranks parents only. |
| `parent_linear_residual_v1` | Candidate diagnostic | Uses the same temporary parent tangent geometry, but ranks by the parent residual left after the current McLachlan velocity. It still records the same support-patch/Schur telemetry for audit. |
| `cached_child_ucb` | Reserved | Currently fails open because no cache-backed measurement contract is promoted. |
| `full_child_block_diagnostic` | Diagnostic-only / non-cheap | Scores the whole child set under each parent. This can reduce classical combinatorics but it has already paid much of the child-level geometry cost, so it must not be described as measurement-saving. |

Cheap parent-tangent scoring reuses the existing retained-support/ridge/supported
inverse convention. The temporary parent block is not added to the APM support,
does not appear as a final patch atom, and is labeled only in telemetry. If the
parent tangent cannot be constructed, the tangent matrix is unavailable, the
active geometry fails parity reconstruction, the score is non-finite, the
residual is above a configured fail-open guard, or exchange is enabled with
`append_macro_scout_exchange_fail_open=True`, the scout preserves the original
child frontier.

The first planned diagnostic run should enable this as a frontier test only,
for example `--append-macro-scout --append-macro-scout-score-mode
parent_tangent_schur_gain --append-macro-scout-parent-cap <M>`, while preserving
the same child-level append gain, Schur novelty, augmented-solve confirmation,
and Paper-I proxy denominator used by the non-scout append ladder. Report
`append_macro_scout_measurement_saving_score_available`,
`append_macro_scout_diagnostic_full_child_set_scoring`, parent counts, and child
counts before/after filtering.

## Support-Patch Schema Fields

Code owner: `SupportPatchControllerConfig`.

These fields define the active full support-patch controller surface. The
current CLI exposes the append ladder, prune-scoring/optional-prune-commit, and
exchange-family search-budget fields. Branch caps are computational search
budgets; they are not scientific batch-size forcing.

| Field | Default | Status |
|---|---:|---|
| `controller_profile` | `support_patch_exchange_family_v1` | Recorded schema profile for the unified stay / append / prune / exchange family. |
| `parameterization_mode_default` | `per_pauli_term` | Records the intended default coordinate mode. |
| `exchange_enabled` / `--support-patch-exchange` | true | Enables true exchange pairing when append and prune scouts are both available. Pure append/prune/stay are still in the finalist set. |
| `branch_scoring_enabled` | true | Records that the controller forms a branch-level finalist set instead of sequential append-then-prune. |
| `support_patch_scoring_workers` / `--support-patch-scoring-workers` | `1` | Classical execution knob. `1` is the serial/default path; values greater than `1` use ordered thread scoring for independent candidate evaluations. This must not change selected patches or trajectory values for fixed settings. |
| `prune_enabled` | dataclass/CLI raw default false; canonical diagnostic profile passes `--support-patch-prune` | Active branch in canonical support-patch diagnostics. Pure append-only runs are limiting-case diagnostics. |
| `prune_commit_enabled` | dataclass/CLI raw default false; canonical prune diagnostics pass `--support-patch-prune-commit` with a finite commit budget | Active commit gate in canonical support-patch diagnostics. Deletions still require persistence, cooldown, ray-distance, differential-miss, and patch-smoothness checks. |
| `max_prune_batch_size` | `0` unless prune is enabled | Live batched prune ladder upper rung for canonical support-patch diagnostics. |
| `prune_rung_set_cap` | `0` | Live runtime enumeration budget for batched prune ladder. |
| `prune_prefilter_size` | `0` | Live singleton-score prefilter for larger prune rungs. |
| `prune_loss_threshold` | `1e-2` | Live grouped deletion-loss authority threshold for support-patch prune; legacy prune has its own threshold. |
| `prune_history_window` | `3` | Live prior-window length for history-aware prune scoring. |
| `prune_history_lambda` | `1.0` | Live deletion-loss history penalty in prune nomination. |
| `prune_condition_lambda_kappa_rel` | `0.0` | Live optional numerator weight for retained-spectrum conditioning relief. Zero preserves the old score. |
| `prune_condition_lambda_schur` | `0.0` | Live optional numerator weight for whole-batch Schur degeneracy. Zero preserves the old score. |
| `prune_condition_lambda_kappa_hist` | `0.0` | Live optional numerator weight for prior conditioning-toxicity history. Zero preserves the old score. |
| `prune_condition_lambda_kappa_dam` | `0.0` | Live optional denominator penalty for deletions that worsen retained-spectrum conditioning. Zero preserves the old score. |
| `prune_persistence_required` | `1` | Live repeated prune evidence gate. In `exact_batch` mode this is the consecutive exact-batch streak; in `atom_history` mode this is the per-atom sighting count. |
| `prune_persistence_mode` | `atom_history` | Live prune persistence gate. `atom_history` counts repeated prune evidence over individual deletion atoms; `exact_batch` preserves the old whole-batch key streak for compatibility diagnostics. |
| `prune_atom_history_fraction` | `1.0` | Live `atom_history` gate fraction: the selected prune batch can reach commit/safety only when at least this fraction of its atoms meet `prune_persistence_required`. |
| `prune_cooldown_steps` | `2` | Live base cooldown after failed prune materialization/safety and the minimum cooldown for smoothness-deferred prune retry. |
| `min_runtime_parameter_count` | `1` | Reserved safety floor. |
| `prune_projection_enabled` | true | Reserved for projection/refit safety. |
| `prune_projection_rounds` | `2` | Reserved projection/refit parameter. |
| `prune_projection_trust_radius` | `0.05` | Reserved projection/refit parameter. |
| `prune_projection_regularization` | `1e-8` | Reserved projection/refit parameter. |
| `prune_ray_distance_tol` | `0.05` | Live same-checkpoint ray-distance safety threshold for prune commits. |
| `prune_differential_miss_tol` | `0.01` | Live same-checkpoint differential expressivity-miss safety threshold for prune commits. |
| `prune_shadow_enabled` | true | Reserved for shadow no-harm checks. |
| `prune_shadow_horizon_steps` | `2` | Reserved shadow horizon. |
| `prune_shadow_score_tol` | `0.01` | Reserved shadow tolerance. |
| `prune_patch_smoothness_enabled` | true | Live same-checkpoint state-space velocity smoothness guard for deletion-containing patches. |
| `prune_patch_smoothness_eta_max` | `1e-3` | Live threshold for normalized pre/post patch velocity jump. A prune candidate with larger severity is deferred, not blacklisted. This matches the Run 112 diagnostic setting promoted as the canonical deletion-containing patch smoothness threshold. |
| `prune_patch_smoothness_cooldown_max_steps` | `8` | Live maximum severity-scaled cooldown for a smoothness-deferred prune batch. |
| `prune_patch_smoothness_severity_scale` | `1.0` | Live scale for mapping excess smoothness severity to retry cooldown. |
| `prune_history_transition` telemetry | support-change dependent | Atom-level prune history/cooldowns are support-invariant for surviving active atoms, while deletion-loss, conditioning, and smoothness-deferred records are support-conditional and clear after append/prune/exchange support mutation. |
| `max_exchange_append_branches` / `--max-exchange-append-branches` | `3` | Live exchange search budget: number of top append scouts retained for true-exchange pairing. |
| `max_exchange_prune_branches` / `--max-exchange-prune-branches` | `3` | Live exchange search budget: number of top prune scouts retained for true-exchange pairing. |
| `max_exchange_pair_count` / `--max-exchange-pair-count` | `0` | Optional live exchange pair enumeration budget. Zero means no extra cap beyond retained branch lists. |
| `exchange_append_score_min` / `--exchange-append-score-min` | `0.0` | Optional live minimum append-scout utility for true-exchange pairing. This is a pairing filter, not forced admission. |
| `exchange_prune_score_min` / `--exchange-prune-score-min` | `0.0` | Optional live minimum prune-scout utility for true-exchange pairing. This is a pairing filter, not forced admission. |
| `exchange_residual_dominance_tol` | `1e-8` | Compatibility dominance tolerance retained in schema. Not a substitute for full patched-support utility. |
| `exchange_cost_dominance_tol` | `1e-8` | Compatibility dominance tolerance retained in schema. Not a substitute for full patched-support utility. |
| `patch_utility_delta_weight` / `--patch-utility-delta-weight` | `1.0` | Live numerator weight on the full-patch normalized support-value change in Paper-II patch utility. |
| `patch_utility_refit_weight` / `--patch-utility-refit-weight` | `0.0` | Live refit-effort risk weight. Current diagnostic implementation records zero-transport refit effort. |
| `patch_utility_velocity_weight` / `--patch-utility-velocity-weight` | `1.0` | Live denominator weight for deletion-containing patch velocity-jump severity. Pure append omits this term. |
| `patch_utility_threshold` / `--patch-utility-threshold` | `0.0` | Live minimum utility for committing a non-stay finalist. Failed true exchange never commits its halves unless pure append/prune independently pass as finalists. |
| `cost_model` | `paper_i_proxy_denominator_v1` | Live for canonical append ranking. |
| `cost_required_for_decisions` | false | No external cost sidecar is required for append decisions. |
| `cost_normalization_mode` | `family_robust_v1` | Live append cost normalization mode. |
| `append_cost_alpha` | `1.0` | Live append cost exponent in `append_gain / denominator^alpha`. |
| `append_cost_lambda_2q` | `0.05` | Live Paper-I proxy denominator weight. |
| `append_cost_lambda_d` | `0.05` | Live Paper-I proxy denominator weight. |
| `append_cost_lambda_1q` | `0.025` | Live Paper-I proxy denominator weight. |
| `append_cost_lambda_theta` | `0.0` | Live Paper-I proxy denominator weight for new runtime coordinates. |
| `append_cost_lambda_shot` | `0.02` | Live Paper-I proxy denominator weight; current AP append telemetry sets online measurement-cache pressure to zero. |
| `append_cost_scale_floor` | `1e-12` | Live robust-normalization scale floor. |
| `append_macro_scout_enabled` / `--append-macro-scout` | false | Optional macro-parent scout prefilter. Final patch atoms remain Pauli/poly children. |
| `append_macro_scout_score_mode` / `--append-macro-scout-score-mode` | `parent_tangent_schur_gain` | Score mode used when the prefilter is enabled. Cheap modes are non-authoritative parent-tangent scouts; `full_child_block_diagnostic` is explicitly non-cheap. |
| `append_macro_scout_parent_cap` / `--append-macro-scout-parent-cap` | `0` | Parent count retained by macro-scout mode. Zero disables this prefilter. |
| `append_macro_scout_score_min` / `--append-macro-scout-score-min` | `0.0` | Optional parent-score floor before top-parent retention. Keep at zero until scout recall is characterized. |
| `append_macro_scout_fail_open` / `--append-macro-scout-fail-open` | true | Preserves the original child frontier when parent-scout scoring is unavailable or uncertified. |
| `append_macro_scout_expand_if_residual_high` / `--append-macro-scout-expand-if-residual-high` | `0.0` | Optional residual-ratio fail-open threshold. Zero disables this guard. |
| `append_macro_scout_exchange_fail_open` / `--append-macro-scout-exchange-fail-open` | true | Preserves the full append frontier for exchange-enabled runs unless a future certified exchange-aware parent bound is implemented. |
| `append_macro_scout_audit_parent_count`, `append_macro_scout_audit_parent_fraction` | `0`, `0.0` | Optional parent-audit telemetry budget for retained and non-retained parents. |
| `append_macro_scout_parent_cost_alpha` | `1.0` | Exponent for parent-score cost proxy weighting. Final child batches still recompute the real Paper-I proxy denominator. |
| `prune_cost_alpha` | `1.0` | Live prune saved-cost pressure exponent. |
| `exchange_cost_alpha` | `1.0` | Compatibility field retained for exchange resource weighting; active exchange utility uses append/prune cost telemetry plus `patch_utility_*` weights. |
| `eps_loss` | `1e-14` | Reserved numerical floor. |
| `append_schur_guard_enabled` | true | Live append-batch Schur novelty guard. |
| `append_schur_min_rank_fraction` | `1.0` | Live append-batch required Schur rank fraction. |
| `append_schur_max_condition_number` | `1e12` | Live append-batch Schur condition cap; `0` disables only this cap. |
| `append_schur_novelty_ridge_lambda` | `0.0` | Live append-batch Schur novelty ridge. |
| `failed_append_reuse_enabled` | false | Optional diagnostic failed-search certificate reuse; disabled by default. |
| `failed_append_reuse_reopen_mode` | `direct_threshold` | Direct or model-change-with-direct-fallback certificate reopen route. |
| `failed_append_reuse_tau_min` | `1e-4` | Direct reopen threshold floor. |
| `failed_append_reuse_tau_margin_scale` | `1.0` | Direct reopen margin scaling. |
| `failed_append_reuse_tau_max` | `1.0` | Direct reopen threshold cap. |
| `failed_append_reuse_eta_reopen` | `0.5` | Model-change safety factor. |
| `failed_append_reuse_model_l_min` | `1e-12` | Minimum finite utility-change scale for model-change route. |
| `failed_append_reuse_naturalization_floor` | `1e-14` | Floor for retained-mode naturalization. |
| `failed_append_reuse_sentinel_count` | `4` | Number of rejected candidate/batch sentinels stored after a failed full search. |
| `failed_append_reuse_secant_wait_min` | `0.0` | Minimum geometry-clock wait for sentinel retry telemetry. |
| `failed_append_reuse_secant_wait_max` | `1.0` | Maximum geometry-clock wait for sentinel retry telemetry. |
| `failed_append_reuse_secant_wait_margin_scale` | `1.0` | Margin scale for base sentinel wait. |
| `failed_append_reuse_secant_positive_safety` | `0.5` | Positive secant trend shortens advisory wait. |
| `failed_append_reuse_secant_negative_growth` | `2.0` | Negative secant trend lengthens advisory wait. |
| `protect_drive_aligned_atoms` | true | Reserved safety rule for drive-aligned support. |
| `uses_reference_for_decision` | false | Enforced guard. Must never be true for AP decisions. |
| `uses_future_exact_forecast_for_decision` | false | Enforced guard. Must never be true for AP decisions. |

## Support-Patch Score Telemetry

Code owner: `pipelines/time_dynamics/ap_mclachlan/support_patch.py`.

Every support patch is represented by removed runtime indices and appended
runtime coordinates. Empty removal plus empty append is no edit; append-only is
append; removal only is prune; both is exchange. The string `insert` is reserved
for legacy payloads or a future explicit internal-placement operation.

Important output fields:

| Field family | Meaning |
|---|---|
| `support_patch_kind` | `no_edit`, `append`, `delete`, or `exchange`; old `insert` payloads are legacy append records. |
| `support_patch_before_gain`, `support_patch_after_gain` | McLachlan gain before and after the proposed support patch. |
| `support_patch_signed_delta_gain` | After-minus-before gain. |
| `support_patch_normalized_score` | Score normalized by `norm_b_sq + epsilon`. |
| `support_patch_append_gain` | Gain credited to appended coordinates. |
| `support_patch_insertion_gain` | Legacy alias for `support_patch_append_gain`. |
| `support_patch_deletion_loss` | Loss credited to deleted coordinates. |
| `support_patch_cost_terms`, `support_patch_cost_weight` | Present in schema; neutral in current AP append ladder. |
| `support_patch_rank_score` | Score used for current ranking. |
| `support_patch_pinv_rcond`, `support_patch_ridge_lambda`, `support_patch_solve_damping` | Inverse policy used for that patch score. |
| `support_patch_schur_novelty` | Scalar Schur novelty summary for appended blocks: rank, full-rank flag, PSD flag, condition number, and eigenvalue extrema. The full matrix is not serialized. |

## Class Settings Lock

Path:
`chtc/generic_time_dynamics_table/input/class_settings/paper_ii_class_settings_lock_v1.json`.

This manifest stores class-level controller settings from the broader
Paper-II dynamics table workflow. It is not the same thing as the standalone
AP runner CLI, but it is the current class-level settings lock for paper-facing
time-dynamics runs.

Top-level lock fields:

| Field | Meaning |
|---|---|
| `schema` | Manifest schema. |
| `lock_status` | Whether the lock is active. |
| `lock_policy` | Promotion policy for this settings lock. |
| `require_canonical_controller_classes` | Requires use of the class-level controller classes. |
| `settings` | Per-class settings records. |

Each settings record includes `tuning_class`, `algorithm_id`, `settings_id`,
`settings_kind`, source paths, strict exact-free feedback status, and a
`settings_payload`.

Current `settings_payload` keys:

| Category | Keys |
|---|---|
| Window / persistence | `active_window_size`, `miss_persistence_spec` |
| Candidate enumeration | `allow_repeats`, `max_probe_positions`, `shortlist_fraction`, `shortlist_size`, `candidate_step_scales`, `include_tangent_secant_proposal` |
| Miss / gain thresholds | `miss_threshold`, `miss_abs_threshold`, `gain_ratio_threshold`, `append_margin_abs`, `high_miss_no_admit_policy` |
| Append no-harm guards | `append_no_harm_guard_enabled`, `append_no_harm_condition_abs_floor`, `append_no_harm_condition_ratio_cap`, `append_no_harm_displacement_ratio_cap`, `append_no_harm_kink_max_condition_ratio`, `append_no_harm_kink_max_displacement_ratio`, `append_no_harm_kink_min_step_gain_delta`, `append_no_harm_rho_only_condition_ratio_cap`, `append_no_harm_rho_only_displacement_ratio_cap`, `append_no_harm_rho_only_min_step_gain_delta`, `append_no_harm_rho_only_step_residual_ratio_cap` |
| Scoring weights | `energy_weight`, `density_slope_weight`, `primary_density_weight`, `site_weight`, `measurement_penalty_weight`, `compile_penalty_weight`, `directional_penalty_weight`, `blend_weight_mode`, `gain_scale_mode`, `step_scale_mode`, `horizon_mode`, `signed_energy_lead_limit` |
| Confirmation scoring | `confirm_score_mode`, `confirm_compress_fraction`, `confirm_compress_max_modes`, `confirm_compress_min_modes`, `oracle_selection_policy` |
| Integrator policy | `integrator_policy`, `integrator_columnarity_threshold`, `integrator_condition_max`, `integrator_curvature_threshold`, `integrator_euler_fs_error_threshold`, `integrator_euler_min_time_fraction`, `integrator_euler_observable_window`, `baseline_step_refine_rounds`, `trust_radius` |
| Inverse / regularization | `pinv_rcond`, `regularization_lambda`, `candidate_regularization_lambda` |
| Fixed-manifold / escape policy | `lock_fixed_manifold`, `below_floor_energy_safe_d_shape_escape`, `below_floor_energy_safe_turn_escape`, `postcross_compare_diag` |
| Prune policy | `prune_mode`, `prune_loss_threshold`, `prune_miss_threshold`, `prune_safe_miss_increase_tol`, `prune_differential_miss_tol`, `prune_persistence_required`, `prune_persistence_mode`, `prune_atom_history_fraction`, `prune_persistence_window`, `prune_theta_block_tol`, `prune_state_jump_l2_tol`, `prune_ray_distance_tol`, `prune_schur_ladder_local_radius`, `prune_condition_lambda_kappa_rel`, `prune_condition_lambda_schur`, `prune_condition_lambda_kappa_hist`, `prune_condition_lambda_kappa_dam`, `prune_appended_origin_bias_enabled`, `prune_appended_origin_grace_steps`, `prune_appended_origin_target_policy` |
| Prune projection / shadow | `prune_projection_mode`, `prune_projection_max_active_runtime`, `prune_projection_regularization`, `prune_projection_rounds`, `prune_projection_trust_radius`, `prune_shadow_enabled`, `prune_shadow_horizon_steps`, `prune_shadow_score_increase_tol` |
| Compatibility exact-free guards | `checkpoint_controller_mode`, `checkpoint_controller_exact_input_mode` |

Compatibility names such as `checkpoint_controller_*` are legacy field names in
the run infrastructure. They should not be used as the reader-facing route name.

## Qiskit Community Cost Reporting Contract

Qiskit-community comparator costs are row-owned evidence, not PDF-owned
annotations. The comparator runner must compile its Qiskit trajectory circuits,
emit accumulated compiled-resource fields in the row bundle, and carry a
matching compile audit into aggregation before any local review PDF is built.

Code owners:

- `pipelines/time_dynamics/benchmarks/qiskit_native.py`: Paper-II Qiskit
  comparator boundary for TrotterQRTE, PVQD, and VarQRTE rows.
- `pipelines/exact_bench/qiskit_community_dynamics_adapter.py`: pinned
  Qiskit-community adapter and compile settings.
- `pipelines/time_dynamics/benchmarks/common.py::_qiskit_community_resources`:
  accumulated Qiskit compiled-resource emitter.
- `pipelines/time_dynamics/tables/dynamics_benchmark_contract.py`: table-field
  mirror for compiled resource metrics.
- `pipelines/reporting/build_paper_ii_hh_local_dynamics_report.py`: LaTeX PDF
  consumer for local HH seed-transfer cost/error rows.

Required completed-row fields for Qiskit-community comparator rows:

| Field | Meaning |
|---|---|
| `resources.resource_policy` | Must be `qiskit_community_compiled_circuit_accumulated_v1`. |
| `resources.qiskit_circuit_record_count` | Number of reported Qiskit circuit records in the trajectory bundle. |
| `resources.compiled_backend_name` | Compile/backend label used for the Qiskit circuit records. |
| `resources.compiled_count_2q_total` | Accumulated two-qubit count over reported trajectory circuits. |
| `resources.compiled_depth_2q_total` | Current repo two-qubit-depth proxy for the Qiskit row. At present this mirrors accumulated two-qubit count until a separate backend-depth estimator is added; do not relabel it as physical hardware depth. |
| `resources.compiled_depth_total` | Accumulated total compiled depth over reported trajectory circuits. |
| `compile_audit.selected_backend` | Mirror of the selected backend and compiled-resource totals used for audit/report consistency. |

`build_paper_ii_hh_local_dynamics_report.py` reads these fields through row
`table_fields`, `metrics`, or `resources` and renders them automatically in the
`Dynamics Cost/Error Rows` section. Agents must not hand-fill Qiskit cost cells
inside the PDF or `.tex` report. If a completed Qiskit row lacks the compiled
resource fields or compile audit, repair/rerun the row or mark the report
blocked until the row bundle carries the evidence.

## Standalone APM Results Pdf Cost Contract

The weak-weak APM progress report is the **Results pdf**:

`output/pdf/ap_mclachlan_weak_weak_snake_progress_diagnostic.pdf`.

This diagnostic artifact is separate from the Qiskit-community comparator
tables above. Its Qiskit cost strip/table means the compiled resource cost of
the final active APM ansatz at the row's terminal trajectory time, usually
`t_final=3`. It is a final-support structural compile for the row being plotted.
It is not a full-horizon repeated cost, not a Qiskit dynamics method, not a
parity check, and not an AP controller input.

For every completed trajectory added to the Results pdf workflow, the
corresponding manifest or cost sidecar must expose at least:

| Field | Meaning |
|---|---|
| `trajectory_json` | Report-row trajectory JSON path. |
| `raw_trajectory_json` | Raw/full trajectory JSON path when the report row is slimmed. |
| `run_index` | Visible Results-pdf run number, newest row highest. |
| `N2q` | Final active ansatz compiled two-qubit count. |
| `D2q` | Final active ansatz compiled two-qubit-depth proxy. |
| `Dc` | Final active ansatz total compiled depth. |
| `backend`, `optimization_level`, `seed_transpiler` | Compile target and transpiler settings. |
| `support_digest` or equivalent | Digest of the final active support compiled for the row. |
| `qiskit_cost_scope` / note | Explicit statement that this is terminal final-support cost and exact/reference data was not used. |

Short smoke rows whose terminal time is not `t=3` may be included only when the
cost note says it is the terminal ansatz for that smoke trajectory. Agents must
not silently treat such smoke costs as `t=3` costs.

## Comparator Boundary

Paper-II comparator rows are managed by the broader benchmark route, not by the
standalone AP runners alone.

| Comparator family | Route note |
|---|---|
| AP-McLachlan | Fixed and append-enabled runners above. |
| AVQDS Method-1 singleton limit | dyn_avqds_tetris with max_layer_width=1; best-singleton limit of the common continuous-RHS kernel. |
| AVQDS(T) Method-3 TETRIS | dyn_avqds_tetris; qubit-disjoint TETRIS append layers using the continuous McLachlan right-hand side. |
| Qiskit TrotterQRTE | Primary Qiskit-community comparator row when routed through the benchmark surface. |
| Qiskit PVQD | Primary Qiskit-community comparator row when routed through the benchmark surface. |
| Qiskit VarQRTE | Primary Qiskit-community comparator row when routed through the benchmark surface. |
| repo-native product formula / PVQD / fixed McLachlan | Legacy/native comparator rows; do not relabel as Qiskit rows. |
| exact diagonalization / exact classical propagation | Post-run diagnostic reference, not a dynamics method branch. |

Comparator evidence must share the same seed hash, drive signature, time grid,
observable set, diagnostic reference policy, and compile target within a
benchmark point. A pool-normalized lane must additionally share the normalized
pool profile, atom count, and ordered atom digest.

## Decision-Data Rule

AP support-patch decisions, integrator choices, and parameter updates must not
use exact/reference trajectories. Exact/reference inputs may be attached only
after a method trajectory exists, for plots and error metrics.

Runtime outputs should expose:

| Field | Required value for strict AP decisions |
|---|---|
| `uses_reference_for_decision` | false |
| `uses_exact_reference_for_decision` | false |
| `uses_future_exact_forecast_for_decision` | false |
| `reference_energy_error_scope` | `post_run_reporting` |

## Update Checklist

Update this file when any of these change:

- a runner adds, removes, or renames a CLI flag;
- `AppendControllerConfig`, `SupportPatchControllerConfig`,
  `SolveRepairConfig`, or `McLachlanInversePolicy` fields change;
- the default AP parameterization mode changes;
- drive operator resolution adds a new promoted channel;
- candidate-pool completeness rules change;
- a reserved support-patch field becomes live runtime behavior;
- the Paper-II class settings lock is replaced.

Suggested verification after edits:

```bash
python3 -m pipelines.time_dynamics.runners.ap_fixed_from_adapt_artifact --help
python3 -m pipelines.time_dynamics.runners.ap_append_from_adapt_artifact --help
python3 -m pytest test/test_ap_mclachlan_state_hamiltonian.py test/test_ap_mclachlan_adaptive_trajectory.py test/test_scaffold_runtime_loader.py -q
```
