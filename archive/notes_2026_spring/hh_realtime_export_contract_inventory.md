# HH realtime export contract inventory

## Purpose

This is a planning-only inventory of the remaining time-dynamics export/controller seam.

Goal:
- document the current contract precisely
- reduce future agent ambiguity
- avoid risky controller refactors until the contract is better isolated

Non-goal:
- no production refactor in this step

---

## Seam overview

Current path:

```text
pipelines.hardcoded.hh_realtime_from_adapt_artifact
  -> pipelines.time_dynamics.hh_realtime_from_adapt_artifact
  -> build_controller_bundle_from_args(...)
  -> RealtimeCheckpointController(...)
  -> RealtimeCheckpointController.run()
       -> _write_progress() -> hh_realtime_progress.build_progress_payload()
       -> _write_partial_payload() -> hh_realtime_progress.build_partial_payload()
       -> returns ControllerRunArtifacts(trajectory, ledger, summary, reference)
  -> build_output_payload(...)
  -> final output JSON
```

Three distinct contracts exist and must not be conflated:

```text
final artifact != partial payload != progress payload
```

Current ownership:
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py`
  - still owns the main export seam
  - owns `ControllerRunArtifacts`
  - owns `_controller_state_payload()`
  - owns final `summary` and final `reference` assembly in `run()`
- `pipelines/time_dynamics/hh_realtime_progress.py`
  - owns progress/partial payload construction + atomic JSON writing
- `pipelines/time_dynamics/hh_realtime_from_adapt_artifact.py`
  - owns wrapper/serializer envelope around controller outputs
- `pipelines/hardcoded/*.py`
  - are live alias surfaces, not inert historical stubs

---

## Producer surfaces

### 1) `ControllerRunArtifacts`

Current in-memory final controller contract:

```text
ControllerRunArtifacts = {
  trajectory: list[dict[str, Any]],
  ledger: list[dict[str, Any]],
  summary: dict[str, Any],
  reference: dict[str, Any],
}
```

Produced by:
- `RealtimeCheckpointController.run()`

Consumed by:
- `build_output_payload(...)` in `hh_realtime_from_adapt_artifact.py`
- successful staged-noise forwarding in `pipelines/hardcoded/hh_staged_noise_workflow.py`

Important property:
- these structures are forwarded largely unchanged, with JSON normalization only at the outer serializer boundary

### 2) `_controller_state_payload()`

Current partial-reference contract:

```text
controller_state = {
  "logical_block_count": int,
  "runtime_parameter_count": int,
  "labels": list[str],
}
```

Used only inside partial payloads under:

```text
partial["reference"]["controller_state"]
```

Tests pin that partial payloads do **not** expose the full final `reference` payload.

### 3) `run()` stage emission order

Current write order is part of the operational contract:

1. `run_start` progress
2. `run_start` partial
3. per checkpoint:
   - `checkpoint_start` progress
   - controller work
   - `checkpoint_done` progress
   - `checkpoint_done` partial
4. optional early stop:
   - `early_stop` progress
   - `early_stop` partial
5. final summary/reference assembly
6. `run_complete` progress
7. `run_complete` partial
8. return `ControllerRunArtifacts`

Why this matters:
- timeout/env-blocked recovery depends on the staged artifacts being present in this order
- early-stop partials may exist before the full final summary exists

### 4) `hh_realtime_progress.py`

#### Progress payload base shape

`build_progress_payload(...)` currently owns this base payload family:

- `status`
- `stage`
- `mode`
- `append_count`
- `prune_count`
- `trajectory_points`
- `ledger_entries`
- `logical_block_count`
- `runtime_parameter_count`
- `total_checkpoints`
- `wallclock_elapsed_s`

Controller stage-specific additions include:

- `checkpoint_start`
  - `checkpoint_index`
  - `time`
  - `physical_time`
  - `drive_term_count`

- `checkpoint_done`
  - `checkpoint_index`
  - `time`
  - `physical_time`
  - `action_kind`
  - `controller_lane`
  - `decision_backend`
  - `oracle_decision_used`
  - `shortlist_size`
  - `oracle_confirm_limit`
  - `oracle_budget_scale`
  - `degraded_reason`

- `early_stop`
  - `status`
  - `checkpoint_index`
  - `time`
  - `early_stop_reason`

- `run_complete`
  - `status`
  - `summary`

#### Partial payload shape

`build_partial_payload(...)` currently owns this payload family:

```text
partial = {
  "status",
  "stage",
  "mode",
  "trajectory",
  "ledger",
  "reference": {"controller_state": ...},
  "summary",
}
```

Important distinction:
- `partial.reference` is intentionally **narrower** than final `result.reference`
- only `controller_state` is guaranteed here

#### Partial fallback summary

When a full final summary does not yet exist, `build_partial_payload(...)` synthesizes a minimal summary with at least:

- `append_count`
- `prune_count`
- `stay_count`
- `executed_decision_backends`
- `final_logical_block_count`
- `final_runtime_parameter_count`

This fallback behavior is frozen by tests and downstream recovery.

#### Atomic write behavior

`write_json_atomic()` is part of the contract in practice:
- staged-noise recovery expects replace-style writes
- controller tests pin failure behavior around throttling / emit-time advancement

---

## Final `reference` contract

Current final `reference` assembled at the tail of `RealtimeCheckpointController.run()` contains:

- `kind`
- `initial_state`
- `times`
- `drive_profile`
- `reference_method`
- `reference_steps_multiplier`
- `projection_time_sampling`
- `geometry_sample_time_policy`

### Nested `drive_profile`

Current nested fields include:
- `A`
- `omega`
- `tbar`
- `phi`
- `pattern`
- `custom_weights`
- `include_identity`
- `time_sampling`
- `t0`

### Metadata flow

`reference_steps_multiplier` flow:

```text
CLI --exact-steps-multiplier
  -> ControllerDriveConfig.exact_steps_multiplier
  -> driven reference-state builder
  -> reference["reference_steps_multiplier"]
```

`reference_method` flow:

```text
drive_time_sampling
  -> src.quantum.drives_time_potential.reference_method_name(...)
  -> reference["reference_method"]
```

This metadata path is pinned by `test/test_exact_steps_multiplier.py`.

---

## Final `summary` contract

Because `build_output_payload(...)` copies `dict(result.summary)` and run-complete progress embeds that same dict, the current `summary` surface is effectively public.

The shape is wide; grouped categories are below.

### Identity / status
- `mode`
- `requested_decision_backend`
- `status`
- `decision_backend`
- `executed_decision_backends`
- `decision_noise_mode`
- `oracle_estimate_kind`

### Controller/config echo
- `oracle_selection_policy`
- `confirm_score_mode`
- `prune_mode`
- `candidate_step_scales`
- `exact_forecast_baseline_step_refine_rounds`
- `exact_forecast_baseline_proposal_mode`
- `exact_forecast_baseline_blend_weights`
- `exact_forecast_baseline_gain_scales`
- `exact_forecast_include_tangent_secant_proposal`
- `exact_forecast_tangent_secant_trust_radius`
- `exact_forecast_tangent_secant_signed_energy_lead_limit`
- `exact_forecast_tracking_horizon_steps`
- `exact_forecast_tracking_horizon_weights`
- `exact_forecast_primary_density_target_mode`
- `exact_forecast_tracking_fidelity_defect_weight`
- `exact_forecast_tracking_primary_density_error_weight`
- `exact_forecast_tracking_staggered_error_weight`
- `exact_forecast_tracking_doublon_error_weight`
- `exact_forecast_tracking_site_occupations_error_weight`
- `exact_forecast_tracking_energy_total_error_weight`
- `exact_forecast_density_slope_weight`
- `exact_forecast_energy_slope_weight`
- `exact_forecast_energy_curvature_weight`
- `exact_forecast_guardrail_mode`

### Drive/baseline behavior
- `drive_aligned_density_active`
- `drive_aligned_density_label`
- `baseline_step_scaling_active`
- `baseline_step_scale_values_used`
- `baseline_blending_active`
- `baseline_blend_weight_values_used`
- `baseline_gain_scaling_active`
- `baseline_gain_scale_values_used`
- `baseline_proposal_kinds_used`

### Counts / execution stats
- `decision_override_count`
- `exact_forecast_veto_count`
- `append_count`
- `prune_count`
- `stay_count`
- `exact_decision_checkpoints`
- `oracle_decision_checkpoints`
- `oracle_attempted_checkpoints`
- `degraded_checkpoints`
- `raw_group_cache_hits`
- `raw_group_cache_misses`
- `raw_group_cache_extensions`
- `final_logical_block_count`
- `final_runtime_parameter_count`

### Final observables / maxima
- `final_fidelity_exact`
- `final_abs_energy_total_error`
- `final_staggered`
- `final_staggered_exact`
- `final_abs_staggered_error`
- `max_abs_staggered_error`
- `final_doublon`
- `final_doublon_exact`
- `final_abs_doublon_error`
- `max_abs_doublon_error`
- `final_site_occupations`
- `final_site_occupations_exact`
- `final_site_occupations_abs_error_max`
- `max_abs_site_occupations_error`

### Oracle provenance
- `oracle_backend_info`
- `oracle_backend_snapshot`
- `oracle_execution_surface`
- `oracle_runtime_profile`
- `oracle_runtime_raw_profile`
- `oracle_runtime_session_policy`
- `oracle_raw_transport`
- `oracle_runtime_job_ids`
- `oracle_runtime_job_count`
- `oracle_compile_request`
- `oracle_compile_observation`

### Audits / termination
- `planning_audit`
- `temporal_measurement_ledger`
- `early_stop_reason`
- `early_stop_checkpoint_index`
- `early_stop_time`

Note:
- tests directly assert only parts of this surface
- but because the serializer forwards it unchanged, the full shape is externally visible today

---

## `trajectory` and `ledger` contract language

Current safest description:
- `trajectory` rows are exported verbatim from controller runtime assembly
- `ledger` rows are exported verbatim from serialized `CheckpointLedgerEntry`
- CLI output and final partial payload copy them through unchanged
- tests mostly freeze parity and pass-through, not a manually curated subset schema

### Minimum ledger fields read by env-blocked reconstruction today

At minimum, downstream recovery logic reads or derives from:
- `action_kind`
- `decision_backend`
- `oracle_attempted`
- `logical_block_count_after`
- `runtime_parameter_count_after`
- `fidelity_exact`
- `abs_energy_total_error`
- `degraded_reason`

---

## Wrapper/output envelope contract

`build_output_payload(...)` in `hh_realtime_from_adapt_artifact.py` produces a larger top-level JSON envelope containing:

- `run_tag`
- `artifact_json`
- `loader_mode`
- `loader_summary`
- `controller_config`
- `drive_config`
- `oracle_config`
- `oracle_capability`
- `logging`
- `summary`
- `trajectory`
- `ledger`
- `reference`

### Nested wrapper-only fields

`loader_summary` includes fields such as:
- `generator_family`
- `fallback_family`
- `resolved_family`
- `handoff_state_kind`
- `family_terms_count`
- `adapt_depth`

`logging` includes:
- `progress_json`
- `partial_payload_json`

Important distinction:
- `summary`, `trajectory`, `ledger`, and `reference` are controller artifacts passed through with JSON normalization only
- `drive_config` may reflect controller-resolved state
- `oracle_config` is normalized request/config payload, not a raw runtime object

---

## Consumer surfaces / test matrix

| Consumer / test | File | What it freezes | Strength |
|---|---|---|---|
| Controller progress/partial tests | `test/test_hh_realtime_checkpoint_controller.py` | run-complete progress summary parity, progress throttle/failure behavior, partial narrow `reference.controller_state`, early-stop fallback summary, drive reference metadata, oracle provenance summary fields | direct / strong |
| Artifact wrapper parity | `test/test_hh_realtime_from_adapt_artifact.py` | pass-through parity for `summary`, `trajectory`, `ledger`, `reference`; logging path serialization; drive-config / oracle-config envelope behavior | direct / strong |
| Exact-steps metadata | `test/test_exact_steps_multiplier.py` | `reference_method_name()` semantics, `reference_steps_multiplier` recording | direct / strong |
| Staged-noise workflow runtime path | `pipelines/hardcoded/hh_staged_noise_workflow.py` | success path forwards controller payload mostly unchanged; env-blocked path reconstructs from progress + partial files | operational / strong |
| Staged-noise workflow tests | `test/test_hh_staged_noise_workflow.py` | forwarding of controller kwargs (`drive_config`, `exact_steps_multiplier`, progress/partial paths, timeout); env-blocked reconstruction from partial/progress artifacts | direct / strong |
| Checkpoint probe | `pipelines/time_dynamics/hh_realtime_checkpoint_probe.py` | secondary consumer of bundle-builder/config surface and optional reference-output JSON | secondary |

Key distinction:
- some fields are **directly asserted** in tests
- others are **externally visible by pass-through** and therefore risky to change even if not asserted field-by-field

---

## Wrapper ambiguity (must be documented explicitly)

The hardcoded TD wrappers are live alias paths, using the pattern:

```py
_module = import_module("pipelines.time_dynamics....")
sys.modules[__name__] = _module
```

Implications:
- `pipelines.hardcoded.*` and `pipelines.time_dynamics.*` are both active contract paths
- imports often use the hardcoded path
- monkeypatches/tests often target the underlying `pipelines.time_dynamics.*` module
- future extraction must preserve both import strings unless migrated atomically

Do **not** describe the hardcoded TD path as legacy-only; that is inaccurate in the current repo state.

---

## Safe later extraction candidates

These are the safest future moves if/when TD refactoring resumes.

### 1) Schema-only extraction
Examples:
- doc-only schema module or `TypedDict` layer for:
  - final artifact envelope
  - progress payload
  - partial payload
  - `controller_state`

Why safe:
- behavior stays in place
- ownership/documentation improves without moving controller logic

### 2) Pure final-reference builder
Extract only final `reference` dict assembly, provided all inputs remain precomputed.

Why relatively safe:
- narrow
- side-effect free
- already conceptually separate from decision logic

### 3) Wrapper-only output-envelope helpers
Further split `build_output_payload()` into loader/logging/oracle serialization helpers.

Why relatively safe:
- outside controller mutation core
- mainly serializer organization

### 4) Stage-name constants / payload labels
Only if string-compatible aliases are preserved.

Why safe-ish:
- low behavior impact
- low value, but low blast radius

---

## Do-not-move-yet zone

These parts should remain frozen for now:

- final `summary` assembly in the tail of `RealtimeCheckpointController.run()`
- stage emission order inside `run()`
- `_controller_state_payload()` field names and nesting under `partial.reference.controller_state`
- `hh_realtime_progress.build_partial_payload()` fallback-summary behavior
- pass-through of `trajectory` / `ledger` row dicts
- hardcoded/time_dynamics alias topology
- `reference_method_name()` as the metadata source for `reference_method`

Why:
- directly pinned by tests
- relied on by env-blocked recovery
- moving them now would create high blast radius for relatively little immediate payoff

---

## Practical recommendation

If TD work resumes soon, safest order is:

1. documentation/schema extraction
2. pure final-reference builder extraction
3. wrapper-envelope helper cleanup
4. only then reconsider any broader controller export refactor

Until then, treat the current controller export seam as frozen-by-contract.
