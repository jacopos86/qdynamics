# HH Adaptive Realtime Checkpoint Controller — Companion Implementation Checklist and Proto-Schema

**Date:** 2026-03-23  
**Status:** design-only / documentation companion / no runtime source edits in this change  
**Target path:** `artifacts/reports/hh_adaptive_realtime_checkpoint_controller_companion_20260323.md`  
**Anchor spec:** `artifacts/reports/hh_adaptive_realtime_checkpoint_controller_spec_20260323.md`

## 1. Summary

This companion translates the March 23 controller spec into a concrete future implementation plan: add a new post-replay adaptive realtime checkpoint controller as a targeted sibling to existing direct-propagation profiles, not as a refactor of ADAPT or `hubbard_pipeline.py`; build it around new controller/types/measurement modules, one additive `src/quantum/compiled_ansatz.py` tangent helper, and narrow workflow/CLI wiring; preserve the hard constraints that exact reuse is same-checkpoint-only, cross-checkpoint data is scheduling-only prior data, V1 centers on baseline-plus-incremental `stay` vs `append_candidate` probing, `MeasurementCacheAudit` remains accounting-only, and unlocked operator-level `phase3_v1` scaffolds such as `artifacts/json/hh_prune_nighthawk_readapt_5op.json` are accepted while locked fixed-scaffold artifacts such as `artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json` are rejected for adaptive ownership.

## 2. Current-state analysis

### 2.1 Live control/data flow today

The live HH staged path is still:

```text
warm start
  -> ADAPT continuation
  -> adapt handoff bundle
  -> replay
  -> direct propagation profiles
```

Concrete call chain from selected code:

1. `pipelines/hardcoded/hh_staged_workflow.py::run_stage_pipeline()`
   - builds `h_poly`, `hmat`, `ordered_labels_exyz`, `coeff_map_exyz`, `psi_hf`
   - runs warm start via `hc_pipeline._run_hardcoded_vqe(...)`
   - runs ADAPT via `adapt_mod._run_hardcoded_adapt_vqe(...)`
   - writes handoff via `_write_adapt_handoff(...)`
   - configures and runs replay
   - returns `StageExecutionResult`

2. `pipelines/hardcoded/hh_staged_workflow.py::run_noiseless_profiles(stage_result, cfg)`
   - calls `_run_noiseless_profile(...)`
   - that calls `hc_pipeline._simulate_trajectory(...)`

3. `pipelines/hardcoded/hubbard_pipeline.py::_simulate_trajectory(...)`
   - only supports `suzuki2`, `piecewise_exact`, `cfqm4`, `cfqm6`
   - has no realtime checkpoint lifecycle

There is no live projected-real-time / McLachlan checkpoint loop in current source.

### 2.2 Existing code that should be reused

#### ADAPT-side probe and insertion vocabulary

`pipelines/hardcoded/hh_continuation_stage_control.py` already gives:

- `allowed_positions(...)`
- `detect_trough(...)`
- `should_probe_positions(...)`
- `StageControllerConfig`
- `StageController`

Current use is ADAPT-only; `adapt_pipeline.py` calls these in the phase-1 candidate loop. This code is reusable for insertion-position vocabulary and policy naming, but not as the future controller lifecycle owner.

#### ADAPT-side scoring, ordering, and accounting helpers

`pipelines/hardcoded/hh_continuation_scoring.py` already gives:

- `MeasurementCacheAudit`
- `measurement_group_keys_for_term(...)`
- `shortlist_records(...)`
- `Phase2NoveltyOracle`
- `Phase2CurvatureOracle`
- `CompatibilityPenaltyOracle`
- `greedy_batch_select(...)`
- `build_candidate_features(...)`

Current call chain in `adapt_pipeline.py`:

- `MeasurementCacheAudit(...)` created once at `adapt_pipeline.py:2773`
- `estimate(...)` used during candidate scoring at `adapt_pipeline.py:3361-3362` and `3615-3616`
- `commit(...)` called after selection at `adapt_pipeline.py:5088-5107`

Hard constraint: the current tests in `test/test_hh_continuation_scoring.py` explicitly lock this object to accounting-only grouped reuse, not measured-value caching.

#### Dataclass and manifest style precedents

`pipelines/hardcoded/hh_continuation_types.py` is the strongest style precedent for future controller runtime data:

- frozen dataclasses
- JSON-friendly field naming
- `MeasurementPlan`
- `MeasurementCacheStats`
- `ScaffoldFingerprintLite`
- `ReplayPlan`
- generator metadata / split-event payload style

`pipelines/hardcoded/handoff_state_bundle.py` is the strongest style precedent for serialized additive output:

- additive nested blocks
- snake_case manifest keys
- only emit optional blocks when populated
- avoid breaking legacy top-level keys

#### Parameterization and runtime index semantics

`src/quantum/ansatz_parameterization.py` already defines the contract the controller must honor:

- `AnsatzParameterLayout`
- `GeneratorParameterBlock`
- `serialize_layout(...)`
- `deserialize_layout(...)`
- `runtime_insert_position(...)`
- `runtime_indices_for_logical_indices(...)`
- `expand_legacy_logical_theta(...)`
- `project_runtime_theta_block_mean(...)`

Hard constraint: the future controller must not invent a second logical/runtime indexing vocabulary.

#### Oracle boundary

`pipelines/exact_bench/noise_oracle_runtime.py` defines the active oracle boundary:

- `OracleConfig` is fixed-shots / fixed-repeats / mean-or-median aggregation
- `ExpectationOracle.evaluate(...)` loops `oracle_repeats` times and returns aggregated estimates plus `raw_values`

Hard constraint: adaptive tier logic belongs above this boundary. The controller should clone `OracleConfig`; it should not move tier policy into `ExpectationOracle`.

### 2.3 Hard constraints and blockers

1. **No checkpoint owner exists.**  
   There is no module that owns:
   - checkpoint identity
   - baseline geometry acquisition
   - exact same-checkpoint reuse
   - candidate-block probes
   - action selection
   - temporal ledgering

2. **Existing “measurement reuse” is not value reuse.**  
   `MeasurementCacheAudit` is planning/accounting only. Its tests assert that role.

3. **Scaffold acceptance is not currently enforced for adaptive realtime use.**  
   The preferred object is the unlocked operator-level 5-op `phase3_v1` scaffold (`artifacts/json/hh_prune_nighthawk_readapt_5op.json`).  
   The locked 7-term fixed scaffold (`artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json`) has:
   - `"pool_type": "fixed_scaffold_locked"`
   - `"structure_locked": true`
   - `"fixed_scaffold_kind": "hh_nighthawk_gate_pruned_7term_v1"`

   V1 adaptive realtime mode should reject this object.

4. **Parameterization-aware exact tangent extraction is missing from the current public quantum helpers.**  
   `src/quantum/compiled_ansatz.py` can prepare states, but the controller will also need tangent columns consistent with the exact ordered compiled ansatz and runtime layout.

5. **Current post-replay integration point is profile-level, not controller-level.**  
   `run_noiseless_profiles(...)` already operates on `StageExecutionResult` after replay. The future controller should integrate at this same level, not inside ADAPT and not inside direct propagation.

## 3. Design

### 3.1 Chosen approach: targeted addition, not refactor

This should be implemented as a targeted addition:

- new `pipelines/hardcoded/hh_realtime_checkpoint_types.py`
- new `pipelines/hardcoded/hh_realtime_measurement.py`
- new `pipelines/hardcoded/hh_realtime_checkpoint_controller.py`
- one additive helper in `src/quantum/compiled_ansatz.py`
- narrow wiring in `pipelines/hardcoded/hh_staged_workflow.py`
- minimal CLI surface in `pipelines/hardcoded/hh_staged_cli_args.py`

This is preferable to a broader refactor because the missing capability is a new lifecycle owner, not a failure of the existing ADAPT or direct-propagation layers. Refactoring `adapt_pipeline.py` or `hubbard_pipeline.py` would create parallel responsibilities and blur the source-of-truth boundaries that are currently clear.

---

### 3.2 Future module boundaries and ownership

#### A. `pipelines/hardcoded/hh_realtime_checkpoint_types.py` — new

**Kind:** module of frozen dataclasses and validation/hash helpers  
**Why:** matches `hh_continuation_types.py` style and keeps controller state contracts centralized

**Owns:**

- immutable checkpoint identity objects
- immutable probe/baseline/ledger summary objects
- controller config objects
- scaffold acceptance helpers
- stable fingerprint helpers for exact reuse keys

**Who creates/owns instances:**

- `RealtimeCheckpointController` creates all per-checkpoint objects
- `hh_staged_workflow.py` creates config objects from CLI/config state
- no global instances

---

#### B. `pipelines/hardcoded/hh_realtime_measurement.py` — new

**Kind:** module with a small stateful cache class plus pure helper functions  
**Why:** exact same-checkpoint value reuse and tier policy are a distinct subsystem; keeping them out of the controller class avoids turning the controller into a cache+planner+serializer monolith

**Owns:**

- per-checkpoint exact value cache
- tier policy normalization and `OracleConfig` cloning
- planning-only reuse summary assembly
- exact-cache lifecycle rules and summaries

**Who creates/owns instances:**

- `RealtimeCheckpointController` creates one planning audit for the full run
- `RealtimeCheckpointController` creates one exact cache per checkpoint
- cache is cleared at checkpoint close and never restored from disk in V1

---

#### C. `pipelines/hardcoded/hh_realtime_checkpoint_controller.py` — new

**Kind:** stateful controller class  
**Why:** the controller owns a sequential lifecycle, mutable run-scoped caches, and temporal ledger append-only state

**Owns:**

- controller config
- run-scoped planning audit
- run-scoped compiled polynomial / Pauli-action caches
- per-checkpoint exact cache lifecycle
- temporal ledger append-only list
- checkpoint history assembly
- `stay` vs `append_candidate` decisions
- controller output manifest assembly

**Who creates/owns instances:**

- `pipelines/hardcoded/hh_staged_workflow.py` creates one controller instance per enabled profile run
- no singleton/global ownership

---

#### D. `src/quantum/compiled_ansatz.py` — modified

**Kind:** additive exact-state helper on existing executor  
**Why:** exact tangent extraction belongs with the compiled ansatz executor that already owns ordered runtime-step semantics and `AnsatzParameterLayout` integration

**Additive interface change:**

- **Before:** `prepare_state(theta, psi_ref) -> np.ndarray`
- **After:** keep `prepare_state(...)` unchanged, add one additive tangent-aware method, e.g.:
  - `prepare_state_and_runtime_tangents(theta, psi_ref, *, runtime_indices=None) -> tuple[np.ndarray, list[np.ndarray]]`

**Call sites to update:**

- new controller module only
- new compiled-ansatz tests only

**Backward compatibility:**

- existing `prepare_state(...)` callers remain unchanged

---

#### E. `pipelines/hardcoded/hh_staged_workflow.py` — modified

**Kind:** workflow/orchestration module  
**Why:** this is the post-replay integration boundary already used by `run_noiseless_profiles(...)`

**Additions:**

- add a new sibling function, not a deep `run_stage_pipeline()` rewrite:
  - `run_adaptive_realtime_checkpoint_profile(stage_result, cfg) -> dict[str, Any]`
- extend `StageExecutionResult` (wherever it is defined; validate exact location) with runtime-only fields needed by the controller:
  - `h_poly`
  - `replay_terms`
  - `candidate_pool`
  - `candidate_generator_metadata`
  - `candidate_family_id`

**Why runtime-only fields are preferred here:**  
V1 should run the controller immediately after replay using in-memory objects instead of inventing a second JSON-to-`AnsatzTerm` loader.

---

#### F. `pipelines/hardcoded/hh_staged_cli_args.py` — modified

**Kind:** CLI argument builder  
**Why:** current staged CLI has replay/direct/noise controls but no checkpoint-controller args

**Additive surface:**

Keep V1 CLI deliberately small:

- `--checkpoint-controller-mode`  
  choices: `off`, `exact_validation_v1`, `fake_backend_v1`  
  default: `off`

- `--checkpoint-shortlist-size`  
  default: `4`

- `--checkpoint-max-probe-positions`  
  default: `6`

All other controller thresholds should live in code defaults in the config dataclass until V1 stabilizes. This avoids another large unstable CLI surface.

**Validation rule:**  
If `--checkpoint-controller-mode != off`, require `oracle_aggregate == "mean"` before controller construction.

---

### 3.3 Proto dataclass sketches

These are shape contracts, not implementation code.

#### Controller config objects

```python
@dataclass(frozen=True)
class MeasurementTierPolicyConfig:
    scout_shot_fraction: float = 0.25
    scout_shot_min: int = 256
    confirm_shot_fraction: float = 0.50
    confirm_shot_min: int = 1024
    commit_shot_fraction: float = 1.0
    scout_repeats: int = 1
    confirm_repeats_min: int = 1
    commit_repeats_min: int = 2
    require_mean_aggregate: bool = True

    rho_miss_confirm: float = 0.15
    rho_miss_calm: float = 0.05
    condition_warn: float = 1e6
    condition_calm: float = 1e4
    margin_ambiguous_ratio: float = 0.10
    margin_strong_ratio: float = 0.25
    displacement_confirm_ratio: float = 0.25
    displacement_calm_ratio: float = 0.10
```

```python
@dataclass(frozen=True)
class RealtimeCheckpointControllerConfig:
    mode: str  # off | exact_validation_v1 | fake_backend_v1
    shortlist_size: int = 4
    max_probe_positions: int = 6
    probe_horizon_steps: int = 1
    live_branches: int = 1

    tau_miss_close: float = 0.05
    tau_miss_open: float = 0.15
    delta_abs_min: float = 1e-8
    delta_rel_min: float = 0.10
    hysteresis_ratio: float = 0.05

    condition_critical: float = 1e8
    stabilization_gap_warn_ratio: float = 0.10
    damping_gap_warn_ratio: float = 0.10
    displacement_ref: float = 1.0

    preferred_scaffold_kind: str = "operator_level_phase3_v1"
    reject_structure_locked: bool = True
    exact_reuse_scope: str = "same_checkpoint_only"
    temporal_prior_scope: str = "scheduling_only"
    grouping_mode: str = "checkpoint_v1_grouped_geometry"
    allow_regrouping_within_checkpoint: bool = False

    tier_policy: MeasurementTierPolicyConfig = ...
```

**Why this split:** tier policy and action policy change at different rates; separating them avoids a flat config blob.

---

#### `CheckpointContext`

```python
@dataclass(frozen=True)
class CheckpointContext:
    checkpoint_index: int
    t_start: float
    t_end: float
    delta_t: float

    branch_id: str
    branch_prefix_hash: str

    scaffold_fingerprint: dict[str, Any]  # serialized ScaffoldFingerprintLite
    parameterization_hash: str
    logical_theta_hash: str
    runtime_theta_hash: str
    prepared_state_fingerprint: str
    oracle_context_hash: str

    grouping_mode: str
    structure_locked: bool
    preferred_scaffold_kind: str

    exact_reuse_fingerprint: str
```

**Notes:**

- `branch_id` is `"main"` in V1 but still stored so the type survives a later multi-branch extension.
- `prepared_state_fingerprint` means “deterministic executable-state identity”; in exact mode it may include normalized-state hashing, but in noisy mode it should be derived from scaffold + theta + time + branch identity, not measured amplitudes.
- `exact_reuse_fingerprint` is a stable hash of the fields that define exact reuse validity.
- Hash payloads should serialize floats as deterministic strings (for example `"{:.16e}"`) before hashing.

---

#### `GeometryValueKey`

```python
@dataclass(frozen=True)
class GeometryValueKey:
    checkpoint_fingerprint: str
    observable_family: str
    group_key: str

    candidate_label: str | None = None
    position_id: int | None = None
    runtime_block_indices: list[int] = field(default_factory=list)

    oracle_context_hash: str = ""
```

**Closed-set `observable_family` values for V1:**

- `baseline_metric`
- `baseline_force`
- `baseline_drift_norm`
- `candidate_cross_metric`
- `candidate_self_metric`
- `candidate_force`

**Hard rule:** `shots` and `oracle_repeats` are **not** part of the key. That is what makes scout→confirm→commit extension possible on the same exact identity.

**Important distinction:** `group_key` here is a controller measurement-request identity, not automatically the same string returned by `measurement_group_keys_for_term(...)`.

---

#### `BaselineGeometrySnapshot`

```python
@dataclass(frozen=True)
class BaselineGeometrySnapshot:
    checkpoint_fingerprint: str
    active_runtime_indices: list[int]

    bar_G: list[list[float]]
    bar_f: list[float]
    dot_theta: list[float]

    drift_norm_sq: float
    stay_objective: float

    epsilon_proj_sq: float
    epsilon_step_sq: float
    rho_miss: float
    stabilization_gap_sq: float
    damping_gap_sq: float

    policy_condition_number: float
    policy_matrix_rank: int
    policy_solve_mode: str
    policy_regularization: float

    measurement_keys: list[dict[str, Any]]
```

**Why dense matrices are acceptable in V1:**  
The intended operator-level HH scaffolds are small; serializing dense `bar_G` is simpler and less error-prone than introducing sparse or factorized persistence.

**Not serialized:** any cached `K^+`, factorization object, or solver handle. Those remain controller-local.

---

#### `CandidateBlockGeometryProbe`

```python
@dataclass(frozen=True)
class CandidateBlockGeometryProbe:
    checkpoint_fingerprint: str

    candidate_label: str
    candidate_pool_index: int
    candidate_family: str

    position_id: int
    logical_insert_position: int
    runtime_insert_start: int
    runtime_block_indices: list[int]

    B_block: list[list[float]]
    C_block: list[list[float]]
    q_block: list[float]

    S_block: list[list[float]]
    w_block: list[float]
    eta_star: list[float] | None

    delta_lambda: float
    predicted_objective_after_append: float

    admissible: bool
    rejection_reason: str | None

    compile_cost_proxy: dict[str, float]
    planning_measurement_stats: dict[str, float]
    selected_measurement_tier: str

    measurement_keys: list[dict[str, Any]]
```

**V1 semantics:**

- `runtime_block_indices` refers to the inserted candidate block’s runtime indices in the trial layout.
- `predicted_objective_after_append = max(stay_objective - delta_lambda, 0.0)`
- `selected_measurement_tier` is recorded even in exact-validation mode for telemetry consistency.

---

#### `TemporalLedgerEntry`

```python
@dataclass(frozen=True)
class TemporalLedgerEntry:
    checkpoint_index: int
    t_start: float
    t_end: float

    action_kind: str  # stay | append_candidate
    candidate_label: str | None
    candidate_pool_index: int | None
    position_id: int | None

    rho_miss: float
    epsilon_step_sq: float
    policy_condition_number: float

    best_append_delta_lambda: float
    shortlist_margin_ratio: float
    predicted_displacement_ratio: float

    action_changed: bool
    selected_measurement_tier: str
    hot_group_keys: list[str]
    notes: list[str]
```

**This is persisted across checkpoints.**  
Its role is scheduling/diagnostics only. It must never be used as exact current geometry.

---

#### Controller-local decision object

This does not need a public persisted type file, but the controller should still use an internal immutable shape:

```python
@dataclass(frozen=True)
class ActionDecision:
    action_kind: str  # stay | append_candidate
    candidate_label: str | None
    candidate_pool_index: int | None
    position_id: int | None

    baseline_objective: float
    selected_objective: float
    selected_delta_lambda: float

    selected_measurement_tier: str
    reason: str
```

This keeps serialization and ledger generation deterministic.

---

### 3.4 Exact state/data flow

#### A. Controller creation

**Trigger:** `--checkpoint-controller-mode != off`  
**Path:** CLI config → `StagedHHConfig` → `hh_staged_workflow.py::run_adaptive_realtime_checkpoint_profile(...)` → `RealtimeCheckpointController(...)`  
**Thread context:** same Python thread as staged workflow  
**Observation path:** controller returns a plain manifest dict that the outer result serializer attaches under `adaptive_realtime_checkpoint`

**Hard validation at creation time:**

1. scaffold is operator-level and not structure-locked
2. `oracle_aggregate == "mean"` if noisy mode is enabled
3. grouping mode is fixed for the run
4. `live_branches == 1` and `probe_horizon_steps == 1`

If any fail and controller mode is explicitly enabled, V1 should hard-fail, not silently downgrade.

---

#### B. Checkpoint start

**Trigger:** each interval on the same time grid already used by direct profiles (`t_final`, `num_times`)  
**Path:** stage result + replay-final scaffold/state → controller builds `CheckpointContext`  
**Thread context:** same thread, synchronous  
**Mutation points:**

- new per-checkpoint exact cache created empty
- no scaffold mutation yet
- temporal ledger unchanged until commit

**Out-of-order/duplication:** not applicable; checkpoints are processed serially

---

#### C. Baseline geometry acquisition

**Trigger:** new checkpoint context  
**Path in exact-validation mode:**

1. normalize scaffold parameterization using `deserialize_layout(...)`
2. normalize runtime theta:
   - if `logical_optimal_point` exists, expand via `expand_legacy_logical_theta(...)` only if needed
   - else if only runtime theta exists, derive logical theta with `project_runtime_theta_block_mean(...)`
3. instantiate `CompiledAnsatzExecutor` for the current logical operator list in runtime mode
4. call new additive helper from `src/quantum/compiled_ansatz.py`:
   - `prepare_state_and_runtime_tangents(...)`
5. compute `H|psi>` with controller-owned compiled caches using `compile_polynomial_action(...)` and `apply_compiled_polynomial(...)`
6. build `bar_G`, `bar_f`, `drift_norm_sq`
7. solve stabilized baseline system once
8. compute `epsilon_proj_sq`, `epsilon_step_sq`, `rho_miss`, conditioning stats
9. package `BaselineGeometrySnapshot`

**Path in fake-backend mode:**

- same controller contract, but grouped-observable requests are routed through `hh_realtime_measurement.py`
- cache lookup happens before measurement
- missing groups are measured using tier-cloned `OracleConfig`
- raw grouped results are stored in the exact cache
- derived baseline quantities are rebuilt from cache records

**Downstream observation:** `BaselineGeometrySnapshot` is stored in current checkpoint history entry

**Duplicate requests:** same `GeometryValueKey` returns exact-cache hit  
**Dropped measurements:** incomplete request is not committed to cache; baseline build fails fast  
**Out-of-order results:** V1 has no parallel requests, so no reordering logic is required

---

#### D. Stay gate

**Trigger:** completed baseline snapshot  
**Rule:** skip append probing and commit `stay` immediately when all are true:

- `rho_miss <= tau_miss_close`
- `policy_condition_number < tier_policy.condition_calm`
- predicted displacement ratio `< tier_policy.displacement_calm_ratio`
- prior ledger entry did not indicate action instability
- no explicit ambiguity flag from the prior checkpoint

This is the only allowed early-exit path in V1. It exists to prevent waste on calm checkpoints.

---

#### E. Candidate and position probing

**Trigger:** stay gate did not short-circuit  
**Path:**

1. build allowed logical positions with direct reuse of `allowed_positions(...)`
2. choose candidate set
   - first exact slice may probe all candidates in the pool because pool sizes are still modest
   - later slice can layer in `shortlist_records(...)`
3. for each `(candidate, position)`:
   - compute planning-only measurement keys via `measurement_group_keys_for_term(...)`
   - record planning-only reuse stats through `MeasurementCacheAudit`
   - build trial operator list with candidate inserted at logical position
   - rebuild trial layout via `build_parameter_layout(...)`  
     **Do not manually patch runtime starts.**
   - compute trial runtime insert start via `runtime_insert_position(...)`
   - build trial runtime theta by zero-inserting the candidate block
   - instantiate trial `CompiledAnsatzExecutor`
   - call `prepare_state_and_runtime_tangents(...)` for only the inserted runtime indices
   - compute `B_block`, `C_block`, `q_block`
   - residualize with the baseline solve
   - compute `S_block`, `w_block`, `eta_star`, `delta_lambda`
   - determine admissibility and reason
   - emit `CandidateBlockGeometryProbe`

**Hard rule:** baseline tangent data is built once per checkpoint and reused. Candidate probes only add `B/C/q`. Full augmented recomputation is allowed only in exact-mode validation tests.

**Performance target:**  
With baseline support size `m` and candidate block size `q`, per-candidate probe should scale like `O(mq + q^2)` after the baseline solve, not like a fresh `O((m+q)^3)` rebuild.

---

#### F. Tier policy and exact-cache lifecycle

**Planning layer:** may persist across checkpoints for accounting and hot-group summaries  
**Exact value layer:** one cache per checkpoint only

**Same-checkpoint exact cache rules:**

- exact hit requires identical checkpoint fingerprint and identical key
- changing checkpoint index, branch prefix, parameter hash, candidate label, position, grouping mode, or oracle context invalidates the key
- `shots` and `oracle_repeats` are excluded from the key so that confirm/commit can extend scout

**Within a checkpoint:**

- lookup before measure
- append batches on confirm/commit
- summarize hits/misses and groups used
- clear cache immediately after commit

**Across checkpoints:**

- cache is discarded
- only `TemporalLedgerEntry` and planning summaries survive

---

#### G. Action decision

For each checkpoint:

- `J_stay = stay_objective`
- `J_append = max(stay_objective - delta_lambda, 0.0)`

Choose `append_candidate` only if all are true:

1. baseline miss is open (`rho_miss >= tau_miss_open`) **or** conditioning is critical
2. candidate probe is admissible
3. `delta_lambda >= delta_abs_min`
4. `delta_lambda / max(stay_objective, 1e-12) >= delta_rel_min`
5. append beats stay by hysteresis:
   - `hysteresis = hysteresis_ratio * max(stay_objective, 1e-12)`

Tie-breaks inside the hysteresis band, in this exact order:

1. lower compile proxy burden
2. lower `groups_new`
3. lower `candidate_pool_index`
4. lower `position_id`

The primary objective remains projected local objective reduction, not ADAPT `simple_score` or `full_v2_score`.

---

#### H. Commit and scaffold mutation

**If action is `stay`:**

- scaffold terms unchanged
- layout unchanged
- theta vectors unchanged

**If action is `append_candidate`:**

- insert candidate at logical position
- rebuild logical term list
- rebuild layout with `build_parameter_layout(...)`
- insert zero logical theta at logical position
- insert zero runtime theta block at the computed runtime insert start
- update branch prefix hash
- clear exact cache
- append ledger entry

**Hard rule:**  
Do not mutate layout by manually shifting existing `runtime_start` values. Always rebuild the layout from the new logical term order.

---

### 3.5 Serialized artifact schema sketch

#### 3.5.1 Serialization recommendation

Use a new additive top-level block:

- key: `adaptive_realtime_checkpoint`
- omit the block entirely when controller mode is `off`
- keep JSON field names in lower_snake_case
- do **not** mirror CamelCase dataclass names as top-level JSON keys

Nested objects should be:

- `"context"` for `CheckpointContext`
- `"baseline"` for `BaselineGeometrySnapshot`
- `"candidate_probes"` for `CandidateBlockGeometryProbe[]`
- `"temporal_ledger"` for `TemporalLedgerEntry[]`

This matches the style of `handoff_state_bundle.py` without forcing post-replay controller data into the pre-replay handoff writer.

#### 3.5.2 Sketch

```json
{
  "adaptive_realtime_checkpoint": {
    "schema_version": "hh_adaptive_realtime_checkpoint_v1",
    "controller_mode": "exact_validation_v1",
    "preferred_scaffold_kind": "operator_level_phase3_v1",

    "exact_cache_policy": {
      "persisted": false,
      "exact_reuse_scope": "same_checkpoint_only",
      "cross_checkpoint_behavior": "temporal_priors_only",
      "grouping_mode_locked_within_checkpoint": true
    },

    "config": {
      "shortlist_size": 4,
      "max_probe_positions": 6,
      "probe_horizon_steps": 1,
      "live_branches": 1,
      "tau_miss_close": 0.05,
      "tau_miss_open": 0.15,
      "delta_abs_min": 1e-8,
      "delta_rel_min": 0.10,
      "hysteresis_ratio": 0.05
    },

    "scaffold_source": {
      "pool_type": "phase3_v1",
      "structure_locked": false,
      "source_artifact_json": "artifacts/json/hh_prune_nighthawk_readapt_5op.json",
      "parameterization": {
        "...": "serialize_layout(layout) payload"
      }
    },

    "checkpoint_history": [
      {
        "context": { "...": "CheckpointContext fields" },
        "baseline": { "...": "BaselineGeometrySnapshot fields" },
        "candidate_probes": [
          { "...": "CandidateBlockGeometryProbe fields" }
        ],
        "decision": {
          "action_kind": "stay",
          "candidate_label": null,
          "position_id": null,
          "baseline_objective": 0.0123,
          "selected_objective": 0.0123,
          "selected_delta_lambda": 0.0,
          "selected_measurement_tier": "scout",
          "reason": "calm_baseline_stay"
        },
        "exact_cache_summary": {
          "entries_created": 14,
          "entries_reused": 9,
          "groups_new": 5,
          "groups_reused": 9,
          "persisted": false
        }
      }
    ],

    "temporal_ledger": [
      { "...": "TemporalLedgerEntry fields" }
    ],

    "planning_reuse_summary": {
      "planning_measurement_cache": {
        "...": "MeasurementCacheAudit.summary() payload"
      },
      "hot_group_keys": ["xz", "yezeze"]
    },

    "validation": {
      "incremental_probe_matches_full_augmented": true,
      "locked_scaffold_rejected": true,
      "exact_reuse_scope_respected": true
    }
  }
}
```

#### 3.5.3 Persistence and compatibility

- This is additive-only.
- No migration of existing handoff or replay schemas is required.
- New code reading old artifacts: controller block absent → controller unavailable/disabled.
- Old code reading new artifacts: should ignore unknown top-level block if readers are permissive; if any reader is strict, gate rollout until that reader is validated.

---

### 3.6 Existing-helper reuse matrix

| Reuse mode | Helper / type | Current file | Future controller role |
|---|---|---|---|
| **Direct reuse** | `allowed_positions(...)` | `hh_continuation_stage_control.py` | insertion-position enumeration |
| **Direct reuse** | `measurement_group_keys_for_term(...)` | `hh_continuation_scoring.py` | planning-only cost/hot-group summaries |
| **Direct reuse** | `shortlist_records(...)` | `hh_continuation_scoring.py` | later cheap shortlist trimming |
| **Direct reuse** | `CompatibilityPenaltyOracle`, `greedy_batch_select(...)` | `hh_continuation_scoring.py` | later probe-order refinement, not first slice |
| **Direct reuse** | `ScaffoldFingerprintLite` | `hh_continuation_types.py` | checkpoint scaffold identity payload |
| **Direct reuse** | `deserialize_layout(...)`, `build_parameter_layout(...)`, `runtime_insert_position(...)`, `runtime_indices_for_logical_indices(...)`, `expand_legacy_logical_theta(...)`, `project_runtime_theta_block_mean(...)` | `ansatz_parameterization.py` | all layout and theta normalization |
| **Direct reuse** | `compile_polynomial_action(...)`, `apply_compiled_polynomial(...)` | `compiled_polynomial.py` | exact `H|psi>` and exact polynomial actions |
| **Direct reuse** | `OracleConfig`, `ExpectationOracle` | `noise_oracle_runtime.py` | tier-cloned oracle execution in noisy mode |
| **Conceptual reuse only** | `MeasurementCacheAudit` object shape | `hh_continuation_scoring.py` | planning summary pattern only |
| **Conceptual reuse only** | `Phase2NoveltyOracle`, `Phase2CurvatureOracle`, `_tangent_data(...)` | `hh_continuation_scoring.py` | geometric shorthand ideas, not exact controller math |
| **Conceptual reuse only** | `detect_trough(...)`, `should_probe_positions(...)`, `StageControllerConfig` | `hh_continuation_stage_control.py` | policy vocabulary only |
| **Conceptual reuse only** | `handoff_state_bundle.py` output style | `handoff_state_bundle.py` | additive manifest style only |
| **Avoid misusing** | `MeasurementCacheAudit` as exact value cache | `hh_continuation_scoring.py` | forbidden |
| **Avoid misusing** | `full_v2_score(...)` as propagation-time action objective | `hh_continuation_scoring.py` | forbidden |
| **Avoid misusing** | `StageController` as checkpoint lifecycle owner | `hh_continuation_stage_control.py` | forbidden |
| **Avoid misusing** | `hubbard_pipeline._simulate_trajectory(...)` as controller owner | `hubbard_pipeline.py` | forbidden |
| **Avoid misusing** | locked 7-term scaffold as adaptive baseline | `hh_prune_nighthawk_gate_pruned_7term.json` | forbidden |
| **Avoid misusing** | `write_handoff_state_bundle(...)` for post-replay controller output | `handoff_state_bundle.py` | avoid in V1 |
| **Avoid misusing** | `oracle_aggregate="median"` in controller mode | `noise_oracle_runtime.py` | forbidden |

---

### 3.7 Recommended tests by file/module

#### `test/test_hh_realtime_checkpoint_types.py` — new

Cover:

- `CheckpointContext` hash stability for identical normalized payloads
- same-checkpoint fingerprint changes when:
  - checkpoint index changes
  - branch prefix changes
  - runtime theta changes
  - grouping mode changes
- `GeometryValueKey` excludes shot tier from identity
- scaffold acceptance:
  - accept `hh_prune_nighthawk_readapt_5op.json`
  - reject `hh_prune_nighthawk_gate_pruned_7term.json` with explicit reason

#### `test/test_hh_realtime_measurement.py` — new

Cover:

- exact cache hit on identical same-checkpoint key
- exact cache miss across checkpoint boundary with same observable family
- cache clear on checkpoint close
- confirm/commit append to the same key instead of creating a new one
- controller mode rejects non-mean aggregation
- grouping mode cannot change within a checkpoint
- planning audit summary remains separate from exact-cache summary

#### `test/test_compiled_ansatz.py` — new

Cover the new additive helper in `src/quantum/compiled_ansatz.py`:

- returned final state matches `prepare_state(...)`
- tangent count matches requested runtime indices
- one-term analytic sanity case at zero angle
- zero-inserted identity block does not change existing tangent columns or final state

#### `test/test_hh_realtime_checkpoint_controller.py` — new

Cover:

- baseline incremental block probe matches full augmented recomputation in exact mode on a small HH fixture
- `stay` chosen on calm baseline
- `append_candidate` chosen when miss is open and gain clears thresholds
- hysteresis keeps `stay` when append improvement is too small
- tie-break order:
  1. compile proxy
  2. `groups_new`
  3. `candidate_pool_index`
  4. `position_id`
- single committed action per checkpoint
- temporal ledger never backfills exact geometry
- locked scaffold rejection propagates before first checkpoint

#### `test/test_hh_staged_cli_args.py` — new

Cover:

- `--checkpoint-controller-mode` defaults to `off`
- explicit mode parses correctly
- no behavior change when mode is off
- controller mode + `oracle_aggregate=median` is rejected

#### `test/test_hh_staged_workflow.py` — new or extended

Cover:

- `run_adaptive_realtime_checkpoint_profile(...)` executes after replay using `StageExecutionResult`
- mode `off` leaves current post-replay behavior unchanged
- enabled mode returns payload with `adaptive_realtime_checkpoint`
- direct propagation profile generation still works unchanged when controller mode is enabled

#### Existing tests that should remain unchanged

- `test/test_hh_continuation_scoring.py`  
  Keep as a regression guard that `MeasurementCacheAudit` stays accounting-only.

- `test/test_hh_continuation_stage_control.py`  
  Keep as a regression guard on `allowed_positions(...)` semantics if reused directly.

---

### 3.8 Do first / do later / do not do yet

#### Do first

1. Add the new report only.
2. Add controller types and scaffold validation.
3. Add exact same-checkpoint cache semantics.
4. Add tangent extraction to `CompiledAnsatzExecutor`.
5. Implement exact/noiseless baseline + incremental `B/C/q` probing.
6. Implement `stay` vs `append_candidate` decision and temporal ledger.
7. Wire the controller as a post-replay sibling profile behind `--checkpoint-controller-mode`.
8. Keep mode off by default.

#### Do later

1. Replace all-candidate probing with shortlist trimming using `shortlist_records(...)`.
2. Add compatibility-aware probe ordering using `CompatibilityPenaltyOracle` / `greedy_batch_select(...)`.
3. Add fake-backend noisy validation using tier-cloned `OracleConfig`.
4. Emit turning/smoothness telemetry; keep it secondary to the main objective.
5. Add more detailed grouped-observable planners for hardware/noisy geometry families.

#### Do not do yet

1. `beam > 1`
2. probe horizon `> 1`
3. cross-checkpoint exact cache persistence
4. hybrid classical/QPU metric splitting
5. classical shadows
6. stochastic metric surrogates
7. IBM Runtime deployment
8. adaptive ownership of locked 7-term scaffolds
9. moving tier policy into `ExpectationOracle`
10. renaming this feature back under legacy `realtime_vqs`

---

### 3.9 Small, sharp invariants

1. **Exact reuse scope invariant**  
   Exact value reuse is valid only inside one checkpoint context.

2. **Planning-vs-value invariant**  
   `MeasurementCacheAudit` may affect ordering/cost telemetry only; it never stores measured geometry.

3. **Baseline reuse invariant**  
   Baseline geometry is built once per checkpoint and reused for every candidate probe.

4. **Incremental probe invariant**  
   Candidate probes only add `B/C/q`; full augmented recomputation is test-only.

5. **Single-action invariant**  
   Exactly one committed action is recorded per checkpoint: `stay` or `append_candidate`.

6. **Scaffold invariant**  
   Adaptive controller mode accepts unlocked operator-level scaffolds and rejects structure-locked fixed scaffolds.

7. **Tier aggregation invariant**  
   Controller mode requires mean aggregation.

8. **Temporal prior invariant**  
   Cross-checkpoint ledger data may change ordering/tiering only; it never fills current geometry entries.

---

### 3.10 Acceptance criteria

1. On a small exact HH fixture, incremental `B/C/q` probing matches full augmented recomputation within numerical tolerance.
2. Same-checkpoint duplicate geometry requests hit the exact cache.
3. The same request on the next checkpoint misses the exact cache.
4. `hh_prune_nighthawk_readapt_5op.json` is accepted as a controller baseline.
5. `hh_prune_nighthawk_gate_pruned_7term.json` is rejected before the first checkpoint.
6. Controller mode off preserves current workflow behavior.
7. Controller mode on emits a new additive `adaptive_realtime_checkpoint` block without changing legacy blocks.
8. Calm checkpoints can terminate at `stay` without probing append candidates.
9. Ambiguous or high-miss checkpoints escalate to confirm tier before commit in noisy mode.
10. No V1 path stores exact cache contents across checkpoints or restarts.

## 4. File-by-file impact

> This section is the future coding plan. The current documentation-only change adds only the report file above.

| File | Change | Why | Validation gate | Dependencies |
|---|---|---|---|---|
| `artifacts/reports/hh_adaptive_realtime_checkpoint_controller_companion_20260323.md` | **New** report | Captures this implementation checklist and proto-schema | Human review against March 23 anchor spec | none |
| `pipelines/hardcoded/hh_realtime_checkpoint_types.py` | **New** dataclasses, scaffold validators, stable hash helpers | Centralize immutable controller contracts and exact-reuse identity rules | `test_hh_realtime_checkpoint_types.py` passes | none |
| `test/test_hh_realtime_checkpoint_types.py` | **New** | Locks exact-reuse identity and scaffold acceptance behavior | positive/negative scaffold fixtures pass | types module |
| `pipelines/hardcoded/hh_realtime_measurement.py` | **New** exact cache + tier config clone helpers + planning/exact summaries | Isolate measurement-tier and same-checkpoint reuse logic from controller loop | `test_hh_realtime_measurement.py` passes | types module |
| `test/test_hh_realtime_measurement.py` | **New** | Proves same-checkpoint-only cache lifecycle and mean-only enforcement | all cache/tier tests pass | measurement module |
| `src/quantum/compiled_ansatz.py` | **Modified** with additive tangent-aware helper on `CompiledAnsatzExecutor` | Controller needs exact tangent columns consistent with ordered runtime layout; duplicating this logic in the controller would be wrong | `test_compiled_ansatz.py` passes and existing callers still compile | none |
| `test/test_compiled_ansatz.py` | **New** | Locks tangent-helper correctness and backward compatibility with `prepare_state(...)` | tangent + identity-insertion tests pass | compiled ansatz change |
| `pipelines/hardcoded/hh_realtime_checkpoint_controller.py` | **New** controller class, baseline/probe math, decision rule, ledger assembly, manifest assembly | This is the missing lifecycle owner | `test_hh_realtime_checkpoint_controller.py` exact-mode suite passes | types, measurement, compiled ansatz |
| `test/test_hh_realtime_checkpoint_controller.py` | **New** | Proves V1 control loop behavior | incremental-vs-full, decision, ledger, rejection tests pass | controller module |
| `pipelines/hardcoded/hh_staged_cli_args.py` | **Modified** with minimal controller CLI helper | Add explicit off-by-default entrypoint for the controller | CLI parser tests pass | config dataclasses |
| `test/test_hh_staged_cli_args.py` | **New** | Locks CLI defaults and invalid combinations | parse + validation tests pass | CLI change |
| `pipelines/hardcoded/hh_staged_workflow.py` | **Modified** to add `run_adaptive_realtime_checkpoint_profile(...)` and runtime-only fields on `StageExecutionResult` (or its owner; validate location) | Post-replay integration belongs here as a sibling to `run_noiseless_profiles(...)` | workflow integration tests pass; mode off unchanged | controller module, CLI config |
| `test/test_hh_staged_workflow.py` | **New/extended** | Locks integration order and additive output behavior | enabled/disabled workflow tests pass | workflow change |

### Explicit non-changes for V1

These files should stay unchanged in the first implementation slice:

- `pipelines/hardcoded/hh_continuation_scoring.py`
- `pipelines/hardcoded/hh_continuation_stage_control.py`
- `pipelines/hardcoded/hh_continuation_types.py`
- `pipelines/hardcoded/handoff_state_bundle.py`
- `pipelines/exact_bench/noise_oracle_runtime.py`
- `pipelines/hardcoded/hubbard_pipeline.py`
- `src/quantum/ansatz_parameterization.py`
- `src/quantum/compiled_polynomial.py`

Reason: they already provide the needed contracts or are the wrong ownership layer.

## 5. Trade-offs and alternatives

### Chosen: new post-replay controller profile  
**Runner-up:** embed controller logic into `run_stage_pipeline()` or `hubbard_pipeline._simulate_trajectory()`  
**Why rejected:** `run_noiseless_profiles(...)` already establishes a post-replay sibling-profile pattern; putting controller logic into direct propagation would mix incompatible responsibilities.

### Chosen: extend `CompiledAnsatzExecutor` with tangent support  
**Runner-up:** duplicate runtime-step derivative logic inside the controller  
**Why rejected:** derivative correctness depends on compiled ansatz ordering and parameterization; the executor already owns that contract.

### Chosen: additive top-level `adaptive_realtime_checkpoint` block  
**Runner-up:** overload legacy `realtime_vqs` naming or write into handoff bundle  
**Why rejected:** legacy `realtime_vqs` is historical lineage, not current validated behavior, and handoff bundles are pre-replay artifacts.

### Chosen: in-memory exact cache only  
**Runner-up:** persisted cross-checkpoint measurement-result cache  
**Why rejected:** it violates the controller spec’s correctness boundary and creates the highest-risk failure mode.

### Chosen: minimal V1 CLI  
**Runner-up:** expose every threshold immediately  
**Why rejected:** the repo already has a wide staged CLI surface; controller thresholds should stabilize in code before becoming user-facing.

## 6. Risks and migration

### Breaking changes

None are required for the documentation-only change.  
For the future runtime implementation, all intended changes are additive and off-by-default.

### Rollback

- If controller mode is off, behavior is unchanged.
- If controller mode is on and downstream readers cannot ignore the new block, keep the writer feature-gated until those readers are validated.
- Exact cache is in-memory only, so rollback does not require persisted-data migration.

### Key risks

1. **Incorrect tangent extraction in the compiled ansatz path**  
   Validate with dedicated unit tests and exact-mode augmented-vs-incremental comparisons.

2. **Accidental use of planning keys as exact cache keys**  
   Prevent by giving `GeometryValueKey.group_key` its own planner-generated identity and by keeping planning summaries in a separate namespace.

3. **Fake-backend confirm/commit aggregation semantics**  
   Validate whether appended oracle batches are statistically independent enough to aggregate as intended; if not, keep the fake-backend path behind a second validation gate.

4. **Unknown final serializer owner**  
   The selected context does not show the outer file that serializes `StageExecutionResult`. Implementation must trace that owner before landing the final output hook.

5. **Candidate-pool source helper location is not shown here**  
   Prefer carrying `candidate_pool` in memory through `StageExecutionResult` instead of reconstructing from JSON in V1.

### Migration strategy

- No migration for existing artifacts.
- New output block is additive.
- New code reading old outputs must tolerate the block being absent.
- Old code reading new outputs should ignore the block or remain on the controller-off path until validated.

## 7. Implementation order

1. **Land this companion report only.**  
   No runtime behavior changes.

2. **Add `hh_realtime_checkpoint_types.py` and `test_hh_realtime_checkpoint_types.py`.**  
   This is the first compilable runtime slice. Lock scaffold acceptance and exact-reuse identities before any controller logic exists.

3. **Add `hh_realtime_measurement.py` and `test_hh_realtime_measurement.py`.**  
   Implement exact-cache lifecycle and tier-config cloning. Keep it independent of the controller first.

4. **Extend `src/quantum/compiled_ansatz.py` with the additive tangent helper and add `test_compiled_ansatz.py`.**  
   This step should land atomically with its tests.

5. **Add `hh_realtime_checkpoint_controller.py` with exact/noiseless baseline + incremental probing only, and add `test_hh_realtime_checkpoint_controller.py`.**  
   First slice may probe all candidates instead of shortlisting. The key validation is incremental-vs-full agreement.

6. **Wire minimal CLI in `hh_staged_cli_args.py` and add the post-replay profile function in `hh_staged_workflow.py`.**  
   This step should remain off by default and land atomically with workflow/CLI tests.

7. **Extend `StageExecutionResult` runtime-only payload to carry the in-memory objects the controller needs.**  
   If the type is defined outside `hh_staged_workflow.py`, update that owner in the same change. This step is atomic with step 6.

8. **Add final result serialization under `adaptive_realtime_checkpoint` after locating the actual writer.**  
   Keep the block omitted when mode is `off`.

9. **Add fake-backend noisy validation mode.**  
   Only start this after exact-mode math and integration are stable.

10. **Add shortlist trimming and compatibility-aware probe ordering.**  
    This is optimization, not the first correctness milestone.

11. **Optionally add richer temporal-prior heuristics and turning telemetry.**  
    These are later quality improvements, not V1 blockers.

If only one future runtime milestone is funded, it should be **steps 2–8 with exact-validation mode only**; that delivers the controller skeleton, exact same-checkpoint reuse boundary, incremental block probing, scaffold acceptance, and additive artifact output without taking on the harder noisy-path risks.
