# HH Adaptive Realtime Checkpoint Controller Specification

**Date:** 2026-03-23  
**Status:** design-only / not implemented in live source  
**Scope:** repo-aware controller spec for future HH adaptive realtime McLachlan dynamics  
**Primary artifact type:** implementation-oriented Markdown spec, not runtime code

## Related files

This spec is grounded in the following existing repo materials:

- `MATH/Math.md` §17 — projective McLachlan geometry, normalized miss, Schur-complement gain, branch objective
- `notes/IMPLEMENT_SOON.md` — measurement reuse as a subsystem, continuation/controller objects, cache/planner interfaces
- `artifacts/reports/investigation_hh_realtime_shortlist_measurement_reuse_20260323.md` — diagnosis of current gaps
- `artifacts/reports/hh_prune_nighthawk_pareto_menu_20260322.md` — operator-level 5-op Re-ADAPT versus locked 7-term scaffold distinction
- `artifacts/json/hh_prune_nighthawk_readapt_5op.json` — preferred operator-level adaptive scaffold object
- `artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json` — locked downstream fixed scaffold, not the adaptive baseline
- `pipelines/hardcoded/hh_continuation_stage_control.py` — ADAPT-side probe/trough primitives
- `pipelines/hardcoded/hh_continuation_scoring.py` — shortlist, batching, compatibility, grouped-cost accounting
- `pipelines/exact_bench/noise_oracle_runtime.py` — current oracle boundary with fixed shots/repeats/aggregation
- `pipelines/hardcoded/hh_staged_workflow.py` — current warm → ADAPT → replay → direct-propagation route
- `pipelines/hardcoded/hubbard_pipeline.py` — current direct propagation methods only (`suzuki2`, `piecewise_exact`, `cfqm4`, `cfqm6`)

---

## 1. Diagnosis and hard requirements

### 1.1 Structural diagnosis

The missing object is **not** a cache in isolation.  
The missing object is a **checkpoint controller** that owns, at each realtime step:

1. which action to evaluate and commit (`stay` vs `append_candidate`),
2. which geometry data can be reused exactly at the current checkpoint,
3. how much measurement effort to spend before committing.

Without that controller:

- shortlist/probe logic stays trapped in ADAPT selection,
- measurement “reuse” degenerates into accounting,
- defect telemetry has no live policy owner,
- shot adaptation has nowhere correct to live.

### 1.2 Exact reuse versus temporal priors

This distinction is mandatory.

**Exact reuse** is valid only when all of the following hold:

- same checkpoint,
- same branch prefix,
- same prepared state,
- same parameter vector,
- same grouped observable identity,
- same oracle/backend/mitigation context.

Across checkpoints, old geometry is **not** exact current reuse.

Across time, previous values of projective metric/force objects are allowed only as:

- scheduling priors,
- stale initial values,
- shortlist hints,
- refresh hints,
- ambiguity-resolution hints.

They are **not** exact current values of the geometry.

### 1.3 Highest-value first implementation

The highest-value V1 is:

> **same-checkpoint incremental geometry reuse during `stay` vs `append_candidate` probe evaluation**

That means:

- measure baseline geometry once per checkpoint,
- reuse it for all candidate probes at that checkpoint,
- probe each candidate by measuring only its marginal block data,
- avoid recomputing the full augmented metric for every candidate.

### 1.4 Non-negotiable scope constraints

This spec assumes all of the following:

- preserve **operator-level** ADAPT compatibility,
- prefer the **5-op Re-ADAPT** operator scaffold as the adaptive baseline,
- reject or disable adaptation on the locked **7-term fixed scaffold**,
- keep hybrid classical/QPU splitting deferred,
- keep classical shadows deferred,
- keep stochastic metric surrogates deferred,
- keep IBM Runtime / wider hardware complications deferred in V1.

---

## 2. Current repo architecture and insertion point

### 2.1 Current live route

Current staged HH execution is effectively:

`warm start -> ADAPT continuation -> handoff bundle -> replay -> direct propagation`

Concretely:

- `pipelines/hardcoded/hh_staged_workflow.py` orchestrates warm start, ADAPT, handoff, replay
- `pipelines/hardcoded/hubbard_pipeline.py::_simulate_trajectory(...)` then performs direct propagation
- direct propagation currently supports only:
  - `suzuki2`
  - `piecewise_exact`
  - `cfqm4`
  - `cfqm6`

There is no live projected-real-time / McLachlan checkpoint loop in current source.

### 2.2 What already exists and should be reused

#### ADAPT-side shortlist/probe primitives

`pipelines/hardcoded/hh_continuation_stage_control.py` already provides:

- `allowed_positions(...)`
- `detect_trough(...)`
- `should_probe_positions(...)`
- `StageController`

These belong to ADAPT selection today, but the underlying policy ideas should be lifted into future realtime control.

#### ADAPT-side scoring/accounting primitives

`pipelines/hardcoded/hh_continuation_scoring.py` already provides:

- `MeasurementCacheAudit`
- `measurement_group_keys_for_term(...)`
- `shortlist_records(...)`
- `Phase2NoveltyOracle`
- `Phase2CurvatureOracle`
- `CompatibilityPenaltyOracle`
- `greedy_batch_select(...)`
- `build_candidate_features(...)`

These are useful as:

- candidate generation and shortlist tools,
- marginal cost models,
- overlap-aware ordering heuristics,
- exact-statevector geometric proxy code.

They are **not** the checkpoint controller.

#### Existing persistence and identity surfaces

The current repo already has good persistence style via:

- `pipelines/hardcoded/handoff_state_bundle.py`
- `pipelines/hardcoded/hh_continuation_types.py`

Notably, `ScaffoldFingerprintLite` is the right style of scaffold identity surface to reuse rather than replace.

### 2.3 Where the future controller belongs

The future controller should be added as a **new sibling layer** in the staged HH workflow.

Conceptually it belongs:

- **after replay**,
- **before direct propagation**,
- operating on the replay-refined operator-level scaffold and current state.

It should **not** be embedded into:

- `adapt_pipeline.py` — wrong lifecycle, ADAPT selection only
- `hubbard_pipeline.py` — wrong responsibility, direct propagation only

The correct future home is a dedicated controller module integrated from `hh_staged_workflow.py`.

---

## 3. Mathematical anchor in projective McLachlan geometry

This controller is anchored in `MATH/Math.md` §17.

### 3.1 Projective objects

At normalized state `|psi>` define:

- horizontal projector:
  `Q_psi = I - |psi><psi|`
- raw tangent columns:
  `tau_j = ∂_{theta_j}|psi(theta; S)>`
- horizontal tangents:
  `bar_tau_j = Q_psi tau_j`
- tangent matrix:
  `bar_T = [bar_tau_1, ..., bar_tau_m]`
- projective drift:
  `bar_b = Q_psi (-i H psi) = -i (H - E_psi) psi`
- drift norm:
  `||bar_b||^2 = Var_psi(H)`

### 3.2 Baseline local solve

The fixed-structure McLachlan solve is:

`bar_G dot(theta)_0 = bar_f`

with:

- `bar_G_ij = Re <bar_tau_i, bar_tau_j>`
- `bar_f_i = Re <bar_tau_i, bar_b>`

Under regularization, define:

`K = bar_G + Lambda`

and solve:

`K dot(theta)_lambda = bar_f`

### 3.3 Baseline defect quantities

The controller must track, at minimum:

- intrinsic model inadequacy:
  `epsilon_proj_sq = ||bar_b||^2 - bar_f^T bar_G^+ bar_f`
- actual damped step mismatch:
  `epsilon_step_sq = ||bar_T dot(theta)_lambda - bar_b||^2`
- normalized miss:
  `rho_miss = epsilon_proj_sq / (||bar_b||^2 + eps)`

Interpretation:

- `epsilon_proj_sq` says how much the current manifold fundamentally misses,
- `epsilon_step_sq` says how much the actual stabilized step misses,
- `rho_miss` is the clean adaptation trigger scale.

### 3.4 Baseline plus candidate-block geometry

At checkpoint `k`, baseline geometry is:

`K_k dot(theta)_k = bar_f_k`

For a candidate appended generator/block `(a,p)`, do **not** rebuild the whole geometry.  
Instead build the block augmentation

```text
[ K_k      B_{a,p} ] [dot(theta)] = [bar_f_k]
[ B^T_{a,p} C_{a,p}+Lambda_c ] [eta] = [q_{a,p}]
```

where:

- `K_k` is the stabilized baseline geometry,
- `B_{a,p}` is the cross block between the current retained support and the candidate block,
- `C_{a,p}` is the candidate self block,
- `q_{a,p}` is the candidate force block,
- `eta` are the new candidate block rates.

The residualized Schur objects are:

- `S_{a,p} = C_{a,p} + Lambda_c - B_{a,p}^T K_k^+ B_{a,p}`
- `w_{a,p} = q_{a,p} - B_{a,p}^T K_k^+ bar_f_k`
- `Delta_{lambda,a,p} = w_{a,p}^T S_{a,p}^+ w_{a,p}`

This is the correct repo-math mapping for the incremental probe.

### 3.5 Scalar shorthand used in planning discussions

For a one-dimensional candidate block, it is often convenient to use the shorthand:

- `m_a` = cross-metric column,
- `s_a` = candidate self-metric,
- `v_a` = candidate force,

so the gain reads schematically as

`Delta_a ∝ (v_a - m_a^T K_k^+ bar_f_k)^2 / (s_a - m_a^T K_k^+ m_a)`

This scalar shorthand must always be understood as a special case of the block form above.

### 3.6 Mandatory implementation principle

At a given checkpoint:

- baseline geometry is built once,
- each probe reuses the same baseline,
- each candidate measures only incremental block data,
- full augmented metric recomputation per candidate is forbidden in V1 except for exact-validation tests.

---

## 4. Preferred adaptive scaffold object

The adaptive baseline object must be **operator-level**.

### 4.1 Preferred object

Use an operator-level scaffold compatible with:

- `artifacts/json/hh_prune_nighthawk_readapt_5op.json`

Required metadata includes:

- logical operator labels,
- generator ids,
- logical/runtime parameter counts,
- parameterization payload,
- generator metadata,
- structure-lock metadata.

### 4.2 Rejected object for adaptive mode

The controller must not treat the locked downstream scaffold as its adaptive baseline:

- `artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json`

That object is already runtime-term expanded / fixed-scaffold locked.

### 4.3 Policy rule

If imported scaffold metadata indicates locked fixed-scaffold mode, then adaptive checkpoint mode must either:

- reject the run, or
- disable `append_candidate` explicitly.

V1 should prefer **rejection** over silent downgrade.

---

## 5. Controller responsibilities

A future `RealtimeCheckpointController` must own all of the following:

1. checkpoint lifecycle,
2. exact same-checkpoint value reuse,
3. temporal scheduling priors,
4. baseline geometry acquisition,
5. candidate shortlist generation,
6. incremental probe ordering,
7. scout / confirm / commit escalation,
8. `stay` vs `append_candidate` action decision,
9. checkpoint ledger emission,
10. exact-cache invalidation at checkpoint close.

It must be synchronous and single-controller-owned in V1.

No parallel probe execution is required in V1.

---

## 6. Core future data objects

The following objects should be introduced later as dedicated types.

### 6.1 `CheckpointContext`

Immutable checkpoint identity object containing at least:

- checkpoint index,
- segment times,
- branch id,
- branch prefix hash,
- scaffold fingerprint,
- parameterization hash,
- logical theta hash,
- runtime theta hash,
- prepared-state fingerprint,
- oracle/backend context hash,
- grouping mode,
- structure-locked flag.

This object defines the identity boundary for exact reuse.

### 6.2 `GeometryValueKey`

Exact grouped-observable reuse key containing at least:

- checkpoint index,
- branch prefix hash,
- prepared-state fingerprint,
- parameter hash,
- observable family,
- group key,
- candidate label if applicable,
- position id if applicable,
- runtime block indices,
- oracle context hash.

**Important:** shot tier is **not** part of the identity key.  
That allows scout → confirm → commit extension on the same exact value record.

### 6.3 `BaselineGeometrySnapshot`

Baseline checkpoint geometry containing at least:

- `bar_G`
- `bar_f`
- `||bar_b||^2`
- `epsilon_proj_sq`
- `epsilon_step_sq`
- `rho_miss`
- `stabilization_gap_sq`
- `damping_gap_sq`
- condition number
- matrix rank
- solve mode
- regularization
- solved `dot(theta)` if available
- exact measurement keys consumed

### 6.4 `CandidateBlockGeometryProbe`

Incremental candidate probe containing at least:

- checkpoint context,
- candidate label,
- candidate pool index,
- candidate family,
- position id,
- runtime block indices,
- `B_block`
- `C_block`
- `q_block`
- `S_block`
- `w_block`
- `eta_star` if solved
- `delta_lambda`
- predicted objective after append
- admissibility flag and rejection reason
- exact measurement keys consumed

### 6.5 `TemporalLedgerEntry`

Cross-checkpoint scheduling summary containing at least:

- checkpoint index,
- time window,
- committed action kind,
- candidate label if any,
- position id if any,
- `rho_miss`, `epsilon_step_sq`, condition number,
- shortlist margin ratio,
- predicted displacement ratio,
- action-changed flag,
- selected measurement tier,
- hot group keys,
- notes.

This object is persisted across checkpoints.  
It is scheduling metadata only, not exact geometry reuse.

---

## 7. Measurement reuse architecture

Reuse must be split into three layers.

### 7.1 Layer A — planning/cost reuse

This layer uses existing tools like:

- `measurement_group_keys_for_term(...)`
- `MeasurementCacheAudit`
- compatibility overlap / batching heuristics

Purpose:

- estimate marginal new-group cost,
- order candidate probes,
- forecast overlap savings,
- support shortlist decisions.

This layer may persist across checkpoints and branches because it is **not exact value reuse**.

### 7.2 Layer B — exact same-checkpoint value reuse

This is the missing layer.

It stores raw grouped observable results at fixed state for the current checkpoint only.

Purpose:

- baseline geometry reuse across all probes at the checkpoint,
- scout → confirm → commit shot extension on the same grouped observable,
- reconstruction of derived quantities from raw groups,
- exact reuse across candidate probes that share the same baseline state.

This layer is in-memory only in V1.

### 7.3 Layer C — temporal ledger / stale priors

This layer stores only scheduling hints such as:

- which groups were hot last checkpoint,
- whether prior checkpoints were calm or unstable,
- which actions repeated,
- whether shortlist ambiguity was high,
- which candidates were recently cheap/expensive.

This layer must never be consumed as if it were exact current geometry.

### 7.4 Raw storage versus derived storage

The value cache should store **raw grouped observable data**, not only final derived matrix entries.

Reason:

- the same grouped data can feed multiple derived quantities,
- stabilization choices may change,
- post-processing may change,
- branch choice may change,
- derived geometry can be rebuilt from the same raw grouped record.

`MeasurementCacheAudit` must remain a marginal cost/accounting model, not the actual value store.

---

## 8. Exact reuse and invalidation rules

### 8.1 Exact reuse is valid only for identical checkpoint context

Exact same-checkpoint reuse is allowed only when all identity components match:

- checkpoint,
- branch prefix,
- prepared state,
- parameter vector,
- grouped observable identity,
- oracle/backend/mitigation context,
- grouping mode.

### 8.2 Reuse decision rules

Allowed exact reuse cases:

- extra shots on the same grouped observable at the same checkpoint,
- confirm extends scout for the same baseline observable,
- confirm extends scout for the same candidate-block observable,
- multiple probes share the same baseline geometry.

Invalid exact reuse cases:

- different checkpoint index,
- different branch prefix,
- parameter update after a solve,
- accepted append changes the manifold,
- different candidate label when probing candidate-specific blocks,
- different position id for the same candidate block,
- regrouping inside the checkpoint,
- backend / mitigation / symmetry mode change.

### 8.3 Temporal prior validity

Temporal priors remain valid for scheduling when exact reuse is invalidated, but only as hints.

They may influence:

- scout tier sizing,
- which candidates to shortlist first,
- which groups are likely expensive,
- whether confirm tier is likely needed.

They may not fill missing current geometry entries.

### 8.4 Exact cache lifecycle

V1 exact cache policy:

- created at checkpoint start,
- used only during that checkpoint’s probe/decision phase,
- cleared at checkpoint close,
- never restored from disk in V1.

Only temporal ledger and planning summaries are persisted.

---

## 9. Checkpoint control loop

The V1 controller loop is:

### Step 1 — checkpoint start

- build `CheckpointContext`
- normalize parameterization/runtime mapping
- bind exact cache and temporal ledger view

### Step 2 — baseline geometry

Acquire baseline grouped observables once and build `BaselineGeometrySnapshot`.

Mandatory emitted telemetry should reuse legacy field names where possible:

- `epsilon_proj_sq`
- `epsilon_step_sq`
- `stabilization_gap_sq`
- `damping_gap_sq`
- `policy_condition_number`
- `policy_matrix_rank`
- `policy_solve_mode`

### Step 3 — stay gate

If `rho_miss` is small and conditioning is healthy, default toward `stay` unless probe ambiguity remains high.

### Step 4 — shortlist generation

Generate operator-level candidates from the allowed adaptive pool.

Use existing ADAPT-side primitives for cheap ordering only:

- grouped cost estimates,
- shortlist trimming,
- compatibility batching,
- overlap heuristics,
- position selection vocabulary.

### Step 5 — incremental candidate probes

For each shortlisted `(candidate, position)`:

- hold baseline fixed,
- hold current prepared state fixed,
- hold current parameter vector fixed,
- measure only marginal `B/C/q` data,
- derive `S/w/Delta_lambda`,
- never recompute the full metric.

### Step 6 — scout / confirm / commit

Apply the tier policy in Section 11.

### Step 7 — action decision

Choose exactly one action in V1:

- `stay`, or
- `append_candidate`

### Step 8 — commit

Extend only the chosen action’s needed grouped observables to commit tier, record the ledger entry, and finalize the checkpoint.

### Step 9 — propagate / advance

Advance time and create the next checkpoint context.

---

## 10. `stay` versus `append_candidate` decision rule

### 10.1 Baseline objective

Let the stabilized baseline objective be

`L_star = ||bar_b||^2 - bar_f^T K^+ bar_f`

Then define:

- `J_stay = L_star`
- `J_append(a,p) = max(L_star - Delta_{lambda,a,p}, 0)`

### 10.2 Admission rule

Choose `append_candidate` only if all are true:

1. baseline miss is meaningfully open **or** conditioning/stabilization is critical,
2. candidate probe is admissible,
3. `Delta_{lambda,a,p}` exceeds both absolute and relative minimum gain thresholds,
4. append beats stay by hysteresis margin.

### 10.3 Proposed V1 defaults

Proposed defaults:

- `tau_miss_close = 0.05`
- `tau_miss_open = 0.15`
- `delta_abs_min = 1e-8`
- `delta_rel_min = 0.10`
- `hysteresis = 0.05 * max(L_star, 1e-12)`

These are proposed controller defaults, not mathematically sacred constants.

### 10.4 Tie-break rules inside hysteresis band

If actions are inside the hysteresis band, use secondary ordering only:

1. lower compile proxy burden,
2. lower marginal new-group count,
3. lower candidate pool index,
4. lower position id.

These are secondary.  
The primary control quantity remains projective local objective reduction.

---

## 11. Scout / confirm / commit measurement policy

This policy lives **above** `ExpectationOracle`, not inside it.

### 11.1 Tier defaults

Starting from base oracle settings:

- `scout_shots = max(256, round(base_shots * 0.25))`
- `confirm_shots = max(1024, round(base_shots * 0.50))`
- `commit_shots = max(base_shots, confirm_shots)`
- `scout_repeats = 1`
- `confirm_repeats = max(1, base_repeats)`
- `commit_repeats = max(2, base_repeats)`

### 11.2 Aggregation restriction

Adaptive checkpoint mode should require:

- `oracle_aggregate == "mean"`

Reason:

- mean aggregation composes naturally with exact same-key shot extension,
- median aggregation does not provide a clean extension rule for scout → confirm → commit accumulation.

### 11.3 Escalation from scout to confirm

Escalate when any of the following holds:

- `rho_miss >= 0.15`
- condition number is high (proposed warning threshold `1e6`)
- stabilization or damping gap enters warning band
- best-vs-second-best action margin is small (proposed `< 10%`)
- objective difference is within roughly `2 * combined_stderr`
- predicted displacement ratio is large (proposed `> 0.25`)
- action kind/label/position changes from prior checkpoint

### 11.4 Commit tier

Commit tier is always used for the chosen action.

Additionally extend baseline groups that dominate uncertainty for the chosen action.

### 11.5 Calm-segment de-escalation

Losers remain at scout tier when all are true:

- `rho_miss <= 0.05`
- condition number is healthy (proposed `< 1e4`)
- winner margin is strong (proposed `>= 25%`)
- predicted displacement ratio is small (proposed `< 0.10`)
- action is unchanged from prior checkpoint

### 11.6 Trigger sources

Tier policy should be driven by:

- defect:
  - `rho_miss`
  - `epsilon_step_sq`
- conditioning:
  - condition number
  - matrix rank
  - stabilization gap
  - damping gap
- ambiguity:
  - stay-vs-best-append margin
  - best-vs-second-best margin
  - uncertainty overlap
- action instability:
  - action kind change
  - candidate label change
  - position change
- predicted motion:
  - displacement surrogate from local tangent speed

---

## 12. Predicted displacement and smoothness / directional-change control

This is a key design target.

### 12.1 Predicted displacement surrogate

At checkpoint `k`, define the physical tangent speed surrogate:

`nu_k_sq = dot(theta)_k^T bar_G_k dot(theta)_k`

and the predicted displacement scale:

`d_k = Delta_t * sqrt(max(nu_k_sq, 0))`

For append probes use the augmented solved rate vector when available.

In V1, use displacement ratio:

`disp_ratio_k = d_k / d_ref`

with proposed default `d_ref = 1.0` as a trust-radius-style normalizer.

### 12.2 Smoothness target

Propagation-side smoothness control is **not already implemented** in the repo.  
This spec defines it as a future controller responsibility.

The mathematically preferred quantity is a whitened turning measure on retained-support velocities.

If the numerical support of `bar_G_k` is

`bar_G_k ≈ V_k Sigma_k^2 V_k^T`

define whitened velocity coordinates:

`z_k = Sigma_k V_k^T dot(theta)_k`

Then for consecutive checkpoints with compatible support, define turning cosine:

`cos(phi_k) = (z_{k-1}^T z_k) / (||z_{k-1}|| ||z_k||)`

This gives a projective-geometry-aware directional-change measure.

### 12.3 V1 use of smoothness information

V1 should not yet hardwire a full smoothness penalty into the primary objective.

V1 should instead use:

- action-change flags,
- displacement ratio,
- near-reversal / high-turning telemetry when available,
- hysteresis against small objective gains.

This is enough to support change-aware measurement escalation and avoid churn.

### 12.4 Future V1.5 / V2 direction

Later controller versions may include a soft propagation-side penalty such as:

- `lambda_turn * (1 - cos(phi_k))`
- support-change admission hysteresis
- gain-per-direction-change tradeoff

But that is intentionally deferred behind the checkpoint controller itself.

---

## 13. Reusing existing repo primitives correctly

### 13.1 Reuse directly

Future controller implementation should directly reuse or adapt:

- `allowed_positions(...)`
- `measurement_group_keys_for_term(...)`
- `shortlist_records(...)`
- `CompatibilityPenaltyOracle`
- `greedy_batch_select(...)`
- `ScaffoldFingerprintLite`
- `handoff_state_bundle` persistence style
- logical/runtime parameterization helpers

### 13.2 Reuse conceptually, not wholesale

Use as inspiration / extracted helpers:

- `Phase2NoveltyOracle`
- `Phase2CurvatureOracle`
- tangent overlap helper logic
- current candidate feature construction patterns

### 13.3 Do not misuse

Do **not** reuse these in the wrong role:

- `MeasurementCacheAudit` as the exact value cache
- `full_v2_score` as the primary propagation-time objective
- `StageController` lifecycle as the checkpoint lifecycle
- `hubbard_pipeline._simulate_trajectory()` as the controller owner

---

## 14. Suggested future file/module map

Future implementation should likely introduce:

- `pipelines/hardcoded/hh_realtime_checkpoint_controller.py`
  - controller loop and lifecycle owner
- `pipelines/hardcoded/hh_realtime_checkpoint_types.py`
  - immutable checkpoint, probe, ledger, key dataclasses
- `pipelines/hardcoded/hh_realtime_measurement.py`
  - exact value cache, tier planner, grouped observable planning helpers

And later extend:

- `pipelines/hardcoded/hh_staged_workflow.py`
  - new controller route after replay
- `pipelines/hardcoded/hh_staged_cli_args.py`
  - new checkpoint-controller mode and threshold knobs
- `pipelines/hardcoded/handoff_state_bundle.py`
  - additive controller metadata block
- `pipelines/exact_bench/noise_oracle_runtime.py`
  - additive helper paths for tier-specific config cloning only

It should not embed controller logic into:

- `adapt_pipeline.py`
- `hubbard_pipeline.py`

---

## 15. Additive artifact schema recommendation

Future controller outputs should live under a new additive block, for example:

`adaptive_realtime_checkpoint`

Recommended top-level fields:

- `schema_version`
- `controller_mode`
- `preferred_scaffold_kind`
- `checkpoint_history`
- `temporal_ledger`
- `planning_reuse_summary`
- `exact_cache_policy`

Recommended exact cache policy fields:

- `persisted: false`
- `exact_reuse_scope: "same_checkpoint_only"`

This avoids overloading legacy `realtime_vqs` naming, which refers to older artifact lineage rather than current validated source behavior.

---

## 16. Risks and validation requirements

### 16.1 Incorrect cross-checkpoint reuse

This is the highest correctness risk.

Prevent it by design:

- strict identity keys,
- in-memory exact cache only,
- no cross-checkpoint exact reuse.

### 16.2 Incremental block math mismatch

Need exact-mode validation that:

- baseline + incremental `B/C/q`
- matches full augmented recomputation

on small HH checkpoints.

### 16.3 Parameterization mismatch

The controller must honor logical/runtime parameterization metadata from imported operator-level scaffolds.  
This needs validation against the 5-op Re-ADAPT artifact and repo parameterization helpers.

### 16.4 Grouping-plan drift

Changing the grouping plan inside a checkpoint breaks exact shot extension semantics.

V1 should forbid regrouping inside a checkpoint.

### 16.5 Smoothness logic overreach

A full turning-angle penalty should not be promoted to the primary objective before the controller exists and baseline incremental probes are validated.

---

## 17. Disciplined V1 scope

V1 should include only:

- one live branch,
- local `stay` vs `append_candidate` decision,
- same-checkpoint baseline reuse,
- incremental candidate-block probes,
- scout / confirm / commit tiers,
- operator-level candidates only,
- exact/noiseless validation path first,
- local fake-backend noisy path only after exact-mode validation.

V1 should explicitly exclude:

- beam > 1,
- longer probe horizon,
- hybrid classical/QPU metric splitting,
- classical shadows,
- stochastic metric surrogates,
- IBM Runtime deployment,
- adaptive baseline from locked 7-term scaffold.

---

## 18. Sharp practical conclusion

The repo’s main gap is not that it “forgot to cache measurements.”  
The main gap is that there is no active realtime checkpoint controller in which:

- shortlist/probe logic,
- exact same-checkpoint incremental geometry reuse,
- and adaptive measurement intensity

can interact.

Once that controller exists, the first high-value reuse is:

> **same-checkpoint reuse of baseline geometry plus incremental candidate-block probing**

Cross-checkpoint information comes later, and mostly as scheduling priors rather than blind reuse of old metric/force values.

---

## 19. Immediate next artifact after this spec

The next concrete artifact should be a controller-contract document or proto-schema that fixes:

- checkpoint/state/cache keys,
- exact reuse validity rules,
- baseline versus incremental geometry objects,
- `stay` vs `append_candidate` admission rule,
- scout / confirm / commit thresholds,
- additive artifact serialization shape.

That is the shortest safe path from investigation to a live, testable HH adaptive realtime controller.
