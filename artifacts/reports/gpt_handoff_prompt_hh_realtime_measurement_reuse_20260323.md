# GPT Handoff Prompt: HH Realtime McLachlan Measurement-Reuse / Shortlist / Smoothness Architecture

Copy-paste prompt:

```text
You are helping design the next architecture for a Python Hubbard-Holstein (HH) repository that already has strong ADAPT and replay machinery, but does NOT currently have an active live realtime McLachlan/projected-real-time controller in the current execution path.

I do NOT want generic VQS advice. I want a repo-aware architectural proposal that respects the current codebase shape, the existing ADAPT machinery, the historical artifacts, and the projective McLachlan math below.

Please reason carefully and produce an implementation-oriented design memo, not just broad ideas.

## What the current repository does

Current live HH staged flow is effectively:

warm start -> ADAPT continuation/operator selection -> matched-family replay -> exact/Suzuki/CFQM propagation

The current live path does NOT run a live projected-real-time / McLachlan controller.

Relevant live files / evidence:
- `pipelines/hardcoded/hh_staged_cli_args.py`
  - dynamics section exposes `--noiseless-methods`, `--t-final`, `--num-times`, `--trotter-steps`, `--exact-steps-multiplier`, CFQM flags
  - noise section exposes static `--shots`, `--oracle-repeats`, `--oracle-aggregate`
- `pipelines/hardcoded/hubbard_pipeline.py`
  - `_simulate_trajectory(...)` supports only `suzuki2`, `piecewise_exact`, `cfqm4`, `cfqm6`
- `pipelines/hardcoded/adapt_pipeline.py`
  - has shortlist/probe/batching logic, but only for ADAPT selection
- `pipelines/hardcoded/hh_continuation_stage_control.py`
  - `allowed_positions(...)`, `detect_trough(...)`, `should_probe_positions(...)`
- `pipelines/hardcoded/hh_continuation_scoring.py`
  - `shortlist_records(...)`
  - `Phase2NoveltyOracle`
  - `Phase2CurvatureOracle`
  - `CompatibilityPenaltyOracle`
  - `greedy_batch_select(...)`
  - `MeasurementCacheAudit` (important: this is accounting-only grouped-reuse tracking, not true persistence/reuse of measured geometry data)
- `pipelines/exact_bench/noise_oracle_runtime.py`
  - `OracleConfig` has fixed `shots`, `oracle_repeats`, `oracle_aggregate`
  - no live change-triggered shot escalation/de-escalation policy

## What historical artifacts show

There are legacy realtime artifacts showing that the repo once carried realtime-side shortlist/probe/beam ideas, but they are not wired into the current live path.

Important legacy artifact:
- `artifacts/json/hh_staged_adapt_vqs_l2_g0p5_nph2_phaseb_only_realtime_vqs.json`

This artifact shows:
- realtime-side shortlist fields like `max_shortlist`
- probe fields like `probe.horizon_steps`
- beam fields like `beam.live_branches`
- branch actions such as `stay` and `append_candidate`
- `beam_pruned` branches

But the winning branch still ends in `mixed_failure`, so treat this as design precedent, not proof of a validated production path.

## Important empirical problem to solve

Historical shot-based measured McLachlan / geometry assembly blew up in measurement cost.
Even a tiny smoke/preflight bundle at `time_end = 0.01` already showed very large transition and ancilla workload.

Relevant artifacts:
- `artifacts/json/phase_a_local_rehearsal_20260317_134642/light_shots_fixed_patch_smoke.bundle.json`
- `artifacts/json/phase_a_local_rehearsal_20260317_134642/active_window_shots_fixed_patch_smoke.bundle.json`

The core pain point is that the workflow appears to remeasure too much and does not sufficiently capitalize on past measurements or checkpoint structure.

## Critical architecture correction

If we use a lean scaffold/manifold for future adaptive realtime work, it should be operator-level, not gate-level.

Specifically:
- the gate-pruned 7-term scaffold is NOT the right object for fresh ADAPT / fresh adaptive realtime geometry
- the better lean candidate is the operator-level 5-op Re-ADAPT artifact
- preferred target lean artifact: `artifacts/json/hh_prune_nighthawk_readapt_5op.json`

So if you discuss a lean starting manifold, please think in terms of operator-level geometry and ADAPT-compatible structure, not post-compile gate-pruned VQE scaffolds.

## The changes I want prioritized

I want a concrete architecture for a future HH adaptive realtime McLachlan stack that prioritizes the following, in this order:

1. **Exact batching/grouping and checkpoint-level measurement reuse**
2. **Coarse-to-fine shortlist/probe for realtime dynamics**
3. **Do NOT fully remeasure the full metric at every checkpoint unless needed**
4. **Penalize non-smoothness / directional changes in the realtime control loop**
5. **Use more measurements when change is detected and fewer when the system is calm**

For now, please DEFER these:
- hybrid classical/QPU metric splitting
- classical shadows
- stochastic metric surrogates / QN-SPSA-style constant-cost metric replacement

Those may become relevant later, but I want the design to first exhaust exact grouped reuse + shortlist/probe + refresh control + smoothness/change-aware measurement scheduling.

## What I think is missing right now

My current understanding is:
- live realtime shortlist/probe is not currently implemented
- live propagation-side non-smoothness/directional-change control is not currently implemented
- current “directional” or “curvature” logic in source is selection-time proxy logic for ADAPT, not timestep-path smoothness control
- measurement policy is effectively static in the active oracle path
- `MeasurementCacheAudit` is not enough; I want actual checkpoint-level persistence/reuse of measured geometry / observables where mathematically legitimate

Please either confirm or correct this understanding explicitly.

## Relevant mathematics (repo notation)

Use the projective McLachlan geometry in `MATH/Math.md`, Section 17, as the mathematical backbone.

### Projective setup
At normalized state `|psi>`:
- horizontal projector:
  `Q_psi = I - |psi><psi|`
- raw tangent vectors:
  `tau_j = ∂_{theta_j} |psi(theta; S)>`
- horizontal tangents:
  `bar_tau_j = Q_psi tau_j`
- tangent matrix:
  `bar_T = [bar_tau_1, ..., bar_tau_m]`
- projective drift:
  `bar_b = Q_psi (-i H psi) = -i (H - E_psi) psi`
- projective drift norm:
  `||bar_b||^2 = Var_psi(H)`

### Fixed-structure McLachlan local solve
The local solve is:
`dot(theta)_0 = argmin_{dot(theta) in R^m} || bar_T dot(theta) - bar_b ||^2`

with:
- projective metric:
  `bar_G_ij = Re <bar_tau_i, bar_tau_j>`
- force:
  `bar_f_i = Re <bar_tau_i, bar_b>`
- stationarity system:
  `bar_G dot(theta)_0 = bar_f`

### Regularized runtime solve
A stabilized solve uses:
`K = bar_G + Lambda`
`K dot(theta)_lambda = bar_f`

and tracks:
- intrinsic model inadequacy:
  `epsilon_proj^2 = ||bar_b||^2 - bar_f^T bar_G^+ bar_f`
- actual damped step mismatch:
  `epsilon_step^2 = || bar_T dot(theta)_lambda - bar_b ||^2`

These are important:
- `epsilon_proj^2` = manifold insufficiency / model inadequacy
- `epsilon_step^2` = actual runtime mismatch after regularization

### Candidate adaptation gain
The repo math also gives residualized candidate gain via Schur complement:
- `S = C + Lambda_c - B^T K^{-1} B`
- `w = q - B^T K^{-1} bar_f`
- `Delta_lambda = w^T S^+ w`

For a single unregularized candidate direction, the clean geometric gain is:
`Delta_0 = ( Re <u_perp, r_0> )^2 / ||u_perp||^2`
where:
- `u_perp = (I - P_barT^R) u`
- `r_0 = (I - P_barT^R) bar_b`

### Adapt trigger
Normalized miss:
`rho_miss = epsilon_proj^2 / (Var_psi(H) + eps)`

Interpretation:
- small `rho_miss` -> current manifold already captures local projective velocity well
- large `rho_miss` -> manifold misses important directions

### Branch / segment objective
The repo math also supports segment / branch scoring using residual accumulation, e.g. integrated residual plus resource/adaptation penalties.

### Important conceptual point
The primary local realtime control quantity should be derivative mismatch / projective residual geometry, not static energy ranking.
Secondary scores can still help for shortlist ranking, but they should not replace the local geometric control objective.

## What I want designed

Please propose a concrete architecture for a future HH realtime controller that fits this repo.
I want you to describe the best design for these five items:

### A. Realtime controller entrypoint
Design a repo-compatible live projected-real-time controller / checkpoint loop that would sit alongside the current staged HH workflow.

I want explicit discussion of:
- where the controller should live conceptually
- whether it should be a sibling of current noiseless profile execution or a dedicated new module
- what state/checkpoint object it should own
- how it should interact with current ADAPT artifacts and replay/handoff machinery

### B. Checkpoint-level measurement reuse subsystem
Design a true measurement reuse subsystem for a realtime checkpoint.

I want a concrete proposal for a “master observable pool” / grouped measurement plan that can reuse data across as many of the following as legitimately possible on the same prepared state:
- metric / QGT block pieces
- force vector pieces
- candidate cross terms (`B`, `C`, `q`-style objects)
- energy
- Hamiltonian variance
- any defect-normalization observables

Please be explicit about:
- what should be cached
- what can be reused exactly vs what must be invalidated
- how cache keys should be defined (state / structure / window / operator set / grouping plan / basis plan / etc.)
- how reuse should interact with active-window changes and ansatz growth
- why this is more than the existing accounting-only `MeasurementCacheAudit`

### C. Realtime shortlist / probe / stay-vs-append policy
I want a coarse-to-fine shortlist/probe controller for realtime dynamics.

I want the runtime controller to be able to consider actions like:
- `stay`
- `append_candidate`
- possibly `expand_window`
- possibly `refresh_metric`
- possibly `defer_adapt`

Please propose:
- how to generate a cheap shortlist
- how to do a horizon-1 or short probe rollout cheaply
- whether a tiny beam should be used (`live_branches = 1` or maybe very small >1)
- how historical artifact ideas like `max_shortlist`, `probe.horizon_steps`, and `stay`/`append_candidate` should be modernized
- how this policy should interact with a lean operator-level 5-op Re-ADAPT style scaffold

### D. Propagation-side non-smoothness / directional-change penalty
This is very important.
I want a mathematically defensible penalty or controller term that discourages undesirable abrupt changes in the realtime path.

For example, I want you to think about things like:
- abrupt turning in `dot(theta)`
- unstable sign flips / directional reversals
- sudden support changes after adaptation
- large control changes not justified by enough geometric improvement

Please propose a projective-geometry-aware way to do this.
I do NOT want a purely ad hoc optimizer penalty unless you can justify it geometrically.

You may consider things like:
- whitened velocity change
- turning-angle penalties between successive retained-support velocities
- residual-improvement-vs-direction-change tradeoffs
- hysteresis / admission thresholds for structure changes

But please choose what you think is mathematically and architecturally best.

### E. Change-triggered measurement escalation / de-escalation
I want the measurement budget to react to detected change.

Desired behavior:
- when geometry is stable and little is changing, use cheaper measurement effort
- when a real change is detected, increase measurements / repeats / precision
- if a shortlist is clearly dominated, do not spend high precision on all candidates
- refine only contenders

Please design a concrete control policy using available or natural telemetry such as:
- `epsilon_proj_sq`
- `epsilon_step_sq`
- `stabilization_gap_sq`
- `damping_gap_sq`
- condition number / rank changes
- active-window changes
- action changes like `stay` vs `append_candidate`
- shortlist score margin collapse / overlap
- branch instability indicators

I want a policy that is exact-grouped-reuse first and adaptive-shot second.
In other words: first avoid duplicated work, then allocate more precision only where needed.

## Constraints / preferences

Please respect these constraints:
- preserve operator-level correctness and ADAPT compatibility
- do not treat the gate-pruned 7-term fixed scaffold as the main adaptive-realtime object
- prefer operator-level 5-op Re-ADAPT / lean operator-level scaffold logic
- do not propose a design that depends first on hybrid classical/QPU splitting, classical shadows, or stochastic metric surrogates
- assume we care about projective geometry and the interpretation of `epsilon_proj`, `epsilon_step`, and related telemetry
- assume this repo already has useful ADAPT-side shortlist/probe/batching machinery that should be reused conceptually where possible
- assume the current live dynamics path is replay + exact/Suzuki/CFQM, so any realtime controller must be added as a distinct architectural layer

## What I want in your answer

Please give me the following sections:

1. **Diagnosis of current architecture**
   - confirm precisely what exists vs what does not

2. **Target architecture**
   - proposed modules / responsibilities / data flow

3. **Mathematical control design**
   - how you would formalize shortlist/probe, smoothness penalty, and change-triggered measurement control using the McLachlan/projective quantities above

4. **Checkpoint cache / reuse design**
   - exact reuse vs invalidation rules
   - what gets cached at each checkpoint

5. **Implementation roadmap in repo terms**
   - likely files/modules to create or extend
   - what should be reused from current ADAPT infrastructure
   - what should stay separate

6. **Risks / failure modes**
   - what could go wrong mathematically or architecturally

7. **Recommended minimal first version**
   - if we wanted a disciplined V1 that gives the highest yield without overbuilding, what exactly should it include?

Be concrete, and feel free to give pseudocode / schemas / control-loop structure.

Do NOT write code unless I ask for it explicitly.
```
