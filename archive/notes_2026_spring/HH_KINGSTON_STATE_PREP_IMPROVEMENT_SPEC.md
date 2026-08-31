# HH Kingston State-Prep Improvement Spec

Date: 2026-03-30
Status: planning spec, repo-specific future addition
Scope: HH fixed-scaffold Kingston-line state-prep improvement
Non-scope: generic optimizer redesign, full raw-QPU architecture rewrite, driven-dynamics planning

## 0. Executive decision

For the Kingston line, the next improvement surface should focus on prepared-state quality, not optimizer cleverness.

The working model is:

```text
epsilon_prep
  ~= epsilon_thermal_init
   + sum epsilon_1q
   + sum epsilon_2q
   + sum_q tau_q / T2_q
   + epsilon_ZZ_crosstalk
```

In repo terms, this means:
- optimize physical schedule quality before changing the optimizer
- improve initialization hygiene before adding heavier mitigation
- use DD, twirling, symmetry filtering, and ZNE as downstream tools, not as substitutes for a bad routed state-prep circuit

The repo already contains part of this direction:
- Kingston-targeted compile scouting exists
- fixed-theta DD / twirling / symmetry post-processing surfaces exist
- gate-stateprep versus readout attribution exists

The repo does **not** yet expose the higher-yield diagnostics as first-class surfaces:
- scheduled duration and max-idle aware compile ranking
- empty-circuit / thermal start checks
- rep-delay sensitivity checks
- reference-state survival checks
- prefix-mirror return-probability localization

This spec defines that missing layer.

---

## 1. Why this addition is needed

Current Kingston evidence says the dominant residual is gate/state-prep, not readout.

Relevant existing evidence:
- [artifacts/reports/investigation_kingston_best_deltae_20260324.md](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/artifacts/reports/investigation_kingston_best_deltae_20260324.md)
- [artifacts/reports/investigation_kingston_partial_trace_and_rerun_plan_20260323.md](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/artifacts/reports/investigation_kingston_partial_trace_and_rerun_plan_20260323.md)

Current repo behavior is still too gate-count-centric for this problem:
- compile scouting selects by routed 2Q burden, depth, and size
- mitigation lanes emphasize readout, twirling, DD, and final ZNE
- no active workflow currently asks: "is the prepared variational state already wrong before measurement?"

That gap matters because:
- a lower parameter count does not guarantee lower prep error
- one removed logical term is much less important than one removed echoed entangler or one removed long idle segment
- an apparently improved noisy objective can still correspond to a less faithful prepared state

The intended new question is:

```text
Which routed Kingston instance prepares the least-wrong state
before measurement and post-processing?
```

---

## 2. Current coverage in the repo

This section is intentionally conservative. It exists to separate current support from future additions.

### 2.1 Implemented now

#### Kingston-targeted compile scouting

The repo already supports Kingston-aware compile scouting for imported fixed scaffolds.

Primary current surfaces:
- [pipelines/hardcoded/adapt_circuit_cost.py](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/hardcoded/adapt_circuit_cost.py)
- [artifacts/json/hh_kingston_6term_compile_scout_20260325T221406Z.json](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/artifacts/json/hh_kingston_6term_compile_scout_20260325T221406Z.json)

Current compile-scout outputs already include:
- physical layout
- compiled two-qubit count
- compiled CX count
- compiled ECR count
- compiled depth
- compiled size

This is useful and should remain the base layer.

#### Fixed-theta DD / twirling / ZNE add-ons

The repo already supports:
- saved-theta local DD probe
- local fake-backend gate twirling
- final ZNE audits on runtime routes

Primary current surfaces:
- [pipelines/exact_bench/noise_oracle_runtime.py](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/exact_bench/noise_oracle_runtime.py)
- [pipelines/hardcoded/hh_staged_cli_args.py](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/hardcoded/hh_staged_cli_args.py)
- [pipelines/hardcoded/hh_staged_noise_workflow.py](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/hardcoded/hh_staged_noise_workflow.py)
- [run_guide.md](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/run_guide.md)

Important current contract:
- DD is treated as a saved-theta local probe, not optimizer-loop mitigation
- twirling is opt-in
- final ZNE is a later audit surface, not the first diagnostic step

#### Symmetry verification and post-processing

The repo already supports:
- `verify_only`
- `postselect_diag_v1`
- `projector_renorm_v1`

These are meaningful for effective prepared-state quality because they can suppress symmetry-breaking damage after acquisition.

Primary current surfaces:
- [pipelines/exact_bench/noise_oracle_runtime.py](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/exact_bench/noise_oracle_runtime.py)
- [pipelines/exact_bench/hh_noise_hardware_validation.py](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/exact_bench/hh_noise_hardware_validation.py)
- [pipelines/hardcoded/adapt_pipeline.py](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/hardcoded/adapt_pipeline.py)

### 2.2 Partial today

#### Layout search exists, but schedule-aware ranking does not

The compile scout can already sweep backend/layout/seed combinations.

But the current selection metric is still:
- compiled two-qubit count
- compiled depth
- compiled size

It does **not** yet rank by:
- scheduled duration
- max per-qubit idle
- idle-by-qubit distribution
- explicit SWAP count
- line-health or calibration-weighted penalties

This is a meaningful gap because the current state-prep problem is dephasing- and routing-sensitive, not just depth-sensitive.

#### Observable grouping exists, but not weak-qubit-aware grouping

The repo already groups observables with QWC / abelian grouping logic.

But it does **not** currently adapt grouping to:
- weak physical qubits
- basis-rotation burden on fragile qubits
- state-prep preservation priorities for a chosen line

### 2.3 Not implemented yet

The following useful diagnostics were not found as first-class repo workflows:
- empty-circuit thermal check
- rep-delay sensitivity sweep
- reference-state survival check
- prefix-mirror `U_k^dagger U_k` return-probability check
- compile-scout exports with scheduled duration and max-idle metrics
- calibration-window-aware line selection summary

These are the main future additions defined below.

---

## 3. Desired future workflow

The intended workflow for Kingston state-prep improvement is:

1. compile/layout screen
2. initialization hygiene checks
3. held-theta DD A/B check
4. energy shot on the best routed instance
5. optional final ZNE or other downstream mitigation

This ordering is deliberate.

### 3.1 Principle

We should prefer:
- better routed circuit
- cleaner initialization
- shorter / better-balanced idle structure

before:
- heavier mitigation
- more optimizer effort
- interpretation of one unusually good noisy call

### 3.2 Compact rule

Optimize:
- physical schedule quality first
- initialization second
- idle suppression third
- mitigation only after those are understood

---

## 4. Proposed additions

This section describes **proposed** new surfaces. None of these names should be treated as already implemented unless a later code change adds them.

### 4.1 Compile scout v2: schedule-aware ranking

Extend the current compile-scout family to report schedule-sensitive metrics alongside gate counts.

#### Proposed new metrics

For every routed candidate, record:
- `scheduled_duration_dt`
- `scheduled_duration_sec` when timing metadata permits
- `max_qubit_idle_dt`
- `idle_by_physical_qubit_dt`
- `swap_count`
- `entangler_count_total`
- `entangler_count_by_link`
- `line_calibration_snapshot`
- `line_score_terms`

#### Proposed ranking score

Rank routed candidates by a weighted physical score rather than parameter count:

```text
S
  = alpha * entangler_count_total
  + beta  * scheduled_duration
  + gamma * max_qubit_idle
  + delta * swap_count
  + eta   * bad_link_penalty
```

Weights can start heuristic and be reported transparently in the artifact.

#### Proposed artifact additions

Add to compile-scout rows:
- chosen physical qubits
- scheduled duration
- max idle
- swap count
- entangler link histogram
- calibration metadata summary for the chosen line

Add to summary:
- best-by-gates candidate
- best-by-schedule candidate
- best-by-composite-score candidate

This avoids hiding tradeoffs.

### 4.2 Initialization hygiene diagnostics

Add explicit workflows that isolate shot-start contamination from circuit damage.

#### Proposed diagnostic mode: `thermal_init_check`

Goal:
- measure raw initialization quality on the exact same chosen physical line

Behavior:
- run an empty or identity-equivalent circuit
- collect computational-basis counts
- report per-qubit and joint-start contamination

Proposed outputs:
- `p1_by_qubit`
- `all_zero_fraction`
- `effective_hot_qubits`
- `rep_delay_used`
- `shots`

#### Proposed diagnostic mode: `rep_delay_sweep`

Goal:
- determine whether longer repetition delay materially improves start-state cleanliness

Behavior:
- repeat the thermal check over a small set of rep delays
- hold layout and backend fixed

Proposed outputs:
- `rep_delay_candidates`
- `p1_by_qubit_by_delay`
- `all_zero_fraction_by_delay`
- `recommended_rep_delay`

### 4.3 Reference-state survival diagnostic

#### Proposed diagnostic mode: `reference_state_survival`

Goal:
- test whether the encoded HH computational reference state survives acquisition well before adding variational structure

Behavior:
- prepare only the reference state for the scaffold
- no variational rotation content
- measure return to the expected computational basis bitstring

Proposed outputs:
- `reference_bitstring`
- `reference_survival_fraction`
- `reference_off_target_mass`
- `line_layout_physical_qubits`

This separates:
- bad initialization
from
- bad variational routing

### 4.4 Prefix-mirror localization

#### Proposed diagnostic mode: `prefix_mirror_check`

Goal:
- localize where the scaffold starts losing prepared-state fidelity

Behavior:
- for prefix depth `k`, build the circuit:

```text
U_k^dagger U_k
```

- start from the same reference state
- measure return probability to the expected reference bitstring

Proposed outputs:
- `prefix_depth`
- `return_probability`
- `compiled_metrics`
- `layout_physical_qubits`

This is the preferred prep-localization diagnostic because it maps failure to a prefix, not just to a final energy.

### 4.5 Held-theta DD A/B diagnostic

#### Proposed diagnostic mode: `held_theta_dd_ab`

Goal:
- test whether the current prep error is materially idle- / dephasing-limited

Behavior:
- same theta
- same routed layout
- same shots
- compare:
  - no DD
  - DD enabled

Optional paired variants later:
- with and without twirling
- same symmetry postprocessing

Proposed outputs:
- `baseline_energy`
- `dd_energy`
- `baseline_uncertainty`
- `dd_uncertainty`
- `delta_energy_dd_minus_baseline`
- `dd_applied_details`

Important contract:
- this remains a held-theta same-layout diagnostic
- it is not evidence that optimizer-loop DD should be enabled by default

### 4.6 Robust-theta preference

The repo should expose a clearer distinction between:
- lowest single noisy objective call
- most repeat-stable theta on the same routed line

#### Proposed additions

For fixed-theta and saved-theta evaluations, report:
- repeat-stability across reruns
- mean and spread at the final theta
- "best single shot" versus "best stable theta" summaries

This does not require a new optimizer. It requires better reporting and selection semantics.

---

## 5. Proposed interfaces

These interfaces are proposed-only.

### 5.1 Proposed CLI surfaces

Potential future flags:
- `--include-state-prep-thermal-check`
- `--include-state-prep-rep-delay-sweep`
- `--include-reference-state-survival`
- `--include-prefix-mirror-check`
- `--prefix-mirror-depths 1,2,3,...`
- `--state-prep-score alpha,beta,gamma,delta,eta`
- `--state-prep-best-of schedule|gates|composite`

If a smaller first slice is preferred, only add:
- `--include-state-prep-thermal-check`
- `--include-held-theta-dd-ab`

### 5.2 Proposed artifact fields

For Kingston-facing summaries, standardize:
- `layout_physical_qubits`
- `compiled_two_qubit_count`
- `swap_count`
- `scheduled_duration_dt`
- `max_qubit_idle_dt`
- `idle_by_physical_qubit_dt`
- `thermal_init`
- `reference_state_survival`
- `held_theta_dd_ab`
- `state_prep_score`

### 5.3 Proposed report-table columns

For human-facing Kingston comparisons:
- scaffold label
- chosen physical qubits
- 2Q count
- swap count
- scheduled duration
- max idle
- thermal all-zero fraction
- reference survival
- held-theta DD delta
- energy result

---

## 6. First safe implementation slice

The first implementation slice should stay narrow and decision-safe.

### Phase 1

Extend compile scouting to include:
- scheduled duration
- max idle
- swap count
- calibration snapshot summary

No workflow changes yet. Only better routing evidence.

### Phase 2

Add initialization diagnostics:
- thermal init check
- reference-state survival

These are low-risk and high-yield.

### Phase 3

Add held-theta DD A/B reporting:
- same layout
- same theta
- same shots

Keep it explicitly diagnostic.

### Phase 4

Add prefix-mirror localization:
- only after the schedule and init surfaces are already stable

This is the most conceptually valuable new diagnostic, but it is not required for the first safe slice.

---

## 7. Non-goals

This future addition does **not** aim to:
- make the optimizer the primary state-prep lever
- replace existing compile scouts
- remove current DD/twirling/ZNE routes
- claim that readout no longer matters at all
- rewrite the repo around a generic hardware-agnostic calibration framework
- solve full raw-QPU architecture questions already covered in `notes/QPU_RAW_VQE_IMPLEMENTATION_SPEC.md`

---

## 8. Acceptance criteria

This future addition should be considered correctly implemented only when:
- the repo can rank Kingston routed candidates by a schedule-aware score, not just 2Q/depth/size
- at least one explicit thermal/init diagnostic exists and emits durable JSON
- at least one explicit held-theta DD A/B diagnostic exists and emits durable JSON
- Kingston comparison reports can show `layout / 2Q / duration / max idle / init result / DD delta` in one table
- documentation clearly distinguishes:
  - current compile scout
  - current mitigation surfaces
  - new state-prep diagnostics

---

## 9. Default execution order for future Kingston campaigns

When this addition exists, the default Kingston improvement sequence should be:

1. transpile 6-term and 7-term across Kingston layout/seed candidates
2. rank by schedule-aware score
3. run thermal/init checks on the chosen line
4. run reference-state survival
5. run held-theta DD A/B
6. run the energy evaluation
7. escalate to final ZNE only if the earlier steps justify it

This sequence keeps the prepared-state question primary and prevents mitigation from masking a bad routed circuit.
