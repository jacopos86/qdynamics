## Final Prompt
<taskname="Full Reopt Repair"/>
<task>
Investigate, document, and minimally repair the Paper-I Hubbard--Holstein SNAKE full-window/full-reoptimization failure described in `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_full_reopt_implementation_error_20260710.md`. Update that investigation note with code findings, implement the smallest code repair that addresses the full-policy final-refit metadata and repeated zero-gain exact-child admission failure, add focused tests, and rerun the relevant diagnostic. Preserve existing windowed and append-only behavior. Do not edit manuscript files.
</task>

<architecture>
- `pipelines/static_adapt/adapt_pipeline.py` is the ADAPT driver: builds Phase-I/II/III candidate records, admits selected candidates, resolves per-step reoptimization windows, writes history rows, and serializes `final_full_refit` metadata.
- `pipelines/static_adapt/engine_support.py` owns `_VALID_REOPT_POLICIES`, `_resolve_reopt_active_indices`, and reduced-objective helpers. It already supports `policy="full"` by activating all parameters per step.
- `pipelines/static_adapt/nested_windows.py` owns `NestedRefitWindow`, `NestedWindowAccounting`, `predict_nested_refit_window`, and serialization. Passing `policy="full"` produces full active post indices and policy telemetry; the suspicious `windowed` geometry telemetry comes from the pipeline fixed-local geometry helper, not this module.
- `pipelines/scaffold/hh_continuation_types.py` defines `CandidateFeatures`, including `novelty`, `phase3_duplicate_penalty`, `family_repeat_cost`, nested/geometry windows, optimizer active indices, and runtime-split identity fields.
- `pipelines/scaffold/hh_continuation_scoring.py` computes Phase-III scores. `FullScoreConfig.duplicate_penalty_weight` defaults to `0.0`, `build_candidate_features` initializes `phase3_duplicate_penalty=0.0`, and `family_repeat_cost_from_history` is family-streak based rather than exact candidate/key duplicate blocking.
- `pipelines/static_adapt/plateau_acquisition.py` contains exact candidate-position duplicate keys and blockers, but those are currently Route-C plateau-specific.
- `test/test_adapt_vqe_integration.py` contains the existing full-policy, nested/geometry window, Route-C final-refit skip, resolver, and final_full_refit tests to extend.
</architecture>

<selected_context>
MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_full_reopt_implementation_error_20260710.md: primary diagnostic note, full file. Records the failed settings, repeated exact child label, zero realized gain, final_full_refit `requested=false/executed=false`, nested vs geometry policy mismatch, and candidate repair ideas.
MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_fullwindow_fullreopt_degradation_diagnostic_20260710.md: companion diagnostic note, full file. Useful for enlarged candidate universe, physical lanes, maturity/controller shortlist context.
pipelines/static_adapt/adapt_pipeline.py: focused implementation slices covering imports, nested/geometry window helpers, `_run_hardcoded_adapt_vqe` config, metadata attachment, plateau duplicate contrast, selection/admission, per-step reopt, rollback/history, final full-refit, and payload serialization.
pipelines/static_adapt/engine_support.py: slices for valid policies, `_resolve_reopt_active_indices`, and reduced objective helpers.
pipelines/static_adapt/nested_windows.py: slices for dataclasses, `predict_nested_refit_window`, serializers, and accounting JSON import.
pipelines/static_adapt/plateau_acquisition.py: full file for existing exact duplicate key semantics and Route-C blocker behavior.
pipelines/scaffold/hh_continuation_types.py: slice with `CandidateFeatures` fields and export behavior.
pipelines/scaffold/hh_continuation_scoring.py: slices for duplicate/family repeat scoring defaults and candidate feature construction.
test/test_adapt_vqe_integration.py: slices for existing full-policy tests, nested/geometry tests, Route-C final-refit skip/geometry decoupling, resolver tests, and final_full_refit integration tests.
</selected_context>

<relationships>
- User settings enter `_run_hardcoded_adapt_vqe(...)` in `adapt_pipeline.py` -> `adapt_reopt_policy_key`, `adapt_window_size_val`, `phase3_geometry_window_size_val`, `adapt_full_refit_every_val`, and `adapt_final_full_refit_val`.
- Per-step reopt path: `adapt_pipeline.py` -> `_resolve_reopt_active_indices(policy=adapt_reopt_policy_key, ...)` in `engine_support.py`; `policy="full"` returns all logical indices and effective policy `"full"`.
- Nested candidate telemetry path: `_predict_nested_refit_window_for_position(... policy=adapt_reopt_policy_key ...)` -> `nested_windows.predict_nested_refit_window` -> serialized into `nested_refit_window`.
- Geometry telemetry path: `_predict_phase3_geometry_window_for_position(... geometry_window_size ...)` currently calls `predict_nested_refit_window(policy="windowed", ...)`; this feeds `phase3_geometry_nested_refit_window` and is intentionally W3/Wopt-decoupled when `phase3_geometry_window_size >= 1`.
- Candidate scoring path: `build_candidate_features` / `build_full_candidate_features` -> `CandidateFeatures` -> record metadata -> `selected_batch_records_for_history` -> history row fields such as `novelty`, `phase3_duplicate_penalty`, `family_repeat_cost`, `generator_id`, runtime-split child fields.
- Duplicate blocking path that already exists: `plateau_acquisition.candidate_key_from_record` + `duplicate_status` is used by `_route_c_plateau_payload_for_record`; non-plateau selection/admission does not use this exact duplicate blocker.
- Final refit path: after the loop, `adapt_pipeline.py` builds `final_full_refit_meta`; current code sets `requested` and executes only when `adapt_reopt_policy_key == "windowed"`, so `policy="full"` plus `adapt_final_full_refit=True` serializes as not requested.
</relationships>

<discoveries>
- The final-refit metadata bug is direct: the final full-prefix refit block says “windowed policy only” and gates `requested`/execution on `adapt_reopt_policy_key == "windowed"`. This contradicts the user setting `adapt_final_full_refit=true` under `adapt_reopt_policy=full`.
- `last_was_full` in that same block already recognizes `history[-1]["reopt_policy_effective"] == "full"` and a full active count, so a minimal full-policy repair can likely mark the refit as requested and skip redundant execution with the existing `last_depth_already_full_prefix` semantics when appropriate.
- The repeated zero-gain admission is not caught by ordinary rollback: rollback only triggers on energy regression greater than `adapt_rollback_tolerance_val`, not on `energy_current == energy_prev` or microscopic no-gain admissions.
- General scoring does not currently suppress exact repeats: `phase3_duplicate_penalty` defaults to zero and is only a weighted tie-break when configured; `family_repeat_cost` is family-level streak cost; exact duplicate candidate-position blocking is Route-C plateau-only.
- The nested/geometry policy mismatch is explained by fixed-local geometry scoring: actual optimizer/nested policy can be full while `phase3_geometry_nested_refit_window.policy_requested` is produced by a helper that hardcodes `policy="windowed"`. Existing tests assert W3/Wopt decoupling, so preserve the decoupled geometry window behavior while making telemetry less misleading if repaired.
- Git status was dirty before handoff; the two selected diagnostic notes are untracked. Avoid reverting or touching unrelated dirty files.
</discoveries>

<repair_constraints>
- Keep append-only and existing windowed behavior intact, including Route-C dormant final-refit skip semantics.
- Keep `phase3_geometry_window_size >= 1` as a fixed local Phase-III scoring window unless there is a deliberate, tested metadata-only change.
- For duplicate/no-gain handling, be conservative: do not ban useful repeats that produce real descent. The target failure is exact repeated child/operator identity with realized refit improvement below the existing energy-step tolerance/noise floor.
- Prefer a narrow helper/metadata repair over broad changes to candidate scoring or lane scheduling.
</repair_constraints>

<test_plan>
Add or update focused tests in `test/test_adapt_vqe_integration.py`:
- Full policy + `adapt_final_full_refit=True` should serialize `final_full_refit.requested is True`; if the last depth already reoptimized the full prefix, assert the existing skip reason rather than requiring a redundant optimizer call.
- Preserve `adapt_final_full_refit=False` skip behavior and windowed Route-C dormant skip behavior.
- Preserve `_resolve_reopt_active_indices(policy="full")` and full-policy active-count behavior.
- Preserve `test_phase3_geometry_window_decouples_from_full_optimizer_refit`; if geometry telemetry is changed, add assertions that distinguish fixed-local scoring window from optimizer policy without changing W3/Wopt decoupling.
- Add a narrow regression for exact repeated candidate/child admission with zero or sub-tolerance realized gain under full policy/full window/repeats allowed, verifying the second no-gain duplicate is rejected/rolled back/cooled down according to the chosen minimal mechanism and that productive repeats are not blocked.
After tests, rerun a small version of the failed weak-weak diagnostic settings and confirm: no long run of the same zero-gain child, coherent `final_full_refit` metadata, and unchanged windowed/append-only targeted tests.
</test_plan>

<ambiguities>
- Exact duplicate identity should be chosen carefully. Route-C uses `(generator_id or candidate_label, position_id)` via `candidate_key_from_record`; the failure note emphasizes the same child label at repeated selected positions. Decide whether the guard key should include position, generator id, label, pool index, or runtime-split child metadata, and document that in the note.
- “Existing energy-step tolerance” is not a single obvious threshold because the failing run disabled eps-energy termination. Candidate thresholds include `eps_energy`, `adapt_rollback_tolerance_val`, or a tiny numeric floor; choose conservatively and test it.
- It is unclear whether `phase3_geometry_nested_refit_window.policy_requested` should change from `"windowed"` to `"full"` or whether a separate metadata field should clarify that the geometry window is fixed-local while optimizer policy is full. Existing decoupling behavior must remain stable.
</ambiguities>

## Selection
- Files: 9 total (3 full, 6 slice)
- Total tokens: 107285 (Auto view)
- Token breakdown: full 11667, slice 95618

### Files
### Selected Files
/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/
├── MATH/
│   └── paper_facing/
│       └── paper_I_static_scaffold/
│           ├── paper_i_hh_full_reopt_implementation_error_20260710.md — 1,840 tokens (full)
│           └── paper_i_hh_fullwindow_fullreopt_degradation_diagnostic_20260710.md — 1,922 tokens (full)
├── pipelines/
│   ├── scaffold/
│   │   ├── hh_continuation_scoring.py — 9,678 tokens (lines 1-125 (Scoring imports and FullScoreConfig/SimpleScoreConfig defaults, including duplicate_penalty_weight defaulting to 0.0.), 1530-1625 (family_repeat_cost_from_history implementation; family-streak cost only, not exact selected candidate/key duplicate suppression.), 2520-2665 (Phase-III auxiliary tie-break and canonical score logic; duplicate penalty is only a weighted tie-break and defaults inactive.), 3520-3635 (build_full_candidate_features score/novelty update path for full candidate records, including phase3_reduced_novelty and canonical score fields.), 5570-5905 (build_candidate_features construction of CandidateFeatures with nested/geometry windows, family repeat cost, and phase3_duplicate_penalty initialized to 0.0.))
│   │   └── hh_continuation_types.py — 5,048 tokens (lines 1-520 (Continuation dataclasses and CandidateFeatures implementation fields, including duplicate/novelty/family repeat, nested/geometry window metadata, optimizer active indices, runtime-split identity, and dict export.))
│   └── static_adapt/
│       ├── adapt_pipeline.py — 54,370 tokens (lines 1-240 (Imports and module dependencies needed by selected ADAPT slices, including scoring/types, nested windows, plateau acquisition helpers, and engine support resolvers.), 960-1065 (Nested refit and Phase-III geometry window prediction helpers; actual policy passthrough vs fixed-local geometry helper currently hardcoding policy='windowed'.), 1500-2019 (_run_hardcoded_adapt_vqe signature and validation for adapt_reopt_policy/window/topk, phase3_geometry_window_size, full/final refit flags, insertion mode, and rollback mode.), 3860-4210 (Phase-III geometry window choice and metadata attachment, including serialized nested_refit_window and phase3_geometry_nested_refit_window payloads and W3/Wopt decoupling fields.), 8260-8425 (Route-C plateau duplicate and failed-family scoring gate using candidate_key_from_record/duplicate_status; exact duplicate blocking is currently plateau-trial specific.), 17820-17935 (Non-plateau selected-record source choice and sorting: source lock, batching, Phase-III shortlist, admission source records, and fallback full_records[0].), 18380-19095 (Admission splice and selected feature capture, nested/geometry payload extraction, then per-step reoptimization setup via _resolve_reopt_active_indices and W3/Wopt geometry-only decoupling.), 19635-19840 (Post-optimization rollback threshold and Route-C plateau failure/success handling; shows ordinary rollback only fires on energy regression above tolerance, not zero realized gain.), 19960-20280 (Realized-gain bookkeeping and history row core fields: energy_before/after, delta_energy, eps_energy_step_abs, rollback flags, reopt active counts, nested/geometry payloads.), 20375-20495 (Continuation telemetry fields in history row including novelty, family_repeat_cost, refit_window_indices, and physical/operator lane metadata relevant to repeated child diagnosis.), 20645-20830 (Phase-III generator/runtime-split identity fields and final history append/admission commit gate, including phase1_features_history update.), 21020-21480 (Final full-prefix refit block; currently requested/executed are gated by adapt_reopt_policy_key == 'windowed' even though last_was_full recognizes full-prefix reoptimization.), 23780-23995 (Final payload assembly serializing adapt_reopt_policy/window/full_refit settings, history, and final_full_refit metadata.))
│       ├── engine_support.py — 4,424 tokens (lines 1-180 (Imports and foundational helpers/types used by engine support; enough context for policy constants and reduced objective helpers.), 900-1142 (Reoptimization policy constants plus _resolve_reopt_active_indices, _make_reduced_parameter_expander, and _make_reduced_objective implementations; confirms policy='full' already optimizes all parameters per step.))
│       ├── nested_windows.py — 5,407 tokens (lines 1-125 (NestedRefitWindow and NestedWindowAccounting dataclasses, including policy_requested/policy_effective telemetry fields.), 360-639 (predict_nested_refit_window and serializers; shows requested/effective policy handling where policy='full' yields full active post indices and serialized policy fields.), 790-835 (nested_window_accounting_from_json round-trip helper used by adapt_pipeline when consuming selected nested window metadata.))
│       └── plateau_acquisition.py — 7,905 tokens (full)
└── test/
    └── test_adapt_vqe_integration.py — 16,691 tokens (lines 1-120 (Test imports and handles for _run_hardcoded_adapt_vqe, reopt resolver, configs, and CLI config.), 4368-4465 (TestAdaptReoptPolicyFull setup and full-policy prefix-change regression test.), 4485-4659 (Invalid reopt policy and wrapper passthrough tests around adapt_reopt_policy/full behavior.), 4850-5099 (Nested/geometry window tests for selector policy, geometry window counts, and actual-window metadata.), 5650-5885 (Route-C final-refit/geometry tests, especially full optimizer refit decoupled from Phase-III geometry window.), 9330-9560 (Pure _resolve_reopt_active_indices and reduced-objective tests verifying full, append_only, and windowed periodic-full behavior.), 9624-10120 (Windowed reopt integration and final_full_refit metadata tests, false-final-refit skip, periodic full refit cadence, and full policy active-count regression.))


---

## Generated Plan

## Chat Send ✅
- **Chat**: `full-reopt-repair-5642E4` | **Mode**: plan

### Response
## 1. **Summary**

Make a targeted repair in `pipelines/static_adapt/adapt_pipeline.py`: treat `adapt_reopt_policy="full"` as a supported final-full-refit policy for metadata, but skip redundant execution when the last adaptive step already optimized the full prefix; add a conservative post-optimization guard that structurally rolls back and future-blocks exact repeated candidate identities only when a duplicate repeat realizes no meaningful energy gain. Extend focused tests in `test/test_adapt_vqe_integration.py`, update only the diagnostic note, and rerun a short weak–weak smoke before any paper-facing rerun.

## 2. **Current-state analysis**

- `engine_support.py::_resolve_reopt_active_indices` already supports `policy="full"` correctly: it returns all logical indices and effective policy `"full"`.
- `adapt_pipeline.py` partially ignores that support after the adaptive loop:
  - `final_full_refit_meta["requested"]` is currently gated by `adapt_reopt_policy_key == "windowed"`.
  - The final refit execution block is also gated by `"windowed"`.
  - Existing `last_was_full` logic already recognizes `"full"` in principle, but the outer gate prevents it from running.
- Per-step admission flow:
  1. Candidate records are built as `CandidateFeatures`.
  2. Records are shortlisted/scored.
  3. A selected candidate is spliced into `selected_ops`/`theta`.
  4. `_resolve_reopt_active_indices(...)` chooses active coordinates.
  5. The reduced/full optimizer runs.
  6. Rollback only fires on energy regression above `adapt_rollback_tolerance_val`.
  7. Zero-gain duplicate admissions remain structurally committed.
- Existing duplicate machinery is not used for normal Phase-III admission:
  - `phase3_duplicate_penalty` defaults to `0.0`.
  - `family_repeat_cost_from_history` is family-streak based, not exact identity blocking.
  - `plateau_acquisition.candidate_key_from_record` and `duplicate_status` are Route-C plateau-only.
- The suspicious `phase3_geometry_nested_refit_window.policy_requested="windowed"` is caused by fixed-local Phase-III geometry prediction, not by optimizer reoptimization. Existing `w3_wopt_decoupled` and `optimizer_active_refit_indices` already distinguish scoring geometry from optimizer policy.

This should be a targeted repair, not a scoring/lane refactor, because the two blocking defects are localized: final-refit gating and post-refit admission acceptance of duplicate no-gain records.

## 3. **Design**

### A. Final full-refit metadata repair

Modify only `adapt_pipeline.py` final-refit section.

Define an internal boolean equivalent to:

```py
final_full_refit_policy_supported =
    adapt_reopt_policy_key in {"windowed", "full"}
```

Use it for:

- `final_full_refit_meta["requested"]`
- final-refit execution guard
- Route-C dormant-record skip guard

Behavior after change:

| Policy | `adapt_final_full_refit=True` | Expected metadata |
| --- | --- | --- |
| `append_only` | true | unchanged: `requested=false` |
| `windowed` | true | unchanged existing behavior |
| `full` | true | `requested=true`; usually `executed=false`, `skipped_reason="last_depth_already_full_prefix"` |
| `full` | false | `requested=false`, `executed=false` |

Also adjust the “last step already full” check to recognize effective policy strings such as:

- `"full"`
- `"full+nested_window_v1"`
- `"windowed_periodic_full"`

Only treat the last step as full if `history[-1]["reopt_active_count"] == len(selected_ops)`.

### B. Zero-gain exact duplicate guard

Add a narrow guard in `adapt_pipeline.py`.

#### Identity choice

Use an identity-level key for blocking, with diagnostic candidate-position key retained for telemetry.

Canonical identity priority:

1. Runtime split child identity if available.
2. `candidate_label`.
3. `generator_id`.
4. `selected_logical_op`.
5. `selected_op`.

Use `plateau_acquisition.candidate_key_from_record(...)` only for telemetry when possible, because its `(identity, position_id)` shape is too position-specific for the observed repeated-child failure.

#### Trigger conditions

The guard triggers only when all are true:

- `allow_repeats` is true.
- The current admission is singleton, not batch.
- Route-C plateau trial is not active.
- The same canonical identity exists in previously committed history rows.
- Realized gain is below tolerance:

```py
realized_gain = energy_prev - energy_current
threshold = max(
    eps_energy,
    adapt_rollback_tolerance_val,
    floating_point_floor_relative_to_energy_prev,
)
trigger = realized_gain <= threshold
```

Ignore prior rows where `structural_rollback` is true so rejected attempts do not count as committed duplicates.

#### Action on trigger

After optimization and after existing regression rollback logic, but before history row construction:

- Restore pre-admission structural state:
  - `selected_ops`
  - `theta`
  - `selected_layout`
  - `available_indices`
  - `selection_counts`
  - `phase2_optimizer_memory`
  - `phase3_split_events`
  - `phase3_runtime_split_summary`
  - prune metadata state
- Set:
  - `depth_rollback=True`
  - `structural_rollback=True`
  - `energy_current=energy_prev`
- Record telemetry in the history row:

```py
zero_gain_duplicate_guard = {
  "schema": "zero_gain_duplicate_guard_v1",
  "triggered": true,
  "action": "structural_rollback_and_block_identity",
  "candidate_identity": "...",
  "candidate_key": {...} or null,
  "realized_gain": ...,
  "threshold": ...,
  "prior_committed_duplicate": true,
  "block_scope": "identity",
}
```

#### Future blocking

Before selected-record source choice around the existing `phase2_selected_records` logic, derive blocked identities from prior history rows where `zero_gain_duplicate_guard.triggered` is true.

Filter all candidate source lists used for non-plateau selection:

- `full_records`
- `phase2_shortlisted_records`
- `phase3_shortlisted_records`
- `admission_source_records`

If every candidate source is filtered away, stop cleanly with:

```text
stop_reason = "zero_gain_duplicate_guard_exhausted"
```

Do not use global mutable state; derive block state from the current branch/history so beam or future branch-local flows remain isolated.

### C. Geometry telemetry

Do not change fixed-local geometry behavior for this repair. Preserve:

- `phase3_geometry_window_policy="fixed_local_v1"`
- `w3_wopt_decoupled=True`
- full optimizer active indices under `adapt_reopt_policy="full"`

Update the investigation note to explain that the `"windowed"` value in `phase3_geometry_nested_refit_window` is geometry-scoring telemetry, not optimizer refit policy.

## 4. **File-by-file impact**

### `pipelines/static_adapt/adapt_pipeline.py`

Change:

- Add internal helper(s):
  - canonical candidate identity extraction
  - committed-history identity collection
  - zero-gain duplicate decision
  - candidate-record filtering
  - optional local structural-restore helper to avoid duplicating rollback state restoration
- Apply duplicate filtering before non-plateau selected-record source choice.
- Apply post-optimization duplicate/no-gain structural rollback before history row construction.
- Add `zero_gain_duplicate_guard` and optional filter summary fields to history rows.
- Update final-full-refit policy support from `"windowed"` only to `{"windowed", "full"}`.
- Preserve append-only behavior.

No changes needed in:

- `engine_support.py`
- `nested_windows.py`
- `hh_continuation_scoring.py`
- `hh_continuation_types.py`
- `plateau_acquisition.py`

### `test/test_adapt_vqe_integration.py`

Add focused tests:

1. Full policy final refit:
   - `adapt_reopt_policy="full"`
   - `adapt_final_full_refit=True`
   - assert `final_full_refit.requested is True`
   - assert skip reason is `last_depth_already_full_prefix` when last depth optimized all coordinates.

2. Full policy false flag:
   - `adapt_final_full_refit=False`
   - assert `requested is False`, `executed is False`.

3. Append-only unchanged:
   - assert append-only still does not request final full refit.

4. Duplicate guard pure/helper tests:
   - repeated committed identity + sub-tolerance gain triggers.
   - repeated committed identity + real descent does not trigger.
   - first occurrence + sub-tolerance gain does not trigger.
   - blocked identity is filtered from candidate source records.

5. Keep existing geometry decoupling test unchanged:
   - `test_phase3_geometry_window_decouples_from_full_optimizer_refit`.

### `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_full_reopt_implementation_error_20260710.md`

Update only this diagnostic note.

Add sections documenting:

- Final-refit root cause and repair.
- Duplicate no-gain admission root cause and guard semantics.
- Geometry telemetry explanation.
- Test results.
- Weak–weak smoke result after rerun.

Do not edit manuscript/table/figure files.

## 5. **Risks and migration**

- JSON output changes are additive except `final_full_refit.requested` for `policy="full"`, which is an intentional correctness fix.
- No persistence migration is required.
- The duplicate guard can suppress a later repeat of an identity after it has once produced a no-gain duplicate. This is acceptable for the repair because productive repeats are still allowed until a realized no-gain duplicate is observed.
- If beam selection has a separate candidate-source path, validate that the history-derived filter is applied there too or explicitly document beam as unaffected.

## 6. **Implementation order**

1. Add zero-gain duplicate helper functions in `adapt_pipeline.py`.
2. Add final-full-refit policy support for `"full"` and update skip detection.
3. Add structural-restore helper or reuse existing rollback restoration for duplicate guard.
4. Insert candidate-source filtering before non-plateau selection fallback.
5. Insert post-optimization duplicate/no-gain guard before history row construction.
6. Add/extend tests in `test/test_adapt_vqe_integration.py`.
7. Run targeted tests:
   - full-policy final-refit tests
   - resolver/full active-count tests
   - geometry decoupling test
   - duplicate guard tests
8. Run short weak–weak diagnostic smoke to depth 8–10 with failed settings.
9. Update the diagnostic note with actual test and smoke outcomes.
10. Only then consider rerunning the six-regime less-aggressive matrix.


> 💡 Continue this plan conversation with ask_oracle(chat_id: "full-reopt-repair-5642E4", new_chat: false)