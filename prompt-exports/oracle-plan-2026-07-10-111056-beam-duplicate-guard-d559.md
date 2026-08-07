## Final Prompt
<taskname="Beam duplicate guard"/>

<task>
Repair the remaining Paper-I HH SNAKE full-window/full-reopt implementation error for beam-enabled runs. The ordinary non-beam duplicate guard and final_full_refit metadata are already present; implement only the smallest additional code change needed so beam branch expansion/selection prevents exact repeated zero-gain child admissions while preserving productive repeats, Route-C plateau behavior, and existing windowed/append-only semantics. Do not update manuscript/PDF artifacts. Add/adjust focused tests and plan a weak--weak validation smoke after the patch.
</task>

<architecture>
- `pipelines/static_adapt/adapt_pipeline.py` contains `_run_hardcoded_adapt_vqe()` and the nested beam execution path. Beam mode is enabled by `beam_policy.beam_enabled`; branch-local helpers `_evaluate_beam_branch()` and `_materialize_beam_child()` live inside this function.
- `pipelines/static_adapt/engine_support.py` defines `_BeamBranchState`, `_BranchExpansionPlan`, `_BranchStepScratch`, and `_BeamParentScratchResult`. `_BeamBranchState.clone_for_child()` deep-copies branch `history`, so duplicate state can be branch-local.
- `pipelines/static_adapt/beam_search.py` is mostly pure policy/telemetry: beam prune/dedup/sort, round diagnostics, branch summaries, and replay payloads. It does not decide admissions.
- `pipelines/static_adapt/batch_ordering.py::_record_candidate_label()` is the fallback label extractor used by `_zero_gain_duplicate_identity()`.
- `pipelines/scaffold/hh_continuation_types.py::CandidateFeatures` carries `runtime_split_child_labels` and `runtime_split_child_generator_ids`; the duplicate identity helper intentionally prefers these child identities over generic parent labels.
- `pipelines/static_adapt/plateau_acquisition.py` is Route-C plateau-specific duplicate machinery. In `adapt_pipeline.py`, Route-C plateau first slice rejects effective beam width > 1, and ordinary duplicate filtering is skipped while `route_c_plateau_trial_active` is true.
</architecture>

<selected_context>
- `pipelines/static_adapt/adapt_pipeline.py` (slices): imports; zero-gain duplicate helpers; `_run_hardcoded_adapt_vqe` options and beam policy serialization; Route-C beam guard; Route-C plateau scoring integration; beam setup/prune/dedup wrappers; `_evaluate_beam_branch()` proposal selection; `_materialize_beam_child()` child insertion, local reopt aftermath, history commit, prune/termination; beam round loop/finalist transfer; ordinary non-beam duplicate filter and post-refit rollback guard for comparison.
- `pipelines/static_adapt/engine_support.py` (full): branch/proposal/scratch dataclasses, branch cloning, reopt active-index policy including `policy_key == "full"`.
- `pipelines/static_adapt/beam_search.py` (full): beam policy adapters, diagnostics, replay/final payloads.
- `pipelines/static_adapt/batch_ordering.py` (full): `_record_candidate_label()` and batch ordering helpers used by proposal records.
- `pipelines/static_adapt/plateau_acquisition.py` (full): Route-C plateau duplicate key/state helpers, to distinguish plateau duplicate policy from ordinary zero-gain duplicate guard.
- `pipelines/scaffold/hh_continuation_types.py` (full): `CandidateFeatures` fields, especially runtime split child identity fields.
- `test/test_static_adapt_full_reopt_duplicate_guard.py` (full): current focused tests for identity/filter helpers and final-full-refit semantics.
- `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_full_reopt_implementation_error_20260710.md` (full): investigation note, failed weak--weak evidence, and checkpoint stating beam settings were active despite no batching.
</selected_context>

<relationships>
- Ordinary path: `_zero_gain_duplicate_blocked_identities(history)` + `_filter_zero_gain_duplicate_records(...)` at `adapt_pipeline.py:17828` filters only identities previously blocked by a triggered guard, and skips Route-C plateau trials.
- Ordinary path: post-refit singleton guard at `adapt_pipeline.py:19919-20002` computes identity from `selected_batch_records_for_history[0]`, compares against `_zero_gain_duplicate_committed_identities(history)`, and rolls back only when `allow_repeats`, non-Route-C, singleton, prior committed duplicate, and `realized_gain <= max(eps_energy, adapt_rollback_tolerance, relative 1e-12)`.
- Productive repeats are preserved by design: ordinary path does not pre-filter all committed identities, only previously blocked zero-gain identities; a committed repeat is rejected only after its realized post-refit gain is known to be non-meaningful.
- Beam proposal path: `_evaluate_beam_branch()` builds `eligible_records_local`, then `beam_selected_records_local` at `adapt_pipeline.py:13187`, then `_BranchExpansionPlan`s. There is currently no branch-local zero-gain duplicate filtering before selected plans are returned.
- Beam commit path: `_materialize_beam_child()` performs local reopt and regression rollback around `adapt_pipeline.py:14418`, then builds `history_row_local` at `14521` and appends it at `15024` (selected slice). It currently lacks the ordinary zero-gain duplicate guard and does not serialize `zero_gain_duplicate_filter` / `zero_gain_duplicate_guard` in beam history rows.
- Beam round loop: `zero_gain_duplicate_filter_payload` / `zero_gain_duplicate_guard_payload` placeholders are initialized at `adapt_pipeline.py:15325`, but discovery found no use before `selected_plans = list(scratch.proposals)` at `15464` or child materialization/frontier selection.
- Beam branch history is local: `base_branch = _beam_base_branch_from_parent_scratch(parent, scratch)` clones parent history; each child appends its own `history_row_local`. Avoid global cross-branch duplicate bans unless explicitly justified.
</relationships>

<edit_points>
- Primary candidate edit: add the ordinary post-refit zero-gain duplicate guard inside `_materialize_beam_child()` after existing beam regression rollback and before `history_row_local` is finalized/appended. Use `base_branch.history` or the child history before appending as the committed-history source, and gate it like ordinary path: `allow_repeats`, singleton only (`len(selected_batch_records_for_history_local) == 1` / `batch_size_local == 1`), not Route-C plateau, not already structurally rolled back, prior committed duplicate identity, realized gain below threshold. On trigger, restore child state from `base_branch`, set `depth_rollback_local` and `structural_rollback_local`, set a `zero_gain_duplicate_guard_payload` with action `structural_rollback_and_block_identity`, and let existing beam structural-rollback handling mark the child terminal.
- Also add beam history telemetry fields mirroring ordinary rows: `zero_gain_duplicate_filter` and `zero_gain_duplicate_guard` in `history_row_local`. The filter payload can initially be inactive unless you also add branch-local blocked-identity filtering before proposal selection.
- Optional/secondary if needed by tests: apply `_zero_gain_duplicate_blocked_identities(branch.history)` in `_evaluate_beam_branch()` after `admission_source_records_local` is formed and before `retained_records_local`/`eligible_records_local`/`beam_selected_records_local`. Keep it branch-local and filter only blocked identities, not all committed duplicates. Avoid adding broad new dataclass plumbing unless telemetry or behavior requires it.
- Do not modify `beam_search.py` for admission behavior unless unavoidable; it only prunes/ranks already-materialized branch states.
</edit_points>

<tests>
- Extend `test/test_static_adapt_full_reopt_duplicate_guard.py` with focused beam coverage. Best low-risk target: a helper-level or small integration test that exercises beam singleton duplicate guard semantics: prior committed same runtime child identity + zero realized gain triggers structural rollback/block payload; same identity with meaningful gain remains allowed.
- Add/verify a beam-enabled smoke path uses `adapt_beam_live_branches > 1`, `adapt_beam_children_per_parent > 1`, and batching disabled (`phase2/phase3 no batching`) so the repaired route is actually beam-enabled.
- Keep existing checks passing: `python3 -m py_compile pipelines/static_adapt/adapt_pipeline.py test/test_static_adapt_full_reopt_duplicate_guard.py` and `pytest -q test/test_static_adapt_full_reopt_duplicate_guard.py -x`.
- After patch/tests, run the requested weak--weak diagnostic smoke to depth 8-10 under the failed settings and verify no exact child dominates repeated zero-gain admissions and `final_full_refit` metadata remains coherent.
</tests>

<ambiguities>
- The raw failed `raw_outputs/.../weak_weak/row_manifest.json` and `result.json` paths were not visible in the loaded RepoPrompt file tree/search results, so they are not selected. The selected investigation note contains the relevant argv/evidence: `--adapt-beam-live-branches 3`, `--adapt-beam-children-per-parent 2`, `--phase2-no-batching`, `--phase3-no-batching`, repeated `paop_full:paop_cloud_p(site=1->phonon=0)::child_set[4]`, and zero `delta_energy` for k>=5.
- The exact smoke command is not selected; derive it from the failed settings/note or existing project runner conventions if available to the next agent.
</ambiguities>

## Selection
- Files: 8 total (7 full, 1 slice)
- Total tokens: 114360 (Auto view)
- Token breakdown: full 46234, slice 68126

### Files
### Selected Files
/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/
├── MATH/
│   └── paper_facing/
│       └── paper_I_static_scaffold/
│           └── paper_i_hh_full_reopt_implementation_error_20260710.md — 2,800 tokens (full)
├── pipelines/
│   ├── scaffold/
│   │   └── hh_continuation_types.py — 7,954 tokens (full)
│   └── static_adapt/
│       ├── adapt_pipeline.py — 68,126 tokens (lines 450-679 (Imports for engine_support beam dataclasses, beam_search policy helpers, batch_ordering record labels/order helpers, plateau acquisition/Route-C modules used by duplicate guard and beam route.), 1400-1569 (Record identity and duplicate-guard helpers: _record_like_mapping, _zero_gain_duplicate_identity, committed/blocked identity collection, record filtering, and final-full-refit helper predicates.), 1620-2010 (_run_hardcoded_adapt_vqe signature and early option normalization, including adapt_reopt_policy/full-refit flags and beam parameters passed into the main static ADAPT run.), 2520-2580 (Beam capacity policy resolution from adapt_beam_live_branches/children_per_parent/terminated_keep.), 3050-3085 (Serialized settings for requested/effective beam policy values in output/config metadata.), 5940-6029 (Route-C plateau configuration payload and explicit guard that Route-C plateau first slice only supports effective beam width 1.), 8340-8629 (Route-C plateau scoring helpers and duplicate-status payloads, showing separate plateau duplicate machinery distinct from normal zero-gain duplicate guard.), 10180-10320 (Beam setup diagnostics plus _beam_prune_current/_beam_dedup_current wrappers used when constructing frontiers/finalists.), 10813-10980 (Beam branch summary/clone/executor helpers and start of _evaluate_beam_branch with branch-local energy, gradients, and state initialization.), 12880-14080 (Beam _evaluate_beam_branch phase2/phase3 shortlist retention, tie-beam/batch handling, eligible record selection, children cap, and _BranchExpansionPlan creation.; End of beam proposal construction, eps-grad scratch returns, final _BranchStepScratch return, and start of _materialize_beam_child.; _materialize_beam_child setup: clone base branch, map plan batch records to selected feature rows/labels/positions, update split telemetry, compute batch score/burden, and resolve local reopt active indices.), 14416-14780 (Beam child after local optimizer: non-improvement rollback, realized energy deltas/drop counters, selected gradients, selector measurement stats, and history_row_local fields through selected_feature_rows/batch/beam metadata.), 14900-15240 (Beam history row admitted-record telemetry, child.history append, cumulative selector/beam costs, committed admission updates, live prune execution, and structural_rollback branch.; Beam post-commit noise-floor/drop/eps-energy/pool-exhausted termination handling before returning child.), 15320-15625 (Beam-enabled outer loop start: duplicate payload placeholders, parent scratch evaluation, base branch construction, selected_plans from scratch.proposals, child materialization, and child_frontier/round_terminals append.; Beam round diagnostics and replay payload construction after child_frontier pruning and terminal fallback.), 15670-15790 (Beam finalists/winner selection and transfer of winner branch state back to main selected_ops/history/runtime variables.), 15800-15900 (Ordinary non-beam loop preamble showing zero_gain payload placeholders and Route-C plateau trial state variables.), 17820-17905 (Ordinary-path pre-admission zero-gain duplicate filter over full/phase2/phase3/admission records, skipped during Route-C plateau trials.), 18960-19040 (Ordinary-path reopt active-index resolution and Route-C plateau active+dormant window override near local optimizer setup.), 19840-20010 (Ordinary-path post-refit rollback and zero-gain duplicate guard: realized gain threshold, prior committed duplicate detection, structural rollback and block action.), 20360-20620 (Ordinary-path history_row serialization including structural_rollback, zero_gain_duplicate_filter, zero_gain_duplicate_guard, selected_feature_rows, and runtime split child identity fields.), 21080-21210 (Ordinary-path post-history commit/prune branch, including structural_rollback handling after duplicate guard and live prune execution suppression.))
│       ├── batch_ordering.py — 5,508 tokens (full)
│       ├── beam_search.py — 9,384 tokens (full)
│       ├── engine_support.py — 11,669 tokens (full)
│       └── plateau_acquisition.py — 7,905 tokens (full)
└── test/
    └── test_static_adapt_full_reopt_duplicate_guard.py — 1,014 tokens (full)


---

## Generated Plan

## Chat Send ✅
- **Chat**: `beam-duplicate-guard-EBA8AF` | **Mode**: plan

### Response
## 1. **Summary**

Implement a targeted beam-path fix in `pipelines/static_adapt/adapt_pipeline.py`: mirror the existing ordinary-path post-refit zero-gain duplicate guard inside `_materialize_beam_child()`. The guard should trigger only for singleton, non-Route-C, `allow_repeats=True` beam admissions whose candidate identity was previously committed on that branch and whose realized post-refit gain is below the same threshold used by the ordinary path. On trigger, structurally restore the child to `base_branch`, serialize guard telemetry in the beam history row, and let existing beam structural-rollback handling mark the child terminal. Do not change final-full-refit logic, Route-C plateau duplicate machinery, beam pruning policy, manuscript files, or PDF artifacts.

---

## 2. **Current-state analysis**

### Relevant ownership and data flow

- `_run_hardcoded_adapt_vqe()` owns both ordinary ADAPT execution and true beam execution.
- Beam mode is enabled through `beam_policy.beam_enabled`, resolved from:
  - `adapt_beam_live_branches`
  - `adapt_beam_children_per_parent`
  - `adapt_beam_terminated_keep`
- Beam branch state is represented by `_BeamBranchState` from `engine_support.py`.
  - Each branch owns its own:
    - `selected_ops`
    - `theta`
    - `history`
    - `available_indices`
    - optimizer memory
    - Phase-I/II/III telemetry state
  - `_BeamBranchState.clone_for_child()` deep-copies `history`, so duplicate-guard state can remain branch-local.

### Existing ordinary duplicate guard

In `adapt_pipeline.py`:

- `_zero_gain_duplicate_identity(record)` builds a canonical identity.
  - It intentionally prioritizes:
    1. `runtime_split_child_labels`
    2. `runtime_split_child_generator_ids`
    3. `candidate_label`
    4. `selected_op`
    5. `selected_logical_op`
    6. `label`
    7. `generator_id`
    8. `_record_candidate_label()`
- `_zero_gain_duplicate_committed_identities(history)` collects identities from non-structural-rollback history rows.
- `_zero_gain_duplicate_blocked_identities(history)` collects identities from prior triggered guard payloads.
- Ordinary path has two protections:
  1. Pre-admission filtering of previously blocked identities.
  2. Post-refit singleton guard:
     - only when `allow_repeats`
     - not during Route-C plateau trial
     - only singleton admission
     - only if same identity was previously committed
     - only if realized gain is below:
       ```text
       max(eps_energy, adapt_rollback_tolerance, 1e-12 * max(1, abs(energy_before)))
       ```
     - action: structural rollback and block identity in telemetry.

This preserves productive repeats because committed duplicate identities are not pre-filtered unless a prior zero-gain guard already triggered.

### Beam path gap

Beam execution currently does:

1. `_evaluate_beam_branch()`:
   - evaluates branch-local candidate scores
   - builds `eligible_records_local`
   - selects `beam_selected_records_local`
   - creates `_BranchExpansionPlan` objects

2. Beam round loop:
   - receives `scratch.proposals`
   - materializes each proposal through `_materialize_beam_child()`

3. `_materialize_beam_child()`:
   - clones `base_branch`
   - inserts selected singleton/batch terms
   - runs local reoptimization
   - applies only regression rollback
   - appends `history_row_local`
   - commits/prunes unless `structural_rollback_local`

Missing today:

- no beam post-refit zero-gain duplicate guard
- no beam history fields:
  - `zero_gain_duplicate_filter`
  - `zero_gain_duplicate_guard`

Therefore a beam branch can repeatedly commit the same runtime-split child with effectively zero realized descent.

### Reusable code

Reuse/extend:

- `_zero_gain_duplicate_identity()`
- `_zero_gain_duplicate_committed_identities()`
- existing ordinary threshold semantics
- existing beam structural-rollback handling in `_materialize_beam_child()`

Do not reuse/modify for this fix:

- `plateau_acquisition.py` Route-C duplicate state: it is plateau-specific.
- `beam_search.py`: it ranks/prunes already-materialized branches and should not decide admissions.
- `engine_support.py`: no new branch dataclass state is required for the minimal fix.

---

## 3. **Design**

### A. Shared duplicate-guard decision helper

Add a small internal helper near the existing zero-gain helpers in `adapt_pipeline.py`.

Purpose: centralize the ordinary-path duplicate decision so beam and ordinary paths use identical identity/threshold semantics.

Suggested shape:

```py
def _zero_gain_duplicate_guard_payload(
    *,
    history_rows,
    selected_record,
    selected_position,
    energy_before,
    energy_after,
    eps_energy,
    rollback_tolerance,
) -> dict[str, Any]:
    ...
```

Contract:

- Pure function; no mutation.
- Uses `_zero_gain_duplicate_identity(selected_record)`.
- Uses `_zero_gain_duplicate_committed_identities(history_rows)`.
- Computes:
  - `realized_gain = energy_before - energy_after`
  - `threshold = max(eps_energy, rollback_tolerance, 1e-12 * max(1.0, abs(energy_before)))`
  - `prior_committed_duplicate = identity in committed_identities`
  - `triggered = prior_committed_duplicate and realized_gain <= threshold`
- Returns payload matching existing schema:
  ```json
  {
    "schema": "zero_gain_duplicate_guard_v1",
    "triggered": false,
    "candidate_identity": "...",
    "candidate_key": {"candidate_identity": "...", "position_id": ..., "key": "..."},
    "realized_gain": 0.0,
    "threshold": 1e-6,
    "prior_committed_duplicate": true,
    "action": "none"
  }
  ```
- If identity or candidate key cannot be built, return `candidate_identity=None`, `candidate_key=None`, `triggered=False`.

Use this helper in the ordinary path too, replacing only the duplicated payload/threshold construction. The ordinary rollback gates and mutations must remain unchanged.

### B. Beam post-refit guard in `_materialize_beam_child()`

Add branch-local guard logic after existing beam regression rollback and before drop/eps-energy counters and `history_row_local` construction.

Required gate:

```text
not structural_rollback_local
and allow_repeats
and not phase3_plateau_cfg.enabled
and batch_size_local == 1
and len(selected_batch_records_for_history_local) == 1
```

Use:

- history source: `base_branch.history`
- selected record: `selected_batch_records_for_history_local[0]`
- selected position: `selected_position_local`
- energy before: `energy_prev_local`
- energy after: `child.energy_current`

On `triggered=True`:

1. Restore child structural state from `base_branch`, matching the existing beam structural rollback fields:
   - `child.selected_ops`
   - `child.theta`
   - `child.available_indices`
   - `child.selection_counts`
   - `child.phase2_optimizer_memory`
   - `child.phase3_split_events`
   - `child.phase3_runtime_split_summary`
   - `child.phase1_prune_metadata`
   - `child.phase1_prune_first_seen_steps`

2. Set:
   - `child.energy_current = energy_prev_local`
   - `depth_rollback_local = True`
   - `structural_rollback_local = True`
   - `zero_gain_duplicate_guard_payload_local["action"] = "structural_rollback_and_block_identity"`

3. Emit a beam-specific log event:
   - event name: `hardcoded_adapt_beam_zero_gain_duplicate_rollback`
   - include:
     - `depth`
     - `branch_id`
     - `parent_branch_id`
     - `candidate_identity`
     - `candidate_key`
     - `realized_gain`
     - `threshold`

4. Do not add candidate-specific global state.
   - The guard remains branch-local.
   - The structural-rollback child becomes terminal through existing beam code.
   - Productive siblings remain eligible.

### C. Beam history telemetry

In `_materialize_beam_child()`, initialize local inactive payloads:

```json
zero_gain_duplicate_filter = {
  "schema": "zero_gain_duplicate_filter_v1",
  "active": false,
  "blocked_identity_count": 0,
  "removed_count": 0,
  "removed_identities": []
}
```

```json
zero_gain_duplicate_guard = {
  "schema": "zero_gain_duplicate_guard_v1",
  "triggered": false,
  "candidate_identity": null,
  "candidate_key": null,
  "realized_gain": null,
  "threshold": null,
  "prior_committed_duplicate": false,
  "action": "none"
}
```

Add both fields to `history_row_local`, mirroring ordinary history rows:

```py
"zero_gain_duplicate_filter": dict(zero_gain_duplicate_filter_payload_local),
"zero_gain_duplicate_guard": dict(zero_gain_duplicate_guard_payload_local),
```

Do not implement beam pre-proposal filtering in this patch. It would require extra scratch telemetry plumbing and is not necessary to prevent committed zero-gain duplicate admissions.

### D. Route-C and batching behavior

- Skip the beam duplicate guard whenever `phase3_plateau_cfg.enabled` is true.
- Skip batch admissions with `batch_size_local > 1`.
- Keep Route-C plateau duplicate handling isolated in `plateau_acquisition.py`.
- Do not alter the existing guard that rejects Route-C plateau with effective beam width greater than 1.

### E. Concurrency and lifecycle

- `_evaluate_beam_branch()` may run in a parent `ThreadPoolExecutor`.
- `_materialize_beam_child()` runs after scratch results are merged, in the serial child-materialization loop.
- The new beam guard runs only inside `_materialize_beam_child()`, so it introduces no shared mutable cross-thread state.
- Branch history remains branch-local through `_BeamBranchState.clone_for_child()`.

---

## 4. **File-by-file impact**

### `pipelines/static_adapt/adapt_pipeline.py`

Modify only this production file.

Changes:

1. Add `_zero_gain_duplicate_guard_payload(...)` near existing zero-gain helpers.
   - Reason: reuse identical ordinary/beam semantics.
   - Depends on existing:
     - `_zero_gain_duplicate_identity`
     - `_zero_gain_duplicate_committed_identities`
     - `PlateauCandidateKey`

2. Replace ordinary-path inline guard payload construction with the helper.
   - Reason: prevent divergence.
   - Must preserve existing gates, rollback mutation, and log event.

3. Inside `_materialize_beam_child()`:
   - Add local inactive filter/guard payloads.
   - Optionally factor existing base-branch restoration into a small nested helper to avoid duplicating rollback field assignments.
   - Add post-refit zero-gain duplicate guard after regression rollback.
   - Add `zero_gain_duplicate_filter` and `zero_gain_duplicate_guard` fields to `history_row_local`.

Do not modify:

- final full-refit logic
- `_evaluate_beam_branch()` proposal selection
- beam prune/dedup policy
- Route-C plateau scoring/duplicate helpers

### `test/test_static_adapt_full_reopt_duplicate_guard.py`

Add focused tests:

1. Helper decision test: prior committed runtime child + zero gain triggers.
   - History row should include `runtime_split_child_labels`.
   - Selected record uses the same child label.
   - Assert:
     - `triggered is True`
     - `prior_committed_duplicate is True`
     - `candidate_identity == "runtime_split_child:<child>"`
     - `action == "none"` before caller rollback mutation.

2. Helper decision test: prior committed runtime child + meaningful gain does not trigger.
   - Assert productive repeat is allowed.

3. Beam telemetry smoke.
   - Run `_run_hardcoded_adapt_vqe()` with:
     - `adapt_beam_live_branches=2`
     - `adapt_beam_children_per_parent=2`
     - `phase2_enable_batching=False`
     - `allow_repeats=True`
     - small existing Hubbard fixture similar to current tests
   - Assert at least one beam history row contains:
     - `beam_branch_id`
     - `zero_gain_duplicate_filter`
     - `zero_gain_duplicate_guard`

Existing tests must remain unchanged and passing.

### Files intentionally not changed

- `pipelines/static_adapt/beam_search.py`
- `pipelines/static_adapt/engine_support.py`
- `pipelines/static_adapt/batch_ordering.py`
- `pipelines/static_adapt/plateau_acquisition.py`
- `pipelines/scaffold/hh_continuation_types.py`
- `MATH/.../paper_i_hh_full_reopt_implementation_error_20260710.md`
- any manuscript/PDF artifacts

---

## 5. **Risks and migration**

- This is an additive internal behavior/telemetry change; no persisted schema migration is required.
- Beam runs may terminate a branch earlier when all children are zero-gain duplicates. This is intended.
- Productive repeats are preserved because the guard is post-refit and threshold-based.
- The only new output fields are additive beam history telemetry fields.
- If the weak--weak smoke shows repeated terminal rollbacks consuming all children, consider a later branch-local blocked-identity prefilter, but do not include that in this minimal patch.

---

## 6. **Implementation order**

1. Add `_zero_gain_duplicate_guard_payload(...)` in `adapt_pipeline.py`.
2. Update ordinary non-beam guard payload construction to call the helper without changing gates or rollback behavior.
3. In `_materialize_beam_child()`:
   - add local inactive filter/guard payloads
   - add or reuse a local restore-to-base helper
   - add the post-refit duplicate guard
   - serialize guard/filter payloads in `history_row_local`
4. Add tests in `test/test_static_adapt_full_reopt_duplicate_guard.py`.
5. Run:
   ```bash
   python3 -m py_compile pipelines/static_adapt/adapt_pipeline.py test/test_static_adapt_full_reopt_duplicate_guard.py
   pytest -q test/test_static_adapt_full_reopt_duplicate_guard.py -x
   ```
6. After tests pass, run weak--weak diagnostic smoke to depth 8–10 using the failed settings:
   - physical operator lane
   - no batching
   - `adapt_reopt_policy=full`
   - full insertion/window settings
   - `allow_repeats=true`
   - `adapt_beam_live_branches=3`
   - `adapt_beam_children_per_parent=2`
7. Validate smoke:
   - no repeated committed zero-gain `paop_cloud_p(... )::child_set[4]` dominance
   - triggered duplicate rows, if present, have `structural_rollback=true`
   - final operators do not accumulate rolled-back duplicate children
   - `final_full_refit` metadata remains coherent.


> 💡 Continue this plan conversation with ask_oracle(chat_id: "beam-duplicate-guard-EBA8AF", new_chat: false)