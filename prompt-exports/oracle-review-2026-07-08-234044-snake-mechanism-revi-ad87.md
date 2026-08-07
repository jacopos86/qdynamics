## Final Prompt
<taskname="Snake mechanism review"/>

<task>
Review the current uncommitted implementation of the Paper-I SNAKE mechanism-resolved algorithmic work reconstructor. Focus only on semantic/accounting correctness and test coverage. Do not suggest PDF/report-builder changes. Verify whether the new helper preserves these constraints: existing snake_algorithmic_work_from_payload remains authoritative; candidate_count_total/records_evaluated/shortlist/retained counts are diagnostics/formula operands only and never promoted directly to S_alg; beam terminal rows use winner-lineage work and keep aggregate beam search work as provenance only; by_phase-only payloads remain mechanism-unclassified; Phase-II metric sub-split is not ratio-allocated and is marked as requiring formula reconstruction.

Context: Files changed are pipelines/exact_bench/snake_table_i_measurement_work.py and test/test_snake_table_i_measurement_work.py. Focus on the newly added SNAKE_MECHANISM_RESOLVED_WORK_SCHEMA_VERSION, snake_mechanism_resolved_work_from_payload, helper functions around controller scope/event parsing, and the new tests. Existing focused tests passed: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_snake_table_i_measurement_work.py -q and PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_static_adapt_controller_measurement_work.py -q.
</task>

<architecture>
- pipelines/exact_bench/snake_table_i_measurement_work.py is the reporting-only SNAKE Table-I measurement-work normalizer. It contains both the pre-existing authoritative S_alg reconstructor and the new mechanism-resolved helper.
- Authoritative S_alg path: snake_algorithmic_work_from_payload -> _scoped_runtime_s_norm_components_from_payload -> terminal/prefix/beam reconstruction -> _runtime_reconstruction_event_ledger -> algorithmic_measurement_work_from_row. This path promotes only validated typed runtime/event-ledger components and blocks missing candidate-work ledgers.
- New mechanism path: snake_mechanism_resolved_work_from_payload calls snake_algorithmic_work_from_payload first, copies coarse S_alg fields into coarse_S_alg, then partitions typed operator-probe counts into mechanism bins. It should reconcile back to the authoritative coarse S_alg and never use diagnostic candidate/group counters as S_alg.
- Controller scope/event parsing lives in new helpers _normalize_controller_phase_name, _parse_controller_work_scope, _mechanism_chargeable_operator_probe_count, _mechanism_event_records_from_controller_summary, and _mechanism_scoped_controller_summary.
- pipelines/static_adapt/selector_measurement_proxy.py supplies controller_proxy_from_history_rows plus candidate-ledger/by_phase/by_scope aggregation used by prefix scopes and beam winner-lineage terminal rows.
- pipelines/exact_bench/generic_static_metric_enrichment.py is included as targeted slices for schema constants, _num, strict numeric parsing, _algorithmic_work_from_event_ledger, and algorithmic_measurement_work_from_row, which is the generic final S_alg ledger promotion helper.
</architecture>

<selected_context>
_git_data/repos/holstein-test-fullclone-c2b04285/2026-07-08/2334/diff/per-file/pipelines__exact_bench__snake_table_i_measurement_work.py.patch: focused diff for the new schema constant and mechanism-resolved helper block added after snake_algorithmic_work_from_payload.
_git_data/repos/holstein-test-fullclone-c2b04285/2026-07-08/2334/diff/per-file/test__test_snake_table_i_measurement_work.py.patch: focused diff for new import, _add_by_scope fixture helper, and new mechanism tests.
pipelines/exact_bench/snake_table_i_measurement_work.py: full implementation; includes existing runtime S_alg reconstruction, beam winner-lineage logic, prefix scope reconstruction, event ledger creation, fair work, normalization/enrichment, and the new mechanism-resolved helper.
test/test_snake_table_i_measurement_work.py: full focused test file; includes fixtures for native runtime, prefix runtime, beam aggregate runtime, the new mechanism tests, and existing S_alg/fair/deterministic regression tests.
pipelines/static_adapt/selector_measurement_proxy.py: full controller measurement proxy helper; includes controller summary finalization, by_phase/by_scope and candidate ledger aggregation, numeric validation, and controller_proxy_from_history_rows.
test/test_static_adapt_controller_measurement_work.py: full supporting controller tests for event recording, actual/common operator-probe fields, candidate counters, and work-scope behavior.
pipelines/exact_bench/generic_static_metric_enrichment.py: slices only; includes imports/schema constants/aliases, _num, _strict_first_num, _algorithmic_work_from_event_ledger, and algorithmic_measurement_work_from_row.
</selected_context>

<relationships>
- snake_mechanism_resolved_work_from_payload -> snake_algorithmic_work_from_payload: coarse S_alg is computed first and should remain authoritative.
- snake_algorithmic_work_from_payload -> _runtime_reconstruction_event_ledger -> algorithmic_measurement_work_from_row: validates typed component totals and emits S_alg/S_alg_N_* fields.
- _mechanism_scoped_controller_summary mirrors _scoped_runtime_s_norm_components_from_payload scope handling: terminal direct summary; beam aggregate terminal summary replaced by controller_proxy_from_history_rows(winner_history); prefix scopes use controller_proxy_from_history_rows(prefix_rows).
- _mechanism_event_records_from_controller_summary event source priority: events first, then by_scope, then by_phase. by_scope keys are parsed by _parse_controller_work_scope into phase/event fields; by_phase rows get event_kind=None and should be mechanism-unclassified/partial.
- _mechanism_chargeable_operator_probe_count accepts only actual_operator_probe_count/actual_operator_probe_count_total with OPERATOR_PROBE_CHARGE_BASIS; candidate_count_total/evaluated_count_total/records_evaluated/shortlist_size_total/retained_count_total appear in candidate_exposure/formula_operands only.
- _mechanism_classify_event maps phase1 append/insertion to gradient bins, phase2_rerank_records to metric.phase2_rerank_unclassified with requires_formula_reconstruction, phase3_reduced_geometry_rerank and batch_union_scoring to metric bins, and unknown phase/event to unclassified bins.
- New tests exercise terminal by_scope reconciliation, by_phase-only partial/unclassified behavior, beam terminal winner-lineage mechanism work with aggregate total as provenance, and display_prefix history-row scoping.
</relationships>

<ambiguities>
- Factual review point: _mechanism_event_records_from_controller_summary prefers summary["events"] over summary["by_scope"] when both exist. Verify whether real native controller summaries can contain both and whether events carry enough mechanism fields for the intended mechanism resolution.
- Test-coverage point to assess: new tests cover happy-path by_scope, by_phase-only, beam terminal, and prefix scope. They do not appear to directly test invalid/missing actual_operator_probe_count inside mechanism event records, charge-basis mismatch, events-vs-by_scope precedence, or a by_scope event whose diagnostic candidate counters conflict with actual_operator_probe_count.
- Do not review or propose changes in PDF/report-builder files; the selected test imports a report-builder constant only for an existing cache-version test.
</ambiguities>

## Selection
- Files: 7 total (6 full, 1 slice)
- Total tokens: 88765 (Auto view)
- Token breakdown: full 82295, slice 6470

### Files
### Selected Files
/Users/jakestrobel/Library/Application Support/RepoPrompt/Workspaces/Workspace-Holstein local active-352B06F4-8772-4E37-BD57-71C363AA7073/_git_data/
└── repos/
    └── holstein-test-fullclone-c2b04285/
        └── 2026-07-08/
            └── 2334/
                └── diff/
                    └── per-file/
                        ├── pipelines__exact_bench__snake_table_i_measurement_work.py.patch — 5,243 tokens (full)
                        └── test__test_snake_table_i_measurement_work.py.patch — 1,986 tokens (full)
/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/
├── pipelines/
│   ├── exact_bench/
│   │   ├── generic_static_metric_enrichment.py — 6,470 tokens (lines 1-170 (Module imports and Table-I measurement-work schema/constants used by snake_table_i_measurement_work, including ALGORITHMIC_MEASUREMENT_WORK_SCHEMA and TABLE_I_EVENT_LEDGER_SCHEMA plus S_alg aliases.), 283-296 (_num numeric parser imported by snake_table_i_measurement_work and used by finite/nonnegative helpers.), 606-1010 (Strict numeric parsing, S_alg explicit/event-ledger component extraction, and algorithmic_measurement_work_from_row; this is the authoritative final S_alg promotion path called by snake_algorithmic_work_from_payload.))
│   │   └── snake_table_i_measurement_work.py — 35,748 tokens (full)
│   └── static_adapt/
│       └── selector_measurement_proxy.py — 19,324 tokens (full)
└── test/
    ├── test_snake_table_i_measurement_work.py — 17,821 tokens (full)
    └── test_static_adapt_controller_measurement_work.py — 2,173 tokens (full)


---

## Code Review

## Chat Send ✅
- **Chat**: `snake-mechanism-review-C64B7F` | **Mode**: review

### Response
## Summary

The change adds a reporting-only, mechanism-resolved SNAKE work helper that first obtains authoritative coarse `S_alg` via `snake_algorithmic_work_from_payload`, then partitions validated typed operator-probe counts into mechanism bins. The tested happy paths preserve the intended constraints: candidate/shortlist counters are not promoted to `S_alg`, beam terminal rows use winner-lineage work, aggregate beam search work is provenance only, by-phase-only direct fixtures remain unclassified, and Phase-II metric work is not ratio-allocated. I found a couple of semantic edge cases and coverage gaps.

## Findings

### P1

1. **Empty `events` / `by_scope` can shadow valid `by_phase` data**  
   **File:** `pipelines/exact_bench/snake_table_i_measurement_work.py::_mechanism_event_records_from_controller_summary`  
   The helper returns immediately when `summary["events"]` is a sequence or `summary["by_scope"]` is a mapping, even if they contain zero usable event records. Finalized controller summaries commonly carry `by_scope: {}` from `_empty_controller_summary()` / `controller_proxy_from_history_rows()`. In that case, a genuine by-phase-only prefix or beam-winner summary will be treated as empty `by_scope`, leaving gradient/metric bins at zero and causing `invalid_reconciliation` instead of the intended partial/unclassified mechanism result.  
   **Suggestion:** Only select `events` or `by_scope` if at least one valid mapping event is recovered; otherwise fall through to `by_phase`. Add a regression with a finalized/prefix summary containing `by_scope: {}` and populated `by_phase`.

2. **Final-prefix mechanism scope can use terminal summary even when coarse `S_alg` used prefix rows**  
   **File:** `pipelines/exact_bench/snake_table_i_measurement_work.py::_mechanism_scoped_controller_summary`  
   For prefix scopes where `history_position == len(history)`, the mechanism helper unconditionally delegates to terminal summary reconstruction. The authoritative scoped `S_alg` path only uses terminal reconstruction if it succeeds; otherwise it falls back to prefix history rows. This can make mechanism partitioning use a terminal summary that the authoritative path rejected, while `coarse_S_alg` came from valid prefix rows.  
   **Suggestion:** Mirror `_scoped_runtime_s_norm_components_from_payload`: either always build prefix mechanism summaries from `controller_proxy_from_history_rows(prefix_rows)`, or only use terminal equivalence after validating that terminal reconstruction succeeded and record that equivalence in metadata.

### P2

1. **Raw `events` formula operands omit real per-event candidate aliases**  
   **File:** `pipelines/exact_bench/snake_table_i_measurement_work.py::snake_mechanism_resolved_work_from_payload`  
   When raw controller `events` are used, `formula_operands["event_records"]` reads fields like `candidate_count_total`, `shortlist_size_total`, and `retained_count_total`. But `ControllerMeasurementWorkAccumulator.record_event()` emits singular fields such as `candidate_count`, `evaluated_count`, `shortlist_size`, `retained_count`, and `rejected_count`. As a result, diagnostics/formula operands can be `None` for real event payloads even though the data exists.  
   **Suggestion:** Populate operands from both singular and `_total` aliases, while continuing to compute mechanism bins only from `actual_operator_probe_count`.

2. **Add negative/guardrail tests for the new mechanism path**  
   **File:** `test/test_snake_table_i_measurement_work.py`  
   The new tests cover important happy paths, but not several safety gates: missing `actual_operator_probe_count` with positive diagnostic counters, charge-basis mismatch, empty `by_scope` fallback to `by_phase`, raw `events` vs `by_scope` precedence, and conflicting candidate counters versus actual probe counts.  
   **Suggestion:** Add focused regressions asserting these cases block or remain diagnostic-only without changing authoritative `S_alg`.


> 💡 Continue this review conversation with ask_oracle(chat_id: "snake-mechanism-review-C64B7F", new_chat: false)