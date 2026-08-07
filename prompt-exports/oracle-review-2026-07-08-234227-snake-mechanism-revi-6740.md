# Oracle Review

## Summary

Based on the described follow-up changes, the implementation now looks conservative in the intended accounting sense: coarse `S_alg` remains delegated to `snake_algorithmic_work_from_payload`, diagnostic candidate/shortlist/retained counters remain formula operands rather than promoted work, beam terminal accounting uses winner-lineage controller history, prefix scopes are reconstructed from prefix rows, by-phase-only summaries remain mechanism-unclassified/partial, and Phase-II metric substructure is explicitly deferred rather than ratio-allocated. I do not see remaining semantic blockers to reporting this as a conservative reconstructor scaffold with full Gram/novelty vs Schur formula reconstruction deferred.

## Findings

### P2

1. **Add one explicit regression for final-prefix non-terminal delegation**  
   **File:** `test/test_snake_table_i_measurement_work.py`  
   You fixed `_mechanism_scoped_controller_summary` so final-prefix scopes always use `controller_proxy_from_history_rows(prefix_rows)`, but the listed new tests do not explicitly mention a regression where terminal summary differs from prefix history at `history_position == len(history)`.  
   **Suggestion:** Add a small test asserting mechanism work follows prefix-row reconstruction even when a terminal controller summary is present and would produce different mechanism bins.

2. **Clarify scaffold status in emitted metadata or caller-facing contract**  
   **File:** `pipelines/exact_bench/snake_table_i_measurement_work.py`  
   Since Phase-II Gram/novelty vs Schur reconstruction is intentionally deferred, downstream users could overinterpret mechanism bins as a complete formula-level decomposition.  
   **Suggestion:** Ensure the returned payload clearly carries a flag such as `requires_formula_reconstruction` / `partial_mechanism_reconstruction` for affected metric bins and that tests assert this flag for Phase-II rerank cases.

## Verdict

No remaining P0/P1 accounting-semantics blockers found from the described changes. This is reasonable to report as a conservative mechanism-resolved reconstructor scaffold, provided the deferred formula reconstruction is stated explicitly.