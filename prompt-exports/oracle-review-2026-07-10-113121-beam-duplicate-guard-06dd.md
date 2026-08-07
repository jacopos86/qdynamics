# Oracle Review

## Summary

The repair appears narrow enough to proceed: behavior remains concentrated in `adapt_pipeline.py` around duplicate identity/admission handling and final-refit metadata, with focused tests and a direct weak–weak rerun reproducing the failed beam/no-batch/full-reopt setting. The validation evidence is strong for the reported failure mode: the pathological repeated child dropped from 24 copies to one, the run terminates via the new duplicate-guard exhaustion path, and `final_full_refit` metadata is coherent.

## Findings

### P2

- `test/test_static_adapt_full_reopt_duplicate_guard.py` — **Beam exhaustion should have a focused regression assertion if it does not already.**  
  The validation run exercised `stop_reason=zero_gain_duplicate_guard_exhausted`, but this is important enough to lock down directly.  
  **Suggestion:** add/ensure one test asserts that when branch-local blocked-identity filtering removes all admissible records, beam expansion returns `zero_gain_duplicate_guard_exhausted` and does **not** fall back to `simple_v1` or materialize rollback-only children.

- `test/test_static_adapt_full_reopt_duplicate_guard.py` — **Route-C/plateau exclusion should be explicitly protected.**  
  Since beam filtering was added before proposal creation, the key invariant is that it remains branch-local and does not interfere with Route-C plateau duplicate machinery.  
  **Suggestion:** if not already covered, add a minimal assertion that duplicate filtering/guarding is inactive during Route-C plateau trial handling, or at least verify the code gate is explicit.

- `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_full_reopt_implementation_error_20260710.md` — **Scope hygiene.**  
  Updating the investigation note is fine if it is purely diagnostic provenance, but the original repair request said not to update manuscript/PDF artifacts.  
  **Suggestion:** keep this change only if it is an internal implementation-error note; avoid carrying any manuscript-facing prose/PDF updates into the rerun commit.

## Proceed decision

I would proceed to the six-regime rerun after the two quick test/invariant checks above are confirmed. I do **not** see a need for further production-code changes before rerunning the matrix. For the rerun, gate acceptance on: no repeated exact zero-gain child dominance, expected branch-local `zero_gain_duplicate_guard`/`filter` telemetry, sane `zero_gain_duplicate_guard_exhausted` stops, and coherent `final_full_refit` metadata across all regimes.