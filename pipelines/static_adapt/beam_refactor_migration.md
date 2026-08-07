# Static ADAPT Beam Refactor Migration Log

Purpose: track behavior-preserving moves out of
`pipelines/static_adapt/adapt_pipeline.py` during the beam refactor. This is an
agent-facing map for later dead-code and archaic-behavior review.

Baseline checkpoint:

- `a972a61` - `checkpoint before static adapt beam refactor`
- Branch: `codex/static-adapt-beam-refactor`
- Fallback branch pushed before refactor: `codex/adapt-generator-geometry-speedups-20260613`

## Current Extracted Module

New module:

- `pipelines/static_adapt/beam_search.py`

Important scope rule: extracted helpers are not deletion decisions. A helper
being moved here means it was still part of the live beam runtime path at the
time of extraction, unless a later note explicitly says otherwise.

## Moved Helpers

| Commit | Helper(s) now in `beam_search.py` | Former role inside `_run_hardcoded_adapt_vqe` |
|---|---|---|
| `e2c7c6a` | `_beam_prune_key_payload_for_policy`, `_beam_sort_key_for_policy`, `_beam_prune_for_policy`, `_beam_dedup_for_policy` | Mode-dependent beam survival, sort, dedup, and audit adapters for legacy versus ordered-batch beam policy. |
| `83905c7` | `_BeamParentRoundPolicy`, `_resolve_beam_parent_round_policy` | Computes parent worker counts, parent-parallel enablement, disabled reason, and per-parent branch worker budget. |
| `8df6b09` | `_beam_round_best_record_value`, `_beam_round_update_best`, `_accumulate_beam_round_frontier_diagnostic` | Accumulates per-round candidate counts, best available scores, and parent stop/expanded reason counts. |
| `0afa326` | `_beam_base_branch_from_parent_scratch` | Copies scratch/evaluation state from a parent branch into the base branch used for stop/admission children. |
| `12c8802` | `_beam_terminal_child_from_scratch` | Creates the terminal/stop child branch from a base branch and branch scratch result. |
| `f87b28b` | `_BeamRoundPruneAuditSummary`, `_beam_round_prune_audit_summary` | Aggregates per-round prune audit payloads, permission-reason counts, and prune child/executed/accepted counters. |
| `3d57dc7` | `_beam_round_stop_reason` | Resolves per-round beam diagnostic stop reason from frontier/proposal availability and parent stop-reason counts. |
| `1ba37a6` | `_beam_round_diagnostics_payload` | Builds the per-round beam diagnostics payload from round counts, parent-parallel state, frontier diagnostics, and prune audit summary. |
| `c362f5a` | `_beam_replay_round_payload` | Builds the per-round beam replay telemetry payload while receiving runtime-local sort, branch-summary, and conversion callbacks from `adapt_pipeline.py`. |
| `b233d33` | `_beam_round_done_log_payload` | Builds the `hardcoded_adapt_beam_round_done` log payload from round counts, parent-parallel state, and frontier diagnostics. |
| `a2f39d6` | `_beam_replay_telemetry_payload` | Builds the beam replay telemetry envelope for current checkpoints, including replay tail, branch summaries, and checkpoint-branch policy. |
| `ecf16f4` | `_beam_branch_replay_summary_payload` | Builds compact per-branch replay summaries for beam current-checkpoint telemetry while receiving the runtime-local prune-key callback from `adapt_pipeline.py`. |
| `4aab9ba` | `_beam_branch_summary_payload` | Builds final per-branch beam diagnostics summaries while receiving generator IDs and the runtime-local prune-key callback from `adapt_pipeline.py`. |
| `this commit` | `_beam_final_diagnostics_payload` | Builds the final beam diagnostics payload and checkpoint-relationship summary while receiving runtime-local sort, fingerprint, prune-key, replay-summary, and branch-summary callbacks from `adapt_pipeline.py`. |

## Still In `adapt_pipeline.py`

The following are intentionally not moved yet:

- `_evaluate_beam_branch`
- `_materialize_beam_child`
- checkpoint payload assembly
- final winner copy-back into the main runner locals
- public CLI and payload compatibility names, including `phase2_*`,
  `phase2_v1`, and `phase3_*`

## Validation Pattern

After each slice so far:

- `python3 -m py_compile pipelines/static_adapt/beam_search.py pipelines/static_adapt/adapt_pipeline.py test/test_adapt_engine_support.py`
- `python3 -m pytest -q test/test_adapt_engine_support.py`
- beam-focused subset from `test/test_adapt_vqe_integration.py`

Current beam-focused subset covers CLI parsing, route-C beam width validation,
finite-angle fallback, shared SPSA seed policy, class-filtered beam behavior,
checkpoint replay telemetry, gradient parallel parity, parent parallel parity,
and generic Phase-III beam formula behavior.
