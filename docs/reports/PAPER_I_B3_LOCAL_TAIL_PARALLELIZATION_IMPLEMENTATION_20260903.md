# Paper-I B3 local tail: ledger and checkpoint serialization implementation (2026-09-03)

Author: Claude (Fable 5.1), executing
`chtc/paper_i_ra_adapt_repair_20260727/ra_adapt_costexp_depthfix_sixregime_20260820_v2_chtc/HANDOFF_LOCAL_TAIL_STREAMING_PARALLELIZATION_20260903.md`
after the audit recorded in `agent_guidance/shared/paper-i-refactor-worklog.md`
("2026-09-03 — B3 strong-sector tail: compile-parallelism audit").

Branch `claude/paper-i-b3-local-tail-20260903` in worktree
`Holstein_test.worktrees/claude-paper-i-b3-local-tail`, based on
`f041a5c1bd6e06a1885f5f83d5494c180ca709ec` (= `origin/paper-i-integration-20260831`
at handoff time). Not pushed. **No scientific continuation ran, no CHTC job was
touched, and neither checkpoint archive was downloaded.** All Python ran under
`pipelines/shell/ram_guard.py` (peak 588 MB across every run).

## Retargeting, stated up front

The handoff's performance hypothesis (hours in full-trial Qiskit compile) was
disproved before implementation began. On the sealed source the live jobs
execute, Phase-I transpiles are serial and cost 57–193 transpiles per round at
0.02–0.08 s each (≤ 10 s per round). The 23.6 → 72.8 min pre-Phase-I block per
round is checkpoint publication (full ledger and full history JSON-encoded
3–5× by the pure-Python encoder), ≥ 11 full ledger re-materializations per
round, and the 6508 × reps(k) Phase-0 sweep. The increments below therefore
target serialization and bookkeeping; Increment 2 as written (parallel
compile) and Increment 3 (surrogate screen) were not implemented because they
cannot remove more than ~10 s per round.

Two handoff corrections carried into the code:

- `hardcoded_adapt_phase2_full_feature_parallel` is a deferred observation
  flushed at the accepted transition, so its timestamp is the transition start,
  not Phase-II completion.
- Continuation jobs do not stream the ledger: `EstimatorCallLedger.from_payload`
  constructs `cls()` without `occurrence_stream_path`, so
  `STATIC_ADAPT_ESTIMATOR_LEDGER_STREAM_DIR` is not honored after resume.

## Increment 0 — baseline and phase/RSS instrumentation

Status: done
Commands run:

```bash
cd /Users/lordskyler/Holstein_test.worktrees/claude-paper-i-b3-local-tail
/Users/lordskyler/Holstein_test/.venv/bin/python /Users/lordskyler/Holstein_test/pipelines/shell/ram_guard.py --limit-mb 8000 -- \
  /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q -p no:cacheprovider \
  test/test_ra_adapt_exact_geometry_backend.py test/test_static_adapt_resume_scaffold.py \
  test/test_static_adapt_occurrence_stable_checkpoint.py test/test_static_adapt_accepted_refit_external_gram.py
```

Result: baseline on `f041a5c1` = **74 passed, 9 skipped** (the handoff's 68/9 was
measured on the `5d6d23a8` plot-refresh base, not this start SHA). Five
stdout-only `AI_LOG` events were added, emitted through `_emit_round_stage_timing`
(never raises; every field guarded by `_telemetry_int`; RSS sampled from
`/proc/self/status` with a `getrusage` fallback):

| Event | Emitted from | Fields |
|---|---|---|
| `hardcoded_adapt_checkpoint_publication_timing` | `_write_current_checkpoint` after publish | stage_depth, elapsed_s, occurrence_count, unique_primitive_count, history_rows, forced, rss |
| `hardcoded_adapt_phase0_position_record_timing` | `run_position_record_phase0` | stage_depth, population_count, occurrence_start, occurrence_count, elapsed_s, rss |
| `hardcoded_adapt_phase1_position_loop_timing` | `_evaluate_default_phase1_positions` after the position loop | stage_depth, record_count, elapsed_s, rss |
| `hardcoded_adapt_accepted_refit_timing` | transition transaction around the Powell refit | stage_depth, parameter_count, elapsed_s, rss |
| `hardcoded_adapt_ledger_materialization` | any occurrence-row materialization ≥ 0.05 s | start, end, row_count, elapsed_s, rss |

The field is `stage_depth`, not `depth`, because
`runtime_heartbeat.normalize_ai_log_progress` merges `depth` from any event
and would have clobbered live progress.

Verification: pristine `f041a5c1` vs this tree on three bounded session-kernel
fixtures (singleton plateau, macro append, position-record natural terminal;
RAM and streaming ledgers; 11 checkpoint publications, 8 accepted refits):
all 45 scientific payload sections identical by canonical sha256; the three
full ledgers (fingerprints included) and the three streaming occurrence
files byte-identical; every ledger sidecar identical after `generated_utc`;
the pre-existing `AI_LOG` stream identical (135 lines) plus exactly 21 new
telemetry lines. The only attributable payload differences are 15
source-provenance digests that hash `adapt_pipeline.py` itself
(`controller_instrumentation_sha256`, and for semantic-materialized protocols
`source_implementation_inventory_sha256` → `protocol_sha256`); `route_sha256`
and `problem_scientific_content_sha256` are unchanged.

Deviations: `ram_guard.py` does not exist on `f041a5c1`, so the main checkout's
copy was used by absolute path. `hardcoded_adapt_phase1_position_loop_timing`
is not exercised by the fixtures (the method returns before the loop unless the
insertion domain is expanded); it fires on `full_commutation_reduced` routes
such as the six-regime arms.

Commit: `d893a6b0` (shared with Increment 2 below).

## Increment 1 — streaming resume hydration

Status: skipped
Reason: the per-round re-materialization cost (Increment 2 below) was the
larger and safer lever; hydration remains the blocker for the local runner.
Finding recorded for it: `from_payload` builds an in-RAM ledger regardless of
`STATIC_ADAPT_ESTIMATOR_LEDGER_STREAM_DIR`, so a resumed job carries every
occurrence in RAM beside the O(U) `_records` dict.

## Increment 2 — retargeted: ledger bookkeeping and checkpoint serialization

Status: done (steps 1 and 2 of the four-step plan agreed in chat; steps 3 and 4
listed under "Remaining")

What changed:

1. `EstimatorCallLedger.occurrence_count()` (O(1)) and
   `occurrence_rows(start, end)` (only the window materialized; streaming mode
   skips lines outside the window without parsing). Every session-kernel and
   transaction call site that used `len(ledger_occurrences())` or
   `ledger_occurrences()[a:b]` now uses them (kernel: occurrence_start/after
   phase0/phase1/phase2/end; transaction: gradient and position Phase-0
   slices; workspace and transition prefix identities; transition-after rows).
   `_default_no_prune_ledger_occurrences` itself now walks only the occurrence
   projection instead of `to_payload()` (entries, fingerprint, summaries were
   discarded by every caller). Ledger-like test doubles that expose only
   `to_payload` keep the historical path (duck-typed fallbacks).
2. `_write_current_checkpoint` reuses `to_payload()["summary"]` and
   `["occurrence_summary"]` instead of walking the ledger twice more.
3. `pipelines/static_adapt/json_streaming.py`: chunked C-encoder output that
   is byte-identical to `json.dumps(sort_keys=True, separators=(",", ":"))`
   and to `json.dumps(sort_keys=True, indent=2)`; mappings are walked key by
   key (documents are never handed to the encoder whole), list children are
   batched into adaptive ~2 MB slices, oversized children are recursed.
4. `current_checkpoint._stable_json_digest` hashes the chunked canonical
   encoding (byte-identical digest); content-addressed sidecars hash the text
   as written instead of re-reading the file; opt-in
   `STATIC_ADAPT_CHECKPOINT_JSON_LAYOUT=compact` writes sorted compact
   documents through the C encoder (default `pretty` keeps the historical
   `indent=2` bytes).
5. Phase-II fork children reset the new counter with their scratch buffer.

Measured on this M4 (ledger-shaped payload: 300k occurrences, 100k entries,
6 history rows with 20k-row populations; 136 MB compact / 182 MB pretty):

| Operation | Before | After |
|---|---|---|
| `_stable_json_digest` (projection digest, per checkpoint ×2) | 2.26 s | 0.44 s |
| Sidecar write, default pretty layout | 3.54 s | 3.54 s (unchanged bytes) |
| Sidecar write, `compact` layout (opt-in) | n/a | 0.44 s, −25 % bytes |
| Sidecar sha256 | second full read | inline |
| Per-round `to_payload()` materializations in the pre-Phase-I window | 6 | 0 (three windowed reads, one O(1) count each) |

Commands run and results:

```bash
# focused + ledger + encoder + new tests
... -m pytest -q test/test_ra_adapt_exact_geometry_backend.py test/test_static_adapt_resume_scaffold.py \
  test/test_static_adapt_occurrence_stable_checkpoint.py test/test_static_adapt_accepted_refit_external_gram.py \
  test/test_estimator_call_ledger.py test/test_estimator_ledger_streaming.py test/test_static_adapt_estimator_call_ledger.py \
  test/test_static_adapt_current_checkpoint_owner.py test/test_paper_i_s_alg_accounting.py test/test_json_streaming.py \
  test/test_estimator_ledger_occurrence_window.py
#   -> 208 passed, 9 skipped
# checkpoint / resume suites after the writer change
... -m pytest -q test/test_current_checkpoint_json_layout.py test/test_json_streaming.py \
  test/test_static_adapt_current_checkpoint_owner.py test/test_static_adapt_occurrence_stable_checkpoint.py \
  test/test_static_adapt_resume_scaffold.py test/test_paper_i_sr_snake_resume_adapter.py \
  test/test_estimator_ledger_occurrence_window.py
#   -> 151 passed, 9 skipped
# route suites (11 files) on this tree vs a pristine f041a5c1 export
#   this tree: 22 failed, 223 passed; pristine: 28 failed, 217 passed
#   the 22 are a subset of the pristine 28; the other 6 are canary tests that
#   fail only in the export (missing chtc/ files). No new failures.
```

Verification: parity harness (scratchpad `parity/drive.py`, `compare.py`) on
the final tree reproduces exactly the difference set of the audit run: 36
run-to-run noise keys (output path, `generated_utc`, and the digest cascade
over them; identical in a base-vs-base control) plus the 15 source-provenance
digests. Adversarial review (opus, 6 findings) produced two majors, both
fixed before commit: the fork-child counter reset and the telemetry `depth`
field. New tests: `test/test_estimator_ledger_occurrence_window.py` (26),
`test/test_json_streaming.py` (56), `test/test_current_checkpoint_json_layout.py` (5).

Deviations: the handoff's Increment 2 (deterministic parallel compile) was not
implemented (≤ 10 s per round at stake). The compact layout is opt-in and
changes file bytes, not content; readers use `json.load` and the recorded
sha256 is always computed from the bytes written. GitNexus: the index follows
`main`, which lacks the session-kernel symbols, so `impact` returned UNKNOWN
for them; callers were verified with `rg` and `detect_changes` (scope `all`,
worktree) reported risk low with no affected processes for every commit.

Commits: `d893a6b0`, `60c93b26`, `ef42dd04`.

## Increment 3 — screening-only surrogate

Status: skipped (compile is ≤ 10 s per round; a surrogate cannot pay for its
own exact rescoring).

## Increment 4 — local runner, CHTC template, SSH helper

Status: not started. Blocked on Increment 1 (bounded-memory hydration) and on
measuring the remaining checkpoint cost with the new telemetry on a real tail.

## Increment 5 — this report

Status: done.

## Remaining blockers, ranked

1. History rows embed a deep copy of the full Phase-0 receipt (population and
   ranking, P·reps(k) rows) and the P-candidate commutation plan table; the
   checkpoint deep-copies the history 3× and pure-Python-encodes it up to 5×.
   Slimming the rows changes checkpoint content, so consumers must be verified
   first (step 3 of the plan).
2. Resume hydration that honors the stream directory (Increment 1).
3. Remaining O(S_alg) passes per round: `closed_occurrence_prefix_summary`
   (1), the two prefix-identity digests (identity strings, cannot change),
   the checkpoint `to_payload` (1), and the pretty `current.json` write unless
   `STATIC_ADAPT_CHECKPOINT_JSON_LAYOUT=compact` is enabled.
4. No `gc` tuning anywhere on the path; a `gc.freeze()` after hydration is an
   untested hypothesis.

## Provenance

Audit workflow `wf_3bbe68f2-abb` (13 agents; conclusions and refutations in
the worklog entry). Verification workflow `wf_e62c8825-c21` (parity harness,
adversarial review, test author, two refuters). Sealed CHTC source used as the
byte-level reference: `source_locked.tar.gz` SHA-256
`fd7b37dd3329e074e3fcb1966a65cae35eaada09cccbee1a02caab6e1eb45d7a`.
