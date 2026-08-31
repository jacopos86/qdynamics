# Deep-Run Segmenting Playbook (CHTC memory diet, part 3)

For strong-sector (nph7) runs beyond ~iteration 50, do not submit one
monolithic job. Chain bounded segments through the validated
checkpoint-continuation mechanism, with the memory-diet switches on.
This bounds every job's RAM and output size regardless of total depth.

## The three controls (all pre-existing, all validated)

1. **Segment bounds** — cap how much one job executes:
   `adapt_segment_target_depth`, `adapt_segment_max_new_admissions`,
   `adapt_segment_wallclock_cap_s`. A segment that hits its bound stops
   cleanly with a resumable checkpoint (`current.json` + sha).
2. **Checkpoint continuation** — the next segment resumes via
   `AcceptedStateResume(checkpoint_path=..., checkpoint_sha256=...)`.
   Validated on CHTC (per-round checkpoint pairs; see the continuation
   kit pattern under
   `chtc/paper_i_ra_adapt_repair_20260727/continuation_kit_20260818/`).
   The resume validator re-verifies route identity and the sha bracket,
   so a corrupted or mismatched chain fails closed.
3. **Memory-diet switches** (2026-08-19) — set both in the job
   environment so the two dominant RAM growth terms live on disk:

   ```text
   STATIC_ADAPT_ESTIMATOR_LEDGER_STREAM_DIR = <scratch>/ledger_stream
   STATIC_ADAPT_RUN_HISTORY_SPILL_DIR       = <scratch>/history_spill
   ```

   Both are observationally identical to in-RAM operation (equivalence
   proofs in `test/test_estimator_ledger_streaming.py` and
   `test/test_spilling_history.py`); S_alg accounting stays live
   in-run and deterministically replayable post-run.

## Segment sizing rule

Strong sector (nph7): segments of **25 controller rounds** (r50 = 2
segments, r70 = 3). Weak sector (nph3): segmenting unnecessary below
r70; use one job. Wallclock cap per segment: 2x the observed per-round
time x 25, so a hung segment dies without burning the slot.

## Chain recipe (per cell)

1. Segment 1: submit with segment bounds + diet env; fetch
   `current.json` + `.sha256` on completion.
2. Segment k+1: stage the prior checkpoint pair as inputs; request
   resume with the recorded sha; same route contract, same seed, same
   bounds. The route identity check makes a wrong-chain submit fail
   closed rather than produce mixed evidence.
3. Final segment emits the ordinary terminal payloads; aggregate as
   usual. The spilled ledger/history files are job outputs — compress
   with the job (gzip in the wrapper) and fetch only if the analysis
   needs occurrence-level audits; the summaries inside the payloads
   carry S_alg either way.

## Storage notes

- Spill/stream files are per-job scratch: name them into the job's
  working directory so CHTC cleans them unless explicitly fetched.
- Apply the artifact-retention contract to fetched raw outputs
  (`agent_guidance/shared/artifact-retention.md`): fetched archives are
  raw class; summaries/payloads are evidence class.
