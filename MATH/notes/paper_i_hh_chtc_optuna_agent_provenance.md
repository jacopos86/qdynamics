# Paper-I HH CHTC Optuna Agent Provenance

Purpose: agent-facing working ledger for the June 2026 Hubbard-Holstein CHTC
Optuna/SPSA runs. This is not a manuscript source, table-promotion artifact, or
human-facing report. It is meant to let the next agent resume monitoring,
retrieval, repair, and aggregation without reconstructing chat context.

Last snapshot: `2026-06-12T07:00:23-05:00` from CHTC access point
`ap2001.chtc.wisc.edu`, remote repo `~/Holstein_phase3_optuna_chtc`.

## Scope

Paper target: Paper I static ADAPT, Hubbard-Holstein Table III / related U8
diagnostic extension.

Active scientific intent from chat:

- Main HH SNAKE novelty Optuna batch: four Table-III regimes, two novelty
  variants where applicable:
  - exponent novelty term;
  - flat novelty term.
- Additional strong-Hubbard diagnostic sector: set strong Hubbard to `U/t=8`.
  This changes:
  - U8 strong-weak: `(U/t, lambda) = (8, 0.25)`;
  - U8 strong-strong: `(U/t, lambda) = (8, 1.25)`.
- U8 comparator SPSA suite: run comparators through their Optuna/SPSA route for
  the new strong-Hubbard sector.

Important semantic correction from chat:

- There was no HH table discrepancy caused by a different SNAKE novelty exponent
  convention. Existing HH tables all used an exponent term in novelty. The new
  batch intentionally compares exponent novelty against flat novelty.

## Non-negotiable Handling Rules

- Do not store or echo credentials in files, commands, logs, or reports.
- Do not kill, remove, hold, release, or delete unrelated jobs.
- Do not delete remote generated output unless it has been fetched locally and
  count plus byte size have been verified.
- Do not call any row paper-promotable or not paper-promotable. Report objective
  evidence only.
- For SNAKE jobs below, live `current_best.json` is missing in the observed
  SNAKE output directories. Best SNAKE values in this ledger are extracted
  directly from completed `trial_*/.../json/result.json` files, using
  `cutoff_diagnostics.abs_error_same_cutoff` when present.

## Current CHTC Storage

Snapshot:

```text
home:       39G / 40G quota-adjacent
raw_outputs 30G
logs        400M
```

## Incremental Status Snapshot — 2026-06-12T11:05:25-05:00

Source: live CHTC `condor_q` / `condor_history` on `ap2001`, plus targeted
scans of the known output directories. This snapshot is additive; it does not
replace the earlier fetched evidence.

Queue state:

| Scope | Cluster | Running procs | Completed ok | Nonzero |
|---|---:|---|---|---|
| Table-III SNAKE exponent/flat | `7646619` | `4,5,6,7` | `0,1,2,3` | none seen |
| SNAKE novelty-surface | `7652257` | `0,3,5` | `1,4` | `2` exit `124` |
| U8 comparator SPSA | `7651653` | `2,4,6,8,9,10` | `0,1,3,5,7,11` | none seen |

Newly visible completed result:

| Job | Record | Trials visible | Best trial | Best same-cutoff deltaE | Remote source | Local fetch |
|---|---|---:|---:|---:|---|---|
| `7652257.1` | `paper_i_hh_snake_novelty_surface_optuna_20260611_v2_tableiii_strong_weak` | 11 | 7 | `2.7608714489363306e-04` | `raw_outputs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2_tableiii_strong_weak/run/hh_L2_nph2_three_model_sym_strong_weak/trial_0007/hh_L2_nph2_three_model_sym_strong_weak/json/result.json` | `raw_outputs/chtc_fetches/paper_i_hh_20260612_incremental/strong_weak_novelty_surface_best/result_trial_0007.json` |

Local fetch verification for the best JSON:

```text
path: raw_outputs/chtc_fetches/paper_i_hh_20260612_incremental/strong_weak_novelty_surface_best/result_trial_0007.json
size: 45991454 bytes
sha256: a4ddb2c635bb1b5c4ccc1b67b979f87bc1a2432bc5b0947d8b0f02ecfff8dbb8
metric: cutoff_diagnostics.abs_error_same_cutoff = 0.00027608714489363306
logs: logs/chtc_fetches/paper_i_hh_20260612_incremental/paper_i_hh_snake_novelty_surface_optuna_20260611_v2.7652257.1.{err,log,out}
```

Active job mapping:

| Job | Scientific row | Evidence status |
|---|---|---|
| `7646619.4` | Table-III weak-strong SNAKE exponent | running; output not transferred |
| `7646619.5` | Table-III weak-strong SNAKE flat | running; output not transferred |
| `7646619.6` | Table-III strong-strong SNAKE exponent | running; output not transferred |
| `7646619.7` | Table-III strong-strong SNAKE flat | running; output not transferred |
| `7652257.0` | Table-III weak-weak SNAKE novelty-surface | running; output not transferred |
| `7652257.3` | Table-III strong-strong SNAKE novelty-surface | running; output not transferred |
| `7652257.5` | U8 strong-strong SNAKE novelty-surface | running; output not transferred |
| `7651653.2` | U8 strong-strong append ADAPT SPSA | running; stderr shows study creation only |
| `7651653.4` | U8 strong-strong Geo-ADAPT SPSA | running; stderr shows trial 0 finished |
| `7651653.6` | U8 strong-strong HEA SPSA | running; stderr shows through trial 8 finished |
| `7651653.8` | U8 strong-strong Qubit/QEB SPSA | running; stderr shows study creation only |
| `7651653.9` | U8 strong-weak Qubit/QEB SPSA | running; stderr shows trial 0 finished |
| `7651653.10` | U8 strong-strong TETRIS SPSA | running; stderr shows study creation only |

Monitoring caveats:

- The U8 comparator submit uses `when_to_transfer_output = ON_EXIT_OR_EVICT`
  and `transfer_output_files = raw_outputs, logs`. Running comparator result
  JSON is therefore not visible on the access point until transfer, although
  Optuna stderr lines reveal some completed trials.
- The remote comparator `raw_outputs/paper_i_hh_u8_comparator_spsa_v1/records`
  directory is currently empty on the access point after prior retrieval/cleanup
  and before the still-running jobs transfer their outputs.
- The newly completed `7652257.1` full output directory is about `1.7G`; only the
  current best `result.json` and matching logs were fetched in this incremental
  step. Fetch the full directory later if every trial JSON is needed locally.

Previous cleanup already performed:

| Remote output dir | Local fetch path | Files | Bytes | Remote state |
|---|---|---:|---:|---|
| `raw_outputs/routeA_paper_i_snake_tau2e4_hh_L2_nph1_clean_weak_marrakesh_v6` | `raw_outputs/chtc_fetches/paper_i_hh_20260611_storage/routeA_paper_i_snake_tau2e4_hh_L2_nph1_clean_weak_marrakesh_v6/` | 925 | 1994391834 | deleted after verification |
| `raw_outputs/routeA_paper_i_clean_phonon_ladder_nph2_ref4_harmonic_kerr_chain_l2_nph2_clean_weak_hk_snake_rerun_v2_v1` | `raw_outputs/chtc_fetches/paper_i_hh_20260611_storage/routeA_paper_i_clean_phonon_ladder_nph2_ref4_harmonic_kerr_chain_l2_nph2_clean_weak_hk_snake_rerun_v2_v1/` | 413 | 1853169702 | deleted after verification |
| `raw_outputs/routeA_paper_i_snake_tau2e4_ttprime_hubbard_L2_clean_strong_marrakesh_v6` | `raw_outputs/chtc_fetches/paper_i_hh_20260611_storage/routeA_paper_i_snake_tau2e4_ttprime_hubbard_L2_clean_strong_marrakesh_v6/` | 980 | 1566637237 | deleted after verification |
| `raw_outputs/hubbard_full_noise_sweep_corrected_20260610_v2` | `raw_outputs/chtc_fetches/paper_i_hh_20260611_storage/hubbard_full_noise_sweep_corrected_20260610_v2/` | incomplete partial | about 496M local | remote vanished during rsync; not deleted by agent |

If quota again approaches 40G, repeat targeted fetch/verify/delete on inactive
directories only. Do not touch active batch directories listed below.

## Live / Recent Condor State

Snapshot query:

```text
condor_q -constraint 'ClusterId == 7646619 || ClusterId == 7652257 || ClusterId == 7651653'
condor_history jsstrobel -limit 300 ...
```

### Batch Summary

| Cluster | Batch | Running procs at snapshot | Completed ok in recent history | Nonzero in recent history | Notes |
|---:|---|---|---|---|---|
| 7646619 | `holstein-paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1` | 4, 5, 6, 7 | 0, 1, 2, 3 | none seen | Table-III SNAKE novelty batch |
| 7652257 | `holstein-paper_i_hh_snake_novelty_surface_optuna_20260611_v2` | 0, 1, 3, 5 | 4 | 2 with exit code 124 | U8 / novelty-surface SNAKE batch; proc 2 needs log diagnosis |
| 7651653 | `holstein-paper-i-hh-u8-comparator-spsa-v1-full` | 2, 4, 6, 8, 9, 10 | 0, 1, 3, 5, 7, 11 | none seen | U8 comparator SPSA suite |

Unrelated or older held jobs still visible:

| Cluster | Procs | Batch | Hold reason |
|---:|---|---|---|
| 7466802 | 0 | `paper-i-hh-snake-structural-continue-recovery-20260604-v2` | memory limit at 65536 MB |
| 7466802 | 1 | `paper-i-hh-snake-structural-continue-recovery-20260604-v2` | user hold |
| 7582403 | 0 | `paper-i-hh-strong-weak-snake-continue-currentbest-20260608-v1` | user plateau stop for retrieval after iteration 129 |
| 7593091 | 0-5 | `paper-i-hh-tableiii-competitor-history-extend-20260609-v1` | transfer output failure: missing `scratch/logs` |

Do not act on held jobs unless the user explicitly brings them back into scope.

## SNAKE Result Snapshot

Metric: `cutoff_diagnostics.abs_error_same_cutoff` from completed
`result.json` files. Lower is better. These are partial Optuna snapshots, not
terminal batch aggregates.

| Regime / sector | Variant | Remote output dir | Completed trial JSONs | Trials seen | Best trial | Best same-cutoff deltaE | Newest result time |
|---|---|---|---:|---|---:|---:|---|
| weak-weak | exponent | `raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_weak_weak_exponent` | 7 | 0, 1, 3, 4, 9, 11, 19 | 4 | 3.7671790687321405e-04 | 2026-06-11T22:01:59 |
| weak-weak | flat | `raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_weak_weak_flat` | 7 | 0, 1, 3, 4, 9, 11, 19 | 4 | 3.7671790687321405e-04 | 2026-06-11T21:32:48 |
| strong-weak | exponent | `raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_strong_weak_exponent` | 5 | 0, 2, 3, 10, 19 | 19 | 2.0797526934657196e-04 | 2026-06-11T21:40:59 |
| strong-weak | flat | `raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_strong_weak_flat` | 5 | 0, 2, 3, 10, 19 | 19 | 2.0797526934657196e-04 | 2026-06-11T20:55:17 |
| weak-strong | novelty-surface v2 | `raw_outputs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2_tableiii_weak_strong` | 0 | none | n/a | n/a | n/a |
| U8 strong-weak | novelty-surface v2 | `raw_outputs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2_u8_strong_weak` | 11 | 0, 1, 2, 3, 4, 7, 8, 9, 10, 13, 17 | 7 | 5.3588643219804055e-05 | 2026-06-11T21:39:36 |

SNAKE caveats:

- No live `current_best.json` was found in the SNAKE directories scanned.
- Some Condor procs are still running, so completed trial counts will change.
- `7652257.2` completed with exit code 124. Diagnose its `run.err`, `run.out`,
  and Condor event log before deciding whether to repair/resubmit. Do not assume
  science failure.
- Flat and exponent variants currently show identical best deltaE for the
  weak-weak and strong-weak records observed. That may be because the same best
  trial settings are represented so far, not because the variants are equivalent.

## U8 Comparator SPSA Snapshot

Source: remote `progress/current_best.json` files under
`raw_outputs/paper_i_hh_u8_comparator_spsa_v1/records/*/progress/`.

| Regime / sector | Method record | Best trial | Best deltaE | Objective | mtime |
|---|---|---:|---:|---:|---|
| U8 strong-strong | `static_family_informed_vqe` | 0 | 4.7942372696688496e-01 | 3.0093370611247554 | 2026-06-12T02:19:20 |
| U8 strong-weak | `static_family_informed_vqe` | 0 | 4.735413338657498e-01 | 3.007486052617863 | 2026-06-11T15:04:56 |
| U8 strong-weak | `static_full_meta_append_adapt_vqe` | 13 | 4.316680658444927e-05 | -0.6609033405307688 | 2026-06-11T23:51:06 |
| U8 strong-weak | `static_geo_adapt_vqe` | 10 | 1.2270497532052005e-05 | -1.2064740910961296 | 2026-06-11T14:41:55 |
| U8 strong-weak | `static_hea_qiskit_vqe` | 14 | 3.097220207705509e-02 | 2.1926501386770987 | 2026-06-11T19:33:01 |
| U8 strong-weak | `static_tetris_qubit_adapt_vqe` | 16 | 1.1970874688738142e-05 | -1.2170490391153395 | 2026-06-11T19:24:24 |

Comparator caveats:

- The snapshot only lists records with a visible `progress/current_best.json`.
- Additional comparator procs were still running at snapshot and may not have
  current-best files on the access point yet.
- Some records are completed in Condor history but should still be retrieved and
  aggregated from final JSONs before any table/support update.

## 2026-06-12 Quota Retrieval And Remote Cleanup

Retrieval window: started from the CHTC status snapshot at
`2026-06-12T07:06:17-05:00`; cleanup verified at
`2026-06-12T07:14:32-05:00`.

Local fetch root:
`raw_outputs/chtc_fetches/paper_i_hh_20260612_quota_retrieval/`

Transfer facts:

- `rsync` completed successfully.
- Total byte size transferred: `5948267090` bytes.
- Local regular-file count after transfer: `2187` files.
- Local byte total after transfer: `5948267090` bytes.
- The transferred tree is rooted under the fetch root with `--relative`, so
  remote `raw_outputs/...` paths appear locally as
  `raw_outputs/chtc_fetches/paper_i_hh_20260612_quota_retrieval/raw_outputs/...`.
- Fetched SNAKE scheduler logs are under
  `raw_outputs/chtc_fetches/paper_i_hh_20260612_quota_retrieval/logs/`.
- Comparator result directories were fetched and verified; comparator scheduler
  logs were not observed in the local fetched `logs/` tree and were therefore
  not deleted remotely.

Verified payload directories before remote deletion:

| Remote payload path | Files | Bytes |
|---|---:|---:|
| `raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_weak_weak_exponent` | 184 | 1463345864 |
| `raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_weak_weak_flat` | 184 | 1463284461 |
| `raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_strong_weak_exponent` | 172 | 931140898 |
| `raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_strong_weak_flat` | 172 | 931081325 |
| `raw_outputs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2_tableiii_weak_strong` | 27 | 6325549 |
| `raw_outputs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2_u8_strong_weak` | 208 | 901215843 |
| `raw_outputs/paper_i_hh_u8_comparator_spsa_v1/records/paper_i_hh_u8_comp_spsa__full__static_family_informed_vqe__hh_u8_strong_strong` | 187 | 6997213 |
| `raw_outputs/paper_i_hh_u8_comparator_spsa_v1/records/paper_i_hh_u8_comp_spsa__full__static_family_informed_vqe__hh_u8_strong_weak` | 187 | 3913813 |
| `raw_outputs/paper_i_hh_u8_comparator_spsa_v1/records/paper_i_hh_u8_comp_spsa__full__static_full_meta_append_adapt_vqe__hh_u8_strong_weak` | 227 | 88448202 |
| `raw_outputs/paper_i_hh_u8_comparator_spsa_v1/records/paper_i_hh_u8_comp_spsa__full__static_geo_adapt_vqe__hh_u8_strong_weak` | 227 | 81228429 |
| `raw_outputs/paper_i_hh_u8_comparator_spsa_v1/records/paper_i_hh_u8_comp_spsa__full__static_hea_qiskit_vqe__hh_u8_strong_weak` | 187 | 2174253 |
| `raw_outputs/paper_i_hh_u8_comparator_spsa_v1/records/paper_i_hh_u8_comp_spsa__full__static_tetris_qubit_adapt_vqe__hh_u8_strong_weak` | 207 | 66206376 |

Remote cleanup performed after local verification:

- Deleted exactly the 12 verified payload directories listed in the table above.
- Deleted fetched SNAKE scheduler logs for `7646619.{0,1,2,3}` and
  `7652257.{2,4}`.
- Did not delete the parent
  `raw_outputs/paper_i_hh_u8_comparator_spsa_v1/` tree because active comparator
  procs are still writing under sibling record directories.
- Did not delete any known running SNAKE directories:
  `7646619.{4,5,6,7}` or `7652257.{0,1,3,5}`.
- Did not delete any known running U8 comparator record directories:
  `7651653.{2,4,6,8,9,10}`.

Post-cleanup remote storage snapshot:

| Path | Size |
|---|---:|
| `/home/jsstrobel` | `34G` |
| `raw_outputs` | `25G` |
| `logs` | `397M` |

Post-cleanup scoped queue snapshot:

| Cluster.proc | Batch | Status |
|---|---|---:|
| `7646619.4` | `holstein-paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1` | `2` |
| `7646619.5` | `holstein-paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1` | `2` |
| `7646619.6` | `holstein-paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1` | `2` |
| `7646619.7` | `holstein-paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1` | `2` |
| `7652257.0` | `holstein-paper_i_hh_snake_novelty_surface_optuna_20260611_v2` | `2` |
| `7652257.1` | `holstein-paper_i_hh_snake_novelty_surface_optuna_20260611_v2` | `2` |
| `7652257.3` | `holstein-paper_i_hh_snake_novelty_surface_optuna_20260611_v2` | `2` |
| `7652257.5` | `holstein-paper_i_hh_snake_novelty_surface_optuna_20260611_v2` | `2` |
| `7651653.2` | `holstein-paper-i-hh-u8-comparator-spsa-v1-full` | `2` |
| `7651653.4` | `holstein-paper-i-hh-u8-comparator-spsa-v1-full` | `2` |
| `7651653.6` | `holstein-paper-i-hh-u8-comparator-spsa-v1-full` | `2` |
| `7651653.8` | `holstein-paper-i-hh-u8-comparator-spsa-v1-full` | `2` |
| `7651653.9` | `holstein-paper-i-hh-u8-comparator-spsa-v1-full` | `2` |
| `7651653.10` | `holstein-paper-i-hh-u8-comparator-spsa-v1-full` | `2` |

Status code `2` is Condor running. Hold reason was `undefined` for all rows
above.

Known failed terminal job with fetched local evidence:

- `7652257.2` exited with code `124`.
- Classification from fetched local logs: walltime/wrapper timeout, not a Python
  traceback or scientific exception.
- Condor event log line: normal termination with return value `124`.
- `stdout` shows the run was still active at depth `14` after about
  `21574` seconds; the final job line is
  `JOB DONE record_id=paper_i_hh_snake_novelty_surface_optuna_20260611_v2_tableiii_weak_strong status=124`.
- `stderr` says `Terminating squashfuse_ll after timeout` and notes that
  timeouts can be caused by a running background process.
- Its local fetched directory is
  `raw_outputs/chtc_fetches/paper_i_hh_20260612_quota_retrieval/raw_outputs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2_tableiii_weak_strong/`.
- Its fetched local logs are
  `raw_outputs/chtc_fetches/paper_i_hh_20260612_quota_retrieval/logs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2.7652257.2.{err,log,out}`.
- Repair direction: same scientific settings, resume/retry with a longer
  walltime or checkpoint-aware continuation if available. Do not change
  scientific settings without user approval.

## Commands For Next Agent

Run these from an already-authenticated CHTC shell at
`~/Holstein_phase3_optuna_chtc`.

Status and quota:

```bash
date -Iseconds
du -sh ~ 2>/dev/null
du -sh raw_outputs logs 2>/dev/null
condor_q -constraint 'ClusterId == 7646619 || ClusterId == 7652257 || ClusterId == 7651653' \
  -af ClusterId ProcId JobBatchName JobStatus HoldReason | sort
condor_history jsstrobel -limit 300 \
  -af ClusterId ProcId JobBatchName JobStatus ExitCode HoldReason \
  | egrep '7646619|7652257|7651653|holstein-paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1|holstein-paper_i_hh_snake_novelty_surface_optuna_20260611_v2|holstein-paper-i-hh-u8-comparator-spsa-v1-full' || true
```

Diagnose the nonzero SNAKE job:

```bash
grep -Hn "ExitCode\\|Traceback\\|ERROR\\|error\\|timed out\\|124\\|TIME" \
  raw_outputs/chtc_fetches/paper_i_hh_20260612_quota_retrieval/logs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2.7652257.2.* \
  2>/dev/null | head -n 200
```

Fetch after jobs are terminal:

```bash
mkdir -p raw_outputs/chtc_fetches/paper_i_hh_20260612_terminal
# Use rsync/scp interactively; do not embed credentials.
# Fetch only scoped output dirs for clusters 7646619, 7652257, and 7651653.
```

After any fetch, verify before remote deletion:

```bash
python3 - <<'PY'
from pathlib import Path
import os, json, sys
p = Path(sys.argv[1])
count = 0
total = 0
for root, dirs, files in os.walk(p):
    for f in files:
        fp = Path(root) / f
        try:
            total += fp.stat().st_size
            count += 1
        except OSError:
            pass
print(json.dumps({"path": str(p), "exists": p.exists(), "files": count, "bytes": total}))
PY /path/to/local/fetched/dir
```

## Next Agent Actions

1. Re-run the status commands above.
2. If `7652257.2` is still the only nonzero scoped job, inspect logs and classify
   exit 124. If it is walltime/plumbing and the intended scientific settings are
   clear, repair/resubmit the same settings. Ask before changing any scientific
   semantics.
3. When a scoped job leaves the queue, retrieve its completed JSONs and logs to a
   local `raw_outputs/chtc_fetches/...` path.
4. Verify local file count and byte size against remote before any remote delete.
5. Update this ledger with each new status snapshot, retrieval, failure
   classification, and best-deltaE table.
6. Only after terminal retrieval, aggregate final JSONs into a machine-readable
   support summary/source map. Do not edit manuscript tables from this ledger.

## Minimal JSONL Snapshot

The following JSONL is copied from the `2026-06-12T07:00:23-05:00` remote scan
and is intended for machine parsing by a later agent.

```jsonl
{"kind":"snake","dir":"raw_outputs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2_tableiii_weak_strong","regime":"weak_strong","variant":"novelty_surface","completed_result_jsons":0,"trial_numbers":[],"best_trial":null,"best_deltaE":null,"metric_key":null,"best_path":null,"newest_mtime_iso":null}
{"kind":"snake","dir":"raw_outputs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2_u8_strong_weak","regime":"u8_strong_weak","variant":"novelty_surface","completed_result_jsons":11,"trial_numbers":[0,1,2,3,4,7,8,9,10,13,17],"best_trial":7,"best_deltaE":5.3588643219804055e-05,"metric_key":"cutoff_diagnostics.abs_error_same_cutoff","best_path":"raw_outputs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2_u8_strong_weak/run/hh_L2_nph2_three_model_sym_u8_strong_weak/trial_0007/hh_L2_nph2_three_model_sym_u8_strong_weak/json/result.json","newest_mtime_iso":"2026-06-11T21:39:36.746595"}
{"kind":"snake","dir":"raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_strong_weak_exponent","regime":"strong_weak","variant":"exponent","completed_result_jsons":5,"trial_numbers":[0,2,3,10,19],"best_trial":19,"best_deltaE":0.00020797526934657196,"metric_key":"cutoff_diagnostics.abs_error_same_cutoff","best_path":"raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_strong_weak_exponent/run/hh_L2_nph2_three_model_sym_strong_weak/trial_0019/hh_L2_nph2_three_model_sym_strong_weak/json/result.json","newest_mtime_iso":"2026-06-11T21:40:59.186650"}
{"kind":"snake","dir":"raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_strong_weak_flat","regime":"strong_weak","variant":"flat","completed_result_jsons":5,"trial_numbers":[0,2,3,10,19],"best_trial":19,"best_deltaE":0.00020797526934657196,"metric_key":"cutoff_diagnostics.abs_error_same_cutoff","best_path":"raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_strong_weak_flat/run/hh_L2_nph2_three_model_sym_strong_weak/trial_0019/hh_L2_nph2_three_model_sym_strong_weak/json/result.json","newest_mtime_iso":"2026-06-11T20:55:17.968327"}
{"kind":"snake","dir":"raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_weak_weak_exponent","regime":"weak_weak","variant":"exponent","completed_result_jsons":7,"trial_numbers":[0,1,3,4,9,11,19],"best_trial":4,"best_deltaE":0.00037671790687321405,"metric_key":"cutoff_diagnostics.abs_error_same_cutoff","best_path":"raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_weak_weak_exponent/run/hh_L2_nph2_three_model_sym_weak_weak/trial_0004/hh_L2_nph2_three_model_sym_weak_weak/json/result.json","newest_mtime_iso":"2026-06-11T22:01:59.807879"}
{"kind":"snake","dir":"raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_weak_weak_flat","regime":"weak_weak","variant":"flat","completed_result_jsons":7,"trial_numbers":[0,1,3,4,9,11,19],"best_trial":4,"best_deltaE":0.00037671790687321405,"metric_key":"cutoff_diagnostics.abs_error_same_cutoff","best_path":"raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_weak_weak_flat/run/hh_L2_nph2_three_model_sym_weak_weak/trial_0004/hh_L2_nph2_three_model_sym_weak_weak/json/result.json","newest_mtime_iso":"2026-06-11T21:32:48.274234"}
{"kind":"comparator","record":"paper_i_hh_u8_comp_spsa__full__static_family_informed_vqe__hh_u8_strong_strong","path":"raw_outputs/paper_i_hh_u8_comparator_spsa_v1/records/paper_i_hh_u8_comp_spsa__full__static_family_informed_vqe__hh_u8_strong_strong/progress/current_best.json","best_trial":0,"best_deltaE":0.47942372696688496,"objective":3.0093370611247554,"mtime_iso":"2026-06-12T02:19:20.327112"}
{"kind":"comparator","record":"paper_i_hh_u8_comp_spsa__full__static_family_informed_vqe__hh_u8_strong_weak","path":"raw_outputs/paper_i_hh_u8_comparator_spsa_v1/records/paper_i_hh_u8_comp_spsa__full__static_family_informed_vqe__hh_u8_strong_weak/progress/current_best.json","best_trial":0,"best_deltaE":0.4735413338657498,"objective":3.007486052617863,"mtime_iso":"2026-06-11T15:04:56.595903"}
{"kind":"comparator","record":"paper_i_hh_u8_comp_spsa__full__static_full_meta_append_adapt_vqe__hh_u8_strong_weak","path":"raw_outputs/paper_i_hh_u8_comparator_spsa_v1/records/paper_i_hh_u8_comp_spsa__full__static_full_meta_append_adapt_vqe__hh_u8_strong_weak/progress/current_best.json","best_trial":13,"best_deltaE":4.316680658444927e-05,"objective":-0.6609033405307688,"mtime_iso":"2026-06-11T23:51:06.611511"}
{"kind":"comparator","record":"paper_i_hh_u8_comp_spsa__full__static_geo_adapt_vqe__hh_u8_strong_weak","path":"raw_outputs/paper_i_hh_u8_comparator_spsa_v1/records/paper_i_hh_u8_comp_spsa__full__static_geo_adapt_vqe__hh_u8_strong_weak/progress/current_best.json","best_trial":10,"best_deltaE":1.2270497532052005e-05,"objective":-1.2064740910961296,"mtime_iso":"2026-06-11T14:41:55.503310"}
{"kind":"comparator","record":"paper_i_hh_u8_comp_spsa__full__static_hea_qiskit_vqe__hh_u8_strong_weak","path":"raw_outputs/paper_i_hh_u8_comparator_spsa_v1/records/paper_i_hh_u8_comp_spsa__full__static_hea_qiskit_vqe__hh_u8_strong_weak/progress/current_best.json","best_trial":14,"best_deltaE":0.03097220207705509,"objective":2.1926501386770987,"mtime_iso":"2026-06-11T19:33:01.242998"}
{"kind":"comparator","record":"paper_i_hh_u8_comp_spsa__full__static_tetris_qubit_adapt_vqe__hh_u8_strong_weak","path":"raw_outputs/paper_i_hh_u8_comparator_spsa_v1/records/paper_i_hh_u8_comp_spsa__full__static_tetris_qubit_adapt_vqe__hh_u8_strong_weak/progress/current_best.json","best_trial":16,"best_deltaE":1.1970874688738142e-05,"objective":-1.2170490391153395,"mtime_iso":"2026-06-11T19:24:24.239176"}
```
