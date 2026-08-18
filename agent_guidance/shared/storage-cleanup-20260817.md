# Storage Cleanup Record 2026-08-17

User-authorized disk-space cleanup executed 2026-08-17 (Claude Code session).
This file is the authoritative record for paths that historical guidance,
handoffs, or provenance tables may still reference. It changes no scientific
identities: recorded SHA-256 values and byte sizes in dated documents remain
the description of what the removed archives contained.

## Deleted (superseded raw CHTC downloads, ~80 GB)

Each directory was verified to contain no file newer than 2026-08-09 before
deletion. All were raw fetched scheduler outputs whose successor repair,
continuation, and retrieval batches (2026-08-09 through 2026-08-17) remain on
disk.

Under `raw_outputs/`:

- `chtc_fetch_paper_i_hh_r50_full48_20260719/`
- `chtc_fetch_paper_i_hh_sr_pool_complements_20260719/`
- `chtc_fetch_paper_i_hh_sr_20260720/`
- `chtc_fetch_paper_i_hh_sr_20260721/`
- `chtc_fetch_paper_i_hh_append_projected_singleton_20260721/`
- `chtc_fetch_paper_i_ra_adapt_stationary_core_v7_9392883_20260729/`
- `chtc_fetch_paper_i_ra_adapt_stationary_core_v8_9392920_20260729/`
- `chtc_quota_archive_retrieval_20260801_legacy_large/`

Under `chtc/paper_i_ra_adapt_repair_20260727/`:

- `live_safety_snapshots_20260803_9401106/`
- `retrieved_chtc_20260731_release_v2/`
- `retrieved_chtc_20260731_append_r70_strong_strong/`
- `retrieved_chtc_20260801_append_r70_intermediate_weak/`
- `retrieved_chtc_20260802_append_r70_remaining/`
- `retrieved_chtc_20260803_historical_average_plateau_r70_cluster_9400878/`
- `retrieved_chtc_20260803_historical_mean_global_singleton_nph3_full/`

Not deleted (still present; do not confuse with the list above):
`retrieved_chtc_20260731_append_r70_strong_weak/`,
`retrieved_chtc_20260801_append_r70_weak_weak/`,
`raw_outputs/_preserved_archives/`, `output/storage_archives/`, and every
batch-local `resume_inputs/` copy.

## Compressed in place (no data loss)

Every run-artifact `*.json` larger than 100 MB, written before 2026-08-17 and
outside any `*20260817*` run directory, was gzip-compressed in place
(80 files, 48.4 GB to 7.1 GB): `result.json.gz`, `estimator_ledger.json.gz`,
`current.json.gz`, `checkpoint`/`estimator_call_ledger` variants. Restore any
file with `gunzip <file>.gz` or read it directly via `gzip.open`. New runs
still write plain `.json`.

In `output/local_runs` harness runs, `result.json` holds only a ledger
summary; the standalone `estimator_ledger.json(.gz)` is the sole full ledger
copy there and was therefore compressed, never deleted. (Only the
2026-08-16 chtc local-runtime batch embedded the full ledger inside
`result.json`; its standalone duplicates were removed after entry-count
verification.)

`resume_inputs/*.tar.gz` copies shared across batch directories are hardlinks
of one another; per-path size sums overcount them.

## Preservation invariants honored

- No completed evidence, manifest, or source lock from 2026-08-09 onward was
  deleted.
- The 2026-08-17 runs (`paper_i_ra_allphase_adaptive_20260817_comparison_data`,
  `paper_i_ra_allphase_adaptive_append_remaining5_maximum_k50_20260817_v1`)
  were excluded from every delete and compress operation.
- Quarantined compatibility/provenance documents were annotated with dated
  pointers only; no identity, checksum, or table value was rewritten.
