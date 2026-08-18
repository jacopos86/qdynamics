# Artifact Retention Contract

Read this file when creating, fetching, aggregating, or cleaning run
artifacts. It exists because raw scheduler downloads accumulated ~160 GB
unnoticed before the 2026-08-17 cleanup
(`agent_guidance/shared/storage-cleanup-20260817.md`). It governs storage
lifecycle only; it never overrides evidence, provenance, or promotion
authority.

## Classification at creation

Every fetched archive and completed run directory is one of:

- **evidence** — summaries, receipts, manifests, source locks, accepted
  results, and anything an active manuscript table or evidence contract
  references. Keep through publication. Never expired by this contract.
- **raw** — fetched scheduler tarballs, full estimator ledgers, checkpoint
  chains, safety snapshots, and superseded batch payloads. Working copies:
  CHTC/cloud retains the originals of fetched archives (user-confirmed
  2026-08-17). Raw artifacts are compressible immediately and expirable once
  superseded.

When a fetch or batch completes, record which class its payload belongs to in
the batch notes or manifest. Unclassified artifacts are treated as raw for
compression and as evidence for deletion (compress freely, never delete
unclassified data without user review).

## Compression rule

A run-artifact JSON larger than 100 MB is compressed in place (`gzip -6`,
producing `<name>.json.gz`) once its run has been complete for 7 days.
Readers use `gunzip` or `gzip.open`. Active runs and anything modified within
24 hours are never touched. New runs still write plain `.json`.

## Agent responsibility

The user does not run the report; agents do. Run
`python3 pipelines/shell/artifact_retention_report.py` and surface any
candidates to the user for approval whenever a task involves a CHTC fetch,
artifact aggregation, storage, disk space, or cleanup — and opportunistically
when a session touches this repository and the report has plausibly not run
within roughly a month. Never delete a candidate without the user's explicit
approval of that batch.

## Expiry rule

A raw artifact becomes an **expiry candidate** 30 days after it is superseded
— a newer batch replaces it, or its results are aggregated into evidence.
Expiry is staged, never automatic:

1. `python3 pipelines/shell/artifact_retention_report.py` prints candidates
   with sizes and newest-file dates. The report mutates nothing.
2. The user approves a listed batch for deletion.
3. Before deleting, verify the candidate contains no file newer than its
   supersession date and no evidence-class payload.

Protected regardless of age: paths matching `*_preserved*` or
`*storage_archives*`, evidence-class artifacts, anything from the current
UTC day, and directories the user has named as kept (current Paper-I append
runs through publication).

## Non-goals

This contract does not authorize any specific deletion, does not apply to
`src/`, `test/`, `MATH/`, guidance, or manuscripts, and does not replace the
per-cleanup dated records under `agent_guidance/shared/`.
