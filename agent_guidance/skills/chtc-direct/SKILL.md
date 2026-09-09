---
name: chtc-direct
description: Direct CHTC submit-node operations — login, fetch/rsync, condor status, storage hygiene, and cleanup for this repository's campaigns. Use when the current request operates against the CHTC submit node (rescue fetch, job status, archive retrieval, quota checks). Do not use for planning runs (paper lane run skills own that) or for local execution.
---

# CHTC direct operations

Recreated 2026-08-31 on the replacement machine (the original was a
user-level skill lost with the previous machine — `RECOVERY-AUDIT.md`
§8, recreation spec `docs/reports/EVIDENCE_PARITY_AUDIT_20260831.md`
§4). This file is deliberately repo-tracked so it survives machine loss.
It is an adapter: the documents below are the authority; this file only
routes and pins the operational rules.

## Authorities (read before acting)

1. `CHTC-FETCH.md` — the rescue-fetch procedure: rsync commands, the
   221-path regeneration grep, and the selective priority list.
2. `docs/reports/EVIDENCE_PARITY_AUDIT_20260831.md` §3 — the current
   prioritized fetch list (P0–P4) with per-item shas and receipts.
3. `chtc/paper_i_ra_adapt_repair_20260727/CAMPAIGN_STATE_20260817.md` —
   remote layout and campaign conventions.
4. `agent_guidance/shared/run-guide.md` — status gates and general run
   discipline.
5. `agent_guidance/shared/chtc-storage-sources.md` — storage hygiene.
6. `agent_guidance/time-dynamics/paper-ii-overnight-campaign.md`
   (lines ~273–290) — the one-Duo-push rule, verbatim authority.

## Connection facts

- Submit host: `jsstrobel@ap2001.chtc.wisc.edu` (128.105.68.112),
  reached via GlobalProtect VPN.
- Remote layout: `/home/jsstrobel/<campaign>/{package,image.sif,logs,transfer,submit_home.sub}`;
  results remapped to `/staging/jsstrobel/<campaign>/transfer/`;
  remote repo root `~/Holstein_phase3_optuna_chtc`; pinned apptainer
  image sha256 `fa5c4ea8…` (full value in `CAMPAIGN_STATE_20260817.md`).
- **Host-key pinning gap:** no ap2001 host-key fingerprint survives in
  the repo. Pin on a user-verified first connect or from CHTC's
  published host keys; record the accepted fingerprint here once
  verified.

## Hard rules

- **Standing authorization.** When authorized CHTC work is due, check the
  shared master and, if needed, initiate one SSH login and one Duo push
  immediately. The author approves the notification; no fresh connection
  request or readiness question is needed. Follow the connection sequence in
  `paper-ii-overnight-campaign.md`, including prompt handling, master reuse
  (~12 h target), and reporting an actually denied or timed-out push.
- **Status reads are one-shot and read-only:** single `condor_q` /
  `condor_history` invocations against exact recorded cluster IDs
  (JobBatchName is unreliable). Bounded `condor_ssh_to_job` extraction
  only per the recipe in `agent_guidance/skills/paper-i-run/SKILL.md`
  (~lines 226–238). Never watch the queue continuously; never conclude
  "no jobs active" from a partial observation.
- **Fetch-only tasks never submit, resubmit, or modify remote state.**
- Verify every fetched archive against its recorded sha before use;
  the P2 cluster-9664325 B3 tarballs are the sole exception (no in-git
  hashes) — record their shas at fetch time.
- Respect quota: `/home/jsstrobel` is 40 GiB soft / 50 GiB hard; run
  `du -sh` before large pulls; fetch to local `~/chtc_rescue/` and back
  it up off-machine before any remote cleanup.
- Remote cleanup/deletion only with explicit user authorization naming
  the paths.

## Standing context (2026-08-31)

- The rescue fetch is time-sensitive (submit-node staging until purge).
- Unreconciled scheduler state at machine loss: cluster 9716077 (siam
  r07 continuation, UNCONFIRMED) — while logged in, capture one valid
  `condor_q` + `condor_history` pair and fetch its outputs if it ran.
- `report_refresh_required.json` under
  `chtc/paper_ii_staged_accuracy_euler_v1_t20_resume_repair_d3301581_20260826/`
  formally blocks new campaign submissions until resolved.
- Items the fetch will NOT return (confirmed absent remotely): Paper II
  staged-lane exact references, the 345-row ledger JSON,
  `staged_report.pdf/json`, the r07 submission receipt, compiled
  manuscript PDFs (`EVIDENCE_PARITY_AUDIT_20260831.md` §3).
