# HANDOFF — CHTC rescue fetch, scheduler reconciliation, and verification

Author: Claude session, 2026-08-31, replacement M4 Pro. Executor: Codex.
Written under `agent_guidance/shared/agent-handoff-contract.md`. Standalone:
no memory of the authoring conversation is required.

## Anchors

| field | value |
|---|---|
| Checkout | `/Users/lordskyler/Holstein_test` (the ONLY local copy; no worktrees active) |
| Branch / SHA at authoring | `main` @ `cb263a05` (a full-suite run may add commits after; `git pull` first, do not rebase or force anything) |
| Test baseline | `.venv/bin/python -m pytest test --collect-only -q` → 7,535 tests / 9 known collection errors (`RECOVERY-AUDIT.md` §12). Do not run the full suite — it may already be running from another session (one heavy job per machine). |
| Python | `.venv/bin/python` (3.12.14), cwd must be repo root for imports |
| Remote | `jsstrobel@ap2001.chtc.wisc.edu` (128.105.68.112), via GlobalProtect VPN (user connects VPN first) |
| Remote layout | `/home/jsstrobel/<campaign>/{package,image.sif,logs,transfer,submit_home.sub}`; results also under `/staging/jsstrobel/<campaign>/transfer/`; remote repo `~/Holstein_phase3_optuna_chtc` |
| Local fetch root | `~/chtc_rescue/` (on the 1 TB disk; expected total ~65–78 GB from `/staging` + ~18.6 GiB `/home`) |
| Authorities | `CHTC-FETCH.md` (procedure + 221-path grep), `docs/reports/EVIDENCE_PARITY_AUDIT_20260831.md` §3 (priorities, shas), `agent_guidance/skills/chtc-direct/SKILL.md` (operating rules), `chtc/paper_i_ra_adapt_repair_20260727/CAMPAIGN_STATE_20260817.md` (layout) |

## Scope

**In scope:** SSH login, host-key pinning, one-shot scheduler-state capture,
quota/inventory reads, rsync fetches, sha verification, writing the report
file named below, committing that report + verification receipts.

**Out of scope, with reasons:**
- **Any `condor_submit`, resubmit, release, or remote edit/cleanup.**
  `chtc/paper_ii_staged_accuracy_euler_v1_t20_resume_repair_d3301581_20260826/report_refresh_required.json`
  formally blocks submissions, and fetch-only tasks never modify remote state
  (worklog §6an rule, encoded in the chtc-direct skill). Remote deletion only
  after the user confirms the local copy is backed up off-machine.
- **Repo code edits.** Another session is mid-restore; touching
  `pipelines/**` or `test/**` risks conflicts. Report, don't fix.
- **Heavy local compute.** 18 GB aggregate agent ceiling
  (`agent_guidance/shared/memory-budget.md`); a full-suite run may be active.
  rsync + shasum are fine.

## Increments (execute all autonomously; stop only at a stated stop condition)

### 1 — Login and pin the host key
Goal: one authenticated SSH session, host key recorded.
```
ssh jsstrobel@ap2001.chtc.wisc.edu   # user connects GlobalProtect first
```
- Exactly **one** Duo push. If it times out or is denied: STOP, report,
  never auto-retry (`agent_guidance/time-dynamics/paper-ii-overnight-campaign.md` ~273–290).
- Keep this session/PTY open and reuse it for everything below (ControlMaster
  or a persistent shell; ~12 h reuse target).
- No host-key fingerprint survives in the repo. On first connect, capture
  `ssh-keygen -lf` of the offered key, have the user verify against
  https://chtc.cs.wisc.edu published keys if available, and record the
  accepted fingerprint in the report AND in
  `agent_guidance/skills/chtc-direct/SKILL.md` (edit that one file's
  "Connection facts" — this edit is pre-authorized here).
Expected: login banner; fingerprint recorded.
Stop condition: cannot authenticate → report; do not try alternate hosts.

### 2 — One-shot scheduler-state capture
Goal: the open dispositions from 2026-08-30 are reconciled.
```
condor_q -allusers jsstrobel -nobatch > ~/chtc_rescue/condor_q_$(date +%Y%m%dT%H%M%S).txt 2>&1
condor_history jsstrobel -limit 200 > ~/chtc_rescue/condor_history_$(date +%Y%m%dT%H%M%S).txt 2>&1
```
(run remotely; copy the two files back). Address by exact cluster ID —
JobBatchName is unreliable:
- **9716077** (siam r07 continuation, UNCONFIRMED since 2026-08-30T20:18Z):
  determine ran/failed/never-ran; if outputs exist, add them to the fetch.
- **9713112 proc3** (resource-rescue TERMINAL_FAILURE_EVIDENCE_PRESERVED):
  confirm its preserved evidence paths still exist.
Expected: both files nonempty; a stated disposition for each cluster.
Stop condition: none — a failed capture is itself a reportable result.

### 3 — Inventory before fetching
Goal: current remote inventory vs the recorded 221 paths.
```
ssh <session> 'du -sh /home/jsstrobel /staging/jsstrobel 2>/dev/null; ls /home/jsstrobel | wc -l'
```
Then regenerate the recorded-path list locally with the grep command at the
bottom of `CHTC-FETCH.md` and diff against a remote `find` listing.
Expected: ~18.6 GiB `/home` (47 top-level items), ~65–78 GB `/staging`
(2026-08-30 figures); note anything already purged.
Stop condition: if `/staging/jsstrobel` is already gone, skip to increment 5
with what remains and mark PURGED items in the report.

### 4 — Priority fetches, sha-verified (P0 first, in this order)
Goal: the irreplaceable archives are local and hash-verified.
Fetch to `~/chtc_rescue/<same-relative-path>` via
`rsync -avh --progress` over the open session. Verify each with `shasum -a 256`.

| P | Item | Verify against |
|---|---|---|
| P0 | `paper_ii_t20_resume_fix_cluster9698264_12.tar.gz` (13,271,737 B) | sha `d42aa3e7…` + per-file: the 36 shas in the t20 receipt (`MATH/paper_facing/paper_II_dynamics/t20_provisional_author_override_receipt.json`) |
| P0 | Paper I L4 matched-k75 RA `result.tar.gz`, campaign `paper_i_l4_qiskit_proxy_extensions_20260825{,_v2,_v3}`, cluster 9697719 | sha `1d0ac841…fca614`; member `./ra_l3_matched_k75/result.json` |
| P1 | `paper_ii_frontier2_final_25.tar.gz` (16.6 MB), `paper_ii_exchange_repair_cluster9698201_22.tar.gz` (9.3 MB), and dir `raw_outputs/paper_ii_staged_accuracy_euler_v1_pre_repair_clusters9697391_9697725_20260825/` (no tarball — rsync recursively) | receipts under `chtc/paper_ii_staged_accuracy_euler_v1_*` |
| P1 | Paper I cluster-9675078 strong-insertion final outputs (Fig 2 partial-prefix tails) | live-tail shas in `ADAPT---Paper-I` figure provenance |
| P2 | `paper_ii_t20_cluster9698250_12.tar.gz`, `paper_ii_threshold_cluster9698247_14.tar.gz`, cluster-9713112 proc3 delta | receipts in repo `chtc/` tree |
| P2 | cluster-9664325 B3 tarballs (`sixregime__*__9664325__NN.tar.gz`) — **no in-git hashes: record shas at fetch time**; cluster-9674089 b3mr archives (18 tarballs) | `agent_guidance/static-adapt/golden/ARCHIVES.sha256` |
| P3 | `/home/jsstrobel/Holstein_siam_nb4_source_446b75f3_20260828/` (7 tarballs, ~6.7 MB) | per-archive shas in receipts |
| P3 | `chtc/generic_time_dynamics_table/input` → `paper_ii_class_settings_lock_v1.json` | needed by the paper-ii-run gate + future calibration skill |
| P4 | prospective archives: HH/Hubbard transfer_archives (incl. both Hubbard exact_reference tarballs), `paperii_combined_cycle_20260830` uploads, hh_apm failure logs | best-effort |

Full sha values: `docs/reports/EVIDENCE_PARITY_AUDIT_20260831.md` §3.
Expected: every fetched archive PASS (or its sha recorded, for 9664325).
Stop condition: any sha MISMATCH → keep the file, mark MISMATCH, continue;
do not re-fetch more than once.

### 5 — Full-tree rescue rsync
Goal: complete copies, not just priorities (disk allows it).
```
rsync -avh --progress jsstrobel@ap2001.chtc.wisc.edu:/home/jsstrobel/ ~/chtc_rescue/home/
rsync -avh --progress jsstrobel@ap2001.chtc.wisc.edu:/staging/jsstrobel/ ~/chtc_rescue/staging/
```
(`CHTC-FETCH.md` is the authority if its commands differ — prefer its exact
flags.) Expected: transfer totals consistent with increment 3's `du`.
Stop condition: local disk under 150 GB free → stop the staging pull, report.

### 6 — Report and commit
Goal: results durable in version control.
Write `chtc/RESCUE_FETCH_REPORT_20260831.md`: one block per increment in the
contract's report-back format (status / commands / result / verification /
deviations), the verification table (per-item sha PASS/MISMATCH/PURGED), the
condor dispositions, and recorded-at-fetch shas for 9664325. Commit that file
+ the chtc-direct fingerprint edit with explicit paths (never `git add -A`),
then push. Note `~/chtc_rescue/` itself stays untracked — remind the user to
back it up off-machine (their choice of destination).

**Do NOT expect these to exist remotely** (confirmed absent 2026-08-31; their
absence is not a failure): Paper II staged-lane exact references, the 345-row
ledger JSON, `staged_report.pdf/json`, the r07 submission receipt, compiled
manuscript PDFs.

## Traps (each of these has already bitten an agent here)

- One Duo push means one. A second push after a timeout locks the user out of
  trust in the automation — report instead.
- `condor_q` filtered by JobBatchName lies; exact cluster IDs only. Never
  conclude "no jobs" from a partial read.
- `output/` and `prompt-exports/` are gitignored — durable artifacts go in
  tracked paths (the report path above is tracked).
- rsync trailing slashes: `src/ dst/` copies contents; `src dst/` nests the
  dir. The commands above are contents-copies.
- Broad `git add` sweeps other agents' work (incident `ba7f2ac9`). Explicit
  paths only.
- A path or `sha256: "missing"` sentinel is not a source lock — a claim of
  "fetched and verified" requires the actual hash comparison in the report.
