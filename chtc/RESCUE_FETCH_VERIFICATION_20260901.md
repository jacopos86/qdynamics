# CHTC Rescue Fetch — Verification, Disposition and Restoration Receipt

- **Date:** 2026-09-01
- **Fetched tree (read-only):** `~/chtc_rescue/` — 143 GB (`home/` = submit-node `/home/jsstrobel` mirror, `staging/` = `/staging` mirror, `inventory/`, `scheduler/` = condor_q + condor_history captures, `verification/` = the stalled Codex session's partial tables)
- **Authorities:** `chtc/HANDOFF_CHTC_RESCUE_FETCH_20260831.md` (fetch list + verification targets) · `docs/reports/EVIDENCE_PARITY_AUDIT_20260831.md` §3 (per-item shas) · `docs/reports/FULL_SUITE_BASELINE_20260831_classification.md` (data-missing rows 7/8/9 + chtc families)
- **Repo state:** base `fe7a74bc`; restoration commit **`0c511b0a`** (83 files, +61,467 lines) — local, **not pushed** (`git log origin/main..HEAD` = 1 commit)
- **Hard rules observed:** `~/chtc_rescue` read-only throughout (all extraction into session scratch; `find -newermt` confirms zero files touched by this session) · no SSH/network · no test-file edits · the one pipeline execution (Task D re-aggregation) wrapped in `.venv/bin/python pipelines/shell/ram_guard.py --limit-mb 8000` (exit 0, peak 231 MB); one heavy execution at a time.
- **Supersedes:** the Codex session's own rescue report, which was never written. Its transfer (the rsync pulls) and its partial verification tables are subsumed into §2 below — one of its two tables is corrected here.

---

## 1. Archive verification — every archive with a recorded sha

### 1.1 P0 / P1 / P2 archives

| # | Archive (path under `~/chtc_rescue/`) | Bytes | sha256 | Status |
|---|---|---|---|---|
| 1 | `home/Holstein_paper_ii_staged_accuracy_euler_v1_resume_fix_20260826/paper_ii_t20_resume_fix_cluster9698264_12.tar.gz` | 13,271,737 | `d42aa3e7ebf46d9dc141f4606f2b2798696fd99d3f0134901e41c5b16453aa74` | **PASS** (vs `completion_receipt_9698264.json`; 60 members) |
| 2 | `home/paper_i_l4_l3_matched_ra_20260826_v3/package/result.tar.gz` (Paper I L4 RA) | 543,900,119 | `1d0ac8416d124ba6a85118f8a1a228b4f754c5567c0eb341db515426fdfca614` | **PASS** — *location correction, see §6(a)* |
| 6 | `home/Holstein_paper_ii_staged_accuracy_euler_v1_20260825/.../paper_ii_frontier2_final_25.tar.gz` | 16,640,685 | `0d316bdabd281db78038a16fc486a64c13f9a24cf054b70f9a0f282b48614358` | **PASS** (vs frontier2 `submission_receipt.json`, cluster 9697725, 43 run.json) |
| 6 | `…/paper_ii_frontier2_code_and_package.tar.gz` | 11,946,939 | `7a2c8045f6c921de2b96967b80334de2461537dc3e932bb8fe1896109e95e981` | **PASS** (sha lives in the receipt's `transfer_archive` node, which has no `remote_path` — a remote_path-keyed scan skips it) |
| 6 | `paper_ii_exchange_repair_cluster9698201_22.tar.gz` | — | — | **NOT-FOUND — expected absence**, receipt records `deleted_after_verified_local_ingestion`; *see §6(b)* |
| P2 | `…/paper_ii_t20_cluster9698250_12.tar.gz` | 5,165,512 | `2c1afc45…` | **PASS** (vs `t20_promotions_20260826/completion_receipt_9698250.json`) |
| P2 | `…/paper_ii_threshold_cluster9698247_14.tar.gz` | 8,462,213 | `e0da222a…` | **PASS** (vs `threshold_extension_after9698201_20260826/completion_receipt_9698247.json`) |
| P2 | `…/resource_rescue_t5_frontier340_source_74da5166_upload.tar.gz` | 5,283,675 | (matches receipt) | **PASS** (vs its `remote_upload_receipt.json`) |

### 1.2 SIAM nb4 receipt-bound archives — `home/Holstein_siam_nb4_source_446b75f3_20260828/` (+ 1 at home root)

| Archive | sha256 (prefix) | Status |
|---|---|---|
| `r00_upload_20260828` (5,539,814 B) | `a6f1f8d0…` | **PASS** |
| `incoming/…frontier9714484` | `9f5e78a2…` | **PASS** |
| `incoming/…frontier9714525` | `0fc077ee…` | **PASS** |
| `incoming/…frontier9714675_upload_clean` | `e3c45aba…` | **PASS** |
| `incoming/…frontier9714816` | `88f48cb2…` | **PASS** |
| `incoming/…frontier9715075` | `b9fb68b7…` | **PASS** |
| `incoming/…frontier9715147` | `8a428364…` | **PASS** |
| `siam_r02_package` | `01852fb2…` | **PASS** |
| `home/siam_nb4_source_446b75f3_frontier9715270_upload_20260830.tar.gz` (1,035,685 B, at home root) | `bd7bcb6b…` | **PASS** |

**9/9 receipt-bound PASS; family total 7,834,227 B.**

Quarantined upload — **not a mismatch**: `incoming/quarantine/…frontier9714675_upload_20260829.appledouble.tar.gz` = `d41b9860789e921b7cf926ff84809707e22b8cb447bf226611f8d83db73a6ba6` (first raw-file anchor). The r04 receipt records only a *quarantine receipt logical* SHA-256 (`199883e8…`), a different digest definition, not a file hash. Bytes are demonstrably the recorded artifact: 46,776 B exactly as recorded, and `tarfile` shows exactly 20 members of which 10 are AppleDouble sidecars — precisely the "46776-byte 20-member attempt" described. (`bsdtar` reports 10 because it merges `._` entries; a listing artifact, not a content difference.) Forensic retention, not paper evidence.

### 1.3 Member-level verification

| Target | Result |
|---|---|
| t20 per-run members, 24 of 36 (12 `run.json` + 12 `terminal_qiskit_cost.json`) vs `t20_provisional_author_override_receipt.json` `rows[].run_sha256`/`cost_sha256` | **24/24 PASS** — independently re-extracted and recomputed twice (Task A, Task D), not spot-checked |
| t20 per-run members, remaining 12 of 36 (`screen_run_sha256`) | **12/12 PASS** after correct source selection — *supersedes the Codex table, see §2* |
| **Item 1 total** | **36/36 PASS** |
| L4 member `./ra_l3_matched_k75/result.json` (46,941,252 B) | **FOUND + first anchor RECORDED** `e8662dd1551445f47891dd4b2e96d02df771a9ef208ee04db2711d5150fc3de5` — no sha existed anywhere in git. Tarball has exactly 9 members, all under `./ra_l3_matched_k75/`. Sole backing for the manuscript's pure-Hubbard L4 matched-k75 block. |
| Unarchived directory fetch `raw_outputs/paper_ii_staged_accuracy_euler_v1_pre_repair_clusters9697391_9697725_20260825/` | **COMPLETE 43/43** local vs remote (87 remote entries = 43 dirs + 43 files + 1 root); 43-row sha table generated at restore-receipt quality |
| Unarchived directory fetch `raw_outputs/paper_ii_staged_accuracy_euler_v1/` (preserved cluster-9698201) | **COMPLETE 40/40** local vs remote |
| 9716077 (siam r07) provenance chain | **PASS** — `run.json` sha `27e1a91400df8de7a96b994bc1d24620d655f20c3955143c88c394f9985ff853` == `trajectory_sha256` in its own `action_receipt.json`; all 6 package inputs match the repo-tracked `submission_receipt.json` (`2a44a36e…`, `c188c655…`, `74a5ef96…`, `5752bc1e…`, `8e91ed87…`, `8919265b…`) |

### 1.4 First-anchor shas recorded for previously unhashed archives

| Archive | Bytes | sha256 |
|---|---|---|
| `home/…frontier2_…/paper_ii_frontier2_partial_18.tar.gz` (superseded partial of final_25, 18 run.json) | 7,384,540 | `0aedb8fb47eceb81b1be30a1b6e8bcc1100af2e5509150ad14419807ce847087` |
| `home/…/paper2_staged_code_20260825.tar.xz` | 3,262,668 | `b70571058e26e8504bd5482a5a1ad2d2812274d9b14c4619aee737ece1356dd8` |
| `staging/ra_adapt_costexp_depthfix_sixregime_20260820_v1/transfer/sixregime__b_depth_append_ra__strong_weak_u8__nph3__9664320__14.tar.gz` | 43,347,118 | `4608f4fde488d48ade3e436c7c13f46e1601a0d7b93fc8149e0fadff5b1e66a9` |
| `staging/ra_adapt_warmstart_sixregime_20260820_v1/transfer/sixregime__c_ws_depth_append_ra__weak_weak__nph3__9665210__0.tar.gz` | 245,221,689 | `5f81dfb4f45f7e4a6b2a6bba27a2f3549a247d7676e8c8ab0d192b388acb15e9` |

The two `sixregime__` rows are recorded **in lieu of** the lost cluster-9664325 set (§3 item 5). `…9664320__14` is the v1-cluster run of the same arm/regime/proc-index cell as the missing `9664325__14` — a **near-relative superseded by v2, not a substitute**.

### 1.5 Archive-read integrity across the full 1,089-tarball / 98.9 GB content scan

1,086 of 1,089 readable. 92 of the 95 listing failures are macOS AppleDouble `./._*` metadata stubs rsync carried over (harmless, not archives). Three genuine:

| Archive | Issue |
|---|---|
| `home/…/paper_i_hh_childpoolfix_v2_source_submit_20260626.tgz` | **TRUNCATED** — 4,096 B on disk vs 2.1 GB gzip-declared |
| `home/tmp_hh_repair_transfer/hh_weak_strong_repair_20260608_v1_submit.tgz` | **TRUNCATED** — 6,789 B vs 283 MB declared |
| `home/paper_i_fig2_fetch_serve_…/paper_i_fig2_json_fetch_7593090_7593091_20260611_105547.tgz` | **Broken symlink only** — its 5,034,354 B target was fetched intact at `home/paper_i_fig2_json_fetch_7593090_7593091_20260611_105547.tgz`; no data loss |

All three are June-era and unrelated to the fetch items.

---

## 2. Codex handoff tables — one validated, one corrected

Codex's partial tables live at `~/chtc_rescue/verification/`. Both were reused, not redone; both were then independently re-derived.

| Table | Codex result | Verdict here |
|---|---|---|
| `t20_9698264_run_cost_sha256.tsv` (24 rows) | 24 PASS | **FULLY VALIDATED.** All 24 rows re-derived from two independent re-extractions; every row agrees on status, expected and actual. Its extraction at `verification/t20_9698264/` is sound (60 files). |
| `t20_screen_run_sha256.tsv` (12 rows) | 9 PASS / **2 MISMATCH** / **1 MISSING** | **WRONG — the 3 non-PASS rows are false alarms. Corrected result: 12/12 PASS.** |

**Root cause (verified directly, not inferred).** The 12 screen shas are **not** in the t20 tarball. Codex hashed all 12 against its `verification/frontier2_final_25/` extraction alone — which for these arms carries the *pre-repair-era* bytes. The receipt-expected bytes are in the preserved cluster-9698201 raw outputs. Re-hashing resolves every row:

| Arm | Receipt expects | Codex reported (from frontier2_final_25 / pre_repair copy) | Byte-exact match found at |
|---|---|---|---|
| `screen_apm_t5__hh_snake_nph1__exchange__delta_theta_5e-5__weakslow__t5__cut3e-06/run.json` | `bbc9f8ba…` | `a149a371…` | `home/Holstein_paper_ii_staged_accuracy_euler_v1_20260825/raw_outputs/paper_ii_staged_accuracy_euler_v1/` → `bbc9f8ba…` ✔ |
| `screen_apm_t5__…__state_motion_1e-3_plus_parameter_5e-5__weakslow__t5__cut3e-06/run.json` | `38b2a2a3…` | `f48e4881…` | same dir → `38b2a2a3…` ✔ |
| `screen_avqds_t5__…__state_motion_5e-5__fastfast__t5__cut3e-07/run.json` | `1e1a1738…` | `MISSING` | same dir → `1e1a1738…` ✔ |

The two reported "actual" values are genuine hashes — of the superseded `…_pre_repair_clusters9697391_9697725_20260825/` copies of the same arm names. So the table is *internally faithful* (its col2 = expected-from-repo, col3 = actual-on-disk, and a 3-row spot-check of each table reproduced col3 exactly) but its **source selection was wrong**. Task D reached 12/12 independently by mapping all 12 screens across both raw_outputs dirs.

> **Action required before anything ships:** do not let `t20_screen_run_sha256.tsv` reach a commit as-is. A spurious MISMATCH on Paper II's only table-and-figure evidence is a false integrity failure. It did **not** enter commit `0c511b0a` — both Codex tables live outside the repo and the commit contains only the 83 restored artifact files.

---

## 3. Inventory dispositions

### 3.1 Fetch-list items (HANDOFF §Fetch list)

| # | Item | Disposition |
|---|---|---|
| 1 | t20 P0 archive + 36 member shas | **FOUND / VERIFIED** — 36/36 PASS |
| 2 | Paper I L4 RA `result.tar.gz` + `result.json` member | **FOUND / VERIFIED** — archive PASS, member first-anchored `e8662dd1…` |
| 3 | 18 cluster-9674089 b3mr tarballs vs `agent_guidance/static-adapt/golden/ARCHIVES.sha256` | **NOT-FOUND (0 of 18) — unrecoverable.** Both transfer dirs (`/home/jsstrobel/paper_i_hh_b3_measured_residual_20260821/transfer`, `/staging/j/jsstrobel/…/transfer`) are **empty on the remote** per the 2026-08-31 find, and empty in the rescue tree. Only 54 job logs (18 procs × .out/.err/.log), `package/` and `image.sif` survive. No `b3mr__*` or `*__9674089__*` anywhere on this machine. `ARCHIVES.sha256`'s recorded root `output/local_runs/…/transfer/` does not exist locally either. → **USER-DECISION** |
| 4 | SIAM nb4 receipt-bound tarballs | **FOUND / VERIFIED** — 9/9 PASS; quarantined upload explained (§1.2); **12 further campaign tarballs unanchored** → USER-DECISION (§3.3) |
| 5 | 12 `sixregime__*__9664325__NN.tar.gz` (Table I estimator ledgers) | **NOT-FOUND (0 of 12) — unrecoverable.** The instruction was to record first-anchor shas; **the bytes do not exist to hash.** Campaign `ra_adapt_costexp_depthfix_sixregime_20260820_v2`; transfer dirs empty in `/home` and `/staging` both remotely and locally; only 66 job logs survive. Across the whole remote inventory just two `sixregime__` tarballs exist at all — both recorded in §1.4. The audit's "only fetch item that cannot be sha-verified after the fact" is moot: it cannot be fetched at all. → **USER-DECISION** |
| 6 | frontier2 P1 family | **FOUND / VERIFIED** — 5/5 present archives PASS; `paper_ii_exchange_repair_cluster9698201_22.tar.gz` is an **expected absence** (deleted by design; its content survives as the preserved raw_outputs, which supply the 3 screen shas above); both directory fetches complete (43/43, 40/40) |

### 3.2 Classification data-missing shopping list (rows 5/7/8/9/15/28 + chtc families)

Resolved against 69,440 filesystem paths **plus a full content index of all 1,089 tarballs** — three of the highest-value items existed **only inside sealed archives** and were invisible to filesystem search.

| Item | Disposition |
|---|---|
| **Row 8 — `GLOBAL_SOURCE_LOCK` `output/pdf/paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715/…20260715.json`** | **RECOVERED BYTE-EXACT & RESTORED.** sha `8e057da082f9d81b308fc6563029ef9f1f309950b54191bd60bc47abfec95922` = the digest pinned at `pipelines/static_adapt/ra_adapt/bundles.py:563`. 1,762,635 B; present in 3 independent sealed archives, extracted and verified identical from 2 (`home/lanes_ablation12_r50_20260806_v1_chtc/source_locked.tar.gz` and `staging/paper_i_gradient_phase0_two_routes_20260807_v1/…/source/source_locked.tar.gz`). The same archive's other two lock members also PASS their pinned digests (`0153bbf8…`, `66a64097…`) — 3/3 checkable locks, proving a faithful source snapshot. **Row 8's "not byte-reproducible even after a full input fetch" is superseded; the re-pin-46-manifests-or-retire ruling is not needed.** |
| **Row 5 — `chtc/phase3_optuna/paper_i_table_i_audit_escalation.py`** | **FOUND & RESTORED.** In `home/rho1_redo_stage_20260609.tgz`; `git hash-object` = **`b9be9e1fc2120532c698af1096624e6289297997`**, byte-identical to `e838eb35:chtc/phase3_optuna/paper_i_table_i_audit_escalation.py`. The CHTC production copy and the July branch blob are the same bytes — row 5's proposed graft is *validated*, not replaced. |
| **Row 9 — SPSA smoke config + input tree** | **FOUND & RESTORED** (config + 5-file input tree). Config `2fef684b627dc3cc575a87e75cc69d4ad64d3daa92449b3166b6f09f7cba9c7f`, byte-identical across two independent archives; their sibling `submit_…_smoke.sub` = `2a318380…`, byte-identical to the repo's already-tracked copy — that is what pins the archives to genuine repo-era bytes. 7 of row 9's 11 configs are recoverable across 8 staged tarballs. **GAP:** `warm_start_schedule_lock.json` (needed at `test/…/test_paper_i_comparator_spsa_calibration_preflight.py:154`) is in **no** archive — 0 hits across all 1,089. → USER-DECISION |
| **Rows 15 + 28 — `chtc/generic_time_dynamics_table/input/`** | **FOUND & RESTORED** (75 non-seed files). Schema re-validated against row 15's warning: `paper_ii_class_tuned_case_manifest_v1`, **20 cases keyed by both tuning_class and table_class** — the required era, *not* Snake's wrong 16-case table_class-only manifest that would silently produce empty-class failures. |
| **Row 7 — `output/siam_screening/nb4_pilot_v1/`** | **NOT-FOUND.** Exhaustively negative on three independent indexes (fetched filesystem, remote find, all 1,089 tarball listings). A **local gitignored orchestration product that never existed on CHTC** — no fetch can return it. Partial substitute verified: CHTC raw evidence for the bound cluster 9713113 exists at `home/Holstein_siam_nb4_source_ba50339c_20260828/` and passes **5/5** against its own sha-bearing `evidence/siam_nb4_avqds_screen_9713113_proc0/disposition.json` (`4d376702…`, `724aed65…`, +3 logs); `condor_history` confirms 9713113.0 completed 8/28 11:19. But the five artifacts `test_siam_stable_bindings.py` asserts on are orchestrator outputs and are unrecoverable. → **USER-DECISION** |
| **Fig 2 tails — cluster 9675078 final outputs** | **NOT-FOUND — never produced.** All six procs (.0–.5) are in `condor_q` at 2026-08-31T21:50:58 in state **H (HELD)**, submitted 8/23 00:07, and **none** appears in `condor_history`. Resume chain 9712814/9712826/9712827/9712836/9712869 also HELD; 9713637/9713638 RUNNING. **Surviving partial state:** `b3_9675078_1_20260828T043705Z.tar.gz` (1,324,476,960 B) and `b3_9675078_2_20260828T044432Z.tar.gz` (1,263,907,130 B), each holding `current.json` + `current.verified_singleton_resume.*.json` + `current.estimator_call_ledger_checkpoint.*.json` — the partial-prefix state Fig 2 needs; remote parity exact; proc 0/4/5 checkpoints named in condor_q are already purged from remote disk. → Fig 2 tail reconstruction is a **USER-DECISION** |
| **Fig 3 raw — `l3_phase0` / `promotion_k15`** | **NOT-FOUND (belief confirmed).** `promotion_k15`: 0 hits in all three indexes. `l3_phase0`: 0 on the filesystem and 0 in the remote find; 22 archive hits collapse to 6 distinct member paths, **all code** (`chtc/ra_adapt_l3_phase0_qiskit_adaptive_20260826/run_arm.py`, `test/test_l3_phase0_qiskit_adaptive.py`, and `.pyc`) inside `source_locked.tar.gz` snapshots. A targeted search for any `result.json`/`run.json` under an `l3_phase0` path returns zero. This extends the parity audit's "no branch holds any l3_phase0 file" **to the submit node**. The runner code survives if a rerun is ever authorized. |

### 3.3 The 9716077 verdict (from the scheduler captures)

**RAN AND COMPLETED SUCCESSFULLY** — resolving the "UNCONFIRMED since 2026-08-30T20:18Z" open item.

- `condor_history`: `9716077.0 jsstrobel 8/30 15:18 0+21:09:30 C 8/31 12:27 run_action_apptainer.sh siam-nb4-full_apm-continue-r07-cut3em06`; correctly absent from `condor_q` (finished).
- Job log: `005 … 2026-08-31 12:27:37 Job terminated. (1) Normal termination (return value 0)`; stderr trace runs to completion at k=500/400, t=5.
- Outputs fetched at `home/Holstein_siam_nb4_source_446b75f3_20260828/raw_outputs/siam_nb4_full_apm_continue_r07_exchangefix_source_36c40d52_20260830/siam-nb4-full_apm-continue-r07-cut3em06/`: `run.json` (31,883,749 B) + `action_receipt.json` (216,118 B) + 3 logs. Full chain PASS (§1.3).
- **Benign non-match, do not misread:** the remote `remote_upload_receipt.json` (`fe62e1be…`) differs from the repo's (`f63dbfd6…`) — a diff shows they are *different documents* (remote = per-action upload receipt keyed by action_id/batch_name; repo = archive-upload receipt with the tar/gzip integrity block), not corruption. The repo copy is authoritative and already tracked.
- **Supersedes** the parity audit's "do NOT expect the r07 submission receipt": the continuation's action receipt and the full 31.9 MB trajectory both survive. Treat as **P0-grade new post-loss evidence**.

### 3.4 Two traps in the fetched tree

1. **Zero-byte `live_tails` are NOT fetched data.** `home/ra_adapt_costexp_depthfix_sixregime_20260820_v2/live_tails_9675078/{9675078.2,.3,.4,.5}.out` are **0 bytes**, dated Aug 31 21:59–22:00 (immediately after Codex's 21:53–21:58 window). `live_tails` has **zero hits** in both the remote find and `rsync_home_20260831.log`. They are local residue of an interrupted `chtc/paper_i_ra_adapt_repair_20260727/results_tooling/fetch_live_tails.sh` run (that script `rm -f`s empty tails; the `rm` never ran) — i.e. **the stalled session wrote into the read-only rescue tree**. Delete/ignore; the tree is no longer a pristine remote mirror.
2. **The remote repo clone is empty.** `home/Holstein_phase3_optuna_chtc/` contains exactly **one** file (`chtc/paper_iv_h2o_reencode_20260818/logs/9662568.log`), and `chtc/phase3_optuna/config` has **zero hits** in the entire 69,435-line remote find. The HANDOFF's "Remote layout" anchor and classification row 9 both assume that clone is the recovery path for the 11 SPSA configs — **it is not**; the staged `.tgz` archives are. A filesystem-only sweep would have declared rows 5/8/9 unrecoverable.

---

## 4. Restorations — commit `0c511b0a` (local, unpushed)

83 files, +61,467 lines. Commit message carries every file's sha256 and provenance path. Every artifact was re-verified **in the repo location**, not accepted on report.

| # | Restored to (repo path) | sha256 | Provenance (`~/chtc_rescue/`) | Criterion |
|---|---|---|---|---|
| 1 | `output/pdf/paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715/paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715.json` | `8e057da082f9d81b308fc6563029ef9f1f309950b54191bd60bc47abfec95922` | `home/lanes_ablation12_r50_20260806_v1_chtc/source_locked.tar.gz` | (a) exact recorded sha — pin at `bundles.py:563` |
| 2 | `chtc/phase3_optuna/paper_i_table_i_audit_escalation.py` | `eb3d9c19d1d10b6b12542b46f5b417b238ba874ec23075f69a11fc4c3de1b10a` (git blob `b9be9e1f…`) | `home/rho1_redo_stage_20260609.tgz` | (a) blob-identical to `e838eb35` |
| 3 | `chtc/phase3_optuna/config/paper_i_comparator_spsa_calibration_v1_smoke.json` (+ `chtc/phase3_optuna/input/paper_i_comparator_spsa_calibration_v1_smoke/` 5 files: `acba2cfb…`, `90bf94aa…` ×2, `3bd0210e…` ×2) | `2fef684b627dc3cc575a87e75cc69d4ad64d3daa92449b3166b6f09f7cba9c7f` | `home/paper_i_comp_spsa_smoke_stage.tgz` (corroborated by `rho1_redo_stage_20260609.tgz`) | (b) unambiguous — byte-identical across 2 archives; sibling `.sub` matches the repo's tracked copy |
| 4 | `chtc/generic_time_dynamics_table/input/` — 75 files, 1.1 MB (key rows `paper_ii_class_tuned_cases_v1.json` `72c34b31895f8c266beda42e4bf930797a56f45a4efe5fef750adf7063c35f44`, `class_settings/paper_ii_class_settings_lock_v1.json` `563e36119a690428e0659364c4d190ab698e6b147c15441db07776ac8e2724a2`) | (75-row table in commit message) | `home/Holstein_paper_ii_production_v1/chtc/generic_time_dynamics_table/input/` | (b) schema-validated required era; byte-identical across both campaign copies |

`output/` is gitignored (`.gitignore:50`), so #1 was committed with `git add -f`; tracking it permanently is a user call. `chtc/phase3_optuna/config/` did not exist and was created. Restore #4 is purely additive (`git ls-tree HEAD` had 0 tracked files under `input/`).

### 4.1 Test-count effect (9 test files, one run each, `--continue-on-collection-errors`)

| Family | Before | After | Δ | Cause |
|---|---|---|---|---|
| **Aggregate (9 files)** | **85 F / 251 P** | **32 F / 304 P** | **+53 green** | — |
| `test_generic_static_benchmark.py` | 36 F / 91 P | 12 F / 115 P | +24 | escalation module (exactly the predicted 24/36) |
| `test_ra_adapt_bundles.py` | 15 F / 12 P | 2 F / 25 P | +13 | GLOBAL_SOURCE_LOCK |
| `test_paper_iii_compatibility_matrix.py` | 9 F / 0 P | 0 F / 9 P | +9 | class_tuned tree |
| `test_paper_iii_production_gate.py` | 4 F / 0 P | 0 F / 4 P | +4 | class_tuned tree |
| `test_hh_realtime_optuna.py` | 3 F / 117 P | 1 F / 119 P | +2 | class settings lock |
| spsa preflight | 8 F / 0 P | 7 F / 1 P | +1 | config + input tree |
| spsa outputs · chtc_time_dynamics_optuna · 8 package-manifest files | 8 F · 2 F/31 P · 26 F/51 P | unchanged | 0 | see corrections below |
| `test_generic_time_dynamics_table_inputs.py` | collection ERROR | collection ERROR | 0 | row 31 mask, §7 |

**Regression check:** `test_generic_dynamics_benchmark.py`, `test_generic_dynamics_ablation_matrix.py`, `test_dynamics_table_lock_contract.py`, `test_ra_adapt_lock_macro_receipts.py` → 1 F / 74 P. The single failure (`test_generic_avqds_t_incomplete_candidate_pool_writes_structured_skip`) was isolated in a throwaway worktree at pre-restore `fe7a74bc` and **fails identically there** — pre-existing, not caused by the restore. Worktree removed and pruned.

The 2 surviving `test_ra_adapt_bundles.py` failures are classification **row 6** (pinned route/protocol digest drift + a sha-literal assert), a different pre-existing family — not the lock.

### 4.2 Deliberate exclusions from the commit

| Held back | Size | Why |
|---|---|---|
| 5 `seed_artifacts_*` dirs in the same input tree (`seed_tracks_v2` 60M, `table_lock_available_phase3` 44M, `matched_fixed_mclachlan` 38M, `diagnostic_smoke_v1` 34M, `hh_seed_tracks_v1` 30M) | 206 MB of the tree's 207 MB | The repo already gitignores `chtc/**/input/seed_artifacts/` (`.gitignore:56`) but the pattern misses these **suffixed** names — a literal "restore the whole 141-file tree" would have put 206 MB of gitignored-in-spirit blobs into git history permanently. No affected test reads them. The 75 non-seed files delivered all the test value at 1.1 MB. |
| `raw_outputs/…_pre_repair_clusters9697391_9697725_20260825/` (43 files) | 155 MB | Gitignored (`.gitignore:48`); its receipt records **the path only, no per-file shas**; no test reads it. Recommend committing the 43-row sha table into the tracked `chtc/` receipt tree instead. |
| r07 outputs `run.json` + `action_receipt.json` | 31.9 MB | Gitignored raw_outputs; **zero suite references** to `raw_outputs/siam_nb4_full_apm_continue_r07`. Fully chain-verified evidence, but restoring it adds 31.9 MB of permanent history for zero test movement. |

### 4.3 Era divergence flagged inside restore #4

`chtc/generic_time_dynamics_table/input/class_settings/paper_ii_class_settings_smoke_lock_v1.json` = `6365177c27fd3068d97c0d3dbe98caeb34fd864c66e58b6618f21a2750236f46` (CHTC production era) **differs** from the copy committed at ancestor `5139669f` (`1b492e1f…`). It overwrites nothing (nothing under `input/` is tracked at HEAD) and is self-marked `manifest_role=local_smoke_only_not_paper_evidence`, but the choice is currently **untested** — `test_smoke_class_settings_lock_is_exact_three_strict_controller_classes` asserts on this file and is unreachable behind the row-31 mask. → **USER-DECISION:** keep CHTC-era (current) or swap to the `5139669f` ancestor.

---

## 5. Rung-2 t20 re-aggregation verdict (Task D)

**VERDICT: the published table is reproduced from the raw fetched trajectories BYTE-IDENTICALLY.**

| Step | Result |
|---|---|
| P0 tarball sha + extraction | **PASS** `d42aa3e7…`, 60 files |
| 24 extracted t20 files vs receipt shas | **PASS 24/24** |
| 12 t5 screen inputs | **FOUND 12/12 byte-identical** to `screen_run_sha256`. The receipt's screen paths (`output/paper_ii_staged/fetched/cluster_{9698201.final22,9697391.live,9697725.final25}`) do not exist locally; all 12 were located across `home/Holstein_paper_ii_staged_accuracy_euler_v1_20260825/raw_outputs/{paper_ii_staged_accuracy_euler_v1, …_pre_repair_clusters9697391_9697725_20260825}/` (mapping in `taskD/screen_map.json`) |
| Seed identity (`chtc/…_t20_resume_repair_d3301581_20260826/input/seed.json`, `7a36a2da…`) | **PASS 12/12 `reference_key`** — content-addressed identity recomputed (schema `paper_ii_exact_reference_store_v1`, propagator `exponential_midpoint_v1`, seed sha + per-run drive_profile + stitched 501-point grid + substeps=40). Confirms seed, grid and drive provenance exactly. |
| Exact reference store (6 keys `e5551b66/4b3c54b2/097c9750/f83fc25a/4826d7df/b0d1e60f`) | **NOT-FOUND** — `output/paper_ii_staged/exact_references` absent from repo *and* the whole rescue tree (matches RECOVERY-AUDIT "vanished"). Rebuilt all 6 from seed+drive via `ExactReferenceStore`; **identity keys regenerate identically**, but `reference.npz`/manifest **bytes** differ 12/12 — expected: floats recomputed on M4 BLAS, and the manifest embeds the data sha. M4-BLAS ulp precedent applies, **bytes only**. |
| Aggregation entry point | HEAD `pipelines/time_dynamics/diagnostics/provisional_t20_manuscript_assets.py` is **schema v4** with the direct raw-t20 table path retired; the committed receipt is **v3**, and the v3 module revision is **unrecoverable** (repo history has only restore commit `e0e6d6cd`; all 32 rescue-tree copies are v4). Re-aggregation therefore drove the surviving v3-equivalent library primitives directly (`physical_errors`, `ExactReferenceStore.get_or_build`, `load_terminal_resources`, t5+t20[1:] stitching, screen+t20 `accepted_internal_substep_count` sum) and reconstructed the v3 row formatting. Driver `taskD/reaggregate.py`, run under `ram_guard --limit-mb 8000` (exit 0, peak 231 MB). |
| **Rebuilt table vs committed `MATH/paper_details/generated/paper_ii_t20_provisional_frontier_table.tex`** | **BYTE PARITY — PASS** `cdb337a9cc81a5d84928ee887bea7af573696f9dea22f82de0ef7d2c42dc1b77`, the same sha the receipt records; all 12 published row lines reproduced exactly |

**Per-row float/byte split (all 12 rows PASS):** every integer — `time_step_count`, `n_theta`, graph tuple `C_G` — is **bit-exact 12/12** (e.g. row 1 apm/weakslow/`delta_theta_5e-5`/cut3e-06: N_step 471,508, N_θ 40, C_G (140,200,298); row 12 avqds/fastfast/`state_motion_5e-5`/cut3e-07: N_step 504,448, N_θ 100, C_G (316,408,672)). **0 of 48** float error metrics are bit-exact, but **all 48 agree within 1.17e-10 relative** worst case (row 10 avqds/strongfast/cut1e-06, `mean_infidelity` ≈ 1.17e-7 — the smallest infidelity in the table, hence the most reference-ulp-sensitive). Per-row worst relative diffs: 1.0e-11, 9.2e-12, 7.6e-12, 9.3e-12, 2.0e-11, 4.8e-11, 1.4e-11, 1.2e-11, 1.2e-11, **1.2e-10**, 1.7e-11, 9.5e-11.

**Interpretation:** the receipt's floats were computed against the original cluster-built exact references, which are lost. Rebuilding the deterministic-identity references on Apple/M4 BLAS shifts the propagated exact statevectors by ~1e-13…1e-10 relative, which fully accounts for the drift — orders of magnitude inside the published 2-significant-digit precision, and the table itself still comes out byte-identical. Byte-level mismatch is confined to the rebuilt `reference.npz`/manifest files. → the float-vs-byte parity split is a **USER-DECISION** for how it is worded in the manuscript's reproducibility note. Full numbers: `taskD/reaggregation_result.json`.

Task D artifacts (session scratchpad): `taskD/{reaggregate.py, reaggregation_result.json, paper_ii_t20_provisional_frontier_table.rebuilt.tex, screen_map.json, exact_references_rebuilt/}`.

---

## 6. Corrections the fetch authorities need (none applied — doc edits were out of scope)

| | Correction |
|---|---|
| (a) | **`chtc/HANDOFF_CHTC_RESCUE_FETCH_20260831.md` line 93** and **`docs/reports/EVIDENCE_PARITY_AUDIT_20260831.md` §3 item 2**: the Paper I L4 RA `result.tar.gz` is under campaign **`paper_i_l4_l3_matched_ra_20260826_v3`**, *not* `paper_i_l4_qiskit_proxy_extensions_20260825{,_v2,_v3}`. Those campaign dirs hold only the 8 per-arm `l4__*__9697719__N.tar.gz` staging transfers. |
| (b) | **HANDOFF line 94** and **audit §3 item 3**: `paper_ii_exchange_repair_cluster9698201_22.tar.gz` is listed as a 9.3 MB fetch target, but its own receipt (`chtc/paper_ii_staged_accuracy_euler_v1_exchange_budget_repair_20260825/submission_receipt.json`) records `remote_archive_state: deleted_after_verified_local_ingestion` with `remote_raw_results_state: preserved`. Mark it **deleted-by-design**; the expectation is stale. |
| (c) | **Audit §2.1 L4 row**: replace "unverifiable-here" — `./ra_l3_matched_k75/result.json` is now anchored at `e8662dd1551445f47891dd4b2e96d02df771a9ef208ee04db2711d5150fc3de5`. |
| (d) | **`agent_guidance/static-adapt/golden/ARCHIVES.sha256`**: add a `sixregime` section for the two surviving relatives (§1.4). Note the file's `b3:` / `b9:` prefixes are **root markers**, so it is not directly consumable by `shasum -c`. |
| (e) | **HANDOFF "Remote layout" anchor + classification row 9**: the recovery path for the SPSA configs is the staged `.tgz` archives, **not** the `~/Holstein_phase3_optuna_chtc` clone (which is empty). |
| (f) | **Classification row 8's "27 = 15 bundles + 12 package-manifest echoes" is wrong.** The lock is worth **13**; the 8 package test files sat at 26 F / 51 P both before and after. Running the validator directly shows they fail earlier, at `validate_package.py:89`, on a **missing per-package `source_locked.tar.gz`** — a separate data-missing item. |

---

## 7. What remains open

### 7.1 Blocking / high value

| Open item | Detail |
|---|---|
| **Off-machine backup of `~/chtc_rescue` (143 GB)** | **Still needed, and it is the single largest exposure.** The remote transfer dirs for items 3 and 5 are already empty; this tree is now the only copy of much of it, and the prior session already wrote stray files into it (§3.4). **USER-DECISION: destination.** |
| **Codex's `t20_screen_run_sha256.tsv` must not be committed as-is** | Corrected verdict 12/12 PASS is recorded in §2. Rewrite or discard the table before any commit. |
| **Row-31 mask on the class_tuned family** | `test/chtc/test_generic_time_dynamics_table_inputs.py` — the file that actually asserts on the tree restored in §4 — cannot be collected, before or after: `ModuleNotFoundError: chtc.generic_time_dynamics_table.build_paper_ii_hh_seed_track_inputs` at line 29 (module only on the `517eca4c` wip snapshot). This silently suppresses ~18 tests, including the ones that would validate the restored tree and the era choice in §4.3. **The "26" class_tuned target is not reachable until row 31 is grafted or the file retired.** Incomplete-graft item, not a data item. |
| **272 package `source_locked.tar.gz` missing; exactly 8 sha-recoverable** | All 33 `source_locked.tar.gz` in the rescue tree were hashed against every tracked `source_archive_manifest.json`. **272 package dirs have no local archive**; exactly **8** have an exact recorded-sha match: `lanes_ablation12_r50_20260806_v1_chtc` (`4e7f0f58…`, the very archive the GLOBAL_SOURCE_LOCK came from) plus 7 dirs sharing `690d54db…` from the staging gradient_phase0 campaign. Criterion-(a) clean, ~2 distinct archives, modest size. **None belongs to the failing package tests** (insertion12 / factorial48 / stationary_core / r70 / sixregime / plateau), so this is evidence preservation, not a test win — but it is the same anchor gap that already cost items 3 and 5, and these 8 are anchorable today. |
| **Anchor gap — 12 SIAM `incremental_exports` tarballs** | `home/Holstein_siam_nb4_source_446b75f3_20260828/incremental_exports/cluster_{9713618,9713835×2,9714484,9714524,9714525,9714675,9714816,9715075,9715147,9715270}/proc0/` (11) + `incoming/…frontier9714675_upload_20260829_v2.tar.gz`. These are the actual per-cluster SIAM raw trajectories — the substantive fetch — and **no sha for any of them exists in `chtc/`, `MATH/` or `docs/`.** Not restored (fails both criteria). |

### 7.2 User decisions queued

| Decision | Options |
|---|---|
| 18 b3mr (9674089) + 12 sixregime (9664325) tarballs | Unrecoverable. Retire the `ARCHIVES.sha256` rows / data-gate the dependent claims, or reconstruct by rerun. |
| Fig 2 tails (9675078) | Rebuild from the two surviving resume-egress checkpoints (2.6 GB, restore to `raw_outputs/ra_adapt_costexp_depthfix_sixregime_20260820_v2/resume_inputs/` recording shas at restore time), or retire. |
| `output/siam_screening/nb4_pilot_v1/` (row 7) | Rebuild-with-new-receipts from the sha-verified 9713113 raw inputs, vs retire/data-gate `test_siam_stable_bindings.py`. |
| Fig 3 raw (`l3_phase0`/`promotion_k15`) | Unrecoverable on CHTC as well as in git; local-regen-as-reconstruction stands as the parity audit's open call. Runner code survives inside the sealed packages. |
| 206 MB `seed_artifacts_*` | Commit the bytes, or (recommended) record a sha table in `chtc/` without the bytes. |
| 155 MB pre_repair tree + 31.9 MB r07 outputs | Commit the sha tables into `chtc/` (cheap, recommended) vs the bytes themselves. |
| `paper_ii_class_settings_smoke_lock_v1.json` era | Keep CHTC-era `6365177c…` (current) vs ancestor `5139669f` copy `1b492e1f…`. |
| 4 residual SPSA files, verified and available | `chtc/phase3_optuna/`: `run_paper_i_comparator_spsa_calibration_task.sh` `69e96e83b6252e56d3171601f5d4b6f934ac7d190510aaa4480c2e54596790bc` and `run_…task_apptainer.sh` `7aa025ef6bc9d9ef752842dbb55bb03dc575a56681390cfcc98000adebae2d1e` (byte-identical across **both** archives — the standard applied to the config that was restored); `requirements-chtc.txt` `a0dd00b4…` and `submit_paper_i_hh_geo_qeb_spsa_repair_v1_full.sub` `e5e01948…` (single-source, `rho1_redo_stage` only). Greenlight = one-command follow-up that should clear most of the residual 7 preflight failures. |
| `warm_start_schedule_lock.json` | In **no** archive (0/1,089). Regenerate or retire the preflight assertion at line 154. |
| Push `0c511b0a` | Commit is local and unpushed. |
| The 6 doc corrections in §6 | Can be applied in one follow-up pass. |

### 7.3 Housekeeping

- Delete/ignore the 4 zero-byte `live_tails_9675078/*.out` placeholders (§3.4) — they are not evidence, and they mean the tree is no longer a pristine remote mirror.
- Note the 2 truncated June-era archives (§1.5) in any future integrity sweep; the Fig-2 symlink is a path artifact, not data loss.
