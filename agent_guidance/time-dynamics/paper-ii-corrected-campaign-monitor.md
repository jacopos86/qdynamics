## 20260905T214114Z — 15-minute post-submission heartbeat (manual)

- `9720203`: {'running': 11}
- `9720204`: not in queue
- `9720205`: not in queue
- `9720069`: {'running': 1}

Decisions: none required. Fetches: none.

## 20260905T214344Z — SIAM r00 screens: startup failure and once-only resubmission

- `9720204.0` and `9720205.0` completed ExitCode 1 in 0.1–0.2 min, 0 MB: `_verify_source_tree_contract` found `pipelines/artifacts/__init__.py` missing on the remote. Cause: the staging rsync's `--exclude 'artifacts'` (from the §3 recipe, intended for output dirs) also dropped the source package `pipelines/artifacts/`. The HH package is unaffected (it extracts its own source tarball).
- Fix: re-uploaded `pipelines/artifacts/`; verified every file in both packages' source-tree contracts is present on the remote. Resubmitted each once with an identical submit description (§7 input-transfer class; attempt 2 of 2). Receipts: `output/chtc_monitor/resource_retry_9720204_0.json`, `resource_retry_9720205_0.json`.
- resubmitted:   siam_nb4_qc_avqds_screen_r00_cut1em02_so: cluster 9720211
- resubmitted:   siam_nb4_qc_full_apm_screen_r00_cut1em02: cluster 9720212
- `9720203`: 11/11 running; `.0` image 0.87→1.44 GB over 10 min (executing).

## 20260905T214824Z — SIAM r00 second startup failure: source-lock mismatch (no further retry)

- `9720211.0`, `9720212.0`: ExitCode 1 at 16 s / 11 s. `_verify_source_tree_contract` line 531: `SIAM source-tree bytes differ: pipelines/time_dynamics/campaigns/paper_ii_corrected_exchange_cells_v1.json`. The contract hashes file contents; the packages are locked to `a3d1cebe` while the uploaded tree is the branch head (campaign files changed at `a43bcd86`). Attempt 2 of 2 exhausted; not resubmitted. Fix is a rebuild of both packages on the uploaded tree's revision, not a retry.
## 20260905T214927Z — SIAM r00 rebuilt on the uploaded revision; remote pre-flight clean

- Packages `chtc/siam_nb4_qc_avqds_screen_r00_cut1em02_source_812d20c3_20260905` and `..._full_apm_..._source_812d20c3_20260905` (`46df0ab1`), source-locked to `812d20c3` whose `pipelines/src` equals the tree on ap2001. Exchange arm on `siam_nb4_pilot_exchange_v1` (frontier 8), AVQDS default.
- Uploaded. All 809 `(path, sha256)` rows of the source-tree contract verified on the remote: 0 mismatches. LIFECYCLE_OK both; dry-run 1 job each. Awaiting user "submit".
- `9720203`: 11 running. `9720069.3`: running.
## 20260905T215118Z — SIAM r00 rebuilt packages submitted (standing authorization)

- AVQDS screen → `9720216`; exchange screen (frontier 8) → `9720217`. Source lock `812d20c3`, byte contract pre-verified on the remote. These replace `9720204/9720211` and `9720205/9720212`, all of which failed at startup and hold no outputs.
