# Page-12 global-singleton gradient-Phase-0 route

This file is the compact execution and preservation pointer for the route shown
on page 12 of the evolving Paper-I results PDF. It is a named candidate route,
not an alias for the ordinary `run_ra_adapt(problem)` default.

## Exact identity

- Package:
  `chtc/paper_i_ra_adapt_repair_20260727/paper_i_ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_phase23_no_lanes_cap24_tau1em4_r50_20260807_v1_chtc`
- Package ID:
  `paper_i_ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_phase23_no_lanes_cap24_tau1em4_r50_20260807_v1_chtc`
- Route ID:
  `ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_plateau`
- Route-contract SHA-256:
  `9811652b332b592bee048a8e5f3048972256abae186921ed7efea52bfd5f3dd8`
- Execution-plan canonical SHA-256:
  `1758a5969638397d0433c1802a901a8b0475abf49f49538acb1f4722a28ea7e7`
- Source-archive SHA-256:
  `690d54dbf5bafcaaf974dc11339ed927cb7f5d117265ed51adbb811785740762`

The route initializes the full executable singleton population. Phase 0 ranks
that population by absolute standard-ADAPT gradient and retains 24 candidates;
Phase 0 uses no Fubini metric and no resource cost. The retained singletons then
pass through singleton Phase I, identity-preserving singleton Phase II, and
singleton Phase III. Phase I uses the structural proxy; Phases II and III use
signed full-trial-ansatz Qiskit transpilation deltas with no physical lanes.
The controller uses stationary source response, Powell-200, seed 7, windowed
refits, and commutation-reduced cumulative plateau insertion with threshold
`1e-4`, patience 1, and no hysteresis. The six-regime source lock selects
`nph=3` for the weak sector and `nph=7` for the strong sector.

Do not reconstruct this route from prose. Execute from the sealed package or
materialize a new source-locked revision whose typed route receipt is compared
against the route-contract digest above.

## Current evidence state

Cluster `9605157` produced five authenticated round-50 archives. The
strong--strong cell is still an incomplete live/running source and is not part
of the completed archive set. The Page-12 adapter explicitly records
`paper_evidence_adopted=false`; this document does not promote the route or
change the canonical default.

The report pointers that must survive local cleanup are:

- `output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_partial_progress.pdf`
  — 3,562,852 bytes, SHA-256
  `838ca3d63a677425c9d03fbec1f9376c007f3b5d1902587d5586e390848c711b`.
- `output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_partial_progress_provenance.json`
  — 1,216,174 bytes, SHA-256
  `cf5010ae8cac4e13cd146384a7a88d9a57da32552d3d758b952a2c5194ebcb27`.
- `output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_global_singleton_gradient_phase0_page12_adapter.json`
  — 136,905 bytes, SHA-256
  `76a8cfe59a7a955ac8d097eba828d6737f9478f093608d7d33e13bee98fe0647`.
- The corresponding Page-12 PDF and PNG in the same directory — respectively
  SHA-256
  `b548b75b3fec70c0977bb1f741cc76ad29a301b7d2d90d2781d2dd66e4f70c4f`
  and
  `ea00e13829bb971296a472852afe8ab029d3cb7fb944fa0f009759f280967b62`.

## Preserved Page-12 archives

All five files below were re-read on 2026-08-10. Exact size and SHA-256 matched
the Page-12 adapter, and every gzip/tar stream was readable.

| Regime | Local archive | Bytes | SHA-256 |
|---|---|---:|---|
| weak--weak | `chtc/paper_i_ra_adapt_repair_20260727/retrieved_phase0_completed_20260809/9605157.0_full.tar.gz` | 401,629,564 | `2d7200a235a98de13ea8c430a5e1524ae7a65d53884b94111a621a728c9738c6` |
| intermediate--weak | `chtc/paper_i_ra_adapt_repair_20260727/retrieved_phase0_completed_20260809/9605157.1_full.tar.gz` | 410,858,501 | `94df41a0b977618ca47cf2f26c2787b94276965c7ce767e12e5b88388bee64b8` |
| strong--weak | `chtc/paper_i_ra_adapt_repair_20260727/retrieved_phase0_completed_20260809/9605157.2_full.tar.gz` | 379,565,071 | `1b43d275067c5293f61dce8aaebe21f6445bdff6ca1b0f04bfc4b3b3a5a1eb7e` |
| weak--strong | `chtc/paper_i_ra_adapt_repair_20260727/retrieved_phase0_completed_20260809/9605157.3_full.tar.gz` | 1,099,219,486 | `ab3096dd82ae499fced09ebbcc462e93fa83195a1183eb532366a4a79cd19429` |
| intermediate--strong | `chtc/paper_i_ra_adapt_repair_20260727/retrieved_phase0_completed_20260809/9605157.4_full.tar.gz` | 1,169,324,852 | `703d805f27b179f483329ea8bcb8b78bd3c1a28c56f15cd3335e14ccfc896978` |

Keep the five adjacent `*_retrieval_receipt.json` and
`*_completed_report_adapter.json` files with these archives.

## Preserved singleton Append-ADAPT round-70 archives

All six files below were re-read on 2026-08-10. Exact size and SHA-256 matched
the Page-12 adapter, and every gzip/tar stream was readable.

| Regime | Local archive | Bytes | SHA-256 |
|---|---|---:|---|
| weak--weak | `chtc/paper_i_ra_adapt_repair_20260727/retrieved_chtc_20260801_append_r70_weak_weak/r70_fresh__weak_weak__nph3__append_singleton__cluster_9398375__proc_1.tar.gz` | 202,606,149 | `e0ff609149ea5035ae9c3936ea2631e98cd3360cde684d31dbcc19a98cdfc050` |
| intermediate--weak | `chtc/paper_i_ra_adapt_repair_20260727/retrieved_chtc_20260801_append_r70_intermediate_weak/r70_fresh__intermediate_weak__nph3__append_singleton__cluster_9398375__proc_3.tar.gz` | 235,895,048 | `62d7ddbc08ef073e9b11eaef05c5176d80a97dee13fa57834773b5c682c8b694` |
| strong--weak | `chtc/paper_i_ra_adapt_repair_20260727/retrieved_chtc_20260731_append_r70_strong_weak/r70_fresh__strong_weak_u8__nph3__append_singleton__cluster_9398375__proc_5.tar.gz` | 99,318,690 | `0b7faa1664e10adf2b596b313da769726e9a2ab8ebd5f72aab1b1cdf1f12435f` |
| weak--strong | `chtc/paper_i_ra_adapt_repair_20260727/retrieved_chtc_20260802_append_r70_remaining/r70_fresh__weak_strong__nph7__append_singleton__cluster_9398375__proc_7.tar.gz` | 2,139,963,100 | `1f2fb266bbfbc9c6849bb8dda4fe908cc2ddf92d3e7dbf32440ebfe5c675a86f` |
| intermediate--strong | `chtc/paper_i_ra_adapt_repair_20260727/retrieved_chtc_20260802_append_r70_remaining/r70_fresh__intermediate_strong__nph7__append_singleton__cluster_9398375__proc_9.tar.gz` | 2,164,046,361 | `f31eda051f0e25bdaf7347b3cb9bbf838c6e067ef11571d2ce05cd7232ee1e32` |
| strong--strong | `chtc/paper_i_ra_adapt_repair_20260727/retrieved_chtc_20260731_append_r70_strong_strong/r70_fresh__strong_strong_u8__nph7__append_singleton__cluster_9398375__proc_11.tar.gz` | 954,111,541 | `6028ddfdad557d5f28c4c3539ee7c957cd249efba1d5a3654a3625d5b4c5bbf6` |

The Append report adapter is
`output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_ra_append_singleton_r70_page6_adapter.json`
(SHA-256
`21ae30bff8c9810d8e0670e162dfb0565dad2f3e8a406e235dc5ad34ba9e63ea`).

## Default decision

Keep this route as an explicit named candidate until the strong--strong cell is
complete and the user decides whether to promote it. The robust near-term seam
is the sealed package plus this pointer; changing the silent canonical default
before completing that evidence would mix a partial experimental route into
ordinary RA-ADAPT calls.

## Local cleanup receipt

After the preservation checks above passed, the 2026-08-10 cleanup increased
Data-volume free space from approximately 13.40 GiB to 108.21 GiB. The first
conservative pass reached 60.37 GiB; a second preservation-manifest-driven pass
removed additional obsolete operational payloads and generated recovery data.
A final post-cleanup check verified all 7,095 protected manifest entries with
zero missing paths, type drift, or size drift, including the exact recorded
sizes of all five Page-12 archives and all six singleton Append-ADAPT round-70
archives above.

The machine-readable keep-set and its reproducible builder are under
`artifacts/storage_cleanup_20260810/`. The manifest protects 67.45 GiB across
source, Git history, manuscripts, PDFs, the recursively resolved evolving-PDF
dependency graph, current packages and continuations, Page-12, and
Append-ADAPT round-70. Its canonical SHA-256 is
`d2dc1befd59ae1a3eab597280af7f6b16a11b785aaa8008e99e11bee3c4d0719`.

The cleanup removed 19 intermediate fetch-snapshot directories named
`heartbeat_*`, `priority_*`, or `status_*` under these three roots:

- `raw_outputs/chtc_fetch_paper_i_hh_r50_full48_20260719/`;
- `raw_outputs/chtc_fetch_paper_i_hh_sr_20260720/`;
- `raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/`.

Those exact snapshots occupied 14.08 GiB, had no hardlinks outside the deletion
set, and were not any completed Page-12 or Append-ADAPT archive listed above.
The cleanup then:

- removed the disposable repository `tmp/` contents and Python/test caches;
- hardlink-deduplicated byte-identical continuation inputs while preserving
  every pathname and byte binding;
- removed seven failed or derived raw-output trees after confirming that no
  completed evidence or external hardlink depended on them;
- removed superseded CHTC safety snapshots only after hashing their later
  authenticated successors, and removed six large extracted ledger/checkpoint
  copies only after streaming identical members from their retained parent
  archives;
- removed `/private/tmp/page12_phase0_audit.8p8t9q9y` only after all 20 files
  matched preserved Page-12 archive members by size and SHA-256; and
- cleared rebuildable package, browser-update, extension, and Chrome
  on-device-model caches. Chrome may download the model again if that feature
  is used;
- removed four extracted Page-10 continuation checkpoint/ledger files only
  after their SHA-256 values matched the authenticated, pointer-closed weak--
  strong and intermediate--strong compact resume archives; and
- cleared additional rebuildable dependency, Conda-package, obsolete VS Code
  extension, speech/media-analysis, and Codex runtime caches.

The second pass additionally:

- removed 57 obsolete checkpoint/ledger/current operational files from older
  local diagnostics, reclaiming 15,959,101,440 allocated bytes, while retaining
  five exact paths referenced by the evolving report and all compact results,
  summaries, trajectories, and receipts;
- removed seven high-confidence obsolete raw-output trees with no current
  report or paper references, reclaiming 11,336,524 KiB; and
- removed 13 generated-data, environment, raw-output, and CHTC roots from the
  old iCloud recovery checkout, reclaiming 24,979,776 KiB while retaining its
  source, `MATH/`, documentation, and `output/pdf/` paper materials.

The 9.1-GiB `/private/tmp/claude-501` scientific scratch tree was deliberately
preserved because it contains unique local checkpoints/results and is still
referenced by TeX `\graphicspath` entries. Final retrieval directories,
protected archives, report adapters, manifests, and source-locked packages
were not removed. Deleted intermediate snapshots and failed-output copies are
not recoverable locally. Generated data removed from the old iCloud recovery
checkout is likewise not locally recoverable from that checkout. All completed
or authenticated evidence listed above remains intact.
