# Paper-III fixed-pool termination evidence — 2026-08-28

## Locked implementation

- Branch: `paper-ii-exchange-selector`
- Implementation commit: `bd4788374aab527cd343b147b578b01d75b47931`
- Commit subject: `paper-iii: lock fixed-pool completion and comparator identity`
- Canonical method: `record_first_adaptive_qse`
- Canonical completion route: `selected_qse_fixed_pool`
- Selection-policy digest: `4e0b76e200821e9cfa4ba9b8a522995bb3a70a17402329bc5d51dc0d35ee5300`

The canonical selected-QSE route does not invoke Davidson. Pool exhaustion and
uncertified target coverage are terminal outcomes of that method. Exact-reference
errors are computed only in the post-run audit and cannot change the online
completion status.

The separately declared `selected_qse_then_davidson` route remains an explicit
hybrid ablation boundary. If it is enabled later, its receipt and cost include
both selected-QSE and Davidson stages. It is not part of the canonical campaign.

## Locked local artifacts

The JSON artifacts are under the repository's ignored `output/` tree. Their
content hashes, source revision, policy digest, and source-path hashes are the
reproducibility anchors.

| Regime set | Artifact | SHA-256 | Cells | Certified | Uncertified |
|---|---|---|---:|---:|---:|
| nph1 | `output/diagnostics/paper_iii_selected_qse_fixed_pool_v1/failure_evidence_nph1.json` | `8db7c5afecb4d177bdc7e05918e4fadfd1c99475e52f903c5e60e10d6e7e3c90` | 6 | 6 | 0 |
| nph3 | `output/diagnostics/paper_iii_selected_qse_fixed_pool_v1/failure_evidence_nph3.json` | `3757b6fcf296c6082ce99b665315bda3dff0402d00e70f0ce2213071a464eb33` | 6 | 3 | 3 |

Both artifacts use schema
`paper_iii_selected_qse_fixed_pool_failure_evidence_v2`, report
`davidson_invoked: false`, and revalidate with `require_clean_sources=True`.
Their provenance receipts state that all relevant source paths were clean and
unchanged during acquisition at the implementation commit above. This
cleanliness claim is scoped to the eight source paths recorded in each receipt,
not to the entire shared checkout.

## Observed completion outcomes

`max residual` is the online physical Ritz-residual maximum on the actual
exchange-maintained terminal support. `terminal exact error` is a post-run,
index-matched diagnostic only.
`selected` is the number of selection events; it must not be confused with the
terminal support cardinality after atomic exchange.

### nph1

| Regime | Status | Selected | Traced terminal k | Terminal support | Retained frame | Max residual | Terminal exact error | Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| weak_weak | `TARGET_WINDOW_CERTIFIED` | 15 | 15 | 15 | 15 | 1.131e-12 | 7.305e-14 | 3.96 |
| intermediate_weak | `TARGET_WINDOW_CERTIFIED` | 16 | 16 | 16 | 15 | 4.953e-10 | 5.896e-11 | 4.13 |
| strong_weak_u8 | `TARGET_WINDOW_CERTIFIED` | 16 | 16 | 15 | 15 | 1.858e-08 | 4.320e-09 | 3.66 |
| weak_strong | `TARGET_WINDOW_CERTIFIED` | 15 | 15 | 15 | 15 | 2.745e-13 | 1.432e-14 | 4.72 |
| intermediate_strong | `TARGET_WINDOW_CERTIFIED` | 16 | 16 | 16 | 15 | 5.252e-13 | 1.954e-14 | 4.57 |
| strong_strong_u8 | `TARGET_WINDOW_CERTIFIED` | 15 | 15 | 15 | 15 | 3.169e-12 | 1.202e-12 | 3.63 |

### nph3

| Regime | Status | Selected | Traced terminal k | Terminal support | Retained frame | Max residual | Terminal exact error | Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| weak_weak | `TARGET_WINDOW_NOT_CERTIFIED` | 71 | 64 | 64 | 40 | 1.242e-03 | 3.193e-07 | 186.26 |
| intermediate_weak | `TARGET_WINDOW_CERTIFIED` | 53 | 53 | 53 | 38 | 7.242e-04 | 9.583e-08 | 223.85 |
| strong_weak_u8 | `TARGET_WINDOW_CERTIFIED` | 88 | 64 | 57 | 33 | 1.161e-04 | 2.160e-09 | 420.11 |
| weak_strong | `POOL_EXHAUSTED` | 63 | 63 | 63 | 51 | 2.469e-01 | 1.799e-02 | 258.68 |
| intermediate_strong | `POOL_EXHAUSTED` | 57 | 57 | 56 | 51 | 2.324e-01 | 1.154e-02 | 218.06 |
| strong_strong_u8 | `TARGET_WINDOW_CERTIFIED` | 73 | 64 | 60 | 38 | 8.422e-04 | 1.140e-07 | 555.61 |

Interpretation locked to these data:

- The nph1 fixed pool certifies all six cells.
- The nph3 fixed pool has a 50% uncertified rate: two genuine
  `POOL_EXHAUSTED` cells and one dimension-cap
  `TARGET_WINDOW_NOT_CERTIFIED` cell.
- The two nph3 `strong_*_u8` cells certify their actual supports at the
  acquisition cap without reaching selector termination. They must not be
  described as selector convergence.
- `POOL_EXHAUSTED` means exhaustion of the usable metric-novel records, not
  literal selection of every raw pool record. Both exhausted cells have a
  closed energy window but residuals above threshold.
- The `weak_weak` nph3 post-run exact error is small even though its online
  residual certificate narrowly fails (1.242e-3 versus the 1e-3 threshold).
  This is expected and is evidence that the exact audit does not rewrite the
  declared online completion boundary.
- These data establish where fixed-pool selected-QSE fails under the locked
  cap and stride. They do not yet establish that the separately benchmarked
  hybrid is cost-effective; that decision requires matched-cost comparison
  with direct Davidson.

## Safeguards and behavior lock

The completion certificate is recomputed on the actual terminal support after
exchange. It uses supported-overlap truncation, retained-rank checks, physical
full-state Ritz residuals, and pressure from every novel unused fixed-pool
record. It does not use exact roots or exact errors.

The support-producing modules were unchanged relative to the pre-increment
commit:

- `pipelines/qse_spectra/record_selection.py`
- `pipelines/qse_spectra/exchange_maintenance.py`
- `pipelines/qse_spectra/core.py`
- `pipelines/qse_spectra/paper_iii_problem.py`

The 48 stored nph1 `(regime, k, support_indices)` rows are bit-identical to
`output/diagnostics/paper_iii_record_first_adaptive_qse_v2/growth_trace_nph1.json`.
Both canonicalized row sets have SHA-256
`f29647a2e1743502820c911abaf1e01b5bf6054d2e8925cb7924f2bc622c5d52`.

## Comparator and route identity

Direct Davidson remains a distinct comparator with construction policy
`direct_davidson_fixed_lr_plus_r_plus_1_low_diagonal_sector_v1`. Its target-sector
probes are ordered deterministically by Hamiltonian diagonal and then basis
index; all initialized probes are charged to comparator cost. A seed block that
cannot fit the declared cap fails closed as
`DAVIDSON_INITIALIZATION_CAP_INSUFFICIENT`.

Campaign, cell-key, action, action-file, package, envelope, attempt, ledger,
qualification, and results-contract identities were migrated to v2. Legacy
hybrid identity `record_first_adaptive_qse_v2` is rejected rather than silently
normalized to canonical selected-QSE.

## Verification

- Focused completion/campaign/contract suite: `112 passed`.
- Wider static-QSE/selector/exchange suite: `227 passed, 1 failed`; the sole
  failure is the pre-existing forbidden import in unchanged
  `pipelines/qse_spectra/siam_static_adapter.py`.
- Selector, exchange-maintenance, core, problem-provider, and SIAM-adapter
  files were all diff-clean relative to the implementation parent.
- `git diff --check` and `git diff --cached --check` passed before the code
  commit.
- No Paper-III manuscript source was edited or staged in this increment.

The required pre-commit `git pull --rebase` was attempted. Git refused because
the shared checkout contained staged and unstaged work. No stash, reset, or
other-session cleanup was performed.
