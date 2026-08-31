# Results — Paper-I L=3 fixed-width noise rerun with Phase 0

All three RA-Append cells completed 20 accepted rounds.  Each protocol records
Phase 0 as active, omits the adaptive Phase-I--III shortlist policy, and fixes
the subsequent widths at `24/12/1`.  Every accepted-round receipt contains a
24-member Phase-0 gradient population and a 24-member Phase-0 shortlist, so
Phase 0 was executed but its cap was nonrestrictive.

## Recorded prefixes

The estimator values below are the runtime's raw `S_alg` occurrence counts.
They are retained for provenance and must undergo the established
phase-reuse recount before entering the manuscript.

| Noise | k | Exact diagnostic error | N2q | D2q | Dc | raw S_alg |
|---|---:|---:|---:|---:|---:|---:|
| low | 10 | 5.617706338334e-06 | 108 | 92 | 383 | 7,178 |
| low | 20 | 1.023920769461e-05 | 132 | 106 | 439 | 22,603 |
| high | 10 | 5.649595215811e-04 | 108 | 92 | 383 | 6,681 |
| high | 20 | 6.577203750497e-04 | 216 | 189 | 781 | 23,571 |
| extreme | 10 | 1.432355136856e-01 | 115 | 98 | 392 | 5,631 |
| extreme | 20 | 1.168826301968e-01 | 148 | 114 | 454 | 21,644 |

The exact diagnostic trajectory is not monotone because candidate gradients
and Powell objectives are noisy while the recorded diagnostic energy is exact.
No intermediate prefix should be substituted for the declared k=10 or k=20
comparison point.

## Evidence checks

- Accepted rounds are contiguous from 1 through 20 in every cell.
- Each cell contains 20 explicit Phase-0 population receipts, all with
  `population_count = shortlist_count = 24`.
- `phase0_gradient_screen = true` and
  `adaptive_phase_shortlists = false` in every serialized request.
- The route contract contains Phase-I/II widths 24/12 and no
  `ra_phase123_shortlist_policy`; singleton Phase III remains fixed at one.
- Accepted refit and comparator constants are matched at Powell
  `xtol=1e-5`, `ftol=1e-12`.
- Targeted protocol and tolerance guards passed: 8 tests.

## Source lock

- Git head: `618b3bf605720cd248ea7ea96af16b1c283d949b`
- Branch: `codex/noise-nonpositive-shortlist-20260824`
- Dirty source diff SHA-256:
  `80068ac191bba8231d2215f6b0e94a705133ee10db6322cd57376f0bb4f59064`

The Git diff does not cover untracked files, so the critical source-file hashes
are recorded separately:

- `run_cell.py`: `403366b9ca021213f14b5dc76e258986fa149f93fa1fd3e66ed6ee75e1f57293`
- `marginal_qiskit_cost.py`: `0dd300ba8e3dc6b83722fd1ab4252662f67fb2a54bc5cd73d4bf36ca8a3cdc3e`
- `appendix_demos.py`: `13efe33639a7d6ad055fb388d5c682278418b8ffa4e7d52671580e6404bd1ba3`
- `engine.py`: `23f4391f3aa2f034b96bd70b698fe2ac38fa08542fd505a758f2983f9d2a0f22`
- `engine_support.py`: `d2f417fae79b47c277b4bfe80c6206cbd78a02d339f8cf9c2ce653a29c45ad14`
- `append.py`: `f46ea592acb5659cb5015f1a4d98a58415e123d5d786da80786723c9b16af964`
- `contracts.py`: `0c01fca2e3abfd9dffd717ce254c50cc51e9669b7c52b5497c37358aa300d5fe`

## Result-file SHA-256

- low: `1eb22babdf294043a78dd4287f5ca8e2a8d2bd6b3fe2b6617f92121284071c25`
- high: `6ef0e1961ee1946d371284877643091e3f93b17e5e080a0cc23f72b0c4ee619d`
- extreme: `5b337f196a4fe2b9f370679b51452b1d565fc47bd0c5132aa43661f50479d27e`

## Promotion status

The RA cells are complete, but the Paper-I figure and table are not updated by
this run.  Before promotion, apply the phase-reuse estimator recount and audit
the AVQE noise comparator under its intended no-cache callback convention.
Then regenerate the paired trajectories and every dependent comparison claim.
