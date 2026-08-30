# Paper III static-QSE numerical source anchor — 2026-08-29

## Authority status

The 13-page source-anchor PDF below supersedes the earlier three-page static
production report at the legacy `paper_iii_static_qse_production_20260829_v1`
path. The old directory is preserved as historical output and is not a
numerical authority for the manuscript revision.

- Branch: `paper-ii-exchange-selector`
- Static scope: Hubbard--Holstein, Peierls--Hubbard, and finite-box SIAM
- Target: six lowest excited roots in the exact-seed sector
- Dynamics: excluded and deferred to a separate paper
- Proposed method: cost-weighted operator-selected QSE (OS-QSE)
- Selector:
  `full_pencil_ky_fan_cost_ratio_novelty_floor_no_reward_v1`
- Hubbard--Holstein pool:
  `hh_full_meta_plus_encoded_boson_pauli_completion_v1`
- Implicit Davidson fallback: absent
- Numerical authority:
  `output/pdf/paper_iii_static_qse_source_anchor_20260829_v2/paper_iii_static_qse_source_anchor.pdf`
- Machine-readable authority:
  `output/pdf/paper_iii_static_qse_source_anchor_20260829_v2/manifest.json`

The results PDF is the source anchor for numerical values, evidence hashes,
route definitions, crossing policy, accounting boundaries, complete matrices,
failures, mechanism evidence, and spectral audits. Paper III supplies the
expanded scientific derivation, interpretation, and framing; it must not
silently replace source-anchor values with legacy generated fragments.

## Production method identity

OS-QSE evaluates the full supported generalized pencil for each admissible
candidate. Its acquisition signal is the proposed-pencil Ky--Fan gain
`Delta` divided by normalized graph-span cost `K`; novelty is an admissibility
floor only. Gain normalization is round-local, while the cost normalization is
fixed once from the initially eligible pool and does not change as candidates
are consumed. The production route preserves:

- double Gram--Schmidt residualization against the retained frame;
- supported-overlap truncation and retained-rank checks;
- the pre-six-root common `h11` ceiling in the Ky--Fan objective;
- the residual-plus-`h11` pool-relative stopping rule;
- deterministic candidate order and tie handling;
- atomic singleton `1<->1` exchange with whole-patch pencil re-solve and
  rejection leaving the support unchanged;
- structured failure outcomes rather than a silent method change.

Direct quantum Davidson, single-seed real-time Krylov, and six-probe
block-seeded real-time Krylov are independent comparator routes. Complete
fixed-class QSE is the conventional static baseline. The two Krylov routes are
never combined case by case into an oracle method.

## Publication crossing correction

The publication definition is the earliest successful prefix in each route's
declared growth order. Every error and resource coordinate is charged at that
same prefix. The sealed Hubbard--Holstein campaign summaries instead selected
the minimum-`K` successful rung. Reconstructing from the immutable full traces
changes three selected-QSE entries:

| Case | target | sealed minimum-`K` rung | publication first prefix | achieved error |
|---|---:|---:|---:|---:|
| `weak_strong_nph1` | `1e-4` | `k=15`, `K=24` | `k=14`, `K=26` | `1.33e-15` |
| `weak_strong_nph1` | `1e-6` | `k=15`, `K=24` | `k=14`, `K=26` | `1.33e-15` |
| `weak_strong_nph3` | `1e-4` | `k=46`, `K=428` | `k=42`, `K=484` | `7.55e-5` |

No scientific route was rerun or mutated. The report manifest records all 48
Hubbard--Holstein route/target reconciliation receipts. Headline counts remain
unchanged.

## Locked findings

- OS-QSE reaches the six-root `1e-6` target in all 12 Hubbard--Holstein cases,
  all three Peierls--Hubbard regimes, and the ten-qubit SIAM `N_b=4` case:
  16/16 total.
- Direct quantum Davidson also reaches 16/16 as an independently initialized
  comparator.
- At each route's publication first crossing, OS-QSE has lower summed Qiskit
  two-qubit structural annotation than Davidson in all 16 cells.
- In Hubbard--Holstein, OS-QSE has `K <= K_DQ` in 10/12 cells at `1e-4` and
  12/12 at `1e-6`; its Qiskit annotation is no larger in 11/12 and 12/12;
  its declared logical queries are no larger in 12/12 at both targets.
- The SIAM transfer case reaches error `5.07e-7` with OS-QSE versus `9.60e-7`
  with Davidson, at Qiskit annotations 622 versus 2,142 two-qubit gates and
  4,666 versus 46,008 logical estimator queries.
- Single-seed RTK reaches 10/16. Six-probe block RTK reaches 11/16. Their
  success sets are complementary, demonstrating initialization-dependent
  spectral support rather than a universal Krylov ranking.
- Every one of the five block-RTK failures has a target direction with
  post-run source-block weight below `1e-20`; extending the same fixed-source
  time grid cannot create that missing component.
- Under the same current selector, seed, overlap cutoff, dimension cap,
  stopping rule, and no-fallback boundary, the base Hubbard--Holstein macro
  pool reaches 10/12 cells at `1e-4`; encoded-boson completion reaches 12/12.
  Completion-family records occur in every completion-pool terminal support.

## Static spectral audit

All currently reported spectra are terminal-manifold audits, not
first-crossing spectra. Hubbard--Holstein OS-QSE has median normalized
broadened-spectrum `L1` error `1.73e-9` and maximum `2.00e-5` across site-0
density and displacement probes. Complete fixed LR-QSE can reproduce selected
bright spectra while missing the full six-root energy window; spectral
response recovery and worst-root coverage are therefore reported separately.

The source anchor does not pool incompatible postprocessing conventions:

- Hubbard--Holstein uses 801 points on
  `[0, max(1, omega_6_exact + 6 eta)]`, `eta=0.05`, trapezoidal integration,
  denominator floor `1e-15`, and transition-strength floor `1e-8`.
- Peierls/SIAM and RTK use 801 points, `eta=0.05`, an upper limit equal to the
  largest exact or approximate pole plus `0.5`, trapezoidal integration, and
  undefined normalized error when exact integrated weight is at most `1e-12`.

## Resource/accounting boundary

The evidence distinguishes three coordinates:

- `K`: online deterministic Marrakesh graph-span cost on the surviving or
  admitted support;
- Qiskit `N2q/D2q`: sums of independently transpiled direction/action
  circuit-structure annotations, with summed depth an aggregate rather than
  one executable-circuit depth;
- logical estimator queries: route-local counts, not shots.

OS-QSE/fixed QSE use a qubit-wise-commuting record-pencil convention;
Davidson/RTK use an ungrouped synthesized-basis convention. Candidate
screening, rejected proposals, exchange-patch evaluation, transition
observables, and measurement reuse are not converted into a common cumulative
hardware-work currency. The report therefore supports compact admitted-support
claims under declared structural coordinates, not total-work, shot, runtime,
or end-to-end hardware claims.

The Hubbard--Holstein traces expose rather than hide this boundary: the source
anchor tabulates every route's online rounds, full proposed-pencil evaluations,
proposed Hamiltonian row elements, terminal support, terminal `K`, and
evaluated exchange locations.

## Remaining evidence and claim limits

Explicitly unresolved:

- current-policy cost-discount-off ablation;
- current-policy exchange-off ablation and accepted-patch ledger;
- a cumulative acquisition-work ledger under one common implementation model;
- Hamiltonian-power QKSD implementation;
- whether qEOM/q-sc-EOM, VQD, and SSVQE remain literature-only scope or gain
  numerical comparators.

Finite-shot work is not planned for this static exact/noiseless paper. The
current evidence does not establish approximate-seed robustness, physical
bosonic-cutoff convergence, universal superiority over Krylov methods,
systematic SIAM generality beyond one `N_b=4` case, or time-dynamics behavior.

## Source-anchor artifacts

Generated artifacts remain in the repository's ignored `output/` tree; the
tracked builder and this evidence record are the reproducible contract.

| Artifact | SHA-256 |
|---|---|
| `pipelines/reporting/build_paper_iii_static_resource_enriched_pdf.py` | `ab510461e0ea29e2f8f1c9b6b39a6adbd396658b0d969f92440e680b5eeed0c4` |
| `test/test_paper_iii_static_resource_enriched_pdf.py` | `329035f25c43ca4aa86dd1c55d703880ebb81e59188e15530303eeef15ff5424` |
| `output/pdf/paper_iii_static_qse_source_anchor_20260829_v2/paper_iii_static_qse_source_anchor.tex` | `2275e65b45678669457a8b1d3710e88c47d249ca8d390328e6cb9226de5ba713` |
| `output/pdf/paper_iii_static_qse_source_anchor_20260829_v2/paper_iii_static_qse_source_anchor.pdf` | `4a9d1224785484d34e448617a486bb912d61da7a93d76272cf77da9853e841a4` |
| `output/pdf/paper_iii_static_qse_source_anchor_20260829_v2/manifest.json` | `d7c274fd3e7387034a8227dc9d92cdba13a55d54037609dca0874864db3b74a7` |

The manifest records source revision `d1717ccdb11fbbda4548f9ada4065f5c9e170cf2`.
The LaTeX invocation records a stable temporary-directory placeholder rather
than the random execution path. Two consecutive controlled builds at that
revision reproduced every output hash byte for byte. The PDF was rendered to
PNG at 110 dpi and inspected page by page. It has 13 letter-sized pages with no
clipped tables, overlapping text, broken glyphs, or missing figures.

## Verification

- Source-anchor contract test module: 5 passed.
- Focused Paper-III reporting guard: 1 passed.
- Two consecutive direct report builds: passed with identical output hashes.
- PDF: 13 pages, page-by-page visual inspection passed.
- Repository-wide reporting import-latency test: seven unrelated pre-existing
  Paper-I builders exceed the five-second threshold; the Paper-III builder
  itself imports below the limit.
- No CHTC job was submitted and no new scientific route was executed.
