# Paper III block-seeded real-time-Krylov matrix evidence (2026-08-29)

## Verdict

The locked block-seeded real-time-Krylov route reaches the one-to-one six-root
energy target at `1e-6` in **11 of 16** cases.  It reaches all six
Hubbard--Holstein `n_ph_max=1` cells, two of six `n_ph_max=3` cells, two of
three Peierls--Hubbard cells, and the SIAM `N_b=4` transfer cell.  The same 11
cells reach `1e-2`, `1e-4`, and `1e-6`; the other five do not reach even
`1e-2` within the locked cap.

This is a complete fixed-policy comparator result, including its failures.  It
does not support a claim that block seeding uniformly repairs the earlier
single-seed route.  The block and single-seed success sets are complementary:
block seeding improves the aggregate success count from 10/16 to 11/16 and
repairs every `n_ph_max=1` failure, but loses four weak/intermediate
`n_ph_max=3` cells and `peierls_weak_weak` that the single-seed route reached.
Together with the post-run source audit, this shows that Krylov window coverage
is materially initialization-dependent.

## Locked route

- Policy ID:
  `block_seeded_realtime_krylov_lowest_diagonal_b6_dt0p5_v1`.
- Six lowest-diagonal computational probes in the declared target sector are
  considered in deterministic `(H_ii, basis_index)` order; the first six
  independent excited directions are admitted.
- The block size is `B=R=6`, the time step is `dt=0.5`, and exact excited
  roots are unavailable to the online route.
- The exact prepared ground is retained as an uncharged reference direction,
  matching the benchmark's common source-state boundary.
- Every computational-probe preparation is charged once per time shell.  These
  probes require one-qubit bit flips but zero two-qubit gates.  Every propagated
  state is additionally charged by its number of repeated Hamiltonian steps.
- The caps are the same raw basis-state caps used for the single-seed route:
  100 for Hubbard--Holstein, 40 for Peierls--Hubbard, and 100 for SIAM.  Because
  growth occurs in complete six-state shells, the terminal raw dimensions are
  97, 37, and 97 respectively.
- The supported pencil uses relative overlap cutoff `1e-12`.  Exact-reference
  errors use one-to-one Hungarian root assignment strictly after termination.
- `K` is aggregate Marrakesh graph-span two-qubit count.  Qiskit columns are
  summed FakeMarrakesh circuit-structure annotations, not a single executable
  circuit.  Query counts use the synthesized-basis pencil convention with no
  QWC reuse.

## Matched `1e-6` outcomes

For a reached cell, the row reports the first successful recorded shell.  For
a failed cell, it reports the terminal error and terminal expenditure; these
are not successful threshold costs.

| case | outcome | raw dim. | rank | max six-root error | K | Qiskit N2q | Qiskit D2q | estimator queries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hh::weak_weak::nph1` | reached | 19 | 13 | 2.803e-12 | 360 | 522 | 432 | 3,420 |
| `hh::intermediate_weak::nph1` | reached | 25 | 14 | 8.995e-13 | 720 | 1,044 | 864 | 5,850 |
| `hh::strong_weak_u8::nph1` | reached | 25 | 14 | 1.020e-09 | 720 | 1,044 | 864 | 5,850 |
| `hh::weak_strong::nph1` | reached | 19 | 13 | 3.331e-14 | 360 | 522 | 432 | 3,420 |
| `hh::intermediate_strong::nph1` | reached | 25 | 14 | 1.141e-13 | 720 | 1,044 | 864 | 5,850 |
| `hh::strong_strong_u8::nph1` | reached | 25 | 14 | 6.408e-11 | 720 | 1,044 | 864 | 5,850 |
| `hh::weak_weak::nph3` | **failed** | 97 | 41 | 2.156e-01 | 48,960 | 96,480 | 48,960 | 152,096 |
| `hh::intermediate_weak::nph3` | **failed** | 97 | 41 | 4.943e-01 | 48,960 | 96,480 | 48,960 | 152,096 |
| `hh::strong_weak_u8::nph3` | reached | 91 | 39 | 4.138e-07 | 42,840 | 84,420 | 42,840 | 133,952 |
| `hh::weak_strong::nph3` | **failed** | 97 | 42 | 5.686e-01 | 48,960 | 96,480 | 48,960 | 152,096 |
| `hh::intermediate_strong::nph3` | **failed** | 97 | 43 | 3.711e-01 | 48,960 | 96,480 | 48,960 | 152,096 |
| `hh::strong_strong_u8::nph3` | reached | 85 | 42 | 7.606e-07 | 37,128 | 73,164 | 37,128 | 116,960 |
| `peierls_weak_weak` | **failed** | 37 | 11 | 3.812e-01 | 9,000 | 12,510 | 11,250 | 21,090 |
| `peierls_weak_strong` | reached | 19 | 11 | 1.388e-15 | 1,800 | 2,502 | 2,250 | 5,700 |
| `peierls_u8_weak` | reached | 19 | 11 | 1.465e-14 | 1,800 | 2,502 | 2,250 | 5,700 |
| `siam_nb4_pilot` | reached | 97 | 88 | 7.898e-07 | 24,480 | 24,480 | 15,840 | 85,554 |

## Failure mechanism

The post-run source audit reconstructs the six declared computational probes
and measures their squared overlap weight on each exact target root.  It never
feeds this information back into source selection.  Every failed cell contains
one target root with source-block overlap weight below `1e-20`:

| failed case | zero-weight target root (one-based) |
|---|---:|
| `hh::weak_weak::nph3` | 4 |
| `hh::intermediate_weak::nph3` | 5 |
| `hh::weak_strong::nph3` | 4 |
| `hh::intermediate_strong::nph3` | 5 |
| `peierls_weak_weak` | 6 |

Thus increasing the number of time shells cannot recover those particular
roots under this fixed source block: unitary propagation preserves the zero
overlap.  Individual weights inside an exactly degenerate eigenspace are
basis-dependent; the artifact records that qualification.  None of the 11
successful cells has a target root below the declared zero-weight floor.

## Provenance and status

- Implementation/audit revision:
  `f158ecc0a8807c02febe8801d4fd4fa15a8893d3` on
  `paper-ii-exchange-selector`.
- Matrix:
  `output/diagnostics/paper_iii_block_seeded_realtime_krylov_matrix_20260829_v1/matrix.json`.
- Matrix SHA-256:
  `93fd614d3214f5f4948962a822d980b7be726f5d7045dd48a96084a1ffae370b`.
- Receipt:
  `output/diagnostics/paper_iii_block_seeded_realtime_krylov_matrix_20260829_v1/receipt.json`.
- The acquisition used the RAM guard with a 3,000 MB limit; peak resident use
  was 262 MB.
- Thirteen focused and neighboring tests passed before acquisition.
- Evidence class remains `candidate`; promotion is an explicit authorial
  decision.  No Paper-III manuscript files were edited by this increment.

## Manuscript boundary if promoted

The evidence supports comparison against two explicitly fixed Krylov policies:
single-seed RTK (10/16) and this six-probe block RTK (11/16).  It does not
support a universal claim against Krylov methods.  Failures should be displayed
as failures with terminal expenditure, and the complementary coverage should
be stated rather than selecting only the favorable route on a case-by-case
basis.
