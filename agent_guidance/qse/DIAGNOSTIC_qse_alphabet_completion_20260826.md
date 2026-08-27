# Paper III QSE alphabet-saturation diagnostic

Date: 2026-08-26

Branch: `paper-ii-exchange-selector`

Tested commits: `95e66867` (selection-round receipts), `29c16f53`
(adaptive seed/synthesis cost separation)

## Verdict

The two failed `n_ph_max=3` cells are genuine fixed-alphabet failures, not
greedy-selection regret, overlap-cutoff loss, exchange damage, a dimension
cap, or root matching.  Retuning the production score cannot repair them.

The current pool-exhaustion hybrid restores accuracy but is not competitive:
it pays for the complete record phase before residual synthesis.  A systematic
encoded-boson completion demonstrates a better tight-accuracy route, but an
always-on enlarged pool changes the greedy path and raises cost in three
already-solved regimes.  It is therefore a candidate alphabet for a versioned
v2, not yet the production algorithm.

Do not start the Paper III dynamics matrix from this state.  First implement
and validate an exact-reference-free safeguard that preserves the base-pool
prefix when completion is unnecessary.

## 1. Fixed-alphabet ceiling

The production selector, its exchanged terminal support, and the complete
128-record pool all span the same retained 51-dimensional space in both
failed regimes.  Lowering the relative overlap cutoff from `1e-10` to `1e-14`
adds no rank and changes no reported root.  Monotone root matching selects the
same first six roots as slot-wise comparison.

| regime | selected records | exchanged records | full-pool records | retained rank | selected/full-pool max error |
|---|---:|---:|---:|---:|---:|
| `weak_strong` | 63 | 63 | 128 | 51 | `1.7988398468e-2` |
| `intermediate_strong` | 57 | 56 | 128 | 51 | `1.1537424390e-2` |

The full-pool roots and terminal selected roots agree to numerical precision.
The selector has therefore recovered the complete span available from the
declared alphabet.  The missing accuracy is representational.

## 2. Correct adaptive-QSE resource accounting

The adaptive comparator starts from 18 declared linear-response records and
then synthesizes Davidson corrections.  The old trace charged every admitted
seed record as a first-order Trotter step of the Hamiltonian.  Those seeds are
literal pool records and cost `N2q=104` in total, rather than
`18 * 68 = 1224`.  Commit `29c16f53` now records

```text
seed-record resources + synthesized-direction count * Trotter-step resources
```

and emits both components in every adaptive trace row.  Old Paper III
adaptive resource ratios are invalid until their artifacts are regenerated.

The corrected hard-regime artifact is
`output/diagnostics/paper_iii_growth_trace_20260826_v2/growth_trace_nph3_hard.json`.

| regime | method | `N2q` at `1e-2` | `N2q` at `1e-4` | `N2q` at `1e-6` |
|---|---|---:|---:|---:|
| `weak_strong` | adaptive QSE | 444 | 920 | 1532 |
|  | pool-exhaustion hybrid | 4818 | 4886 | 5022 |
| `intermediate_strong` | adaptive QSE | 2212 | 2620 | 2824 |
|  | pool-exhaustion hybrid | 4324 | 4392 | 5072 |

Thus the existing hybrid is an accuracy repair, not a cost improvement.

## 3. Alphabet-enrichment probes

### Hamiltonian closure

Appending the physical vectors `H Omega_r |psi_0>` to the complete record
span raises retained rank from 51 to 60.  One closure layer reaches machine
precision in `weak_strong` and `5.3e-9` in `intermediate_strong`; a second
layer reaches machine precision in both.  However, representing these vectors
as explicit polynomial records `H Omega_r` is too costly: the unchanged
`Delta/K` selector crosses the target ladder only after approximately
`1.8e4`--`5.4e4` two-qubit gates.  Hamiltonian-product records are not the
competitive repair under the current oracle.

### Encoded-boson completion

The missing directions can instead be supplied by cheap sector-preserving
Pauli records.  The tested regime-independent class is

```text
{ P_b tensor I_f, P_b tensor Z_j :
  P_b is any nonidentity Pauli word on the encoded boson register,
  j is one of the four fermion qubits }.
```

At `n_ph_max=3` this adds 1275 candidates.  Selection still uses the unchanged
production two-term `Delta/K` rule and compiled costs.  The table compares the
existing base alphabet with an always-on union of the base and completion
alphabets.  `--` means the base alphabet cannot reach the target.

| regime | alphabet | `N2q` at `1e-2` | `N2q` at `1e-4` | `N2q` at `1e-6` |
|---|---|---:|---:|---:|
| `weak_weak` | base | 618 | 618 | 1254 |
|  | completion union | 584 | 584 | 584 |
| `intermediate_weak` | base | 294 | 294 | 294 |
|  | completion union | 724 | 724 | 724 |
| `strong_weak_u8` | base | 280 | 684 | 684 |
|  | completion union | 900 | 900 | 900 |
| `weak_strong` | base | -- | -- | -- |
|  | completion union | 828 | 850 | 872 |
| `intermediate_strong` | base | -- | -- | -- |
|  | completion union | 614 | 614 | 614 |
| `strong_strong_u8` | base | 258 | 258 | 258 |
|  | completion union | 920 | 920 | 920 |

The ideal envelope of the two alphabets beats corrected adaptive QSE in all
six `1e-4` and `1e-6` cells.  It is not an operating algorithm: choosing the
cheaper arm from exact errors would be reference-tuned.  At `1e-2`, adaptive
QSE remains cheaper in `weak_weak`, `intermediate_weak`, and `weak_strong`.

## 4. Required v2 gate

The new round receipts record retained rank, maximum target residual, spectral
window pressure, remaining candidate capture, and the admitted score
components without changing selected supports.  Retained-rank stagnation is a
useful warning but is not a sufficient gate by itself: it also occurs in
already-solved regimes.  Likewise, residual convergence alone is Ritz-blind
and can miss a root.

A production completion gate must satisfy all of the following without exact
energies:

1. preserve every base-alphabet prefix before the gate, so existing cheap
   crossings remain available;
2. detect loss of useful base-pool span from measured rank/residual/window
   receipts, not from regime labels or exact-reference error;
3. retain a window-completeness guard so the seed-orthogonal triplet root in
   `intermediate_strong` cannot be skipped;
4. admit completion records through the same `Delta/K` rule after activation;
5. beat corrected adaptive QSE on the full `nph1`/`nph3` fixed-accuracy matrix,
   including estimator work, before any manuscript superiority claim or
   dynamics campaign is resumed.

No score, cutoff, ridge, or residual threshold was retuned in this diagnostic.
