# Paper-I L=3 manuscript-cost rerun — 2026-08-24

## Locked route

- Problem: half-filled open \(L=3\) Hubbard chain, \(U/t=1.25\), six qubits.
- Candidate supply: fixed 24-generator intact pool shared with AVQE.
- Phase 0: active absolute-gradient screen with cap 24 (nonrestrictive here).
- Phases I--III: adaptive inverse-Simpson shortlists with frontier ratio 0.9.
- RA accepted refits: Fubini--Study-whitened coordinate chart.
- AVQE refits: native Powell coordinates.
- Powell: `xtol=1e-5`, `ftol=1e-12`, `maxiter=200` for both methods.
- Compiled coordinates: exact FakeMarrakesh transpilation, optimization level 1,
  transpiler seed 7, common nonzero structural angle.
- Phase cost: population median/MAD normalization of signed
  \(\Delta N_{2q}/\Delta D_{2q}/\Delta D_c\), compressed by `(2/pi) atan`,
  weighted 0.30/0.30/0.25, and applied as `1 - C_bar/2`.

## Results at k=12

All estimator values below preserve every Powell/Hamiltonian evaluation and
remove only repeated staged gradient or metric primitives already available in
the same selector round. Beam uses selected-lineage accounting. Pruning charges
all executed delete--refit work.

| arm | N2q | D2q | Dc | corrected S_alg | terminal error |
|---|---:|---:|---:|---:|---:|
| AVQE | 344 | 331 | 1393 | 7477 | comparator payload |
| RA-Append | 300 | 288 | 1182 | 7749 | 0.0 |
| RA + batch | 300 | 284 | 1173 | 7939 | 2.66e-15 |
| RA + beam | 232 | 208 | 857 | 7880 | 7.55e-15 |
| RA + pruning | 292 | 280 | 1148 | 8570 | 1.78e-15 |

Relative to AVQE, the circuit reductions are:

- RA-Append: 12.8% / 13.0% / 15.1% in N2q / D2q / Dc; estimator work +3.6%.
- Batch: 12.8% / 14.2% / 15.8%; estimator work +6.2%.
- Beam: 32.6% / 37.2% / 38.5%; selected-lineage estimator work +5.4%.
- Pruning: 15.1% / 15.4% / 17.6%; estimator work +14.6%.

The previous request to place the pruning marker at k=9 or k=10 is not
supported by this rerun: pruning and RA-Append are identical through k=10.
The first circuit separation occurs at k=11.

## Result payload SHA-256

- AVQE: `3e86b66078fb84c106404438504491e07171fc375c6b364750c2482fab38fb52`
- RA-Append: `01a5351f2f100b4679871833139357ae455dd9bad7ae9cfd449d56256d99e774`
- Batch: `c065ca7ada9cf3c0b50139e2828fd94749c1db31773710fa2aa06905afcb64e0`
- Beam: `369dd77551fab5d929543f71b3fab7136e17b6c463933ad090806becf6076489`
- Pruning: `9e5d4d4f5655cbe9032da8fd20fc614e0fe861d8889e5715a1705daa383fba1f`

## Reproduction

- Single arm: `run_arm.py`
- Bounded local matrix (two workers): `run_parallel.py`
- Raw logs and payloads: `results/`

The synthetic-noise diagnostics are not part of this matrix. Their adaptive
shortlist has no positive feasible population at the first noisy round, so a
fixed-width rerun would be a different evaluated selector and was not silently
substituted here.
