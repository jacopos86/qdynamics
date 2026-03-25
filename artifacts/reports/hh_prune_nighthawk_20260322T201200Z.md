# Nighthawk Pruning Analysis

**Source:** `artifacts/json/hh_backend_adapt_fullhorse_powell_20260322T194155Z_fakenighthawk.json`
**Prune threshold:** |θ| < 1.0e-04
**Generated:** 2026-03-22T20:13:17.421839+00:00

## Original Circuit

- Logical depth: 16
- Logical params: 16
- Runtime params: 27
- Compiled 2Q gates: 50
- Compiled depth: 125
- Compiled size: 234
- |ΔE|: 5.62e-05
- Pareto |ΔE|/2Q: 1.12e-06

## Pruning Candidates

| Index | Label | |θ| | Decision |
|-------|-------|-----|----------|
| 0 | `hh_termwise_ham_quadrature_term(yezeee)` | 1.46e-19 | REMOVE |
| 2 | `hh_termwise_ham_quadrature_term(yeeeze)` | 4.68e-10 | REMOVE |
| 13 | `hh_termwise_ham_quadrature_term(eyezee)` | 3.41e-10 | REMOVE |
| 14 | `paop_full:paop_disp(site=1)` | 6.12e-07 | REMOVE |
| 15 | `hh_termwise_ham_quadrature_term(yezeee)` | 0.00e+00 | REMOVE |

**Layers removed:** 5 / 16
**Pruned depth:** 11

## Surviving Scaffold

1. `hh_termwise_ham_quadrature_term(yezeee)` (θ=-1.500893)
2. `hh_termwise_ham_quadrature_term(yeeeze)` (θ=-1.348605)
3. `hh_termwise_ham_quadrature_term(eyeeez)` (θ=+0.076385)
4. `hh_termwise_ham_quadrature_term(eyezee)` (θ=+0.026517)
5. `hh_termwise_ham_quadrature_term(eyezee)` (θ=-0.026481)
6. `uccsd_ferm_lifted::uccsd_sing(alpha:0->1)` (θ=+0.785398)
7. `uccsd_ferm_lifted::uccsd_sing(beta:2->3)` (θ=+0.816540)
8. `paop_lf_full:paop_dbl_p(site=0->phonon=0)` (θ=-0.022554)
9. `paop_lf_full:paop_dbl_p(site=1->phonon=1)` (θ=-0.471671)
10. `paop_full:paop_hopdrag(0,1)::child_set[0,2]` (θ=-0.785398)
11. `paop_full:paop_disp(site=0)` (θ=-0.043453)

## Path B: Re-ADAPT Phase 3

- Logical depth: 10
- Logical params: 10
- Runtime params: 16
- |ΔE|: 5.62e-05
- Compiled 2Q gates: 120
- Compiled depth: 398
- Compiled size: 509
- Pareto |ΔE|/2Q: 4.68e-07

## Pareto Comparison: |ΔE| / Cost

| Variant | Depth | 2Q Gates | |ΔE| | |ΔE|/2Q |
|---------|------:|--------:|---------:|--------:|
| Original (fullhorse) | 16 | 50 | 5.62e-05 | 1.12e-06 |
| Path B (re-ADAPT) | 10 | 120 | 5.62e-05 | 4.68e-07 |
