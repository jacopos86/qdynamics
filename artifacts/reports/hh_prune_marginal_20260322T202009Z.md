# Marginal Pruning Analysis — Nighthawk Pruned Scaffold

**Source:** `artifacts/json/hh_prune_nighthawk_pruned_scaffold.json`
**Generated:** 2026-03-22T20:20:50.364350+00:00

## Baseline

- Operators: 11
- Runtime params: 21
- Energy: 0.1587240824
- |ΔE|: 5.62e-05
- Exact: 0.1586679041
- Compiled 2Q gates (FakeNighthawk): 53

## Leave-One-Out Results

Each row shows what happens when that single operator is removed and remaining params are re-optimized.

| Idx | Operator | |θ| | ΔE regression | |ΔE| after | Cheap? |
|-----|----------|-----|---------------|-----------|--------|
| 3 | `hh_termwise_ham_quadrature_term(eyezee)` | 2.65e-02 | -9.22e-13 | 5.62e-05 | yes |
| 4 | `hh_termwise_ham_quadrature_term(eyezee)` | 2.65e-02 | -9.21e-13 | 5.62e-05 | yes |
| 2 | `hh_termwise_ham_quadrature_term(eyeeez)` | 7.64e-02 | -9.21e-13 | 5.62e-05 | yes |
| 0 | `hh_termwise_ham_quadrature_term(yezeee)` | 1.50e+00 | -9.16e-13 | 5.62e-05 | yes |
| 1 | `hh_termwise_ham_quadrature_term(yeeeze)` | 1.35e+00 | -9.01e-13 | 5.62e-05 | yes |
| 7 | `paop_lf_full:paop_dbl_p(site=0->phonon=0)` | 2.26e-02 | +1.39e-05 | 7.01e-05 | yes |
| 10 | `paop_full:paop_disp(site=0)` | 4.35e-02 | +4.20e-03 | 4.25e-03 |  |
| 9 | `paop_full:paop_hopdrag(0,1)::child_set[0,2]` | 7.85e-01 | +3.28e-01 | 3.28e-01 |  |
| 5 | `uccsd_ferm_lifted::uccsd_sing(alpha:0->1)` | 7.85e-01 | +5.99e-01 | 5.99e-01 |  |
| 6 | `uccsd_ferm_lifted::uccsd_sing(beta:2->3)` | 8.17e-01 | +5.99e-01 | 5.99e-01 |  |
| 8 | `paop_lf_full:paop_dbl_p(site=1->phonon=1)` | 4.72e-01 | +7.80e-01 | 7.80e-01 |  |

### Interpretation

Operators sorted by cheapness of removal (smallest regression first). Operators with regression < 1e-4 are candidates for further pruning where the gate savings likely justify the small accuracy cost.

## Greedy Multi-Prune (regression budget)

Sequential greedy removal of cheapest operators, re-optimizing after each.

| Budget | Removed | Remaining depth | |ΔE| after | Regression | Layers cut |
|--------|---------|----------------:|----------:|-----------:|-----------:|
| 1e-06 | [3], [4], [2], [0], [1] | 6 | 5.62e-05 | +9.57e-11 | 5 |
| 1e-05 | [3], [4], [2], [0], [1] | 6 | 5.62e-05 | -6.38e-13 | 5 |
| 1e-04 | [3], [4], [2], [0], [1], [7] | 5 | 7.01e-05 | +1.39e-05 | 6 |
| 5e-04 | [3], [4], [2], [0], [1], [7] | 5 | 7.01e-05 | +1.39e-05 | 6 |
| 1e-03 | [3], [4], [2], [0], [1], [7] | 5 | 7.01e-05 | +1.39e-05 | 6 |
| 5e-03 | [3], [4], [2], [0], [1], [7] | 5 | 7.01e-05 | +1.39e-05 | 6 |

## Pareto Menu

These are the pruning options available, from most conservative to most aggressive:

| Variant | Depth | Est. 2Q gates | |ΔE| | Notes |
|---------|------:|-------------:|---------:|-------|
| Current pruned | 11 | 53 | 5.62e-05 | Baseline (no further pruning) |
| Budget 1e-06 | 6 | ~28 | 5.62e-05 | -5 ops: hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term |
| Budget 1e-05 | 6 | ~28 | 5.62e-05 | -5 ops: hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term |
| Budget 1e-04 | 5 | ~23 | 7.01e-05 | -6 ops: hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, paop_lf_full:paop_dbl_p |
| Budget 5e-04 | 5 | ~23 | 7.01e-05 | -6 ops: hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, paop_lf_full:paop_dbl_p |
| Budget 1e-03 | 5 | ~23 | 7.01e-05 | -6 ops: hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, paop_lf_full:paop_dbl_p |
| Budget 5e-03 | 5 | ~23 | 7.01e-05 | -6 ops: hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, hh_termwise_ham_quadrature_term, paop_lf_full:paop_dbl_p |
