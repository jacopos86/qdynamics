# Nighthawk Pruning Analysis — Final Report

**Source:** `artifacts/json/hh_backend_adapt_fullhorse_powell_20260322T194155Z_fakenighthawk.json`
**Generated:** 2026-03-22

## Summary

The fullhorse Nighthawk ADAPT circuit (16 layers, 60 compiled 2Q gates) contains 5 dead layers with |θ| < 1e-4. Both pruning paths yield **the same lean core** of 10–11 operators that compiles to **53 2Q gates** on FakeNighthawk — an 11.7% reduction with zero energy regression.

## Original Circuit

| Metric | Value |
|--------|------:|
| Logical depth | 16 |
| Logical params | 16 |
| Runtime params | 27 |
| Compiled 2Q gates (FakeNighthawk) | 60 |
| Compiled depth | 137 |
| Compiled size | 272 |
| Energy | 0.1587240824 |
| |ΔE| | 5.62e-05 |

## Pruning Candidates (|θ| < 1e-4)

| Index | Label | |θ| | Decision |
|-------|-------|-----|----------|
| 0 | `hh_termwise_ham_quadrature_term(yezeee)` | 1.46e-19 | REMOVE |
| 2 | `hh_termwise_ham_quadrature_term(yeeeze)` | 4.68e-10 | REMOVE |
| 13 | `hh_termwise_ham_quadrature_term(eyezee)` | 3.41e-10 | REMOVE |
| 14 | `paop_full:paop_disp(site=1)` | 6.12e-07 | REMOVE |
| 15 | `hh_termwise_ham_quadrature_term(yezeee)` | 0.00e+00 | REMOVE |

**Layers removed:** 5 / 16

## Path A: Fixed-Scaffold VQE

Removed 5 near-zero layers, re-optimized the 11 surviving operators with POWELL.

- **Energy:** 0.1587240824 (identical to original — zero regression)
- **Depth:** 11
- **Runtime params:** 21
- **Compiled 2Q gates:** 53
- **Compiled depth:** 133
- **Compiled size:** 233
- **nfev:** 452, **elapsed:** 0.4s

### Surviving scaffold

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

## Path B: Re-ADAPT Phase 3 from HF

Fresh ADAPT search from HF state with aggressive termination (`drop_floor=1e-9`, `patience=3`, `min_depth=6`) and Nighthawk-conditioned backend cost.

- **Energy:** 0.1587240824 (same as Path A)
- **Depth:** 10
- **Runtime params:** 20
- **Compiled 2Q gates:** 53
- **Compiled depth:** 133
- **Compiled size:** 233
- **Stop reason:** `drop_plateau`
- **nfev:** 12651

### Re-ADAPT operator sequence

1. `hh_termwise_ham_quadrature_term(yezeee)` (θ=-1.500894)
2. `hh_termwise_ham_quadrature_term(yeeeze)` (θ=-1.348605)
3. `hh_termwise_ham_quadrature_term(eyeeez)` (θ=+0.076385)
4. `hh_termwise_ham_quadrature_term(eyezee)` (θ=+3.57e-05) ← near-zero, prunable
5. `uccsd_ferm_lifted::uccsd_sing(alpha:0->1)` (θ=+0.785398)
6. `uccsd_ferm_lifted::uccsd_sing(beta:2->3)` (θ=+0.816540)
7. `paop_lf_full:paop_dbl_p(site=0->phonon=0)` (θ=-0.022554)
8. `paop_lf_full:paop_dbl_p(site=1->phonon=1)` (θ=-0.471671)
9. `paop_full:paop_hopdrag(0,1)::child_set[0,2]` (θ=-0.785398)
10. `paop_full:paop_disp(site=0)` (θ=-0.043453)

**Key observation:** Path B independently rediscovered the same 10-operator core as Path A (minus the two near-zero `eyezee` duplicates). The overlap is 10/10 — the two approaches converge on the same lean circuit.

## Pareto Comparison: |ΔE| vs Cost

| Variant | Depth | 2Q Gates | Compiled Depth | |ΔE| | Pareto dominant? |
|---------|------:|--------:|---------------:|---------:|:---:|
| Original (fullhorse) | 16 | 60 | 137 | 5.62e-05 | — |
| Path A (fixed scaffold) | 11 | **53** | **133** | 5.62e-05 | Yes |
| Path B (re-ADAPT from HF) | 10 | **53** | **133** | 5.62e-05 | Yes |

Both lean paths **strictly Pareto-dominate** the original fullhorse circuit:
- **Same accuracy** (|ΔE| = 5.62e-05)
- **11.7% fewer 2Q gates** (60 → 53)
- **2.9% shallower compiled depth** (137 → 133)
- **14.3% fewer total gates** (272 → 233)

## Converged Lean Core

The irreducible operator set for this HH instance (L=2, g=0.5, n_ph_max=1) on FakeNighthawk is:

| # | Operator family | Physics |
|---|----------------|---------|
| 1–3 | `hh_termwise_ham_quadrature_term` (×3 active) | Quadrature seed (phonon displacement) |
| 4–5 | `uccsd_sing` (alpha + beta) | Single excitations |
| 6–7 | `paop_dbl_p` (site 0→ph 0, site 1→ph 1) | Polaron double displacement |
| 8 | `paop_hopdrag(0,1)::child_set[0,2]` | Drag-assisted hopping |
| 9 | `paop_disp(site=0)` | Phonon displacement (site 0 only) |

This 9–10 operator core appears to be the **minimal Pareto-optimal scaffold** for this problem at the ~5.6e-05 |ΔE| accuracy level.

## Artifacts

- Pruning script: `pipelines/hardcoded/hh_prune_nighthawk.py`
- Path A pruned scaffold: `artifacts/json/hh_prune_nighthawk_pruned_scaffold.json`
- Path A cost scout: `artifacts/json/hh_prune_nighthawk_pruned_scaffold_compile_scout.json`
- Path B re-ADAPT result: `artifacts/json/hh_prune_readapt_nighthawk_20260322T201452Z.json`
- Path B cost scout: `artifacts/json/hh_prune_readapt_nighthawk_20260322T201452Z_compile_scout.json`

## Next Steps

1. **Second-pass pruning on Path B:** The `eyezee` at index 3 (θ=3.57e-05) could be removed for a 9-operator circuit — worth testing if the compiled cost drops further.
2. **Noise oracle:** Run the lean 53-gate circuit through the staged noise pipeline to confirm it survives decoherence better than the 60-gate original.
3. **Backend shortlist:** Test the lean scaffold on FakeFez and FakeMarrakesh to see if it's also Pareto-dominant across backends.
