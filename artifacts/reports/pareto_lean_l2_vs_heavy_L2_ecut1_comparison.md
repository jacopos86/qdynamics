# Pareto-Lean L2 vs Heavy Full-Meta: L=2 n_ph_max=1 ADAPT-VQE Comparison

## Verdict

The 11-operator `pareto_lean_l2` pool still matches the heavy 46-operator `full_meta` run at essentially the same energy on a fresh rerun dated **2026-03-21**, and the current direct per-Pauli runtime/cost path yields an honest transpiled cost of **97 CX** on `FakeGuadalupeV2` (default `opt_level=1`).

## Run Artifacts

- Heavy scaffold extraction: [adapt_hh_L2_ecut1_full_meta_phase3_heavy_powell_20260321T014500Z.full_scaffold_for_pruning.json](../json/adapt_hh_L2_ecut1_full_meta_phase3_heavy_powell_20260321T014500Z.full_scaffold_for_pruning.json)
- Heavy selected-ops log summary: [adapt_hh_L2_ecut1_full_meta_phase3_heavy_powell_20260321T014500Z.selected_ops_from_log.json](../json/adapt_hh_L2_ecut1_full_meta_phase3_heavy_powell_20260321T014500Z.selected_ops_from_log.json)
- Historical lean artifact: [adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell.json](../json/adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell.json)
- Current-code rerun (2026-03-21): [adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell_rerun_currentcode_20260321T171615Z.json](../json/adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell_rerun_currentcode_20260321T171615Z.json)
- Motif analysis: [hh_L2_ecut1_scaffold_motif_analysis.md](hh_L2_ecut1_scaffold_motif_analysis.md)

## Physics

All compared runs use the same HH instance:
- `L=2`, `t=1.0`, `u=4.0`, `dv=0.0`, `omega0=1.0`, `g_ep=0.5`
- `n_ph_max=1`, `boson_encoding=binary`, `ordering=blocked`, `boundary=open`
- Total qubits: 6 (fermion=4, boson=2)
- Exact filtered GS energy: `0.1586679041257264`

## Pool Comparison

| | Heavy (`full_meta`) | Lean (`pareto_lean_l2`) |
|---|---:|---:|
| Pool size (post-dedup) | 46 | **11** |
| UCCSD singles | 2 | 2 |
| UCCSD doubles | 1 | **0** |
| HH quadrature terms | 4 | 4 |
| HH unit terms | 16 | **0** |
| HVA layers | 4 | **0** |
| `paop_cloud_p` | 2 | 2 |
| `paop_cloud_x` | 2 | **0** |
| `paop_disp` | 2 | **0** |
| `paop_dbl` | 2 | **0** |
| `paop_hopdrag` | 1 | 1 |
| `paop_dbl_p` | 4 | 2 |
| `paop_dbl_x` | 4 | **0** |
| `paop_curdrag` | 1 | **0** |
| `paop_hop2` | 1 | **0** |

The lean pool is still a **76% reduction** in candidate count relative to heavy (`46 -> 11`).

## Result Summary

| Metric | Heavy (`full_meta`) | Historical lean artifact | Current-code rerun |
|---|---:|---:|---:|
| Final energy | 0.15872408303679303 | 0.15872408323495057 | **0.15872408236037028** |
| `abs_delta_e` | 5.6178911068e-05 | 5.6179109224e-05 | **5.6178234644e-05** |
| Logical depth | 97 (manual interrupt) | 16 | **14** |
| Runtime parameters | — | 16 | **25** |
| Logical parameters | — | 16 | **14** |
| Stop reason | manual interrupt | `drop_plateau` | `drop_plateau` |

### Rerun-vs-historical lean delta

- Energy difference: `-8.7458e-10`
- `abs_delta_e` difference: `-8.7458e-10`
- The rerun is therefore **energy-equivalent for practical purposes**.

## What changed in the current-code rerun

The fresh rerun does **not** preserve the exact historical 16-layer scaffold.

What stayed the same:
- Positions 1–8 still follow the same physical core:
  1. four HH quadrature seed terms,
  2. two lifted-UCCSD singles,
  3. two `paop_dbl_p` terms.

What changed:
- The rerun diverges at logical position 9.
- It converges to a **14-layer logical scaffold** instead of 16.
- Under the current per-Pauli runtime parameterization, those 14 logical layers expand to **25 runtime parameters**.

## Current-code rerun scaffold (14 logical operators)

```text
 1. hh_termwise_ham_quadrature_term(yezeee)
 2. hh_termwise_ham_quadrature_term(yeeeze)
 3. hh_termwise_ham_quadrature_term(eyeeez)
 4. hh_termwise_ham_quadrature_term(eyezee)
 5. uccsd_ferm_lifted::uccsd_sing(alpha:0->1)
 6. uccsd_ferm_lifted::uccsd_sing(beta:2->3)
 7. paop_lf_full:paop_dbl_p(site=0->phonon=1)
 8. paop_lf_full:paop_dbl_p(site=1->phonon=0)
 9. paop_full:paop_hopdrag(0,1)::child_set[0,2]
10. paop_full:paop_cloud_p(site=1->phonon=0)
11. paop_full:paop_cloud_p(site=1->phonon=0)::child_set[1]
12. paop_full:paop_cloud_p(site=0->phonon=1)
13. paop_lf_full:paop_dbl_p(site=0->phonon=1)::child_set[3]
14. paop_lf_full:paop_dbl_p(site=0->phonon=1)::child_set[3]
```

Interpretation:
- The rerun still validates the **11-operator pool**.
- It does **not** validate the stricter claim that the historical 16-entry lean scaffold is uniquely reproduced under the current code.
- The physically relevant claim that survives is stronger: **the pruned pool still reaches the same accuracy target without needing the heavy pool**.

## Honest current gate-count result (supersedes the old Suzuki estimate)

This gate-count result comes from running the current `adapt_circuit_cost.py` on the fresh rerun artifact above.

### Cost basis

- Circuit semantics: ordered **per-Pauli rotations** matching the current runtime execution path
- Backend: `FakeGuadalupeV2`
- Transpiler setting: default `opt_level=1`
- Source artifact: `adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell_rerun_currentcode_20260321T171615Z.json`

### Circuit cost

| Metric | Value |
|---|---:|
| Logical parameters | 14 |
| Runtime parameters | 25 |
| Abstract gates | 203 |
| Abstract depth | 85 |
| Transpiled gates | 178 |
| Transpiled depth | 97 |
| 2-qubit gates | **97 CX** |
| 1Q gates reported | 81 (`59 rz + 22 sx`) |

### Measurement cost

| Metric | Value |
|---|---:|
| Hamiltonian terms | 17 |
| QWC groups | 4 |
| Total shots @ `1.6e-3` target precision | 3,027,346 |

## Important reporting update

The earlier version of this report quoted a **Suzuki-synthesized depth-12 estimate** around **284–326 CX**. That number is now historical only.

Why it changed:
- it was based on an older circuit-cost path using Suzuki-style synthesis,
- it evaluated a historical depth-12 scaffold rather than the fresh current-code rerun,
- the current code now uses the direct per-Pauli runtime semantics, and the refreshed honest cost on the rerun artifact is **97 CX**.

So the best current statement is:
- **same physics**,
- **same practical energy quality**,
- **smaller logical scaffold than the historical lean artifact**,
- **much lower honest 2Q gate count than the old Suzuki-based estimate**.

## Overall conclusions

1. **The 11-operator `pareto_lean_l2` pool remains validated.** The fresh rerun reproduces the target energy quality at essentially the same `abs_delta_e` as both the heavy run and the historical lean artifact.

2. **The exact logical scaffold is not rigid across code versions.** The old 16-layer lean scaffold and the fresh 14-layer rerun differ starting at position 9, but this does not materially affect the final energy.

3. **The heavy 46-operator pool is still unnecessary for L=2, `n_ph_max=1`, `ecut_1`.** The fresh rerun supports the same practical conclusion as before: the heavy pool does not buy meaningful extra accuracy here.

4. **The current honest circuit cost is excellent.** On the current runtime semantics, the rerun compiles to **97 CX** on `FakeGuadalupeV2`, with only 4 QWC measurement groups.

5. **The old Suzuki-based hardware-cost section should no longer be used for decision-making.** The direct per-Pauli cost above is the relevant number for the current code path.
