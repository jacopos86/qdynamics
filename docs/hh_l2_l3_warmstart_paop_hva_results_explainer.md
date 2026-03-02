# L2/L3 HH Warm-Start PAOP+HVA Results (Human-Readable PDF Report)

## Part I - L=3 Results

## 1. Parameter Manifest

- Model family/name: `Hubbard` (HH specialization).
- Ansatz types covered: `hh_hva_tw`, `hh_hva_ptw`, ADAPT(`paop_lf_std`), ADAPT trend pools (`uccsd+paop`, `uccsd+paop+hva`).
- Drive enabled: both `false` (static sections) and `true` (driven section from `drive_from_fix1_warm_start_B_full.json`).
- Core physical parameters: Part I (L3) uses `t=1.0, U=4.0, dv=0.0`; Part II (L2 analog) also uses `t=1.0, U=4.0, dv=0.0`.
- HH-defining parameters for L3/L2 analog sections: `omega0=1.0`, `g_ep=0.5`, `n_ph_max=1`, boson encoding `binary`, ordering `blocked`, boundary `open`.
- Legacy baseline note: the `L2 VQE` row retained in Part I scoreboard is historical provenance at `U=2.0`, `g_ep=1.0`.
- Reproducibility sector references: L2 `(N_up,N_dn)=(1,1)`; L3 `(N_up,N_dn)=(2,1)`.
- Driven-run timing/drive: L3 uses `t_final=15.0`, `num_times=201`, `trotter_steps=192`, `exact_steps=384`; L2 analog uses `t_final=15.0`, `num_times=201`, `trotter_steps=128`, `exact_steps=256`; both use `A=0.5`, `omega=2.0`, `phi=0.0`, `t0=0.0`, `tbar=3.0`, pattern `staggered`.

### 1.1 What “Accessibility Rerun” Means (Exhaustive Definition)

In this report, “Accessibility rerun” does **not** mean a UI/accessibility feature and does **not** mean driven Trotter dynamics. It means a second-pass static ADAPT optimization run where we keep a fixed practical compute budget and re-run the warm-start branch under those fixed limits to measure what quality is realistically reachable.

Concretely, the accessibility runs come from `ACC_JSON = artifacts/useful/L3/l3_hh_accessibility_fixes_under8pct.json`. Each rung (for example, `Acc B` and `Acc C`) specifies explicit ADAPT limits such as `adapt_max_depth`, `adapt_maxiter`, `eps_grad`, and `wallclock_cap_s=1200`. The purpose is to compare outcomes under controlled run limits, not to change physics definitions.

The non-accessibility counterpart in this PDF is `B export` from `B_EXPORT_JSON = artifacts/useful/L3/warmstart_states/fix1_warm_start_B_full_state.json`, which is the direct exported rebuild artifact for warm-start branch B. That is why `B export` and `Acc B` can be very close in `DeltaE` but still differ in recorded depth/parameter count (`42` vs `43`) and runtime fields: they are separate executions with separate bookkeeping.

For this reason, throughout this document:
- `Accessibility rerun` = fixed-budget static ADAPT rerun (`Acc B`, `Acc C`).
- `Driven dynamics` = time-evolution branch analysis from `drive_from_fix1_warm_start_B_full.json`.

## 2. Executive Summary

- L2 strong VQE reaches `DeltaE=6.029e-07` (near exact).
- L3 warm VQE alone remains at `DeltaE=1.969e-02`.
- Warm-start + ADAPT improves L3 to `DeltaE=4.393e-03` (Accessibility C).
- Trend runs reach `DeltaE~2.622e-04` (best shown point).
- Drive branch diagnostics show PAOP branch fidelity near `0.995893` vs HVA near `0.827750` against the same filtered exact manifold.

## 3. Math and Metrics (Compact)

$$H = H_t + H_U + H_{ph} + H_{e-ph}, \quad H(t)=H+H_{drive}(t).$$
$$E_{exact,sector}=\min_{\psi\in\mathcal{H}_{(N_\uparrow,N_\downarrow)}}\langle\psi|H|\psi\rangle.$$
$$\Delta E = |E_{best}-E_{exact,sector}|, \quad \varepsilon_{rel}=\Delta E/|E_{exact,sector}|.$$
$$|\psi_d\rangle = e^{-i\theta_d G_d}\cdots e^{-i\theta_1 G_1}|\psi_0\rangle, \quad g_m=i\langle\psi_d|[H,G_m]|\psi_d\rangle.$$

Cost metrics used in this report:
$$\kappa_{eval}=nfev/P, \quad \tau_{eval}=runtime\_s/nfev.$$

## 4. Run Labels and Measurement Dictionary

### 4.1 Run Label Glossary (What each row name means)

| Label | Short definition |
|---|---|
| `L2 VQE` | Hardcoded HH VQE on L=2 (`hh_hva_tw`, `L-BFGS-B`). |
| `L3 warm VQE` | Warm-start seed VQE stage on L=3 (`hh_hva_ptw`, `COBYLA`). |
| `B export` | Exported fix1 warm-start branch B state (`depth=42`, `pool=paop_lf_std`). |
| `Acc B` | Accessibility rerun branch B (separate execution from `B export`, reached `depth=43`). |
| `Acc C` | Accessibility rerun branch C (different rung settings). |
| `A_medium` | Trend pool A (`uccsd+paop`), medium budget (`depth=20`). |
| `A_heavy` | Trend pool A (`uccsd+paop`), heavy budget (`depth=36`). |
| `B_medium` | Trend pool B (`uccsd+paop+hva`), medium budget (`depth=20`). |
| `B_heavy` | Trend pool B (`uccsd+paop+hva`), heavy budget (`depth=36`). |

Provenance key:
- `L2_JSON` = `artifacts/useful/L2/H_L2_hh_termwise_regular_lbfgs_t1.0_U2.0_g1_nph1.json`
- `L3_WARM_JSON` = `artifacts/json/hh_L3_hh_hva_ptw_heavy.json`
- `B_EXPORT_JSON` = `artifacts/useful/L3/warmstart_states/fix1_warm_start_B_full_state.json`
- `ACC_JSON` = `artifacts/useful/L3/l3_hh_accessibility_fixes_under8pct.json`
- `TREND_JSON` = `artifacts/useful/L3/l3_uccsd_paop_hva_trend_full_20260302T000521.json`

Clarification on naming:
- In this report, `Acc B` means \"Accessibility rung B\" (same concept you referred to as \"BACC\").
- `B export` and `Acc B` are related but not identical runs; that is why depth can appear as `42` vs `43`.

### 4.2 Measurement Dictionary (What each metric is)

Core optimization metrics:

| Metric | Definition | Units |
|---|---|---|
| `E_best` | `vqe.energy` or `result.E_best` | energy |
| `E_exact` | `ground_state.exact_energy_filtered` or `exact.E_exact_sector` | energy |
| `DeltaE` | `|E_best - E_exact|` | energy |
| `eps_rel` | `DeltaE / |E_exact|` | unitless |
| `Params P` | `vqe.num_parameters` or `result.num_parameters` | count |
| `nfev` | `vqe.nfev` or `result.nfev_total` | count |
| `Depth` | VQE proxy=`reps`; ADAPT depth = `ansatz_depth` or `adapt_depth_reached` | count |
| `Runtime (s)` | `runtime_s` (or nearest runtime field) | seconds |
| `kappa_eval` | `nfev / P` | eval/param |
| `tau_eval` | `runtime_s / nfev` | s/eval |

Dynamics metrics:

| Metric | Definition | Units |
|---|---|---|
| `F_paop` / `F_hva` | `fidelity_paop_trotter`, `fidelity_hva_trotter` | unitless |
| `dE_total_paop` / `dE_total_hva` | `|E_total_exact - E_total_trotter|` per branch | energy |
| `dD_paop` / `dD_hva` | `|D_exact - D_trotter|` per branch | unitless |

## 5. Readable Static Scoreboard

| Run | DeltaE | Depth | Params P | nfev | E_best | E_exact | eps_rel | Optimizer |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| L2 VQE | 6.029e-07 | 6 | 108 | 15042 | -0.389552500 | -0.389553103 | 1.548e-06 | L-BFGS-B |
| L3 warm VQE | 1.969e-02 | 3 | 39 | 4000 | 0.264627255 | 0.244940700 | 8.037e-02 | COBYLA |
| B export | 6.508e-03 | 42 | 42 | n/a | 0.251448234 | 0.244940700 | 2.657e-02 | ADAPT |
| Acc B | 6.508e-03 | 43 | 43 | 8063 | 0.251448234 | 0.244940700 | 2.657e-02 | ADAPT |
| Acc C | 4.393e-03 | 38 | 38 | 6227 | 0.249334000 | 0.244940700 | 1.794e-02 | ADAPT |
| A_medium | 2.622e-04 | 20 | 20 | 12640 | 0.245202940 | 0.244940700 | 1.071e-03 | ADAPT |
| A_heavy | 2.629e-04 | 36 | 36 | 26833 | 0.245203648 | 0.244940700 | 1.074e-03 | ADAPT |
| B_medium | 2.622e-04 | 20 | 20 | 12640 | 0.245202940 | 0.244940700 | 1.071e-03 | ADAPT |
| B_heavy | 2.629e-04 | 36 | 36 | 26833 | 0.245203648 | 0.244940700 | 1.074e-03 | ADAPT |

Note: this table intentionally rounds values for readability; all source values are retained in JSON artifacts.

![Static error comparison](hh_l2_l3_report_assets/fig01_deltae_bar.png)

![Cost vs accuracy](hh_l2_l3_report_assets/fig02_cost_vs_accuracy.png)

## 6. Accessibility and Warm-Start Convergence

Accuracy/cost view:

| Run | DeltaE_best | Depth | Params | nfev | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| B export | 6.508e-03 | 42 | 42 | n/a | 1201.4 |
| Acc B | 6.508e-03 | 43 | 43 | 8063 | 1261.9 |
| Acc C | 4.393e-03 | 38 | 38 | 6227 | 1208.3 |

Warm-start gain view:

| Run | DeltaE_warm | DeltaE_best | Gain factor (`DeltaE_warm/DeltaE_best`) | E_warm | E_best |
|---|---:|---:|---:|---:|---:|
| B export | 1.969e-02 | 6.508e-03 | 3.025 | 0.264627255 | 0.251448234 |
| Acc B | 1.969e-02 | 6.508e-03 | 3.025 | 0.264627255 | 0.251448234 |
| Acc C | 1.234e-02 | 4.393e-03 | 2.809 | 0.257282637 | 0.249334000 |

![Accessibility convergence histories](hh_l2_l3_report_assets/fig03_accessibility_histories.png)

![Selected operator family counts](hh_l2_l3_report_assets/fig10_family_counts.png)

### 6.1 Operator Family Symbol Definitions (Figure 4)

The three family labels in Figure 4 correspond to the PAOP generators:

$$
P_i = i(b_i^{\dagger}-b_i), \quad
n_i = n_{i\uparrow}+n_{i\downarrow}, \quad
\tilde n_i = n_i-\bar n.
$$

$$
K_{ij}=\sum_{\sigma\in\{\uparrow,\downarrow\}}\left(c_{i\sigma}^{\dagger}c_{j\sigma}+c_{j\sigma}^{\dagger}c_{i\sigma}\right), \quad
J_{ij}=i\sum_{\sigma\in\{\uparrow,\downarrow\}}\left(c_{i\sigma}^{\dagger}c_{j\sigma}-c_{j\sigma}^{\dagger}c_{i\sigma}\right).
$$

- `Disp`: $G^{\text{disp}}_i=\tilde n_i\,P_i$
- `Hop Drag`: $G^{\text{hopdrag}}_{ij}=K_{ij}(P_i-P_j)$
- `Cur Drag`: $G^{\text{curdrag}}_{ij}=J_{ij}(P_i-P_j)$

`Cur Drag` is the same `paop_curdrag` family (current-drag channel); this is the channel you referred to as Kerr-drag in discussion.

## 7. Trend Experiment (A/B pools)

- Raw pool sizes: `UCCSD=8`, `PAOP=7`, `HVA=13`.
- Pool A (`uccsd+paop`): `dedup_total=15`, `overlap_count=0`.
- Pool B (`uccsd+paop+hva`): `dedup_total=28`, `overlap_count=0`.

| Trend run | DeltaE | Depth | Params P | nfev | runtime (s) | final max|grad| |
|---|---:|---:|---:|---:|---:|---:|
| A_medium | 2.622e-04 | 20 | 20 | 12640 | 624.0 | 4.564e-03 |
| A_heavy | 2.629e-04 | 36 | 36 | 26833 | 1722.5 | 7.722e-03 |
| B_medium | 2.622e-04 | 20 | 20 | 12640 | 720.7 | 4.564e-03 |
| B_heavy | 2.629e-04 | 36 | 36 | 26833 | 2379.3 | 7.722e-03 |

![Trend energy traces](hh_l2_l3_report_assets/fig04_trend_energy_traces.png)

![Trend gradient traces](hh_l2_l3_report_assets/fig05_trend_grad_traces.png)

Section role boundary: Section 7 is static ADAPT trend optimization; Section 8 is driven branch dynamics.

## 8. Drive Dynamics (Branch-Aware)

Branch order: `exact_gs_filtered`, `exact_paop`, `trotter_paop`, `exact_hva`, `trotter_hva`.

### 8.0 Dynamics Reference Trajectory and How It Is Generated

For dynamics, the reference is the time-dependent exact trajectory

$$
E_{ref}(t) \equiv E_{exact\_gs\_filtered}(t) = \\texttt{energy\\_total\\_exact}(t),
$$

taken from the `exact_gs_filtered` branch in `drive_from_fix1_warm_start_B_full.json`.

This is not a single static scalar; it is a full time series used as the baseline for branch comparisons. In this run:
- `E_ref(t=0) = 0.24494070012791422`
- `E_ref(t=t_final=15) = 0.26953974487978255`

Repo options/fields used to generate this dynamics dataset (same file, `settings.drive` + top-level settings):
- drive enabled: `true`
- waveform: `A=0.5`, `omega=2.0`, `phi=0.0`, `t0=0.0`, `tbar=3.0`, pattern=`staggered`
- Trotter setup: `suzuki_order=2`, `trotter_steps=192`
- exact reference setup: `reference_steps_multiplier=2`, `reference_steps=384`
- exact reference method: `exponential_midpoint_magnus2_order2`
- backend: `scipy_sparse_expm_multiply`

### 8.1 Exact-GS vs PAOP/Trotter-PAOP (Primary Dynamics Comparison)

Naming map used in this section (`drive_from_fix1_warm_start_B_full.json`):
- `exact_gs_filtered`: exact/reference propagation from filtered-sector exact ground-state initialization.
- `exact_paop`: exact propagation from imported PAOP ADAPT state (`source=adapt_json`, `pool=paop_lf_std`, `depth=42`).
- `trotter_paop`: Trotter propagation from the same PAOP initialization.
- `exact_hva`: exact propagation from regular hardcoded VQE (`source=regular_vqe`, `ansatz=hh_hva_ptw`).
- `trotter_hva`: Trotter propagation from the same HVA initialization.

Primary PAOP triad (total-energy distances):

| Metric pair | t=0 | mean over time | max over time | t=t_final |
|---|---:|---:|---:|---:|
| `|E_gs - E_exact_paop|` | 6.508e-03 | 6.941e-03 | 7.938e-03 | 6.923e-03 |
| `|E_gs - E_trotter_paop|` | 6.508e-03 | 6.936e-03 | 7.910e-03 | 6.789e-03 |
| `|E_exact_paop - E_trotter_paop|` | 0.000e+00 | 1.380e-04 | 1.014e-03 | 1.340e-04 |

Companion HVA comparison (same metric form):

| Metric pair | t=0 | mean over time | max over time | t=t_final |
|---|---:|---:|---:|---:|
| `|E_gs - E_exact_hva|` | 1.777e-01 | 1.859e-01 | 1.905e-01 | 1.876e-01 |
| `|E_gs - E_trotter_hva|` | 1.777e-01 | 1.859e-01 | 1.904e-01 | 1.868e-01 |
| `|E_exact_hva - E_trotter_hva|` | 0.000e+00 | 4.360e-04 | 2.766e-03 | 7.359e-04 |

Fidelity summary (vs filtered exact GS manifold projector):

| Branch | mean | min | max | final |
|---|---:|---:|---:|---:|
| `paop` | 0.995893 | 0.995732 | 0.996043 | 0.995881 |
| `hva` | 0.827750 | 0.827244 | 0.828695 | 0.828071 |

Driven overlays requested in this revision:
- Plot A combines absolute energies and energy-error trajectories in one figure.
- Plot B combines absolute energies, fidelities, and observables in one figure.

![Drive absolute energies + energy errors](hh_l2_l3_report_assets/fig11_drive_energy_abs_and_error.png)

![Drive energies + fidelities + observables](hh_l2_l3_report_assets/fig12_drive_energy_fidelity_observables.png)

### 8.2 L=3 3D Energy Panels (Static vs Driven)

Static-energy 3D panel from the L3 drive artifact (`energy_static_*` lanes; this is an observable view, not a separate no-drive run).
Source JSON: `artifacts/useful/L3/drive_from_fix1_warm_start_B_full.json`.

![L3 3D static-energy lanes](hh_l2_l3_report_assets/fig13_l3_static_energy_3d.png)

Driven total-energy 3D panel from the same artifact (`energy_total_*` lanes).
Source JSON: `artifacts/useful/L3/drive_from_fix1_warm_start_B_full.json`.

![L3 3D driven total-energy lanes](hh_l2_l3_report_assets/fig14_l3_driven_energy_3d.png)

### 8.3 Additional Branch Aggregates

| Metric | PAOP branch | HVA branch |
|---|---:|---:|
| Fidelity mean | 0.995892975 | 0.827750111 |
| Fidelity final | 0.995881343 | 0.828070675 |
| max |Delta E_total| | 1.014e-03 | 2.766e-03 |
| mean |Delta E_total| | 1.380e-04 | 4.360e-04 |
| max |Delta doublon| | 5.556e-03 | 4.733e-03 |

![Drive fidelity](hh_l2_l3_report_assets/fig06_drive_fidelity.png)

![Drive total-energy error](hh_l2_l3_report_assets/fig07_drive_energy_error.png)

![Drive doublon error](hh_l2_l3_report_assets/fig08_drive_doublon_error.png)

![Staggered observable overlay](hh_l2_l3_report_assets/fig09_staggered_overlay.png)

## 9. Interpretable Time-Snapshot Appendix (No Raw Dump)

Snapshot rows every 10th time index for readability (21 rows total).

| idx | time | F_paop | F_hva | dE_total_paop | dE_total_hva | dD_paop | dD_hva |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.995909 | 0.827522 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 10 | 0.750 | 0.995909 | 0.827521 | 1.663e-06 | 1.320e-06 | 9.340e-06 | 1.044e-05 |
| 20 | 1.500 | 0.995909 | 0.827517 | 7.830e-06 | 1.727e-05 | 7.975e-06 | 9.208e-06 |
| 30 | 2.250 | 0.995904 | 0.827514 | 6.163e-06 | 3.898e-05 | 5.051e-05 | 5.763e-05 |
| 40 | 3.000 | 0.995906 | 0.827580 | 1.832e-05 | 1.211e-04 | 7.286e-05 | 8.990e-05 |
| 50 | 3.750 | 0.995910 | 0.827601 | 4.004e-05 | 4.533e-05 | 1.343e-04 | 8.513e-05 |
| 60 | 4.500 | 0.995910 | 0.827710 | 6.052e-05 | 2.118e-04 | 2.869e-04 | 2.994e-04 |
| 70 | 5.250 | 0.995936 | 0.827729 | 1.259e-05 | 8.408e-05 | 3.732e-04 | 3.713e-04 |
| 80 | 6.000 | 0.995924 | 0.827590 | 4.948e-05 | 2.581e-04 | 4.923e-04 | 4.956e-04 |
| 90 | 6.750 | 0.995951 | 0.827563 | 1.030e-04 | 1.305e-04 | 3.830e-04 | 4.241e-04 |
| 100 | 7.500 | 0.995918 | 0.827248 | 1.660e-04 | 6.827e-04 | 3.472e-04 | 3.184e-04 |
| 110 | 8.250 | 0.995908 | 0.827536 | 9.220e-05 | 8.233e-05 | 4.452e-04 | 1.238e-04 |
| 120 | 9.000 | 0.995863 | 0.827396 | 2.033e-04 | 3.001e-04 | 7.400e-04 | 9.808e-04 |
| 130 | 9.750 | 0.995822 | 0.827926 | 1.248e-04 | 5.640e-04 | 9.674e-04 | 1.164e-03 |
| 140 | 10.500 | 0.995833 | 0.827907 | 9.978e-05 | 1.265e-04 | 1.854e-03 | 2.196e-03 |
| 150 | 11.250 | 0.995769 | 0.828147 | 3.302e-04 | 1.730e-04 | 6.147e-04 | 9.901e-04 |
| 160 | 12.000 | 0.995880 | 0.828300 | 2.780e-04 | 9.137e-04 | 1.374e-03 | 1.826e-03 |
| 170 | 12.750 | 0.995851 | 0.828023 | 4.194e-05 | 1.302e-03 | 1.309e-04 | 3.459e-04 |
| 180 | 13.500 | 0.995954 | 0.828678 | 7.711e-04 | 2.766e-03 | 3.432e-03 | 1.780e-03 |
| 190 | 14.250 | 0.995912 | 0.827734 | 1.078e-04 | 1.247e-03 | 3.170e-03 | 3.969e-03 |
| 200 | 15.000 | 0.995881 | 0.828071 | 1.340e-04 | 7.359e-04 | 5.556e-03 | 4.733e-03 |

## 10. Quantile Appendix (Trajectory Error Distributions)

| quantile | dE_total_paop | dE_total_hva | dD_paop | dD_hva |
|---|---:|---:|---:|---:|
| min | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| q25 | 1.345e-05 | 4.970e-05 | 9.817e-05 | 8.990e-05 |
| median | 6.784e-05 | 2.426e-04 | 4.109e-04 | 3.386e-04 |
| q75 | 1.702e-04 | 6.503e-04 | 1.030e-03 | 9.851e-04 |
| q90 | 3.296e-04 | 1.141e-03 | 1.963e-03 | 2.055e-03 |
| q99 | 9.172e-04 | 2.538e-03 | 4.475e-03 | 4.095e-03 |
| max | 1.014e-03 | 2.766e-03 | 5.556e-03 | 4.733e-03 |

## Part II - L=2 Analog Run (New)

## 11. L=2 Analog Run (New)

### 11.1 Provenance

- Exported warm-start state:
  - `artifacts/useful/L2/warmstart_states/fix1_warm_start_B_l2_state.json`
- Static L2 analog run:
  - `artifacts/useful/L2/l2_open_blocked_static_from_fix1B.json`
- Driven L2 analog run:
  - `artifacts/useful/L2/l2_open_blocked_drive_from_fix1B.json`
- Strict trend crosscheck:
  - `artifacts/useful/L2/l2_uccsd_paop_hva_trend_crosscheck.json`

### 11.2 L2 Static Scoreboard (DeltaE-first ordering)

| Run | DeltaE | Depth | Params | nfev | Stop/Notes |
|---|---:|---:|---:|---:|---|
| L2 accessibility export (`fix1_warm_start_C`) | 1.087e-03 | 78 | 78 | 36889 | `wallclock_cap` |
| L2 internal HVA VQE (static pipeline) | 8.056e-03 | 2 | 16 | 1200 | `hh_hva_ptw`, `COBYLA` |
| L2 trend crosscheck (`A_heavy`) | 3.283e-01 | 6 | 6 | 264 | `eps_energy` |

### 11.3 L2 Drive Dynamics Triad (same format as L3)

Primary PAOP triad (static-energy distances in driven run):

| Metric pair | t=0 | mean over time | max over time | t=t_final |
|---|---:|---:|---:|---:|
| `|E_gs - E_exact_paop|` | 1.102e-03 | 1.294e-03 | 2.007e-03 | 1.262e-03 |
| `|E_gs - E_trotter_paop|` | 1.102e-03 | 1.857e-03 | 4.761e-03 | 1.561e-03 |
| `|E_exact_paop - E_trotter_paop|` | 0.000e+00 | 7.150e-04 | 3.500e-03 | 2.995e-04 |

Companion HVA comparison:

| Metric pair | t=0 | mean over time | max over time | t=t_final |
|---|---:|---:|---:|---:|
| `|E_gs - E_exact_hva|` | 8.056e-03 | 8.942e-03 | 2.228e-02 | 8.452e-03 |
| `|E_gs - E_trotter_hva|` | 8.056e-03 | 9.691e-03 | 2.220e-02 | 1.057e-02 |
| `|E_exact_hva - E_trotter_hva|` | 0.000e+00 | 9.449e-04 | 5.423e-03 | 2.115e-03 |

L2 fidelity summary:

| Branch | mean | min | max | final |
|---|---:|---:|---:|---:|
| `paop` | 0.999196 | 0.998322 | 0.999389 | 0.998603 |
| `hva` | 0.998070 | 0.997306 | 0.998337 | 0.997497 |

### 11.4 L2 3D Energy Panels (Static vs Driven)

All panels in this subsection are generated from L2 JSON only (no L3 trajectory source is used here).

L2 static no-drive 3D energy lanes.
Source JSON: `artifacts/useful/L2/l2_open_blocked_static_from_fix1B.json`.

![L2 3D static-energy lanes](hh_l2_l3_report_assets/fig15_l2_static_energy_3d.png)

L2 driven total-energy 3D lanes.
Source JSON: `artifacts/useful/L2/l2_open_blocked_drive_from_fix1B.json`.

![L2 3D driven total-energy lanes](hh_l2_l3_report_assets/fig16_l2_driven_energy_3d.png)

### 11.5 L2 Operator Family Counts (Cur-Drag/Kerr-Drag Consistency)

L2 includes the same family labels as L3 (`disp`, `hopdrag`, `curdrag`), with `curdrag` explicitly present.

| L2 accessibility run | Disp | Hop Drag | Cur Drag | Total selected depth |
|---|---:|---:|---:|---:|
| `fix1_warm_start_B` | 2 | 5 | 62 | 69 |
| `fix1_warm_start_C` | 4 | 73 | 1 | 78 |

![L2 selected operator family counts](hh_l2_l3_report_assets/fig17_l2_family_counts.png)

### 11.6 Scale Note (Why driven can look “closer” visually)

L2 quantitative scale check:
- No-drive static axis excursion (`E_gs`): ~`2.50e-16` (near-flat line).
- Driven static-observable excursion (`E_gs`): ~`8.673e-02`.
- Mean gap `|E_gs-E_hva|` no-drive: `8.056e-03`.
- Mean gap `|E_gs-E_hva|` driven: `8.942e-03` (slightly larger, not smaller).

Interpretation: visual overlap depends strongly on y-axis excursion; absolute error tables are the oracle for closeness.

Analogous L3 reminder:
- L3 static-observable `E_gs` excursion in drive artifact: ~`1.006e-01`.
- L3 driven total-energy `E_gs` excursion: ~`5.560e-02`.
- Mean `|E_gs-E_hva|` is still large in both views (`~1.856e-01` static-observable, `~1.859e-01` driven-total).

## 12. Interpretation Notes

1. Large `nfev` does not guarantee lower `DeltaE`; pool geometry and stopping criteria dominate late-stage behavior.
2. The `42 vs 43` depth discrepancy for B is a provenance difference across two executions (export artifact vs accessibility rerun), not a contradiction.
3. In driven outputs, internal HVA branch quality can differ substantially from imported PAOP branch quality; always interpret by branch source first.
4. This report intentionally removes unreadable raw dumps; all raw values remain in JSON artifacts listed in Section 1.
