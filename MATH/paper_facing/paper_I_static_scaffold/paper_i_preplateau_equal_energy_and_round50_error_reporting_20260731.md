# Paper-I pre-plateau equal-energy costs and round-50 errors

Date: 2026-07-31

Status: working numerical record for the next manuscript-claim pass. This file does not itself promote any run or modify the manuscript.

## Reporting decision

Use two distinct views.

1. **Resource comparison at a pre-plateau common attainable error.** For the macro representation, compare the stationary-source always-insertion RA trajectory from page 1 of the evolving stationary-core report with macro Append-ADAPT from that same page-1 provenance. For the singleton representation, use all available page-2 trajectories from that report and attach costs only to plateau-insertion RA and singleton Append-ADAPT. Report the prefix cost tuple
   \[
   (N_{2q},D_{2q},D_c,W_{1q},S_{\mathrm{alg}}),
   \qquad
   S_{\mathrm{alg}}=N_{H,\mathrm{outer}}+N_{H,\mathrm{refit}}+N_{\mathrm{grad}}+N_{\mathrm{metric}}.
   \]
2. **Terminal comparison at controller round 50.** Compare absolute same-cutoff ED energy errors only. Do not attach round-50 costs to this view.

The cost tables use `Append / RA` ratios. A ratio greater than one means that RA used fewer resources; a ratio below one means that Append used fewer resources.

## Pre-plateau selector

The calculation uses the typed `paper_i_effective_plateau_v1` and the established shared-window common-accuracy selector in `pipelines/reporting/paper_i_run_summary.py`.

For each trajectory, the effective plateau is the earliest prefix whose absolute energy error is no more than 110% of that trajectory's best error over the complete 50-round horizon. If the two effective plateau rounds are \(k_{\mathrm{pl}}^{\mathrm{RA}}\) and \(k_{\mathrm{pl}}^{\mathrm{A}}\), the shared window ends at

\[
k_\star=\min(k_{\mathrm{pl}}^{\mathrm{RA}},k_{\mathrm{pl}}^{\mathrm{A}}).
\]

Within this window, define the common attainable target

\[
\epsilon_\star=
\max\left[
\min_{k\le k_\star}\epsilon_{\mathrm{RA}}(k),
\min_{k\le k_\star}\epsilon_{\mathrm{A}}(k)
\right].
\]

Each method is costed at its first prefix \(k\le k_\star\) satisfying \(\epsilon(k)\le\epsilon_\star\). The crossings need not have identical errors because accepted rounds are discrete; both merely meet the same target.

## Common-target crossing audit

| Representation | Regime | Plateau rounds RA/A | Shared end | Common target \(\epsilon_\star\) | Crossing rounds RA/A | Crossing errors RA/A |
|---|---:|---:|---:|---:|---:|---:|
| Macro, always insertion | WW | 15 / 32 | 15 | 9.469101e-4 | 6 / 15 | 9.375982e-4 / 9.469101e-4 |
| Macro, always insertion | IW | 12 / 29 | 12 | 5.215884e-2 | 4 / 12 | 1.273516e-2 / 5.215884e-2 |
| Macro, always insertion | SW | 9 / 10 | 9 | 1.749468e-6 | 9 / 9 | 1.434625e-6 / 1.749468e-6 |
| Macro, always insertion | WS | 17 / 10 | 10 | 4.054334e-2 | 9 / 10 | 3.535526e-2 / 4.054334e-2 |
| Macro, always insertion | IS | 13 / 10 | 10 | 2.576231e-2 | 7 / 10 | 2.568160e-2 / 2.576231e-2 |
| Macro, always insertion | SS | 11 / 13 | 11 | 5.542285e-5 | 10 / 11 | 4.978185e-5 / 5.542285e-5 |
| Singleton, plateau insertion | WW | 38 / 37 | 37 | 1.001780e-9 | 35 / 37 | 6.149146e-10 / 1.001780e-9 |
| Singleton, plateau insertion | IW | 37 / 34 | 34 | 1.443102e-8 | 34 / 30 | 1.443102e-8 / 7.352090e-9 |
| Singleton, plateau insertion | SW | 13 / 18 | 13 | 1.407003e-6 | 12 / 13 | 1.406479e-6 / 1.407003e-6 |
| Singleton, plateau insertion | WS | 50 / 48 | 48 | 6.387528e-4 | 35 / 48 | 5.207833e-4 / 6.387528e-4 |
| Singleton, plateau insertion | IS | 49 / 50 | 49 | 1.412750e-4 | 33 / 49 | 3.331624e-5 / 1.412750e-4 |
| Singleton, plateau insertion | SS | 45 / 50 | 45 | 4.421326e-8 | 45 / 42 | 4.421326e-8 / 4.301073e-8 |

## Pre-plateau equal-energy resource costs

Tuples are ordered as `(N2q, D2q, Dc, W1q, S_alg)`.

### Macro: stationary-source always-insertion RA versus macro Append

| Regime | RA crossing cost | Append crossing cost | Append/RA ratios `(N2q, D2q, Dc, W1q, S_alg)` |
|---|---:|---:|---:|
| WW | unavailable: exact prefix archive retired | (1,002, 893, 4,173, 1,918, 40,881) | — |
| IW | (84, 80, 480, 180, 3,259) | (970, 872, 3,954, 1,902, 54,510) | (11.548, 10.900, 8.238, 10.567, 16.726) |
| SW | (192, 171, 1,185, 364, 14,695) | (256, 204, 1,337, 528, 4,418) | (1.333, 1.193, 1.128, 1.451, 0.301) |
| WS | (208, 177, 998, 376, 17,061) | (328, 184, 1,189, 552, 7,572) | (1.577, 1.040, 1.191, 1.468, 0.444) |
| IS | (70, 59, 324, 152, 10,911) | (328, 184, 1,189, 552, 7,640) | (4.686, 3.119, 3.670, 3.632, 0.700) |
| SS | (748, 645, 3,694, 1,134, 23,801) | (612, 352, 2,097, 998, 13,119) | (0.818, 0.546, 0.568, 0.880, 0.551) |

Across the five macro cells with recompilable always-insertion prefixes, RA is lower in all four compiled-circuit coordinates in 4/5. Append is lower in all four circuit coordinates in strong--strong. RA has lower \(S_{\mathrm{alg}}\) only in intermediate--weak; Append has lower \(S_{\mathrm{alg}}\) in the other 4/5. The weak--weak trajectory and crossing remain reportable, but its RA cost tuple is not substituted because the exact prefix archive was retired locally.

### Singleton: plateau-insertion RA versus singleton Append

| Regime | RA crossing cost | Append crossing cost | Append/RA ratios `(N2q, D2q, Dc, W1q, S_alg)` |
|---|---:|---:|---:|
| WW | (146, 112, 487, 285, 90,910) | (194, 176, 727, 357, 281,774) | (1.329, 1.571, 1.493, 1.253, 3.099) |
| IW | (120, 90, 525, 256, 93,269) | (140, 124, 619, 254, 250,765) | (1.167, 1.378, 1.179, 0.992, 2.689) |
| SW | (28, 19, 139, 78, 9,110) | (50, 44, 251, 111, 19,896) | (1.786, 2.316, 1.806, 1.423, 2.184) |
| WS | (128, 95, 491, 277, 126,847) | (262, 226, 867, 488, 1,030,044) | (2.047, 2.379, 1.766, 1.762, 8.120) |
| IS | (118, 89, 465, 257, 165,467) | (266, 235, 931, 479, 1,175,668) | (2.254, 2.640, 2.002, 1.864, 7.105) |
| SS | (188, 148, 846, 375, 200,972) | (192, 165, 776, 360, 821,233) | (1.021, 1.115, 0.917, 0.960, 4.086) |

Across all six singleton cells, RA has lower \(N_{2q}\), \(D_{2q}\), and \(S_{\mathrm{alg}}\) in 6/6, lower \(D_c\) in 5/6, and lower \(W_{1q}\) in 4/6. The Append/RA ranges are 1.021--2.254 for \(N_{2q}\), 1.115--2.640 for \(D_{2q}\), and 2.184--8.120 for \(S_{\mathrm{alg}}\). RA is strictly lower in all five cost coordinates in 4/6 singleton cells.

### Combined descriptive count

Across the 11 comparisons with complete cost tuples, RA is lower in \(N_{2q}\) for 10/11, \(D_{2q}\) for 10/11, \(D_c\) for 9/11, \(W_{1q}\) for 8/11, and \(S_{\mathrm{alg}}\) for 7/11. It is strictly lower across the full five-coordinate tuple in 5/11 cells. These are descriptive counts over heterogeneous regimes, not a statistical average or an uncertainty statement.

## Round-50 same-cutoff energy error only

No resource costs belong in this section. The ratio is `Append error / RA error`; values above one favor RA and values below one favor Append. Very small differences near unity should be described as comparable rather than as meaningful wins.

### Macro: stationary-source always-insertion RA versus macro Append

| Regime | RA round-50 absolute error | Append round-50 absolute error | Append/RA error ratio |
|---|---:|---:|---:|
| WW | 3.733213e-4 | 4.893695e-4 | 1.311 |
| IW | — | 2.670662e-2 | unavailable: RA paused at \(k=27\) |
| SW | — | 1.386808e-6 | unavailable: RA paused at \(k=27\) |
| WS | — | 3.961709e-2 | unavailable: RA paused at \(k=43\) |
| IS | — | 2.483639e-2 | unavailable: RA paused at \(k=41\) |
| SS | — | 3.953301e-5 | unavailable: RA paused at \(k=21\) |

### Singleton: plateau-insertion RA versus singleton Append

| Regime | RA round-50 absolute error | Append round-50 absolute error | Append/RA error ratio |
|---|---:|---:|---:|
| WW | 1.287859e-14 | 9.416689e-10 | 7.3119e4 |
| IW | 9.482493e-11 | 3.197095e-9 | 33.716 |
| SW | 8.006202e-7 | 8.010331e-7 | 1.00052 |
| WS | 4.240144e-5 | 6.059548e-4 | 14.291 |
| IS | 4.655843e-6 | 1.085717e-4 | 23.319 |
| SS | 4.073966e-8 | 1.161507e-8 | 0.285 |

The WW singleton RA value is near the numerical floor and should not support a precise multi-order superiority claim without a numerical-tolerance qualifier. The SW singleton pair is effectively tied at the displayed scale even though its raw ratio falls slightly on one side of unity.

The page-2 stationary singleton endpoints used as plot context are:

| Regime | Always-insertion RA | Plateau-insertion RA | No-insertion RA | Append-ADAPT |
|---|---:|---:|---:|---:|
| WW | 2.795091e-10 | 1.287859e-14 | 2.997602e-15 | 9.416689e-10 |
| IW | 5.070305e-8 | 9.482493e-11 | 1.296002e-11 | 3.197095e-9 |
| SW | pending | 8.006202e-7 | 1.381829e-6 | 8.010331e-7 |
| WS | pending | 4.240144e-5 | 4.240144e-5 | 6.059548e-4 |
| IS | pending | 4.655843e-6 | 5.056445e-6 | 1.085717e-4 |
| SS | pending | 4.073966e-8 | 2.279102e-5 | 1.161507e-8 |

## Candidate claim language for the manuscript pass

Subject to the evidence qualifications below, the most defensible cost statement is:

> At a common attainable accuracy selected within a window ending at the earlier effective plateau, plateau-insertion singleton RA used fewer two-qubit gates, lower two-qubit depth, and less logical estimator work than Append-ADAPT in all six regimes, with Append/RA ratios of 1.021--2.254, 1.115--2.640, and 2.184--8.120, respectively. In the five macro panels whose always-insertion prefix can presently be recompiled, RA used fewer two-qubit gates and lower two-qubit and total circuit depth in four; logical estimator work favored RA only in intermediate--weak.

For the terminal view, use the per-regime energy-error table and do not combine it with resource-cost claims. In particular, avoid language implying that a round-50 error advantage was obtained at a comparable round-50 cost.

## Evidence qualifications

1. **Partial-report status.** The evolving stationary-core report marks itself as cross-revision partial progress and not final paper evidence. Its page-1 and page-2 trajectories are being used for the present manuscript candidate framing, with their source qualifications retained.
2. **Paused macro always-insertion trajectories.** Weak--weak reaches round 50; intermediate--weak, strong--weak, weak--strong, intermediate--strong, and strong--strong end at authenticated rounds 27, 27, 43, 41, and 21. These prefixes support the pre-plateau matched-error comparison, not a uniform round-50 claim.
3. **Weak--weak macro cost limitation.** The weak--weak always-insertion trajectory is reportable, but its exact selected-prefix cost tuple is unavailable because the source archive needed for recompilation was retired locally. No different trajectory or cost was substituted.
4. **Singleton availability.** Page 2 has complete always-insertion singleton trajectories only for weak--weak and intermediate--weak; the other four always-insertion cells remain pending and are omitted from those panels.
5. **WS singleton diagnostic status.** The weak--strong singleton plateau trajectory failed only the G5 plateau-domain exercise guard and remains a qualified diagnostic observation.
6. **No uncertainty inference.** These are deterministic same-seed, same-optimizer trajectory comparisons. Counts and ratios are descriptive and do not establish statistical significance.

## Provenance

- Evolving stationary-core report: `output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_partial_progress.pdf`.
- Evolving stationary-core provenance: `output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_partial_progress_provenance.json`, SHA-256 `09317815aec1d3b7794f025084ab1173b7270f784d2e4d689b237c840cc15ebc`.
- Page-1 paused-prefix provenance: `output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_partial_progress_local_paused_prefix_page1_provenance.json`, SHA-256 `f03966a7eec9b1919d17893b155d3afac38f3f03ca762df53b8ee8cbc510860d`.
- Recovery adapter used for weak--weak macro always insertion, weak--weak/intermediate--weak singleton always insertion, and qualified weak--strong singleton plateau insertion: `raw_outputs/paper_i_ra_adapt_stationary_core_recovery_20260730/recovery_adapter.json`, SHA-256 `4d3f4bded2f7fbde1965df24c4ced3119cd07927dfd82582a339dc5a35e6e485`.
- Stationary curve caches: `MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723/paper_i_hh_macro_common_accuracy_20260723_stationary_page1_macro_curve_cache.json` and `MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723/paper_i_hh_macro_common_accuracy_20260723_stationary_page2_singleton_curve_cache.json`.
- V7 singleton no-insertion processes: 5 (WW), 13 (IW), 21 (SW), 29 (WS), 37 (IS), and 45 (SS). Plateau processes: 6, 14, 22, 30, 38, and 46, with WS plotted from the explicit G5-qualified recovery adapter.
- Canonical singleton Append registry: `agent_guidance/static-adapt/reporting/canonical-append-registry-v1.json`, SHA-256 `2d59ee3d92ccf79d7c8f5fa826516159576220011872e7d1142a6b5b612f722a`.
- Campaign disposition audit: `output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/chtc_campaign_audit_20260730.md`.
- Selection implementation: `select_paper_i_effective_plateau` and `select_paper_i_common_accuracy` in `pipelines/reporting/paper_i_run_summary.py`.
- Prefix reconstruction and compilation: `_ra_prefix_from_archive`, `_append_prefix_from_archive`, and `_compile_cost` in `pipelines/reporting/build_paper_i_ra_vs_adapt_common_accuracy_cost_pdf.py`.
