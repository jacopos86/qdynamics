# Paper I Data Staging: HH k_pl Markers and Fidelity

Generated: 2026-06-19

This is a support dossier only. It does not update `MATH/paper_details/static_adapt_paper_I.tex`, does not replace manuscript figures, and does not rebuild the Paper-I PDF.

## Generated Support Artifacts

- HH fidelity/marker audit JSON: `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_native200_kpl_fidelity_marker_audit_20260619.json`
- HH fidelity/marker audit CSV: `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_native200_kpl_fidelity_marker_audit_20260619.csv`
- Same-cutoff HH exact-state cache: `MATH/paper_facing/paper_I_static_scaffold/cache/paper_i_hh_same_cutoff_exact_state_cache_20260619.npz`
- Same-cutoff HH exact-state cache manifest: `MATH/paper_facing/paper_I_static_scaffold/cache/paper_i_hh_same_cutoff_exact_state_cache_20260619.json`
- Source HH plateau audit: `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_native200_first_plateau_prefix_audit_20260619.json`
- Source HH manuscript-support JSON: `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_native200_manuscript_update_20260619.json`

## Exact-State Cache

The reusable object is the same-cutoff exact ED state, not the fidelity scalar. The cache stores one exact state vector for each displayed Hubbard--Holstein regime and cutoff; new ansatz data can reuse these states and only recompute the overlap. The cache should be regenerated only if the Hamiltonian, cutoff, profile, encoding, or comparison space changes.

## Current Paper-I Visible Data Snapshot

### Hubbard Model

Weak column: `U/t=0.5`; strong column: `U/t=1.5`.

| Method | weak |Delta E| | weak 1-F | weak N2q | weak D2q | weak Dc | weak S | strong |Delta E| | strong 1-F | strong N2q | strong D2q | strong Dc | strong S |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HEA VQE | 3.10e-8 | -- | 6 | 5 | 35 | 48000 | 5.86e-8 | -- | 6 | 5 | 35 | 48000 |
| family VQE | 2.81e-8 | -- | 56 | 52 | 278 | 48000 | 5.92e-8 | -- | 56 | 52 | 278 | 48000 |
| Append | 1.56e-2 | -- | 1948 | 1863 | 9942 | 1100000 | 8.69e-5 | -- | 168 | 132 | 848 | 15300 |
| TETRIS-ADAPT | 1.56e-2 | -- | 1524 | 1412 | 7846 | 1090000 | 1.36e-1 | -- | 180 | 138 | 940 | 62800 |
| Geo | 3.15e-4 | -- | 1816 | 1752 | 9024 | 30400 | 5.23e-6 | -- | 432 | 416 | 2176 | 19700 |
| Qubit/QEB | 1.82e-4 | -- | 1552 | 1459 | 7800 | 19100 | 2.14e-7 | -- | 388 | 367 | 1958 | 14100 |
| Snake | 6.66e-16 | 0 | 56 | 52 | 219 | 21919 | 1.33e-14 | 0 | 56 | 52 | 219 | 21919 |

### Spin-Boson/Rabi Model

Weak column: `g/omega0=0.05`, `n_ph=1`; strong column: `g/omega0=0.1`, `n_ph=2`.

| Method | weak |Delta E| | weak 1-F | weak N2q | weak D2q | weak Dc | weak S | strong |Delta E| | strong 1-F | strong N2q | strong D2q | strong Dc | strong S |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HEA VQE | 3.05e-8 | -- | 6 | 5 | 35 | 48000 | 4.87e-3 | -- | 10 | 7 | 37 | 48000 |
| family VQE | 1.67e-3 | -- | 31 | 29 | 245 | 48000 | 6.70e-3 | -- | 101 | 78 | 609 | 48000 |
| Append | 6.16e-7 | -- | 91 | 61 | 717 | 184000 | 1.04e-5 | -- | 227 | 201 | 1596 | 4067 |
| TETRIS-ADAPT | 6.16e-7 | -- | 119 | 107 | 750 | 173390 | 9.97e-6 | -- | 711 | 623 | 3544 | 5823 |
| Geo | 6.16e-7 | -- | 91 | 89 | 1095 | 23600 | 9.97e-6 | -- | 565 | 549 | 3944 | 55600 |
| Qubit/QEB | 1.67e-3 | -- | 971 | 971 | 4477 | 300000 | 6.70e-3 | -- | 1017 | 1017 | 4753 | 301000 |
| Snake | 6.16e-7 | 3.08e-7 | 34 | 34 | 124 | 30043 | 9.97e-6 | 5.01e-6 | 213 | 198 | 725 | 38440 |

### Hubbard-Holstein Current Table Snapshot

#### weak--weak $(U/t,\lambda)=(0.25,0.25)$, $M=2$
| Method | k_pl | |Delta E| | N2q | D2q | Dc | S |
| --- | --- | --- | --- | --- | --- | --- |
| Append | 4 | 2.82e-3 | 96 | 68 | 487 | 1970 |
| Geo | 9 | 2.87e-2 | 2560 | 2553 | 8595 | 43033 |
| SNAKE | 13 | 4.71e-4 | 116 | 107 | 746 | 23069 |

#### intermediate--weak $(U/t,\lambda)=(1.25,0.25)$, $M=2$
| Method | k_pl | |Delta E| | N2q | D2q | Dc | S |
| --- | --- | --- | --- | --- | --- | --- |
| Append | 5 | 7.83e-2 | 124 | 96 | 695 | 2460 |
| Geo | 7 | 1.29e-2 | 214 | 210 | 1147 | 33491 |
| SNAKE | 10 | 4.64e-4 | 86 | 80 | 577 | 15374 |

#### strong--weak $(U/t,\lambda)=(8,0.25)$, $M=2$
| Method | k_pl | |Delta E| | N2q | D2q | Dc | S |
| --- | --- | --- | --- | --- | --- | --- |
| Append | 15 | 2.49e-4 | 3312 | 3260 | 11793 | 7305 |
| Geo | 24 | 1.35e-4 | 1056 | 1053 | 5169 | 114598 |
| SNAKE | 16 | 1.02e-3 | 48 | 47 | 330 | 38664 |

#### weak--strong $(U/t,\lambda)=(0.25,1.25)$, $M=4$
| Method | k_pl | |Delta E| | N2q | D2q | Dc | S |
| --- | --- | --- | --- | --- | --- | --- |
| Append | 8 | 4.52e-2 | 2180 | 1985 | 7584 | 3972 |
| Geo | 5 | 5.02e-2 | 328 | 206 | 1149 | 26850 |
| SNAKE | 17 | 1.90e-2 | 752 | 743 | 4117 | 21207 |

#### intermediate--strong $(U/t,\lambda)=(1.25,1.25)$, $M=4$
| Method | k_pl | |Delta E| | N2q | D2q | Dc | S |
| --- | --- | --- | --- | --- | --- | --- |
| Append | 13 | 2.83e-2 | 9948 | 9681 | 32968 | 6422 |
| Geo | 17 | 9.79e-3 | 1496 | 1213 | 6469 | 91050 |
| SNAKE | 17 | 8.70e-3 | 972 | 965 | 5402 | 16262 |

#### strong--strong $(U/t,\lambda)=(8,1.25)$, $M=4$
| Method | k_pl | |Delta E| | N2q | D2q | Dc | S |
| --- | --- | --- | --- | --- | --- | --- |
| Append | 12 | 1.88e-3 | 8632 | 8527 | 29691 | 5934 |
| Geo | 17 | 1.80e-4 | 1372 | 1147 | 6210 | 91050 |
| SNAKE | 14 | 5.78e-5 | 1512 | 1361 | 7694 | 5907 |

## HH k_pl Marker and Fidelity Audit

Marker coordinates are the candidate `k_pl` display coordinate on the same-cutoff error curve. Fidelity is same-cutoff `1-F`. For non-SNAKE rows, fidelity is recomputed by replaying the accepted operator prefix at `k_pl`; for SNAKE rows, fidelity uses the source-stopped displayed ansatz `adapt_vqe.exact_state_fidelity` field because separate per-prefix SNAKE state/cost sidecars are not available in the retrieved artifacts.

### weak--weak
Preview plot: `MATH/paper_facing/paper_I_static_scaffold/preview_plots/paper_i_hh_native200_kpl_marker_weak_weak_20260619.png`

| Method | marker k_pl | marker |Delta E| | 1-F | fidelity status | N2q | D2q | Dc | S |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Append | 4 | 0.00282 | 0.00114 | computed | 96 | 68 | 487 | 1970 |
| Geo | 9 | 0.0287 | 0.0107 | computed | 2560 | 2553 | 8595 | 43033 |
| SNAKE | 13 | 0.000471 | 0.000216 | computed | 116 | 107 | 746 | 23069 |

### intermediate--weak
Preview plot: `MATH/paper_facing/paper_I_static_scaffold/preview_plots/paper_i_hh_native200_kpl_marker_intermediate_weak_20260619.png`

| Method | marker k_pl | marker |Delta E| | 1-F | fidelity status | N2q | D2q | Dc | S |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Append | 5 | 0.0783 | 0.0197 | computed | 124 | 96 | 695 | 2460 |
| Geo | 7 | 0.0129 | 0.00392 | computed | 214 | 210 | 1147 | 33491 |
| SNAKE | 10 | 0.000464 | 0.00014 | computed | 86 | 80 | 577 | 15374 |

### strong--weak
Preview plot: `MATH/paper_facing/paper_I_static_scaffold/preview_plots/paper_i_hh_native200_kpl_marker_strong_weak_20260619.png`

| Method | marker k_pl | marker |Delta E| | 1-F | fidelity status | N2q | D2q | Dc | S |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Append | 15 | 0.000249 | 0.000203 | computed | 3312 | 3260 | 11793 | 7305 |
| Geo | 24 | 0.000135 | 1.72e-05 | computed | 1056 | 1053 | 5169 | 114598 |
| SNAKE | 16 | 0.00102 | 0.000256 | computed | 48 | 47 | 330 | 38664 |

### weak--strong
Preview plot: `MATH/paper_facing/paper_I_static_scaffold/preview_plots/paper_i_hh_native200_kpl_marker_weak_strong_20260619.png`

| Method | marker k_pl | marker |Delta E| | 1-F | fidelity status | N2q | D2q | Dc | S |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Append | 8 | 0.0452 | 0.0328 | computed | 2180 | 1985 | 7584 | 3972 |
| Geo | 5 | 0.0502 | 0.0351 | computed | 328 | 206 | 1149 | 26850 |
| SNAKE | 17 | 0.019 | 0.0119 | computed | 752 | 743 | 4117 | 21207 |

### intermediate--strong
Preview plot: `MATH/paper_facing/paper_I_static_scaffold/preview_plots/paper_i_hh_native200_kpl_marker_intermediate_strong_20260619.png`

| Method | marker k_pl | marker |Delta E| | 1-F | fidelity status | N2q | D2q | Dc | S |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Append | 13 | 0.0283 | 0.00973 | computed | 9948 | 9681 | 32968 | 6422 |
| Geo | 17 | 0.00979 | 0.00692 | computed | 1496 | 1213 | 6469 | 91050 |
| SNAKE | 17 | 0.0087 | 0.00509 | computed | 972 | 965 | 5402 | 16262 |

### strong--strong
Preview plot: `MATH/paper_facing/paper_I_static_scaffold/preview_plots/paper_i_hh_native200_kpl_marker_strong_strong_20260619.png`

| Method | marker k_pl | marker |Delta E| | 1-F | fidelity status | N2q | D2q | Dc | S |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Append | 12 | 0.00188 | 0.00114 | computed | 8632 | 8527 | 29691 | 5934 |
| Geo | 17 | 0.00018 | 3.82e-05 | computed | 1372 | 1147 | 6210 | 91050 |
| SNAKE | 14 | 5.78e-05 | 1.5e-05 | computed | 1512 | 1361 | 7694 | 5907 |

## Preview Plot Links

- weak--weak: `MATH/paper_facing/paper_I_static_scaffold/preview_plots/paper_i_hh_native200_kpl_marker_weak_weak_20260619.png`
  ![weak--weak](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/MATH/paper_facing/paper_I_static_scaffold/preview_plots/paper_i_hh_native200_kpl_marker_weak_weak_20260619.png)
- intermediate--weak: `MATH/paper_facing/paper_I_static_scaffold/preview_plots/paper_i_hh_native200_kpl_marker_intermediate_weak_20260619.png`
  ![intermediate--weak](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/MATH/paper_facing/paper_I_static_scaffold/preview_plots/paper_i_hh_native200_kpl_marker_intermediate_weak_20260619.png)
- strong--weak: `MATH/paper_facing/paper_I_static_scaffold/preview_plots/paper_i_hh_native200_kpl_marker_strong_weak_20260619.png`
  ![strong--weak](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/MATH/paper_facing/paper_I_static_scaffold/preview_plots/paper_i_hh_native200_kpl_marker_strong_weak_20260619.png)
- weak--strong: `MATH/paper_facing/paper_I_static_scaffold/preview_plots/paper_i_hh_native200_kpl_marker_weak_strong_20260619.png`
  ![weak--strong](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/MATH/paper_facing/paper_I_static_scaffold/preview_plots/paper_i_hh_native200_kpl_marker_weak_strong_20260619.png)
- intermediate--strong: `MATH/paper_facing/paper_I_static_scaffold/preview_plots/paper_i_hh_native200_kpl_marker_intermediate_strong_20260619.png`
  ![intermediate--strong](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/MATH/paper_facing/paper_I_static_scaffold/preview_plots/paper_i_hh_native200_kpl_marker_intermediate_strong_20260619.png)
- strong--strong: `MATH/paper_facing/paper_I_static_scaffold/preview_plots/paper_i_hh_native200_kpl_marker_strong_strong_20260619.png`
  ![strong--strong](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/MATH/paper_facing/paper_I_static_scaffold/preview_plots/paper_i_hh_native200_kpl_marker_strong_strong_20260619.png)

## Notes

- The HH marker/fidelity values are staged for possible reuse in the next paper update; no automatic manuscript transfer is implied.
- Current manuscript snapshots above are copied for convenience and may become stale when new results replace the paper tables.
- `S` values are rendered as ordinary decimal numerals.
