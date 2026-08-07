# Archive spectral comparison routes

Use the following names consistently.  The phrase `cone correction` by itself
is forbidden because two distinct propagated models contain cone operations.

| route key | plot label | propagated state | online dynamics or correction |
|---|---|---|---|
| `exact_cutoff16` | exact cutoff-16 Hamiltonian | full cutoff-16 wavefunction | DOP853 Schrodinger propagation; offline reference only |
| `archive_eom` | archive EOM (uncorrected) | 31 real archive moments | archive moment EOM without a physicality correction |
| `regular_eom_correction` | regular EOM correction (31D joint-Gram) | the same 31 real archive moments | archive EOM plus the minimum-Euclidean-norm velocity correction enforcing the retained electronic/joint-Gram, correlation-trace, and correction-energy conditions |
| `apcm_m4_prototype` | McLachlan-type M4 correction (APCM prototype) | 60 real coordinates: the raw archive chart plus preparation-dependent relative-mode moments | implemented entrance-layer source repair, positive `M4` completion, hidden-stage retraction, and retained joint-Gram controller |

The last route is the implemented APCM entrance-layer prototype.  It does not
contain the full adaptive moment-metric McLachlan projection proposed in the
design memorandum, so plots and prose must retain the word `prototype`.

## Canonical four-route spectrum

The common stored horizon is set by the `M4` prototype, which ends at `t=20`.
Use the post-pulse interval `4 <= t <= 20`, common sampling `0.2`, mean
subtraction, and a Hann window.  This gives angular-frequency spacing
`0.387851 t_hop`; widths from this comparison are observed finite-window FWHM
values and are resolution dominated.  Do not call them lifetimes or
deconvolved broadening.

Regenerate the plot and its route/provenance manifest with:

```bash
MPLCONFIGDIR=/tmp/paper-v-mpl-cache PYTHONPATH=paper_5/src:. python3 \
  pipelines/open_dynamics/analyze_archive_m4_polarization_spectra.py \
  output/local_runs/paper_v_archive_observable_trajectories_t1000_20260803_v1 \
  output/local_runs/paper_v_apcm_spin_exchange_blocks_controller_t20_h0025_20260805_v1 \
  output/plots/paper_v_results_progression_20260804 \
  --prefix archive_m4_four_route_polarization_spectrum
```

The generated JSON is the machine-readable source manifest:

`output/plots/paper_v_results_progression_20260804/archive_m4_four_route_polarization_spectrum.json`.

The separate weak/strong three-route analysis uses the longer common interval
`10 <= t <= 100` and has finer frequency spacing.  It answers the coupling
trend question but contains no `M4` prototype and must not be presented as the
four-route comparison.
