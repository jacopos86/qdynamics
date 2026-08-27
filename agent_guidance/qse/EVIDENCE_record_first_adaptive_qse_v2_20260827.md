# Record-first adaptive QSE v2 evidence

Date: 2026-08-27
Branch: `paper-ii-exchange-selector`
Status: verified local statevector comparison used for the authorized Paper III
method/results synchronization in this increment.

## Canonical construction

`record_first_adaptive_qse_v2` is the complete method.

1. Acquire operator-generated records with the production two-term score
   `S = Δ/K` and apply certified exchange at every recorded support.
2. If the record-selector receipt is `residual_converged`, return the maintained
   record basis without invoking residual synthesis.
3. If the receipt is `pool_exhausted`, preserve the actual exchange-maintained
   terminal support and use it as the seed for preconditioned Davidson residual
   synthesis.
4. Stop residual synthesis at its declared residual tolerance or dimension cap.

Exact reference energies annotate the trace and determine the post-run
fixed-accuracy crossings. They never choose a record, trigger the handoff, or
stop either phase. `record_only` is the fixed-alphabet ablation.

## Behavior verification

- Focused method/comparator and selector behavior tests: 32 passed.
- nph1 support behavior lock: 93/93 acquisition-prefix supports were
  bit-identical between the `HEAD` implementation and this increment. For
  every prefix of each regime's recorded acquisition order, both versions
  reran certified exchange and returned the same ordered support. Neither the
  production record selector nor `exchange_maintenance.py` was modified.
- At `n_ph_max=1`, all six regimes remain in the record phase.
- At `n_ph_max=3`, the fallback fires only in the two diagnosed
  fixed-alphabet ceiling regimes:

| regime | record receipt | handoff support | record-only terminal error | record-first terminal error |
|---|---:|---:|---:|---:|
| weak--strong | `pool_exhausted` | 63 | 1.79884e-2 | 1.88738e-15 |
| intermediate--strong | `pool_exhausted` | 57 | 1.15374e-2 | 3.96905e-15 |

The other four `n_ph_max=3` regimes reach the `1e-6` energy target without
residual synthesis. No Δ/K retuning is used.

## Corrected compiled two-qubit crossings

Cells are `N2q` at the first post-run fixed-accuracy crossing. Record seeds
retain their actual compiled record costs; only residual-synthesized directions
are charged as first-order Hamiltonian directions.

| cutoff | regime | method | 1e-2 | 1e-4 | 1e-6 |
|---:|---|---|---:|---:|---:|
| 1 | weak--weak | record-first | 32 | 32 | 32 |
| 1 | weak--weak | direct Davidson | 156 | 176 | 176 |
| 1 | intermediate--weak | record-first | 26 | 26 | 26 |
| 1 | intermediate--weak | direct Davidson | 156 | 156 | 156 |
| 1 | strong--weak | record-first | 26 | 26 | 26 |
| 1 | strong--weak | direct Davidson | 156 | 156 | 176 |
| 1 | weak--strong | record-first | 28 | 28 | 28 |
| 1 | weak--strong | direct Davidson | 156 | 176 | 176 |
| 1 | intermediate--strong | record-first | 26 | 34 | 34 |
| 1 | intermediate--strong | direct Davidson | 156 | 156 | 156 |
| 1 | strong--strong | record-first | 38 | 38 | 38 |
| 1 | strong--strong | direct Davidson | 156 | 176 | 176 |
| 3 | weak--weak | record-first | 618 | 618 | 1210 |
| 3 | weak--weak | direct Davidson | 240 | 2688 | 2960 |
| 3 | intermediate--weak | record-first | 294 | 294 | 294 |
| 3 | intermediate--weak | direct Davidson | 240 | 2348 | 2552 |
| 3 | strong--weak | record-first | 276 | 276 | 322 |
| 3 | strong--weak | direct Davidson | 2416 | 2484 | 2892 |
| 3 | weak--strong | record-first | 4818 | 4886 | 5022 |
| 3 | weak--strong | direct Davidson | 444 | 920 | 1532 |
| 3 | intermediate--strong | record-first | 4324 | 4392 | 5072 |
| 3 | intermediate--strong | direct Davidson | 2212 | 2620 | 2824 |
| 3 | strong--strong | record-first | 258 | 258 | 258 |
| 3 | strong--strong | direct Davidson | 2484 | 2892 | 2892 |

Interpretation:

- At `n_ph_max=1`, the record phase is 4.1--6.8 times cheaper in `N2q`
  than direct Davidson over the reported targets.
- At `n_ph_max=3` and targets `1e-4` and `1e-6`, the record phase is cheaper
  in four of six regimes, by factors 2.4--11.2.
- In the two fallback regimes, completeness is recovered but costs 1.7--5.3
  times more `N2q` than starting with direct Davidson.
- At the loose `1e-2` target, direct Davidson is cheaper in four of six
  `n_ph_max=3` regimes. The method therefore has a conditional record-economy
  advantage, not universal cost dominance.

## Source records

- `output/diagnostics/paper_iii_record_first_adaptive_qse_v2/growth_trace_nph1.json`
- `output/diagnostics/paper_iii_record_first_adaptive_qse_v2/growth_trace_nph3.json`
- `output/diagnostics/paper_iii_record_first_adaptive_qse_v2/growth_trace_nph3_cap64_refresh.json`
- `output/diagnostics/paper_iii_record_first_adaptive_qse_v2/growth_trace_nph3_canonical.json`

The consolidated `n_ph_max=3` record retains per-regime dimension caps. The
three cap-64 refresh regimes are weak--weak, weak--strong, and
intermediate--strong. The other three cap-60 traces are retained only after
all three reported energy targets were already crossed. Krylov uses its
separately declared dimension cap of 40 throughout.
