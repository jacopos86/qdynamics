# Evidence — the energy window metric is partly measuring invisible states

Date: 2026-08-27
Branch: `paper-ii-exchange-selector`
Status: diagnostic finding on the `n_ph_max=1` tier. Affects interpretation
of `C*(eps_E)`, not any committed number.

## Question that prompted it

Do the comparator methods (direct Davidson, real-time Krylov) need their own
transition/Green-function products, or is the energy-only comparison enough?
The decision hinges on whether matching on energy accuracy also matches on
spectral weight: energies are second-order accurate in wavefunction error,
transition amplitudes only first-order.

## Method

Along the canonical record acquisition order (with certified exchange at every
prefix), compare at each `k`:

- `max_r |E_r^QSE - E_r^exact|` over the six target roots -- the metric
  `C*(eps_E)` is read from;
- spectral-weight error `sum_r |S_r^QSE - S_r^exact| / sum_r S_r^exact`,
  where `S_r = |<Psi_r|O|psi_0>|^2` is what actually builds a spectral
  function.

Probes: one `uccsd_ferm_lifted` and one `hh_phonon` pool operator per regime.
**These are illustrative probes, not a declared spectroscopic observable set**
-- which roots count as dark is observable-dependent, so the magnitudes below
are the mechanism, not paper-grade numbers.

## Finding: the two metrics decouple, non-monotonically

`weak_weak`, `n_ph_max=1`, exact strengths per root:

| root | E_exact | uccsd probe | phonon probe |
|---:|---:|---:|---:|
| 1 | -0.012374 | 1.41e-2 | 5.11e-1 |
| 2 | 0.121098 | ~1e-31 | 4.75e-1 |
| 3 | 1.000000 | 4.73e-1 | ~1e-32 |
| 4 | 1.022884 | ~1e-29 | ~1e-32 |
| 5 | 1.301002 | 5.12e-1 | 1.41e-2 |
| 6 | 2.000000 | ~1e-32 | ~1e-32 |

Roots 4 and 6 are **dark**: no probe couples to them. Then:

| k | N2q | max\|dE\| (all roots) | weight error | what is actually wrong |
|---:|---:|---:|---:|---|
| 12 | 24 | **2.07e-1** | **2e-14 / 3e-15** | only dark roots 4, 6 are off; every bright root is exact to 1e-15 |
| 13 | 28 | 1.96e-4 | 7.5e-4 / 3.4e-4 | dark root 6 fixed, but all four bright roots *degrade* |
| 14 | 32 | 6.3e-14 | 1.6e-14 | everything converged |

At `k=12` the subspace is spectroscopically exact -- every root any observable
can see has both its energy and its weight right to machine precision -- while
the reported window error is `2e-1`, driven entirely by a dark root at
`E=2.0`. At `k=13` the energy metric improves by three orders while the
spectral weights get ten orders *worse*.

## How much it costs us, across the tier

`eps_E = 1e-6`, all six `n_ph_max=1` regimes; `k(all)` is the current
crossing, `k(bright)` the crossing over weight-carrying roots only:

| regime | dark roots | k(all) | N2q | k(bright) | N2q | overcharge |
|---|---:|---:|---:|---:|---:|---:|
| weak_weak | 2 | 14 | 32 | 12 | 24 | **1.33x** |
| intermediate_weak | 3 | 16 | 26 | 16 | 26 | 1.00x |
| strong_weak_u8 | 3 | 14 | 26 | 11 | 22 | **1.18x** |
| weak_strong | 2 | 14 | 32 | 14 | 32 | 1.00x |
| intermediate_strong | 3 | 16 | 34 | 16 | 34 | 1.00x |
| strong_strong_u8 | 3 | 15 | 38 | 15 | 38 | 1.00x |

Dark roots exist in every regime (2--3 of 6), but they set the crossing in
only two of six, at 18--33% extra `N2q`.

## Consequences

1. **The current metric is conservative, not wrong.** It charges our method
   for resolving states no observable can see, so reported `N2q` is an upper
   bound. Existing headline wins (4.1--6.8x at `n_ph_max=1`) survive
   unchanged -- they would only widen.
2. **Comparators do not need spectral products for the present claim.** The
   paper compares excitation energies; the trace stage already serves all
   four public methods. Green/spectral functions remain our own downstream
   deliverable.
3. **They would be needed if the claim becomes spectroscopic.** "Cheaper
   spectral functions" cannot be inferred from an energy-matched crossing,
   because the two metrics demonstrably decouple at fixed `k`.
4. Energy convergence is **not monotone in spectral weight** (`k=12` -> `13`
   above). Any weight-aware criterion must be read from the whole trace, the
   same way `C*(eps_E)` already is.

## Open, if the author wants this paper-grade

- Declare the actual spectroscopic observable set for Hubbard--Holstein; the
  dark/bright split above is only as meaningful as the probes chosen.
- Decide whether to keep the all-root metric (simplest, defensible,
  self-penalizing) or add a weight-aware crossing alongside it.

Reproduce: `pipelines/qse_spectra/paper_iii_problem.py` provider plus
`compute_qse_spectra` with transition observables over exchange-maintained
prefixes; no new machinery required.
