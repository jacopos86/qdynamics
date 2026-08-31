# Exploratory early-pruning sweep

This sweep is diagnostic evidence, not a source of paper-facing values.  The
canonical `metric_ablation_gated` result remains the untuned reference.

## Fixed profile

- Pure Hubbard: open, half-filled `L=3`, `U/t=1.25`.
- Horizon: 12 controller rounds.
- Global 24-generator pool, active Phase 0, adaptive `gamma=0.9` shortlists.
- Exact Qiskit compiled marginal costs in every scored phase.
- Fubini--Study-whitened accepted refits; Powell `xtol=1e-5`, `ftol=1e-12`.
- Delete--refit verification remains enabled.

## Predeclared variants

| Label | `small_theta_abs` | `protect_steps` | `max_regression` | retained-gain ratio | Purpose |
|---|---:|---:|---:|---:|---|
| canonical | `1e-3` | 1 | `1e-8` | 0.5 | Locked reference; reuse existing run. |
| protect0 | `1e-3` | 0 | `1e-8` | 0.5 | Remove the one-round admission protection. |
| theta1e-2_protect0 | `1e-2` | 0 | `1e-8` | 0.5 | Moderately widen the amplitude screen. |
| theta1e-1_protect0 | `1e-1` | 0 | `1e-8` | 0.5 | Aggressively widen the amplitude screen. |
| theta1e-1_protect1 | `1e-1` | 1 | `1e-8` | 0.5 | Isolate the effect of protection at the wide screen. |
| theta1e-1_eps1e-6 | `1e-1` | 0 | `1e-6` | 0.5 | Test a looser absolute regression gate. |
| theta1e-1_zeta025 | `1e-1` | 0 | `1e-8` | 0.25 | Test a looser retained-gain gate. |

`protect_steps` is the relevant patience control for the first live deletion.
The internal checkpoint period does not delay a live pass after an accepted
admission, because accepted admission itself opens the pruning cadence.

## Evaluation rule

Report the first delete--refit trial and first accepted deletion separately.
The primary endpoint is the earliest accepted deletion subject to final
absolute energy error no larger than `1e-12`.  Report `N2q`, `D2q`, `Dc`, and
`S_alg` at round 12.  No variant is promoted from this sweep solely because it
looks best; any promoted setting requires a fresh confirmatory run declared
after this sweep.
