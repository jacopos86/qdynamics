# L=3 early-pruning sweep results

Status: exploratory sweep complete; selected setting independently confirmed.

Source commit: `618b3bf605720cd248ea7ea96af16b1c283d949b` plus the
uncommitted Paper-I L=3 cost-route work recorded in this directory.

All estimator totals below use the Paper-I phase-reuse convention: Hamiltonian
objective occurrences remain charged, while staged gradient and metric
primitives already available at the same mathematical state are reused.
Rejected delete--refit siblings remain charged as executed work.

## Results at round 12

| Variant | First trial | First accepted deletion | Accepted deletions | `N2q` | `D2q` | `Dc` | corrected `S_alg` | Final absolute error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Canonical: `theta=1e-3`, protect 1 | 11 | 11 | 1 | 292 | 280 | 1148 | 8570 | `1.78e-15` |
| `theta=1e-3`, protect 0 | 10 | 10 | 3 | 276 | 264 | 1083 | 8676 | `1.78e-15` |
| `theta=1e-2`, protect 0 | 10 | 10 | 3 | 276 | 264 | 1083 | 8676 | `1.78e-15` |
| `theta=1e-1`, protect 0 | 5 | 10 | 3 | 276 | 264 | 1083 | 9837 | `1.78e-15` |
| `theta=1e-1`, protect 1 | 9 | 11 | 1 | 292 | 280 | 1148 | 10689 | `1.78e-15` |
| `theta=1e-1`, protect 0, `epsilon=1e-6` | 5 | 10 | 3 | 276 | 264 | 1083 | 9837 | `1.78e-15` |
| `theta=1e-1`, protect 0, retained ratio 0.25 | 5 | 10 | 3 | 276 | 264 | 1083 | 9837 | `1.78e-15` |

The matched no-prune RA-Append endpoint is `300/288/1182` with corrected
`S_alg=7749` and zero displayed absolute error.

## Interpretation

- Removing the one-round admission-protection window is the useful change.
  It advances the first accepted deletion from round 11 to round 10 and then
  accepts deletions in rounds 10--12.
- Relative to canonical pruning, the protect-0 route lowers `N2q`, `D2q`, and
  `Dc` by 5.5%, 5.7%, and 5.7%, respectively, for 1.2% more corrected
  estimator work.
- Relative to RA-Append, it lowers the same circuit measures by 8.0%, 8.3%,
  and 8.4%, while using 12.0% more corrected estimator work.
- Widening the amplitude screen to `1e-1` causes early delete--refit trials at
  rounds 5 and 9, but both are rejected because they remove real energy gain.
  It does not advance the first safe deletion and increases estimator work.
- Raising the absolute tolerance to `1e-6` or reducing the retained-gain ratio
  to 0.25 does not change a decision in this trajectory.
- The internal checkpoint period is not the effective patience control for
  these live passes: every accepted admission already opens the pruning
  cadence.  The admission-protection window is the controlling delay.

## Confirmation

The selected `theta=1e-3`, protect-0 setting was rerun after the exploratory
sweep under `prune_protect0_confirm_k12`.  It reproduced the three deletion
rounds, final error, circuit tuple, raw estimator total, and corrected estimator
total exactly.

- Confirmation result SHA-256:
  `73c30af203d02e8d22c1dbc6917d04bc4cd2bda23b18b524f366f608fbd95cbf`
- Matched canonical result SHA-256:
  `af48059a8ce032a55993ef0e324b1c887c87cad04d2197d5a94f1a889973fabf`
- Sweep summary SHA-256:
  `625044462972d10fb1f20bec8884dc90f7e6b888e9009c3cd2f1450cbd0e029b`
