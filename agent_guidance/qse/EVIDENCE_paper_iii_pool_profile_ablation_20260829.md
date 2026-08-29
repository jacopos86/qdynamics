# Paper-III fixed-pool profile ablation evidence — 2026-08-29

## Decision

The declared encoded-boson completion pool is the first fixed pool in this
lane to pass the full 12-cell coverage gate. With the locked full-proposed-
pencil Ky Fan selector, it certifies the target window and reaches the
post-run exact-error target of $10^{-4}$ in all 12 Hubbard--Holstein cells.
The original macro pool does both in 10/12 cells and terminates as
`POOL_EXHAUSTED` in `weak_strong_nph3` and `intermediate_strong_nph3`.

This evidence supports using the completion alphabet when full fixed-pool
coverage is required. It does not by itself choose between always exposing
that alphabet and a separately declared staged base-to-completion method:
always-on completion costs more than direct Davidson at $10^{-4}$ in two
cells, while the base alphabet is cheaper in several easy cells.

## Locked implementation and campaign

- Branch: `paper-ii-exchange-selector`
- Source revision: `4542a895b7670cfd4233f67b1ed4447f05eec510`
- Source revision subject: `paper-iii: automate fixed-pool profile evidence`
- Selector policy:
  `full_pencil_ky_fan_cost_ratio_novelty_floor_no_reward_v1`
- Selected-QSE construction policy:
  `selected_qse_fixed_pool_full_pencil_ky_fan_v2`
- Direct comparator construction policy:
  `direct_davidson_fixed_lr_plus_r_plus_1_low_diagonal_sector_v1`
- Source-lock digest:
  `096888afb3a211fbee75ab50f89d63d2d7b30bcf67320c6821d87b3aa3f64080`
- Target window: roots $r=1,\ldots,6$
- Online residual threshold: $10^{-3}$
- Post-run exact-error targets: $10^{-2},10^{-4},10^{-6}$
- Dimension cap: 64
- Primary reported cost: compiled two-qubit count $N_{2q}$
- Routes: 2 pool profiles x 12 cells x 2 methods = 48

The selection score is the full proposed-pencil Ky Fan decrease per
incremental cost,

\[
  \mathcal S_\ell(A)
  = \frac{F_{\mathcal R}(\mathcal B_\ell)
  -F_{\mathcal R}(\mathcal B_\ell\cup\{A\})}{\Delta K_\ell(A)}.
\]

The supported-image novelty floor remains an admissibility guard. There is no
additive novelty reward. Exact roots are post-run audit data and select no
record, stop no route, and rewrite no terminal status.

## Pool identities

| Policy | nph1 records | nph3 records | Construction |
|---|---:|---:|---|
| `hh_full_meta_macro_v1` | 94 | 128 | Shared provider's original full-meta macro pool |
| `hh_full_meta_plus_encoded_boson_pauli_completion_v1` | 169 | 1403 | Original pool as an unchanged prefix, plus 75/1275 deterministic sector-preserving encoded-boson Pauli records |

The completion enumerates every nonidentity Pauli word on the encoded boson
register, tensored with fermion identity and with each one-body fermion
$Z_j$. It does not flip fermion occupation sectors. Its added record costs
use the same Marrakesh graph-span two-qubit estimator as the base pool.

Direct Davidson always uses the unmodified problem provider and is independent
of the selected-QSE pool profile. Its online rows, exact-audit crossings, and
pool digest were pairwise identical across the two independently executed
profiles in all 12 cases.

## Coverage and cost results

At the declared $10^{-4}$ lock target:

| Selected pool | Online certified | Target reached | No more expensive than direct among comparable cells | Maximum selected/direct $N_{2q}$ ratio | Coverage-qualified |
|---|---:|---:|---:|---:|---|
| Base macro | 10/12 | 10/12 | 7/10 | 1.992 | no |
| Encoded-boson completion | 12/12 | 12/12 | 10/12 | 1.407 | yes |

Per-cell results follow. A dash means that the route did not reach that exact-
error target; the online status remains independent of that audit.

| Case | Base online status | Base $N_{2q}@10^{-4}$ | Completion $N_{2q}@10^{-4}$ | Direct $N_{2q}@10^{-4}$ | Completion/direct | Completion $N_{2q}@10^{-6}$ | Direct $N_{2q}@10^{-6}$ |
|---|---|---:|---:|---:|---:|---:|---:|
| `weak_weak_nph1` | `TARGET_WINDOW_CERTIFIED` | 18 | 18 | 76 | 0.237 | 24 | 76 |
| `intermediate_weak_nph1` | `TARGET_WINDOW_CERTIFIED` | 22 | 26 | 56 | 0.464 | 28 | 56 |
| `strong_weak_u8_nph1` | `TARGET_WINDOW_CERTIFIED` | 10 | 14 | 76 | 0.184 | 14 | 76 |
| `weak_strong_nph1` | `TARGET_WINDOW_CERTIFIED` | 16 | 24 | 76 | 0.316 | 24 | 76 |
| `intermediate_strong_nph1` | `TARGET_WINDOW_CERTIFIED` | 64 | 26 | 56 | 0.464 | 28 | 56 |
| `strong_strong_u8_nph1` | `TARGET_WINDOW_CERTIFIED` | 18 | 12 | 76 | 0.158 | 20 | 76 |
| `weak_weak_nph3` | `TARGET_WINDOW_CERTIFIED` | 226 | 242 | 172 | 1.407 | 382 | 512 |
| `intermediate_weak_nph3` | `TARGET_WINDOW_CERTIFIED` | 478 | 242 | 240 | 1.008 | 260 | 376 |
| `strong_weak_u8_nph3` | `TARGET_WINDOW_CERTIFIED` | 160 | 160 | 172 | 0.930 | 218 | 308 |
| `weak_strong_nph3` | `POOL_EXHAUSTED` | — | 428 | 852 | 0.502 | 526 | 1396 |
| `intermediate_strong_nph3` | `POOL_EXHAUSTED` | — | 428 | 1124 | 0.381 | 440 | 1192 |
| `strong_strong_u8_nph3` | `TARGET_WINDOW_CERTIFIED` | 200 | 216 | 308 | 0.701 | 286 | 308 |

Every completion-pool cell has online status `TARGET_WINDOW_CERTIFIED`.
Direct Davidson has online status `DAVIDSON_RESIDUAL_CERTIFIED` and reaches
both reported exact targets in all 12 cells. At $10^{-6}$, completion is no
more expensive than direct in 12/12 cells; its largest ratio is 0.929.

The two completion losses at $10^{-4}$ are localized:

- `weak_weak_nph3`: 242 versus Davidson's 172, ratio 1.407.
- `intermediate_weak_nph3`: 242 versus Davidson's 240, ratio 1.008.

The completion pool repairs both genuine base-pool failures without invoking
Davidson:

- `weak_strong_nph3`: base `POOL_EXHAUSTED`; completion reaches $10^{-4}$
  at 428 rather than Davidson's 852.
- `intermediate_strong_nph3`: base `POOL_EXHAUSTED`; completion reaches
  $10^{-4}$ at 428 rather than Davidson's 1124.

## Artifact and resume lock

The full artifacts remain in the canonical repository's ignored `output/`
tree; this tracked evidence file is their promotion anchor.

| Artifact | SHA-256 |
|---|---|
| `output/diagnostics/paper_iii_hh_pool_profile_ablation_20260828_v2/pool_profile_ablation.json` | `7202a8a51ebce5abfb2835072292dd7965868245e66db95a2bcf770b36a04945` |
| `output/diagnostics/paper_iii_hh_pool_profile_ablation_20260828_v2/pool_profile_ablation.md` | `88eea3f966edecc955db5397a169689c5ca9555aeafd99ae4fc72a5bae365425` |

Both profile receipts name source revision `4542a895b7670cfd4233f67b1ed4447f05eec510`
and the same source-lock digest listed above. Re-running the campaign CLI
against the completed output root returned in 1.5 seconds and preserved both
artifact hashes byte-for-byte, demonstrating deterministic resume without
route recomputation.

## Safeguard regression discovered by the automation

The first campaign pass exposed a false terminal-certificate failure: the
certifier tested unused candidates with `norm_sq > 0`, whereas selection used
the declared supported-image absolute cutoff. Floating-point residue from the
projected identity was therefore resurrected as the only threatening
candidate. A real `intermediate_weak_nph1` regression test was made red before
the certifier was aligned with the selector cutoff. The final artifacts above
were regenerated from the committed fix.

Verification on the source revision:

- Focused pool/campaign/completion suite: 78 passed.
- Wider Paper-III/QSE suite: 217 passed.
- Default nph1 growth behavior lock: all 384 support-bearing rows are
  bit-identical to the pre-increment baseline.
- Neither `exchange_maintenance.py` nor the shared problem provider was
  edited.
- No Paper-III manuscript source was edited or staged by this increment.

## Next algorithm decision

Do not insert an implicit selected-QSE-to-Davidson fallback. The enlarged
fixed pool already supplies complete 12-cell coverage and remains a pure
selected-QSE route.

The next justified ablation is a separately versioned, declared two-stage
alphabet route: begin with the base macro pool and expose the encoded-boson
completion only after the base stage returns an online pool-insufficiency
status. Both stages must be charged to its receipt. Compare that route against
always-on completion and direct Davidson. This test asks whether the easy-cell
economy of the base alphabet and the hard-cell coverage of the completion
alphabet can coexist without crowding the canonical selector with a hidden
fallback.
