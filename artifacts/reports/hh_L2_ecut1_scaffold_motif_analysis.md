# HH L=2 n_ph_max=1 Heavy Scaffold: Motif Analysis & Pruning Summary

## Source Artifact

- Log: [adapt_hh_L2_ecut1_full_meta_phase3_heavy_powell_20260321T014500Z.log](../logs/adapt_hh_L2_ecut1_full_meta_phase3_heavy_powell_20260321T014500Z.log)
- Scaffold JSON: [adapt_hh_L2_ecut1_full_meta_phase3_heavy_powell_20260321T014500Z.full_scaffold_for_pruning.json](../json/adapt_hh_L2_ecut1_full_meta_phase3_heavy_powell_20260321T014500Z.full_scaffold_for_pruning.json)

## Physics

- `L=2`, `t=1.0`, `u=4.0`, `dv=0.0`, `omega0=1.0`, `g_ep=0.5`
- `n_ph_max=1`, `boson_encoding=binary`, `ordering=blocked`, `boundary=open`
- Total qubits: 6 (fermion=4, boson=2)
- Exact GS energy: `0.15866790412572532`

## ADAPT Settings

- `adapt_pool=full_meta`, `adapt_continuation_mode=phase3_v1`
- `adapt_inner_optimizer=POWELL`, `adapt_max_depth=160`, `adapt_maxiter=12000`
- `adapt_reopt_policy=windowed`, `adapt_window_size=999999` (full refit)
- `adapt_allow_repeats=true`
- `phase3_runtime_split_mode=shortlist_pauli_children_v1`
- `adapt_drop_floor=-1.0` (disabled — run was manually interrupted)

## Outcome

- Final energy: `0.15872408303679303`
- `abs_delta_e`: `5.618e-5`
- Scaffold depth at interruption: 97 (plus 4 seed ops = 101 total entries)
- `eps_energy_low_streak`: 85 (plateaued for 85 consecutive steps)
- Only 20 unique operators selected (81 repeats out of 101 entries)

## Pool Inventory: 46 operators available

| Class | Count in pool | Used | Notes |
| --- | ---: | --- | --- |
| `hh_termwise_unit` | 16 | **0/16** | Never selected |
| `hh_termwise_quadrature` | 4 | **4/4** | All used (seed block) |
| `uccsd_sing` | 2 | **2/2** | All used |
| `uccsd_dbl` | 1 | **0/1** | Never selected |
| HVA layers | 4 | **0/4** | Never selected |
| `paop_cloud_p` | 2 | **2/2** | All used — **dominant class** (87 of 101 entries) |
| `paop_cloud_x` | 2 | **0/2** | Never selected |
| `paop_disp` | 2 | **0/2** | Never selected |
| `paop_dbl` | 2 | **0/2** | Never selected |
| `paop_hopdrag` | 1 | **1/1** | Used (4 entries via child sets) |
| `paop_dbl_p` | 4 | **2/4** | 2 used (site=0->ph=1, site=1->ph=0); 2 unused (site=0->ph=0, site=1->ph=1) |
| `paop_dbl_x` | 4 | **0/4** | Never selected |
| `paop_curdrag` | 1 | **0/1** | Never selected |
| `paop_hop2` | 1 | **0/1** | Never selected |

**Summary: 11 operators used out of 46 available (24%). 35 operators never selected.**

## Five Motifs

### 1. `paop_cloud_p` completely dominates the scaffold

`paop_cloud_p` accounts for **87 of 101** scaffold entries (86%). Only 2 parents exist (`site=0->phonon=1` and `site=1->phonon=0`), selected a combined 52+35=87 times including child sets. The ADAPT loop is hammering these two operators and their child atoms over and over, squeezing out tiny energy improvements each time. This is the electron-phonon cloud coupling — the single most important physics at this coupling strength.

### 2. HH quadrature is the backbone (all 4 used as seed)

All 4 `hh_termwise_quadrature` terms enter as seed operators (positions 1–4). They are not repeated later — unlike L=3 where they appeared twice. At L=2 the quadrature seed fully covers the available boson-fermion coupling directions.

### 3. UCCSD singles essential, doubles unused

Both `uccsd_sing` operators are selected (positions 5–6), immediately after the seed block. The single `uccsd_dbl` (`ab:0,2->1,3`) is never used. At L=2 with only 2 sites, the double excitation space is too constrained — singles suffice.

### 4. `paop_hopdrag` and `paop_dbl_p` provide supporting structure

- `paop_hopdrag(0,1)`: 4 entries (positions 9–12) via child sets — all 4 possible child set combinations used
- `paop_dbl_p`: 2 parents used (4 entries including child sets), 2 parents unused (the `phonon=0` and `phonon=1` variants for the "wrong" site-phonon pairing)

These provide the correlated hopping and long-range e-ph coupling that `cloud_p` alone can't capture.

### 5. Everything else is dead weight

**Zero usage** of:
- All 16 `hh_termwise_unit` terms (diagonal Hamiltonian terms)
- All 4 HVA layerwise macros
- All `x`-type PAOPs (`cloud_x`, `dbl_x`)
- `paop_disp` (both sites)
- `paop_dbl` (both sites)
- `paop_curdrag`, `paop_hop2`
- The one `uccsd_dbl`

This matches the L=3 pattern exactly: `p`-type operators dominate, `x`-type operators are inert, unit terms and HVA layers contribute nothing.

## Comparison with L=3 Motifs

| Motif | L=2 (n_ph_max=1) | L=3 (n_ph_max=1) |
| --- | --- | --- |
| Quadrature | 4/4 used (seed only) | 6/6 used (seed + repeated) |
| UCCSD singles | 2/2 used | 4/4 used |
| UCCSD doubles | 0/1 used | 3/4 used |
| `paop_cloud_p` | **2/2 — dominant (86%)** | 4/4 — strong but not dominant |
| `paop_dbl_p` | 2/4 used | 7/7 used |
| `paop_disp` | 0/2 unused | 2/3 used |
| `paop_hopdrag` | 1/1 used (child sets) | 2/2 used (child sets) |
| `x`-type PAOPs | 0 used | 0 used |
| HVA layers | 0 used | 0 used |
| Unit terms | 0 used | 0 used |

Key differences:
- **`paop_cloud_p` is far more dominant at L=2** — it's basically the only operator the ADAPT loop wants after the initial seed+UCCSD+hopdrag block. At L=3 the scaffold is more balanced across classes.
- **`paop_disp` is unused at L=2** but used at L=3. The dispersive coupling may only become relevant with more sites.
- **`uccsd_dbl` is unused at L=2** but heavily used at L=3. Doubles need the larger orbital space to contribute.
- **`paop_dbl_p` is partially used at L=2** (2/4) vs fully used at L=3 (7/7). Again, more sites = more useful long-range couplings.

## Implied Pareto Pool for L=2 (n_ph_max=1)

**Keep (core) — 11 generators:**

| Class | Count | Role |
| --- | ---: | --- |
| `hh_termwise_quadrature` | 4 | Energy backbone (seed) |
| `uccsd_sing` | 2 | Fermionic singles correlation |
| `paop_cloud_p` | 2 | e-ph cloud coupling — dominant class |
| `paop_hopdrag` | 1 | Correlated hopping (via child sets) |
| `paop_dbl_p` | 2 | Long-range e-ph doubles (site=0->ph=1, site=1->ph=0 only) |

**Drop entirely — 35 generators:**

| Class | Count | Reason |
| --- | ---: | --- |
| `hh_termwise_unit` | 16 | Never selected |
| HVA layers | 4 | Never selected |
| `paop_cloud_x` | 2 | x-type never selected |
| `paop_dbl_x` | 4 | x-type never selected |
| `paop_disp` | 2 | Never selected at L=2 |
| `paop_dbl` | 2 | Never selected |
| `paop_curdrag` | 1 | Never selected |
| `paop_hop2` | 1 | Never selected |
| `uccsd_dbl` | 1 | Never selected at L=2 |
| `paop_dbl_p` (unused variants) | 2 | site=0->ph=0 and site=1->ph=1 never selected |

This takes the pool from **46 → 11**, a 76% reduction. Even more aggressive than L=3's 60% cut.

## Observation: Scaffold Depth vs Useful Information

The 101-entry scaffold has only 20 unique operators. After position ~14, the ADAPT loop is exclusively recycling `paop_cloud_p` parents and child sets with diminishing returns (~1e-12 per step by the end). The effective "information content" of this scaffold is concentrated in the first ~14 positions.

A truncated scaffold at depth 14 would use only the 11 unique operators identified above and likely capture the vast majority of the energy improvement, at a fraction of the circuit cost.
