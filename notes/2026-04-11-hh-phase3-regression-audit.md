# HH L=2 Phase-3 Scaffold Regression Audit

- Date: 2026-04-11
- Scope: current checkout, `L=2`, HH, `t=1`, `U=4`, `g=0.5`, `omega0=1`, `n_ph_max=1`
- Objective: explain why the current direct HH ADAPT path still fails to recover the historical `81`-2Q fullhorse oracle at the same `|\Delta E| \sim 10^{-4}` target

## Issue statement

The frozen March 22 legacy route still reproduces the best known raw-HF-start cost-vs-energy point in this checkout:

- legacy oracle: `artifacts/agent_runs/20260409_hh_l2_hist81_legacy_current_compare_d16_v3/legacy_20260322/`
- result: `|\Delta E| = 5.617823464482141e-05`
- compile scout: `81` two-qubit gates, depth `151`

The current static route can still reach essentially the same energy band, but only with materially heavier scaffolds.

## Executive update (must-read)

The most important late result from this audit is the legacy-shell geometry isolation bundle:

- `artifacts/agent_runs/20260411_hh_l2_legacy_proxy_vs_exact_reduced_v1/json/summary.json`

Under the frozen legacy shell, `proxy_reduced` and `exact_reduced` are identical:

- both reach `|\Delta E| = 5.617823464482141e-05`
- both first enter the good band at depth `6`
- both compile to `81` 2Q, depth `151`, size `348` on `FakeMarrakesh`

So the strongest current statement is:

> Exact-reduced geometry alone does **not** explain the current-route regression. When transplanted into the frozen legacy shell, it still recovers the historical `81`-2Q oracle scaffold.

The other important repo-state update is the `L=3` routed-cost bundle:

- `artifacts/agent_runs/20260411_hh_l3_routed_cost_legacy_vs_current_v1/json/summary.json`

That comparison shows a regime split rather than a single cost ordering:

- historical strong-coupling success: `618` 2Q, depth `1462`, `|\Delta E| = 8.4942074026e-05`
- current weak-coupling success: `342` 2Q, depth `905`, `|\Delta E| = 4.1313925567e-04`
- current strong-coupling direct-surface failure: `159` 2Q, depth `400`, but `|\Delta E| = 5.0332982161e-01`

So `L=3` is not “legacy always cheaper” or “current always heavier”; the present state is that the cheap current strong-coupling line is cheap because it is wrong.

## Locked reference runs

| Role | Artifact | Result |
| --- | --- | --- |
| Legacy oracle | `artifacts/agent_runs/20260409_hh_l2_hist81_legacy_current_compare_d16_v3/legacy_20260322/` | `|\Delta E|=5.617823464482141e-05`, `81` 2Q, depth `151`, runtime params `27` |
| Legacy shell with exact-reduced Phase-3 geometry | `artifacts/agent_runs/20260411_hh_l2_legacy_proxy_vs_exact_reduced_v1/cases/hybrid_exact_reduced/` | `|\Delta E|=5.617823464482141e-05`, `81` 2Q, depth `151`, runtime params `27` |
| Zero-burden reduced selector | `artifacts/agent_runs/20260411_hh_l2_phase3_raw_vs_reduced_geometry_v1/cases/control_reduced/` | `|\Delta E|=5.7163718280933695e-05`, `179` 2Q, depth `551`, runtime params `40` |
| Zero-burden raw-exact selector | `artifacts/agent_runs/20260411_hh_l2_phase3_raw_vs_reduced_geometry_v1/cases/raw_exact/` | `|\Delta E|=5.720162546546392e-05`, `169` 2Q, depth `498`, runtime params `38` |
| Best current exact-path line so far | `artifacts/agent_runs/20260411_hh_l2_phase3_burden_sweep_v1/cases/raw_exact_compile_only/` | `|\Delta E|=5.717731863025266e-05`, `163` 2Q, depth `450`, runtime params `36` |

## Established findings

### 1. The regression is real

The legacy route is not a phantom artifact. In the current checkout, the frozen legacy entrypoint still reproduces the `81`-2Q scaffold at the expected energy band.

So the current repo state is:

- low-energy convergence is still reachable on the current path
- low-cost scaffold recovery is what regressed

### 2. Shortlists and batching are not the primary cause

The score-3-only compare removed shortlist pressure and disabled batching, but the first divergence survived.

At depth `3`:

- legacy admits `paop_lf_full:paop_dbl_p(site=0->phonon=0)`
- current still admits `paop_full:paop_disp(site=0)`

So the root issue is not simply shortlist truncation or batch shell behavior.

### 3. Phase-3 geometry changed in the current route

Legacy phase 3 uses the proxy reduced geometry

- `\widetilde h_r^{legacy} = F_r^{metric} - o_r^T (G_r + \lambda_H I)^{-1} o_r`
- `\mathcal N_3^{legacy}(r) = 1 - o_r^T (G_r + \varepsilon I)^{-1} o_r / F_r^{metric}`
- trust-region radius also uses `F_r^{metric}`

Current phase 3 uses exact derivative-based reduced geometry

- `\widetilde h_r = h_r - b_r^T R_r^{-1} b_r`
- `\mathcal N_3^{now}(r) = 1 - (q_r^{red})^T (Q_r + \varepsilon I)^{-1} q_r^{red} / F_r^{red}`
- trust-region radius uses `F_r^{red}`

The important audit point is that the current route changed the actual geometric objects fed into phase 3, not just weights around the same proxy.

### 4. Raw-vs-reduced ablation: post-window reduction is only part of the issue

We added a diagnostic-only hidden selector flag:

- `--phase3-selector-geometry-mode reduced`
- `--phase3-selector-geometry-mode raw_exact`

Result:

- `reduced`: `179` 2Q, first band depth `7`
- `raw_exact`: `169` 2Q, first band depth `6`

So scoring before inherited-window reduction helps modestly.

But it does **not** repair the early family choice:

- both lanes still admit displacement at depth `3`
- neither lane restores the legacy LF-double start

Therefore:

- post-window reduction contributes to the regression
- exact raw geometry is still not enough to recover the proxy-optimal scaffold family

### 5. Burden sweep: cost can be trimmed, but not fixed

We then swept burden on top of `raw_exact`.

Result:

- full burden ladders (`light`, `default`, `strong`, `default+lifetime`) all converged to the same `171`-2Q family
- `raw_exact_compile_only` improved further to `163` 2Q
- `reduced_burden_default` became worse than the zero-burden reduced control and entered the good band later

So burden shaping can compress the current exact path somewhat, but it does not restore the legacy low-cost family.

### 6. Current best statement

Current evidence supports the following statement:

> For the cost-vs-energy objective of this repo, the current exact phase-3 geometry path remains worse than the legacy proxy path for recovering the low-cost scaffold family, even when we remove post-window reduction from selector ranking.

This is stronger than saying only that reduced geometry is bad. The exact raw path is also still wrong on the decisive early family ranking.

### 7. Legacy-shell exact-reduced isolation: exact geometry alone does not break the oracle

We then ran a stricter isolation test on the frozen March 22 shell:

- control bundle: `artifacts/agent_runs/20260411_hh_l2_legacy_proxy_vs_exact_reduced_v1/cases/control_proxy_reduced/`
- hybrid bundle: `artifacts/agent_runs/20260411_hh_l2_legacy_proxy_vs_exact_reduced_v1/cases/hybrid_exact_reduced/`
- summary: `artifacts/agent_runs/20260411_hh_l2_legacy_proxy_vs_exact_reduced_v1/json/summary.json`

The hybrid lane kept the frozen legacy Phase-1 / Phase-2 / selection shell, but replaced the legacy proxy-reduced Phase-3 geometry with the current exact-reduced geometry. That swap **did not** make the legacy route heavier.

Both lanes produced the same result:

- `|\Delta E| = 5.617823464482141e-05`
- first good-band depth `6`
- `16` logical ops
- `27` runtime params
- `81` 2Q, depth `151`, size `348` on `FakeMarrakesh`

They also kept the same decisive early admission order:

- depth `3`: `paop_lf_full:paop_dbl_p(site=0->phonon=0)`
- depth `4`: `paop_lf_full:paop_dbl_p(site=1->phonon=1)`
- depth `5`: `paop_full:paop_hopdrag(0,1)::child_set[0,2]`
- depth `6`: `paop_full:paop_disp(site=0)`

So the updated statement is sharper:

- the current-route regression is **not** explained by exact-reduced geometry alone
- the failure requires the broader current route, not just substituting exact-reduced geometry into the legacy shell

### 8. Related L=3 routed-cost state

We also created a routed-cost comparison for `L=3`:

- bundle: `artifacts/agent_runs/20260411_hh_l3_routed_cost_legacy_vs_current_v1/`
- summary: `artifacts/agent_runs/20260411_hh_l3_routed_cost_legacy_vs_current_v1/json/summary.json`

Important result:

- historical strong broad-pool scaffold (reconstructed for compile scout): `618` 2Q, depth `1462`, `|\Delta E| = 8.4942074026e-05`
- current weak successful scaffold: `342` 2Q, depth `905`, `|\Delta E| = 4.1313925567e-04`
- current strong direct-surface failed scaffold: `159` 2Q, depth `400`, but `|\Delta E| = 5.0332982161e-01`

So `L=3` is not a simple “legacy cheaper vs current heavier” story. The strong historical success is much heavier than the current weak success, while the cheap present strong-coupling direct-surface line is scientifically invalid because it sits in the wrong basin.

## What is ruled out

- not just a stopping-depth issue
- not just shortlist truncation
- not just phase-2 batching
- not just post-window reduction
- not exact-reduced geometry alone, when transplanted into the frozen legacy shell

## What is still plausible

- child-set construction / child scoring drift still matters later in the trajectory
- exact derivative geometry may still be misaligned with the scaffold-cost objective on the full current route, even though it does not break the frozen legacy shell by itself
- the legacy proxy may be acting as a better structural heuristic for low-cost scaffold discovery than the exact local geometry

## Current repo-state conclusion

For `L=2` HH, the main scientific objective remains:

- achieve the historical legacy energy band
- while matching or beating the historical legacy scaffold cost

As of this audit:

- legacy oracle still wins that objective at `81` 2Q
- best current exact-path route is `raw_exact_compile_only` at `163` 2Q
- the current route remains in a displacement-first family that the legacy route avoids

## Primary artifact paths

- legacy oracle result: `artifacts/agent_runs/20260409_hh_l2_hist81_legacy_current_compare_d16_v3/legacy_20260322/json/result.json`
- legacy oracle compile scout: `artifacts/agent_runs/20260409_hh_l2_hist81_legacy_current_compare_d16_v3/legacy_20260322/json/compile_scout_fake_marrakesh.json`
- raw-vs-reduced summary: `artifacts/agent_runs/20260411_hh_l2_phase3_raw_vs_reduced_geometry_v1/json/summary.json`
- burden sweep summary: `artifacts/agent_runs/20260411_hh_l2_phase3_burden_sweep_v1/json/summary.json`
- legacy proxy-vs-exact-reduced isolation: `artifacts/agent_runs/20260411_hh_l2_legacy_proxy_vs_exact_reduced_v1/json/summary.json`
- L=3 routed-cost bundle: `artifacts/agent_runs/20260411_hh_l3_routed_cost_legacy_vs_current_v1/json/summary.json`
