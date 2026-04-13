# HH Phase-3 Selector Geometry Matrix

- Date: 2026-04-12
- Objective: determine whether `--phase3-selector-geometry-mode proxy_reduced` is a narrowly useful HH recovery lever or a generally better current-route selector.
- Current recommendation: keep `proxy_reduced` as an explicit HH recovery/debug knob only. Do not broaden it toward the default route based on current evidence.

## Executive summary

The matrix result is mixed in the exact way that matters for repo policy:

1. **Positive on the regression/fullhorse anchor**
   - `proxy_reduced` recovers the good basin and is cheaper than current `reduced`.
2. **Negative on the public/deployment anchor**
   - `proxy_reduced` is slightly worse in energy gap and uses a larger scaffold.
3. **Negative on a solved off-anchor static HH point**
   - `proxy_reduced` matches energy but becomes materially heavier.

So the present best statement is:

> `proxy_reduced` is a useful **targeted recovery lever** for the legacy/fullhorse regression lane, but it does **not** generalize cleanly to the public anchor or to an off-anchor solved HH point.

## Matrix

| Case | Physics / route | Reduced lane | Proxy-reduced lane | Verdict |
| --- | --- | --- | --- | --- |
| Regression/fullhorse anchor | `L=2`, HH, `t=1 U=4 g=0.5 omega0=1 n_ph_max=1`, `full_meta + phase3_v1 + POWELL`, current route | `artifacts/agent_runs/20260412_hh_l2_builtin_reduced_control_v1/json/result.json` → `|ΔE|=0.32829231021489486`, logical ops `18`, transpiled `122` 2Q, depth `363` | `artifacts/agent_runs/20260412_hh_l2_builtin_proxy_reduced_validate_v1/json/result.json` → `|ΔE|=5.617823464409977e-05`, logical ops `18`, transpiled `110` 2Q, depth `317` | **Strong positive**. This is the lane where `proxy_reduced` is clearly useful. |
| Public / deployment anchor | `L=2`, HH, `t=1 U=4 g=0.5 omega0=1 n_ph_max=1`, public SPSA route, proxy backend-cost mode | `artifacts/agent_runs/20260412_hh_l2_public_anchor_reduced_matrix_v1/json/run.json` → `|ΔE|=1.4383993769573333e-04`, logical ops `21` | `artifacts/agent_runs/20260412_hh_l2_public_anchor_proxy_reduced_matrix_v1/json/run.json` → `|ΔE|=1.4665747987888111e-04`, logical ops `23` | **Negative**. Public anchor gets slightly worse and larger. |
| Off-anchor solved static HH point (`lam0p2`) | `L=2`, HH, `t=1 U=0 dv=1 omega0=0.5 g=0.2236068 n_ph_max=1`, `full_meta + phase3_v1 + POWELL`, fixed-`dv` sweep case | Baseline reuse: `artifacts/agent_runs/20260411T203500Z_hh_l2_u0_w0p5_lambda_sweep_legacy_vs_current_fixed_dv_v1/cases/lam0p2/current_head/json/result.json` → `|ΔE|=5.738127468916665e-04`, logical ops `13`, transpiled `91` 2Q, depth `302` | `artifacts/agent_runs/20260412_hh_l2_lam0p2_proxy_reduced_matrix_v1/json/result.json` → `|ΔE|=5.738127468918885e-04`, logical ops `13`, transpiled `130` 2Q, depth `466` | **Negative**. Energy is effectively unchanged, but scaffold cost is materially worse. |

## What changed structurally

### 1. Regression/fullhorse anchor

This is the only row where `proxy_reduced` clearly improves the current route.

- Reduced lane falls into the bad basin.
- Proxy-reduced lane recovers the good basin.
- The earlier shadow-causal audit showed the decisive seam is the Phase-3 selector geometry, not a blanket legacy-score revert.

### 2. Public anchor

The public anchor does **not** benefit from the same switch.

- Reduced lane: `|ΔE|=1.4383993769573333e-04`, `21` logical ops.
- Proxy-reduced lane: `|ΔE|=1.4665747987888111e-04`, `23` logical ops.

Important nuance:

- This public route uses `phase3_backend_cost_mode=proxy`, so the most trustworthy comparison fields here are the energy gap and scaffold size, not routed `2Q` / depth.
- On those public-anchor fields, `proxy_reduced` loses.

The operator lists agree through the early shared prefix and diverge later; the first visible scaffold divergence in the saved operator lists occurs after the first seven operators.

### 3. Off-anchor solved point (`lam0p2`)

This is the cleanest generalization check because the point is already solved on the current route.

- Reduced baseline: `|ΔE|=5.738127468916665e-04`, `91` 2Q, depth `302`.
- Proxy-reduced replay: `|ΔE|=5.738127468918885e-04`, `130` 2Q, depth `466`.

So here `proxy_reduced` is not rescuing anything; it is simply making the scaffold heavier for no meaningful energy gain.

The saved operator lists remain identical through the first eight operators and first diverge afterward, which is consistent with a later structural drift rather than an immediate basin flip.

## Interpretation

The matrix now gives a boundary statement strong enough to use operationally:

- **Yes**: `proxy_reduced` is valuable as a recovery knob on the specific current-route fullhorse regression lane.
- **No**: current evidence does **not** support treating it as a generally better selector for the repo's public/direct HH route.
- **No**: current evidence does **not** support broadening it based on off-anchor static HH behavior.

That means the safest repo policy is:

1. keep `--phase3-selector-geometry-mode proxy_reduced` explicit and opt-in,
2. use it when the objective is **legacy/fullhorse regression recovery**,
3. do not move it toward the default public route without new positive evidence on public/off-anchor surfaces.

## Artifact list

### Positive recovery row
- `artifacts/agent_runs/20260412_hh_l2_builtin_reduced_control_v1/json/result.json`
- `artifacts/agent_runs/20260412_hh_l2_builtin_proxy_reduced_validate_v1/json/result.json`

### Public anchor row
- `artifacts/agent_runs/20260412_hh_l2_public_anchor_reduced_matrix_v1/json/run.json`
- `artifacts/agent_runs/20260412_hh_l2_public_anchor_proxy_reduced_matrix_v1/json/run.json`

### Off-anchor row
- `artifacts/agent_runs/20260411T203500Z_hh_l2_u0_w0p5_lambda_sweep_legacy_vs_current_fixed_dv_v1/cases/lam0p2/current_head/json/result.json`
- `artifacts/agent_runs/20260412_hh_l2_lam0p2_proxy_reduced_matrix_v1/json/result.json`
