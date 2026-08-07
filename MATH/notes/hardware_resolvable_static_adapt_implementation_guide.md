# Hardware-Resolvable Static ADAPT Implementation Guide

Created: 2026-05-16  
Source of truth: `MATH/paper_details/adaptive_selection_staged_continuation.tex`  
Target implementation area: static ADAPT/SNAKE selector code and benchmark telemetry

## Purpose

The math manuscript was updated so the SNAKE/static-ADAPT selector is hardware-resolvable rather than only shot-resolvable. The implementation must now distinguish three objects:

1. **Resolution-adjusted gain**: the predicted energy gain after subtracting shot/statistical and hardware-resolution floors.
2. **Selector score**: the dimensionless acquisition ranking score using geometry, novelty, and `K_k`.
3. **Net hardware-resolvable utility**: the energy-scale value used for admission, stopping, and runway forecasting.

Do not collapse these into one score.

## Notation Contract

- Use `k` for phase index: `k in {0,1,2,3}`.
- Use `t` for selector/ADAPT iteration.
- Do not use `Omega` for hardware opacity. The manuscript reserves `Omega_{mn}` for algebraic support overlap.
- Use `Lambda_hw(c)` for compiled-circuit hardware opacity.
- Use `pi_hw(c) = 1 - exp(-Lambda_hw(c))` for the compact circuit-corruption proxy.
- Treat `K_k(r;t)` as a **dimensionless acquisition-burden penalty**, not an energy-error bound.

## New Hardware-Resolution Objects

For a compiled measurement circuit `c`,

```text
Lambda_hw(c)
  = sum_eta n_eta(c) [-log(1 - epsilon_eta)]
  + sum_q T_q(c) / T_q^coh
  + sum_q [-log(1 - r_q)]
  + Lambda_xtalk(c)
  + Lambda_drift(c)

pi_hw(c) = 1 - exp(-Lambda_hw(c))
```

For an observable measured in commuting groups,

```text
b_O_hw(t) = sum_b sum_{ell in B_b} |c_{b ell}| b_ell_hw(c_b)
b_ell_hw(c_b) <= beta_O pi_hw(c_b)
```

For a candidate-gradient observable,

```text
epsilon_g_res(r;t)
  = z_alpha(t,r) sigma_g_shot(r;t)
  + b_g_hw(r;t)
  + b_g_drift(r;t)

g_hw_lcb(r;t) = max(|g_hat_r(t)| - epsilon_g_res(r;t), 0)
```

Implementation note: `epsilon_g_res` is an aggregate resolution floor. Keep the decomposition internally so shot uncertainty, hardware bias, drift, and future mitigation terms can be logged separately.

## Phase 0 Pilot Screen

Phase 0 is a cheap raw-gradient pilot screen before Fubini--Study geometry or curvature.

For each broad candidate-position record `r = (m,p)`:

```text
g0(r) = d/dvartheta E(O plus_p m, theta plus_p vartheta)|_{vartheta=0}

epsilon_g0_res(r;t)
  = z_alpha(t,r) sigma_0(r;t)
  + b_g0_hw(r;t)
  + b_g0_drift(r;t)

g0_upper_hw(r;t) = |g0_hat(r;t)| + epsilon_g0_res(r;t)
DeltaE0_upper_hw(r;t) = alpha_0 g0_upper_hw(r;t)
```

Phase 0 shortlists by threshold or top `N_0`:

```text
S0 = { r in R0 : DeltaE0_upper_hw(r;t) >= tau_0_lane(r) }
R1 = S0
```

Important: Phase 0 is a one-sided rejection screen. It does not test Fubini--Study displacement, tangent novelty, curvature, or recoverability.

## Phase I/II/III Gain Updates

Every place that previously used:

```text
g_lcb(r)
```

must use:

```text
g_hw_lcb(r;t)
```

Keep the existing geometry and trust-region definitions. Do not rewrite the Fubini--Study or Schur geometry merely to add hardware resolution.

Examples:

```text
DeltaE1_TR_hw(r;t)
  = max_{|alpha| <= rho / sqrt(F_raw(r))}
      [ g_hw_lcb(r;t)|alpha| - 0.5 lambda_F F_raw(r) alpha^2 ]

DeltaE2_TR_hw(r;t)
  = max_{|alpha| <= rho / sqrt(F_raw(r))}
      [ g_hw_lcb(r;t)|alpha| - 0.5 max(h_r,0) alpha^2 ]

DeltaE3_TR_hw(r;t)
  = max_{|alpha| <= rho / sqrt(F_red_safe(r))}
      [ g_hw_lcb(r;t)|alpha| - 0.5 max(h_tilde_r,0) alpha^2 ]
```

Selector score remains:

```text
S_k(r;t) = DeltaE_k_hw(r;t) N_k(r;t) / (1 + K_k(r;t))
```

`K_k` is a relative acquisition burden. Do not interpret it as an energy error and do not subtract it from energy gains.

## Admission and Stopping Utility

Define a separate energy-scale burden for future energy-resolution degradation:

```text
Delta b_E_inc(C;t)
  = max(b_E_hw(U_t plus C) - b_E_hw(U_t), 0)
```

Here `C` is a payload: either a singleton `{r}` or a batch `B`.

Define hardware-resolvable value:

```text
V_hw(C;t)
  = DeltaE_hw(C;t) N(C;t)
  - gamma_hw Delta b_E_inc(C;t)
```

Stopping residual:

```text
Xi_hw(t) = max_{C in C_t} max(V_hw(C;t), 0)
```

Stop when:

```text
Xi_hw(t) <= tau_stop(t)
```

Candidate family `C_t` must include:

- live singleton candidate surface;
- eligible geometric batch family when batching is enabled.

Do not claim this is exact ADAPT stationarity or global convergence. It is hardware-resolvable stationarity over the searched/live candidate family.

## Runway Forecast Replacement

The useful-runway clock must be driven by historical **net hardware-resolvable gains**, not raw energy drops.

For accepted payload `C_s`,

```text
Delta_adm_lcb(s)
  = Delta_adm_hat(s)
  - z_alpha_s sigma_DeltaE_shot(s)
  - b_E_Delta_cmp(s)
```

In noiseless ideal simulation, unavailable terms may be dropped.

Net admitted gain:

```text
Y_hw(s)
  = Delta_adm_lcb(s)
  - gamma_hw Delta b_E_inc(C_s;s)

Z_hw(s) = asinh(Y_hw(s) / tau_Delta)
```

Use valid historical admissions:

```text
I_adm(t)
```

not `I_t^+`, because `Y_hw` may be signed and negative net admissions should not be hidden.

Runway summaries:

```text
m_hw(t) = EWMA_{s in I_adm(t)} Z_hw(s)
s_hw(t) = max(s_min, MAD_{s in I_adm(t)} Z_hw(s))

rho_hw(t)
  = clip(
      EWMA_recent([Y_hw]_+) /
      (EWMA_older([Y_hw]_+) + eps),
      rho_min,
      1
    )

gamma_hw_decay(t) = max(-log(rho_hw(t)), 0)
u_hw(t) = asinh(tau_stop(t) / tau_Delta)
```

Future usefulness probability:

```text
Q_hw(q;t)
  = Phi((m_hw(t) - gamma_hw_decay(t)(q-1) - u_hw(t)) / s_hw(t))
```

Survival:

```text
P_surv_hw(q;t) = product_{j=1}^{q-1} (1 - h_hw(j;t))
```

Remaining useful admissions:

```text
N_rem_hw(t) = sum_{q=1}^{D_left(t)} P_surv_hw(q;t) Q_hw(q;t)
R_hw(t) = N_rem_hw(t) / (D_left(t) + eps)
e_hw(t) = R_hw(t)^a
g_hw(t) = (1 - R_hw(t))^b
```

All maturity schedules should use `R_hw`, `e_hw`, and `g_hw`.

## Schedules Driven by Hardware-Resolvable Runway

Replace raw `R_t`, `e_t`, `g_t` in schedule logic with:

```text
R_hw(t), e_hw(t), g_hw(t)
```

Affected schedules:

- shortlist pressure;
- shortlist breadth;
- refit-window contraction;
- phase retirement;
- shot-count schedule;
- pruning/ablation pressure;
- expensive phase retirement order.

## Hardware-Aware Shot Schedule

Replace the old SNR floor denominator with a hardware-aware denominator:

```text
N_shot_snr_hw(k;t)
  = ceil(
      kappa_k^2 sigma_hat_{t,k}^2 /
      max(delta_hat_{t,k}^2,
          tau_delta_hw(k;t)^2,
          delta_floor_k^2)
    )
```

Use a pilot resolvability gate:

```text
Gamma_res(t,k)
  = 1[
      delta_hat_pilot(t,k)
      > tau_delta_hw(k;t) + z_alpha sigma_delta_pilot(t,k)
    ]
```

Effective shots:

```text
if Gamma_live(k;t) == 0:
    N_shot_eff(k;t) = 0
elif Gamma_res(t,k) == 0:
    N_shot_eff(k;t) = N_diag_k
else:
    N_shot_eff(k;t)
      = min(N_shot_cap_k,
            max(N_shot_sched(k;t), N_shot_snr_hw(k;t)))
```

Do not use zero shots merely because the pilot is unresolved. Use the diagnostic floor `N_diag_k` to preserve telemetry.

## Batch Update

Batch gradients must become hardware-aware.

For batch `B`,

```text
g_hat_B(t) = (g_hat_r(t))_{r in B}
Sigma_B_shot(t) = covariance of jointly estimated batch-gradient vector
b_B_hw(a;t) = sum_{r in B} a_r b_g_hw(r;t)
```

Robust batch trust-region gain:

```text
DeltaE_B_TR(t)
  = max_{a >= 0, a^T G_B(t) a <= rho_B(t)^2}
      [
        |g_hat_B(t)|^T a
        - z_alpha sqrt(a^T Sigma_B_shot(t) a)
        - b_B_hw(a;t)
        - 0.5 a^T H_tilde_B_plus(t) a
      ]
```

Contextual singleton gain inside the same batch reduction:

```text
DeltaE_{r|B}_TR(t)
  = max_{beta >= 0, beta^2 G_B_rr(t) <= rho_B(t)^2}
      [
        g_hw_lcb(r;t) beta
        - 0.5 H_tilde_B_rr_plus(t) beta^2
      ]
```

Batch hardware value:

```text
V_hw(B;t)
  = DeltaE_B_TR(t) N_3(B;t)
  - gamma_hw Delta b_E_inc(B;t)
```

Admit a batch only if:

```text
V_hw(B;t) > tau_stop(t)
```

Batch stopping must consider eligible batches, not only singletons. A batch of individually subthreshold directions can still be jointly useful unless a subadditivity theorem is imposed.

## Modes

### Ideal/noiseless mode

Use:

```text
b_g_hw = 0
b_g_drift = 0
b_E_hw = 0
Delta b_E_inc = 0
```

The selector then reduces to shot/noiseless behavior, with the same structural equations.

### Compiled-resource proxy mode

If no calibrated noisy-versus-ideal backend data exist, hardware floors may be proxied from compiled-resource features. These must be logged as proxy controller floors, not calibrated hardware error bars.

### Calibrated hardware mode

Use backend calibration, compiled measurement circuits, layout, readout, drift, and mitigation metadata to instantiate `b_O_hw`, `b_g_hw`, and `b_E_hw`.

## Confidence and Theorem Caution

Pointwise `z_alpha` confidence is enough for heuristic ranking. For theorem-level stopping claims, confidence must be simultaneous or sequentially budgeted over the searched candidate family:

```text
sum_{t,C} alpha_{t,C} <= alpha_total
```

Hardware-resolvable stopping means:

```text
no searched payload has lower-confidence net value above tau_stop
```

It does not mean:

- exact ADAPT pool exhaustion;
- exact gradient stationarity;
- ground-state convergence;
- calibrated hardware validation without calibrated hardware floors.

## Implementation Checklist

- Add hardware opacity/resolution data structures without reusing `Omega`.
- Add aggregate `epsilon_g_res` fields while preserving decomposed telemetry.
- Replace all `g_lcb` calls with `g_hw_lcb`.
- Add Phase 0 upper-confidence raw-gradient screen.
- Keep `K_k` in the score denominator only.
- Add `Delta b_E_inc` and `V_hw` for admission/stopping.
- Replace raw runway telemetry with `Y_hw`, `Z_hw`, `N_rem_hw`, `R_hw`, `e_hw`, `g_hw`.
- Drive schedule pressure/retirement/window/shot/prune logic from hardware-aware runway.
- Add hardware-aware SNR shot schedule plus diagnostic floor.
- Update batch gain and batch stopping to include hardware-aware robust objective.
- Preserve ideal/noiseless and proxy modes explicitly.

