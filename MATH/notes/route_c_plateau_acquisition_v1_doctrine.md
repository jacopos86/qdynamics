---
title: "Route C Plateau Acquisition v1.2 Doctrine"
date: "2026-06-08"
status: "implementation source-of-truth draft; repo-agent contract"
companion_pdf: "MATH/notes/route_c_plateau_acquisition_v1_doctrine.pdf"
---

# Route C Plateau Acquisition v1.2 Doctrine

## Purpose

This Markdown note is the **repo-agent implementation contract** for Route C
Plateau Acquisition v1.2. It is intentionally operational: symbols, required
state, score definitions, gates, telemetry, tests, and implementation seams.

The companion PDF at
`MATH/notes/route_c_plateau_acquisition_v1_doctrine.pdf` is the pedagogical
source-of-truth derivation. The PDF explains why the log-volume score is the
right plateau acquisition statistic. This Markdown file records what a repo
agent should implement from that derivation.

Route C remains:

> Route A until the resolved Phase-III gain frontier is flat; then a plateau
> selector buys geometrically useful admitted coordinates, runs the ordinary
> enlarged refit, and commits nonzero amplitudes only after validated unlock.

Route C is not a new optimizer and not a frozen-coordinate routine.

## v1.2 change summary

The v1.1 plateau selector used a residual **fraction**
`N3_plat/(1 + K3)` as the primary plateau acquisition score. v1.2 changes the
primary acquisition statistic to **incremental resolved Fubini--Study
log-volume growth** against the admitted scaffold.

Implementation target:

```text
phase3_plateau_acquisition_score = log_volume_v1
```

Compatibility/diagnostic mode:

```text
phase3_plateau_acquisition_score = fractional_residual_v1
```

Core changes:

1. Keep the plateau context as the admitted logical scaffold:
   \(C_t^{\rm plat}=I_t^{\log}\), including dormant plateau records.
2. Use raw horizontal tangents for plateau span-growth geometry.
3. Keep Schur/Phase-III reduced objects for energy prediction, negative-curvature
   diagnostics, ordinary ranking, and refit/unlock validation.
4. Gate plateau candidates by both absolute residual strength and fractional
   residual novelty.
5. Rank plateau candidates by lower-confidence log-volume gain divided by cost.
6. Keep the existing unlock-or-zero-commit energy rule unchanged.

## Imported nouns and notation

Candidate-position records follow `MATH/Math.md`:

\[
r=(m,p),
\]

where `m` is an operator/generator candidate and `p` is an insertion position.
The staged selector chain remains

\[
\mathcal R_k(t)\to\mathcal S_k(t)\to\mathcal R_{k+1}(t):=\mathcal S_k(t).
\]

The ordinary Phase-III score remains

\[
S_3(r;t)=
\frac{\Delta E_{\rm TR}(r;t)\,\mathcal N_3(r;t)}{1+K_3(r;t)}.
\]

The Phase-III gain frontier is

\[
\Delta_{\max}^{(3)}(t)
:=
\max_{r\in\mathcal R_3(t)}\Delta E_{\rm TR}(r;t).
\]

## State model

Route C needs one admitted scaffold with multiple accounting views.

Logical scaffold:

\[
\mathcal O_t^{\log}=(o_1,\ldots,o_{n_t}),
\qquad
I_t^{\log}=\{1,\ldots,n_t\}.
\]

Per-coordinate status must be provenance-based:

\[
\operatorname{status}_{t,j}\in
\{\operatorname{ordinary\_admitted},\operatorname{dormant\_plateau}\}.
\]

Dormant plateau block:

\[
D_t^0:=\{j\in I_t^{\log}:
\operatorname{status}_{t,j}=\operatorname{dormant\_plateau}\}.
\]

Do **not** define dormant status as every coordinate with `theta == 0`. An
ordinary Route-A coordinate may become numerically zero without being a Route-C
dormant acquisition.

Executable nonzero circuit:

\[
I_t^{\rm exec}:=
\{j\in I_t^{\log}:|\theta_{t,j}|>\theta_{\rm zero}\}.
\]

Repo contract:

- Dormant records are admitted scaffold coordinates.
- Dormant records have committed amplitude zero.
- Dormant records are included in novelty geometry, duplicate logic, and trial
  refit.
- Dormant records are omitted from executed nonzero gates while their committed
  amplitude is zero.

## Plateau entry

Ordinary Route A mode is used while

\[
\Delta_{\max}^{(3)}(t)>\varepsilon_{\rm plat}(t).
\]

Plateau acquisition mode is entered when

\[
\Delta_{\max}^{(3)}(t)\le\varepsilon_{\rm plat}(t).
\]

The current implementation may use a drop-streak proxy while the exact frontier
predicate is maturing. Proxy entries must log the entry source, e.g.

```text
entry_source = drop_plateau_proxy_v1
```

If negative-curvature singleton escape is available, expose the diagnostic

\[
\Delta E_{\rm nc}(r;t)=
\frac12[-\widetilde h_r]_+
\frac{\rho^2}{F_{r,\rm safe}^{\rm red}}.
\]

A strict plateau predicate should require both the ordinary trust gain and the
negative-curvature diagnostic to be below their configured floors, unless the
route explicitly delegates negative curvature to later joint unlock.

## Plateau candidate surface

Plateau acquisition should select from a declared plateau candidate surface,
not silently from an already-collapsed energy-only shortlist.

\[
\mathcal R_{\rm plat}(t):=
\begin{cases}
\mathcal R_3(t), & \mathcal R_3(t)\ne\varnothing\text{ and Phase III geometry is live},\\
\mathcal R_2(t), & \mathcal R_3(t)=\varnothing\text{ or Phase III shortlist collapsed},\\
\mathcal R_1(t)\text{ or a gated fresh surface}, & \text{if later surfaces are empty}.
\end{cases}
\]

Telemetry must record the source surface actually used.

## Plateau geometry context

For Route C v1.2, the default plateau context is the full admitted logical
scaffold:

\[
C_t^{\rm plat}(r):=I_t^{\log}.
\]

If a contracted context is used for cost reasons, it must be explicit and logged,
e.g.

\[
C_t^{\rm plat}(r):=W_r^{\rm act}\cup D_t^0.
\]

At the reportable state, define raw horizontal tangents

\[
\tau_j(t):=
Q_{\psi_t}\partial_{\theta_j}
|\psi(\mathcal O_t^{\log},\theta_t^{\log})\rangle,
\qquad j\in C_t^{\rm plat}(r),
\]

and candidate tangent

\[
\tau_r(t):=
Q_{\psi_t}\partial_{\alpha_r}
|\psi(\mathcal O_t^{\log}\oplus r,
(\theta_t^{\log},\alpha_r))\rangle\big|_{\alpha_r=0}.
\]

Build

\[
T_C=[\tau_j(t)]_{j\in C_t^{\rm plat}(r)}.
\]

Primitive Gram data:

\[
G_C:=\Re(T_C^\dagger T_C),
\qquad
q_r:=\Re(T_C^\dagger\tau_r),
\qquad
F_r:=\Re\langle\tau_r,\tau_r\rangle.
\]

These are Fubini--Study/QIM tangent objects, not energy-Hessian objects.

## Absolute and fractional residual gates

Raw supported residual strength:

\[
\sigma_r^\perp:=
\left[F_r-q_r^\top G_C^+q_r\right]_+.
\]

Fractional residual novelty:

\[
\mathcal N_r^{\rm frac}:=
\operatorname{clip}_{[0,1]}
\frac{\sigma_r^\perp}{\max(F_r,\varepsilon_{\rm red})}.
\]

Route C v1.2 should use both gates:

\[
\sigma_{r,\rm lcb}^\perp\ge\sigma_{\rm plat}^{\min},
\qquad
\mathcal N_{r,\rm lcb}^{\rm frac}\ge\nu_{\rm plat}^{\min}.
\]

Deterministic local diagnostics may set LCB values equal to raw values. Noisy or
measurement-facing routes should subtract the configured uncertainty margin.

Repo meaning:

- The absolute residual gate rejects tiny directions that look novel only by
  ratio.
- The fractional gate rejects large directions that are mostly already explained
  by the admitted scaffold.
- `fractional_residual_v1` may still report `N3_plat`; it is no longer the
  preferred primary ranking statistic.

## Primary v1.2 score: log-volume gain

Use ridge-stabilized residual geometry with

\[
\lambda_{\rm vol}>0,
\qquad
A_C:=G_C+\lambda_{\rm vol}I.
\]

Ridge residual:

\[
\sigma_{r|C}^{\lambda}:=
\left[F_r-q_r^\top A_C^{-1}q_r\right]_+.
\]

Incremental regularized log-determinant gain:

\[
\Delta V_{r|C}^{\lambda}:=
\log\left(1+\frac{\sigma_{r|C}^{\lambda}}{\lambda_{\rm vol}}\right).
\]

Convention: the physical Fubini--Study volume element uses `sqrt(det G)`, so the
physical log-volume gain is `0.5 * DeltaV`. The implementation uses the
log-determinant convention above. The factor `0.5` is positive and global, so it
does not change ranking; thresholds such as `phase3_plateau_volume_min` must use
the same convention as the implementation.

Primary plateau acquisition score:

\[
\boxed{
S_{3,\rm plat}^{\rm vol}(r;t):=
\frac{[\Delta V_{r|C}^{\lambda}]_{\rm lcb}}{1+K_3(r;t)}.
}
\]

Selection:

\[
r_t\in
\operatorname*{arg\,max}_{r\in\mathcal R_{\rm plat}(t)}
S_{3,\rm plat}^{\rm vol}(r;t),
\]

subject to duplicate blocking, compile/feasibility gates, absolute residual gate,
and fractional residual gate.

Repo contract:

- Plateau score must not multiply by `DeltaE_TR`, raw trust gain, gradient
  magnitude, or Schur energy gain.
- Cost remains in the denominator.
- `log_volume_v1` is the preferred implementation target.
- `fractional_residual_v1` is a compatibility/diagnostic selector, not the
  recommended Route-C default after v1.2 review.

## Optional residual-pool leverage

A stronger Geo-style diagnostic may form the residual-pool Gram over
\(\mathcal R_{\rm plat}(t)\):

\[
K_{uv|C}^{\lambda}:=
\Re\langle\tau_u,\tau_v\rangle
-q_u^\top(G_C+\lambda_{\rm vol}I)^{-1}q_v.
\]

Ridge leverage:

\[
\ell_r^{\lambda_P}:=
\left[K_{\mathcal R|C}^{\lambda}
(K_{\mathcal R|C}^{\lambda}+\lambda_P I)^{-1}\right]_{rr}.
\]

Recommended use for v1.2 implementation:

- optional telemetry;
- optional tie-breaker;
- optional gate `ell_lcb >= ell_plat_min`.

Do not make leverage mandatory for the first implementation unless tests and
runtime cost justify it.

## Block-aware extension

Plateau escape is expected to be collective. If selecting a block
\(B=\{r_1,\ldots,r_s\}\), define

\[
T_B=[\tau_{r_1},\ldots,\tau_{r_s}],
\qquad
F_{BB}=\Re(T_B^\dagger T_B),
\qquad
Q_{CB}=\Re(T_C^\dagger T_B).
\]

Residual block Gram:

\[
\Sigma_{B|C}^{\lambda}:=
F_{BB}-Q_{CB}^\top(G_C+\lambda_{\rm vol}I)^{-1}Q_{CB}.
\]

Block log-volume gain:

\[
\Delta V_{B|C}^{\lambda}:=
\log\det(I_s+\lambda_{\rm vol}^{-1}\Sigma_{B|C}^{\lambda}).
\]

Initial implementation may remain singleton-greedy. If block acquisition is
added, use greedy marginal increments:

\[
r_{\rm next}\in
\operatorname*{arg\,max}_{r\in\mathcal R_{\rm plat}(t)\setminus B}
\frac{[\Delta V_{r|C\cup B}^{\lambda}]_{\rm lcb}}{1+K_3(r;t)}.
\]

## Schur response role

Schur-refitted tangents are not the primary v1.2 plateau novelty primitive.
When the novelty projector uses the same admitted context, adding an in-context
Schur response does not change out-of-context residual geometry:

\[
\tau_r^\star=\tau_r+T_Cs_r
\quad\Longrightarrow\quad
(I-P_C)\tau_r^\star=(I-P_C)\tau_r.
\]

Repo contract:

- Use raw horizontal tangent geometry for plateau log-volume acquisition.
- Keep Schur/Phase-III objects for ordinary Route-A scoring, energy diagnostics,
  negative-curvature diagnostics, and refit/unlock validation.
- If compatibility mode computes Schur-refitted `N3_plat`, label it as such.

## Trial refit and unlock rule

The trial scaffold is

\[
\mathcal O_t^{\rm trial}:=\mathcal O_t^{\log}\oplus r_t.
\]

Initialize

\[
x_t^{(0)}=(\theta_t^{\log},0),
\]

where `theta_t^log` already includes zeros for dormant coordinates. Then run the
ordinary inner optimizer over the full enlarged vector:

\[
x_t^{\rm opt}\approx
\operatorname*{arg\,min}_x
E(\mathcal O_t^{\rm trial},x).
\]

Raw trial drop:

\[
\widehat\Delta_t^{\rm trial}:=E_t^{\rm rep}-E_t^{\rm trial}.
\]

Validated lower-confidence drop:

\[
Y_t^{\rm trial}:=
\widehat\Delta_t^{\rm trial}
-z_\alpha\sigma_{\Delta E,t}
-b_{E,\Delta}^{\rm cmp}(t)
-\gamma_{\rm hw}\Delta b_{E,r_t}^{\rm inc}(t).
\]

Unlock succeeds iff

\[
Y_t^{\rm trial}\ge\varepsilon_{\rm unlock}(t).
\]

Failed unlock:

\[
Y_t^{\rm trial}<\varepsilon_{\rm unlock}
\Longrightarrow
E_{t+1}^{\rm rep}=E_t^{\rm rep},
\quad
\theta_{D_{t+1}^0}=0,
\quad
r_t\text{ committed with }\operatorname{status}=\operatorname{dormant\_plateau}.
\]

Successful unlock:

\[
Y_t^{\rm trial}\ge\varepsilon_{\rm unlock}
\Longrightarrow
E_{t+1}^{\rm rep}=E_t^{\rm trial},
\quad
\theta_{t+1}^{\log}=x_t^{\rm opt}.
\]

Plateau-acquired coordinates with amplitude above `theta_zero` are promoted out
of `dormant_plateau`; those still below tolerance remain dormant.

## SPSA and finite-amplitude probes

SPSA inclusion remains required:

\[
i\in D_t^0\cup\{r_t\}
\Longrightarrow
\Delta_i^{(s)}\text{ exists and }c_s\Delta_i^{(s)}\ne0.
\]

Finite-amplitude seed/probe paths are optimizer-strength diagnostics. They do
not replace the geometry score and do not weaken the unlock rule.

Current diagnostic surface:

```text
phase3_plateau_seed_probe_mode = off | dormant_new_random_v1
phase3_plateau_seed_probe_count
phase3_plateau_seed_probe_radius
phase3_plateau_seed_probe_seed
```

## Plateau exhaustion

Route C must not admit zero records forever. Exhaustion should terminate, widen,
or switch fallback when any declared guard fires:

\[
|D_t^0|\ge D_{\rm dormant}^{\max},
\]

or

\[
\max_{r\in\mathcal R_{\rm plat}(t)}[\Delta V_{r|C}^{\lambda}]_{\rm lcb}
< V_{\rm plat}^{\min},
\]

or

\[
\max_{r\in\mathcal R_{\rm plat}(t)}\sigma_{r,\rm lcb}^\perp
<\sigma_{\rm plat}^{\min},
\]

or rank/log-volume failure persists for `M_rankfail` plateau admissions.

Telemetry must distinguish:

```text
route_c_exit_status = plateau_acquisition_exhausted
route_c_exit_status = plateau_surface_widened
route_c_exit_status = plateau_geometry_fallback
```

## Configuration surface

Required/target config fields:

```text
phase3_plateau_acquisition_mode
phase3_plateau_acquisition_score = fractional_residual_v1 | log_volume_v1
phase3_plateau_context = full_logical_scaffold_v1 | active_window_plus_dormant_v1
phase3_plateau_lambda_vol
phase3_plateau_sigma_min
phase3_plateau_nu_min
phase3_plateau_volume_min
phase3_plateau_leverage_mode = off | tie_break | gate | multiplicative
phase3_plateau_lambda_pool
phase3_plateau_leverage_min
phase3_plateau_unlock_margin
phase3_plateau_dormant_max
phase3_plateau_rank_eps
phase3_plateau_rankfail_patience
phase3_plateau_candidate_surface_policy
phase3_plateau_seed_probe_mode/count/radius/seed
```

Paper-facing Route C must manifest these values or explicitly mark optional
guards as disabled diagnostic knobs.

## Telemetry surface

For each plateau selection, log:

```text
route_c_plateau_entry_source
route_c_plateau_candidate_surface
route_c_plateau_acquisition_score
route_c_plateau_context
candidate_key
K3
DeltaE_TR_max
epsilon_plat
DeltaE_nc_max              # if available
sigma_perp
sigma_perp_lcb
N_frac
N_frac_lcb
lambda_vol
DeltaV_log
DeltaV_log_lcb
residual_pool_leverage      # optional
rank_context
rank_context_plus_candidate
D_dormant_count
trainable_indices
trainable_mask_dormant
seed_probe_*                # if enabled
E_report_before
E_trial
trial_drop
Y_trial
epsilon_unlock
unlock_success
promoted_logical_indices
remaining_dormant_indices
E_report_after
route_c_exit_status         # if exit/widen/fallback fires
```

## Minimum tests

1. **Identity safety**

   \[
   E(\mathcal O\oplus r,(\theta,0))=E(\mathcal O,\theta).
   \]

2. **Dormant context inclusion**

   A candidate whose tangent lies in the dormant-plus-active admitted span has
   near-zero residual and near-zero log-volume gain.

3. **Tiny-norm rejection**

   A candidate with high fractional novelty but
   \(\sigma_r^\perp<\sigma_{\rm plat}^{\min}\) fails the plateau pass gate.

4. **Large-redundant rejection**

   A candidate with large `F_r` but
   \(\mathcal N_r^{\rm frac}<\nu_{\rm plat}^{\min}\) fails the plateau pass gate.

5. **Log-volume ranking**

   Given two passing candidates, the one with larger
   \([\Delta V_{r|C}^{\lambda}]_{\rm lcb}/(1+K_3)
   ranks first.

6. **Marginal block behavior**

   A candidate's marginal log-volume gain decreases after an equivalent tangent
   has already been selected into the plateau block/context.

7. **SPSA inclusion**

   Dormant and selected plateau coordinates appear in the trainable vector and
   perturbation vector during plateau refit.

8. **Failed unlock rollback**

   Failed unlock commits zero amplitudes, restores reportable energy, and does
   not leak failed optimizer state.

9. **Successful collective unlock**

   A synthetic joint fixture activates multiple dormant coordinates only after
   validated energy unlock.

## Implementation order

1. Add the `log_volume_v1` score as a new plateau acquisition option.
2. Preserve `fractional_residual_v1` for compatibility and ablation.
3. Add residual gates and config plumbing.
4. Add telemetry fields before HH reruns.
5. Add focused unit tests for the geometry helper.
6. Only then rerun diagnostic HH cases.

## Handoff summary

Route C v1.2 keeps the Route-C state machine and energy safety rule from v1.1,
but replaces the primary plateau acquisition statistic. Plateau mode should buy
coordinates that increase the resolved local Fubini--Study volume of the admitted
logical scaffold. The selector uses raw horizontal tangent geometry against
`I_t^log`, including dormant plateau records, gates by absolute and fractional
residual novelty, ranks by lower-confidence log-volume gain over cost, and then
runs the ordinary enlarged refit. Energy unlock remains the only path from
zero-committed dormant coordinates to active nonzero coordinates.
