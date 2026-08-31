# Cached tangent--Schur prune screen implementation handoff

## Purpose

Add an intermediary prune screen between cheap nomination and the expensive
remove-refit prune trial. The nomination set is no longer amplitude-only:
amplitude collapse detects near-identity coordinates, while cached Schur and
tangent projection screens detect nonzero but locally recoverable coordinates.

The intended doctrine is:

$$
\text{amplitude, commutation, and cached surrogates nominate}
\to
\text{cached geometry screens}
\to
\text{ablation--refit decides}.
$$

This is not a code change by itself, and it is not a deletion certificate
supplied by commutation. Small amplitude is only a near-identity nomination
signal, and bare commutation only identifies structured compensator windows
worth testing. The cached tangent--Schur screen decides whether the current
branch geometry makes a mature coordinate locally recoverable enough to spend an
ablation/refit trial. The final prune decision remains the existing remove-refit
confidence rule with rollback.

## Required cached quantities

For a candidate coordinate \(j\) and a proposed compensator window
\(W\subseteq I_{-j}\), the implementation needs:

- current branch identity and state: \((\mathcal O_t^+,\theta_t^+)\);
- horizontal coordinate tangents \(\bar\tau_i=Q_{\psi_t}\partial_{\theta_i}|\psi_t\rangle\);
- tangent overlap block \(Q_j(W;t)\);
- tangent overlap vector \(q_j(W;t)\);
- tangent norm \(F_j(t)\);
- local gradient entries \(g_j(t)\) and \(g_W(t)\);
- Hessian block \(H_W(t)\);
- mixed Hessian vector \(h_{Wj}(t)\);
- deleted-coordinate curvature \(h_{jj}(t)\).

The cache is valid only if all of these were evaluated at the current branch
state and for the same projected prune window:

$$
\operatorname{valid}_{j}^{\mathrm{cache}}(W;t)=1.
$$

If the branch scaffold, parameters, refit window, or coordinate indexing has
changed since the cache was formed, the screen must be skipped or recomputed.
Stale geometry must not authorize a prune trial.

## Projection windows

Use two projection windows.

Flat-sector window:

$$
W_{j,\mathrm{flat}}^{-}(t)
:=
\{i\in I_{-j}:
\Omega_{m_i m_j}=1,\ \mathsf C_{m_i m_j}=1,\ 
\mathsf q_{m_i m_j}=\mathrm{exact}\}.
$$

This window contains only exact support-overlapping commuting compensators.
Support-disjoint commuting coordinates are not redundancy evidence unless the
tangent projection test itself succeeds.

Broader projected prune window:

$$
W_j^{-}(t)\subseteq I_{-j}.
$$

In the manuscript this is usually one of the typed prune ladder windows
\(W_j^{[s]}\). It may include correlated or noncommuting compensators depending
on the rung being tested.

If a projection window is empty, the projected screen abstains for that window.

## Screen definitions

For \(i,k\in W\), compute

$$
\bigl(Q_j(W;t)\bigr)_{ik}
:=
\operatorname{Re}\langle \bar\tau_i,\bar\tau_k\rangle,
$$

$$
\bigl(q_j(W;t)\bigr)_i
:=
\operatorname{Re}\langle \bar\tau_i,\bar\tau_j\rangle,
$$

$$
F_j(t)
:=
\operatorname{Re}\langle \bar\tau_j,\bar\tau_j\rangle+\varepsilon_F.
$$

With \(\lambda_{\mathrm{pr}}>0\), compute the normalized projected tangent
residual

$$
\rho_j^{\mathrm{tan}}(W;t)
:=
\operatorname{clip}_{[0,1]}
\left(
1-
\frac{
q_j(W;t)^\top
\bigl(Q_j(W;t)+\lambda_{\mathrm{pr}}I\bigr)^{-1}
q_j(W;t)
}{
F_j(t)
}
\right).
$$

Small \(\rho_j^{\mathrm{tan}}\) means coordinate \(j\)'s horizontal tangent is
already nearly spanned by the retained window. The actual prune screen should
use the amplitude-scaled Fubini deletion length. Define

$$
\delta_{W,F}^{\star}(j;t)
:=
\theta_{t,j}
\bigl(Q_j(W;t)+\lambda_{\mathrm{pr}}I\bigr)^{-1}q_j(W;t),
$$

and

$$
D_{j\mid W,F}^{2}(t)
:=
\theta_{t,j}^2F_j(t)
-2\theta_{t,j}q_j(W;t)^\top\delta_{W,F}^{\star}(j;t)
+
\bigl(\delta_{W,F}^{\star}(j;t)\bigr)^\top
Q_j(W;t)\delta_{W,F}^{\star}(j;t).
$$

In the unregularized ideal case,

$$
D_{j\mid W,F}^{2}(t)
=
\theta_{t,j}^2
\left(
F_j(t)-q_j(W;t)^\top Q_j(W;t)^+q_j(W;t)
\right).
$$

Thus \(\rho_j^{\mathrm{tan}}\) is a scale-free diagnostic, while
\(D_{j\mid W,F}^{2}\) is the state-space prune-screen quantity.

For the energy-side screen, compute the frozen deletion proxy

$$
\widehat L_j^{\mathrm{frz}}(t)
:=
-g_j(t)\theta_{t,j}
+\frac12 h_{jj}(t)\theta_{t,j}^2.
$$

Then compute

$$
R_j(W;t)
:=
\frac12\left(H_W(t)+H_W(t)^\top\right)+\lambda_H I,
$$

and, when the branch is not near a stable local refit, use the gradient-aware
Schur surrogate

$$
\widehat L_j^{\mathrm{Schur}}(W;t)
:=
-g_j(t)\theta_{t,j}
+\frac12h_{jj}(t)\theta_{t,j}^2
-\frac12
\bigl(g_W(t)-h_{Wj}(t)\theta_{t,j}\bigr)^\top
R_j(W;t)^{-1}
\bigl(g_W(t)-h_{Wj}(t)\theta_{t,j}\bigr).
$$

Near a locally refit state, the default projected Schur screen may use

$$
\widetilde h_j^{(-)}(W;t)
:=
h_{jj}(t)-h_{jW}(t) R_j(W;t)^{-1}h_{Wj}(t),
$$

and

$$
L_j^{\mathrm{Schur}}(W;t)
:=
\frac12\theta_{t,j}^2\max\{\widetilde h_j^{(-)}(W;t),0\}.
$$

The screen gate is

$$
\Gamma_j^{\mathrm{proj}}(W;t)
:=
\mathbf 1[\operatorname{valid}_{j}^{\mathrm{cache}}(W;t)=1]\,
\mathbf 1[D_{j\mid W,F}^{2}(t)\le(\tau_F^{\mathrm{pr}}(t))^2]\,
\mathbf 1[L_j^{\mathrm{Schur}}(W;t)\le\tau_{\mathrm{Schur}}^{\mathrm{pr}}(t)].
$$

## Candidate classification

Flat projected candidate:

$$
j\in\mathcal C_{\mathrm{flat}}^{\mathrm{proj}}(t)
\quad\Longleftrightarrow\quad
\Gamma_j^{\mathrm{proj}}(W_{j,\mathrm{flat}}^{-}(t);t)=1.
$$

Curvature-compensated projected candidate:

$$
j\in\mathcal C_{\mathrm{curv}}^{\mathrm{proj}}(t)
\quad\Longleftrightarrow\quad
\Gamma_j^{\mathrm{proj}}(W_j^{-}(t);t)=1
\ \text{and}\
\Gamma_j^{\mathrm{proj}}(W_{j,\mathrm{flat}}^{-}(t);t)=0.
$$

Flat projected candidates may use the ordinary prune confirmation path.
Curvature-compensated projected candidates may enter the prune trial path only
with the stricter curvature confirmation guard already defined by the manuscript.

The full nomination set sent to the probe cap is the union of amplitude,
cached-Schur, cached-Fubini-displacement, flat-projected, and
curvature-projected lanes:

$$
\mathcal C_{\mathrm{prune}}^{\mathrm{nom}}(t)
:=
\mathcal P_{\mathrm{amp}}(t)
\cup
\mathcal P_{\mathrm{Schur}}(t)
\cup
\mathcal P_{F}(t)
\cup
\mathcal C_{\mathrm{flat}}^{\mathrm{proj}}(t)
\cup
\mathcal C_{\mathrm{curv}}^{\mathrm{proj}}(t).
$$

Candidates outside the projected sets can still enter through the amplitude,
Schur, or tangent lanes, but the final deletion decision is still the
remove-refit confidence rule.

## Integration contract

Insert this screen after the existing cheap/stale/activity nomination and before
the expensive remove-refit trial.

Expected implementation flow:

1. Build the mature eligible set.
2. Build the amplitude/activity lane from the existing cheap score.
3. For mature coordinates with valid cache, build Schur and tangent-projection
   lanes.
4. Build exact flat and broader projected windows.
5. Check cache validity for each window.
6. Compute \(\rho_j^{\mathrm{tan}}\) and \(L_j^{\mathrm{Schur}}\) only from valid
   cached or recomputed current-branch geometry; use \(D_{j\mid W,F}^{2}\), not
   \(\rho_j^{\mathrm{tan}}\), as the state-space prune gate.
7. Classify the coordinate as flat projected, curvature-compensated projected,
   or protected/deprioritized.
8. Send only candidates in
   \(\mathcal C_{\mathrm{prune}}^{\mathrm{nom}}(t)\), or scheduled compression
   overrides, into the existing remove-refit ladder.
9. Preserve the existing final acceptance rule:

$$
U_j^{[s,N]}
\le
\Delta_{\mathrm{pr}}^{\mathrm{tol}}(t),
\qquad
\Gamma_{\mathrm{ret}}(j,t;s,N)=1,
\qquad
\Gamma_{\mathrm{curv}}(j,t;s,N)=1.
$$

The projected screen must not weaken the final energy-safety, retained-gain, or
rollback requirements.

## Acceptance criteria for the implementation agent

- The screen is categorical, not a continuous bonus in \(R_j^{\mathrm{cheap}}\).
- Amplitude collapse is one nomination lane, not the whole prune shortlist.
- Nonzero coordinates can be nominated through cached Schur or tangent-span
  recoverability.
- Cache validity is checked before any projected score is used.
- Exact support-overlapping commuting compensators define the flat-sector screen.
- Support-disjoint commutation never acts as redundancy evidence by itself.
- Noncommuting compensation is labeled curvature-compensated and uses stricter
  confirmation.
- Existing remove-refit confidence and rollback semantics remain authoritative.
