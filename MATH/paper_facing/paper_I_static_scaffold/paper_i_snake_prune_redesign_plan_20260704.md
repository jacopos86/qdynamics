# Paper-I SNAKE Prune Redesign Plan

Status: planning note only. No implementation change is authorized by this file.

Created: 2026-07-04

Target: replace the current lexicographic prune nomination key with a scalar nomination score plus hard gates, while preserving the rule that measured delete-refit energy safety is the only deletion-commit authority.

## Current Design Decisions

As of 2026-07-04, the new route should be tracked as an alternative prune route, not as an immediate replacement of every existing behavior.

- Keep the current lexicographic ranking key only as a legacy/baseline option.
- Add a new option whose nomination object is a scalar score rather than the tuple

\[
K_j=
\left(
\Lambda_j^\ominus,
b_j,
\alpha_j,
-s_j,
\beta_j,
|\bar\theta_j|,
\ell_j
\right).
\]

Here \(K_j\) is the current lexicographic prune ranking key for deletion coordinate \(j\), \(\Lambda_j^\ominus\) is the Schur-predicted deletion loss, \(b_j\) is the stored admission score divided by stored burden, \(\alpha_j\) is the small-angle pool flag, \(s_j\) is the smallness/stagnation score, \(\beta_j\) is the stored selector burden, \(\bar\theta_j\) is the wrapped angle, and \(\ell_j\) is the deterministic coordinate label.

- Replace the raw Hessian-only Schur deletion model in the new route by a state-metric-regularized Schur model.
- Do not treat "conditioning of the ansatz" as a prune objective. If a condition number is recorded, it is telemetry for the local matrix inverted inside the Schur computation.
- Keep measured delete-refit energy safety as the only deletion-commit authority.

## 1. Current Behavior Summary

At adaptive step \(k\), write the current ordered ansatz as

\[
\mathcal O_k=(d_1,\ldots,d_{n_k}),
\qquad
E_k=E(\mathcal O_k,\boldsymbol\theta_k).
\]

Each coordinate \(d_j\) carries at least an admission step \(a_j\) and a cooldown value \(c_j(k)\). The current recoverability path uses the broad eligibility set

\[
\mathcal E_k^\ominus
=
\left\{
j\in\{1,\ldots,n_k\}:
k-a_j\ge p_{\rm protect},
\quad
c_j(k)\le 0
\right\}.
\]

Thus eligibility means "old enough" and "not on cooldown." Older small-angle, stale-age, and stagnation fields exist as telemetry, but the recoverability path is effectively protection-plus-cooldown.

For each eligible coordinate, the current Schur surrogate estimates a deletion-loss proxy \(\Lambda_j^\ominus\). The Schur value is used to rank or nominate candidates; it is not an acceptance certificate. In the current Paper-I path, `surrogate_exact_trial_cap=1`, so after Schur screening only one candidate is sent to the measured delete-refit test.

The current conceptual key is

\[
K_j=
\left(
\Lambda_j^\ominus,
b_j,
\alpha_j,
-s_j,
\beta_j,
|\bar\theta_j|,
\ell_j
\right),
\]

with smaller \(K_j\) preferred lexicographically. This is mechanically deterministic, but it is a poor conceptual control object: after the Schur entry, all other terms behave like a long tie-breaker ladder instead of a coherent prune utility.

Deletion acceptance is still governed by measured nonlinear energy safety. With the tested deletion coordinate \(j\),

\[
\Delta E_j^\ominus
:=
E(\mathcal O_k\ominus d_j,\boldsymbol\theta_{k,-j})-E_k,
\]

and deletion is accepted only if the measured delete-refit test passes the energy-regression, retained-gain, and curvature-safety guards. Schur does not commit deletion.

On rejection, the temporary delete branch is discarded. The post-admission child branch is restored, the generator remains in the ansatz, optimizer state is restored, and the rejected coordinate receives cooldown. The cooldown prevents immediate retry, but it does not by itself encode a long-term "this deletion has been unsafe before" penalty.

## 2. Replacement Object: State-Metric-Regularized Schur Score

The new route should make prune nomination match the Phase-III principle more closely: use an energy-curvature model, but regularize the compensating response by state-space displacement.

For deletion coordinate \(j\), let

\[
\alpha_j:=-\theta_{k,j}.
\]

Here \(\theta_{k,j}\) is the current optimized angle on generator \(d_j\), and \(\alpha_j\) is the coordinate displacement that deletes \(d_j\) by moving its parameter to zero.

Let \(W_j\) be the survivor-coordinate window opened for local compensation. The deletion-plus-compensation displacement is

\[
x_j(\delta\theta_W)
=
\alpha_j e_j
+
\sum_{i\in W_j}\delta\theta_i e_i .
\]

Here \(e_j\) is the coordinate basis vector for the deleted generator, \(e_i\) are coordinate basis vectors for survivor generators in \(W_j\), and \(\delta\theta_W=(\delta\theta_i)_{i\in W_j}\) is the compensating parameter displacement on those survivors.

The state-metric-regularized deletion model is

\[
\mathcal L_{j,\mu}^{\ominus}(\delta\theta_W)
=
g^\top x_j
+
\frac12 x_j^\top H x_j
+
\frac{\mu}{2}x_j^\top G x_j .
\]

Here \(g\) is the local energy-gradient vector over coordinates \(\{j\}\cup W_j\), \(H\) is the local energy Hessian over the same coordinates, \(G\) is the Fubini--Study Gram matrix over the same coordinates, and \(\mu\ge0\) controls how strongly state-space displacement penalizes the compensating response.

Writing the survivor blocks explicitly,

\[
M_{WW}^{(\mu)}
:=
H_{WW}+\mu G_{WW}+\lambda I,
\]

\[
r_{W}^{(\mu)}
:=
g_W+
\left(
H_{Wj}+\mu G_{Wj}
\right)\alpha_j .
\]

Here \(H_{WW}\) and \(G_{WW}\) are the Hessian and Fubini--Study blocks on the survivor window, \(H_{Wj}\) and \(G_{Wj}\) couple the deleted coordinate to that window, \(g_W\) is the survivor-gradient block, \(\lambda I\) is the numerical ridge used in the local solve, and \(r_W^{(\mu)}\) is the linear term seen by the survivor compensation variables.

The optimal local compensation is

\[
\delta\theta_{W,\mu}^{\star}
=
-
\left(
M_{WW}^{(\mu)}
\right)^+
r_W^{(\mu)} .
\]

Here \((\cdot)^+\) is the damped/pseudoinverse solve used for the local Schur block. This is the prune analogue of the Phase-III inherited-coordinate response, but with the Fubini--Study metric added to the local quadratic model.

The route's Schur nomination loss is

\[
\Lambda_{j,\mu}^{\ominus}(W_j)
:=
\min_{\delta\theta_W}
\mathcal L_{j,\mu}^{\ominus}(\delta\theta_W)
=
\mathcal L_{j,\mu}^{\ominus}(\delta\theta_{W,\mu}^{\star}) .
\]

Here \(\Lambda_{j,\mu}^{\ominus}(W_j)\) is still only a nomination model. The measured delete-refit energy change \(\Delta E_j^\ominus\) remains the acceptance quantity. Setting \(\mu=0\) recovers the raw Hessian-Schur route, while \(\mu>0\) penalizes compensations that look cheap in energy curvature but large in Fubini--Study state displacement.

## 3. Scalar Prune Nomination Score

The replacement should separate hard admissibility gates from a scalar utility score. The hard-gated nomination domain is

\[
\mathcal G_k^\ominus
=
\left\{
j\in\mathcal E_k^\ominus:
C_j^{\rm saved}>0,
\quad
\Lambda_{j,\mu}^\ominus<\infty,
\quad
\nu_j^\ominus\le \nu_{\max}
\right\}.
\]

Here \(\mathcal G_k^\ominus\) is the set of eligible coordinates that pass the new route's hard gates, \(C_j^{\rm saved}\) is the estimated resource saving from deleting \(d_j\), \(\Lambda_{j,\mu}^\ominus\) is the state-metric-regularized Schur nomination loss, and \(\nu_j^\ominus\) is the Fubini--Study deletion novelty. The novelty gate should be configurable; a first conservative version can make it telemetry-only by setting \(\nu_{\max}=1\).

For \(j\in\mathcal G_k^\ominus\), define the scalar prune nomination score

\[
R_j^\ominus
=
\frac{
C_j^{\rm saved}\,
w_j^{\rm small}\,
w_j^{\rm stale}\,
w_j^{\rm red}
}{
\epsilon_R
+\Lambda_{j,\mu,+}^\ominus
+\lambda_\nu\nu_j^\ominus
+\lambda_{\rm rej}\bar L_{j}^{\rm rej}
},
\qquad
\Lambda_{j,\mu,+}^\ominus:=\max\{0,\Lambda_{j,\mu}^\ominus\}.
\]

Here \(R_j^\ominus\) is the scalar prune priority, \(w_j^{\rm small}\), \(w_j^{\rm stale}\), and \(w_j^{\rm red}\) are smooth preference weights, \(\bar L_j^{\rm rej}\) is the accumulated prior rejection penalty, and \(\epsilon_R,\lambda_\nu,\lambda_{\rm rej}\) are route weights. The denominator deliberately excludes a standalone ansatz-conditioning term.

The nomination before persistence is

\[
j_k^\star
\in
\arg\max_{j\in\mathcal G_k^\ominus} R_j^\ominus.
\]

This score has the intended sign structure:

- large saved resource burden increases prune priority;
- small angle, stale age, and redundancy increase prune priority;
- predicted deletion loss, tangent novelty, and prior rejected loss decrease prune priority;
- negative predicted Schur loss is not allowed to collapse the denominator because \(\Lambda_{j,\mu,+}^\ominus\) is clipped at zero.

### Saved Burden

Let deleting coordinate \(d_j\) reduce the compiled and measurement proxies by

\[
\Delta N_{2q,j}^{+},
\quad
\Delta D_{2q,j}^{+},
\quad
\Delta D_{c,j}^{+},
\quad
\Delta S_j^{+},
\]

where \(x^+=\max\{0,x\}\). A normalized saved-burden score is

\[
C_j^{\rm saved}
=
\omega_N\frac{\Delta N_{2q,j}^{+}}{N_0+\epsilon_C}
+\omega_{D2}\frac{\Delta D_{2q,j}^{+}}{D_{2,0}+\epsilon_C}
+\omega_{Dc}\frac{\Delta D_{c,j}^{+}}{D_{c,0}+\epsilon_C}
+\omega_S\frac{\Delta S_j^{+}}{S_0+\epsilon_C}.
\]

The normalizers \(N_0,D_{2,0},D_{c,0},S_0\) can be current ansatz costs, recent median costs, or fixed benchmark scales. The weights \(\omega_\bullet\) define the prune resource model. A first implementation should record each component separately in telemetry even if the initial score uses only a subset.

### Small-Angle Preference

Use the wrapped coordinate amplitude

\[
\bar\theta_j=\operatorname{wrap}_{[-\pi,\pi)}(\theta_j),
\qquad
a_j^\theta=|\bar\theta_j|.
\]

Let

\[
\tau_{\theta,k}
=
\max\left\{
\tau_{\theta,\min},
\eta_\theta\,\operatorname{median}_{i\in\mathcal E_k^\ominus}
|\bar\theta_i|
\right\}.
\]

A smooth small-angle preference is

\[
w_j^{\rm small}
=
\left[
1+
\left(
\frac{a_j^\theta}{\tau_{\theta,k}+\epsilon_\theta}
\right)^{p_\theta}
\right]^{-1}.
\]

This replaces a binary small-angle flag with a continuous pressure. It is close to one for small angles and decays smoothly for large angles.

### Stale-Coordinate Preference

Let the branch-local admission age be

\[
A_j(k)=k-a_j.
\]

After the protection interval, stale age can be rewarded by

\[
w_j^{\rm stale}
=
1+\eta_{\rm stale}
\left[
1-
\exp\left(
-\frac{(A_j(k)-p_{\rm protect})_+}{\tau_{\rm age}}
\right)
\right].
\]

This does not make old coordinates automatically deletable. It only increases their nomination score when the Schur, novelty, history, and measured safety terms are also favorable.

### Redundancy Preference

The redundancy preference should be a bounded decreasing function of deletion novelty:

\[
w_j^{\rm red}
=
\sigma\left(
\frac{\nu_\star-\nu_j^\ominus}{\tau_\nu}
\right),
\qquad
\sigma(x)=\frac{1}{1+e^{-x}}.
\]

Here \(\nu_\star\) is the novelty level below which the coordinate is considered redundant enough to prefer deletion. This keeps \(w_j^{\rm red}\in(0,1)\). The same \(\nu_j^\ominus\) also appears in the denominator as an additive risk penalty; this is acceptable because the numerator acts like a soft preference and the denominator acts like risk control. If this double-counts novelty in practice, set either \(w_j^{\rm red}=1\) or \(\lambda_\nu=0\) in the first ablation.

## 4. Deletion Novelty / Redundancy

Prune novelty should be the deletion analogue of candidate tangent novelty. For pruning, high novelty protects the coordinate because it means the coordinate supplies a tangent direction not reconstructed by the remaining ansatz.

Let \(\tilde t_i\) be the horizontal Fubini--Study tangent direction associated with coordinate \(d_i\), evaluated at \((\mathcal O_k,\boldsymbol\theta_k)\). For a survivor set \(W_j\subseteq\{1,\ldots,n_k\}\setminus\{j\}\), define the Gram blocks

\[
G_{jj}=\operatorname{Re}\langle \tilde t_j,\tilde t_j\rangle,
\qquad
G_{jW}=
\left(
\operatorname{Re}\langle \tilde t_j,\tilde t_i\rangle
\right)_{i\in W_j},
\]

\[
G_{WW}=
\left(
\operatorname{Re}\langle \tilde t_i,\tilde t_\ell\rangle
\right)_{i,\ell\in W_j}.
\]

The survivor-explained fraction of \(d_j\)'s tangent is

\[
\eta_j^{\rm span}
=
\frac{
G_{jW}(G_{WW}+\lambda_G I)^+G_{Wj}
}{
G_{jj}+\epsilon_G
}.
\]

The deletion novelty is the clipped residual fraction

\[
\nu_j^\ominus
=
\Pi_{[0,1]}
\left(
1-\eta_j^{\rm span}
\right).
\]

Interpretation:

- \(\nu_j^\ominus\approx 0\): survivor tangents explain the deleted coordinate; deletion is geometrically plausible.
- \(\nu_j^\ominus\approx 1\): the coordinate contributes a tangent direction not spanned by survivors; deletion should be penalized or hard-gated.

The survivor set \(W_j\) can initially be the same Schur nomination window \(W_j^{\rm Schur}\), but a better geometry diagnostic is often an enlarged window or all survivors. The design should support

\[
W_j\in
\left\{
W_j^{\rm Schur},
\;W_j^{8},
\;W_j^{12},
\;\{1,\ldots,n_k\}\setminus\{j\}
\right\}.
\]

## 5. Persistence Gate

Exact-batch persistence is too brittle because the exact top batch can change even when the same coordinate family repeatedly appears unsafe or attractive. Use coordinate/label/support-family persistence.

Let \(\mathcal N_r^\ominus\) be the scored prune-nomination list at earlier step \(r\), before the measured delete-refit cap is applied. Define a stable family key

\[
\phi(j)\in
\left\{
\text{coordinate identity},
\text{generator label},
\text{Pauli support family},
\text{operator-class family}
\right\}.
\]

The persistence count for coordinate \(j\) is

\[
p_j(k)
=
\sum_{r=\max\{0,k-W_p\}}^{k-1}
\mathbf 1
\left\{
j\in\mathcal N_r^\ominus
\;\lor\;
\phi(j)\in \phi(\mathcal N_r^\ominus)
\right\}.
\]

Measured deletion is allowed only if

\[
p_j(k)\ge p_{\min}.
\]

The scalar score \(R_j^\ominus\) still chooses the candidate. Persistence is a stability gate after scoring:

\[
j_k^\star
=
\arg\max_{j\in\mathcal G_k^\ominus}R_j^\ominus,
\qquad
\text{measure delete}(j_k^\star)
\quad\text{only if}\quad
p_{j_k^\star}(k)\ge p_{\min}.
\]

If \(p_{j_k^\star}(k)<p_{\min}\), the step should record a persistence wait, update nomination history, and skip the measured delete-refit call. This avoids paying energy-measurement cost for one-step prune spikes.

## 6. History Penalty

Cooldown and history penalty should be separate objects.

Cooldown:

\[
c_j(k)>0
\quad\Longrightarrow\quad
j\notin\mathcal E_k^\ominus.
\]

Cooldown prevents immediate retry after rejection.

History penalty:

\[
\bar L_{j,k+1}^{\rm rej}
=
(1-\rho_{\rm rej})\bar L_{j,k}^{\rm rej}
+\rho_{\rm rej}
\max\{0,\Delta E_j^\ominus-\epsilon_k^\ominus\}
\]

after a rejected measured deletion trial for \(j\). If coordinate \(j\) is not tested at step \(k\), use decay only:

\[
\bar L_{j,k+1}^{\rm rej}
=
(1-\rho_{\rm idle})\bar L_{j,k}^{\rm rej},
\qquad
0\le\rho_{\rm idle}\le\rho_{\rm rej}.
\]

The penalty enters the denominator of \(R_j^\ominus\). A coordinate that repeatedly fails measured delete-refit by a large margin becomes harder to nominate later, even after cooldown expires. A coordinate rejected only by a tiny margin is penalized weakly.

## 7. Local Schur-Solve Telemetry

The route should not define "conditioning of the ansatz" as a prune score. The only conditioning quantity worth logging is the numerical stability of the matrix inverted inside the local Schur computation.

\[
M_{WW}^{(\mu)}
=
H_{WW}+\mu G_{WW}+\lambda I .
\]

Here \(M_{WW}^{(\mu)}\) is the local survivor block used to compute \(\delta\theta_{W,\mu}^{\star}\), \(H_{WW}\) is the survivor Hessian block, \(G_{WW}\) is the survivor Fubini--Study block, \(\mu\) is the state-metric regularization weight, and \(\lambda\) is the numerical ridge.

A local solve-stability diagnostic is

\[
\kappa_{j,\mu}^{\rm Schur}
:=
\kappa\!\left(M_{WW}^{(\mu)}\right).
\]

Here \(\kappa(\cdot)\) is the matrix condition number. Large \(\kappa_{j,\mu}^{\rm Schur}\) means the local Schur compensation solve is numerically fragile. It does not prove that the ansatz is poorly conditioned, and it does not measure physical deletion safety.

A companion rank diagnostic is

\[
r_{j,\mu}^{\rm Schur}
:=
\operatorname{rank}_{\tau}
\left(
M_{WW}^{(\mu)}
\right).
\]

Here \(\operatorname{rank}_{\tau}\) is numerical rank at threshold \(\tau\). This telemetry can explain unstable Schur nominations, but the first implementation should not include it in \(R_j^\ominus\) or in hard gates.

If a later ablation shows that solve instability is a common failure mode, add an optional gate such as

\[
\kappa_{j,\mu}^{\rm Schur}\le\kappa_{\max}^{\rm Schur}.
\]

Here \(\kappa_{\max}^{\rm Schur}\) would be a route setting. This gate should be off by default.

## 8. Schur Accuracy Improvements

The scalar score will only be useful if \(\Lambda_{j,\mu}^\ominus\) is a tolerable predictor of measured deletion loss. Add these ablation knobs before promoting a new prune rule:

1. Compare raw Hessian-Schur against state-metric-regularized Schur:

\[
\mu\in\{0,\mu_1,\mu_2,\mu_3\}.
\]

Here \(\mu=0\) is the existing raw Hessian-Schur deletion model, and \(\mu>0\) activates Fubini--Study regularization in the local compensation solve.

2. Increase local Schur survivor window size:

\[
w_{\rm Schur}\in\{4,8,12,\mathrm{full}\}.
\]

3. Replace the current one-left/three-right interior window for \(w_{\rm Schur}=4\) with a centered survivor window. For even window size \(w\), a centered interior choice is

\[
W_j^{\rm centered}
=
\{d_{j-w/2},\ldots,d_{j-1},d_{j+1},\ldots,d_{j+w/2}\},
\]

with boundary shifts only near the ends of the ansatz.

4. Increase measured delete-refit trial cap:

\[
M_{\rm exact}^\ominus\in\{1,2,3\}.
\]

The score can still produce a ranked nomination list; the cap controls how many Schur-safe candidates are measured.

5. Audit Schur prediction against measured regression on all existing prune trials:

\[
\left(
\Lambda_{j,\mu}^\ominus,\;
\Delta E_j^\ominus,\;
\Delta E_j^\ominus-\Lambda_{j,\mu}^\ominus,\;
A_j^\ominus,\;
\mu,\;
\kappa_{j,\mu}^{\rm Schur}
\right).
\]

Here the tuple records predicted deletion loss, measured deletion loss, prediction error, final accept/reject result, metric-regularization strength, and local Schur-solve conditioning. Report rank correlation, false-safe rate, false-dangerous rate, and error by \(\mu\) and window size. The critical failure mode is a low \(\Lambda_{j,\mu}^\ominus\) candidate with high measured \(\Delta E_j^\ominus\).

## 9. Measurement-Cost Implications

Energy regression is already the measured delete safety test:

\[
\Delta E_j^\ominus
=
E(\mathcal O_k\ominus d_j,\boldsymbol\theta_{k,-j})-E_k.
\]

This is the correct authoritative guard because it directly measures the post-delete energy loss.

State-metric regularization in \(\Lambda_{j,\mu}^\ominus\) requires the Fubini--Study blocks \(G_{WW}\) and \(G_{Wj}\). If those blocks are already estimated or cached in the SNAKE scoring path, the new route reuses them. If not, estimating them on hardware requires additional metric measurements. The first implementation should be cache-aware and should emit whether \(G\)-blocks were reused or newly measured.

State-overlap checks, state fidelity checks, and direct tangent-overlap diagnostics are useful simulator diagnostics, but they should not become default hardware-facing acceptance guards without a separate justification and a measurement-budget analysis.

Recommended default:

- use energy regression for acceptance;
- use state-metric-regularized Schur, novelty, history, and resource saved terms for nomination and telemetry;
- record local Schur-solve conditioning as telemetry only;
- allow overlap/fidelity diagnostics in simulator reports;
- require an explicit setting before any extra overlap measurement becomes part of hardware-facing prune acceptance.

## 10. Proposed Algorithmic Contract

At step \(k\):

1. Build \(\mathcal E_k^\ominus\) from protection and cooldown.
2. For each \(j\in\mathcal E_k^\ominus\), compute or retrieve \(C_j^{\rm saved}\), \(\Lambda_{j,\mu}^\ominus\), \(\nu_j^\ominus\), \(\bar L_j^{\rm rej}\), and local Schur-solve telemetry \(\kappa_{j,\mu}^{\rm Schur}\).
3. Apply hard gates to form \(\mathcal G_k^\ominus\).
4. Rank by scalar score \(R_j^\ominus\), not the lexicographic key \(K_j\).
5. Record the top scored list \(\mathcal N_k^\ominus\) for persistence.
6. Let \(j_k^\star=\arg\max R_j^\ominus\). If \(p_{j_k^\star}(k)<p_{\min}\), do not measure deletion; record persistence wait.
7. If persistence passes, run measured delete-refit for the top \(M_{\rm exact}^\ominus\) candidates.
8. Commit deletion only if measured energy safety passes.
9. On rejection, restore the post-admission child branch, restore optimizer state, keep the generator, apply cooldown, and update \(\bar L_j^{\rm rej}\).

The measured acceptance variable can remain structurally close to the current one:

\[
A_j^\ominus
=
\mathbf 1\{\Delta E_j^\ominus\le\epsilon_k^\ominus\}
\mathbf 1\{G_j^{\rm ret}\ge\rho_{\rm ret}G_k^{\rm adm}\}
\mathbf 1\{C_j^{\rm curv}=1\}.
\]

The redesign changes how \(j\) is nominated; it should not initially change the measured acceptance authority.

## 11. Implementation Later

Likely code hooks, based on the current live path:

- `pipelines/scaffold/hh_continuation_pruning.py`
  - `PruneConfig`: add score-mode options, Schur model mode, metric regularization \(\mu\), score weights, novelty settings, persistence settings, history-penalty settings, Schur-window variant, exact trial cap options, and solve-telemetry flags.
  - `build_static_prune_surrogate_scores`: emit both raw \(\Lambda_j^\ominus\) and metric-regularized \(\Lambda_{j,\mu}^\ominus\), with window diagnostics under multiple window choices.
  - `rank_prune_candidates`: keep legacy lexicographic ranking as a baseline mode; add scalar \(R_j^\ominus\), hard gates, and a scored nomination list for the new route.
  - `apply_pruning`: keep measured delete-refit acceptance authoritative; only extend returned decision telemetry if needed.

- `pipelines/static_adapt/adapt_pipeline.py`
  - `_build_phase1_prune_cfg`: expose the new settings.
  - `_build_prune_schur_nomination_scores`: add metric-regularized Schur, novelty, saved-cost, rejection-history, and solve-telemetry fields if the required cached geometry/cost data exist.
  - `_execute_live_mature_prune_pass`: maintain persistence history and rejection-loss history; continue restoring branch and optimizer state on rejection.
  - metadata transport around admission/prune: persist \(\bar L_j^{\rm rej}\), family keys \(\phi(j)\), nomination history, and cooldown.

- `pipelines/static_adapt/prune_schur_payloads.py`
  - extend payloads to report scalar score components, hard-gate outcomes, persistence waits, raw Schur versus metric-Schur values, and Schur-vs-measured audit rows.

- `pipelines/static_adapt/prune_risk_dataset.py`
  - extend diagnostic extraction to include \(R_j^\ominus\), \(\Lambda_{j,\mu}^\ominus\), \(\nu_j^\ominus\), \(\bar L_j^{\rm rej}\), \(\kappa_{j,\mu}^{\rm Schur}\), and measured \(\Delta E_j^\ominus\).

- `pipelines/static_adapt/cli_config.py` and `pipelines/static_adapt/output_artifacts.py`
  - expose settings and emit them in normalized manifests.

Do not change Paper-I benchmark results or manuscript claims until the redesigned prune path is implemented, audited against existing prune telemetry, and rerun under a user-approved benchmark contract.
