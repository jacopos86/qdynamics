# Prune Recoverability Implementation Handoff

Date: 2026-04-30

This note is for repo agents implementing the prune doctrine that was updated in the math source of truth. It summarizes what changed in `MATH/Math.md` and translates it into repo implementation tasks. No production code was changed by the math update.

## Source Of Truth

Authoritative source:

- `MATH/Math.md`, especially section `11.5.1 Branch-local ADAPT prune-recoverability ladder: nomination, refit, and rollback`.
- `MATH/Math.md`, time-dynamics section `17A.9` and surrounding hybrid checkpoint-controller definitions for the projected prune, McLachlan-Schur, shadow-observable, and persistence/homotopy doctrine.

Generated artifacts:

- `MATH/Math.tex`
- `MATH/Math.pdf`
- `MATH/adaptive_selection_staged_continuation.tex`
- `MATH/adaptive_selection_staged_continuation.pdf`

Do not edit generated TeX/PDF directly. Edit `MATH/Math.md`, then regenerate with:

```bash
python3 MATH/build/build_math_from_md.py
```

## High-Level Doctrine

The old live prune behavior treated prune mostly as deletion of small or collapsed coordinates. The updated doctrine treats prune as compression under recoverability.

The compact contract is:

$$
\boxed{
\text{ADAPT prune}
=
\text{cheap compensability nomination}
\to
\text{windowed remove-refit ladder}
\to
\text{shot-escalated confirmation}.
}
$$

For time dynamics, the compact contract is:

$$
\boxed{
\text{dynamics prune}
=
\text{cached McLachlan-Schur cost ladder}
\to
\text{projected reduced-state initialization}
\to
\text{short shadow observable certificate}.
}
$$

The important asymmetry is:

$$
\begin{aligned}
\text{time dynamics:}\qquad
&\text{Schur is a live cached McLachlan geometric object},\\
\text{static ADAPT:}\qquad
&\text{Schur is a local quadratic idealization of recoverability}.
\end{aligned}
$$

## 2026-04-30 Addendum: Surrogate Authority Boundary

The Hessian/Schur surrogate should be implemented for static ADAPT, but its
authority is limited. It is a live prune objective for ranking, compensator
window selection, and shot allocation; it is not the final deletion
certificate.

Static ADAPT target:

$$
\delta E_B^\star(t)
:=
\min_{\xi\in\Theta_{\mathcal O_t^{(-B)}}}
E(\mathcal O_t^{(-B)},\xi)
-
E(\mathcal O_t,\theta_t^+).
$$

Static ADAPT surrogate:

$$
\widetilde q_B^{(m)}(t)
:=
\frac12\theta_B^\top
\left[
\widetilde H_{BB}
-
\widetilde H_{BW_B^{(m)}}
\left(
\widetilde H_{W_B^{(m)}W_B^{(m)}}+\lambda I
\right)^+
\widetilde H_{W_B^{(m)}B}
\right]\theta_B.
$$

Implementation rule:

$$
\boxed{
\text{surrogate Schur ranking}
\to
\text{surrogate-selected remove-refit window}
\to
\text{shot-controlled energy confirmation}
\to
\text{commit or rollback}.
}
$$

The quasi-Newton cache must be damped, metric-scaled, symmetrized, PSD-projected,
ridged, and invalidated/embedded across scaffold mutations. A nonmonotone
surrogate Schur ladder is a health failure and should not aggressively prune.

Time dynamics is different. Do not import the static ADAPT Hessian surrogate
into the Chapter 17A realtime controller. Time dynamics uses the measured or
ideal-observable McLachlan geometry \(K_t=G_t+\Lambda_t\) and \(f_t\):

$$
\boxed{
\text{time dynamics}
=
\text{McLachlan-Schur prune-loss ladder}
\to
\text{projected reduced-state initialization}
\to
\text{differential miss + persistence}
\to
\text{short observable shadow certificate}.
}
$$

Exact ED/reference trajectories remain diagnostic side channels only. They may
appear in reports and overlays, but they must not decide ranking, admission,
shadow acceptance, persistence, or Optuna online feedback for a QPU-faithful
route.

## What Changed In Static ADAPT Math

### 1. The deletion target changed

Old effective target:

$$
\text{delete coordinate } j
\quad\text{when local small-amplitude or collapse proxies pass.}
$$

New target:

$$
\boxed{
\delta E_j^\star(t)
:=
\inf_{\xi\in\Theta_{\mathcal O_{t,-j}^+}}
E(\mathcal O_{t,-j}^+,\xi)
-
E(\mathcal O_t^+,\theta_t^+).
}
$$

The key implication is:

$$
|\theta_j|\approx0
\Longrightarrow
\delta E_j^\star(t)\approx0,
\qquad
\delta E_j^\star(t)\approx0
\not\Longrightarrow
|\theta_j|\approx0.
$$

Implementation meaning: small angle is only evidence. It is not the prune class. The prune class is compensable coordinates whose deletion can be recovered by refitting survivors.

### 2. Eligibility changed

Old behavior in the current code tends to require staleness, stagnation, and small angle before a candidate can even reach the prune trial.

New mature set:

$$
\boxed{
\mathcal M_t
:=
\Bigl\{
j:\ 
\operatorname{age}_{j,t}\ge a_{\min},
\ 
j\notin\mathcal P_{\mathrm{protect}}(t),
\ 
c_j(t)=0
\Bigr\}.
}
$$

Implementation meaning: staleness and stagnation should rank or prioritize candidates, not hard-block them. Protection and cooldown remain hard gates.

### 3. Nomination changed

Old behavior:

$$
\mathcal P_{\mathrm{probe}}
\subseteq
\mathcal J_{\angle}
$$

where the probe set is effectively restricted to small-angle coordinates.

New cheap recoverability prior:

$$
\boxed{
R_j^{\mathrm{cheap}}(t)
:=
w_A\widetilde A_j
+
w_V\widetilde V_j
+
w_G\widetilde G_j
-
w_C C_j^{\mathrm{comp}}
-
w_T C_j^{\mathrm{tan}}
-
w_{\kappa}\widetilde\kappa_j
-
w_{\mathrm{age}}z_j^{\mathrm{age}}
-
w_{\mathrm{stale}}z_j^{\mathrm{stale}}.
}
$$

with

$$
\boxed{
C_j^{\mathrm{comp}}(t)
:=
\max_{i\ne j}
\left\{
|\operatorname{corr}_W(\theta_j,\theta_i)|,\,
|\operatorname{corr}_W(g_j,g_i)|,\,
|\operatorname{corr}_W(\Delta\theta_j,\Delta\theta_i)|
\right\},
}
$$

and, only when tangent data are already cached,

$$
\boxed{
C_j^{\mathrm{tan}}(t)
:=
\max_{i\ne j}
\frac{
\left|\operatorname{Re}\langle\bar T_j,\bar T_i\rangle\right|
}{
\|\bar T_j\|_2\,\|\bar T_i\|_2+\varepsilon_T
}.
}
$$

Nomination set:

$$
\boxed{
\mathcal P_t^{(0)}
:=
\operatorname{Bottom}_{Q_{\mathrm{pr}}(t)}
\left(
R_j^{\mathrm{cheap}}(t);
j\in\mathcal M_t
\right).
}
$$

Implementation meaning: small-angle, amplitude collapse, stagnation, and old age are feature channels. They do not define the probe set by themselves. A non-small but compensable coordinate must be allowed into the trial layer.

### 4. Schur changed role in ADAPT

Old intended direction from the earlier draft:

$$
\widehat{\delta E}_{j,\mathrm{Schur}}
\quad\text{as if it were the default live ADAPT prune object.}
$$

New role:

$$
\widehat{\delta E}_{j,\mathrm{Schur}}
\quad\text{is optional simulation or quasi-Newton evidence.}
$$

The default ADAPT object is the refit ladder:

$$
\boxed{
\delta E_j^{(m)}(t)
:=
\min_{\xi\in\Theta_{\mathcal O_{t,-j}^+}}
\left\{
E(\mathcal O_{t,-j}^+,\xi):
\xi_i=((\theta_t^+)_{-j})_i
\ \forall i\notin W_j^{(m)}
\right\}
-
E(\mathcal O_t^+,\theta_t^+).
}
$$

with monotonicity:

$$
\boxed{
\delta E_j^{(0)}(t)
\ge
\delta E_j^{(1)}(t)
\ge
\cdots
\ge
\delta E_j^{(M)}(t)
\ge
\delta E_j^\star(t).
}
$$

Implementation meaning: frozen ablation is rung 0. Later rungs should broaden compensating refit windows. Terminal or scheduled compression may allow a full survivor refit.

### 5. Acceptance changed

Old behavior:

$$
\operatorname{Prune}_j(t)=1
\iff
\Gamma_{\mathrm{prune}}^{\mathrm{perm}}(t)
\wedge
j\in\widetilde{\mathcal M}_t
\wedge
\Gamma_j^{\mathrm{amp}\downarrow}(t)
\wedge
\Gamma_j^{\mathrm{safe}}(t)
\wedge
\Gamma_j^{\mathrm{ret}}(t).
$$

New behavior:

$$
\boxed{
\operatorname{Prune}_j(t)=1
\Longleftrightarrow
\Gamma_{\mathrm{prune}}^{\mathrm{perm,new}}(t)=1
\ \wedge\
j\in\mathcal M_t
\ \wedge\
\exists(m,N):
j\in\mathcal P_t^{(0)}
\ \wedge\
\mathsf D_j^{(m,N)}=\mathrm{accept}.
}
$$

where

$$
\boxed{
\mathsf D_j^{(m,N)}
:=
\begin{cases}
\mathrm{accept},
&
U_j^{(m,N)}
\le
\Delta_{\mathrm{pr}}^{\mathrm{tol}}(t)
\ \wedge\
\Gamma_{\mathrm{ret}}(j,t;m,N)=1,\\
\mathrm{reject},
&
m=M,\ N=N_{\mathrm{hi}},\
L_j^{(m,N)}
>
\Delta_{\mathrm{pr}}^{\mathrm{tol}}(t),\\
\mathrm{escalate},
&
\text{otherwise.}
\end{cases}
}
$$

Implementation meaning: low confidence or borderline evidence should escalate shot budget or refit window, not immediately reject. Acceptance is recoverability-first.

### 6. Safety tolerance changed

Old issue: retained-gain logic effectively made prune stricter exactly when the last admitted gain was small.

New tolerance:

$$
\boxed{
\Delta_{\mathrm{pr}}^{\mathrm{tol}}(t)
:=
\max\!\left(
\Delta_{\mathrm{num}},
c_{\mathrm{shot}}\sigma_E(t),
c_{\mathrm{scr}}\Delta_{\mathrm{scr}}(t),
\Delta_{\mathrm{chem}},
c_{\mathrm{rel}}|E_t^+-E_{\mathrm{target}}|
\right).
}
$$

Retained-gain guard is conditional:

$$
\Gamma_{\mathrm{ret}}(j,t;m,N)
:=
\mathbf 1[\Delta_{\mathrm{adm}}(t)<\Delta_{\mathrm{ret,on}}]
\vee
\mathbf 1\!\left[
E_t^-
-
\widehat E_N(\mathcal O_{t,-j}^+,\widehat\theta_{-j}^{(m,N)})
\ge
\eta_{\mathrm{ret}}\Delta_{\mathrm{adm}}(t)
+
z_\alpha\widehat\sigma_j(m,N)
\right].
$$

Implementation meaning: when the last admitted gain is below the useful floor, judge prune by absolute recoverability. Only protect a large just-admitted gain with retained-gain logic.

## Static ADAPT Implementation Targets

Primary code surfaces:

- `pipelines/static_adapt/adapt_pipeline.py`
  - `_execute_live_mature_prune_pass`
  - `_prune_refit_window_indices_live`
  - final checkpoint prune block near the existing `phase1_prune_final_mode` logic
  - `post_admission_prune` telemetry and beam prune diagnostics
- `pipelines/scaffold/hh_continuation_pruning.py`
  - `PruneConfig`
  - `rank_prune_candidates`
  - `cheap_prune_score`
  - `amplitude_collapse_witness`
  - `apply_pruning`
- CLI / Optuna surfaces:
  - `pipelines/static_adapt/optimization/phase3_policy_optuna.py`
  - parser/config declarations in `pipelines/static_adapt/adapt_pipeline.py`
- Tests:
  - `test/test_hh_continuation_pruning.py`
  - `test/test_adapt_vqe_integration.py`
  - `test/optimization/test_phase3_policy_optuna.py`
  - `test/test_hh_cost_energy_optuna.py`

Recommended implementation steps:

1. Add a new prune-policy version flag rather than silently changing old semantics. Suggested value: `phase1_prune_policy="recoverability_ladder_v1"`, with legacy behavior available behind a legacy value until tests and run recipes are migrated.

2. Change `rank_prune_candidates` so eligibility is:

   $$
   \operatorname{age}_{j,t}\ge a_{\min},\quad
   j\notin\mathcal P_{\mathrm{protect}}(t),\quad
   c_j(t)=0.
   $$

   Do not hard-require `stale_age`, `stagnation_threshold`, or small angle. Use them as features in `R_j^{cheap}`.

3. Replace the small-angle probe set with `R_j^{cheap}` ranking. Implement a helper that computes:

   - normalized late amplitude;
   - normalized recent motion;
   - cached gradient or stationarity signal when available;
   - selector burden;
   - age and staleness features;
   - compensability by coordinate-history correlations;
   - optional tangent overlap only when already available.

4. Keep `amplitude_collapse_witness` as diagnostic or ranking evidence. It must not be a required final acceptance gate under `recoverability_ladder_v1`.

5. Convert the existing one-window refit into a ladder:

   - rung 0: frozen ablation;
   - rung 1: current local window;
   - rung 2: local window plus correlated or old compensators;
   - final scheduled/terminal rung: optional full survivor refit.

   The current `_prune_refit_window_indices_live` can become a window builder that returns `list[list[int]]`.

6. Replace `max_regression` as the sole safety threshold with `Delta_pr_tol`. In noiseless local simulation, this can initially be:

   $$
   \Delta_{\mathrm{pr}}^{\mathrm{tol}}
   =
   \max(\Delta_{\mathrm{num}}, c_{\mathrm{scr}}\Delta_{\mathrm{scr}}, \Delta_{\mathrm{chem}})
   $$

   while shot variance terms are omitted when unavailable.

7. Make retained-gain conditional. Existing unconditional `retained_gain_ratio * admitted_gain` behavior should only apply when:

   $$
   \Delta_{\mathrm{adm}}(t)\ge \Delta_{\mathrm{ret,on}}.
   $$

8. Implement escalation semantics:

   - accept when the confidence upper bound is below tolerance and conditional retained-gain passes;
   - reject only after max rung and high budget fail;
   - otherwise escalate in refit width or shots.

   In deterministic/noiseless mode, set `sigma=0` and treat `N` as a nominal rung label.

9. Preserve exact rollback:

   $$
   \Xi_t^{\mathrm{pr}}
   :=
   (\mathcal O_t^+,\theta_t^+,E_t^+,\{c_i(t)\}_i,\text{optimizer state}).
   $$

   A failed prune must not modify scaffold, theta, energy, cooldowns except for the rejected candidate cooldown, optimizer memory, or metadata.

10. Add telemetry for gate-failure histograms:

   - permission open/closed reason;
   - mature eligible count;
   - prior-ranked candidate count;
   - small-angle count as diagnostic only;
   - amplitude witness status as diagnostic only;
   - selected rung;
   - window sizes;
   - frozen regression;
   - refit regression;
   - tolerance;
   - conditional retained-gain threshold;
   - accept/reject/escalate reason.

## What Changed In Time Dynamics Math

The earlier math update also changed the time-dynamics prune doctrine. Dynamics may use a genuine cached McLachlan-Schur loss because the controller already owns local geometry.

Exact local prune loss:

$$
K_k:=\bar G_k+\Lambda_k,
\qquad
f_k:=\operatorname{Re}(\bar T_k^\dagger\bar b_k).
$$

For active set \(A\):

$$
\epsilon_k^2(A)
:=
\|\bar b_k\|^2
-
f_{k,A}^{\top}K_{k,AA}^{+}f_{k,A}.
$$

For deleting block \(B\), with \(C=I_k\setminus B\):

$$
\boxed{
L_{B,k}^{\mathrm{pr,exact}}
=
\epsilon_k^2(C)-\epsilon_k^2(I_k)
=
f_{k,I}^{\top}K_{k,II}^{+}f_{k,I}
-
f_{k,C}^{\top}K_{k,CC}^{+}f_{k,C}.
}
$$

The prune ladder is an upper-bound ladder:

$$
\boxed{
L_{B,k}^{(0)}
\ge
L_{B,k}^{(1)}
\ge
\cdots
\ge
L_{B,k}^{(M)}
=
L_{B,k}^{\mathrm{pr,exact}}.
}
$$

Unlike ADAPT, the live dynamics controller can and should use this cached Schur/McLachlan geometry.

The dynamics update also replaced raw coordinate projection:

$$
(\mathcal O_k^{(-B)},\Pi_{-B}\theta_k)
$$

with reduced-state projection:

$$
\boxed{
\mathsf S_{\mathrm{prune}(B)}
(\mathcal O_k,\theta_k)
=
\left(
\mathcal O_k^{(-B)},
\mathcal P_{-B}(\theta_k)
\right).
}
$$

where \(\mathcal P_{-B}\) should minimize a local state/observable/velocity mismatch objective.

The final dynamics accept condition includes shadow observable safety:

$$
\boxed{
\rho_{B,k}^{\mathrm{pr},(m)}
\le
\tau_{\mathrm{pr}},
\qquad
\mathcal J_{B,k}^{\mathrm{shadow}}
\le
\tau_{\mathrm{shadow}},
\qquad
\Gamma_k^{\mathrm{calm}}=1.
}
$$

## Time Dynamics Implementation Targets

Primary code surfaces:

- `pipelines/time_dynamics/ap_mclachlan/controller.py`
  - `_cached_prune_loss`
  - `_prune_permitted`
  - `_prune_candidates`
  - `_build_pruned_runtime_state`
  - `_prune_no_harm_guard_reason`
  - `_select_prune_action`
  - action commit handling for `prune_coordinate`
- `pipelines/time_dynamics/ap_mclachlan/types.py`
  - prune config fields and telemetry fields
- `pipelines/time_dynamics/optimization/hh_realtime_optuna.py`
  - prune search-space knobs and objective reporting
- CLI adapter:
  - `pipelines/time_dynamics/runners/hh_from_adapt_artifact.py`
- Tests:
  - `test/test_hh_staged_cli_args.py`
  - `test/test_hh_realtime_checkpoint_types.py`
  - `test/test_hh_realtime_from_adapt_artifact.py`
  - `test/optimization/test_hh_realtime_optuna.py`

Recommended implementation steps:

1. Make `_cached_prune_loss` explicitly compute the McLachlan-Schur loss:

   $$
   f_I^\top K_{II}^{+}f_I
   -
   f_C^\top K_{CC}^{+}f_C,
   $$

   using the same ridge/regularization policy as the baseline solve. Keep the normalized telemetry field, but distinguish raw loss and normalized loss.

2. Implement a compensator ladder for prune candidates. The current candidate record has one `cached_prune_loss`; extend it to include rung rows:

   - compensator subset;
   - raw loss;
   - normalized loss;
   - monotonicity status;
   - selected rung.

3. Relax the absolute low-miss gate in `_prune_permitted`. The math allows high-miss prune trials when the candidate has small differential miss loss:

   $$
   \Delta\rho_{B,k}^{\mathrm{pr}}
   \le
   \tau_\rho^{\mathrm{pr}}.
   $$

   Keep strict/QPU-faithful data-flow rules: exact target/reference data cannot affect the decision.

4. Replace raw deletion in `_build_pruned_runtime_state` with reduced-state projection. The current delete-theta-block behavior is the spike risk. Add a projection routine that initializes survivor theta by minimizing a local objective with terms for:

   - Fubini-Study or prepared-state distance to incumbent;
   - observable bundle mismatch;
   - tangent velocity mismatch.

   In a first implementation, a bounded local least-squares/refit over survivor coordinates is acceptable if it uses only prepared-state observable/geometry quantities.

5. Upgrade `_prune_no_harm_guard_reason` from one-step score/rho checks to a shadow observable certificate. Compare the incumbent stay branch and projected-reduced branch over a short horizon \(H_{\mathrm{pr}}\) using the same integrator policy and Hamiltonian schedule. Include at least:

   - total energy;
   - staggered density;
   - doublon;
   - site occupations or density vector when available;
   - finite differences of those observables.

6. Add persistence before hard deletion where practical:

   - mark a candidate dormant-prunable when Schur loss, differential miss, and shadow observable score pass;
   - require persistence across `q_req` of the last `q_pers` checkpoints before commit; or
   - implement a homotopy deletion variable later if direct persistence is too invasive.

7. Preserve QPU-faithful decision boundaries. Exact ED/reference quantities may be computed as diagnostics only. Do not route exact future fidelity, exact future observable error, or exact target states into prune candidate ranking, accept/reject, shadow score, or online Optuna feedback for strict runs.

8. Add telemetry:

   - prune blocker reason categories;
   - `rho_expr`, `rho_real`, `rho_num` if available;
   - `NeedsSolveRepair`;
   - candidate Schur ladder rows;
   - differential miss;
   - projection objective terms;
   - shadow observable score;
   - persistence counters;
   - post-prune observable deltas.

## Tests To Add Or Rewrite

Static ADAPT:

- Candidate ranking test where a non-small but highly compensable coordinate outranks a small isolated coordinate.
- Test that small-angle and amplitude-collapse witnesses are diagnostic under `recoverability_ladder_v1`.
- Test that staleness affects ranking but not hard eligibility.
- Test that retained-gain is bypassed when `admitted_gain < Delta_ret_on`.
- Test rollback restores scaffold/theta/energy/optimizer metadata after reject.
- Test ladder escalation: frozen rung fails, wider refit rung accepts.
- Update integration assertions that currently require `probe_indices subset small_angle_pool_indices`.

Time dynamics:

- Test McLachlan-Schur prune loss equals direct recomputed reduced projection loss for a small synthetic geometry.
- Test prune ladder losses are monotone nonincreasing as compensator sets grow.
- Test raw projection is not used when projected initialization is enabled.
- Test shadow observable rejection catches a candidate with low geometric loss but large observable spike.
- Test high-miss differential prune can be permitted when candidate differential miss is safe.
- Test strict/QPU-faithful contract rejects exact-reference leakage in prune ranking or shadow acceptance.

## Migration Notes

- Keep legacy flags working initially, but route new behavior through explicit policy names to avoid silently changing old run recipes.
- Emit both old and new telemetry fields during transition when possible. Existing reports and tests consume fields like `small_angle_pool_indices`, `amplitude_witness_rows`, `cached_prune_loss`, and `post_prune_state_jump_l2`.
- Prefer implementation behind opt-in defaults first, then promote once the gate-failure histograms show expected behavior.
- The intended pressure increase is not looser final safety. Pressure increases by opening more opportunities, testing more candidates, allowing broader refits/projections, and escalating uncertain trials.
