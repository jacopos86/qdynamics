# L=2 Noisy Surface Run Handoff

For the **local/offline L=2 HH noisy surface** (`phase3_v1` + `pareto_lean_l2` + `FakeNighthawk`), see [MATH/l2_noisy_surface_run.md](l2_noisy_surface_run.md).

This is the terminal-agent handoff note for the three mitigation lanes:

- `none`
- `readout`
- `full`

It is intentionally the front-page pointer so the noisy-surface run recipe is visible before the rest of the manuscript.

---

# ⚡ Real QPU Settings (April 5, 2026 Validated)

> **This is the recommended configuration for real QPU deployment.**
> Validated via narrow 36-case local sweep confirming optimality in the `pareto_lean` + `phase3_v1` regime.

## Anchor Configuration

**Use this exact command for real QPU L=2 ADAPT runs:**

```bash
python -u -m pipelines.hardcoded.adapt_pipeline \
  --L 2 --problem hh --t 1.0 --u 4.0 --dv 0.0 \
  --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --boson-encoding binary --ordering blocked --boundary open --term-order sorted \
  --adapt-pool pareto_lean --adapt-continuation-mode phase3_v1 \
  --adapt-max-depth 160 --adapt-eps-grad 5e-7 --adapt-eps-energy 1e-9 --adapt-seed 5 \
  --adapt-inner-optimizer SPSA \
  --adapt-spsa-a 0.1 --adapt-spsa-c 0.02 --adapt-spsa-A 5.0 \
  --adapt-spsa-callback-every 5 --adapt-spsa-progress-every-s 30 \
  --adapt-maxiter 3200 \
  --adapt-state-backend compiled \
  --adapt-reopt-policy windowed --adapt-window-size 999999 --adapt-window-topk 999999 \
  --adapt-full-refit-every 8 --adapt-final-full-refit true \
  --adapt-beam-live-branches 2 --adapt-beam-children-per-parent 2 --adapt-beam-terminated-keep 2 \
  --phase1-prune-enabled --phase1-prune-fraction 0.25 --phase1-prune-max-candidates 6 --phase1-prune-max-regression 1e-8 \
  --phase1-probe-max-positions 999999 --phase1-trough-margin-ratio 1.0 --phase1-shortlist-size 64 \
  --phase2-shortlist-fraction 1.0 --phase2-shortlist-size 64 \
  --phase2-frontier-ratio 0.85 \
  --phase2-lambda-H 1e-06 --phase2-rho 0.25 --phase2-gamma-N 1.0 \
  --phase2-w-depth 0.1 --phase2-w-group 0.075 --phase2-w-shot 0.075 --phase2-w-optdim 0.05 --phase2-w-reuse 0.05 --phase2-w-lifetime 0.05 \
  --phase2-eta-L 0.0 --phase2-motif-bonus-weight 0.05 --phase2-duplicate-penalty-weight 0.0 \
  --phase2-compat-overlap-weight 0.4 --phase2-compat-comm-weight 0.2 --phase2-compat-curv-weight 0.2 \
  --phase2-compat-sched-weight 0.2 --phase2-compat-measure-weight 0.2 \
  --phase2-remaining-evaluations-proxy-mode auto \
  --phase3-frontier-ratio 0.85 \
  --phase3-symmetry-mitigation-mode verify_only --phase3-enable-rescue \
  --phase3-backend-cost-mode proxy --phase3-runtime-split-mode off --phase3-lifetime-cost-mode off \
  --adapt-drop-floor 0.0005 --adapt-drop-patience 3 --adapt-drop-min-depth 12 \
  --initial-state-source adapt_vqe --skip-pdf --phase2-no-batching
```

## Expected Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Energy | 0.15877612622031317 | Exact ground state reference |
| \|ΔE\| | **1.082e-4** | Excellent accuracy for NISQ |
| Ansatz depth | 20 | Shallow = low error on real hardware |
| Parameters | 52 | Efficient parameterization |
| Wall-clock | ~43s | Fast convergence |

## Validation Results (April 5 Sweep)

A 36-case narrow local sweep confirmed this configuration is optimal:

- ✅ **SPSA maxiter=3200**: Goldilocks zone (2400 undershoots, 4000 overshoots)
- ✅ **Frontier ratio 0.85**: All values (0.80–0.90) converge to same energy; 0.85 balances quality
- ✅ **Beam (2,2,2)**: Sufficient; wider beam (3,2,3) costs 50% more time, zero quality gain
- ✅ **Drop policy (0.0005, 3, 12)**: Optimal shallowness; tighter drop → shallower circuits for QPU

### Critical Finding

Increasing SPSA `maxiter` beyond 3200 **degrades** performance (1.27–1.54e-4 vs 1.082e-4):
- Suggests pool saturation: `pareto_lean` operators are exhausted at this budget
- SPSA noise amplification: longer optimization in finite ansatz space doesn't help

**Do not deviate from maxiter=3200 in this regime.**

## What NOT to Change

Based on sweep validation:
- ❌ Pool: Do not switch from `pareto_lean`
- ❌ Continuation: Do not move away from `phase3_v1`
- ❌ Runtime split: Keep `off` for L=2
- ❌ SPSA budget: Do not increase `maxiter` beyond 3200
- ❌ Frontier: Do not vary frontier ratio (insensitive in 0.80–0.90 range)
- ❌ Beam: Do not use (3,2,3); keep (2,2,2)

## For Real QPU Deployment

1. Use the command above exactly as written
2. Expect |ΔE| ≈ 1.08e-4 (this is the best-known trustworthy result)
3. Ansatz depth = 20 is excellent for NISQ mitigation
4. No further local tuning will improve this configuration (sweep confirms)

## Companion Driven Candidate

The leading **driven realtime-controller** candidate for the same L=2 HH scaffold family is now the heavier-target
`secant_lead100` setting from
`artifacts/agent_runs/20260405_controller_drive_heavier_secant_lead_scan_v1/json/secant_lead100.json`.

This should be tracked alongside the winning static/noiseless L=2 ADAPT anchor above when choosing real-QPU-facing settings, because it is currently the best controller law we have for recovering visible driven dynamics on the stronger `A=0.55`, `tbar=4.0`, `t_final=8.0` benchmark.

### Current driven-controller headline metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Controller law | `secant_lead100` | Signed-energy-lead-tapered secant + signed blend |
| Drive target | `A=0.55`, `omega=2.0`, `tbar=4.0`, `t0=0.0` | Heavier than the earlier fine-resolution regime |
| Mean `|ΔE|` | **1.633e-3** | Best whole-window value among current heavier-target candidates |
| Max `|ΔE|` | **6.501e-3** | Large improvement over prior signed-blend and raw secant anchors |
| Early-window MAE (`t≤3.35`) | **1.176e-3** | Recovers the startup dip far better than `signed_mild_front_current` |
| Late-window MAE (`t≥3.35`) | **1.964e-3** | Still slightly better than the previous late-time winner |
| Energy span ratio | `1.032` | Slight overshoot, but close to exact |
| Min fidelity | `0.9626` | Remains bounded with no append events |

### Leading QPU-facing pair

- **Static / ground-state anchor:** the validated `pareto_lean + phase3_v1` L=2 ADAPT command above.
- **Driven / dynamics anchor:** the heavier-target `secant_lead100` controller result above.

These are the two front-page candidates we should currently treat as the leading settings when optimizing toward a real QPU run.

---

---
title: "Hubbard-Holstein Mathematical Notes (Symbolic-Only Edition)"
author: "Jake Skyler Strobel"
date: "March 20, 2026"
geometry: margin=0.8in
fontsize: 10pt
toc: true
toc-depth: 3
---

> Canonical naming note: `MATH/Math.md` is the symbolic manuscript and corresponds to `MATH/Math.pdf`. The archival repo-oriented manuscript is `MATH/archaic_repo_math.md`.

# 1. Parameter Manifest and Reader Contract

This duplicate is intentionally symbolic-first. It keeps the same broad manuscript shape as the existing mathematical notes,

1. primitives first,
2. composite operators second,
3. explicit substitutions third,
4. reduced master forms last,

but it suppresses almost all implementation labels, file names, mode strings, and executable-surface commentary. The goal is to make the document readable as mathematics rather than as a repository audit.

## 1.1 Required parameter manifest

A complete symbolic run or derivation should specify at least the following data.

- Model family: Hubbard or Hubbard-Holstein.
- Lattice size: $L$.
- Ordering map: interleaved or blocked, i.e. a choice of $p(i,\sigma)$.
- Boundary condition: open or periodic.
- Fermionic parameters: hopping $t$, onsite interaction $U$, and static site potentials $v_i$.
- Bosonic parameters: phonon frequency $\omega_0$, electron-phonon coupling $g$, and cutoff $n_{\mathrm{ph,max}}$.
- Boson encoding choice: binary or unary.
- Variational family, when relevant: layerwise, physical-termwise, excitation-based, or adaptive.
- Control parameters, when relevant: optimizer family, propagation rule, drive waveform parameters, and benchmark grid.

## 1.2 Reader contract

This manuscript favors explicit substitution over compressed abstraction.

- If a primitive operator exists, it is written first.
- If a composite object is built from primitives, the primitive form is shown.
- If a later equation can be simplified by inserting an earlier primitive explicitly, that insertion is made.
- If multiple mathematical surfaces exist, such as different index orderings or boson encodings, the surface is named locally rather than hidden behind an unnamed default.

### 1.2.1 Quick notation glossary for adaptive and continuation sections

This subsection is a compact map for the symbols that recur in the adaptive-selection, split-aware, replay, and beam-style sections.

- $g,m$: generic candidate generators. The manuscript uses both letters in different sections; they play the same role.
- $p$: candidate insertion position in the ordered scaffold.
- $r=(m,p)$: a candidate-position record, i.e. a generator together with a chosen insertion position.
- $\mathcal G$: the full master generator pool.
- $\mathcal G^{(1)},\mathcal G^{(2)},\mathcal G^{(3)}$: progressively more restricted candidate universes, with
  $$
  \mathcal G^{(3)}\subseteq\mathcal G^{(2)}\subseteq\mathcal G^{(1)}\subseteq\mathcal G.
  $$
- $\mathcal P_g$ or $\mathcal P_m$: admissible insertion positions for candidate $g$ or $m$.
- $W(p)$: inherited refit window used if a candidate is inserted at position $p$.
- $W_{\mathrm{new}}(p)$: the newest-parameter portion of the refit window.
- $W_{|\theta|}(p)$: the older large-amplitude carry subset of the refit window.
- $\mathcal P_{\mathrm{avail}}^{(3)}$: the full available set of Phase-3 candidate-position records.
- $\mathcal C_{\mathrm{cheap}}$: the horizon-sized coarse shortlist obtained by cheap-score ranking over the full admissible Phase-3 candidate-position pool.
- $\mathcal S_2,\mathcal S_3$: the Phase-2 and Phase-3 shortlists that survive cheap screening before full reranking.
- $\mathcal C_{\mathrm{split}}(m)$: the family of split children generated from macro generator $m$.
- $\Gamma_{\mathrm{prune}}^{\mathrm{perm}}$: post-admission prune-permissibility gate deciding whether local scaffold simplification is attempted.
- $\mathcal J_{\mathrm{exp}}$: the post-admission set of locally expendable scaffold coordinates or generators considered by the prune pass.
- $\mathcal O_*$: the finally selected adaptive scaffold, i.e. the ordered generator sequence admitted by the adaptive stage.
- $\mathcal M_{\mathrm{warm}},\mathcal M_{\mathrm{refine}},\mathcal M_{\mathrm{replay}}$: archival appendix-only warm-start, optional refine, and replay manifolds.
- $\mathcal B$: a portable handoff/state bundle carrying manifest, amplitudes, energies, and continuation metadata.
- $\mathcal C$: continuation payload / provenance payload attached to a handoff bundle.
- $\Gamma_{\mathrm{stage}}$: stage-admissibility gate.
- $\Gamma_{\mathrm{sym}}$: symmetry-admissibility gate.
- $\mathfrak b$: a beam branch state. In §11 it indexes alternative ADAPT scaffolds; in §17 it indexes projected-dynamics branches over checkpoint segments.
- $\mathcal H_b$: admitted-record history carried by scaffold branch $b$.
- $\mathcal A_b$: branch-local admissible admission set retained for scaffold-beam expansion.
- $\mathcal Q(b)$: scaffold-beam proposal family, usually $\{\mathrm{stop}\}\cup\mathcal A_b$.
- $\eta_b$: branch-local stage label for the scaffold beam, e.g. core/seed versus residual.
- $W_{\mathrm{act}}(\mathfrak b)$: the currently active position-probe window on branch $b$.
- $\mathfrak F_c$: live frontier at beam round $c$.
- $\mathfrak T_c$: terminal-branch pool at beam round $c$.
- $\mathfrak W_{\mathrm{final}}$: final union of surviving frontier and terminal branches.
- $J_b$: cumulative branch objective tuple used to rank or prune branch $b$; its coordinates are section-specific.

### 1.2.2 Quick parameter glossary for adaptive and continuation sections

The symbols below are the main scalar controls that appear in the adaptive-selection, split-aware, and beam-style formulas.

- $\lambda_{\mathrm{2q}}$: weight on two-qubit proxy burden.
- $\lambda_d$: weight on depth-style proxy burden.
- $\lambda_\theta$: weight on marginal parameter-count burden.
- $\lambda_F$: cheap-stage normalization weight multiplying the tangent metric term.
- $\lambda_H$: regularization strength for inherited-window Hessian blocks.
- $\rho$: trust-region radius scale used in one-dimensional local improvement models.
- $N_{\mathrm{cheap}}^{\max}$: absolute cap on the horizon-sized Phase-3 coarse shortlist.
- $N_{\max}$: maximum shortlist size.
- $f_{\mathrm{short}}$: shortlist fraction used in Phase 3.
- $w_R$: weight on useful-horizon lifetime burden.
- $w_D$: weight on lifetime depth/compile burden.
- $w_G$: weight on newly introduced grouped-measurement burden.
- $w_C$: weight on newly introduced compile or circuit burden.
- $w_c$: weight on any extra cheap-path correction burden.
- $\gamma_N$: exponent applied to novelty factors in full rerank scores.
- $\varepsilon$: small positive stabilizer used in denominators and tie-safe quotients.
- $\varepsilon_{\mathrm{nov}}$: novelty regularizer used when inverting or stabilizing tangent-overlap matrices.
- $z_\alpha$: confidence multiplier used to convert a raw gradient estimate into a lower-confidence gradient.
- $\sigma(g,p)$ or $\sigma_r$: uncertainty estimate attached to the candidate-position gradient signal.
- $\tau_1$: Phase-2 shortlist threshold.
- $\tau_{\mathrm{split}}$: split-replacement margin required before a split child or child-set beats its parent.
- $\tau_{\mathrm{drop}}$: minimum per-depth absolute-error improvement regarded as meaningful by the stage controller.
- $M$: patience window used when counting repeated small-drop depths.
- $D_{\mathrm{left}}(t)$: remaining controller depth budget at selector step $t$.
- $S_t(k)$: survival factor for having a still-useful admission opportunity $k$ selector steps ahead.
- $Q_t(k)$: conditional usefulness factor for the $k$-ahead admission opportunity.
- $\widehat N_{\mathrm{rem}}(t)$: heuristic forecast of useful remaining admissions at selector step $t$.
- $C_t$: confidence attached to the current useful-runway forecast.
- $K_{\mathrm{coarse}}(t)$: dynamic coarse-cap induced by the current useful-runway forecast.
- $H_t$: effective useful-depth horizon,
  $$
  H_t=\min\bigl(D_{\mathrm{left}}(t),\widehat N_{\mathrm{rem}}(t)\bigr).
  $$
- $W_{\mathrm{sat}}(t)$: saturation pressure induced by the current useful-runway forecast.
- $\pi_t$: selector mode tag, e.g. fast / mixed / high-fidelity.
- $\mathrm{regime}_{\mathrm{sat}}(t)$: hungry / watch / plateau saturation regime.
- $\eta_{\mathrm{ret}}$: minimum retained-gain fraction required by the post-admission prune accept rule.
- $\beta_{\mathrm{keep}}$: repo-facing retained-gain floor used in prune acceptance; in this manuscript it plays the same role as $\eta_{\mathrm{ret}}$.
- $\tau_{\mathrm{miss}}$: projected-miss threshold used in manifold-growth logic.
- $B_{\mathrm{child}}$: beam child cap per parent branch-expansion round.
- $B_{\mathrm{target}}(t)$: active batch target size at selector step $t$.
- $B_{\mathrm{live}}$: live-frontier beam width retained after pruning.
- $B_{\mathrm{term}}$: cap on stored terminal branches.
- $M_{\mathrm{probe}}$: cap on distinct probed insertion positions retained in order.
- $a_j(t)$: admission age of scaffold coordinate or generator $j$ at selector step $t$.
- $\varrho_j(t)$: normalized admission-rank score of coordinate or generator $j$ at selector step $t$.
- $\mathcal P_{\mathrm{protect}}(t)$: hard-protected coordinate set excluded from the live prune pass.
- $c_j^{\mathrm{cool}}(t)$: local prune cooldown on coordinate $j$.
- $S_\theta(j,t)$: optional small-angle prescreen score used only to save local prune work.
- $\tau_{\mathrm{stale}}$: minimum local expendability score required before a prune trial is even considered.
- $\Xi_t^{\mathrm{trial}}$: exact rollback snapshot taken before a local prune/remove-refit trial.
- $\Delta k_{\mathrm{cp}}$: checkpoint advance used in the projected-dynamics beam of §17, not in the scaffold beam of §11.
- $\Delta k_{\mathrm{probe}}$: probe advance used in the projected-dynamics beam of §17, not in the scaffold beam of §11.
- $C$: final beam round index / number of retained beam-update rounds.

In §11, the index $t$ denotes a selector/controller step rather than the real-time variable used later in the dynamics sections.

## 1.3 Non-negotiable representational conventions

### 1.3.1 Internal Pauli alphabet

In formulas below we use the standard alphabet
$$
\{I,X,Y,Z\}.
$$
If one prefers the lower-case alphabet $\{e,x,y,z\}$, the identification is simply
$$
e \leftrightarrow I,
\qquad
x \leftrightarrow X,
\qquad
y \leftrightarrow Y,
\qquad
z \leftrightarrow Z.
$$

### 1.3.2 Pauli-word and qubit ordering

Pauli words and computational-basis labels are written left-to-right as
$$
q_{N_q-1}\cdots q_1 q_0,
$$
with qubit $q_0$ the rightmost symbol and also the least-significant bit in basis-index arithmetic.

### 1.3.3 Canonical algebra sources

The canonical mathematical primitives are:

- Jordan-Wigner ladder operators $\hat c_p^\dagger$ and $\hat c_p$,
- fermion number operators $\hat n_p$,
- boson ladder operators $\hat b_i^\dagger$ and $\hat b_i$,
- Hermitian Pauli words and Pauli polynomials,
- ordered exponentials generated from those primitives.

### 1.3.4 Surface-specific defaults

This symbolic version does not pretend there is a single universal surface. The mathematics admits several equally valid choices:

- interleaved or blocked fermion ordering,
- open or periodic boundary conditions,
- binary or unary boson encoding,
- static or driven Hamiltonians,
- fixed-structure or adaptive variational manifolds.

Whenever a formula depends on the choice of surface, that dependence is shown locally.

## 1.4 Canonical mathematical anchors

The main objects carried through the manuscript are:

- the ordering map $p(i,\sigma)$,
- the ladder primitives $\hat c_p^\dagger$, $\hat c_p$, $\hat b_i^\dagger$, $\hat b_i$,
- the density operators $\hat n_{i\sigma}$, $\hat n_i$, and $\hat D_i$,
- the Hubbard and Hubbard-Holstein Hamiltonians,
- the reference state $|\psi_{\mathrm{ref}}\rangle$,
- the variational manifold $\mathcal M$ and its ansatz map $\theta \mapsto |\psi(\theta)\rangle$,
- the propagation rule $U(t)$,
- the continuation or handoff bundle $\mathcal B$.

# 2. Ordering, Indexing, and Register Layout

## 2.1 Site, spin, and mode indices

The site index is
$$
i \in \{0,1,\dots,L-1\},
$$
and the spin label is stored as
$$
\sigma \in \{\uparrow,\downarrow\} \equiv \{0,1\}.
$$
The fermion mode index is the Jordan-Wigner qubit index.

### 2.1.1 Interleaved ordering

The interleaved map is
$$
p(i,\sigma)=2i+\sigma,
$$
so that
$$
p(i,\uparrow)=2i,
\qquad
p(i,\downarrow)=2i+1.
$$

### 2.1.2 Blocked ordering

The blocked map is
$$
p(i,\uparrow)=i,
\qquad
p(i,\downarrow)=L+i.
$$

## 2.2 Pauli-word placement and basis-index extraction

If a Pauli letter acts on qubit $q$, then in a printed word of length $N_q$ it sits at string position
$$
\operatorname{pos}(q)=N_q-1-q.
$$

If the computational-basis index is $k$, then the occupation bit on qubit $q$ is
$$
b_q(k)=\left\lfloor\frac{k}{2^q}\right\rfloor \bmod 2.
$$
Thus the printed bitstring and integer basis index obey the same rightmost-$q_0$ convention.

## 2.3 Full HH register layout

The fermion register uses
$$
N_{\mathrm{ferm}}=2L
$$
qubits.

If the local phonon cutoff is $n_{\mathrm{ph,max}}$, then the local phonon Hilbert dimension is
$$
d=n_{\mathrm{ph,max}}+1.
$$
The number of phonon qubits per site is
$$
q_{\mathrm{pb}}=
\begin{cases}
\max\{1,\lceil \log_2 d\rceil\}, & \text{binary},\\
d, & \text{unary}.
\end{cases}
$$
Therefore the total HH qubit count is
$$
N_q=2L+Lq_{\mathrm{pb}}.
$$

In qubit-index order, the register is naturally partitioned as
$$
[\text{fermions}\;|\;\text{site-0 phonons}\;|\;\cdots\;|\;\text{site-(L-1) phonons}].
$$
In printed order $q_{N_q-1}\cdots q_0$, the high-index phonon blocks appear on the left.

# 3. Fermionic Primitives and Direct Substitution

## 3.1 Jordan-Wigner ladder primitives

For mode $p$, the creation operator is
$$
\hat c_p^\dagger
=
\frac{1}{2}(X_p-iY_p)\prod_{r=0}^{p-1}Z_r,
$$
and the annihilation operator is
$$
\hat c_p
=
\frac{1}{2}(X_p+iY_p)\prod_{r=0}^{p-1}Z_r.
$$

In fully expanded printed-word form, the operator acts on the word ordered as $q_{N_q-1}\cdots q_0$ with the Jordan-Wigner $Z$-string extending through all modes below $p$.

## 3.2 Number primitive

The fermion number operator is
$$
\hat n_p=\hat c_p^\dagger\hat c_p=\frac{I-Z_p}{2}.
$$

## 3.3 Site densities and doublon operator

At site $i$, define
$$
\hat n_i = \hat n_{i\uparrow}+\hat n_{i\downarrow},
$$
and the doublon operator
$$
\hat D_i = \hat n_{i\uparrow}\hat n_{i\downarrow}.
$$
By direct substitution,
$$
\hat D_i = \frac{1}{4}(I-Z_{p(i,\uparrow)})(I-Z_{p(i,\downarrow)}).
$$

## 3.4 Worked ordering substitutions

### 3.4.1 Interleaved

Under interleaved ordering,
$$
\hat n_i = I - \frac{1}{2}(Z_{2i}+Z_{2i+1}),
$$
and
$$
\hat D_i = \frac{1}{4}(I-Z_{2i})(I-Z_{2i+1}).
$$
For nearest-neighbor spin-up hopping in a one-dimensional chain,
$$
\hat c_{i\uparrow}^\dagger \hat c_{i+1,\uparrow} + \hat c_{i+1,\uparrow}^\dagger \hat c_{i\uparrow}
=
\frac{1}{2}\Bigl(X_{2i} Z_{2i+1} X_{2i+2} + Y_{2i} Z_{2i+1} Y_{2i+2}\Bigr).
$$

### 3.4.2 Blocked

Under blocked ordering,
$$
\hat n_i = I - \frac{1}{2}(Z_i + Z_{L+i}),
$$
and
$$
\hat D_i = \frac{1}{4}(I-Z_i)(I-Z_{L+i}).
$$
For spin-up nearest-neighbor hopping, the Jordan-Wigner string is empty because the two blocked spin-up modes are adjacent, so
$$
\hat c_{i\uparrow}^\dagger \hat c_{i+1,\uparrow} + \hat c_{i+1,\uparrow}^\dagger \hat c_{i\uparrow}
=
\frac{1}{2}\Bigl(X_i X_{i+1} + Y_i Y_{i+1}\Bigr).
$$
The general blocked-ordering formula uses the same Jordan-Wigner interval rule as before, with an empty product for adjacent modes.

# 4. Boson Primitives and Encodings

## 4.1 Local boson Hilbert space

For each site $i$, the truncated boson Hilbert space is
$$
\mathcal H_{b,i}=\operatorname{span}\{|n_i\rangle : 0\le n_i\le n_{\mathrm{ph,max}}\}.
$$
The ladder operators satisfy
$$
\hat b_i^\dagger |n\rangle = \sqrt{n+1}\,|n+1\rangle,
\qquad
\hat b_i |n\rangle = \sqrt{n}\,|n-1\rangle,
$$
with
$$
\hat n_{b,i}=\hat b_i^\dagger \hat b_i,
\qquad
\hat x_i=\hat b_i+\hat b_i^\dagger,
\qquad
\hat P_i=i(\hat b_i^\dagger-\hat b_i).
$$

## 4.2 Binary encoding

In binary encoding, each local occupation number $n\in\{0,\dots,d-1\}$ is mapped to a binary word on $q_{\mathrm{pb}}=\lceil \log_2 d\rceil$ qubits:
$$
|n\rangle \longmapsto |\operatorname{bin}(n)\rangle.
$$
The truncated number operator is then
$$
\hat n_{b,i} = \sum_{n=0}^{d-1} n\,|n\rangle\langle n|,
$$
understood as an operator on the binary-encoded subspace.

## 4.3 Unary encoding

In unary encoding, each local occupation $n$ is represented by a one-hot basis state on $d$ qubits:
$$
|n\rangle \longmapsto |0\cdots 010\cdots 0\rangle,
$$
with the single $1$ marking the occupied unary slot. The number operator remains
$$
\hat n_{b,i} = \sum_{n=0}^{d-1} n\,|n\rangle\langle n|,
$$
but now the basis states are embedded in a one-hot code space.

## 4.4 Phonon vacuum

### 4.4.1 Binary vacuum

The local binary vacuum is
$$
|0\rangle_{b,i} \longmapsto |00\cdots 0\rangle.
$$

### 4.4.2 Unary vacuum

The local unary vacuum is the one-hot state carrying occupancy zero,
$$
|0\rangle_{b,i} \longmapsto |10\cdots 0\rangle,
$$
or whichever one-hot convention is chosen consistently for the lowest occupation.

# 5. Hubbard Hamiltonian by Explicit Substitution

Write the static Hubbard Hamiltonian as
$$
\hat H_{\mathrm{Hub}} = \hat H_t + \hat H_U + \hat H_v.
$$

## 5.1 Kinetic term

The kinetic term is
$$
\hat H_t = -t\sum_{\langle i,j\rangle,\sigma}
\left(
\hat c_{i\sigma}^\dagger \hat c_{j\sigma}
+
\hat c_{j\sigma}^\dagger \hat c_{i\sigma}
\right).
$$
For two fermion modes $p<q$, direct Jordan-Wigner substitution gives
$$
\hat c_p^\dagger \hat c_q + \hat c_q^\dagger \hat c_p
=
\frac{1}{2}
\left(
X_p Z_{p+1}\cdots Z_{q-1} X_q
+
Y_p Z_{p+1}\cdots Z_{q-1} Y_q
\right).
$$

## 5.2 Onsite interaction

The onsite interaction term is
$$
\hat H_U = U\sum_i \hat n_{i\uparrow}\hat n_{i\downarrow}
= U\sum_i \hat D_i.
$$
Substituting the number primitive,
$$
\hat H_U = \frac{U}{4}\sum_i (I-Z_{p(i,\uparrow)})(I-Z_{p(i,\downarrow)}).
$$

## 5.3 Static potential term

The static potential term is
$$
\hat H_v = \sum_{i,\sigma} v_i \hat n_{i\sigma}.
$$
After substitution,
$$
\hat H_v = \sum_i v_i\left(I-\frac{1}{2}(Z_{p(i,\uparrow)}+Z_{p(i,\downarrow)})\right).
$$

## 5.4 Fully substituted Hubbard Hamiltonian

Combining the pieces,
$$
\hat H_{\mathrm{Hub}}
=
-t\sum_{\langle i,j\rangle,\sigma}
\left(
\hat c_{i\sigma}^\dagger \hat c_{j\sigma} + \hat c_{j\sigma}^\dagger \hat c_{i\sigma}
\right)
+
\frac{U}{4}\sum_i (I-Z_{p(i,\uparrow)})(I-Z_{p(i,\downarrow)})
+
\sum_i v_i\left(I-\frac{1}{2}(Z_{p(i,\uparrow)}+Z_{p(i,\downarrow)})\right),
$$
with the kinetic term expanded further into Pauli strings as needed.

# 6. Hubbard-Holstein Hamiltonian by Explicit Substitution

The driven Hubbard-Holstein Hamiltonian is
$$
\hat H_{\mathrm{HH}}(t)
=
\hat H_t + \hat H_U + \hat H_v + \hat H_{\mathrm{ph}} + \hat H_g + \hat H_{\mathrm{drive}}(t).
$$

## 6.1 Phonon energy

The phonon energy is
$$
\hat H_{\mathrm{ph}} = \omega_0\sum_i \left(\hat n_{b,i}+\frac{1}{2}I\right).
$$

### 6.1.1 Unary explicit form

In unary form, $\hat n_{b,i}$ is diagonal on the one-hot occupation basis:
$$
\hat n_{b,i} = \sum_{n=0}^{d-1} n\,|n\rangle_i\langle n|.
$$
For the full qubit-level Pauli reduction in the spin-block JW + unary setting, including the explicit $I/Z$ form of $\hat H_{\mathrm{ph}}$, see Section 6.2.3.4(a).

### 6.1.2 Binary explicit form

In binary form,
$$
\hat n_{b,i} = \sum_{n=0}^{d-1} n\,|\operatorname{bin}(n)\rangle_i\langle \operatorname{bin}(n)|,
$$
understood as an operator on the truncated binary code space.

## 6.2 Electron-phonon coupling

A natural density-shifted coupling is
$$
\hat H_g = g\sum_i \hat x_i\bigl(\hat n_i-\bar n I\bigr),
$$
where $\bar n = N_e/L$ is the reference density, often $\bar n=1$ at half filling.

### 6.2.1 Unary explicit form

In unary encoding, $\hat x_i$ acts as nearest-neighbor hopping on the truncated local occupation chain:
$$
\hat x_i = \sum_{n=0}^{d-2} \sqrt{n+1}\Bigl(|n+1\rangle_i\langle n| + |n\rangle_i\langle n+1|\Bigr).
$$
Thus
$$
\hat H_g = g\sum_i \hat x_i\bigl(\hat n_i-\bar n I\bigr)
$$
with $\hat x_i$ understood in that unary basis. For the detailed spin-block JW + unary Pauli reduction of the Holstein sector, including the explicit $ZXX/ZYY$ expansion, see Sections 6.2.3.2–6.2.3.4.

### 6.2.2 Binary explicit form

In binary encoding, the same operator is written on the binary code space:
$$
\hat x_i = \sum_{n=0}^{d-2} \sqrt{n+1}\Bigl(|\operatorname{bin}(n+1)\rangle_i\langle \operatorname{bin}(n)| + |\operatorname{bin}(n)\rangle_i\langle \operatorname{bin}(n+1)|\Bigr),
$$
with the same density factor $(\hat n_i-\bar n I)$.

## 6.2.3 Worked Holstein-only reduction after Jordan-Wigner fermions and unary phonons

Set
$$
N_b = n_{\mathrm{ph,max}}.
$$
In this worked subsection, isolate the Holstein part of the Hubbard-Holstein Hamiltonian:
$$
\hat H_{\mathrm{Hol}}
=
\omega_0\sum_{i=0}^{L-1}\hat b_i^\dagger \hat b_i
\;+
\;g\sum_{i=0}^{L-1}(\hat b_i^\dagger+\hat b_i)(\hat n_i-1),
\qquad
\hat n_i=\hat n_{i\uparrow}+\hat n_{i\downarrow}.
$$

### 6.2.3.1 Qubit indexing used

For fermions, use spin-block Jordan-Wigner ordering:
$$
q(i,\sigma)=i+\sigma L,
\qquad
\sigma=0\equiv\uparrow,
\quad
\sigma=1\equiv\downarrow.
$$
Thus
$$
q_{i\uparrow}=i,
\qquad
q_{i\downarrow}=i+L,
\qquad
i=0,\dots,L-1.
$$

For phonons, use a unary register with cutoff $N_b$. Per site $i$, allocate $N_b+1$ qubits labelled by $n=0,\dots,N_b$, and place all fermions first and phonons second:
$$
q_{i,n}^{(b)} = 2L + i(N_b+1)+n,
\qquad
n=0,\dots,N_b.
$$

Let $X_q$, $Y_q$, and $Z_q$ denote the Pauli operators on qubit $q$.

### 6.2.3.2 Jordan-Wigner fermionic number operators

Jordan-Wigner gives
$$
\hat n_{i\sigma} = \frac{I-Z_{q_{i\sigma}}}{2},
$$
so
$$
\hat n_i-1
=
\left(\frac{I-Z_{q_{i\uparrow}}}{2}+\frac{I-Z_{q_{i\downarrow}}}{2}\right)-1
=
-\frac{Z_{q_{i\uparrow}}+Z_{q_{i\downarrow}}}{2}.
$$

### 6.2.3.3 Unary phonon ladder operators in Pauli form

Define the single-qubit combinations
$$
(+)_{q}=\frac{X_q+iY_q}{2},
\qquad
(-)_{q}=\frac{X_q-iY_q}{2}.
$$

Then the unary truncated ladder operators are
$$
\hat b_i^\dagger
=
\sum_{n=0}^{N_b-1}\sqrt{n+1}\,(+)_{q_{i,n}^{(b)}}\,(-)_{q_{i,n+1}^{(b)}},
$$
$$
\hat b_i
=
\sum_{n=1}^{N_b}\sqrt{n}\,(-)_{q_{i,n-1}^{(b)}}\,(+)_{q_{i,n}^{(b)}}.
$$

A convenient real Pauli expansion is
$$
\hat b_i^\dagger+\hat b_i
=
\frac12\sum_{n=0}^{N_b-1}\sqrt{n+1}
\Big(
X_{q_{i,n}^{(b)}}X_{q_{i,n+1}^{(b)}}
+
Y_{q_{i,n}^{(b)}}Y_{q_{i,n+1}^{(b)}}
\Big).
$$

### 6.2.3.4 Explicit Holstein Hamiltonian in the qubit basis

#### (a) Phonon energy term

On the unary one-hot code space, the phonon number operator is diagonal and may be written as
$$
\hat b_i^\dagger\hat b_i \equiv \hat n_i^{(b)}
=
\sum_{n=0}^{N_b} n\,\frac{I-Z_{q_{i,n}^{(b)}}}{2}
\qquad
\text{(exact on the one-hot subspace)}.
$$
Therefore
$$
\boxed{
\hat H_{\mathrm{ph}}
=
\omega_0\sum_{i=0}^{L-1}\sum_{n=0}^{N_b} n\,\frac{I-Z_{q_{i,n}^{(b)}}}{2}
}.
$$

Equivalently, as constant plus Pauli-$Z$,
$$
\hat H_{\mathrm{ph}}
=
\underbrace{
\omega_0\frac12\sum_i\sum_{n=0}^{N_b} n
}_{\displaystyle \omega_0 L\frac{N_b(N_b+1)}{4}}
I
-
\frac{\omega_0}{2}\sum_{i=0}^{L-1}\sum_{n=0}^{N_b} n\,Z_{q_{i,n}^{(b)}}.
$$
The global constant shift may be dropped if desired.

#### (b) Electron-phonon coupling term

Using
$$
\hat n_i-1 = -\frac{Z_{q_{i\uparrow}}+Z_{q_{i\downarrow}}}{2}
$$
and the unary expansion of $\hat b_i^\dagger+\hat b_i$, one obtains
$$
\boxed{
\hat H_{e\text{-}ph}
=
-\frac{g}{4}\sum_{i=0}^{L-1}\sum_{n=0}^{N_b-1}\sqrt{n+1}
\bigl(Z_{q_{i\uparrow}}+Z_{q_{i\downarrow}}\bigr)
\Big(
X_{q_{i,n}^{(b)}}X_{q_{i,n+1}^{(b)}}
+
Y_{q_{i,n}^{(b)}}Y_{q_{i,n+1}^{(b)}}
\Big)
}.
$$

The fully expanded Pauli-string sum is
$$
\hat H_{e\text{-}ph}
=
-\frac{g}{4}\sum_{i=0}^{L-1}\sum_{n=0}^{N_b-1}\sqrt{n+1}
\Big[
Z_{q_{i\uparrow}}X_{q_{i,n}^{(b)}}X_{q_{i,n+1}^{(b)}}
+
Z_{q_{i\uparrow}}Y_{q_{i,n}^{(b)}}Y_{q_{i,n+1}^{(b)}}
+
Z_{q_{i\downarrow}}X_{q_{i,n}^{(b)}}X_{q_{i,n+1}^{(b)}}
+
Z_{q_{i\downarrow}}Y_{q_{i,n}^{(b)}}Y_{q_{i,n+1}^{(b)}}
\Big].
$$
Each summand is three-local.

#### Final Holstein Hamiltonian in JW + unary form

Collecting the phonon-energy and electron-phonon pieces,
$$
\boxed{
\hat H_{\mathrm{Hol}}^{\mathrm{(JW+unary)}}
=
\omega_0\sum_{i=0}^{L-1}\sum_{n=0}^{N_b} n\,\frac{I-Z_{q_{i,n}^{(b)}}}{2}
-
\frac{g}{4}\sum_{i=0}^{L-1}\sum_{n=0}^{N_b-1}\sqrt{n+1}
\bigl(Z_{q_{i\uparrow}}+Z_{q_{i\downarrow}}\bigr)
\Big(
X_{q_{i,n}^{(b)}}X_{q_{i,n+1}^{(b)}}
+
Y_{q_{i,n}^{(b)}}Y_{q_{i,n+1}^{(b)}}
\Big)
}.
$$

## 6.3 Time-dependent density drive

A general density drive may be written as
$$
\hat H_{\mathrm{drive}}(t) = \sum_i v_i(t)\hat n_i.
$$
A useful waveform class is
$$
v_i(t)=A s_i \sin(\omega t + \phi)\exp\!\left[-\frac{(t-t_0)^2}{2\bar t^2}\right],
$$
where $A$ is the amplitude, $s_i$ is a static spatial pattern, $\omega$ is the drive frequency, $\phi$ is the phase, and $\bar t$ is the envelope width.

### 6.3.1 Generic density-drive waveform surface

After inserting $\hat n_i=I-\tfrac12(Z_{p(i,\uparrow)}+Z_{p(i,\downarrow)})$, the drive separates into an identity offset and a sum of time-dependent $Z$-coefficients. Thus the time dependence resides in the coefficients rather than in the operator labels.

## 6.4 Two HH assembly surfaces

### 6.4.1 Exact Hamiltonian surface

The exact surface treats $\hat H_{\mathrm{HH}}(t)$ as an operator to be diagonalized, exponentiated, projected, or applied directly to statevectors.

### 6.4.2 Variational HH ansatz surface

The variational surface treats the same physical pieces as generator families from which one builds a structured manifold
$$
|\psi(\theta)\rangle = U(\theta)|\psi_{\mathrm{ref}}\rangle.
$$

## 6.5 Fully substituted HH master form

The full symbolic master form is therefore
$$
\hat H_{\mathrm{HH}}(t)
=
\hat H_{\mathrm{Hub}}
+
\omega_0\sum_i \left(\hat n_{b,i}+\frac{1}{2}I\right)
+
 g\sum_i \hat x_i(\hat n_i-\bar n I)
+
\sum_i v_i(t)\hat n_i,
$$
with $\hat n_i$, $\hat n_{b,i}$, and $\hat x_i$ expanded according to the chosen fermion ordering and boson encoding.

# 7. Reference States and Exact Sector ED

## 7.1 Fermionic Hartree-Fock determinant

Let $\mathcal O$ denote the occupied spin-orbital set in a Slater determinant reference. Then
$$
|\Phi_{\mathrm{HF}}\rangle = \prod_{p\in\mathcal O} \hat c_p^\dagger |\mathrm{vac}_f\rangle.
$$

## 7.2 Hubbard-Holstein reference state

The natural HH reference is the tensor product of the phonon vacuum and fermionic Hartree-Fock state:
$$
|\psi_{\mathrm{ref}}^{\mathrm{HH}}\rangle = |\mathrm{vac}_{\mathrm{ph}}\rangle \otimes |\Phi_{\mathrm{HF}}\rangle.
$$

## 7.3 Exact HH sector basis

Fixing the fermion particle numbers $(N_\uparrow,N_\downarrow)$, the exact sector basis is
$$
\mathcal H_{\mathrm{sector}} = \operatorname{span}\{|f\rangle\otimes|\mathbf n_b\rangle : |f\rangle\in\mathcal H_{N_\uparrow,N_\downarrow},\; 0\le n_{b,i}\le n_{\mathrm{ph,max}}\}.
$$

### 7.3.1 Binary index map

For binary boson encoding, one may order basis states by the pair
$$
(f,\mathbf n_b),
$$
where $f$ is the fermionic bit pattern and $\mathbf n_b=(n_{b,0},\dots,n_{b,L-1})$ is the vector of local phonon occupations, each encoded in binary.

### 7.3.2 Unary index map

For unary boson encoding, the same sector is indexed by
$$
(f,\mathbf e_{n_{b,0}},\dots,\mathbf e_{n_{b,L-1}}),
$$
where $\mathbf e_n$ denotes the unary one-hot representation of occupation $n$.

## 7.4 Exact HH matrix elements

### 7.4.1 Diagonal elements

For basis state $|f\rangle\otimes|\mathbf n_b\rangle$, the diagonal contribution is
$$
U\sum_i n_{i\uparrow}(f)n_{i\downarrow}(f)
+
\sum_i v_i\,n_i(f)
+
\omega_0\sum_i \left(n_{b,i}+\frac{1}{2}\right).
$$

### 7.4.2 Hopping matrix elements

If $|f'\rangle$ differs from $|f\rangle$ by a single allowed nearest-neighbor hop with the same phonon occupations, then
$$
\langle f',\mathbf n_b|\hat H_t|f,\mathbf n_b\rangle
=
-t\,(-1)^{\chi(f;p,q)},
$$
where $\chi(f;p,q)$ is the Jordan-Wigner parity count between the two fermion modes $p$ and $q$.

### 7.4.3 Electron-phonon matrix elements

At fixed fermion configuration, the electron-phonon term changes the local phonon number by one. For site $i$,
$$
\langle n_{b,i}+1|\hat x_i|n_{b,i}\rangle = \sqrt{n_{b,i}+1},
\qquad
\langle n_{b,i}-1|\hat x_i|n_{b,i}\rangle = \sqrt{n_{b,i}}.
$$
Hence the off-diagonal electron-phonon matrix elements are proportional to
$$
g\,(n_i(f)-\bar n)\sqrt{n_{b,i}+1}
\quad\text{or}\quad
g\,(n_i(f)-\bar n)\sqrt{n_{b,i}},
$$
according to whether the phonon occupation increases or decreases.

# 8. Statevector Action, Expectation, and Exponential Primitives

## 8.1 Basis-state primitive

A statevector on $N_q$ qubits is written as
$$
|\psi\rangle = \sum_{k=0}^{2^{N_q}-1} a_k |k\rangle
= \sum_{b\in\{0,1\}^{N_q}} a_b |b\rangle.
$$

## 8.2 Explicit Pauli-word action

Let $P$ be a Pauli word. Then its action on a computational basis state can be written in the form
$$
P|b\rangle = \omega_P(b)\,|b\oplus \chi_P\rangle,
$$
where $\chi_P$ records which qubits are flipped by $X$ or $Y$, and $\omega_P(b)$ is the phase determined by the $Y$ and $Z$ letters acting on the bitstring $b$.

## 8.3 Expectation values

If
$$
\hat H = \sum_\alpha h_\alpha P_\alpha,
$$
then the expectation value is
$$
E(\psi)=\langle \psi|\hat H|\psi\rangle = \sum_\alpha h_\alpha \langle \psi|P_\alpha|\psi\rangle.
$$

## 8.4 Pauli rotations

For any Hermitian Pauli word with $P^2=I$,
$$
e^{-i\theta P} = \cos\theta\,I - i\sin\theta\,P.
$$
This identity is the primitive behind fast statevector application of Pauli exponentials.

## 8.5 Exponential of a Pauli polynomial

For a Pauli polynomial
$$
A = \sum_\alpha a_\alpha P_\alpha,
$$
one has
$$
U_A(\theta)=e^{-i\theta A}.
$$
If the $P_\alpha$ commute pairwise, then
$$
U_A(\theta)=\prod_\alpha e^{-i\theta a_\alpha P_\alpha}.
$$
If they do not commute, one may use either exact exponentiation on the working subspace or an ordered product formula.

## 8.6 Exact sector energy target

The natural exact comparison target in a fixed fermion sector is
$$
E_{\mathrm{exact,sector}} = \min_{\psi\in\mathcal H_{\mathrm{sector}}} \langle \psi|\hat H|\psi\rangle.
$$
This is distinct from the unrestricted full-Hilbert minimum when a sector constraint is imposed.

# 9. Fixed VQE Ansatz Families

This section records the main symbolic manifold families without tying them to any implementation names.

## 9.1 Hubbard layerwise ansatz

For the static Hubbard model, write
$$
\hat H_{\mathrm{Hub}} = \hat H_t + \hat H_U + \hat H_v.
$$
A layerwise ansatz uses one parameter per physical group per layer:
$$
U_{\mathrm{Hub}}^{(\ell)}
=
\exp(-i\theta_t^{(\ell)}\hat H_t)
\exp(-i\theta_U^{(\ell)}\hat H_U)
\exp(-i\theta_v^{(\ell)}\hat H_v),
$$
with the understanding that each grouped generator may itself be split into Pauli exponentials internally while sharing the same scalar parameter within the group.

## 9.2 Hubbard excitation-based surface

Relative to a Hartree-Fock occupied/virtual partition, single- and double-excitation generators take the form
$$
G_{i\to a}^{(\sigma)} = i\left(\hat c_{a\sigma}^\dagger \hat c_{i\sigma} - \hat c_{i\sigma}^\dagger \hat c_{a\sigma}\right),
$$
and
$$
G_{ij\to ab}^{(\sigma,\sigma')} = i\left(\hat c_{a\sigma}^\dagger \hat c_{b\sigma'}^\dagger \hat c_{j\sigma'}\hat c_{i\sigma} - \text{h.c.}\right).
$$
A layered excitation manifold is then
$$
U_{\mathrm{exc}}(\theta)=\prod_r e^{-i\theta_r G_r}.
$$

## 9.3 HH layerwise ansatz

For the HH model, define grouped generators
$$
G_t,\;G_U,\;G_v,\;G_{\mathrm{ph}},\;G_g,\;G_{\mathrm{drive}}.
$$
One layer is
$$
U_{\mathrm{HH,lw}}^{(\ell)}
=
\exp(-i\theta_{t,\ell}G_t)
\exp(-i\theta_{U,\ell}G_U)
\exp(-i\theta_{v,\ell}G_v)
\exp(-i\theta_{\mathrm{ph},\ell}G_{\mathrm{ph}})
\exp(-i\theta_{g,\ell}G_g)
\exp(-i\theta_{\mathrm{drive},\ell}G_{\mathrm{drive}}),
$$
with inactive groups omitted.

## 9.4 HH physical-termwise ansatz

A physical-termwise layer sits between the grouped and fully split extremes. Let
$$
\mathcal A_{\mathrm{phys}} = \{A_m\}
$$
be the collection of unsplit physical HH generators such as bond hopping terms, onsite interactions, local phonon energies, local couplings, and local drive terms. Then
$$
U_{\mathrm{HH,phys}}^{(\ell)} = \prod_{A_m\in\mathcal A_{\mathrm{phys}}} e^{-i\phi_{m,\ell}A_m}.
$$
Because the generators remain grouped at the physical-operator level, this fixed-family surface stays aligned with the same conserved-sector story as the HH layerwise surface.
In this symbolic HH manuscript, the canonical fixed-family surface is intentionally restricted to the sector-preserving HH layerwise and HH physical-termwise families.

## 9.5 Reference-state and ansatz pairing principle

The reference state and variational family should be paired by three criteria:

1. the conserved sector they preserve,
2. the expressive scale needed by the target Hamiltonian,
3. the conditioning and optimization burden induced by the chosen parameterization.

# 10. Polaronic Operator Families

## 10.1 Primitive polaronic ingredients

### 10.1.1 Shifted density

If the total electron number is $N_e$, define the mean density
$$
\bar n = \frac{N_e}{L},
$$
and the shifted density
$$
\tilde n_i = \hat n_i - \bar n I.
$$
At half filling $N_e=L$, this reduces to
$$
\tilde n_i = -\frac{1}{2}(Z_{p(i,\uparrow)}+Z_{p(i,\downarrow)}).
$$

### 10.1.2 Phonon primitives $P_i$ and $x_i$

Define
$$
P_i=i(\hat b_i^\dagger-\hat b_i),
\qquad
x_i=\hat b_i+\hat b_i^\dagger.
$$

### 10.1.3 Local squeeze primitive

A natural local squeeze generator is
$$
S_i = i\left((\hat b_i^\dagger)^2-\hat b_i^2\right).
$$

### 10.1.4 Doublon primitive

The doublon projector is
$$
D_i = \hat n_{i\uparrow}\hat n_{i\downarrow}.
$$

### 10.1.5 Even hopping channel

A bond-even hopping operator is
$$
T_{ij}^{(+)} = \sum_\sigma\left(\hat c_{i\sigma}^\dagger\hat c_{j\sigma}+\hat c_{j\sigma}^\dagger\hat c_{i\sigma}\right).
$$

### 10.1.6 Odd current channel

A bond-odd current operator is
$$
J_{ij}^{(-)} = i\sum_\sigma\left(\hat c_{i\sigma}^\dagger\hat c_{j\sigma}-\hat c_{j\sigma}^\dagger\hat c_{i\sigma}\right).
$$

## 10.2 Representative dressed channels

The exact family boundaries can vary, but the symbolic equivalents of the common channels can be written as follows.

### 10.2.1 Conditional displacement

$$
G_i^{(\mathrm{cd})} = \tilde n_i P_i.
$$

### 10.2.2 Doublon dressing

$$
G_i^{(\mathrm{dd})} = D_i P_i.
$$

### 10.2.3 Dressed hopping drag

$$
G_{ij}^{(\mathrm{hd})} = T_{ij}^{(+)}(P_i-P_j).
$$

### 10.2.4 Odd-channel drag

$$
G_{ij}^{(\mathrm{od})} = J_{ij}^{(-)}(x_i-x_j).
$$

### 10.2.5 Second-order even channel

$$
G_{ij}^{(2)} = T_{ij}^{(+)}(x_i-x_j)^2.
$$

### 10.2.6 Third-order odd current channel

$$
G_{ij}^{(3)} = J_{ij}^{(-)}(x_i-x_j)^3.
$$

### 10.2.7 Fourth-order even hopping channel

$$
G_{ij}^{(4)} = T_{ij}^{(+)}(x_i-x_j)^4.
$$

### 10.2.8 Bond-conditioned displacement and squeeze channels

With a bond-sensitive scalar observable $B_{ij}$, one may write
$$
G_{ij}^{(\mathrm{bd})}=B_{ij}P_i,
\qquad
G_{ij}^{(\mathrm{bs})}=B_{ij}S_i.
$$
A simple choice is $B_{ij}=\hat n_i-\hat n_j$.

### 10.2.9 Local squeeze channels

$$
G_i^{(\mathrm{sq})}=S_i,
\qquad
\tilde G_i^{(\mathrm{sq})}=\tilde n_i S_i.
$$

### 10.2.10 Extended cloud channels

A symbolic extended cloud family is
$$
G_i^{(\mathrm{cloud})}=\tilde n_i\sum_{r\in\mathcal N(i)} \alpha_{ir} P_r,
$$
with coefficients $\alpha_{ir}$ controlling cloud shape.

### 10.2.11 Doublon-translation and doublon-squeeze channels

$$
G_{ij}^{(\mathrm{DT})}=D_i T_{ij}^{(+)},
\qquad
G_i^{(\mathrm{DS})}=D_i S_i.
$$

## 10.3 Pool-family map

A useful abstract partition is
$$
\mathcal G = \mathcal G_{\mathrm{local}} \cup \mathcal G_{\mathrm{bond}} \cup \mathcal G_{\mathrm{squeeze}} \cup \mathcal G_{\mathrm{cloud}} \cup \mathcal G_{\mathrm{doublon}}.
$$
These families can be opened gradually rather than all at once.

### 10.3.1 Compact displacement/squeeze macro families

Two especially compact subfamilies are
$$
\mathcal G_{\mathrm{VLF}} = \{\tilde n_i P_i\}_i,
\qquad
\mathcal G_{\mathrm{SQ}} = \{S_i,\tilde n_i S_i\}_i.
$$
They act as coarse polaronic and squeeze envelopes.

# 11. Adaptive Selection and Staged Continuation

## 11.1 Cumulative phase construction for adaptive selection

Rather than treating Phase 1, Phase 2, and Phase 3 as three unrelated selector passes tried in reverse order, the current HH continuation logic is best read cumulatively. We write
$$
\mathcal G^{(3)} \subseteq \mathcal G^{(2)} \subseteq \mathcal G^{(1)} \subseteq \mathcal G,
$$
where $\mathcal G$ is the full expensive master pool, but the later phases reuse the earlier scoring and refinement machinery rather than replacing it with an unrelated fallback pass.

In that cumulative reading:

- **Phase 1** provides the broad cheap-screen surface over candidate-position records,
- **Phase 2** reuses the Phase-1 records and adds shortlist-and-full-rerank refinement,
- **Phase 3** reuses the Phase-1 and Phase-2 machinery and adds the highest-level augmentations used in current HH mainline.

Separately, the runtime controller may broaden the currently exposed candidate family through controller-stage transitions such as `seed/core $\to$ residual`; that controller-stage evolution should not be confused with the Phase-1/2/3 sophistication labels.

So the exposition below is written in the same constructive order **Phase 1 $\to$ Phase 2 $\to$ Phase 3** that the cumulative selector surface itself follows.

## 11.2 Phase 1 broad fallback selector surface

### 11.2.1 Core signal, append position, and refit window

Let $g\in\mathcal G^{(1)}$ be a candidate generator and $p$ a candidate insertion position. Let $W(p)$ denote the parameter window that is allowed to refit after insertion at $p$. The local post-refit drop is abstractly
$$
\delta E(g,p) = E(\theta) - \min_{\vartheta\in W(p)} E\bigl(\theta \oplus_p (g,\vartheta)\bigr).
$$

ie the old scaffold energy - energy-imporvement caused by reoptimizing generators  within a window inserted about position p

### 11.2.2 Gates, burdens, and feature ingredients

Each candidate-position pair is filtered by admissibility gates and assigned a burden
$$
B(g,p)=\lambda_{\mathrm{2q}} C_{\mathrm{2q}}(g,p)+\lambda_d D(g,p)+\lambda_\theta \Delta K(g,p),
$$
where $C_{\mathrm{2q}}$ is a two-qubit proxy cost, $D$ is a depth proxy, and $\Delta K$ is the marginal parameter count.

### 11.2.3 Cheap score

A Phase-1 broad score closes immediately to
$$
\begin{aligned}
S_1(g,p)
&=\frac{\underline{\delta E}(g,p)}{B(g,p)+\varepsilon}\\
&=\frac{\underline{\delta E}(g,p)}{\lambda_{\mathrm{2q}}C_{\mathrm{2q}}(g,p)+\lambda_dD(g,p)+\lambda_\theta\Delta K(g,p)+\varepsilon},
\end{aligned}
$$
where $\underline{\delta E}$ is a lower-bound or surrogate estimate of useful post-refit energy drop.

This broad score is intentionally early-stage: it privileges immediate drop estimates when the controller still has long useful runway. Later phases keep the candidate-position language but no longer value raw $\underline{\delta E}$ in the same dominant way; the later cheap/full surfaces progressively downweight immediate-drop dominance as the useful horizon contracts.

### 11.2.4 Position probing, trough detection, and effective selector

Rather than selecting only by generator identity, one probes positions $p$ and computes
$$
(g_1,p_1)=\arg\max_{(g,p)} S_1(g,p).
$$
If the best score remains below threshold across all positions, that can be treated as a plateau or trough signal.

### 11.2.5 Full-pool limiting case

If one removes the structured shortlist machinery and keeps only the final full-pool backstop, the limiting Phase-1 rule becomes: choose the admissible generator with largest gradient magnitude,
$$
g_* = \arg\max_{g\in\mathcal G^{(1)}_{\mathrm{admissible}}} |\partial_\epsilon E(\epsilon;g)|_{\epsilon=0}|,
$$
subject to symmetry and cost gates.

## 11.3 Phase 2 relaxed selector surface

Phase 2 should be read as an extension of the Phase-1 cheap screen: it keeps the same candidate-position language, but adds shortlist formation and a richer reranking model on top of the earlier score. More precisely, Phase 1 does not hand Phase 2 only the single best record from §11.2.4; it hands forward a retained family of candidate-position records, and the explicit retained set is the shortlist $\mathcal S_2$ defined below by threshold or Top-$K$ on $S_1$.

### 11.3.1 Core signal, append position, refit window, and cheap screen

The Phase-2 selector works over candidate-position pairs with
$$
g\in\mathcal G^{(2)},
\qquad
p\in\mathcal P_g,
$$
and the cheap screening surface closes as
$$
\begin{aligned}
\delta E(g,p)
&=E(\theta)-\min_{\vartheta\in W(p)}E\bigl(\theta\oplus_p(g,\vartheta)\bigr),\\
B(g,p)
&=\lambda_{\mathrm{2q}}C_{\mathrm{2q}}(g,p)+\lambda_dD(g,p)+\lambda_\theta\Delta K(g,p),\\
S_1(g,p)
&=\frac{\underline{\delta E}(g,p)}{B(g,p)+\varepsilon}
=\frac{\underline{\delta E}(g,p)}{\lambda_{\mathrm{2q}}C_{\mathrm{2q}}(g,p)+\lambda_dD(g,p)+\lambda_\theta\Delta K(g,p)+\varepsilon}.
\end{aligned}
$$

### 11.3.2 Shortlist rule

Define the shortlist by threshold or cap:
$$
\begin{aligned}
\mathcal S_2
&=\{(g,p):S_1(g,p)\ge\tau_1\}\quad\text{or}\quad \operatorname{Top}_K\{(g,p)\text{ by }S_1(g,p)\}\\
&=\left\{(g,p):\underline{\delta E}(g,p)\ge\tau_1\bigl[\lambda_{\mathrm{2q}}C_{\mathrm{2q}}(g,p)+\lambda_dD(g,p)+\lambda_\theta\Delta K(g,p)+\varepsilon\bigr]\right\}\\
&\qquad\text{or}\quad \operatorname{Top}_K\left\{(g,p)\text{ by }\frac{\underline{\delta E}(g,p)}{\lambda_{\mathrm{2q}}C_{\mathrm{2q}}(g,p)+\lambda_dD(g,p)+\lambda_\theta\Delta K(g,p)+\varepsilon}\right\}.
\end{aligned}
$$
If $\pi_g(g,p)=g$ denotes projection onto generator identity, then the generators actually carried forward by the shortlist are $\pi_g(\mathcal S_2)$. We therefore do not write $\mathcal G^{(2)}=\mathcal S_2$: $\mathcal G^{(2)}$ lives in generator space, whereas $\mathcal S_2$ is a state-dependent shortlist of candidate-position records.

### 11.3.3 Reduced-path geometry, novelty, curvature, and trust-region drop

For a shortlisted record $(g,p)$, let

$$
|\psi\rangle = U_{>p}\,U_{\le p}\,|\psi_0\rangle,
\qquad
|\psi_{(m,p)}(\alpha)\rangle = U_{>p}\,e^{-i\alpha A_m}\,U_{\le p}\,|\psi_0\rangle,
$$

$$
\widetilde A_{(m,p)} := U_{>p}\,A_m\,U_{>p}^{\dagger},
\qquad
U_{>p}:=U_K(\theta_K)\cdots U_{p+1}(\theta_{p+1}),
$$

$$
|\psi_{(m,p)}(\alpha)\rangle = e^{-i\alpha \widetilde A_{(m,p)}}|\psi\rangle,
\qquad
\left.\partial_{\alpha}|\psi_{(m,p)}(\alpha)\rangle\right|_{\alpha=0}
=
-i\,\widetilde A_{(m,p)}|\psi\rangle.
$$

$$
\begin{aligned}
U_{g,p}(\alpha)&=e^{-i\alpha \widetilde A_{g,p}},&
|\psi_{g,p}(\alpha)\rangle&=U_{g,p}(\alpha)|\psi\rangle,&
Q_\psi&=I-|\psi\rangle\langle\psi|,\\
|d_{g,p}\rangle&=\left.\partial_\alpha |\psi_{g,p}(\alpha)\rangle\right|_{\alpha=0}=-i\widetilde A_{g,p}|\psi\rangle,&
|d^{(2)}_{g,p}\rangle&=\left.\partial_\alpha^2 |\psi_{g,p}(\alpha)\rangle\right|_{\alpha=0}=-\widetilde A_{g,p}^2|\psi\rangle,\\
\tau_{g,p}&=Q_\psi |d_{g,p}\rangle=Q_\psi\left.\partial_\alpha |\psi_{g,p}(\alpha)\rangle\right|_{\alpha=0}=-iQ_\psi\widetilde A_{g,p}|\psi\rangle,\\
g(g,p)&=\left.\partial_\alpha\langle \psi_{g,p}(\alpha)|H|\psi_{g,p}(\alpha)\rangle\right|_{\alpha=0}=2\Re\langle \psi|H|d_{g,p}\rangle,\\
g_{\mathrm{lcb}}(g,p)&=\max\{|g(g,p)|-z_\alpha\sigma(g,p),0\}.
\end{aligned}
$$
If $\{\tau_j\}_{j\in W(p)}$ are the current horizontal tangents in the inherited window, then
$$
\begin{aligned}
F(g,p)&=\|\tau_{g,p}\|^2=\langle \tau_{g,p},\tau_{g,p}\rangle=\langle d_{g,p}|Q_\psi|d_{g,p}\rangle,\\
\mathcal N(g,p)&=1-\max_{j\in W(p)} \frac{|\langle \tau_{g,p},\tau_j\rangle|}{\|\tau_{g,p}\|\,\|\tau_j\|}
=1-\max_{j\in W(p)} \frac{|\langle d_{g,p}|Q_\psi|\partial_{\theta_j}\psi\rangle|}{\sqrt{\langle d_{g,p}|Q_\psi|d_{g,p}\rangle}\,\sqrt{\langle \partial_{\theta_j}\psi|Q_\psi|\partial_{\theta_j}\psi\rangle}},\\
\kappa(g,p)&=h_{g,p}=\left.\partial_\alpha^2\langle \psi_{g,p}(\alpha)|H|\psi_{g,p}(\alpha)\rangle\right|_{\alpha=0}
=2\Re\!\left(\langle d_{g,p}|H|d_{g,p}\rangle + \langle \psi|H|d^{(2)}_{g,p}\rangle\right),\\
\delta E_{\mathrm{TR}}(g,p)&=\max_{|\alpha|\le \rho/\sqrt{F(g,p)}}\left[g_{\mathrm{lcb}}(g,p)|\alpha|-\frac12\max\bigl(h_{g,p},0\bigr)\alpha^2\right],\\
\delta E_{\mathrm{quad}}(g,p)&=\frac{g_{\mathrm{lcb}}(g,p)^2}{2h_{g,p}}\qquad\text{when }h_{g,p}>0\text{ and the unconstrained minimizer lies inside the trust region.}
\end{aligned}
$$

### 11.3.4 Full rerank score and effective selector

The Phase-2 rerank score is not just symbolically
$$
S_2(g,p)=\frac{\delta E_{\mathrm{TR}}(g,p)\,\mathcal N(g,p)}{B(g,p)(1+\kappa(g,p)) + \varepsilon},
$$
but, after substitution of the local burden, miss, curvature, and tangent geometry,
$$
\begin{aligned}
S_2(g,p)
&=\frac{\delta E_{\mathrm{TR}}(g,p)\,\mathcal N(g,p)}{\bigl[\lambda_{\mathrm{2q}}C_{\mathrm{2q}}(g,p)+\lambda_dD(g,p)+\lambda_\theta\Delta K(g,p)\bigr]\bigl(1+h_{g,p}\bigr)+\varepsilon}\\
&=\frac{\displaystyle \max_{|\alpha|\le \rho/\sqrt{\langle d_{g,p}|Q_\psi|d_{g,p}\rangle}}\left[\max\{|2\Re\langle \psi|H|d_{g,p}\rangle|-z_\alpha\sigma(g,p),0\}|\alpha|-\frac12\max\!\Bigl(2\Re\!\bigl(\langle d_{g,p}|H|d_{g,p}\rangle+\langle \psi|H|d^{(2)}_{g,p}\rangle\bigr),0\Bigr)\alpha^2\right]}{\bigl[\lambda_{\mathrm{2q}}C_{\mathrm{2q}}(g,p)+\lambda_dD(g,p)+\lambda_\theta\Delta K(g,p)\bigr]\Bigl(1+2\Re\!\bigl(\langle d_{g,p}|H|d_{g,p}\rangle+\langle \psi|H|d^{(2)}_{g,p}\rangle\bigr)\Bigr)+\varepsilon}\\
&\qquad\times\left(1-\max_{j\in W(p)} \frac{|\langle d_{g,p}|Q_\psi|\partial_{\theta_j}\psi\rangle|}{\sqrt{\langle d_{g,p}|Q_\psi|d_{g,p}\rangle}\,\sqrt{\langle \partial_{\theta_j}\psi|Q_\psi|\partial_{\theta_j}\psi\rangle}}\right).
\end{aligned}
$$
The selected candidate is then
$$
(g_2,p_2)=\arg\max_{(g,p)\in\mathcal S_2} S_2(g,p).
$$

## 11.4 Phase 3 primary selector surface

This section describes the full symbolic Phase-3 selector surface: position-aware candidate records, inherited refit windows, split-aware reranking, and post-admission local simplification. In current HH mainline, Phase 3 is not a separate reverse-order fallback pass; it reuses the earlier cheap-screen and shortlist/full-rerank machinery, then adds the highest-level augmentations used in runtime policy.

The formulas in §§11.4.1--11.4.4 are the corrected canonical Phase-3 target contract. Current repo code may still lag some details of this contract; that implementation lag is stated explicitly later in §19.4 rather than folded back into the symbolic definitions here.

### 11.4.1 Core signal, append position, refit window, and cheap screen

The cumulative Phase-3 selector is most naturally written over candidate-position pairs
$$
r=(m,p),
\qquad
m\in\mathcal G^{(3)}_{\mathrm{avail}},
\quad
p\in\mathcal P_m.
$$
Here $m$ is a candidate generator, $p$ is a candidate insertion position, and $\mathcal P_m$ is the set of admissible positions for $m$.

The inherited refit window attached to position $p$ is denoted $W(p)$. A useful decomposition is
$$
W(p)=W_{\mathrm{new}}(p)\cup W_{|\theta|}(p),
$$
where $W_{\mathrm{new}}$ keeps the newest coordinates and $W_{|\theta|}$ optionally preserves a small set of older large-amplitude coordinates.

The raw candidate-position universe entering the cumulative Phase-3 selector is
$$
\mathcal P_{\mathrm{avail}}^{(3)}
=
\left\{
r=(m,p):
m\in\mathcal G^{(3)}_{\mathrm{avail}},
\ p\in\mathcal P_m
\right\}.
$$

Let $t$ denote the current selector step, let $D_{\mathrm{left}}(t)$ be the remaining controller depth budget, let $\widehat N_{\mathrm{rem}}(t)$ be a heuristic forecast of useful remaining admissions, and define the effective useful horizon
$$
H_t=\min\!\bigl(D_{\mathrm{left}}(t),\widehat N_{\mathrm{rem}}(t)\bigr).
$$
One useful controller-level estimator writes
$$
S_t(k)=\prod_{j=1}^{k}\bigl(1-h_t(j)\bigr),
\qquad
\widehat N_{\mathrm{rem}}(t)=\sum_{k=1}^{D_{\mathrm{left}}(t)} S_t(k)\,Q_t(k),
$$
where $S_t(k)$ is a survival factor for still having a useful admission opportunity $k$ selector steps ahead, $Q_t(k)$ is the conditional usefulness of that future opportunity, and $h_t(j)$ is the corresponding controller hazard. A compact controller-level realization is
$$
u_{\mathrm{pat}}(t)=\operatorname{clip}\!\left(\max\!\left(\frac{\mathrm{plateau\_streak}(t)}{P},\frac{\mathrm{low\_streak}(t)}{L}\right),0,1\right),
$$
$$
u_{\mathrm{front}}(t)=
\left(1-\frac{s_1^{\mathrm{cheap}}(t)-s_2^{\mathrm{cheap}}(t)}{s_1^{\mathrm{cheap}}(t)+\varepsilon}\right)
\left(1+\frac{s_1^{\mathrm{cheap}}(t)}{c_s\,s_{\mathrm{ref}}^{\mathrm{cheap}}(t)+\varepsilon}\right)^{-1},
$$
$$
h_t(j)=\sigma\!\Bigl(\beta_0+\beta_{\mathrm{pat}}\nu_{\mathrm{pat}}(t)+\beta_{\mathrm{front}}\nu_{\mathrm{front}}(t)+\eta_h(j-1)\Bigr),
$$
$$
m_t=\operatorname{EWMA}_{s\in W_t^+}\!\bigl(\log \Delta_{\mathrm{adm}}(s)\bigr),
\qquad
\rho_t=\operatorname{clip}\!\left(\frac{\operatorname{EWMA}_{\mathrm{recent}}(\Delta_{\mathrm{adm}})}{\operatorname{EWMA}_{\mathrm{older}}(\Delta_{\mathrm{adm}})+\varepsilon},\rho_{\min},1\right),
$$
$$
\gamma_t=[-\log \rho_t]_+,
\qquad
s_t=\max\!\Bigl(s_{\min},\operatorname{MAD}_{s\in W_t^+}\!\bigl(\log \Delta_{\mathrm{adm}}(s)\bigr)\Bigr),
$$
$$
Q_t(k)=\Phi\!\left(\frac{m_t-\gamma_t(k-1)-\log \tau_{\mathrm{use},t}}{s_t}\right).
$$
If optimistic and pessimistic runway envelopes
$$
\widehat N_{\mathrm{rem}}^{\mathrm{lo}}(t)\le \widehat N_{\mathrm{rem}}(t)\le \widehat N_{\mathrm{rem}}^{\mathrm{hi}}(t)
$$
are also tracked, one convenient band construction is
$$
\operatorname{logit} h_t^{\mathrm{pess}}(j)=\operatorname{logit} h_t(j)+\Delta_h,
\qquad
\operatorname{logit} h_t^{\mathrm{opt}}(j)=\operatorname{logit} h_t(j)-\Delta_h,
$$
$$
m_t^{\mathrm{pess}}=m_t-\Delta_m,\quad s_t^{\mathrm{pess}}=s_t+\Delta_s,
\qquad
m_t^{\mathrm{opt}}=m_t+\Delta_m,\quad s_t^{\mathrm{opt}}=\max(s_{\min},s_t-\Delta_s),
$$
$$
\widehat N_{\mathrm{rem}}^{\mathrm{lo}}(t)=\sum_{k=1}^{D_{\mathrm{left}}(t)} S_t^{\mathrm{pess}}(k)\,Q_t^{\mathrm{pess}}(k),
\qquad
\widehat N_{\mathrm{rem}}^{\mathrm{hi}}(t)=\sum_{k=1}^{D_{\mathrm{left}}(t)} S_t^{\mathrm{opt}}(k)\,Q_t^{\mathrm{opt}}(k),
$$
and then a compact confidence and saturation summary is
$$
C_t=\frac{\widehat N_{\mathrm{rem}}^{\mathrm{lo}}(t)}{\widehat N_{\mathrm{rem}}^{\mathrm{hi}}(t)+\varepsilon},
\qquad
W_{\mathrm{sat}}(t)=1-\frac{\widehat N_{\mathrm{rem}}(t)}{D_{\mathrm{left}}(t)+\varepsilon}.
$$
The corresponding selector mode and saturation regime may be written as
$$
\pi_t=
\begin{cases}
\mathrm{fast},&\widehat N_{\mathrm{rem}}^{\mathrm{lo}}(t)>n_{\mathrm{fast}},\\
\mathrm{high\mbox{-}fidelity},&\widehat N_{\mathrm{rem}}^{\mathrm{hi}}(t)\le n_{\mathrm{slow}},\\
\mathrm{mixed},&\text{otherwise},
\end{cases}
$$
$$
\mathrm{regime}_{\mathrm{sat}}(t)=
\begin{cases}
\mathrm{hungry},&W_{\mathrm{sat}}(t)<\tau_{\mathrm{off}},\\
\mathrm{plateau},&W_{\mathrm{sat}}(t)>\tau_{\mathrm{on}},\\
\mathrm{watch},&\text{otherwise}.
\end{cases}
$$
The quantity $\widehat N_{\mathrm{rem}}(t)$ is controller telemetry rather than a physical observable: it summarizes recent selector progress and remaining useful runway, but it is not itself a hard admissibility certificate.

For $r=(m,p)$, let
$$
\begin{aligned}
U_r(\alpha)&=e^{-i\alpha \widetilde A_r},\\
|\psi_r(\alpha)\rangle&=U_r(\alpha)|\psi\rangle,\\
Q_\psi&=I-|\psi\rangle\langle\psi|,\\
|d_r\rangle&=\left.\partial_\alpha|\psi_r(\alpha)\rangle\right|_{\alpha=0}=-i\widetilde A_r|\psi\rangle,\\
t_r&=Q_\psi|d_r\rangle,\\
F_{\mathrm{raw}}(r)&=\|t_r\|^2=\langle d_r|Q_\psi|d_r\rangle,\\
g_r&=\left.\partial_\alpha\langle \psi_r(\alpha)|H|\psi_r(\alpha)\rangle\right|_{\alpha=0}=2\Re\langle \psi|H|d_r\rangle,\\
g_{\mathrm{lcb}}(r)&=\max\{|g_r|-z_\alpha\sigma_r,0\},\\
\Gamma_{\mathrm{stage}}(r)&:=\Gamma_{\mathrm{stage}}(m),\\
\Gamma_{\mathrm{sym}}(r)&:=\Gamma_{\mathrm{sym}}(m,p),\\
K_{\mathrm{cheap}}(r;t)
&=1+w_R\bar R_{\mathrm{rem}}(r;t)+w_D\bar D_{\mathrm{life}}(r)\\
&\quad +w_G\bar G_{\mathrm{new}}(r)+w_C\bar C_{\mathrm{new}}(r)+w_c\bar c(r).
\end{aligned}
$$
with
$$
p_{\mathrm{app}}:=|\mathcal O|
$$
for the current append slot, and
$$
\bar D_{\mathrm{life}}(r)=\frac{D_{\mathrm{raw}}(r)}{\mathrm{depth}_{\mathrm{ref}}},
\qquad
D_{\mathrm{raw}}(r)
=
\sum_{\ell\in\mathcal L_r}
\left[
2\max(\mathrm{wt}(\ell)-1,0)
+
\frac12\bigl(2\,\#_{X,Y}(\ell)+1\bigr)
\right]
+
|p_{\mathrm{app}}-p|,
$$
$$
\bar G_{\mathrm{new}}(r)=\frac{G_{\mathrm{new}}(r)}{\mathrm{group}_{\mathrm{ref}}},
\qquad
\bar C_{\mathrm{new}}(r)=\frac{C_{\mathrm{new}}(r)}{\mathrm{shot}_{\mathrm{ref}}},
\qquad
\bar c(r)=\frac{c(r)}{\mathrm{reuse}_{\mathrm{ref}}},
$$
$$
C_{\mathrm{new}}(r)=G_{\mathrm{new}}(r)\,n_{\mathrm{shots/group}},
\qquad
c(r)=\mathrm{groups}_{\mathrm{new}}(r),
\qquad
\bar P_{\mathrm{opt}}(r)=\frac{|W(p)|}{\mathrm{optdim}_{\mathrm{ref}}},
$$
$$
\bar R_{\mathrm{rem}}(r;t)
=
H_t\Bigl(
\bar D_{\mathrm{life}}(r)+\bar G_{\mathrm{new}}(r)+\bar C_{\mathrm{new}}(r)+\bar c(r)+\bar P_{\mathrm{opt}}(r)
\Bigr)\mathbf 1_{\mathrm{lifetime}}.
$$
Here $\bar D_{\mathrm{life}}$ is circuit/compile burden at the candidate record, $\bar G_{\mathrm{new}}$ and $\bar C_{\mathrm{new}}$ are measurement-group and shot burden of the new term, $\bar c$ is the reuse penalty, and $\bar R_{\mathrm{rem}}$ is the horizon-weighted lifetime multiplier applied only when lifetime-cost mode is on. The offset $|p_{\mathrm{app}}-p|$ is a local insertion-displacement penalty; it measures how far the proposal sits from the append slot and should not be conflated with expected remaining depth.

To interpolate between a wide early-stage gradient screen and a later-stage cost-aware screen, define
$$
\lambda_t=\frac{H_t}{H_t+c_\lambda},
\qquad c_\lambda>0,
$$
$$
K_{\mathrm{coarse}}(t)=K_{\min}+\left\lfloor (K_{\max}-K_{\min})\frac{\widehat N_{\mathrm{rem}}^{\mathrm{hi}}(t)}{D_{\mathrm{left}}(t)+\varepsilon}\right\rfloor,
$$
and
$$
\begin{aligned}
S_{3,\nabla}(r)
&=
\Gamma_{\mathrm{stage}}(r)\Gamma_{\mathrm{sym}}(r)\,
\frac{g_{\mathrm{lcb}}(r)^2}{2\lambda_FF_{\mathrm{raw}}(r)},\\
S_{3,\mathrm{late}}(r;t)
&=
\Gamma_{\mathrm{stage}}(r)\Gamma_{\mathrm{sym}}(r)\,
\frac{g_{\mathrm{lcb}}(r)^2}{2\lambda_FF_{\mathrm{raw}}(r)}
\frac{1}{K_{\mathrm{cheap}}(r;t)+\varepsilon},\\
S_{3,\mathrm{cheap}}(r;t)
&=
\lambda_t\,S_{3,\nabla}(r)
+
\bigl(1-\lambda_t\bigr)S_{3,\mathrm{late}}(r;t).
\end{aligned}
$$
Then the active coarse-shortlist size is
$$
N_{\mathrm{cheap}}(t)
=
\min\!\left\{
\left|\mathcal P_{\mathrm{avail}}^{(3)}\right|,
N_{\mathrm{cheap}}^{\max},
K_{\mathrm{coarse}}(t)
\right\},
$$
and the Phase-3 cheap universe is the full-pool cheap-score shortlist
$$
\mathcal C_{\mathrm{cheap}}
=
\operatorname{Top}_{N_{\mathrm{cheap}}(t)}\!\left(
\mathcal P_{\mathrm{avail}}^{(3)};
S_{3,\mathrm{cheap}}(\cdot;t)
\right).
$$
So the cumulative Phase-3 cheap screen is applied over the full admissible candidate-position pool, not behind a separate upstream $g_{\mathrm{lcb}}$-only cap. The gates $\Gamma_{\mathrm{stage}}$ and $\Gamma_{\mathrm{sym}}$ determine admissibility, while $S_{3,\mathrm{cheap}}$ ranks only within that admissible pool. The horizon $H_t$ smooths cheap-stage width and burden, but it is not itself a new hard gate and it is not a theorem-level invariant of the Hamiltonian.

The term $S_{3,\nabla}(r)$ is the direct continuation of the early drop-dominant surface $S_1$. It matters most when useful runway is still long. The late-stage factor $S_{3,\mathrm{late}}(r;t)$ matters more once that runway contracts.

From first principles, $K_{\mathrm{cheap}}(r;t)$ is a dimensionless burden functional: the numerator of $S_{3,\mathrm{cheap}}(r;t)$ estimates local utility, and $K_{\mathrm{cheap}}(r;t)$ discounts that utility by how expensive it is to admit $r$ now and, through $\bar R_{\mathrm{rem}}(r;t)$, how expensive that decision is expected to remain over the useful horizon of the current run. The $2\max(\mathrm{wt}(\ell)-1,0)$ piece is a two-qubit-style burden proxy, the $\frac12(2\,\#_{X,Y}(\ell)+1)$ piece is a basis-change / one-qubit rotation proxy, and $|p_{\mathrm{app}}-p|$ penalizes inserting far from the current append location.

### 11.4.2 Shortlist, reduced-path geometry, novelty, full rerank score, and split-aware augmentation

The shortlist size is
$$
N_{\mathrm{short}}=\min\left\{|\mathcal C_{\mathrm{cheap}}|,N_{\max},\left\lceil f_{\mathrm{short}}|\mathcal C_{\mathrm{cheap}}|\right\rceil\right\},
\qquad
\mathcal S_3\subseteq\mathcal C_{\mathrm{cheap}}.
$$
So the flow is
$$
\mathcal P_{\mathrm{avail}}^{(3)}
\longrightarrow
\mathcal C_{\mathrm{cheap}}
\longrightarrow
\mathcal S_3
\longrightarrow
\arg\max_{r\in\mathcal S_3} S_{3,\mathrm{base}}(r)
\ \text{or}\
\arg\max_{r\in\mathcal S_3} S_{3,\mathrm{aug}}(r),
$$
where the first arrow is the horizon-smoothed cheap ranking over the full admissible pool, the middle arrow forms the internal shortlist, and the later arrows perform the richer reranking and Phase-3 augmentation built on top of the earlier surfaces. The horizon forecast enters here only as a controller-side burden/width modulator; it does not turn shortlist membership into a theorem of global optimality.

For each shortlisted record $r=(m,p)$, let $W_r=W(m,p)$ be the inherited window that would actually be refit if $m$ were admitted at position $p$, and let $\{t_j\}_{j\in W_r}$ be the current horizontal tangents in that inherited window. Then
$$
\begin{aligned}
F_{\mathrm{raw}}(r)&=\langle t_r,t_r\rangle,\\
(Q_r)_{jk}&=\Re\langle t_j,t_k\rangle,\\
(q_r)_j&=\Re\langle t_j,t_r\rangle,\\
E(\alpha,\delta\theta_{W_r};r)
&\approx E_0+g_r\alpha+\tfrac12 h_r\alpha^2\\
&\quad +\alpha b_r^\top\delta\theta_{W_r}+\tfrac12\delta\theta_{W_r}^\top H_{W_r}\delta\theta_{W_r},\\
M_r&=\tfrac12(H_{W_r}+H_{W_r}^{\top})+\lambda_H I,\\
\widetilde h_r&=h_r-b_r^\top M_r^{-1}b_r,\\
F_r^{\mathrm{red}}&=F_{\mathrm{raw}}(r)-2q_r^\top M_r^{-1}b_r+b_r^\top M_r^{-1}Q_rM_r^{-1}b_r,\\
q_r^{\mathrm{red}}&=q_r-Q_rM_r^{-1}b_r,\\
\nu_r&=\operatorname{clip}_{[0,1]}\!\left(1-\frac{(q_r^{\mathrm{red}})^\top(Q_r+\varepsilon_{\mathrm{nov}}I)^{-1}q_r^{\mathrm{red}}}{F_r^{\mathrm{red}}}\right),\\
\Delta E_{\mathrm{TR}}(r)
&=\max_{|\alpha|\le \rho/\sqrt{F_r^{\mathrm{red}}}}
\left[g_{\mathrm{lcb}}(r)|\alpha|-\tfrac12\max(\widetilde h_r,0)\alpha^2\right],\\
K_{\mathrm{full}}(r;t)
&=1+w_R\bar R_{\mathrm{rem}}(r;t)+w_D\bar D_{\mathrm{life}}(r)\\
&\quad +w_G\bar G_{\mathrm{new}}(r)+w_C\bar C_{\mathrm{new}}(r)+w_c\bar c(r).
\end{aligned}
$$
So the base rerank score closes to
$$
\begin{aligned}
S_{3,\mathrm{base}}(r;t)
&=\Gamma_{\mathrm{stage}}(m)\Gamma_{\mathrm{sym}}(r)\,\nu_r^{\gamma_N}\frac{\Delta E_{\mathrm{TR}}(r)}{K_{\mathrm{full}}(r;t)+\varepsilon}\\
&=\Gamma_{\mathrm{stage}}(m)\Gamma_{\mathrm{sym}}(r)
\left[\operatorname{clip}_{[0,1]}\!\left(1-\frac{(q_r^{\mathrm{red}})^\top(Q_r+\varepsilon_{\mathrm{nov}}I)^{-1}q_r^{\mathrm{red}}}{F_r^{\mathrm{red}}}\right)\right]^{\gamma_N}
\frac{\Delta E_{\mathrm{TR}}(r)}{K_{\mathrm{full}}(r;t)+\varepsilon}.
\end{aligned}
$$
The augmented selector score is therefore
$$
\begin{aligned}
S_{3,\mathrm{aug}}(r;t)
&=S_{3,\mathrm{base}}(r;t)+\beta_{\mathrm{split}}\Sigma(r)+\beta_{\mathrm{motif}}M(r)+\beta_{\mathrm{sym}}Y(r)-\beta_{\mathrm{dup}}D_{\mathrm{dup}}(r)\\
&=\Gamma_{\mathrm{stage}}(m)\Gamma_{\mathrm{sym}}(r)
\left[\operatorname{clip}_{[0,1]}\!\left(1-\frac{(q_r^{\mathrm{red}})^\top(Q_r+\varepsilon_{\mathrm{nov}}I)^{-1}q_r^{\mathrm{red}}}{F_r^{\mathrm{red}}}\right)\right]^{\gamma_N}
\frac{\Delta E_{\mathrm{TR}}(r)}{K_{\mathrm{full}}(r;t)+\varepsilon}\\
&\qquad+\beta_{\mathrm{split}}\Sigma(r)+\beta_{\mathrm{motif}}M(r)+\beta_{\mathrm{sym}}Y(r)-\beta_{\mathrm{dup}}D_{\mathrm{dup}}(r).
\end{aligned}
$$
Here $\Sigma$ measures the gain available from selective macro-splitting, $M$ measures motif compatibility or transferable local pattern alignment, $Y$ measures symmetry quality or mitigation value, and $D_{\mathrm{dup}}$ penalizes near-duplicate continuation directions.^[These four addends are intentionally higher-level selector covariates. They should be read as implementation-facing heuristic modifiers layered on top of the primitive geometric score $S_{3,\mathrm{base}}$, not as unique primitive invariants with a single repo-independent closed form.] If a shortlisted macro generator $m$ admits a split family
$$
\mathcal C_{\mathrm{split}}(m)=\{m_1,\dots,m_{K_m}\},
$$
then each split child inherits the same position $p$ and therefore defines a child record $r_j=(m_j,p)$, with replacement rule
$$
r\leadsto r_j^*\quad\text{if}\quad \max_j S_{3,\mathrm{aug}}(r_j) > S_{3,\mathrm{aug}}(r)+\tau_{\mathrm{split}}.
$$

$$
\mathcal S_{3}=\{r_{1},\dots,r_{N}\},
\qquad
s(r)=S_{3,\mathrm{aug}}(r)\ \text{or}\ S_{3,\mathrm{base}}(r),
$$
and let the shortlist be sorted so that
$$
s(r_{1})\ge s(r_{2})\ge \cdots \ge s(r_{N}).
$$
A batch selector does not choose a single winner; it chooses a set
$$
\mathcal B_{\mathrm{batch}}\subseteq \mathcal S_{3},
\qquad
1\le |\mathcal B_{\mathrm{batch}}|\le B_{\mathrm{cap}},
$$
by starting from the top record and then greedily adding near-degenerate compatible records.

$$
r\in\mathcal B_{\mathrm{batch}}
\quad\Longrightarrow\quad
s(r)\ge \eta_{\mathrm{nd}}\,s(r_{\max}),
$$
where $r_{\max}$ is the top shortlisted record and $0<\eta_{\mathrm{nd}}\le 1$ is the near-degeneracy ratio. In other words, only candidates whose score is sufficiently close to the best score are even allowed to compete for the same batch.

Then define a baseline pairwise incompatibility penalty
$$
\Pi_0(r,r')
=
w_{\mathrm{ov}}\,\Omega(r,r')
+
w_{\mathrm{comm}}\,\Xi(r,r')
+
w_{\mathrm{curv}}\,\mathcal K(r,r')
+
w_{\mathrm{sched}}\,\mathcal A(r,r')
+
w_{\mathrm{meas}}\,\mathcal M(r,r'),
$$
with
$$
\Omega(r,r')=\frac{|\operatorname{supp}(r)\cap \operatorname{supp}(r')|}{|\operatorname{supp}(r)\cup \operatorname{supp}(r')|},
\qquad
\Xi(r,r')=
\begin{cases}
0,&\text{if the candidate generators commute},\\
1,&\text{otherwise},
\end{cases}
$$
$$
\mathcal K(r,r')=\text{cross-curvature / tangent-overlap proxy},
\qquad
\mathcal A(r,r')=\frac{|W_r\cap W_{r'}|}{|W_r\cup W_{r'}|},
\qquad
\mathcal M(r,r')=1-\text{measurement-overlap}(r,r').
$$
^[Batch note: $\mathcal K$ and $\mathcal M$ are proxy penalties rather than primitive-closed observables. On paper they are best interpreted as controller-supplied overlap costs used to discourage mutually awkward co-admissions.]

The greedy admission rule is then
$$
r\ \text{is appended to}\ \mathcal B_{\mathrm{batch}}
\quad\Longrightarrow\quad
s(r)-\sum_{r'\in\mathcal B_{\mathrm{batch}}}\Pi_t(r,r')>0,
$$
together with
$$
|\mathcal B_{\mathrm{batch}}|<B_{\mathrm{target}}(t)
\quad\text{or at worst}\quad
|\mathcal B_{\mathrm{batch}}|<B_{\mathrm{cap}}.
$$
Here the batch target should itself depend on useful remaining runway. A canonical controller-level rule is
$$
B_{\mathrm{target}}(t)
=
B_{\min}
+
\left\lfloor
(B_{\max}-B_{\min})
\frac{\widehat N_{\mathrm{rem}}(t)}{D_{\mathrm{left}}(t)+\varepsilon}\,
C_t
\right\rfloor,
$$
where $B_{\min}$ and $B_{\max}$ are controller lower/upper batch-width bounds. So batch growth is largest only when useful runway is still long and the runway estimate is confident. The scheduler-scaled compatibility penalty is
$$
\Pi_t(r,r')=\gamma_{\mathrm{compat}}(t)\,\Pi_0(r,r'),
\qquad
\gamma_{\mathrm{compat}}(t)=\gamma_{\min}+(\gamma_{\max}-\gamma_{\min})W_{\mathrm{sat}}(t).
$$
Here $\gamma_{\min}$ and $\gamma_{\max}$ are controller lower/upper compatibility-pressure bounds.
Thus both the allowed batch width and the effective pairwise compatibility pressure depend on expected useful depth remaining; batch is deliberately more permissive when the selector still expects several useful admissions and tighter when saturation pressure is high.
So batch selection is a greedy set-building rule:
$$
\mathcal B_{\mathrm{batch},0}=\varnothing,
\qquad
\mathcal B_{\mathrm{batch},k+1}=
\mathcal B_{\mathrm{batch},k}\cup\{r\}
\ \text{only if }\
r\ \text{is near-degenerate and still has positive net value after penalties.}
$$

Qualitatively, batch means “several shortlisted candidates are so similarly good that we admit more than one at the same ADAPT depth, provided they do not clash too much.” It is not beam: beam keeps multiple alternative scaffolds alive, while batch commits multiple compatible admissions into one scaffold immediately. It is also not a joint global optimizer over subsets; it is a greedy compatibility-filtered multi-admission rule.

### 11.4.3 Post-admission local simplification: permissibility, expendability, and gain-retention safety

After Phase 3 admits its winning payload, the selector surface does not terminate immediately. Let $\mathcal A_d^\star$ denote the payload admitted at adaptive depth $d$: either a singleton $\{r^\star\}$ or a greedy batch $\mathcal B_{\mathrm{batch}}^\star$. Let
$$
(\mathcal O_d,\theta_d,E_d)
$$
be the scaffold state immediately before that admission, and let
$$
(\mathcal O_d^+,\theta_d^+,E_d^+)
$$
be the post-insertion state after the ordinary local reoptimization on the admitted scaffold has completed. The admitted-step gain is
$$
\Delta_{\mathrm{adm}}(\mathcal A_d^\star)=E_d-E_d^+.
$$

The local simplification pass is guarded first by a prune-permissibility gate
$$
\Gamma_{\mathrm{prune}}^{\mathrm{perm}}(\mathcal O_d^+,\mathcal A_d^\star)\in\{0,1\},
$$
which decides whether any post-admission removal attempt is allowed at all. A canonical controller-facing form is
$$
\Gamma_{\mathrm{prune}}^{\mathrm{perm}}(t)
=
\mathbf 1[\text{accepted admission at }t]\,
\mathbf 1[\text{rollback-safe}]\,
\mathbf 1\!\Bigl[
W_{\mathrm{sat}}(t)\ge \tau_{\mathrm{hi}}
\ \vee\
\bigl(W_{\mathrm{sat}}(t)\ge \tau_{\mathrm{med}}
\wedge
\Delta_{\mathrm{adm}}(\mathcal A_d^\star)\le c_{\mathrm{weak}}\Delta_{\mathrm{scr}}(t)\bigr)
\Bigr],
$$
with screening scale
$$
\Delta_{\mathrm{scr}}(t)=\max\!\Bigl(\tau_{\mathrm{use},t},\operatorname{EWMA}(\Delta_{\mathrm{adm}}^+),\Delta_{\mathrm{num}}\Bigr).
$$
Here $\tau_{\mathrm{hi}},\tau_{\mathrm{med}},c_{\mathrm{weak}},\tau_{\mathrm{use},t}$, and $\Delta_{\mathrm{num}}$ are controller thresholds or numerical safety scales.
So the permissibility gate depends on global saturation/runway state and local step quality, but it is not itself a ranking score.

If $\Gamma_{\mathrm{prune}}^{\mathrm{perm}}=1$, define the locally expendable set
$$
\mathcal J_{\mathrm{exp}}(\mathcal A_d^\star)\subseteq\{1,\dots,|\mathcal O_d^+|\},
$$
obtained from the mature eligible set
$$
\mathcal M_t
=
\Bigl\{
j:
a_j(t)\ge a_{\min},\ 
j\notin\mathcal P_{\mathrm{protect}}(t),\ 
c_j^{\mathrm{cool}}(t)=0
\Bigr\},
$$
after excluding coordinates introduced by the just-admitted payload. Thus prune permissibility and local expendability remain separate notions: the first asks whether simplification may be attempted, while the second asks which previously active coordinates are locally weakest once admission has already succeeded.

If a cheap local prescreen is desired, define
$$
\theta_{\mathrm{ref}}(t)=\max\!\Bigl(\theta_{\mathrm{abs}},c_\theta\,\operatorname{median}_{j\in\mathcal M_t}|\operatorname{wrap}(\theta_j(t))|\Bigr),
$$
$$
S_\theta(j,t)=\left[1+\left(\frac{|\operatorname{wrap}(\theta_j(t))|}{\theta_{\mathrm{ref}}(t)+\varepsilon_\theta}\right)^{p_\theta}\right]^{-1},
$$
and let
$$
\mathcal P_{\mathrm{probe}}(t)=
\begin{cases}
\operatorname{Top}_{K_p}\bigl(\mathcal M_t;S_\theta(\cdot,t)\bigr),&\text{if angle prescreen is enabled},\\
\mathcal M_t,&\text{otherwise}.
\end{cases}
$$
This prescreen is computational only. It is not the authoritative local prune score.

For each $j\in\mathcal P_{\mathrm{probe}}(t)$, let $E_{-j}^{\mathrm{frz}}$ denote the frozen-ablation energy obtained by removing coordinate $j$ from the post-admission scaffold while holding all remaining post-admission coordinates fixed at their values in $\theta_d^+$. The local frozen-ablation loss is then
$$
L_j^{\mathrm{frz}}(\mathcal A_d^\star)=E_{-j}^{\mathrm{frz}}-E_d^+.
$$
Coordinates with smaller $L_j^{\mathrm{frz}}$ are more expendable, so the canonical local prune ranking is
$$
X_j(t)
=
\left[
1+\frac{L_j^{\mathrm{frz}}(\mathcal A_d^\star)}{\kappa_\delta\Delta_{\mathrm{scr}}(t)+\varepsilon}
\right]^{-1},
$$
and the live prune candidate set is
$$
\mathcal C_{\mathrm{exp}}(t)=\{j\in\mathcal P_{\mathrm{probe}}(t):X_j(t)\ge \tau_{\mathrm{stale}}\}.
$$
A weak age/rank regularizer may then be written as
$$
B_j(t)=1+\lambda_{\mathrm{age}}\bigl(2\varrho_j(t)-1\bigr),
\qquad 0\le \lambda_{\mathrm{age}}\le 0.1,
$$
$$
R_j(t)=X_j(t)\,B_j(t),
\qquad
j_t^\star=\arg\max_{j\in\mathcal C_{\mathrm{exp}}(t)}R_j(t).
$$
Thus $X_j(t)$ is the authoritative local expendability score, while $B_j(t)$ is only a weak policy bias. The frozen-ablation loss is therefore a local, solver-relative expendability score; it is not a theorem that the corresponding coordinate is globally redundant in the full optimization landscape. The global saturation variable $W_{\mathrm{sat}}(t)$ belongs in $\Gamma_{\mathrm{prune}}^{\mathrm{perm}}(t)$, not in $X_j(t)$ or $R_j(t)$ themselves.

Before the actual remove-refit trial, take the exact rollback snapshot
$$
\Xi_t^{\mathrm{trial}}
=
\bigl(
\mathcal O_d^+,\theta_d^+,E_d^+,\mathcal M_{\mathrm{opt}}^+,\text{local prune metadata}
\bigr).
$$
Each ranked removal attempt is then tested by a local remove-refit trial on the reduced scaffold. If $E_t^{(-j_t^\star)}$ is the post-refit energy of the scaffold with $j_t^\star$ removed, then the retained admitted-step gain is
$$
\Delta_{\mathrm{keep}}^{(j_t^\star)}=E_d-E_t^{(-j_t^\star)}.
$$
The canonical Phase-3 prune accept rule requires both a numerical safety guard and gain retention:
$$
E_t^{(-j_t^\star)}\le E_d^+ + \Delta_{\mathrm{safe}},
\qquad
\Delta_{\mathrm{keep}}^{(j_t^\star)}
\ge
\eta_{\mathrm{ret}}\,\Delta_{\mathrm{adm}}(\mathcal A_d^\star).
$$
In repo-facing notation, with
$$
E_t^-:=E_d,\qquad E_t:=E_d^+,\qquad \delta_t:=\Delta_{\mathrm{adm}}(\mathcal A_d^\star),\qquad \beta_{\mathrm{keep}}:=\eta_{\mathrm{ret}},
$$
the same accept rule is
$$
E_t^{(-j_t^\star)}
\le
E_t^- - \beta_{\mathrm{keep}}\delta_t + \Delta_{\mathrm{safe}}.
$$
The first form shows explicit retained gain. The second is the concise repo-facing energy inequality. If a trial fails, the state is restored exactly to $\Xi_t^{\mathrm{trial}}$.

So the post-admission local simplification contract is
$$
\text{admit }\mathcal A_d^\star
\;\longrightarrow\;
\text{local reopt}
\;\longrightarrow\;
\text{snapshot }\Xi_t^{\mathrm{trial}}
\;\longrightarrow\;
\begin{cases}
\text{keep that state},&\Gamma_{\mathrm{prune}}^{\mathrm{perm}}=0,\\
\text{rank } \mathcal C_{\mathrm{exp}}(t) \text{ by }R_j(t),&\Gamma_{\mathrm{prune}}^{\mathrm{perm}}=1,
\end{cases}
$$
followed, when pruning is permitted, by sequential remove-refit trials with exact rollback on every rejected trial. A failed local prune attempt restores the last committed post-admission state; prune rejection never invalidates the original admission event itself. In particular, prune permissibility is not a local expendability score, frozen ablation is not a global redundancy theorem, and global progress or stopping logic remain controller-level surfaces rather than consequences of the local prune ranking alone.

### 11.4.4 Beam-adapt over scaffold alternatives: branch state, pruning, and effective selector

In this section beam-adapt is **purely structural**: each branch carries an alternative ADAPT scaffold together with its locally refit amplitudes. There is no time checkpoint, probe rollout, or projected-dynamics integral in this section. Those belong to the separate time-dynamics beam surface developed in §17.

Within each branch, the local cheap-screen / shortlist chain reuses §§11.4.1--11.4.2: the branch-local cheap universe is obtained by ranking the full branch-local admissible record set with $S_{3,\mathrm{cheap}}$, not by inserting a separate upstream gradient-only cap. The branch-level frontier-prune operators introduced below remain beam-management surfaces and are distinct from the post-admission scaffold-simplification prune contract of §11.4.3.

The cleanest way to read a live scaffold branch is
$$
\mathfrak b
=
(\text{identity/history};\ \text{current scaffold and amplitudes};\ \text{current energy/depth};\ \text{cumulative selector totals/status}).
$$
More explicitly, write
$$
\begin{aligned}
\mathfrak b
\!&=\bigl(\operatorname{id}_b,\pi_b,\mathcal H_b,\mathcal O_b,\theta_b,E_b,d_b,S_b^{\mathrm{cum}},K_b^{\mathrm{cum}},\sigma_b,\tau_b\bigr),\\
\mathcal H_b&=(r_{b,1},\dots,r_{b,d_b}),
\qquad
r_{b,j}=(m_{b,j}^{\mathrm{adm}},p_{b,j}^{\mathrm{adm}}),\\
\mathcal O_b&=(m_{b,1},\dots,m_{b,|\mathcal O_b|}),
\qquad
E_b=E(\theta_b;\mathcal O_b),\\
S_b^{\mathrm{cum}}&=\sum_{j=1}^{d_b}s_{b,j},
\qquad
K_b^{\mathrm{cum}}=\sum_{j=1}^{d_b}\kappa_{b,j}.
\end{aligned}
$$
Here $\operatorname{id}_b$ and $\pi_b$ are the branch and parent identifiers, $\mathcal H_b$ is the ordered history of admitted records, $\mathcal O_b$ is the scaffold currently carried by branch $b$, $\theta_b$ is the refit parameter vector on that scaffold, $E_b$ is the current variational energy, $d_b$ is the branch ADAPT depth, $s_{b,j}$ and $\kappa_{b,j}$ are the selector score and burden increment contributed by the $j$th admitted record in $\mathcal H_b$, and $(\sigma_b,\tau_b)$ record live/terminal status and termination label.

If beam search is entered from an incumbent scaffold $(\mathcal O^{(0)},\theta^{(0)})$, the root branch is
$$
\begin{aligned}
\mathfrak b_{\mathrm{root}}
=
\bigl(
\operatorname{id}_0,\varnothing,\mathcal H^{(0)},\mathcal O^{(0)},\theta^{(0)},E(\theta^{(0)};\mathcal O^{(0)}),d_0,S_{\mathrm{root}}^{\mathrm{cum}},K_{\mathrm{root}}^{\mathrm{cum}},\mathrm{frontier},\varnothing
\bigr),
\end{aligned}
$$
with
$$
d_0=|\mathcal H^{(0)}|,
\qquad
S_{\mathrm{root}}^{\mathrm{cum}}=\sum_{j=1}^{d_0}s_j^{(0)},
\qquad
K_{\mathrm{root}}^{\mathrm{cum}}=\sum_{j=1}^{d_0}\kappa_j^{(0)}.
$$
If the beam is started from a fresh scaffold, then $\mathcal H^{(0)}=\varnothing$, $d_0=0$, and both cumulative sums vanish.

One beam round is best read as four structural steps.

**(i) Recompute the branch-local cumulative Phase-3 screening chain.**  
Each live branch $\mathfrak b\in\mathfrak F_c$ induces its own raw candidate-position surface. Let
$$
\eta_b\in\{\mathrm{core/seed},\mathrm{residual}\}
$$
be the branch-local stage label, and let the branch-local enabled generator family be
$$
\mathcal G_{\eta_b}^{(3)}(\mathfrak b)
=
\begin{cases}
\mathcal G_{\mathrm{core/seed}}^{(3)}(\mathfrak b),&\eta_b=\mathrm{core/seed},\\
\mathcal G_{\mathrm{residual}}^{(3)}(\mathfrak b),&\eta_b=\mathrm{residual}.
\end{cases}
$$
Let $\mathcal P_{\mathrm{residual}}(\mathfrak b)\subseteq\mathcal G^{(3)}$ denote the residual-only generator family on branch $b$.^[Implementation note: this is a controller-defined subset of the Phase-3 pool rather than a primitive Hamiltonian object. In practice it is decided from the branch's current stage/state and can depend on runtime policy rather than only on algebraic data.] Then the controller-stage gate and the branch-local available Phase-3 generator family are
$$
\Gamma_{\mathrm{stage},b}(m)
=
\begin{cases}
1,&\eta_b=\mathrm{residual},\\
1,&\eta_b=\mathrm{core/seed}\ \text{and}\ m\notin\mathcal P_{\mathrm{residual}}(\mathfrak b),\\
0,&\eta_b=\mathrm{core/seed}\ \text{and}\ m\in\mathcal P_{\mathrm{residual}}(\mathfrak b),
\end{cases}
\qquad
\mathcal G_{\mathrm{avail}}^{(3)}(\mathfrak b)
=
\{\,m\in\mathcal G^{(3)}:\Gamma_{\mathrm{stage},b}(m)=1\,\}.
$$
In the branch-local beam surface, the optional transient `seed` stage is absorbed into the label $\mathrm{core/seed}$ and resolves immediately before persistent beam branching. A convenient closed stage-label transition rule consistent with the stage-controller surface of §11.6 is^[Reader note: $\eta_b$, $n_d^{\mathrm{small}}$, $d_{\min}$, $M$, $\tau_{\mathrm{drop}}$, and $\chi_b^{\mathrm{trough}}$ are best read as controller-state quantities. The manuscript gives a mathematically usable surface for them, but they are still runtime decision variables/signals rather than primitive observables derived directly from the Hamiltonian.] 
$$
\eta_{\mathrm{root}}\in\{\mathrm{core/seed},\mathrm{residual}\},
\qquad
\eta_{b'}=
\begin{cases}
\mathrm{residual},&\eta_b=\mathrm{residual},\\
\mathrm{residual},&\eta_b=\mathrm{core/seed},\ d_{b'}\ge d_{\min},\ n_{d_{b'}}^{\mathrm{small}}=M,\ \chi_{b'}^{\mathrm{trough}}=0,\\
\mathrm{core/seed},&\text{otherwise},
\end{cases}
$$
where $\chi_{b'}^{\mathrm{trough}}\in\{0,1\}$ is the branch-local trough indicator. Thus $\eta_{\mathrm{root}}$ is inherited from the incoming controller state, once a branch reaches residual stage it remains there, and the core/seed $\to$ residual transition occurs only after a branch-local plateau-patience hit without trough detection.
Now let
$$
n_b=|\mathcal O_b|,
\qquad
p_{\mathrm{app}}(\mathfrak b)=n_b,
\qquad
W_{\mathrm{act}}(\mathfrak b)\subseteq\{0,\dots,n_b-1\}
$$
be the current scaffold length, append slot, and active position-probe window on branch $b$. The ordered probe list is
$$
\mathcal L_{\mathrm{probe}}(\mathfrak b)
:=
\bigl(p_{\mathrm{app}}(\mathfrak b),\,0,\,W_{\mathrm{act}}(\mathfrak b)\bigr),
$$
where the entries of $W_{\mathrm{act}}(\mathfrak b)$ are read in their native order. The distinct retained probe positions are^[Algorithmic note: $\operatorname{Dedup}_{\mathrm{ord}}$ means stable order-preserving deduplication of the probe list, and $\operatorname{Head}_{M_{\mathrm{probe}}}$ means truncation to the first $M_{\mathrm{probe}}$ surviving entries. These are algorithmic selection operators rather than primitive algebraic maps.]
$$
\mathcal P_{\mathrm{probe}}(\mathfrak b)
=
\operatorname{Head}_{M_{\mathrm{probe}}}
\Bigl(
\operatorname{Dedup}_{\mathrm{ord}}\bigl(\mathcal L_{\mathrm{probe}}(\mathfrak b)\bigr)
\Bigr)
\subseteq
\{0,\dots,n_b\}.
$$
For each candidate-position record $r=(m,p)$, let the branch-local symmetry gate be
$$
\Gamma_{\mathrm{sym},b}(m,p)
=
\begin{cases}
0,&\text{the symmetry audit for }(m,p;\mathcal O_b)\text{ returns a hard violation},\\
1,&\text{otherwise}.
\end{cases}
$$
Hence the branch-local admissible insertion-position set for generator $m$ is
$$
\mathcal P_m(\mathfrak b)
=
\{\,p\in\mathcal P_{\mathrm{probe}}(\mathfrak b):\Gamma_{\mathrm{sym},b}(m,p)=1\,\},
$$
and the resulting raw candidate-position surface is
$$
\mathcal R_b^{\mathrm{raw}}
=
\{\,r=(m,p):m\in\mathcal G_{\mathrm{avail}}^{(3)}(\mathfrak b),\ p\in\mathcal P_m(\mathfrak b)\,\},
$$
so the generator stage gate and the position-level symmetry gate are already absorbed into $\mathcal R_b^{\mathrm{raw}}$. In the maximally wide limit,
$$
W_{\mathrm{act}}(\mathfrak b)=\{0,\dots,n_b-1\},
\qquad
M_{\mathrm{probe}}\ge n_b+1
\quad\Longrightarrow\quad
\mathcal P_{\mathrm{probe}}(\mathfrak b)=\{0,\dots,n_b\},
$$
and the branch can test every insertion position on its current scaffold. The local screening flow is then
$$
\begin{aligned}
N_{\mathrm{cheap}}(\mathfrak b)
&=
\min\!\left\{N_{\mathrm{cheap}}^{\max},|\mathcal R_b^{\mathrm{raw}}|,K_{\mathrm{coarse}}(\mathfrak b)\right\},\\
\mathcal C_{\mathrm{cheap}}(\mathfrak b)
&=
\operatorname{Top}_{N_{\mathrm{cheap}}(\mathfrak b)}\!\left(\mathcal R_b^{\mathrm{raw}};S_{3,\mathrm{cheap}}\right),\\
N_{\mathrm{short}}(\mathfrak b)
&=
\min\!\left\{|\mathcal C_{\mathrm{cheap}}(\mathfrak b)|,N_{\max},\left\lceil f_{\mathrm{short}}|\mathcal C_{\mathrm{cheap}}(\mathfrak b)|\right\rceil\right\},\\
\mathcal S_3(\mathfrak b)
&=
\operatorname{Top}_{N_{\mathrm{short}}(\mathfrak b)}\!\left(\mathcal C_{\mathrm{cheap}}(\mathfrak b);S_{3,\mathrm{cheap}}\right).
\end{aligned}
$$
where
$$
H_b=\min\!\bigl(D_{\mathrm{left}}(\mathfrak b),\widehat N_{\mathrm{rem}}(\mathfrak b)\bigr)
$$
is the branch-local useful horizon. The same controller layer may also induce
$$
K_{\mathrm{coarse}}(\mathfrak b)
=
K_{\min}+\left\lfloor (K_{\max}-K_{\min})\frac{\widehat N_{\mathrm{rem}}^{\mathrm{hi}}(\mathfrak b)}{D_{\mathrm{left}}(\mathfrak b)+\varepsilon}\right\rfloor,
$$
$$
B_{\mathrm{target}}(\mathfrak b)
=
B_{\min}+\left\lfloor (B_{\max}-B_{\min})\frac{\widehat N_{\mathrm{rem}}(\mathfrak b)}{D_{\mathrm{left}}(\mathfrak b)+\varepsilon}C(\mathfrak b)\right\rfloor,
$$
$$
\Pi_{\mathfrak b}(r,r')=\gamma_{\mathrm{compat}}(\mathfrak b)\,\Pi_0(r,r'),
\qquad
\gamma_{\mathrm{compat}}(\mathfrak b)=\gamma_{\min}+(\gamma_{\max}-\gamma_{\min})W_{\mathrm{sat}}(\mathfrak b),
$$
and mode/regime labels
$$
\pi(\mathfrak b)\in\{\mathrm{fast},\mathrm{mixed},\mathrm{high\mbox{-}fidelity}\},
\qquad
\mathrm{regime}_{\mathrm{sat}}(\mathfrak b)\in\{\mathrm{hungry},\mathrm{watch},\mathrm{plateau}\}.
$$
Using the same cumulative chain from §§11.4.1--11.4.2 — first the reused cheap screen, then the shortlist/full rerank, and finally the Phase-3 augmentations — this branch-local screen is now evaluated on the current branch state $(\mathcal O_b,\theta_b)$ with
$$
\Gamma_{\mathrm{stage}}(m)=\Gamma_{\mathrm{stage},b}(m),
\qquad
\Gamma_{\mathrm{sym}}(r)=\Gamma_{\mathrm{sym},b}(m,p)
\quad\text{for }r=(m,p).
$$
Also set
$$
S_{3,\mathrm{cheap}}(r;\mathfrak b):=s_{b,\mathrm{cheap}}(r),
\qquad
S_{3,\mathrm{base}}(r;\mathfrak b):=s_{b,\mathrm{base}}(r),
$$
so the branch-local shortlist operators are using the same score names as §§11.4.1--11.4.2.
Denote the resulting local selector by
$$
s_b(r)=S_{3,\mathrm{aug}}(r;\mathfrak b)
\quad\text{or}\quad
s_b(r)=S_{3,\mathrm{base}}(r;\mathfrak b).
$$
More explicitly, the branch-local score flow is
$$
\begin{aligned}
s_{b,\mathrm{cheap}}(r)
&=
\Gamma_{\mathrm{stage},b}(m)\Gamma_{\mathrm{sym},b}(m,p)
\frac{\max\!\left\{\left|2\Re\langle\psi_b|H|d_r^{(b)}\rangle\right|-z_\alpha\sigma_r(\mathfrak b),0\right\}^2}{2\lambda_F\langle d_r^{(b)}|Q_{\psi_b}|d_r^{(b)}\rangle\,(K_{\mathrm{cheap}}(r;\mathfrak b)+\varepsilon)},\\
|\psi_b\rangle
&=
U(\theta_b;\mathcal O_b)|\phi_0\rangle,
\qquad
Q_{\psi_b}=I-|\psi_b\rangle\langle\psi_b|,\\
|d_r^{(b)}\rangle
&=
-i\widetilde A_r^{(b)}|\psi_b\rangle,
\qquad
t_r^{(b)}=Q_{\psi_b}|d_r^{(b)}\rangle,
\qquad
F_{\mathrm{raw}}(r;\mathfrak b)=\langle d_r^{(b)}|Q_{\psi_b}|d_r^{(b)}\rangle,\\
t_j^{(b)}
&=
Q_{\psi_b}\partial_{\theta_j}|\psi_b\rangle,
\qquad
(Q_r(\mathfrak b))_{jk}=\Re\langle t_j^{(b)},t_k^{(b)}\rangle,
\qquad
(q_r(\mathfrak b))_j=\Re\langle t_j^{(b)},t_r^{(b)}\rangle,\\
\widetilde h_r(\mathfrak b)
&=
h_r(\mathfrak b)-b_r(\mathfrak b)^\top M_r(\mathfrak b)^{-1}b_r(\mathfrak b),\\
F_r^{\mathrm{red}}(\mathfrak b)
&=
F_{\mathrm{raw}}(r;\mathfrak b)
-2q_r(\mathfrak b)^\top M_r(\mathfrak b)^{-1}b_r(\mathfrak b)
+b_r(\mathfrak b)^\top M_r(\mathfrak b)^{-1}Q_r(\mathfrak b)M_r(\mathfrak b)^{-1}b_r(\mathfrak b),\\
q_r^{\mathrm{red}}(\mathfrak b)
&=
q_r(\mathfrak b)-Q_r(\mathfrak b)M_r(\mathfrak b)^{-1}b_r(\mathfrak b),\\
\nu_r(\mathfrak b)
&=
\operatorname{clip}_{[0,1]}\!\left(
1-\frac{(q_r^{\mathrm{red}}(\mathfrak b))^\top(Q_r(\mathfrak b)+\varepsilon_{\mathrm{nov}}I)^{-1}q_r^{\mathrm{red}}(\mathfrak b)}{F_r^{\mathrm{red}}(\mathfrak b)}
\right),\\
\Delta E_{\mathrm{TR}}(r;\mathfrak b)
&=
\max_{|\alpha|\le \rho/\sqrt{F_r^{\mathrm{red}}(\mathfrak b)}}
\left[
g_{\mathrm{lcb}}(r;\mathfrak b)|\alpha|-\frac12\max(\widetilde h_r(\mathfrak b),0)\alpha^2
\right],\\
s_{b,\mathrm{base}}(r)
&=
\Gamma_{\mathrm{stage},b}(m)\Gamma_{\mathrm{sym},b}(m,p)
\left[
\operatorname{clip}_{[0,1]}\!\left(
1-\frac{(q_r^{\mathrm{red}}(\mathfrak b))^\top(Q_r(\mathfrak b)+\varepsilon_{\mathrm{nov}}I)^{-1}q_r^{\mathrm{red}}(\mathfrak b)}{F_r^{\mathrm{red}}(\mathfrak b)}
\right)
\right]^{\gamma_N}\\
&\qquad\times
\frac{
\max_{|\alpha|\le \rho/\sqrt{F_r^{\mathrm{red}}(\mathfrak b)}}
\left[
\max\!\left\{\left|2\Re\langle\psi_b|H|d_r^{(b)}\rangle\right|-z_\alpha\sigma_r(\mathfrak b),0\right\}|\alpha|
-\frac12\max(\widetilde h_r(\mathfrak b),0)\alpha^2
\right]
}{
K_{\mathrm{full}}(r;\mathfrak b)+\varepsilon
},\\
s_b(r)
&=
s_{b,\mathrm{base}}(r)
+\beta_{\mathrm{split}}\Sigma(r;\mathfrak b)
+\beta_{\mathrm{motif}}M(r;\mathfrak b)
+\beta_{\mathrm{sym}}Y(r;\mathfrak b)
-\beta_{\mathrm{dup}}D_{\mathrm{dup}}(r;\mathfrak b).
\end{aligned}
$$
So the branch-local screening/rerank chain closes as
$$
\mathcal R_b^{\mathrm{raw}}
\xrightarrow{\ s_{b,\mathrm{cheap}}\ }
\mathcal C_{\mathrm{cheap}}(\mathfrak b)
\xrightarrow{\ s_{b,\mathrm{cheap}}\ }
\mathcal S_3(\mathfrak b)
\xrightarrow{\ s_b\ }
\mathcal A_b^{\mathrm{raw}}
\xrightarrow{\operatorname{Top}_{K_b}}
\mathcal A_b.
$$
For $r=(m,p)\in\mathcal S_3(\mathfrak b)$, let the runtime-split child family be
$$
\mathcal C_{\mathrm{split}}(m)=\{m_1,\dots,m_{K_m}\}.
$$
For a child subset $S\subseteq\mathcal C_{\mathrm{split}}(m)$, define the combined child-set polynomial
$$
m_S:=\sum_{u\in S}u.
$$
For any real-coefficient Pauli polynomial
$$
m=\sum_{w}c_wP_w
$$
written in the internal $e/x/y/z$ word alphabet, define the canonical polynomial signature
$$
\operatorname{Sig}_{\mathrm{poly}}(m;\tau_{\mathrm{sig}})
:=
\operatorname{sort}
\left\{
\bigl(w,\operatorname{rnd}_{12}(c_w)\bigr):
|c_w|>\tau_{\mathrm{sig}}
\right\},
\qquad
\operatorname{rnd}_{12}(x):=10^{-12}\operatorname{round}(10^{12}x).
$$
Thus $\operatorname{Dedup}_{\mathrm{sig}}$ retains one representative from each common value of $\operatorname{Sig}_{\mathrm{poly}}(\cdot;\tau_{\mathrm{sig}})$ after collecting like terms. Each singleton child is re-audited at the same insertion position $p$, and the symmetry-safe combined child-set family is
$$
\mathcal A_{\mathrm{split}}(m,p;\mathfrak b)
=
\operatorname{Dedup}_{\mathrm{sig}}
\Bigl[
\{\,S\subseteq\mathcal C_{\mathrm{split}}(m):\Gamma_{\mathrm{sym},b}(m_S,p)=1,\ m_S\not\equiv m\,\}
\Bigr].
$$
Write
$$
r_j=(m_j,p),
\qquad
r_S=(m_S,p)\quad (|S|\ge 2),
$$
for the singleton and child-set representatives, where $r_S$ denotes the combined child-set polynomial placed at the same position $p$. Then the full split-aware promotion universe of the parent record is
$$
\mathcal U_b(r)
=
\{r\}
\cup
\{\,r_j:\Gamma_{\mathrm{sym},b}(m_j,p)=1\,\}
\cup
\{\,r_S:S\in\mathcal A_{\mathrm{split}}(m,p;\mathfrak b),\ |S|\ge 2\,\}.
$$
Unsafe child atoms and unsafe child sets are excluded from $\mathcal U_b(r)$ by the symmetry gate and therefore may remain in provenance only, not in the admissible promotion surface. Now define
$$
r_b^{\star}(r)
=
\arg\max_{u\in\mathcal U_b(r)} s_b(u),
$$
and then the split-aware promoted representative is
$$
\widehat r_b(r)
=
\begin{cases}
r_b^{\star}(r),&\text{if }s_b\!\bigl(r_b^{\star}(r)\bigr)>s_b(r)+\tau_{\mathrm{split}},\\
r,&\text{otherwise}.
\end{cases}
$$
The actual raw admission set passed to the beam is therefore
$$
\mathcal A_b^{\mathrm{raw}}
=
\{\,\widehat r_b(r):r\in\mathcal S_3(\mathfrak b),\ s_b(\widehat r_b(r))>0\,\},
$$
and the retained candidate-admission set is
$$
\mathcal A_b
=
\operatorname{Top}_{K_b}\!\left(\mathcal A_b^{\mathrm{raw}};s_b\right),
\qquad
K_b\le B_{\mathrm{child}}-1.
$$
^[Beam-width note: $\operatorname{Top}_{K_b}$ is the score-ordered child selector on branch $b$. The inequality fixes only an admissible cap; the exact runtime choice of $K_b$ remains a controller policy decision unless specified separately.]

**(ii) Materialize either stop or one new admission.**  
The proposal family is
$$
\mathcal Q(b)=\{\mathrm{stop}\}\cup\mathcal A_b.
$$
The symbol $\mathrm{stop}$ means “terminate this scaffold as-is,” while each $r=(m,p)\in\mathcal A_b$ means “insert one new generator into this branch and refit.” If
$$
\mathcal O_b=(m_{b,1},\dots,m_{b,n_b}),
\qquad
0\le p\le n_b,
$$
then the scaffold insertion map is
$$
\operatorname{Insert}(\mathcal O_b,m,p)
=
(m_{b,1},\dots,m_{b,p},m,m_{b,p+1},\dots,m_{b,n_b}),
$$
and the insertion index transport is
$$
I_p(j)
=
\begin{cases}
j,&1\le j\le p,\\
j+1,&p<j\le n_b.
\end{cases}
$$
The zero-lifted parameter vector is therefore
$$
\theta_b^{\uparrow(r)}
=
(\theta_{b,1},\dots,\theta_{b,p},0,\theta_{b,p+1},\dots,\theta_{b,n_b}).
$$
Equivalently,
$$
(\theta_b^{\uparrow(r)})_{I_p(j)}=\theta_{b,j},
\qquad
(\theta_b^{\uparrow(r)})_{p+1}=0.
$$
For newest-window cap $w_{\mathrm{new}}$ and amplitude-carry cap $k_{|\theta|}$, let
$$
w_{\mathrm{eff}}^{(p)}=\min\{w_{\mathrm{new}},n_b+1\},
\qquad
W_{\mathrm{new}}^{(p)}=\{n_b+2-w_{\mathrm{eff}}^{(p)},\dots,n_b+1\},
$$
$$
W_{|\theta|}^{(p)}
=
\operatorname{Top}_{k_{|\theta|}}
\Bigl(
\{1,\dots,n_b+1\}\backslash W_{\mathrm{new}}^{(p)};
\bigl|(\theta_b^{\uparrow(r)})_j\bigr|
\Bigr).
$$
Hence the inherited refit window on the inserted scaffold is
$$
W_r=W(m,p)=W_{\mathrm{new}}^{(p)}\cup W_{|\theta|}^{(p)}.
$$
Then the branch-local refit map is
$$
\operatorname{Refit}_{W_r}(\theta_b;r)
:=
\arg\min_{\vartheta}
\left\{
E(\vartheta;\operatorname{Insert}(\mathcal O_b,m,p))
:\ 
\vartheta_j=(\theta_b^{\uparrow(r)})_j\ \forall j\notin W_r
\right\}.
$$
The transfer map is
$$
\mathcal T(\mathfrak b,\mathrm{stop})=\mathfrak b'
\quad\text{with}\quad
\mathcal H_{b'}=\mathcal H_b,\ 
\mathcal O_{b'}=\mathcal O_b,\ 
\theta_{b'}=\theta_b,\ 
\ E_{b'}=E_b,\ 
\ d_{b'}=d_b,\ 
\ S_{b'}^{\mathrm{cum}}=S_b^{\mathrm{cum}},\ 
\ K_{b'}^{\mathrm{cum}}=K_b^{\mathrm{cum}},\ 
\sigma_{b'}=\mathrm{terminal},
\qquad
\tau_{b'}=
\begin{cases}
\mathrm{empty},&\mathcal A_b=\varnothing,\\
\mathrm{stop},&\mathcal A_b\neq\varnothing,
\end{cases}
$$
and for $r=(m,p)$,
$$
\mathcal T(\mathfrak b,r)=\mathfrak b'
\quad\text{with}\quad
\mathcal O_{b'}=\operatorname{Insert}(\mathcal O_b,m,p),
\qquad
\theta_{b'}=\operatorname{Refit}_{W_r}(\theta_b;r),
$$
$$
\mathcal H_{b'}=(\mathcal H_b,r),
\qquad
E_{b'}=E(\theta_{b'};\mathcal O_{b'}),
\qquad
d_{b'}=d_b+1,
\qquad
S_{b'}^{\mathrm{cum}}=S_b^{\mathrm{cum}}+s_b(r),
\qquad
K_{b'}^{\mathrm{cum}}=K_b^{\mathrm{cum}}+K_{\mathrm{full}}(r),
\qquad
\sigma_{b'}=
\begin{cases}
\mathrm{terminal},&d_{b'}\ge d_{\max}\ \text{or}\ \mathcal A_{b'}=\varnothing,\\
\mathrm{frontier},&d_{b'}<d_{\max}\ \text{and}\ \mathcal A_{b'}\neq\varnothing,
\end{cases}
\qquad
\tau_{b'}=
\begin{cases}
\mathrm{depth\_cap},&d_{b'}\ge d_{\max},\\
\mathrm{empty},&d_{b'}<d_{\max}\ \text{and}\ \mathcal A_{b'}=\varnothing,\\
\varnothing,&d_{b'}<d_{\max}\ \text{and}\ \mathcal A_{b'}\neq\varnothing,
\end{cases}
$$
where $\mathcal A_{b'}$ is recomputed from step (i) on the child branch itself. So a non-stop child is exactly one more ADAPT admission applied to the parent scaffold, followed by the inherited-window refit already defined above. If $\mathcal A_b=\varnothing$, then $\mathcal Q(b)=\{\mathrm{stop}\}$ and the branch contributes only a terminal child.

**(iii) Classify, deduplicate, and prune the scaffold children.**  
The frontier and terminal descendants of branch $b$ are
$$
\mathfrak C_b^{\mathrm{front}}
=
\{\mathfrak b' : \mathfrak b'=\mathcal T(\mathfrak b,r),\ r\in\mathcal A_b,\ \sigma_{b'}=\mathrm{frontier}\},
$$
$$
\mathfrak C_b^{\mathrm{term}}
=
\{\mathfrak b' : \mathfrak b'=\mathcal T(\mathfrak b,q),\ q\in\mathcal Q(b),\ \sigma_{b'}=\mathrm{terminal}\}.
$$
Two branches are treated as duplicates when they represent the same scaffold and essentially the same refit point, for example under the fingerprint
$$
\operatorname{rnd}_{10}(x):=10^{-10}\operatorname{round}(10^{10}x),
\qquad
\operatorname{round}_{10}(\theta_b)
:=
\bigl(\operatorname{rnd}_{10}(\theta_{b,1}),\dots,\operatorname{rnd}_{10}(\theta_{b,n_b})\bigr),
$$
$$
\mathrm{fp}(\mathfrak b)
=
\bigl(d_b,\operatorname{labels}(\mathcal O_b),\operatorname{round}_{10}(\theta_b)\bigr).
$$
Among near-equivalent branches, keep the one with best lexicographic prune key
$$
\kappa_{\mathrm{prune}}(\mathfrak b)
=
\bigl(E_b,-S_b^{\mathrm{cum}},K_b^{\mathrm{cum}},|\mathcal O_b|,\operatorname{labels}(\mathcal O_b),\operatorname{round}_{10}(\theta_b),\operatorname{id}_b\bigr).
$$
This means that, within the symbolic scaffold beam, lower current energy is primary; stronger cumulative admitted selector value is secondary; lower accumulated burden and then smaller/simpler equivalent scaffolds break the remaining ties.

The beam operators are therefore
$$
\operatorname{Dedup}(\mathcal X)
=
\{\text{best branch per fingerprint under }\kappa_{\mathrm{prune}}\},
\qquad
\operatorname{Prune}(\mathcal X;B)
=
\text{lowest-}B\text{ branches in }\operatorname{Dedup}(\mathcal X)\text{ by }\kappa_{\mathrm{prune}}.
$$
Hence the live and terminal pools update by
$$
\mathfrak F_{c+1}
=
\operatorname{Prune}\!\left(\bigcup_{b\in\mathfrak F_c}\mathfrak C_b^{\mathrm{front}};B_{\mathrm{live}}\right),
\qquad
\mathfrak T_{c+1}
=
\operatorname{Prune}\!\left(\mathfrak T_c\cup\bigcup_{b\in\mathfrak F_c}\mathfrak C_b^{\mathrm{term}};B_{\mathrm{term}}\right).
$$

**(iv) Choose the final scaffold branch.**  
After $C$ scaffold-beam rounds, the surviving candidate set is
$$
\mathfrak W_{\mathrm{final}}=\mathfrak F_C\cup\mathfrak T_C,
$$
and the effective beam selector is
$$
\mathfrak b_*
=
\arg\min_{\mathfrak b\in\mathfrak W_{\mathrm{final}}}\kappa_{\mathrm{prune}}(\mathfrak b).
$$
Equivalently,
$$
\mathfrak b_*
=
\arg\min_{\mathfrak b\in\mathfrak F_C\cup\mathfrak T_C}
\bigl(E_b,-S_b^{\mathrm{cum}},K_b^{\mathrm{cum}},|\mathcal O_b|,\operatorname{labels}(\mathcal O_b),\operatorname{round}_{10}(\theta_b),\operatorname{id}_b\bigr).
$$

A compact scaffold-beam algorithm block is therefore
$$
\begin{aligned}
\text{A: }&
\mathfrak F_0=\{\mathfrak b_{\mathrm{root}}\},
\qquad
\mathfrak T_0=\varnothing,\\
\text{B: }&
\mathcal R_b^{\mathrm{raw}}
\to
\mathcal C_{\mathrm{cheap}}(\mathfrak b)
\to
\mathcal S_3(\mathfrak b)
\to
\mathcal A_b
\qquad(\mathfrak b\in\mathfrak F_c),\\
\text{C: }&
\mathcal Q(b)=\{\mathrm{stop}\}\cup\mathcal A_b,
\qquad
\mathfrak C_b=\{\mathcal T(\mathfrak b,q):q\in\mathcal Q(b)\},\\
\text{D: }&
\mathfrak C_b^{\mathrm{front}}
=
\{\mathfrak b'\in\mathfrak C_b:\sigma_{b'}=\mathrm{frontier}\},
\qquad
\mathfrak C_b^{\mathrm{term}}
=
\{\mathfrak b'\in\mathfrak C_b:\sigma_{b'}=\mathrm{terminal}\},\\
\text{E: }&
\mathfrak F_{c+1}
=
\operatorname{Prune}\!\left(\bigcup_{b\in\mathfrak F_c}\mathfrak C_b^{\mathrm{front}};B_{\mathrm{live}}\right),
\qquad
\mathfrak T_{c+1}
=
\operatorname{Prune}\!\left(\mathfrak T_c\cup\bigcup_{b\in\mathfrak F_c}\mathfrak C_b^{\mathrm{term}};B_{\mathrm{term}}\right),\\
\text{F: }&
\mathfrak W_{\mathrm{final}}=\mathfrak F_C\cup\mathfrak T_C,
\qquad
\mathfrak b_*=\arg\min_{\mathfrak b\in\mathfrak W_{\mathrm{final}}}\kappa_{\mathrm{prune}}(\mathfrak b).
\end{aligned}
$$

In words: each round keeps several scaffold alternatives alive, recomputes the local cumulative Phase-3 screening chain on each surviving scaffold, branches by admitting one additional generator or stopping, deduplicates nearly equivalent descendants, prunes back to the beam budgets, and finally returns the best surviving scaffold under the lexicographic selector key.

The separate time-dynamics / projected-dynamics beam surface, with checkpoint advances, probe rollouts, and integrated residual objectives, is developed later in §17.

## 11.5 Active HH pool surface and selected scaffold

The active HH adaptive surface is carried by the nested pool family
$$
\Omega_{\mathrm{HH}}^{(3)}\subseteq\Omega_{\mathrm{HH}}^{(2)}\subseteq\Omega_{\mathrm{HH}}^{(1)},
\qquad
\mathcal G_{\mathrm{adapt}}^{(k)}=\{\tau_m\}_{m\in\Omega_{\mathrm{HH}}^{(k)}}.
$$
The adaptive output is the selected scaffold
$$
\mathcal O_*=(\tau_{m_1},\dots,\tau_{m_d}),
\qquad
\tau_{m_j}\in\mathcal G_{\mathrm{adapt}}^{(3)},
$$
together with its fitted amplitude vector and resulting state,
$$
(\mathcal O_*,\theta_*^{\mathrm{adapt}},|\psi_*\rangle),
\qquad
|\psi_*\rangle=U(\theta_*^{\mathrm{adapt}};\mathcal O_*)|\phi_0\rangle.
$$
Thus the active HH scaffold manifold selected by ADAPT may be written as
$$
\mathcal M_{\mathrm{scaf}}(\mathcal O_*)
=
\{\,U(\vartheta;\mathcal O_*)|\phi_0\rangle:\vartheta\in\mathbb R^{|\mathcal O_*|}\,\}.
$$
The obsolete historical warm-start $\to$ refine $\to$ ADAPT $\to$ replay chain is no longer part of the mainline symbolic flow and is recorded only in Appendix A.2.

## 11.6 Stage controller

Let the absolute error and its depth-to-depth improvement be
$$
\begin{aligned}
\Delta_d^{\mathrm{abs}}&=|E_d-E_{\mathrm{exact}}|,\\
\rho_d&=\Delta_{d-1}^{\mathrm{abs}}-\Delta_d^{\mathrm{abs}}=|E_{d-1}-E_{\mathrm{exact}}|-|E_d-E_{\mathrm{exact}}|.
\end{aligned}
$$
A compact stage-controller surface is then
$$
\begin{aligned}
n_d^{\mathrm{small}}
&=\sum_{\ell=\max\{1,d-M+1\}}^d \mathbf 1\!\bigl(\rho_\ell\le\tau_{\mathrm{drop}}\bigr),\\
\text{continue}
&\iff d<d_{\min}\ \text{or}\ n_d^{\mathrm{small}}<M,\\
\text{handoff/stop}
&\iff d\ge d_{\min}\ \text{and}\ n_d^{\mathrm{small}}=M.
\end{aligned}
$$
Gradient floors remain secondary diagnostics rather than the primary stop signal.

# 12. SPSA and Optimizer Semantics

## 12.1 SPSA schedules

The standard SPSA schedules are
$$
c_k=\frac{c}{(k+1)^\gamma},
\qquad
a_k=\frac{a}{(A+k+1)^\alpha}.
$$

## 12.2 Two-point stochastic gradient

At iteration $k$, with Rademacher direction $\Delta_k$, evaluate
$$
y_+=f(x_k+c_k\Delta_k),
\qquad
y_-=f(x_k-c_k\Delta_k),
$$
and form the gradient estimate
$$
\hat g_k = \frac{y_+-y_-}{2c_k}\,\Delta_k.
$$

## 12.3 Update and projection

The update is
$$
x_{k+1}=x_k-a_k\hat g_k.
$$
If projection or clipping is imposed, it is applied to both the perturbed points and the updated iterate.

## 12.4 Repeat aggregation

If the same point is sampled multiple times, the repeated evaluations may be aggregated by
$$
\operatorname{mean}
\qquad\text{or}\qquad
\operatorname{median}.
$$

## 12.5 Return policy

Two standard return policies are:

- return an averaged late-trajectory iterate,
- return the best sampled point observed during the run.

## 12.6 Surface note

Stochastic inner optimizers are especially natural for large, noisy, or nondifferentiable effective surfaces, while deterministic optimizers are often effective on smaller and smoother fixed-structure manifolds.

# 13. Drive, Exact Propagation, and Propagator Semantics

## 13.1 Static drive labels and time-dependent coefficients

Let $\mathcal L_{\mathrm{drive}}$ denote the fixed set of density operators appearing in the drive. Then the drive always has the form
$$
\hat H_{\mathrm{drive}}(t)=\sum_{\lambda\in\mathcal L_{\mathrm{drive}}} c_\lambda(t) P_\lambda,
$$
with fixed labels $P_\lambda$ and time-dependent coefficients $c_\lambda(t)$.

## 13.2 Reference coefficient map

For a site-density drive,
$$
\Delta c[Z_{p(i,\uparrow)}](t)=\Delta c[Z_{p(i,\downarrow)}](t)=-\frac{1}{2}v_i(t),
$$
and the identity shift is
$$
\Delta c[I](t)=\sum_i v_i(t)
$$
if one chooses to retain the scalar offset explicitly.

## 13.3 Propagator surfaces

A common propagator menu is:

- second-order Suzuki,
- piecewise exact propagation,
- fourth-order CFQM,
- sixth-order CFQM.

For CFQM methods, the key mathematical point is that the stage times are fixed scheme nodes $c_j$, so a macro-step from $t_n$ to $t_{n+1}=t_n+\Delta t$ samples the Hamiltonian at
$$
t_n+c_j\Delta t,
$$
not at a left, midpoint, or right rule chosen independently.

## 13.4 Reference propagator versus reported trajectory propagator

Write the reported propagator as $U_{\mathrm{traj}}(t)$ and the high-accuracy reference propagator as $U_{\mathrm{ref}}(t)$. These need not be the same approximation.

### 13.4.1 Static reference branch

In the static case, one may take
$$
U_{\mathrm{ref}}(t)=e^{-it\hat H}.
$$

### 13.4.2 Drive-enabled reference branch

For time-dependent $\hat H(t)$, a refined piecewise-constant reference is
$$
U_{\mathrm{ref}}(t_{n+1},t_n) \approx e^{-i\Delta t\hat H(t_n^*)},
$$
with $t_n^*$ sampled on a refined grid and the full reference obtained as an ordered product over the refined slices.

## 13.5 CFQM macro-step mathematics

A generic commutator-free quasi-Magnus macro-step takes the form
$$
U_{n+1,n}^{\mathrm{CFQM}} = \prod_{r=1}^{s} \exp\!\left[-i a_r \Delta t\,\hat H(t_n+c_r\Delta t)\right],
$$
with method-specific coefficients $a_r$ and nodes $c_r$. The formal order is achieved when the stage exponentials are themselves treated at the intended accuracy.

## 13.6 Filtered exact manifold and subspace fidelity

Let $\Pi_{\mathrm{sector}}$ denote the projector onto the target symmetry sector. Then the filtered exact energy is
$$
E_{\mathrm{exact,filtered}} = \min_{\psi\in\Pi_{\mathrm{sector}}\mathcal H} \langle \psi|\hat H|\psi\rangle,
$$
and a sector-aware fidelity may be defined by
$$
F_{\mathrm{sector}}(t) = \frac{|\langle \psi_{\mathrm{exact}}(t)|\Pi_{\mathrm{sector}}|\psi(t)\rangle|^2}{\langle \psi(t)|\Pi_{\mathrm{sector}}|\psi(t)\rangle}.
$$

## 13.7 Branch-resolved observable contracts

If several trajectory branches are carried simultaneously, each observable should be indexed by branch label $b$:
$$
\mathcal O_b(t),
\qquad
b\in\{\mathrm{var},\mathrm{traj},\mathrm{ref},\mathrm{noisy},\mathrm{ideal}\}.
$$
Typical observables are energy, density, doublon density, and fidelity.

## 13.8 State import and handoff contract

A portable state bundle can be written abstractly as
$$
\mathcal B = (\Lambda,\theta,a,E,\mathcal C),
$$
where $\Lambda$ is the physical manifest, $\theta$ are variational parameters when available, $a$ is an amplitude map for the working state, $E$ stores energy summaries, and $\mathcal C$ is continuation metadata.

## 13.9 Highest-tier continuation surfaces

At the highest continuation tier, $\mathcal C$ may include split-event summaries, motif data, symmetry diagnostics, replay policy data, and limited branch histories. These are naturally viewed as auxiliary mathematical metadata rather than as part of the operator algebra itself.

# 14. Continuation and Handoff Contract

## 14.1 Handoff state bundle

A handoff bundle may contain:

- the parameter manifest $\Lambda$,
- the current energy summary,
- the normalized statevector amplitudes,
- an exact comparison target,
- optional continuation metadata.

One convenient symbolic amplitude map is
$$
\{b_{N_q-1}\cdots b_0 \mapsto (\Re a_b,\Im a_b)\}.
$$

## 14.2 Continuation block

The continuation payload can be treated abstractly as
$$
\mathcal C = (m,\mathcal S,\mathcal M_{\mathrm{opt}},\Gamma,\Sigma,\mathcal L,\mathcal U,\Xi,\mathcal R,\mathcal P),
$$
where:

- $m$ is the continuation mode,
- $\mathcal S$ is inherited manifold-structure data,
- $\mathcal M_{\mathrm{opt}}$ is auxiliary optimizer state,
- $\Gamma$ is selected-generator metadata,
- $\Sigma$ is split-event data,
- $\mathcal L$ is a motif library,
- $\mathcal U$ is motif usage,
- $\Xi$ is symmetry information,
- $\mathcal R$ is rescue or recovery history,
- $\mathcal P$ is optional historical replay-policy metadata when such provenance is still being carried.

When controller forecast telemetry is persisted, quantities such as $D_{\mathrm{left}}(t)$, $\widehat N_{\mathrm{rem}}(t)$, and $H_t$ belong naturally to this policy/provenance layer. They remain heuristic controller state rather than operator-level observables.

## 14.3 General handoff map

Let $\mathcal H_{A\to B}$ be a handoff map from source manifold $\mathcal M_A$ to target manifold $\mathcal M_B$:
$$
\mathcal H_{A\to B}: (|\psi_A\rangle,\theta_A,\mathcal C_A) \mapsto (|\psi_B^{(0)}\rangle,\theta_B^{(0)},\mathcal C_B).
$$
The map is intentionally abstract here: it records state, coordinates, and continuation payload transfer between manifolds without committing the main body to any obsolete replay-specific burn-in policy.

## 14.4 Symmetry and tier-three payload note

Symmetry information belongs in the handoff if later stages will use it for validation, postselection, or projector-renormalized estimation. Mathematically, the carried object is a projector or a list of commuting symmetry generators.

## 14.5 Historical warm/refine chain note

The obsolete warm-start $\to$ refine $\to$ ADAPT $\to$ replay path is kept only in Appendix A.2 and is not part of the active main-body symbolic contract.

# 15. Cross-Checks and Exact-Benchmark Contracts

## 15.1 Trial matrix by problem family

For the Hubbard model, a natural benchmark trial set is
$$
\mathcal T_{\mathrm{Hub}} = \{\text{layerwise},\;\text{excitation-based},\;\text{adaptive(exc)},\;\text{adaptive(full-H)}\}.
$$
For the HH model, the canonical sector-preserving core set is
$$
\mathcal T_{\mathrm{HH}} = \{\text{HH-layerwise},\;\text{HH-physical-termwise},\;\text{adaptive(full-H)}\}.
$$
A seed-surface extension can enlarge $\mathcal T_{\mathrm{HH}}$ by adding polaronic refinements built on the same sector-preserving fixed-family surface.

## 15.2 Auto-scaled parameter resolution

Let
$$
S_{\mathrm{trot}}(L,n_{\mathrm{ph,max}}),
\quad
R_{\mathrm{restart}}(L,n_{\mathrm{ph,max}}),
\quad
I_{\max}(L,n_{\mathrm{ph,max}})
$$
be monotone nondecreasing control maps for Trotter steps, restart count, and optimizer effort. The key contract is monotonicity with problem size, not a single universal constant setting.

## 15.3 Exact target and shared reference-state construction

Within a benchmark matrix, all trial families should be compared against the same exact target and the same conserved-sector reference construction. Thus every trial is judged against the same $E_{\mathrm{exact,sector}}$ and the same physical Hamiltonian instance.

## 15.4 Per-trial energy and trajectory semantics

Each trial should at minimum record:

- the variational ground-state energy,
- the exact comparison target,
- trajectory observables on a shared time grid when dynamics are studied,
- fidelity or overlap relative to a common reference branch.

## 15.5 Artifact contracts

Every report artifact should begin with a concise parameter manifest listing:

- model family,
- ansatz family,
- drive enabled or disabled,
- $t$, $U$, $\omega_0$, $g$, and $n_{\mathrm{ph,max}}$,
- any propagation parameters needed to reproduce the result.

## 15.6 HH seed-surface preset and resource proxies

A seed-surface comparison is naturally evaluated not only by energy but also by simple resource proxies such as
$$
C_{\mathrm{2q}},
\qquad
D_{\mathrm{proxy}},
\qquad
C_{\mathrm{1q}}.
$$
One can then compare improvement per marginal resource,
$$
\frac{\Delta E}{C_{\mathrm{2q}}},
\qquad
\frac{\Delta E}{D_{\mathrm{proxy}}}.
$$

# 16. Noise Validation, Symmetry-Mitigation, and Parity Contracts

## 16.1 Filtered versus full exact targets

The filtered target is
$$
E_{\mathrm{exact,filtered}} = \min_{\psi\in\mathcal H_{\mathrm{sector}}} \langle \psi|\hat H|\psi\rangle,
$$
while the unrestricted target is
$$
E_{\mathrm{exact,full}} = \min \operatorname{spec}(\hat H).
$$
The two coincide only if the physical and computational sectors match exactly.

## 16.2 Noisy-minus-ideal observable deltas

For any observable $\mathcal O(t)$, define
$$
\Delta \mathcal O(t) = \mathcal O_{\mathrm{noisy}}(t)-\mathcal O_{\mathrm{ideal}}(t).
$$
In particular,
$$
\Delta E(t)=E_{\mathrm{noisy}}(t)-E_{\mathrm{ideal}}(t),
\qquad
\Delta D(t)=D_{\mathrm{noisy}}(t)-D_{\mathrm{ideal}}(t).
$$

## 16.3 Anchor parity gate

If a new idealized path is compared against a fixed anchor trajectory, a strict parity gate takes the form
$$
\max_{t\in\mathcal T}\,|\mathcal O_{\mathrm{new}}(t)-\mathcal O_{\mathrm{anchor}}(t)| \le \varepsilon_{\mathrm{parity}}.
$$
This is an anchor-based equality test, not merely a qualitative agreement statement.

## 16.4 Same-anchor paired comparison semantics

Two noisy methods should be compared against the same ideal anchor, on the same time grid, with the same initial state, so that differences can be attributed to the noise or mitigation model rather than to different starting conditions.

## 16.5 Symmetry-mitigation mode surface

Three increasingly strong symmetry surfaces are:

1. verify-only sector diagnostics,
2. bitstring postselection,
3. projector-renormalized estimators.

If $\Pi$ is the projector onto the target symmetry sector, the renormalized estimator is
$$
\langle O\rangle_{\Pi} = \frac{\langle \psi|\Pi O \Pi|\psi\rangle}{\langle \psi|\Pi|\psi\rangle}.
$$

## 16.6 Parameter manifest and provenance digest

Even in a symbolic setting, it is useful to view each artifact as carrying a manifest $\mathfrak M$ and a provenance digest $h(\mathfrak M)$. The purpose is not algebraic; it is to guarantee that the reported mathematics can be traced back to a unique physical parameter set.

# 17. Projective McLachlan Real-Time Geometry and Piecewise Beam-Adaptive Control

Below is the cleanest mathematical story for what the projected real-time implementation is doing. It is most naturally written as a **piecewise-smooth control problem on a family of variational manifolds**, with $\hbar=1$, normalized pure states, and real parameters. The two geometric facts underneath everything are:

1. the McLachlan principle projects the exact Schrödinger velocity onto a **real-linear** tangent space using the real inner product $g(u,v)=\Re\langle u,v\rangle$, and
2. physical pure states are rays, so the natural state space is **projective Hilbert space**, which means global-phase directions should be removed from the local scoring geometry.

## 17.1 What problem is being solved?

Fix a Hilbert space $\mathcal H\cong\mathbb C^d$, a Hamiltonian $H(t)$, and a current ansatz structure $\mathcal S$. That structure defines a parameterized family of states
$$
|\psi(\theta;\mathcal S)\rangle,
\qquad
\theta\in\mathbb R^m.
$$

At time $t$, the exact Schrödinger equation asks for the state velocity
$$
\frac{d}{dt}|\psi_{\mathrm{ex}}(t)\rangle
=
-iH(t)|\psi_{\mathrm{ex}}(t)\rangle.
$$

The local projected-dynamics question is therefore:

> **At the current state, how closely can the current ansatz manifold reproduce the exact instantaneous quantum velocity?**

This is the local question answered by real-time McLachlan dynamics. It is also the quantity that drives adaptive manifold growth: enlarge the ansatz when the instantaneous projected defect becomes too large.

## 17.2 Why projective Hilbert space is the right setting

A normalized ket $|\psi\rangle$ and $e^{i\phi}|\psi\rangle$ represent the same physical state. So the physical state space is projective Hilbert space $\mathbb P(\mathcal H)$ rather than the unit sphere itself. At a normalized state $|\psi\rangle$, define the horizontal projector
$$
Q_\psi:=I-|\psi\rangle\langle\psi|.
$$

For each parameter $\theta_j$, the raw tangent vector is
$$
\tau_j:=\partial_{\theta_j}|\psi(\theta;\mathcal S)\rangle,
$$
and the horizontal tangent is
$$
\bar\tau_j:=Q_\psi\tau_j.
$$

Collect the horizontal tangents as columns:
$$
\bar T:=[\bar\tau_1,\dots,\bar\tau_m]\in\mathbb C^{d\times m}.
$$

The exact Schrödinger drift is
$$
b:=-iH\psi,
$$
but the physically relevant local velocity is its horizontal part
$$
\bar b:=Q_\psi b
=
-i(H-E_\psi)\psi,
\qquad
E_\psi:=\langle\psi|H|\psi\rangle.
$$

Its squared norm is
$$
\|\bar b\|^2
=
\langle\psi|(H-E_\psi)^2|\psi\rangle
=
\operatorname{Var}_\psi(H).
$$

So the squared norm of the exact projective Schrödinger velocity is the energy variance. This is why $\operatorname{Var}_\psi(H)$ is the natural normalization scale for projected McLachlan defects.

## 17.3 The current manifold and its allowed velocities

Because the parameters are real, the attainable instantaneous velocities form the **real-linear** tangent cone
$$
T_\psi\mathcal M_{\mathcal S}
=
\left\{
\bar T\dot\theta:\dot\theta\in\mathbb R^m
\right\}.
$$

This is not a complex-linear subspace in general. So the correct local geometry is built from the real inner product
$$
g(u,v):=\Re\langle u,v\rangle.
$$

The variational velocity associated with parameter rates $\dot\theta$ is
$$
v(\dot\theta)=\bar T\dot\theta.
$$

## 17.4 The fixed-structure McLachlan step

The fixed-structure local problem is
$$
\dot\theta_0=\arg\min_{\dot\theta\in\mathbb R^m}\|\bar T\dot\theta-\bar b\|^2.
$$
Its quadratic form, projective metric, force vector, and solved stationarity system close to
$$
\begin{aligned}
\|\bar T\dot\theta-\bar b\|^2
&=\|\bar b\|^2-2\Re\langle \bar T\dot\theta,\bar b\rangle+\dot\theta^\top\Re(\bar T^\dagger\bar T)\dot\theta,\\
(\bar G)_{ij}&=\Re\langle \bar\tau_i,\bar\tau_j\rangle=\Re\langle \partial_{\theta_i}\psi|Q_\psi|\partial_{\theta_j}\psi\rangle,\\
\bar f_i&=\Re\langle \bar\tau_i,\bar b\rangle=\Re\langle Q_\psi\partial_{\theta_i}\psi,-i(H-E_\psi)|\psi\rangle,\\
\mathcal L_0(\dot\theta)&=\|\bar b\|^2-2\bar f^\top\dot\theta+\dot\theta^\top\bar G\dot\theta,\\
\bar G\dot\theta_0&=\bar f,\\
\dot\theta_0&=\bar G^+\bar f\qquad\text{if }\bar G\text{ is singular or rank-deficient},\\
0&=\Re\langle \bar\tau_i,\bar T\dot\theta_0-\bar b\rangle\qquad \forall i.
\end{aligned}
$$

## 17.5 What regularization means

A stabilized runtime step solves
$$
\dot\theta_\lambda=\arg\min_{\dot\theta\in\mathbb R^m}\left(\|\bar T\dot\theta-\bar b\|^2+\dot\theta^\top\Lambda\dot\theta\right),
\qquad
\Lambda\succeq 0.
$$
With
$$
K:=\bar G+\Lambda,
$$
the regularized geometry closes to
$$
\begin{aligned}
K\dot\theta_\lambda&=\bar f,\\
\epsilon_{\mathrm{proj}}^2&:=\min_{\dot\theta}\|\bar T\dot\theta-\bar b\|^2=\|\bar b\|^2-\bar f^\top\bar G^+\bar f,\\
\epsilon_{\mathrm{step}}^2&:=\|\bar T\dot\theta_\lambda-\bar b\|^2,\\
\mathcal L_\lambda^\star&:=\min_{\dot\theta}\left(\|\bar T\dot\theta-\bar b\|^2+\dot\theta^\top\Lambda\dot\theta\right)=\|\bar b\|^2-\bar f^\top K^{-1}\bar f\qquad (K\text{ invertible}),\\
\mathcal L_\lambda^\star&=\epsilon_{\mathrm{step}}^2+\dot\theta_\lambda^\top\Lambda\dot\theta_\lambda.
\end{aligned}
$$
So $\epsilon_{\mathrm{proj}}^2$ measures intrinsic model inadequacy, $\epsilon_{\mathrm{step}}^2$ measures the actual mismatch of the damped runtime step, and $\mathcal L_\lambda^\star$ is the stabilized solver objective.

## 17.6 Why the residual is the right local error quantity

Choose a local gauge in which the representatives are horizontal at the current time. Then the exact and variational projective velocities are
$$
\bar b
\qquad\text{and}\qquad
\bar T\dot\theta.
$$

Define the horizontal residual
$$
\bar r:=\bar T\dot\theta-\bar b.
$$

Then, for a short step $h$,
$$
|\psi_{\mathrm{ex}}(t+h)\rangle
=
|\psi\rangle+h\,\bar b+O(h^2),
$$
$$
|\psi_{\mathrm{var}}(t+h)\rangle
=
|\psi\rangle+h\,\bar T\dot\theta+O(h^2),
$$
so
$$
|\psi_{\mathrm{var}}(t+h)\rangle-|\psi_{\mathrm{ex}}(t+h)\rangle
=
h\,\bar r+O(h^2).
$$

So the correct local score is an **instantaneous derivative-mismatch score**. Static energy scores or novelty scores can help in secondary ranking, but they are not the primary real-time control quantity.

## 17.7 The geometric projector behind the formulas

Because the tangent space is real-linear, the correct projector is
$$
P_{\bar T}^{\mathbb R}v
=
\bar T\,\bar G^+\,\Re(\bar T^\dagger v).
$$

The unregularized optimal residual is therefore
$$
\bar r_0
=
\bar T\dot\theta_0-\bar b
=
\left(P_{\bar T}^{\mathbb R}-I\right)\bar b.
$$

So the ansatz follows the component of the exact projective Schrödinger vector field that lies in the current real tangent plane, and the residual is the orthogonal complement.

## 17.8 What a candidate generator really is

Suppose the current structure is an ordered product ansatz
$$
|\psi(\theta;\mathcal S)\rangle
=
e^{-i\theta_m A_m}\cdots e^{-i\theta_1 A_1}|\phi_0\rangle.
$$

Now insert a candidate block $A_c$ at position $p$, with new parameter $\eta$, initialized at zero:
$$
|\psi'(\theta,\eta)\rangle
=
e^{-i\theta_m A_m}\cdots e^{-i\theta_p A_p}
e^{-i\eta A_c}
e^{-i\theta_{p-1}A_{p-1}}\cdots e^{-i\theta_1 A_1}|\phi_0\rangle.
$$

At $\eta=0$,
$$
|\psi'(\theta,0)\rangle=|\psi(\theta)\rangle.
$$
So the state is continuous, but the tangent space changes immediately. The new tangent column is
$$
u_c:=\left.\partial_\eta|\psi'(\theta,\eta)\rangle\right|_{\eta=0},
\qquad
\bar u_c:=Q_\psi\nu_c.
$$
So a candidate is not merely “a promising operator.” It is a **new tangent direction that becomes available at the current state without moving the state**. For a block candidate carrying $q$ new real parameters, collect the new columns into
$$
\bar U\in\mathbb C^{d\times q}.
$$

## 17.9 The exact local gain formula

Now augment the local problem:
$$
\min_{\dot\theta,\eta}\Big(\|\bar T\dot\theta+\bar U\eta-\bar b\|^2+\dot\theta^\top\Lambda\dot\theta+\eta^\top\Lambda_c\eta\Big),
\qquad
\Lambda_c\succeq 0.
$$
With the mixed overlaps
$$
\begin{aligned}
B_{ia}&=\Re\langle \bar\tau_i,\bar u_a\rangle,&
B&=\Re(\bar T^\dagger\bar U)\in\mathbb R^{m\times q},\\
C_{ab}&=\Re\langle \bar u_a,\bar u_b\rangle,&
C&=\Re(\bar U^\dagger\bar U)\in\mathbb R^{q\times q},\\
q_a&=\Re\langle \bar u_a,\bar b\rangle,&
q&=\Re(\bar U^\dagger\bar b)\in\mathbb R^q,
\end{aligned}
$$
the augmented normal equations, Schur complement, and gain reduce to
$$
\begin{aligned}
\begin{bmatrix}K&B\\B^\top&C+\Lambda_c\end{bmatrix}\begin{bmatrix}\dot\theta\\\eta\end{bmatrix}&=\begin{bmatrix}\bar f\\q\end{bmatrix},\\
\dot\theta(\eta)&=K^{-1}(\bar f-B\eta),\\
\mathcal L_{\mathrm{aug},\lambda}^\star(\eta)&=\mathcal L_\lambda^\star-2\eta^\top w+\eta^\top S\eta,\\
S&=C+\Lambda_c-B^\top K^{-1}B,\\
w&=q-B^\top K^{-1}\bar f,\\
\eta_\star&=S^+w,\\
\Delta_\lambda&:=\mathcal L_\lambda^\star-\mathcal L_{\mathrm{aug},\lambda}^\star=w^\top S^+w\\
&=\bigl(q-B^\top K^{-1}\bar f\bigr)^\top\bigl(C+\Lambda_c-B^\top K^{-1}B\bigr)^+\bigl(q-B^\top K^{-1}\bar f\bigr).
\end{aligned}
$$
For a single real candidate direction ($q=1$), this reduces to
$$
\Delta_\lambda=\frac{w^2}{S}.
$$

## 17.10 What $S$ and $w$ mean

The Schur-complement pair and resulting score may be read in one chain:
$$
\begin{aligned}
S&=C+\Lambda_c-B^\top K^{-1}B,\\
w&=q-B^\top K^{-1}\bar f,\\
\Delta_\lambda&=w^\top S^+w\\
&=\bigl(q-B^\top K^{-1}\bar f\bigr)^\top\bigl(C+\Lambda_c-B^\top K^{-1}B\bigr)^+\bigl(q-B^\top K^{-1}\bar f\bigr).
\end{aligned}
$$
So $S$ is the metric of the new block after removing the part already explainable by the old tangent space, $w$ is the residualized overlap with the exact projective drift, and $\Delta_\lambda$ is residualized force squared divided by residualized norm.

## 17.11 The unregularized geometric meaning

Set $\Lambda=\Lambda_c=0$. For a single real candidate direction $\bar u$, define
$$
\bar u_\perp:=(I-P_{\bar T}^{\mathbb R})\bar u,
\qquad
\bar r_0:=(I-P_{\bar T}^{\mathbb R})\bar b.
$$
Then the exact one-direction gain closes to
$$
\begin{aligned}
\Delta_0
&=\frac{\bigl(\Re\langle \bar u_\perp,\bar r_0\rangle\bigr)^2}{\|\bar u_\perp\|^2}\\
&=\frac{\bigl(\Re\langle (I-P_{\bar T}^{\mathbb R})\bar u,(I-P_{\bar T}^{\mathbb R})\bar b\rangle\bigr)^2}{\|(I-P_{\bar T}^{\mathbb R})\bar u\|^2}.
\end{aligned}
$$
The numerator uses $\Re\langle \bar u_\perp,\bar r_0\rangle$ rather than $|\langle \bar u_\perp,\bar r_0\rangle|$ because one is adding **one real parameter**, not an arbitrary complex tangent line.

## 17.12 When should the ansatz adapt?

The intrinsic normalized miss closes to
$$
\begin{aligned}
\rho_{\mathrm{miss}}
&:=\frac{\epsilon_{\mathrm{proj}}^2}{\|\bar b\|^2+\varepsilon}
=\frac{\|\bar b\|^2-\bar f^\top\bar G^+\bar f}{\|\bar b\|^2+\varepsilon}\\
&=\frac{\operatorname{Var}_\psi(H)-\bar f^\top\bar G^+\bar f}{\operatorname{Var}_\psi(H)+\varepsilon}
=\frac{\langle \psi|(H-E_\psi)^2|\psi\rangle-\bar f^\top\bar G^+\bar f}{\langle \psi|(H-E_\psi)^2|\psi\rangle+\varepsilon}.
\end{aligned}
$$
Interpretation:

- $\rho_{\mathrm{miss}}\approx 0$: the current manifold already captures the projective velocity well,
- $\rho_{\mathrm{miss}}$ large: the manifold is missing important directions.

A clean rule is therefore
$$
\text{adapt if }\rho_{\mathrm{miss}}>\tau_{\mathrm{miss}}.
$$

One may also adapt for numerical unreliability or after a maximum interval without any structural update.

## 17.13 Which candidates should be admissible?

A candidate may have nonzero gain and still be numerically poor because it is nearly redundant. So before ranking by $\Delta_\lambda$, one should inspect the residualized geometry:

- compute singular values of $S$, or of the corresponding whitened residualized block,
- reject candidates whose retained support is too ill-conditioned.

An invariant way to do this is to whiten the old tangent metric. If
$$
\bar G\approx V\Sigma^2V^\top
$$
on its numerical support, define whitened coordinates
$$
z:=\Sigma V^\top\dot\theta.
$$

Then penalizing $\|z\|^2$ is a penalty on **physical tangent speed** rather than coordinate speed. This is the least chart-dependent way to stabilize an overparameterized manifold.

## 17.14 Two different residuals should be tracked

The projective residual is
$$
\bar r:=\bar T\dot\theta_\lambda-\bar b.
$$

This is the correct gauge-invariant quantity for local candidate scoring and local manifold-growth decisions.

The raw Hilbert-space residual is
$$
r_{\mathrm{raw}}
:=
T\dot\theta_\lambda-(-iH\psi)
=
T\dot\theta_\lambda+iH\psi.
$$

These are related by
$$
\bar r=Q_\psi r_{\mathrm{raw}}.
$$

So:

- use $\bar r$ for **local structure growth**,
- use $r_{\mathrm{raw}}$ when a rigorous Hilbert-space accumulated-error bound is desired.

## 17.15 Why integrated residual is the right branch-pruning quantity

Let a branch be propagated on a fixed structure over a segment $[t_a,t_b]$. Let $|\psi_{\mathrm{ex}}\rangle$ be the exact state and $|\psi_{\mathrm{var}}\rangle$ the variational state with the same initial condition at $t_a$, and define
$$
e(t):=|\psi_{\mathrm{var}}(t)\rangle-|\psi_{\mathrm{ex}}(t)\rangle.
$$
Then the error evolution, Duhamel representation, norm bound, and segment cost close to
$$
\begin{aligned}
\dot\psi_{\mathrm{ex}}&=-iH\psi_{\mathrm{ex}},&
\dot\psi_{\mathrm{var}}&=T\dot\theta_\lambda,&
r_{\mathrm{raw}}&=T\dot\theta_\lambda+iH\psi,\\
\dot e&=-iHe+r_{\mathrm{raw}},\\
e(t)&=\int_{t_a}^t U(t,s)\,r_{\mathrm{raw}}(s)\,ds,\\
\|e(t)\|&\le\int_{t_a}^t\|r_{\mathrm{raw}}(s)\|\,ds\le\sqrt{t-t_a}\left(\int_{t_a}^t\|r_{\mathrm{raw}}(s)\|^2\,ds\right)^{1/2},\\
J_{\mathrm{seg}}^{\mathrm{raw}}&:=\int_{t_a}^{t_b}\|r_{\mathrm{raw}}(t)\|^2\,dt=\int_{t_a}^{t_b}\|T\dot\theta_\lambda(t)+iH(t)\psi(t)\|^2\,dt.
\end{aligned}
$$
So $J_{\mathrm{seg}}^{\mathrm{raw}}$ is a natural upper-bound surrogate for accumulated state-vector error. If one wants a fully gauge-invariant pruning metric instead, replace $\|r_{\mathrm{raw}}(t)\|^2$ by $\|\bar r(t)\|^2=\|Q_\psi r_{\mathrm{raw}}(t)\|^2$ and interpret the result as a ray/projective error surrogate.

## 17.16 The branch objective

A branch is a sequence of structures $\mathcal S_0,\mathcal S_1,\dots,\mathcal S_J$ used over consecutive checkpoint segments, and the cumulative objective closes as
$$
\begin{aligned}
J_{\mathrm{branch}}
&=
\sum_{j=0}^{J-1}J_{\mathrm{seg},j}
+\beta\,C_{\mathrm{hw}}
+\gamma\,N_{\mathrm{adapt}}
+\mu_E\sum_{j=0}^{J-1}\int_{t_j}^{t_{j+1}}
\left|
\frac{d}{dt}\langle H(t)\rangle_j-\langle\partial_t H(t)\rangle_j
\right|^2dt\\
&=
\sum_{j=0}^{J-1}\int_{t_j}^{t_{j+1}}\|r_{\mathrm{raw}}^{(j)}(t)\|^2\,dt
+\beta\,C_{\mathrm{hw}}
+\gamma\,N_{\mathrm{adapt}}
+\mu_E\sum_{j=0}^{J-1}\int_{t_j}^{t_{j+1}}
\left|
\frac{d}{dt}\langle H(t)\rangle_j-\langle\partial_t H(t)\rangle_j
\right|^2dt\\
&=
\sum_{j=0}^{J-1}\int_{t_j}^{t_{j+1}}
\bigl\|T^{(j)}(t)\dot\theta_\lambda^{(j)}(t)+iH(t)\psi^{(j)}(t)\bigr\|^2\,dt
+\beta\,C_{\mathrm{hw}}
+\gamma\,N_{\mathrm{adapt}}
+\mu_E\sum_{j=0}^{J-1}\int_{t_j}^{t_{j+1}}
\left|
\frac{d}{dt}\langle H(t)\rangle_j-\langle\partial_t H(t)\rangle_j
\right|^2dt.
\end{aligned}
$$
If one wants a fully gauge-invariant branch surrogate instead, replace $\|r_{\mathrm{raw}}^{(j)}(t)\|^2$ by $\|\bar r^{(j)}(t)\|^2=\|Q_{\psi^{(j)}}r_{\mathrm{raw}}^{(j)}(t)\|^2$.

## 17.17 The full process, in one coherent sequence

At each time $t$, with current structure $\mathcal S$, the full loop can be written compactly as
$$
\begin{aligned}
\text{A: }&
|\psi\rangle=|\psi(\theta;\mathcal S)\rangle,
\qquad
Q_\psi=I-|\psi\rangle\langle\psi|,
\qquad
\bar T=Q_\psi T,
\qquad
\bar b=Q_\psi(-iH\psi)=-i(H-E_\psi)\psi,\\
\text{B: }&
\bar G=\Re(\bar T^\dagger\bar T),
\qquad
\bar f=\Re(\bar T^\dagger\bar b),
\qquad
K=\bar G+\Lambda,
\qquad
\dot\theta_\lambda=K^+\bar f,\\
&
\epsilon_{\mathrm{proj}}^2=\|\bar b\|^2-\bar f^\top\bar G^+\bar f,
\qquad
\epsilon_{\mathrm{step}}^2=\|\bar T\dot\theta_\lambda-\bar b\|^2,
\qquad
\bar r=\bar T\dot\theta_\lambda-\bar b,
\qquad
r_{\mathrm{raw}}=T\dot\theta_\lambda+iH\psi,\\
\text{C: }&
\rho_{\mathrm{miss}}
=
\frac{\epsilon_{\mathrm{proj}}^2}{\|\bar b\|^2+\varepsilon}
=
\frac{\epsilon_{\mathrm{proj}}^2}{\operatorname{Var}_\psi(H)+\varepsilon},
\qquad
\text{adapt iff }\rho_{\mathrm{miss}}>\tau_{\mathrm{miss}},\\
\text{D: }&
B_c=\Re(\bar T^\dagger\bar U_c),
\qquad
C_c=\Re(\bar U_c^\dagger\bar U_c),
\qquad
q_c=\Re(\bar U_c^\dagger\bar b),\\
&
S_c=C_c+\Lambda_c-B_c^\top K^+B_c,
\qquad
w_c=q_c-B_c^\top K^+\bar f,
\qquad
\Delta_{\lambda,c}=w_c^\top S_c^+w_c.
\end{aligned}
$$
For each shortlisted admissible candidate $c$, one then creates a zero-initialized child, propagates it over the next probe segment, and prunes on the accumulated branch objective:
$$
\begin{aligned}
\text{E: }&
\mathcal S\mapsto\mathcal S_c^+,
\qquad
|\psi(\theta,0;\mathcal S_c^+)\rangle=|\psi(\theta;\mathcal S)\rangle,\\
\text{F: }&
J_{\mathrm{seg},c}^{\mathrm{raw}}
=
\int_{t_a}^{t_b}\|r_{\mathrm{raw},c}(t)\|^2\,dt
=
\int_{t_a}^{t_b}\bigl\|T_c(t)\dot\theta_{\lambda,c}(t)+iH(t)\psi_c(t)\bigr\|^2\,dt,\\
\text{G: }&
J_{\mathrm{branch},c}^{\mathrm{new}}
=
J_{\mathrm{branch}}^{\mathrm{old}}
+
J_{\mathrm{seg},c}^{\mathrm{raw}}
+
\beta\,C_{\mathrm{hw},c}
+
\gamma\,N_{\mathrm{adapt},c},
\qquad
\mathfrak B_{\mathrm{next}}
=
\operatorname{Prune}\!\left(\{(\mathcal S_c^+,J_{\mathrm{branch},c}^{\mathrm{new}})\}_c\right).
\end{aligned}
$$
That is the whole piecewise-manifold control loop.

## 17.18 One concise interpretation

The method is doing this:

> The exact quantum dynamics asks for a projective velocity $\bar b$.  
> The current ansatz only provides a real tangent plane.  
> McLachlan chooses the closest available tangent velocity.  
> If the miss is too large, the algorithm enlarges the tangent plane by adding zero-initialized candidate directions.  
> The local gain formula tells us which candidate most reduces the current miss.  
> Short probe rollouts then decide which structural choices remain best after actual propagation.

## 17.19 Explicit symbolic--mathematical differences between the clean story and the current implementation

The clean story above is the right mathematical idealization. The current implementation differs from that idealization in a few explicit ways:

1. **Sign convention.** The clean story uses
   $$
   \bar b=-iQ_\psi H\psi,
   \qquad
   \bar r=\bar T\dot\theta-\bar b.
   $$
   The implementation instead uses
   $$
   r=\bar T\dot\theta+iQ_\psi H\psi,
   \qquad
   f_i=\Im\langle \bar\tau_i|H|\psi\rangle,
   \qquad
   \Re\langle \bar\tau_i,-iH\psi\rangle=\Im\langle \bar\tau_i,H\psi\rangle.
   $$

2. **Regularization surface.** Rather than a general positive-semidefinite penalty $\Lambda$ or $\Lambda_c$, the implementation uses a scalar $\lambda\ge 0$ on either chart coordinates or whitened metric coordinates:
   $$
   \begin{aligned}
   K_{\mathrm{chart}}&=\bar G+\lambda I,\\
   \bar G&\approx V\Sigma^2V^\top,
   \qquad
   z=\Sigma V^\top\dot\theta,
   \qquad
   (I+\lambda I)z=\Sigma^{-1}V^\top\bar f,
   \qquad
   \dot\theta=V\Sigma^{-1}z.
   \end{aligned}
   $$

3. **Propagation solve versus policy solve.** The clean story presents one stabilized local solve, whereas the implementation may separate the propagated step from the policy/telemetry step:
   $$
   (\dot\theta_{\mathrm{prop}},\epsilon_{\mathrm{step}}^{2,\mathrm{prop}})
   \neq
   (\dot\theta_{\mathrm{policy}},\epsilon_{\mathrm{step}}^{2,\mathrm{policy}})
   \quad\text{in general},
   $$
   even when both are derived from the same local geometry.

4. **Variance normalization and guards.** The clean story writes
   $$
   \rho_{\mathrm{miss}}=\frac{\epsilon_{\mathrm{proj}}^2}{\operatorname{Var}_\psi(H)+\varepsilon}.
   $$
   The implemented ratio surface is closer to
   $$
   \rho_{\mathrm{miss}}
   =
   \begin{cases}
   \epsilon_{\mathrm{proj}}^2/(\operatorname{Var}_\psi(H)+\varepsilon),&\operatorname{Var}_\psi(H)\ge \nu_{\min},\\
   \text{suppressed},&\operatorname{Var}_\psi(H)<\nu_{\min},
   \end{cases}
   $$
   with an auxiliary guard flag recording the suppressed regime.

5. **Active-parameter windows.** The clean story is written on the full tangent space, but the implementation may restrict to an active suffix or oracle-supplied index set:
   $$
   T\mapsto T_{W_{\mathrm{act}}},
   \qquad
   \dot\theta\mapsto(\dot\theta_{W_{\mathrm{act}}},0),
   \qquad
   \bar G\mapsto \bar G_{W_{\mathrm{act}}W_{\mathrm{act}}},
   $$
   so inactive coordinates are frozen.

6. **Candidate gain evaluation.** The clean story scores candidates by the Schur-complement gain
   $$
   \Delta_\lambda=w^\top S^+w.
   $$
   The implementation instead zero-initializes one append candidate, rebuilds the augmented local geometry, and scores by direct defect drops
   $$
   \Delta_0=\epsilon_{\mathrm{proj,current}}^2-\epsilon_{\mathrm{proj,aug}}^2,
   \qquad
   \Delta_\lambda=\epsilon_{\mathrm{step,current}}^2-\epsilon_{\mathrm{step,aug}}^2.
   $$

7. **Position-aware current branch growth.** The clean story allows arbitrary insertion positions and multi-parameter blocks. The helper layer now supports position-aware insertion and position-centered post-insert refit windows,
   $$
   \mathcal T(\mathfrak b,m,p)=\mathfrak b',
   \qquad
   \mathcal O_{b'}=\mathcal O_b\oplus_p m,
   \qquad
   \theta_{b'}=\theta_b\oplus_p 0,
   $$
   and the active probe set is built from the append slot, the left boundary, and the active reoptimization window. Hence, when the active window is made maximally large and the probe-position cap is also made large enough, the implementation can evaluate the full insertion set $p\in\{0,\dots,|\theta|\}$ rather than append only. The live surface still remains zero-initialized **single-parameter** growth, but not append-only growth.

8. **Checkpoint-trigger logic.** The clean story says “adapt when $\rho_{\mathrm{miss}}$ is too large.” The implemented trigger surface is richer:
   $$
   \mathbf 1_{\mathrm{checkpoint}}
   =
   \mathbf 1_{\mathrm{cadence}}
   \lor
   \mathbf 1\!\bigl(\epsilon_{\mathrm{proj}}^2>\tau_{\mathrm{proj}}\bigr)
   \lor
   \mathbf 1\!\bigl(\epsilon_{\mathrm{step}}^2>\tau_{\mathrm{step}}\bigr)
   \lor
   \mathbf 1\!\bigl(\rho_{\mathrm{miss}}>\tau_{\mathrm{miss}}\text{ when defined}\bigr),
   $$
   together with a model-versus-damping failure classification.

9. **Branch objective surface.** The clean story uses a scalar objective such as
   $$
   J_{\mathrm{scalar}}=\sum J_{\mathrm{seg}}+\beta C_{\mathrm{hw}}+\gamma N_{\mathrm{adapt}}.
   $$
   The implementation instead prunes by the lexicographic key
   $$
   \kappa_{\mathrm{prune}}
   =
   \bigl(J^{\mathrm{rt}},J^{\mathrm{res}},J^{\mathrm{model}},J^{\mathrm{damp}},J^{\mathrm{cons}},|\mathcal O|,k,\operatorname{labels}(\mathcal O),\operatorname{round}_{10}(\theta),\operatorname{id}\bigr),
   $$
   after deduplicating by time index, selected-operator labels, and rounded parameter values.

10. **Projective versus raw residual accumulation.** The clean story distinguishes a projective residual $\bar r$ from a raw Hilbert-space residual $r_{\mathrm{raw}}$. The implemented rollout telemetry is closer to
   $$
   \bigl(\Delta J^{\epsilon},\Delta J^{\mathrm{model}},\Delta J^{\mathrm{damp}}\bigr)
   =
   \left(
   \int\epsilon_{\mathrm{step}}^2,
   \int\epsilon_{\mathrm{proj}}^2,
   \int\max\!\bigl(\epsilon_{\mathrm{step}}^2-\epsilon_{\mathrm{proj}}^2,0\bigr)
   \right),
   $$
   rather than a separately stored raw Hilbert-space residual bound.

11. **Geometry source.** The clean story is written as if the tangent vectors and $H\psi$ are assembled directly every time. The implementation may instead start from externally supplied local data:
   $$
   (\bar G,\bar f,\operatorname{Var}_\psi(H))
   =
   \mathcal O_{\mathrm{geom}}(\psi,H,\mathcal S),
   $$
   after which the host solves the McLachlan system on that returned geometry.

So the mathematical backbone is the same, but the implemented restricted surface closes schematically to
$$
\begin{aligned}
\text{projective policy geometry}
&+\text{ scalar chart/whitened damping}
+\text{ append-only zero-init growth}\\
&+\text{ checkpointed short-horizon rollout}
+\text{ lexicographic prune key}
+\text{ optional oracle-supplied local geometry}.
\end{aligned}
$$

# 18. Final Primitive-Closed Summary

The symbolic substitution chain is linear and closed:

1. choose the fermion ordering $p(i,\sigma)$,
2. write the Jordan-Wigner ladder primitives $\hat c_p^\dagger$ and $\hat c_p$,
3. reduce number operators to $(I-Z_p)/2$,
4. substitute them into $\hat H_t$, $\hat H_U$, and $\hat H_v$,
5. write $\hat b_i$, $\hat b_i^\dagger$, $\hat n_{b,i}$, $\hat x_i$, and $\hat P_i$,
6. substitute them into $\hat H_{\mathrm{ph}}$ and $\hat H_g$,
7. reduce the drive term to explicit identity-plus-$Z$ coefficients,
8. propagate those operators into statevector primitives,
9. build fixed and adaptive variational manifolds from the same operators,
10. define projective McLachlan tangent geometry on those manifolds,
11. define branch growth, probe rollout, and beam pruning on top of the local defect geometry,
12. define continuation and handoff on top of those manifolds, with historical replay paths kept in the appendix,
13. define benchmark and noise-validation contracts relative to the same exact targets.

This is the point of the symbolic duplicate: the same mathematical skeleton remains, but implementation labels are stripped away so the document can serve as a clean theory-facing reference.

# 19. Data

This final chapter records empirical evidence for the selector and pruning claims developed in the main body. The goal is not to replace the symbolic formulas above, but to show that the live Phase-3 HH selector surfaces described in §§11 and 17 materially change the observed outcomes on matched runs.

## 19.1 Matched novelty ablation

The freshest clean ablation available is a matched four-run HH study with
$$
L=2,\qquad t=1,\qquad U=4,\qquad \omega_0=1,\qquad g=0.5,\qquad n_{\mathrm{ph,max}}=1,
$$
using binary bosons, blocked ordering, open boundaries, the same seed, the same compiled backend path, the same windowed reoptimization policy, the same live pruning machinery, and the same rescue/symmetry/lifetime settings. The only changes are:

1. operator pool: `pareto_lean_l2` versus `full_meta`,
2. novelty exponent: $\gamma_N=1$ versus $\gamma_N=0$.

Writing the final filtered-energy error as
$$
\lvert \Delta E \rvert = \bigl|E_{\mathrm{final}}-E_{\mathrm{exact,filtered}}\bigr|,
$$
the matched runs give:

| Pool | $\gamma_N$ | $E_{\mathrm{final}}$ | $\lvert \Delta E \rvert$ | Depth | Parameters | Stop reason | Prune kept |
|---|---:|---:|---:|---:|---:|---|---:|
| `pareto_lean_l2` | `1.0` | `0.15872408236037028` | `5.6178234644 \times 10^{-5}` | `14` | `25` | `drop_plateau` | `4 / 5` |
| `pareto_lean_l2` | `0.0` | `0.48696021436469644` | `3.2829231024 \times 10^{-1}` | `14` | `23` | `drop_plateau` | `4 / 5` |
| `full_meta` | `1.0` | `0.15872408236037047` | `5.6178234644 \times 10^{-5}` | `13` | `28` | `drop_plateau` | `5 / 5` |
| `full_meta` | `0.0` | `0.48699090951182056` | `3.2832300539 \times 10^{-1}` | `13` | `22` | `drop_plateau` | `5 / 5` |

If
$$
R_{\mathrm{nov}}(\mathcal G)
=
\frac{\lvert \Delta E \rvert_{\gamma_N=0,\mathcal G}}{\lvert \Delta E \rvert_{\gamma_N=1,\mathcal G}},
$$
then the two matched pool ratios are
$$
R_{\mathrm{nov}}(\texttt{pareto\_lean\_l2})\approx 5.84376\times 10^3,
\qquad
R_{\mathrm{nov}}(\texttt{full\_meta})\approx 5.84431\times 10^3.
$$

So, on this matched HH benchmark, turning novelty weighting off worsens the final filtered-energy error by roughly a factor of $5.8\times 10^3$ in both a lean pool and a heavy pool.

## 19.2 Interpretation of the novelty ablation

Several points survive this comparison cleanly.

1. **Novelty-on recovers the low-error line.** In both pools, $\gamma_N=1$ reaches the same final error scale,
   $$
   \lvert \Delta E \rvert \approx 5.62\times 10^{-5}.
   $$

2. **Novelty-off collapses to a much worse terminal scaffold.** In both pools, $\gamma_N=0$ ends near
   $$
   E_{\mathrm{final}}\approx 0.487,
   \qquad
   \lvert \Delta E \rvert \approx 3.28\times 10^{-1},
   $$
   far above the filtered exact target near $0.158668$.

3. **This is not merely a depth effect.** The on/off pairs stop at the same logical depth within each pool, so the dominant difference is the operator-ranking trajectory rather than a simple “more steps” explanation.

4. **This is not merely a pruning-on versus pruning-off effect.** Pruning stays active in all four runs. What changes is the upstream ranking rule. In particular, novelty telemetry is still assembled when $\gamma_N=0$; what is ablated is the multiplicative use of novelty inside the selector score, not the underlying geometric information itself.

These runs therefore support the claim that the reduced-tangent-space novelty factor is doing real work inside the ADAPT ranking rule rather than serving as passive telemetry.

## 19.3 Pool-family redundancy

The same data story also sharpens the pool-reduction claim. For the studied HH regime
$$
L=2,\qquad n_{\mathrm{ph,max}}=1,
$$
the heavy `full_meta` pool has size $46$, while the lean `pareto_lean_l2` pool has size $11$. Hence the pool-reduction fraction is
$$
\eta_{\mathrm{pool}}
=
1-\frac{11}{46}
=
\frac{35}{46}
\approx 0.7609.
$$

Equivalently, the lean pool removes about $76\%$ of the heavy-pool candidates while still retaining the same practical final energy scale,
$$
E_{\mathrm{final}}\approx 0.15872408,
\qquad
\lvert \Delta E \rvert \approx 5.62\times 10^{-5},
$$
on the current-code lean rerun.

The generator classes discussed in this L=2 comparison may be represented symbolically by
$$
\begin{aligned}
A^{\mathrm{sing}}_{ia,\sigma}
&=
i\!\left(\hat c^\dagger_{a\sigma}\hat c_{i\sigma}-\hat c^\dagger_{i\sigma}\hat c_{a\sigma}\right),\\
A^{\mathrm{dbl}}_{ijab}
&=
i\!\left(\hat c^\dagger_a\hat c^\dagger_b\hat c_j\hat c_i-\mathrm{h.c.}\right),\\
G_i^{(\mathrm{cloud})}
&=
\tilde n_i\sum_{r\in\mathcal N(i)}\alpha_{ir}P_r,\\
G_{ij}^{(\mathrm{hd})}
&=
T_{ij}^{(+)}(P_i-P_j),\\
G_i^{(\mathrm{dd})}
&=
D_iP_i,\\
G_{ij}^{(\mathrm{od})}
&=
J_{ij}^{(-)}(x_i-x_j),\\
G_{ij}^{(2)}
&=
T_{ij}^{(+)}(x_i-x_j)^2,\\
x_i
&=
\hat b_i+\hat b_i^\dagger,
\qquad
P_i
=
i(\hat b_i^\dagger-\hat b_i).
\end{aligned}
$$
On this $L=2$ benchmark, the retained/useful classes were $A^{\mathrm{sing}}$, the $P$-type polaronic classes $G_i^{(\mathrm{cloud})}$, $G_{ij}^{(\mathrm{hd})}$, and $G_i^{(\mathrm{dd})}$, together with the quadrature seeds.
By contrast, $A^{\mathrm{dbl}}$, HVA, and the x-type / higher-order channels $G_{ij}^{(\mathrm{od})}$ and $G_{ij}^{(2)}$ were not needed to preserve the achieved error scale.

Empirically, the following heavy-pool families are absent from the lean pool yet are not needed to retain that achieved accuracy scale on this benchmark:

- lifted UCCSD doubles,
- HH unit terms,
- HVA layers,
- `paop_cloud_x`,
- `paop_disp`,
- `paop_dbl`,
- `paop_dbl_x`,
- `paop_curdrag`,
- `paop_hop2`.

This should be read as a scoped data claim, not as a universal proof: for the present HH $L=2$, $n_{\mathrm{ph,max}}=1$ target problem, the heavy pool carries substantial operator-family redundancy relative to the observed energy objective.

### 19.3.1 Weak-coupling $L=2$ class-prune redo

A second $L=2$ class-prune line at weaker coupling sharpens the same conclusion. For
$$
L=2,\qquad t=1,\qquad U=0.5,\qquad \omega_0=1,\qquad g=0.2,\qquad n_{\mathrm{ph,max}}=1,
$$
using binary bosons, blocked ordering, open boundaries, the compiled backend path, full-window reoptimization, beam width $B=3$, and the direct `phase3_v1` HH selector, the heavy parent `full_meta` run reached
$$
E_{\mathrm{parent}}=-0.7763497696348323,
\qquad
\lvert\Delta E\rvert_{\mathrm{parent}}=2.7761269927\times 10^{-5},
\qquad
F_{\mathrm{parent}}=0.9999861889450502,
$$
at logical depth $d_{\mathrm{parent}}=7$.

From that parent, a class-first pruning loop removed one operator family at a time and reran fresh ADAPT after every proposed removal. A removal was accepted only if the rerun satisfied
$$
\lvert\Delta E\rvert \le 5\times 10^{-4},
\qquad
F \ge F_{\mathrm{parent}}-10^{-4}.
$$
The accepted removals were

- `hh_termwise_unit`,
- `hva_layer`,
- `paop_cloud_x`,
- `paop_curdrag`,
- `paop_dbl`,
- `paop_dbl_p`,
- `paop_dbl_x`,
- `paop_disp`,
- `paop_hop2`,
- `paop_hopdrag`,
- `hh_termwise_quadrature`.

Hence the final validated keep-set was simply
$$
\mathcal G_{\mathrm{lean,weak}}
=
\{\texttt{uccsd\_sing},\ \texttt{uccsd\_dbl},\ \texttt{paop\_cloud\_p}\}.
$$
A fresh rerun using only this class-filtered pool then produced
$$
E_{\mathrm{lean}}=-0.7763497696348325,
\qquad
\lvert\Delta E\rvert_{\mathrm{lean}}=2.7761269927\times 10^{-5},
\qquad
F_{\mathrm{lean}}=0.9999861889437017,
$$
at logical depth $d_{\mathrm{lean}}=4$.
Thus, to numerical precision,
$$
E_{\mathrm{lean}}\approx E_{\mathrm{parent}},
\qquad
\lvert\Delta E\rvert_{\mathrm{lean}}\approx \lvert\Delta E\rvert_{\mathrm{parent}},
\qquad
F_{\mathrm{lean}}\approx F_{\mathrm{parent}},
$$
while the logical depth drops from $7$ to $4$.

A separate optimizer check on this same weak-coupling direct Phase-3 beam surface asked a narrower question: whether SPSA, still run locally/noiselessly with the pre-transpile proxy-cost surface (`phase3_backend_cost_mode=proxy`) and still using the same reduced class-filtered pool with full-window refits, could at least reach the practical noise-threshold regime even if it did not exactly recover the Powell basin. The best completed local SPSA artifact on that surface was
$$
E_{\mathrm{SPSA,local}}=-0.7763064877481185,
\qquad
\lvert\Delta E\rvert_{\mathrm{SPSA,local}}=7.1043156641\times 10^{-5},
\qquad
F_{\mathrm{SPSA,local}}=0.9999653825260001,
$$
from the direct beam-enabled class-filtered run `20260328_hh_l2_u05_g02_local_beam_b3_spsa_case4_direct_m3200_a0p1_c0p02_A5`. This did not match the Powell/class-pruned reference value $2.7761269927\times 10^{-5}$, but it did enter the sub-$10^{-4}$ regime and therefore satisfied the more practical gate of crossing below the anticipated noise floor. In that sense the local SPSA question at weak coupling was answered positively: exact-basin recovery remained unresolved, but threshold-level local recovery was achieved strongly enough to justify a direct noisy/backend follow-on on the same algorithmic surface.

A direct noisy transfer follow-on then kept the same reduced pool
$\{\texttt{uccsd\_sing},\texttt{uccsd\_dbl},\texttt{paop\_cloud\_p}\}$,
the same beam/full-window `phase3_v1` surface, and the same SPSA recipe
$(a,c,A,\texttt{maxiter})=(0.1,0.02,5.0,3200)$, but turned on
`backend_scheduled` FakeMarrakesh scoring with `expectation_v1`, `512` shots,
`1` repeat, and no mitigation; the exact command is recorded at
`artifacts/agent_runs/20260328_hh_l2_u05_g02_phase3_beam_b3_fake_marrakesh_spsa_case4_recipe_attempt3_supervised/logs/command.sh`.
During that supervised run, the best observed heartbeat reached
$$
E_{\mathrm{SPSA,noisy,best}}=-0.7762934955401648,
\qquad
\lvert\Delta E\rvert_{\mathrm{SPSA,noisy,best}}=8.4035364595\times 10^{-5},
$$
at depth $95$, as preserved in
`artifacts/agent_runs/20260328_hh_l2_u05_g02_phase3_beam_b3_fake_marrakesh_spsa_case4_recipe_attempt3_supervised/best_so_far_snapshot.json`.
So, before final max-depth termination, the same reduced-pool direct Phase-3 beam SPSA lane already crossed the practical noisy sub-$10^{-4}$ regime under transpiled FakeMarrakesh selection/scoring.

### 19.3.2 Live four-run supervised status block

```text
Objective<As of March 28, 2026 at 10:59:12 CDT, all 4 noisy
  reduced-pool backend_scheduled SPSA runs are still genuinely
  running with live child PIDs and no stderr in artifacts/
  agent_runs/20260328_hh_l2_u05_g02_phase3_beam_b3_fake_marrake
  sh_spsa_case4_recipe_attempt4_dropstop/progress.json,
  artifacts/
  agent_runs/20260328_hh_l2_u05_g02_phase3_beam_b3_fake_marrake
  sh_spsa_case4_recipe_attempt5_dropstop_lifetimeoff/
  progress.json, artifacts/
  agent_runs/20260328_hh_l2_u05_g02_phase3_beam_b4_c3_k4_fake_m
  arrakesh_spsa_case4_recipe_attempt6_dropstop_lifetimeoff/
  sh_spsa_case4_recipe_attempt7_dropstop_transpile_single/
  progress.json. Best observed noisy gaps so far are: attempt4
  3.3871153069520155e-05 at depth 76 in artifacts/
  agent_runs/20260328_hh_l2_u05_g02_phase3_beam_b3_fake_marrake
  sh_spsa_case4_recipe_attempt4_dropstop/logs/stdout.log,
  attempt6 3.733262477279009e-05 at depth 49 in artifacts/
  agent_runs/20260328_hh_l2_u05_g02_phase3_beam_b4_c3_k4_fake_m
  arrakesh_spsa_case4_recipe_attempt6_dropstop_lifetimeoff/
  logs/stdout.log, attempt7 3.899366517123859e-05 at depth 67
  in artifacts/
  agent_runs/20260328_hh_l2_u05_g02_phase3_beam_b3_fake_marrake
  sh_spsa_case4_recipe_attempt7_dropstop_transpile_single/logs/
  stdout.log, and attempt5 1.0855648531227224e-04 at depth 67
  in artifacts/
  agent_runs/20260328_hh_l2_u05_g02_phase3_beam_b3_fake_marrake
  sh_spsa_case4_recipe_attempt5_dropstop_lifetimeoff/logs/
  stdout.log.>
  Why/Intent<So the current result ordering is: baseline clean-
  stop attempt4 best, then wider-beam+lifetime-off attempt6,
  then transpile_single_v1 attempt7, with lifetime-off only
  attempt5 trailing. None has written a final JSON yet because
  the clean stop is intentionally gated by --adapt-drop-min-
  depth 96, so even though three runs are already well below
  the practical 1e-4 threshold, the normal drop_plateau exit
  cannot fire before depth 96.>
  Suggested Next step/how this fits into broader picture<I
  recommend we keep these running until the first one crosses
  depth 96 and exits normally, because that will give us the
  first clean final JSON on this surface without interrupting a
  good basin. If the ranking stays similar, attempt4 should
  finish first, while attempt6 and attempt7 already look
  scientifically competitive enough to justify keeping them
  alive as the main comparison axes.>
```

The rejected removals show which families remained indispensable under this contract:

- removing `paop_cloud_p` gave $\lvert\Delta E\rvert\approx 1.0813\times 10^{-2}$ and $F\approx 0.996632$,
- removing `uccsd_sing` gave $\lvert\Delta E\rvert\approx 2.19934$ and $F\approx 0.214233$,
- removing `uccsd_dbl` gave $\lvert\Delta E\rvert\approx 1.31029\times 10^{-2}$ and $F\approx 0.996691$.

So at this weaker-coupling $L=2$ point the validated lean pool is even smaller than the earlier heavy/lean $U=4$, $g=0.5$ line: the heavy search may still visit quadrature seeds, but the final class-level support needed to preserve the achieved energy/fidelity scale is just
$$
\texttt{uccsd\_sing}\;\cup\;\texttt{uccsd\_dbl}\;\cup\;\texttt{paop\_cloud\_p}.
$$

For the backend-facing follow-up at this same weak-coupling point, the Heron/Marrakesh optimization should be done as a **transpile-only** sweep rather than through the noisy staged wrapper: run `python -m pipelines.hardcoded.adapt_circuit_cost` on the promoted locked scaffold `hh_l2_u05_g02_full_meta_class_pruned_lean4_locked_scaffold_v1.json` over `\texttt{optimization\_level}\in\{1,2\}` and `\texttt{seed\_transpiler}\in\{0,\dots,9\}`, then rank by compiled two-qubit count, depth, and size. On `\texttt{FakeMarrakesh}` the winning setting was `\texttt{optimization\_level}=2`, `\texttt{seed\_transpiler}=5`, reducing the same 4-operator, 6-runtime-term scaffold from about `25` two-qubit gates, depth `63`, size `109` to `18` two-qubit gates, depth `43`, size `83` without changing the physics or operator order.

## 19.4 Relation to live Phase-3 selector/pruning implementation

The canonical Phase-3 target contract is the one stated in §§11.4.1--11.4.3. Current live repo code still lags that contract in two visible ways.

First, the live selector path still applies a hard uncertainty-adjusted gradient pre-cap before the cheap screen and still uses a raw remaining-depth / remaining-evaluations proxy inside the lifetime-burden term. The corrected canonical contract in §11.4.1 instead ranks the full admissible candidate-position pool with the horizon-smoothed cheap score $S_{3,\mathrm{cheap}}(r;t)$, forms a horizon-sized coarse shortlist $\mathcal C_{\mathrm{cheap}}$, and uses the useful horizon
$$
H_t=\min\!\bigl(D_{\mathrm{left}}(t),\widehat N_{\mathrm{rem}}(t)\bigr)
$$
inside the lifetime multiplier. Here $\widehat N_{\mathrm{rem}}(t)$ is controller telemetry and therefore a ranking/burden aid, not a hard admissibility theorem.

Second, the live pruning path still ranks removals primarily by small amplitude together with weaker prior proxy-benefit tie-breaks and accepts removals through a standalone post-refit regression threshold. The corrected canonical contract in §11.4.3 instead separates prune permissibility from local expendability, ranks permissible removals by frozen-ablation loss, and uses retained admitted-step gain as the principal energetic safety condition. Thus frozen ablation is to be read as a local, solver-relative expendability heuristic, while the gain-retention inequality is the actual prune safety condition.

So the honest status note is: current code still trails the canonical Phase-3 mathematics, but the intended direction of alignment is
$$
\text{live implementation}\longrightarrow \text{corrected canonical contract of §§11.4.1--11.4.3},
$$
not the reverse.

## 19.5 Data provenance note

The novelty-ablation numbers in this chapter come from the fresh direct `adapt_pipeline` artifacts `pi_ablate_pareto_lean_l2_novelty_on.json`, `pi_ablate_pareto_lean_l2_novelty_off.json`, `pi_ablate_full_meta_novelty_on.json`, and `pi_ablate_full_meta_novelty_off.json`, all dated `2026-03-24`. The heavy-versus-lean pool-family comparison is anchored by the `2026-03-21` comparison bundle summarized in `pareto_lean_l2_vs_heavy_L2_ecut1_comparison.md` together with the corresponding lean rerun artifact. The data are therefore matched and current enough for PI-facing evidence.

The honesty caveat is the same one stated in the reporting notes: these matched ablations share the same physics, seed, and selector family, but they are not claimed to be byte-for-byte reproductions of every earlier March 21/22 runtime-resolved default. That caveat does not weaken the on/off comparisons above, because the resolved settings were shared inside each matched pair.

## 19.6 Machine-agent redo note for the successful L=2 pruning session

For the validated HH prune line, keep the problem fixed at
$$
L=2,\qquad t=1,\qquad U=4,\qquad \omega_0=1,\qquad g=0.5,\qquad n_{\mathrm{ph,max}}=1,
$$
with the saved fullhorse parent artifact as the starting point. A machine agent can reproduce the session in three steps from repo root:

```bash
python -m pipelines.hardcoded.hh_prune_nighthawk \
  --input-json artifacts/json/hh_backend_adapt_fullhorse_powell_20260322T194155Z_fakenighthawk.json \
  --mode both \
  --prune-threshold 1e-4 \
  --output-json artifacts/json/hh_prune_nighthawk_<timestamp>.json

python -m pipelines.hardcoded.hh_prune_nighthawk \
  --mode export_fixed_scaffolds \
  --source-artifact-json artifacts/json/hh_prune_nighthawk_aggressive_5op.json

python -m pipelines.hardcoded.adapt_circuit_cost \
  --artifact-json artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json \
  --backend-name ibm_marrakesh \
  --seed-transpiler 7 --optimization-level 1 \
  --sweep-backends
```

The redo is not complete unless it records **fidelity as well as energy**. For each surviving scaffold and exported fixed scaffold, save at least:
`abs_delta_e`, exact-state fidelity (`exact_state_fidelity`, `local_exact_state_fidelity`, or `expected_local_exact_state_fidelity`), retained exyz labels, backend, transpiler seed, optimization level, compiled `count_2q`, compiled depth, and compiled size. The validated executable anchor is the `gate_pruned_7term` artifact, not `circuit_optimized_7term`: the trustworthy target is
$$
\lvert\Delta E\rvert\approx 4.23\times 10^{-4},\qquad F\approx 0.9998,
$$
with Marrakesh compile cost near `25` two-qubit gates and depth `63-75`. A leaner 6-term executable may reach about `14` two-qubit gates and depth `48`, but its exact regression is materially worse, near
$$
\lvert\Delta E\rvert\approx 4.61\times 10^{-3}.
$$

## 19.7 Direct noisy `backend_scheduled` L=2 fastprobe note

For the same HH point
$$
L=2,\qquad t=1,\qquad U=4,\qquad \omega_0=1,\qquad g=0.5,\qquad n_{\mathrm{ph,max}}=1,
$$
a useful data line is the direct local noisy `phase3_v1` fastprobe on `FakeNighthawk` with `backend_scheduled` oracle evaluation, `512` shots, `2` repeats, and optional `mthree` readout mitigation. The March 26 command family is recorded in `artifacts/runstate/hh_phase3_v1_local_noisy_backend_scheduled_fastprobe_retry2_20260326_cmd.sh` and the stabilized readout follow-up `artifacts/runstate/hh_phase3_v1_local_noisy_backend_scheduled_fastprobe_retry3_20260326_cmd.sh`, with the no-readout comparator in `artifacts/runstate/hh_phase3_v1_local_noisy_backend_scheduled_noreadout_retry1_20260326/`.

These probes were intentionally weak (`\texttt{adapt\_max\_depth}=1`) and were stopped after the first scout evaluation timing split was captured, so they are debug-quality runtime evidence rather than convergence-quality VQE runs. The first measured `plus` evaluation on the readout+mthree path gave
$$
T_{\mathrm{total,readout}}\approx 37.04\,\mathrm{s},
\qquad
T_{\mathrm{term}}\approx 26.39\,\mathrm{s},
\qquad
T_{\mathrm{calib}}\approx 10.64\,\mathrm{s},
$$
with readout-apply time negligible at about `0.01s`. The no-readout comparator gave
$$
T_{\mathrm{total,no\,readout}}\approx 26.84\,\mathrm{s},
\qquad
T_{\mathrm{term}}\approx 26.83\,\mathrm{s}.
$$
So on this $L=2$ HH debug line the dominant cost is backend-scheduled term execution itself, while readout mitigation contributes a secondary first-hit warmup rather than the main runtime burden. The practical lesson is that `backend_scheduled` is honest enough for a very small late shortlist or confirmation pass, but too expensive to act as the broad pre-shortlist scout surface.

## 19.8 Fixed-scaffold compile-control scout note

For the locked Marrakesh/Heron 6-term HH scaffold `artifacts/json/hh_marrakesh_fixed_scaffold_6term_drop_eyezee_20260323T171528Z.json`, the useful local compile-control test is a transpile-only scout on `FakeMarrakesh` with fixed circuit/parameters, `shots=4096`, `oracle_repeats=8`, readout/mthree mitigation, and grid $(\texttt{optimization\_level},\texttt{seed\_transpiler})\in\{1,2\}\times\{0,\dots,4\}$, ranking by $\Delta E=E_{\mathrm{noisy}}-E_{\mathrm{ideal}}$ then by compiled cost; among the saved `8/10` completed candidates in `artifacts/agent_runs/20260328_direct_fixed_scaffold_compile_scout_fullaccess_attempt2_timeout7200/json/20260328_direct_fixed_scaffold_compile_scout_fullaccess_attempt2_timeout7200.json`, the best-so-far setting was `opt1_seed4` with $\Delta E\approx 9.9899\times 10^{-2}$, two-qubit count `14`, and depth `48`, whereas the completed lower-depth `opt2_*` candidates kept the same two-qubit count `14` but had depth `38` and worse $\Delta E$: `opt2_seed0\approx 1.1726\times 10^{-1}`, `opt2_seed1\approx 1.2245\times 10^{-1}`, `opt2_seed2\approx 1.2029\times 10^{-1}`, so lower compiled depth alone did not improve noisy accuracy and the current hardware-facing compile preset is `(optimization_level=1, seed_transpiler=4)`.

As an $L=2$ run-planning prior rather than a theorem, the earlier local FakeMarrakesh ablations were mixed between readout-only and twirl while DD usually underperformed, but the completed 12-cell saved-$\theta^\star$ mitigation matrix in `artifacts/json/20260328_fixed_scaffold_saved_theta_mitigation_matrix_attempt5_timeout86400_delta_logs_fixed_scaffold_saved_theta_mitigation_matrix.json` now supersedes that partial prior: the top three cells were `opt2_seed0__zne_on__twirl_dd` with $\Delta E\approx-6.6431\times 10^{-4}$ and stderr $\approx1.2322\times 10^{-2}$ at 14 two-qubit gates and depth 38, `opt1_seed4__zne_on__twirl_dd` with $\Delta E\approx8.2990\times 10^{-3}$ and stderr $\approx2.6104\times 10^{-2}$ at 14 two-qubit gates and depth 48, and `opt2_seed0__zne_on__twirl` with $\Delta E\approx1.4807\times 10^{-2}$ and stderr $\approx1.1157\times 10^{-2}$ at 14 two-qubit gates and depth 38. For a real-QPU first pass on this locked 6-term scaffold, the current best local starting lane is therefore `(optimization_level=2, seed_transpiler=0)` with `ZNE on + twirl + DD`, with `(optimization_level=1, seed_transpiler=4)` plus the same mitigation stack as the nearest fallback if one wants to hedge against ZNE-fit variance while staying on the same two-qubit count.

For the weak-coupling locked scaffold `artifacts/json/useful/L2/hh_l2_u05_g02_full_meta_class_pruned_lean4_locked_scaffold_v1.json`, the completed 4-cell readout+mthree shortlist in `artifacts/json/20260329_weak_fixed_scaffold_top4_mitigation_readout_opt2seed5_fixed_scaffold_saved_theta_mitigation_matrix.json` had top two cells `opt2_seed5__zne_on__twirl_dd` with $\Delta E\approx1.0800\times 10^{-2}$ and stderr $\approx4.7014\times 10^{-3}$, and `opt2_seed5__zne_on__dd` with $\Delta E\approx1.6984\times 10^{-2}$ and stderr $\approx1.9163\times 10^{-3}$, both at 18 two-qubit gates and depth 43.

The later readout-on versus readout-off ablations showed that base readout mitigation was helping rather than hurting on both fixed-scaffold winner lines. For the strong-coupling local winner, the same `opt2_seed0__zne_on__twirl_dd` lane moved from $\Delta E\approx-6.6431\times 10^{-4}$ with readout on in `artifacts/json/20260328_fixed_scaffold_saved_theta_mitigation_matrix_attempt5_timeout86400_delta_logs_fixed_scaffold_saved_theta_mitigation_matrix.json` to $\Delta E\approx5.4456\times 10^{-2}$ with readout off in `artifacts/json/20260329_strong_fixed_scaffold_winner_readout_off_ablation_opt2seed0_twirl_dd_retry2_fixed_scaffold_saved_theta_mitigation_matrix.json`, at the same compiled cost (14 two-qubit gates, depth 38). For the weak-coupling 18-two-qubit compile `opt2_seed5`, the analogous readout-off follow-up in `artifacts/json/20260330_weak_fixed_scaffold_top2_readout_off_ablation_opt2seed5_fixed_scaffold_saved_theta_mitigation_matrix.json` gave `opt2_seed5__zne_on__twirl_dd` at $\Delta E\approx5.6830\times 10^{-2}$ and `opt2_seed5__zne_on__dd` at $\Delta E\approx5.8131\times 10^{-2}$, versus the readout-on values $\approx1.0800\times 10^{-2}$ and $\approx1.6984\times 10^{-2}$ respectively, so turning readout off worsened the best weak lanes by about $4.60\times10^{-2}$ and $4.11\times10^{-2}$.

The surprising regime comparison is therefore not in the noiseless scaffold bias but in the noisy response. The strong locked scaffold had $E_{\mathrm{ideal}}-E_{\mathrm{exact}}\approx4.6131\times10^{-3}$ from `artifacts/json/hh_marrakesh_fixed_scaffold_6term_drop_eyezee_20260323T171528Z.json`, while the weak locked scaffold had $E_{\mathrm{ideal}}-E_{\mathrm{exact}}\approx2.7761\times10^{-5}$ from `artifacts/json/useful/L2/hh_l2_u05_g02_full_meta_class_pruned_lean4_locked_scaffold_v1.json`. Thus the weak scaffold was much closer to exact in the noiseless sense, yet still produced a worse noisy $\Delta E$ after the current suppression stack; the deficit is therefore a noise-response / mitigation-effect issue rather than a larger ideal-state approximation error.

## 19.9 Heron transpile Pareto front and noisy replay for fixed L=2 scaffolds

All scaffolds share $L=2$, $n_{\mathrm{ph,max}}=1$, binary bosons, blocked ordering, open boundaries.
Each was transpiled to `FakeMarrakesh` over $(\texttt{optimization\_level},\texttt{seed\_transpiler})\in\{1,2\}\times\{0,\dots,9\}$; the table reports the best compiled two-qubit count per scaffold.
Term-pruned variants start from the 7-term locked scaffold and drop Pauli rotation terms, then reoptimize $\theta$ via Powell.

| Scaffold | $\lvert\Delta E\rvert$ | CZ gates | Depth | $F$ | Coupling | Source |
|---|---:|---:|---:|---:|---|---|
| `weak_lean4_locked` (4 ops) | $2.78\times 10^{-5}$ | 18 | 43 | 0.9999 | $U{=}0.5,g{=}0.2$ | class-prune |
| `aggressive_5op` (12 terms) | $5.62\times 10^{-5}$ | 26 | 72 | — | $U{=}4,g{=}0.5$ | op-prune |
| `gate_pruned_7term` (7 terms) | $4.23\times 10^{-4}$ | 21 | 51 | 0.9998 | $U{=}4,g{=}0.5$ | term-prune |
| `6term_drop_eyezee` (6 terms) | $4.61\times 10^{-3}$ | 14 | 38 | 0.9985 | $U{=}4,g{=}0.5$ | term-prune |
| `6term_drop_eyeeez` (6 terms) | $4.61\times 10^{-3}$ | 14 | 56 | 0.9985 | $U{=}4,g{=}0.5$ | term-prune |
| `5term_drop_both_disp` (5 terms) | $6.92\times 10^{-3}$ | 12 | 48 | 0.9986 | $U{=}4,g{=}0.5$ | term-prune |

The Pareto front over $(\lvert\Delta E\rvert,\,\text{CZ count})$ contains two points when both coupling regimes are pooled:
$$
\bigl(2.78\times 10^{-5},\;18\;\text{CZ}\bigr)_{\text{weak}},
\qquad
\bigl(4.61\times 10^{-3},\;14\;\text{CZ}\bigr)_{\text{strong}}.
$$
Within strong coupling alone, the three-point front is $\{(5.62\times 10^{-5},26),\;(4.61\times 10^{-3},14),\;(6.92\times 10^{-3},12)\}$.
The dominated scaffolds (`pruned_scaffold_11op` at 51 CZ, `ultra_lean_6op`/`readapt_6op` at 38 CZ, `gate_pruned_7term` at 21 CZ) offer no accuracy benefit relative to cheaper alternatives.

**Noisy replay on `FakeMarrakesh` (`backend_scheduled`)** for the strong-coupling Pareto-optimal 6-term scaffold (14 CZ, depth 38, `optimization_level=2`, `seed_transpiler=0`):

| Configuration | $E_{\mathrm{noisy}}$ | $E_{\mathrm{noisy}}-E_{\mathrm{exact}}$ |
|---|---:|---:|
| Saved $\theta^*$ + readout/mthree | $0.2773$ | $+0.119$ |
| Saved $\theta^*$ + readout + gate twirling | $0.2208$ | $+0.062$ |
| SPSA-128 best + readout + gate twirling | $0.1368$ | $-0.022$ |
| Saved $\theta^*$ ideal (noiseless) | $0.1633$ | $+0.005$ |
| Exact GS | $0.1587$ | $0$ |

Mitigation stack: readout/mthree plus local 2Q Pauli twirling, symmetry `verify_only`, no ZNE, no DD.
SPSA: 128 iterations, 1024 shots, 1 repeat, seed 19.
The SPSA-optimized noisy energy undershoots exact by $0.022$, indicating noise-induced bias below the variational floor; the ideal energy at the SPSA-best $\theta$ is $0.189$, confirming parameter drift.
At the noiseless-optimal $\theta^*$, gate twirling halves the noise bias from $0.119$ to $0.062$.

Reproduction:
```bash
python -m pipelines.exact_bench.heron_pareto_sweep --backend FakeMarrakesh --coupling both
python -m pipelines.hardcoded.hh_staged_noise \
  --L 2 --t 1.0 --u 4.0 --g-ep 0.5 --omega0 1.0 --n-ph-max 1 \
  --ordering blocked --boundary open --boson-encoding binary \
  --include-fixed-scaffold-noisy-replay \
  --fixed-final-state-json artifacts/json/hh_marrakesh_fixed_scaffold_6term_drop_eyezee_20260323T171528Z.json \
  --use-fake-backend --backend-name FakeMarrakesh \
  --fixed-scaffold-runtime-transpile-optimization-level 2 \
  --fixed-scaffold-runtime-seed-transpiler 0 \
  --final-method SPSA --final-maxiter 128 --final-seed 19 \
  --mitigation readout --local-readout-strategy mthree --local-gate-twirling \
  --symmetry-mitigation-mode verify_only \
  --shots 1024 --oracle-repeats 1 --noise-seed 7 \
  --smoke-test-intentionally-weak --skip-pdf
```

Artifacts: `artifacts/json/heron_pareto_front_20260327T223120Z.json`, `artifacts/json/hh_6term_pareto_readout_twirl_fast_20260328.json`.

# Appendix A. Spec-only or not-yet-formalized surfaces kept out of the main body

## A.1 Broader continuation-theory items

Several topics belong naturally in a later symbolic pass rather than in the present core manuscript:

- multi-source motif transfer and motif tiling,
- richer split-event formalism,
- more detailed rescue dynamics,
- exact handoff-policy closure beyond the present abstract maps,
- measured-geometry and backend-specific closure beyond the exact symbolic statevector surface.

## A.2 Obsolete historical warm-start $\to$ refine $\to$ ADAPT $\to$ replay path

The historical staged path can be written symbolically as
$$
\begin{aligned}
\mathcal M_{\mathrm{warm}}
&=\{\,U_{\mathrm{warm}}(\theta_{\mathrm{w}})|\phi_0\rangle:\theta_{\mathrm{w}}\in\mathbb R^{m_{\mathrm{w}}}\,\},\\
\mathcal M_{\mathrm{refine}}
&=\{\,U_{\mathrm{refine}}(\theta_{\mathrm{r}})U_{\mathrm{warm}}(\theta_{\mathrm{w}}^*)|\phi_0\rangle:\theta_{\mathrm{r}}\in\mathbb R^{m_{\mathrm{r}}}\,\},\\
\mathcal M_{\mathrm{replay}}(\mathcal O_*)
&=\{\,U_{\mathrm{replay}}(\varphi;\mathcal O_*)|\phi_0\rangle:\varphi\in\mathbb R^{m_{\mathrm{rep}}}\,\}.
\end{aligned}
$$
The corresponding historical handoff chain is
$$
\begin{aligned}
(\theta_{\mathrm{w}}^*,|\psi_{\mathrm{warm}}\rangle)
&\in\mathcal M_{\mathrm{warm}}
\mapsto
(\theta_{\mathrm{r}}^*,|\psi_{\mathrm{refine}}\rangle)
\in\mathcal M_{\mathrm{refine}}\\
&\mapsto
(\mathcal O_*,\theta_*^{\mathrm{adapt}},|\psi_*\rangle)\\
&\mapsto
(\varphi^{(0)},|\psi_{\mathrm{replay}}^{(0)}\rangle)
=
\mathcal H_{\mathrm{replay}}(|\psi_*\rangle,\theta_*^{\mathrm{adapt}},\mathcal O_*,\mathcal C_*)
\in\mathcal M_{\mathrm{replay}}(\mathcal O_*).
\end{aligned}
$$
A typical replay-style historical policy is:

1. preserve inherited scaffold parameters,
2. initialize residual parameters at zero,
3. optionally freeze inherited coordinates for a burn-in stage,
4. then unfreeze and refit in a controlled window.

This path is kept for archival provenance only and is not part of the active main-body symbolic contract.

## A.3 Drive-amplitude comparison path

A drive-amplitude comparison path can be expressed symbolically by comparing several amplitudes $A_0,A_1,\dots$ at fixed Hamiltonian parameters and examining
$$
\Delta E_A(t),\qquad \Delta D_A(t),\qquad F_A(t)
$$
across the amplitude family. The formal structure is clear even if the exact report layout is deferred.

## A.4 Appendix rule

The rule of this symbolic manuscript is therefore:

1. main body = closed mathematical definitions and comparison contracts,
2. appendix = promising but not yet fully formalized extensions.

# References

1. Grimsley, Economou, Barnes, and Mayhall, An adaptive variational algorithm for exact molecular simulations on a quantum computer, Nature Communications 10, 3007 (2019), DOI: 10.1038/s41467-019-10988-2.

2. Provost and Vallée, Riemannian structure on manifolds of quantum states, Communications in Mathematical Physics 76, 289-301 (1980), DOI: 10.1007/BF02193559.

3. Stokes, Izaac, Killoran, and Carleo, Quantum Natural Gradient, Quantum 4, 269 (2020), DOI: 10.22331/q-2020-05-25-269.

4. Nocedal and Wright, Trust-Region Methods, in Numerical Optimization, Springer, DOI: 10.1007/0-387-22742-3_4.

5. Ramôa, Santos, Mayhall, Barnes, and Economou, Reducing measurement costs by recycling the Hessian in adaptive variational quantum algorithms, Quantum Science and Technology 10, 015031 (2025), DOI: 10.1088/2058-9565/ad904e.

6. Ramôa, Anastasiou, Santos, Mayhall, Barnes, and Economou, Reducing the resources required by ADAPT-VQE using coupled exchange operators and improved subroutines, npj Quantum Information (2025), DOI: 10.1038/s41534-025-01039-4.

7. Verteletskyi, Yen, and Izmaylov, Measurement optimization in the variational quantum eigensolver using a minimum clique cover, The Journal of Chemical Physics 152, 124114 (2020), DOI: 10.1063/1.5141458.

8. Yen, Verteletskyi, and Izmaylov, Measuring All Compatible Operators in One Series of Single-Qubit Measurements Using Unitary Transformations, Journal of Chemical Theory and Computation (2020), DOI: 10.1021/acs.jctc.0c00008.

9. Ikhtiarudin, Sunnardianto, Fathurrahman, Agusta, and Dipojono, Shot-Efficient ADAPT-VQE via Reused Pauli Measurements and Variance-Based Shot Allocation, arXiv:2507.16879 (2025).

10. Stadelmann, Übelher, Ramôa, Sambasivam, Barnes, and Economou, Strategies for Overcoming Gradient Troughs in the ADAPT-VQE Algorithm, arXiv:2512.25004 (2025).

### Addendum (2026-04-04): manuscript-aligned direct HH ADAPT / `phase3_v1` sweep notes

This addendum records the new manuscript-aligned direct HH ADAPT behavior on the public `phase3_v1` surface, with `runtime_split=off` treated as the canonical manuscript/public setting.

Physics point for the main recovery study.
- `L=2`
- `t=1.0`
- `U=4.0`
- `g_ep=0.5`
- `omega0=1.0`
- `n_ph_max=1`
- `problem=hh`
- binary bosons
- blocked ordering
- open boundary

Authority targets for interpretation.
- Manuscript matched-ADAPT target from §19.1: `E = 0.1587240823603703`, `|ΔE| = 5.6178234644e-05`, `stop_reason = drop_plateau`.
- Fixed-scaffold anchor from §19.6: `|ΔE| ≈ 4.23e-04`, `F ≈ 0.9998`.

#### Current status on the new manuscript-aligned surface

The best live strong-coupling beam line on the direct public surface is
`beam_runtime_split-off__batching-off__lifetime-off__profile-tight__gamma-1.0`
with
- `E = 0.15872511715315915`
- `|ΔE| = 5.721302743347256e-05`
- `ansatz_depth = 20`
- `logical_num_parameters = 20`
- `num_parameters = 48`
- `stop_reason = drop_plateau`
- `exact_state_fidelity = 0.9999715655261739`

This line clearly beats the fixed-scaffold anchor, but it does not yet numerically match the manuscript matched-ADAPT target. The residual gap to the manuscript matched-ADAPT line is `1.0347927889e-06` in `|ΔE|`.

A historical diagnostic reoptimization run with windowed `COBYLA` reached
- `E = 0.15872468271899687`
- `|ΔE| = 5.6778593271189504e-05`
- residual gap to the manuscript matched-ADAPT line: `6.0035862640e-07`

This is useful as a numerical diagnostic, but it is not the forward recipe for new runs. Going forward, the intended optimizer family is `SPSA`, not `COBYLA`.

#### Executed run families and outcomes

1. Strong-coupling beam sweep.
- Artifact root: `artifacts/agent_runs/20260403_hh_l2_u4_g05_beam_sweep_stage1/`
- Best case: `beam_runtime_split-off__batching-off__lifetime-off__profile-tight__gamma-1.0`
- Result: `|ΔE| = 5.721302743347256e-05`
- Interpretation: beam search nearly recovers the manuscript matched-ADAPT quality.

2. Strong-coupling runtime sweep.
- Artifact root: `artifacts/agent_runs/20260403_hh_l2_u4_g05_runtime_sweep_stage1/`
- Best case still bad: `|ΔE| = 0.3282923042519191`
- Interpretation: the single-trajectory runtime line collapses into the same bad basin seen in the manuscript novelty-off failure scale.

3. Phase-3 beam-vs-batching diagnostic.
- Artifact root: `artifacts/agent_runs/20260403_hh_l2_u4_g05_phase3_diag_beam_vs_batch_v1/`
- Single-trajectory cases stayed at `|ΔE| ≈ 0.3282923`.
- Beam `b3,c2,k3` immediately recovered `|ΔE| = 5.721302743347256e-05`.
- Interpretation: the decisive lever is frontier diversity from beam search, not batching.

4. Beam-capacity recovery slice.
- Artifact root: `artifacts/agent_runs/20260403_hh_l2_u4_g05_phase3_beam_capacity_recovery_v1/`
- `b3,c2,k3` and `b4,c2,k4` tied at `|ΔE| = 5.721302743347256e-05`.
- Interpretation: larger beam width by itself does not improve the recovered solution once the beam is already large enough.

5. Prune/shortlist micro-sweep.
- Artifact root: `artifacts/agent_runs/20260404_hh_l2_u4_g05_phase3_prune_shortlist_micro_v1/`
- All cases were numerically identical to saved precision.
- Interpretation: `phase1_prune_fraction`, `phase1_prune_max_candidates`, and shortlist width are not the active driver of the remaining manuscript gap on this beam-fixed line.

6. Reoptimization slice.
- Artifact root: `artifacts/agent_runs/20260404_hh_l2_u4_g05_phase3_reopt_slice_v1/`
- Windowed `COBYLA` improved the gap to `|ΔE| = 5.6778593271189504e-05`.
- `POWELL` windowed stayed at the original beam winner.
- `POWELL` periodic/full refits got worse.
- `append_only` failed badly at `|ΔE| = 0.6244148638396572`.
- Interpretation: optimizer/refit policy can move the line, but the `COBYLA` improvement should be treated as diagnostic only, not as the new manuscript-forward recipe.

7. Score-weight slice.
- Artifact root: `artifacts/agent_runs/20260404_hh_l2_u4_g05_phase3_score_weight_slice_v1/`
- `near=0.90` gave a small improvement to `|ΔE| = 5.721150236287498e-05` and also reduced the scaffold to `logical_num_parameters = 19`, `num_parameters = 46`, `ansatz_depth = 19`.
- Other `rho` and `lambda_H` variants were essentially flat at the saved precision.

8. Seed robustness slice.
- Artifact root: `artifacts/agent_runs/20260404_hh_l2_u4_g05_phase3_seed_robustness_slice_v1/`
- Seeds `5, 7, 11, 13, 17` all reproduced the same `POWELL` baseline result exactly to saved precision.
- Interpretation: the current beam-fixed `POWELL` plateau is deterministic across this seed set.

9. Weak-coupling diagnostic lanes.
- Canonical weak sweep root: `artifacts/agent_runs/20260403_hh_l2_u05_g02_weak_logical_scaffold_canonical_sweep_v1/`
- Weak beam sweep root: `artifacts/agent_runs/20260403_hh_l2_u05_g02_beam_sweep_stage1_parent_full_meta/`
- Best weak result in both lanes stayed at `|ΔE| ≈ 0.0131029358626`.
- Parent-quality target from §19.3.1 is much smaller, so weak coupling remains a diagnostic regime rather than a recovered manuscript-quality regime.

#### Good parameters versus bad parameters

Parameters/choices that consistently helped or were at least required for recovery.
- Direct HH ADAPT on the public `phase3_v1` surface.
- `runtime_split=off`.
- Beam search on.
- Beam at or above `live_branches=3`, `children_per_parent=2`, `terminated_keep=3`.
- `gamma_N=1.0` on the public strong-coupling line.
- `batching=off` was slightly better than `batching=on` in the best recovered beam lane.
- `lifetime_cost_mode=off` was at least as good as `phase3_v1` within the explored resolution.
- `near=0.90` was the only score-weight variation that gave a measurable improvement in the explored `POWELL` lane.

Parameters/choices that were neutral or nearly inert in the explored region.
- Beam width beyond the `b3,c2,k3` threshold.
- `profile=tight` versus `profile=wide`.
- `phase1_prune_fraction` in the explored `0.15` to `0.35` range.
- `phase1_prune_max_candidates` in the explored `4` to `8` range.
- shortlist width in the explored micro-sweep.
- Most `rho` and `lambda_H` score-weight perturbations.

Parameters/choices that consistently hurt.
- Single-trajectory runtime mode with no beam.
- Runtime line behavior that reproduces `|ΔE| ≈ 0.3282923`.
- `runtime_split=shortlist_pauli_children_v1` relative to `off` on this manuscript/public lane.
- `POWELL` periodic/full refits relative to the windowed baseline.
- `append_only` refit policy, which failed catastrophically.

#### Practical interpretation

For the manuscript-aligned direct HH ADAPT path at strong coupling, the main qualitative lesson is now clear: beam-induced frontier diversity is the key ingredient that recovers the near-manuscript solution, while pruning pressure, shortlist width, profile choice, and modest score-weight changes are second-order. The live public-surface recipe already beats the fixed-scaffold anchor by a large margin, but it still does not numerically match the manuscript matched-ADAPT target.

#### 2026-04-04 update: public-CLI runtime weights and reduced-pool sweeps

After exposing the manuscript-facing selector/runtime knobs on the public direct `phase3_v1` CLI, two additional conclusions emerged for the same strong-coupling $L=2$ point.

1. Public-CLI `full_meta + SPSA` improved once frontier ratios were decoupled from the near-degenerate batch shell.
- Artifact root: `artifacts/agent_runs/20260404_hh_l2_u4_g05_phase3_public_weights_frontier_spsa_v1/`
- Best public-CLI `full_meta + SPSA` line:
  $$
  E=0.15878350593338042,\qquad |\Delta E|=1.1560180765\times 10^{-4},
  $$
  with `logical_num_parameters=21`, `num_parameters=53`, `ansatz_depth=21`, and `fidelity=0.9999480673`.
- The corresponding `weights_zero` line was only slightly worse,
  $$
  |\Delta E|=1.1722877711\times 10^{-4},
  $$
  while the tighter coupled-frontier setting
  $$
  (\text{phase2 frontier},\text{phase3 frontier})=(0.98,0.98)
  $$
  reproduced the older worse basin
  $$
  |\Delta E|=1.4383993770\times 10^{-4}.
  $$
- So the main gain from that patch was not a delicate burden-weight choice; it was fixing the implementation coupling between shortlist frontiering and the batch-shell threshold so that `phase2_frontier_ratio`, `phase3_frontier_ratio`, and `eta_{nd}` became genuinely independent controls.

2. The reduced-pool `pareto_lean + SPSA` lane remained rigid.
- Artifact roots:
  - `artifacts/agent_runs/20260404_hh_l2_u4_g05_phase3_pool_family_spsa_fixed_v1/`
  - `artifacts/agent_runs/20260404_hh_l2_u4_g05_phase3_pareto_lean_gamma_spsa_fixed_v1/`
  - `artifacts/agent_runs/20260404_hh_l2_u4_g05_phase3_pareto_lean_drop_policy_spsa_fixed_v1/`
  - `artifacts/agent_runs/20260404_hh_l2_u4_g05_pareto_lean_new_cli_weights_spsa_v1/`
  - `artifacts/agent_runs/20260404_hh_l2_u4_g05_pareto_lean_compat_proxy_spsa_v1/`
  - `artifacts/agent_runs/20260404_hh_l2_u4_g05_pareto_lean_frontier_ratio_spsa_v1/`
- Baseline reduced-pool result:
  $$
  E=0.1587895750333393,\qquad |\Delta E|=1.2167090761\times 10^{-4},
  $$
  with `logical_num_parameters=21`, `num_parameters=52`, `ansatz_depth=21`, and `fidelity=0.9999443566`.
- This value persisted across all tested variations of:
  - `gamma_N` in the explored `0.75` to `1.5` range,
  - drop-policy parameters,
  - motif bonus,
  - duplicate penalty,
  - leakage penalty `eta_L`,
  - compatibility-prescreen weights,
  - remaining-evaluations proxy mode,
  - frontier ratios from `(0.85,0.85)` up through `(0.95,0.95)` and the asymmetric `(0.90,0.75)`, `(0.90,0.95)`, `(0.95,0.90)` cases.
- Loosening the phase-2 frontier too far did hurt: `(0.75,0.75)` gave
  $$
  |\Delta E|=1.5234589586\times 10^{-4},
  $$
  and `(0.75,0.90)` gave
  $$
  |\Delta E|=1.5457582580\times 10^{-4}.
  $$
- Hence the reduced `pareto_lean` line appears structurally plateaued under the currently exposed runtime-law knobs: it is not rescued by more local tuning of `gamma_N`, batching pressure, motif/repetition heuristics, or compatibility/proxy terms.

3. Cost-aware ranking versus raw-energy ranking now separate.
- On raw strong-coupling energy error alone, the best historical rerank observed in these sweeps was the diagnostic `COBYLA` line
  $$
  E=0.15872468271899687,\qquad |\Delta E|=5.6778593271\times 10^{-5},
  $$
  with `logical_num_parameters=20`, `num_parameters=50`, `ansatz_depth=20`.
- However, the best currently observed $|\Delta E|/K$-style point among runs that reported logical scaffold size was the `near=0.90` strong-coupling line from the score-weight slice:
  $$
  E=0.15872511562808855,\qquad |\Delta E|=5.7211502363\times 10^{-5},
  $$
  with
  $$
  K_{\text{logical}}=19,\qquad K_{\text{runtime}}=46,\qquad d=19.
  $$
- So the present picture is:
  - best raw-$|\Delta E|$ line: slightly better than the old beam winner, but achieved on a non-preferred optimizer branch,
  - best cost-aware line: slightly worse in $|\Delta E|$ but leaner in both logical scaffold size and runtime parameter count,
  - best reduced-pool SPSA line: still stuck near $1.22\times 10^{-4}$ and therefore materially behind the broad-beam public lane.

#### 2026-04-04 update: completed reduced-pool `L=2/L=3` matrix

The overnight reduced-pool matrix `artifacts/agent_runs/20260404_hh_phase3_reduced_pool_matrix_v1/` completed through the strong confirmation and weak-transfer waves. Its campaign summary and final strong templates are recorded in

- `artifacts/agent_runs/20260404_hh_phase3_reduced_pool_matrix_v1/campaign_summary.json`,
- `artifacts/agent_runs/20260404_hh_phase3_reduced_pool_matrix_v1/final_strong_templates_by_L.json`,
- `artifacts/agent_runs/20260404_hh_phase3_reduced_pool_matrix_v1/matrix_leaderboard.json`.

Data field: the completed matrix contained `212` successful cases, `36` failed cases, and `184` unique successful configurations.

For the strong-coupling $L=2$ target point, the best matrix result was
$$
E=0.15877612622031317,\qquad |\Delta E|=1.0822209459\times 10^{-4},
$$
with
$$
K_{\mathrm{logical}}=20,\qquad K_{\mathrm{runtime}}=52,\qquad d=20,\qquad F=0.9999519211,
$$
from the confirmed `pareto_lean` template with
$$
(\text{beam},\text{frontier})=(2,2,2;\,0.85,0.85).
$$
This improved over the earlier reduced-pool SPSA plateau near $1.2167\times10^{-4}$, but it still did not recover the manuscript-quality strong-coupling target at $5.6178234644\times10^{-5}$ and did not beat the earlier broad-lane live recovery line near $5.7213027433\times10^{-5}$.

For the strong-coupling $L=3$ point, the reduced-pool matrix failed badly. Its best confirmed case was
$$
E=-0.2564741176143438,\qquad |\Delta E|=5.0141481774\times 10^{-1},
$$
with fidelity only
$$
F\approx 5.81\times 10^{-8}.
$$
The best matrix lane there was `uccsd_paop_lf_full`, not the more aggressively reduced families. Hence the completed reduced-pool matrix should be read as negative evidence for the current public reduced execution surface at $L=3$, not as evidence that the historical $L=3$ HH target itself is unattainable.

For weak coupling, the transferred $L=2$ reduced-pool line remained poor:
$$
E=-0.7632521982953032,\qquad |\Delta E|=1.3125332609\times10^{-2},
$$
so the matrix did not close the weak-coupling parent-quality gap there. By contrast, the transferred $L=3$ weak baseline was numerically much better,
$$
E=-1.0439438365685594,\qquad |\Delta E|=4.1313925567\times10^{-4},
$$
with fidelity
$$
F=0.9998116001.
$$
That $L=3$ weak result is still not a manuscript anchor, but it does show that the reduced families are not uniformly broken across all $L=3$ HH regimes; the main failure is the strong-coupling $L=3$ execution surface.

To make that contrast explicit, the surviving historical broad-pool $L=3$ reference from `artifacts/reports/hh_heavy_scaffold_best_yet_20260321.md` was
$$
E=0.24502564220194084,\qquad |\Delta E|=8.4942074026\times10^{-5},
$$
while the matched lean comparison report `artifacts/reports/pareto_lean_vs_heavy_L3_comparison.md` recorded the lean/class-pruned line at about
$$
|\Delta E|\approx 1.15\times10^{-4}.
$$
Both are therefore qualitatively incompatible with the present reconstructed direct-surface reruns, which all collapsed to the same bad basin.

In particular, the explicit three-way reproduction run `artifacts/agent_runs/20260404_hh_l3_u4_g05_historical_surface_pool_compare_v1/` reconstructed the documented historical settings
$$
\texttt{phase3\_v1},\ \texttt{POWELL},\ \texttt{phase1 shortlist}=256,\ \texttt{phase2 shortlist}=128,\ \texttt{split}=\texttt{shortlist\_pauli\_children\_v1},
$$
and compared

- `full_meta`,
- class-filtered `full_meta`,
- `pareto_lean_l3`.

All three produced the same failed result,
$$
E=-0.25838912148429044,\qquad |\Delta E|=5.0332982161\times10^{-1},
$$
with negligible fidelity. Hence the current issue is not simply which reduced family is chosen. Rather, some additional execution-surface ingredient from the older successful $L=3$ lane is still missing from the present direct CLI reconstruction.
