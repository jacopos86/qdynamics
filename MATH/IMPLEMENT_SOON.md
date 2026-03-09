# Hubbard–Holstein ADAPT→Replay Continuation Implementation Specification

## Purpose

This document specifies the **remaining implementation work** needed to reach the
target HH ADAPT->replay continuation architecture.

The current repo already has parts of the continuation baseline, but several
items below are still transitional or unimplemented. Read this document as the
**target design specification**, not as a claim that every mechanism described
here already exists in production code.

### Current repo audit snapshot

#### Implemented now
- ADAPT->replay continuation exists; replay is no longer framed as a blind
  restart from a fresh fixed-structure circuit.
- Replay seed construction supports scaffold-aware continuation semantics
  (`scaffold_plus_zero` / `residual_only`) instead of default tiled
  `\theta^\star` replay.
- ADAPT supports `append_only`, `full`, and `windowed` reoptimization modes.
- Windowed ADAPT reoptimization includes periodic full-prefix refits and a final
  full-prefix refit option.

#### Partially implemented / transitional baseline
- Replay continuation semantics exist, but replay-side freeze/unfreeze burn-in
  and trust-region handoff are not yet a first-class production subsystem.
- Windowed ADAPT refits exist, but the full refit-aware selection and scoring
  stack described below is not yet wired through the executable scaffold.
- Current code still exposes `full_meta` as an available HH pool and some older
  run guidance still points agents toward depth-0 `full_meta`; this document
  treats that as legacy/current behavior, not the target pool curriculum.

#### Planned / not yet implemented
- Pool curriculum that keeps `full_meta` closed at depth 0 and opens residual
  enrichment only after plateau diagnosis.
- Core measurement-reuse subsystem for shared energy/gradient/metric/curvature
  work.
- Prune-before-replay with post-prune refit as part of the standard handoff.
- Persisted executable-scaffold state beyond angles alone, including compiled
  artifacts and optimizer memory.
- Explicit compile-cost oracle/input and exact pre-admission exposure of the
  optimizer window \(W(p)\) for refit-aware scoring.
- Replay freeze/unfreeze burn-in and inherited-parameter protection as an
  explicit staged replay policy.

The goal of this document is to convert the remaining atomic suggestions into a **single, coherent, implementation-ready architecture** for an AI coding agent. The document is intentionally written at the level of **mathematical model + control logic + interfaces + pseudocode**, with generic placeholders that a repo-aware agent can later bind to concrete files, functions, classes, and JSON schemas.

The design target is a **hardware-oriented HH pipeline**:

\[
\text{tiny HH seed} \;\to\; \text{physics-aligned ADAPT core} \;\to\; \text{controlled residual enrichment} \;\to\; \text{prune} \;\to\; \text{scaffolded replay}.
\]

The central principle is:

> Select operators by **predicted useful energy decrease on the executable scaffold, after the refit you actually intend to do, divided by effective marginal lifetime cost**.

---

## Executive Summary

The remaining work should be implemented as a layered control system, not as isolated heuristics.

### Mandatory architecture
1. **Pool curriculum**
   - Do not open `full_meta` at depth 0.
   - Use a narrow HH physics-aligned pool first.
   - Only open residual enrichment after plateau diagnosis.

2. **Position-aware operator selection**
   - Operator identity alone is insufficient.
   - Insertion position becomes a control variable.
   - Probe alternative positions before declaring convergence or opening a bigger pool.

3. **Score candidates by predicted post-refit energy drop per lifetime hardware cost**
   - The mathematically complete target score uses:
     - gradient confidence lower bound,
     - Fubini–Study metric element,
     - effective curvature after window relaxation,
     - tangent-space novelty,
     - additive hardware burdens.
   - The first production implementation may use a simplified surrogate, provided the API is designed so the full score can replace it later without architectural churn.

4. **Measurement reuse becomes a subsystem**
   - Reuse grouped Pauli measurements across energy, gradient, metric, and curvature estimation whenever algebraically possible.
   - Preserve current compiled-action exact-statevector acceleration as the simulator/noiseless backend.

5. **Prune before replay**
   - Prune a fully refit scaffold.
   - Refit once after pruning.
   - Replay only from the cleaned scaffold.

6. **Replay is continuation, not restart**
   - Preserve scaffold state, compiled artifacts, measurement partitions, layout choices, and optimizer memory.
   - Keep residual blocks zero-initialized.
   - Use trust-region or freeze-then-unfreeze protection on inherited parameters.

### Recommended implementation order
- **Phase 1**: pool curriculum, simple score, insertion probes, pruning, cache scaffolding, telemetry.
- **Phase 2**: shortlist/full score, curvature reuse, QN-SPSA replay refresh, batch selection.
- **Phase 3**: motif tiling, symmetry expansion hooks, overlap-guided rescue mode.

---

## Baseline Assumptions

This specification starts from a **mixed baseline**:

### Implemented now
- Replay means **continue around the ADAPT scaffold**, not replace it with a
  fresh fixed-structure circuit.
- Replay seed construction supports residual-capacity continuation semantics
  rather than default tiled \(\theta^\star\) replay.
- ADAPT no longer uses pure append-only reoptimization as its only available
  policy.
- Windowed reoptimization exists.
- Periodic full-prefix refits exist.
- Final full-prefix refits exist.

### Partially implemented
- Replay can represent scaffold-aware initialization, but the staged
  freeze-then-unfreeze replay optimizer flow below should still be read as
  planned implementation work unless a later repo audit proves otherwise.
- Windowed ADAPT refits exist, but the exact optimizer window \(W(p)\) is not
  yet universally exposed to the scoring/control machinery described below.
- Current code still supports depth-0 `full_meta`; this document does **not**
  treat that as the intended target curriculum.

### Planned / not yet implemented
- Replay burn-in and constrained unfreeze as a first-class replay control
  policy.
- Measurement reuse, compile-cost-aware admission, prune-before-replay, and
  optimizer-memory persistence as core subsystems.

### Consequences
This document extends the implemented pieces and specifies the missing ones.

---

## Non-negotiable Repository Guardrails

The implementation must preserve the repository contracts below.

### Operator conventions
- Internal Pauli symbols use `e/x/y/z`.
- Pauli words are ordered left-to-right as \(q_{n-1}\dots q_0\).
- Qubit \(0\) is the rightmost character.

### Jordan–Wigner source of truth
- Do not rederive JW ladder operators ad hoc.
- Use the repository JW construction utilities already designated as canonical.

### Core operator files
- The operator algebra core must remain stable.
- New behavior should be added via wrappers/shims/services around existing operator-core files, not by rewriting the core algebra layer.

### Core VQE / ADAPT path
- The production path remains NumPy/statevector centric.
- Qiskit is not introduced into the core hardcoded VQE/ADAPT path.
- The current compiled Pauli-action caching invariant for ADAPT gradients must be preserved.

### Time-evolution readiness
- Any new ansatz/generator object must remain compatible with the repository’s first-class
  \[
  \exp(-i \theta \, \text{PauliTerm}), \qquad \exp(-i \theta \, \text{PauliPolynomial})
  \]
  execution style.

### Replay contract
- Replay must remain provenance-aware.
- Do not silently regress to tiled-\(\theta^\star\) seed semantics.
- Prepared-state vs reference-state handoff semantics must remain explicit and stable.

### Policy-vs-code mismatch rule
- If this specification, the repository policy documents, and current code behavior diverge, the coding agent must stop and surface the mismatch explicitly before proceeding.
- The implementation plan must therefore distinguish:
  - documented target behavior,
  - current code behavior,
  - requested migration action.

### Reoptimization contract
- Windowed refit behavior already exists; the new score must use the **same unlocked window the optimizer will actually use**, not a hypothetical one.

---

## Scope

### In scope
This document specifies how to implement the remaining design ideas:

- pool curriculum;
- position-aware selection and trough detection;
- normalized, hardware-aware, refit-aware operator scoring;
- batched ADAPT with compatibility penalties;
- measurement reuse subsystem;
- selective QN-SPSA use;
- small polaronic HH seed;
- prune-before-replay;
- macro-generators and selective splitting;
- uncertainty-aware control logic;
- trust-region replay handoff;
- persistence-based `full_meta` admission;
- compiled artifact persistence;
- symmetry use in construction and mitigation;
- optimizer-memory persistence;
- motif tiling for larger systems;
- optional overlap-guided rescue mode.

### Out of scope
This document does not specify:
- exact repo file edits;
- exact CLI flags;
- final JSON field names;
- final plotting/report formatting;
- concrete low-level measurement grouping implementation details;
- any rewrite of the operator algebra core.

Those are for a repo-aware implementation pass later.

---

## Design Philosophy

The correct abstraction is not “ADAPT picks an operator, then replay does VQE.”

The correct abstraction is:

1. Maintain an **executable scaffold**:
   \[
   \mathcal S = (\text{generator sequence}, \text{angles}, \text{compiled artifacts}, \text{measurement plan}, \text{optimizer memory}, \text{layout/decomposition state}).
   \]

2. At each ADAPT decision point, score a candidate by the **useful energy reduction it is expected to deliver once admitted into that scaffold and once the intended refit is performed**.

3. Charge the candidate for the **lifetime burden** it introduces:
   - compiled depth / entangler cost,
   - new commuting measurement groups,
   - additional shots after reuse,
   - added optimizer dimension,
   - repeated family reuse.

4. Use pool curriculum and insertion probes to avoid two false moves:
   - opening an overly large pool too early;
   - interpreting an append-position gradient trough as genuine convergence.

5. After the scaffold has plateaued, **clean it** (prune) before growing around it again (replay).

This turns the whole pipeline into a continuation method with resource awareness.

---

## Mathematical Setup

Let the current scaffold be

\[
U_{\mathrm{scaf}}(\theta^\star)
=
U_K(\theta^\star_K)\cdots U_1(\theta^\star_1),
\qquad
|\psi\rangle = U_{\mathrm{scaf}}(\theta^\star)|\psi_0\rangle .
\]

Each unitary block is generated by a Hermitian operator \(A_j\):

\[
U_j(\theta_j)=e^{-i\theta_j A_j}.
\]

A candidate operator \(A_m\) may be inserted at a position \(p\) within the scaffold.
If \(U_{>p}\) is the suffix after the insertion point, define the **dressed generator**

\[
\widetilde A_{m,p}=U_{>p}A_mU_{>p}^\dagger.
\]

The zero-initialized energy gradient for this insertion is

\[
g_{m,p}
=
\frac{\partial E}{\partial \alpha}\Big|_{\alpha=0}
=
2\,\mathrm{Im}\,\langle H\psi|\widetilde A_{m,p}|\psi\rangle,
\]
where the candidate parameter is \(\alpha\).

The corresponding Fubini–Study / QGT diagonal element is

\[
F_{m,p}
=
\langle \widetilde A_{m,p}^2\rangle_\psi
-
\langle \widetilde A_{m,p}\rangle_\psi^2
=
\|(\widetilde A_{m,p}-\langle \widetilde A_{m,p}\rangle_\psi)|\psi\rangle\|^2.
\]

This is the correct local state-space norm of the candidate direction.

### Why \(F\) matters
A raw gradient magnitude is not fair across heterogeneous generator families because
\[
A_m \mapsto \kappa A_m
\quad\Rightarrow\quad
g_{m,p}\mapsto \kappa g_{m,p}.
\]
A scale-aware score must remove this arbitrary normalization dependence.
Using \(F\), curvature, and a trust region yields a score invariant under such rescaling.

---

## Effective Curvature Under the Planned Refit

Suppose that, after admitting a candidate at position \(p\), the optimizer will unlock a window of inherited parameters \(W(p)\).
Let \(\delta\theta_W\) be the local displacement of that unlocked window.

A local quadratic model is

\[
E(\alpha,\delta\theta_W)
\approx
E_0
+
g_{m,p}\alpha
+
\frac12 h_{m,p}\alpha^2
+
\alpha\, b_{m,p}^{\top}\delta\theta_W
+
\frac12 \delta\theta_W^{\top}H_{W(p)}\delta\theta_W .
\]

Because the current point is already locally optimized over the active window,
there is no linear term in \(\delta\theta_W\).

Minimizing over \(\delta\theta_W\) gives

\[
\delta\theta_W^\star
=
-(H_{W(p)}+\lambda_H I)^{-1}b_{m,p}\,\alpha,
\]
and substituting back yields the reduced 1D model
\[
E_{\mathrm{red}}(\alpha)
\approx
E_0
+
g_{m,p}\alpha
+
\frac12 \widetilde h_{m,p}\alpha^2,
\]
with effective curvature
\[
\widetilde h_{m,p}
=
h_{m,p}
-
b_{m,p}^{\top}(H_{W(p)}+\lambda_H I)^{-1}b_{m,p}.
\]

Interpretation:

> \(\widetilde h_{m,p}\) is the curvature of the genuinely new direction **after** allowing the inherited window that you actually plan to refit to relax.

This is the correct curvature for continuation logic.

---

## Trust-Region Predicted Energy Drop

On hardware, or even in simulator continuation, a new operator should not be admitted on the basis of an arbitrarily large local step.
The local state-space line element induced by \(\alpha\) is

\[
ds^2 = F_{m,p}\,\alpha^2.
\]

If the allowed trust radius is \(\rho\), then admissible steps satisfy

\[
|\alpha| \le \frac{\rho}{\sqrt{F_{m,p}}}.
\]

Define a lower-confidence gradient magnitude
\[
g^\downarrow_{m,p}
=
\big(|\hat g_{m,p}| - z_\alpha \hat \sigma_{m,p}\big)_+,
\qquad
(x)_+=\max(x,0).
\]

Then the predicted useful drop is the maximized 1D trust-region model

\[
\Delta \widehat E^{\mathrm{TR}}_{m,p}
=
\max_{|\alpha|\le \rho/\sqrt{F_{m,p}}}
\left[
g^\downarrow_{m,p}|\alpha|
-\frac12 \widetilde h_{m,p}^{+}\alpha^2
\right],
\]
with
\[
\widetilde h_{m,p}^{+}=\max(\widetilde h_{m,p},0).
\]

Closed form:
\[
\Delta \widehat E^{\mathrm{TR}}_{m,p}
=
\begin{cases}
\dfrac{(g^\downarrow_{m,p})^2}{2\widetilde h^+_{m,p}},
&
\dfrac{g^\downarrow_{m,p}}{\widetilde h^+_{m,p}}
\le
\dfrac{\rho}{\sqrt{F_{m,p}}}, \\[1.2ex]
\dfrac{\rho\, g^\downarrow_{m,p}}{\sqrt{F_{m,p}}}
-
\dfrac{\rho^2\,\widetilde h^+_{m,p}}{2F_{m,p}},
&
\text{otherwise.}
\end{cases}
\]

This is the right numerator for an admission score because it answers:

> How much useful energy drop should this candidate produce on the current scaffold, under the refit policy I will actually execute, without taking a state-space step that is too large?

---

## Tangent-Space Novelty

A candidate should not be rewarded if its tangent direction is already spanned by the unlocked scaffold window.

Define the centered tangent direction
\[
t_{m,p}
=
-i\big(\widetilde A_{m,p}-\langle\widetilde A_{m,p}\rangle_\psi\big)|\psi\rangle.
\]

For tangent vectors \(\{t_j\}_{j\in W(p)}\) of the unlocked window, define
\[
[s_{m,p}]_j = \mathrm{Re}\,\langle t_j|t_{m,p}\rangle,
\qquad
[S_{W(p)}]_{jk} = \mathrm{Re}\,\langle t_j|t_k\rangle.
\]

Then define novelty
\[
N_{m,p}
=
1
-
\frac{s_{m,p}^{\top}(S_{W(p)}+\epsilon_N I)^{-1}s_{m,p}}{F_{m,p}}.
\]

Numerically clip \(N_{m,p}\) to \([0,1]\).

Interpretation:
- \(N_{m,p}\approx 1\): mostly outside the current window tangent span.
- \(N_{m,p}\approx 0\): mostly redundant with the current window.

Novelty is geometric; it is more principled than a crude family-repeat penalty.

---

## Lifetime Hardware Burden

A candidate should be ranked by predicted benefit per effective marginal burden.

Define
\[
K_{m,p}
=
1
+
w_D \bar D_{m,p}
+
w_G \bar G^{\mathrm{new}}_{m,p}
+
w_C \bar C^{\mathrm{new}}_{m,p}
+
w_P \bar P_{m,p}
+
w_c \bar c_m.
\]

Where:
- \(D_{m,p}\): compiled depth / entangler / 2-qubit burden increment;
- \(G^{\mathrm{new}}_{m,p}\): number of new commuting measurement groups introduced;
- \(C^{\mathrm{new}}_{m,p}\): extra shots required after reuse;
- \(P_{m,p}\): optimizer-dimension burden;
- \(c_m\): family reuse count or repeat count.

All costs should be normalized:
\[
\bar x = \frac{x}{x_{\mathrm{ref}}+\epsilon},
\]
where \(x_{\mathrm{ref}}\) is either a hard budget or a current-pool robust statistic such as a median.

### Additive, not multiplicative
The denominator should be additive because wall-clock cost, new measurement groups, shots, and optimizer dimension are first-order additive resources.
Do **not** multiply unrelated penalty factors.

---

## Primary Target Score

For candidate \(m\), evaluate a small set of allowed positions \(\mathcal P_m\).

Define the candidate score as

\[
S_m
=
\max_{p\in\mathcal P_m}
\left[
\mathbf 1(m\in\Omega_{\mathrm{stage}})
\mathbf 1(L_{m,p}\le L^{(\mathrm{stage})}_{\max})
e^{-\eta_L L_{m,p}}
N_{m,p}^{\gamma_N}
\frac{\Delta \widehat E^{\mathrm{TR}}_{m,p}}{K_{m,p}}
\right].
\]

Where:
- \(\Omega_{\mathrm{stage}}\): active pool for the current curriculum stage;
- \(L_{m,p}\): symmetry/leakage penalty;
- \(L_{\max}^{(\mathrm{stage})}\): hard leakage cap for the stage;
- \(\eta_L\): soft leakage penalty scale;
- \(\gamma_N\): novelty exponent.

This is the **full target score**, not the initial implementation requirement.

---

## Simplified Production-First Score

The first implementation may use a reduced score, provided the interface is built for a later upgrade.

### Simplified score
\[
S^{\mathrm{simple}}_m
=
\max_{p\in\mathcal P_m}
\left[
\mathbf 1(m\in\Omega_{\mathrm{stage}})
\mathbf 1(L_{m,p}\le L^{(\mathrm{stage})}_{\max})
e^{-\eta_L L_{m,p}}
\frac{
\big(|\hat g_{m,p}| - z_\alpha \hat \sigma_{m,p}\big)_+^2
}{
2 \lambda_F F_{m,p}
\left(1+w_D \bar D_{m,p}+w_C \bar C^{\mathrm{new}}_{m,p}+w_c \bar c_m\right)
}
\right].
\]

Interpretation:
- \(g_{\mathrm{lcb}}^2/(2\lambda_F F)\) is a Levenberg–Marquardt / trust-region surrogate for predicted energy drop;
- \(F\) gives proper generator normalization;
- hardware costs enter additively.

### Mandatory design requirement
Even if the first implementation uses `simple_v1`, the code architecture must already reserve slots for:
- \(h_{m,p}\),
- \(b_{m,p}\),
- \(H_{W(p)}\),
- \(N_{m,p}\),
- \(G^{\mathrm{new}}_{m,p}\),
- \(P_{m,p}\).

The score engine should therefore expose a stable feature container that can later be filled more richly.

---

## Why the Simplified Score Is Acceptable First

The simplified score is acceptable for an initial implementation because it already fixes the biggest defects of raw \(|g|\)-ranking:

1. **Scale fairness** through \(F\).
2. **Noise awareness** through \(g^\downarrow\).
3. **Resource awareness** through additive burden terms.
4. **Position awareness** by maximizing over allowed positions.
5. **Curriculum awareness** through stage gating and leakage caps.

What it does **not** yet capture:
- window-relaxed curvature via the Schur complement;
- tangent-space novelty;
- optimizer dimension burden;
- new measurement-group burden as a distinct term.

These can be added later without changing the external score interface.

---

## Control-Plane Architecture

The implementation should be split into control planes.

### 1. Scaffold plane
Owns the current executable ansatz:
- generator sequence;
- angles;
- insertion positions;
- replay family;
- compiled artifacts;
- layout/decomposition choices;
- measurement partitions;
- optimizer memory.

### 2. Pool/stage plane
Owns:
- active pool stage;
- stage transitions;
- `full_meta` promotion rules;
- allowed macro-generator families.

### 3. Scoring plane
Owns:
- feature estimation;
- simple and full scores;
- shortlist logic;
- position maximization.

### 4. Measurement plane
Owns:
- grouped Pauli measurement planning;
- cache keys and cache reuse;
- variance estimates;
- shot allocation.

### 5. Optimization plane
Owns:
- windowed refits;
- full-prefix refits;
- replay freeze/unfreeze;
- trust-region protection;
- QN-SPSA refresh.

### 6. Pruning plane
Owns:
- prune metrics;
- prune acceptance rules;
- post-prune refit.

### 7. Telemetry plane
Owns:
- per-depth score features;
- candidate shortlists;
- cache hits/misses;
- position probes;
- batch compatibility decisions;
- prune decisions;
- replay phase changes.

The implementation should not collapse all of this into a single monolithic ADAPT loop.

---

## End-to-End Pipeline

The intended pipeline is

\[
\text{HH tiny seed}
\to
\text{ADAPT core}
\to
\text{plateau diagnosis}
\to
\text{controlled residual enrichment}
\to
\text{full refit}
\to
\text{prune}
\to
\text{post-prune refit}
\to
\text{scaffolded replay}.
\]

### Stage A: tiny HH seed
Use a deliberately small 2–6 parameter seed to remove the most obvious HH structure before greedy selection begins.

The seed should target:
- phonon displacement;
- leading drag/current-drag behavior;
- possibly the lightest HVA-inspired block needed to capture obvious coupled electron–phonon structure.

The seed should **not** become a second adaptive algorithm.
Its purpose is entropy reduction, not full optimization.

### Stage B: ADAPT core with narrow physics-aligned pool
The active pool should be restricted to the narrow HH core:
- PAOP displacement-focused directions;
- PAOP/LF drag-like directions;
- any minimal HVA/UCCSD-lifted directions needed to avoid missing obvious couplings.

The pool should remain small enough that:
- gradient ranking is meaningful;
- measurement cost is tractable;
- compiled-depth penalties remain informative.

### Stage C: plateau diagnosis
Before opening a larger pool, diagnose whether the plateau is:
- genuine convergence; or
- a gradient trough caused by append-position bias.

The first action is **not** “open `full_meta`”.
The first action is **probe alternative insertion positions**.

### Stage D: controlled residual enrichment
If plateau persists after insertion probes, open a **small residual enrichment stage** from `full_meta` or lifted-UCCSD/HVA families.

Admission must be conservative:
- persistence across checkpoints,
- novelty beyond the current scaffold span,
- compiled-cost cap,
- limited number of residual admissions per epoch.

### Stage E: full-prefix refit
Do a full-prefix refit before pruning.

### Stage F: pruning
Prune redundant or fading operators only after a full-prefix refit.

### Stage G: post-prune refit
Refit once after pruning.

### Stage H: replay
Replay around the cleaned scaffold:
- residual-only burn-in;
- then controlled unfreeze of inherited parameters;
- optionally periodic QN-SPSA refresh.

---

## Pool Curriculum

The curriculum is

\[
\text{seed} \;\to\; \text{core} \;\to\; \text{residual enrichment}.
\]

### Stage definitions

#### Seed stage
Allowed operators:
- tiny HH-specific seed family only.

Use case:
- warm-start / entropy reduction.
- no large adaptive branching.

Exit condition:
- fixed seed depth reached or marginal gain saturates.

#### Core ADAPT stage
Allowed operators:
- PAOP/LF core;
- minimal HVA/UCCSD-lifted support if explicitly needed.

Use case:
- main physics-aligned adaptive growth.

Exit condition:
- repeated low-drop depths **and**
- no significant alternative insertion position discovered.

#### Residual enrichment stage
Allowed operators:
- small admitted subset of `full_meta`.

Use case:
- fill residual expressivity gaps only after the core has genuinely plateaued.

Exit condition:
- persistence rule fails for all candidates;
- or post-admission improvement falls below stage threshold;
- or cost cap reached.

---

## `full_meta` Admission Rule

Residual `full_meta` operators must **not** be admitted based on a single gradient spike.

A candidate residual operator should satisfy all of:

1. **Persistence**
   - It remains near the top of the shortlist under both raw residual gradient and normalized score for \(p_{\mathrm{persist}}\) consecutive checkpoints.

2. **Novelty**
   - Its tangent direction has significant component outside the span of the current scaffold directions or the active refit window:
     \[
     N_{m,p} \ge N_{\min}^{\mathrm{resid}}.
     \]

3. **Compiled cost cap**
   - Its compiled marginal burden is below a residual-stage cap:
     \[
     D_{m,p} \le D_{\max}^{\mathrm{resid}}.
     \]

4. **Statistical significance**
   - The lower-confidence gradient is positive:
     \[
     g^\downarrow_{m,p} > 0.
     \]

Admit at most \(1\)–\(2\) residual operators per enrichment epoch.

### Rationale
Residual enrichment must remain an exception layer, not a silent reversion to “full pool from the start”.

---

## Insertion Position as a Control Variable

Append-only selection can produce a **gradient trough**:
the append position may appear flat even when the same or similar operator inserted elsewhere has a much larger gradient.

### Allowed positions
For each candidate \(m\), consider a small set \(\mathcal P_m\), not all positions.

The default low-cost set should be:
- append;
- prepend only when trough probing is triggered;
- one midpoint cut;
- one heuristic interior cut.

### Good heuristic interior cuts
A generic implementation should support cut-selection by any of:
- boundary between seed and core;
- midpoint of scaffold depth;
- boundary of the active refit window;
- cut near largest cumulative \(|\theta_j|\);
- cut near recent admitted block(s);
- cut at macro-generator family boundaries.

The initial implementation may use a simple fixed subset:
\[
\mathcal P_m = \{\text{append}\}
\]
during normal growth, and
\[
\mathcal P_m = \{\text{append}, \text{prepend}, \text{midpoint}, \text{window-boundary}\}
\]
during trough probes.

### Trough-probe trigger
Probe alternative positions when all of the following hold:
1. recent accepted depths give low energy-error drop;
2. append-position top score is below a historical or stage-relative threshold;
3. stage is not already residual enrichment;
4. the optimizer is not currently in a temporary noisy spike regime.

### Trough declaration
Declare a trough if:
\[
\max_{p \neq \text{append}} S_{m,p}
\ge
\tau_{\mathrm{pos}}
\cdot
\max_{p = \text{append}} S_{m,p}
\]
for some candidate, with \(g^\downarrow > 0\).

Then choose the best-position candidate rather than opening a bigger pool.

### Genuine convergence
Only if the probed non-terminal positions are also flat should the algorithm treat the plateau as genuine.

---

## Windowed Refit Integration

The score must use the same window that the optimizer will actually unlock after admission.

Let \(W(p)\) be the active window for insertion at \(p\).
This window may depend on the already implemented policy, e.g.
- newest \(w\) parameters,
- plus top-\(k\) legacy parameters by \(|\theta_j|\) or sensitivity.

### Why this matters
If the score predicts post-admission behavior assuming one set of relaxed inherited parameters, but the optimizer actually unlocks a different set, then the score is optimizing the wrong continuation protocol.

### Rule
Any refit-aware score must receive the exact `W(p)` from the same planner that will control the post-admission optimizer.

Do not duplicate the window selection logic in multiple inconsistent places.

---

## Curvature Estimation Ladder

The design should support multiple accuracy levels.

### Level 0: no curvature data
Use the simplified score:
\[
\Delta \widehat E \approx \frac{g_{\mathrm{lcb}}^2}{2\lambda_F F}.
\]

### Level 1: diagonal surrogate
Approximate
\[
\widetilde h_{m,p} \approx \lambda_F F_{m,p}
\]
or another diagonal metric/Hessian proxy.

### Level 2: low-rank recycled window curvature
Use recycled optimizer memory to obtain:
- approximate inverse Hessian on the active window;
- approximate candidate-window cross curvature.

### Level 3: explicit Schur complement
Estimate:
- \(h_{m,p}\),
- \(b_{m,p}\),
- \(H_{W(p)}\),
and compute
\[
\widetilde h_{m,p}
=
h_{m,p}
-
b_{m,p}^{\top}(H_{W(p)}+\lambda_H I)^{-1}b_{m,p}.
\]

### Practical rule
- Full-pool screening uses Level 0 or 1.
- Only the shortlist uses Level 2 or 3.

---

## Batch Selection

Batched ADAPT should be limited and conditional.

### When batching is allowed
Only in early ADAPT, and only when:
1. candidate scores are near-degenerate;
2. compiled placement is compatible;
3. support overlap and cross-curvature penalties are small;
4. the batch size is small (preferably \(2\), at most \(3\)).

### When batching is not allowed
- late ADAPT;
- residual enrichment stage;
- replay.

### Compatibility penalty
For a candidate pair \(m,n\), define a compatibility penalty
\[
\Pi_{mn}
=
w_{\mathrm{ov}} O_{mn}
+
w_{\mathrm{comm}} C_{mn}
+
w_{\mathrm{curv}} X_{mn}
+
w_{\mathrm{sched}} Y_{mn},
\]
where:
- \(O_{mn}\): support overlap penalty;
- \(C_{mn}\): noncommutation / ordering penalty;
- \(X_{mn}\): pairwise cross-curvature proxy;
- \(Y_{mn}\): compiled scheduling incompatibility penalty.

Then select a small batch \(B\) by maximizing
\[
\sum_{m\in B} S_m - \sum_{m<n\in B}\Pi_{mn}.
\]

The first implementation may use a greedy approximation.

### Batch ordering
Even within a batch, preserve a deterministic order:
- descending score;
- then deterministic tie-break;
- then stable position ordering.

---

## Macro-Generators and Selective Splitting

A generator should be treated as a first-class **macro-generator** when:
- it is composed of commuting pieces on the same support or a compilation-friendly support pattern;
- it preserves the target symmetries;
- it has a natural circuit template;
- grouping the pieces reduces compiled depth or entangler count.

### When to keep as a macro-generator
Keep the generator grouped if the grouped implementation:
- has lower compiled cost,
- preserves symmetry more naturally,
- does not destroy measurement efficiency.

### When to split
Split only if:
- the grouped generator has unacceptable compiled depth;
- internal noncommutation forces a poor circuit template;
- finer-grained pruning or novelty analysis is necessary;
- the grouped parameter is too coarse for replay continuation.

### Repository-level rule
Selection, replay, pruning, and telemetry should all operate on the **generator object as implemented on hardware**, not on an artificial Pauli-atom expansion, unless a deliberate split has been chosen.

---

## Symmetry Used Twice

Symmetry should be used twice:
1. to constrain construction;
2. to mitigate bias/noise.

### In construction
Every pool family should carry symmetry metadata:
- particle number preservation;
- spin / sector preservation;
- phonon-number or other model-specific constraints where appropriate;
- known leakage risks.

Use symmetry metadata for:
- stage gating,
- hard rejection,
- leakage penalty \(L_{m,p}\),
- macro-generator grouping eligibility.

### In mitigation
The same conserved quantities should feed a post-measurement mitigation hook:
- symmetry verification,
- symmetry expansion,
- postselection-like filtering where appropriate.

### Architectural rule
Construction and mitigation should reference the same `SymmetrySpec` object or equivalent shared metadata source, not two independent ad hoc representations.

---

## Measurement Reuse as a First-Class Subsystem

Measurement reuse should be elevated from “optimization” to “architecture”.

### Two distinct reuse layers
#### Layer A: observable-planning reuse
Independent of the current parameter values:
- grouped Pauli partitions;
- basis-rotation templates;
- commuting-group plans;
- transform maps needed for gradient/metric/curvature observables.

This layer should be reusable across nearby states and across repeated ADAPT iterations.

#### Layer B: measurement-result reuse
Specific to the current executable state:
- measured means;
- variances;
- shot counts;
- confidence estimates;
- covariance summaries if available.

This layer is valid only for the exact circuit context in which it was measured.

### Cache keys
A measurement cache key should include at least:
- executable scaffold fingerprint;
- insertion-position context if relevant;
- compiled layout/decomposition fingerprint;
- observable group identity;
- backend/noise profile;
- symmetry-mitigation mode;
- shot allocation profile.

### Minimum cache behavior
The measurement subsystem must support:
1. **cache lookup** before new measurement requests;
2. **partial reuse** when only some observable groups are missing;
3. **variance-aware shot extension** for groups that exist but need tighter precision;
4. **separate tracking** of reused vs newly acquired shots.

### Score integration
The score must consume:
- \(\hat g_{m,p}\),
- \(\hat \sigma_{m,p}\),
- \(G^{\mathrm{new}}_{m,p}\),
- \(C^{\mathrm{new}}_{m,p}\),
all from the measurement subsystem.

### Simulator/noiseless rule
The current exact-statevector compiled \(H|\psi\rangle\) acceleration remains the preferred backend in the noiseless path.
The measurement subsystem should sit beside it as the hardware/noisy path, not replace it.

---

## Suggested Measurement-Subsystem API

The design should support generic interfaces of the form:

```python
class MeasurementPlanner:
    def plan_energy_groups(self, hamiltonian, scaffold_ctx): ...
    def plan_gradient_groups(self, candidate, position, scaffold_ctx): ...
    def plan_metric_groups(self, candidate, position, scaffold_ctx): ...
    def plan_curvature_groups(self, candidate, position, window_ctx, scaffold_ctx): ...
```

```python
class MeasurementCache:
    def lookup(self, cache_key): ...
    def store(self, cache_key, grouped_results): ...
    def extend_shots(self, cache_key, additional_budget): ...
```

```python
class ShotAllocator:
    def allocate(self, requested_groups, target_precision, prior_stats): ...
```

The first implementation may stub `plan_curvature_groups` and use only:
- energy groups,
- gradient groups,
- metric groups.

But the API should already exist.

---

## Scoring Workflow

The scoring workflow should be two-stage.

### Stage 1: cheap pass on full active pool
For each candidate \(m\) and allowed position \(p\):
- stage gate;
- leakage gate;
- estimate \(\hat g_{m,p}\), \(\hat \sigma_{m,p}\), \(F_{m,p}\);
- estimate cheap burdens \(D_{m,p}\), \(C^{\mathrm{new}}_{m,p}\), \(c_m\);
- compute `simple_v1`.

### Stage 2: full pass on shortlist
Take the top \(M\) candidates (per candidate family or globally, depending on stage).

For each shortlisted \((m,p)\):
- estimate/refine \(h_{m,p}\);
- estimate/refine \(b_{m,p}\);
- obtain or recycle \(H_{W(p)}\);
- estimate \(N_{m,p}\);
- refine \(D, G^{\mathrm{new}}, C^{\mathrm{new}}, P\);
- compute `full_v2`.

### Final selection
Choose:
- best candidate/position if single-add mode;
- small best-compatible batch if batch mode.

### Why shortlist
The expensive quantities are only worthwhile on a small shortlist:
- curvature,
- tangent novelty,
- compiled full placement cost,
- pairwise compatibility penalties.

---

## Compile-Cost Oracle

Hardware awareness requires an explicit compile-cost oracle.

### Oracle outputs
For an admitted candidate at position \(p\), the oracle should estimate:
- marginal 2-qubit gate count increment;
- marginal 2-qubit depth increment;
- marginal single-qubit cost if desired;
- marginal native-block count;
- estimated remaining-lifetime burden:
  \[
  D^{\mathrm{life}}_{m,p} = N_{\mathrm{rem}} \, D^{\mathrm{per\,eval}}_{m,p}.
  \]

### Important rule
The cost must be evaluated on the **actual executable scaffold + candidate at position \(p\)**, not on the abstract naked generator.

### First implementation
A first implementation may use a proxy, e.g.
- compiled depth increment;
- compiled entangler proxy;
- template-level gate proxy.

But the interface must remain capable of later replacing the proxy with an actual compiled marginal burden.

---

## Optimizer Policy

### Default optimizer philosophy
- Plain SPSA remains the general-purpose optimizer for warm-start and most ADAPT behavior.
- QN-SPSA is deployed selectively where conditioning matters most.

### QN-SPSA deployment rule
Use QN-SPSA:
- only in replay;
- or only every \(M\)-th replay iteration as a preconditioning refresh;
- or simulator-only as a diagnostic/curvature oracle if hardware budget is tight.

### Why
Replay is where:
- the scaffold is already useful;
- conditioning matters more;
- extra metric-aware cost is more justified.

### Optimizer-memory persistence
Parameter recycling is not enough.
The scaffold should preserve:
- inverse-Hessian approximations when available;
- diagonal/low-rank curvature memory;
- QGT/Fisher approximations if already built;
- per-parameter SPSA scales;
- recent gradient/variance summaries.

This memory should survive:
- depth growth;
- pruning;
- replay handoff.

---

## Trust-Region Replay Handoff

Replay should begin as a continuation method.

### Phase 1: residual-only burn-in
- inherited scaffold parameters frozen;
- newly introduced residual parameters active;
- residual parameters zero-initialized.

### Phase 2: constrained unfreeze
- inherited parameters become active;
- inherited-parameter motion is restricted by either:
  - a freeze-then-unfreeze schedule; or
  - an explicit trust region
    \[
    \|\delta\theta_{\mathrm{old}}\|_2 \le \rho_t.
    \]

### Phase 3: fully active replay
- all parameters active;
- optional periodic QN-SPSA refresh.

### Relaxation schedule
A simple trust-radius schedule:
\[
\rho_t = \rho_0 \, r^t
\]
with \(r>1\), clipped to a maximum.
Any monotone relaxation schedule is acceptable if deterministic and logged.

### Replay invariant
Do not allow replay to degenerate into a disguised fixed-structure restart.

---

## Compiled Artifacts Are Part of the Scaffold

The scaffold is not just:
- operator labels,
- parameter vector.

It also includes:
- compiled/native decomposition choices;
- qubit layout/routing choices when present;
- measurement partitions;
- calibration-friendly block structure;
- macro-generator templates;
- optimizer memory.

### Rule
Whenever possible, replay and continuation stages should inherit these artifacts rather than rebuild them from scratch.

### Why
In a hardware-aware setting, the physical executable circuit is part of the optimization object.

---

## Pruning

Pruning should occur **after** a full-prefix refit and **before** replay.

### Why this timing
Early noisy ADAPT growth is the wrong time to prune.
After a full refit:
- operator usefulness is more honestly expressed;
- stale parameters are more visible;
- pruning decisions are less distorted by temporary optimizer lag.

### Prune criteria
A generator becomes prune-eligible if several of the following hold:
1. \(|\theta_j|\) is persistently small.
2. local re-zeroing causes negligible predicted or measured energy loss after local refit.
3. novelty relative to neighboring scaffold directions is very small.
4. its compiled cost is high compared to its retained usefulness.
5. it is dominated by a nearby macro-generator or successor direction.
6. its removal lowers replay-noise sensitivity or optimizer dimension substantially.

### Prune acceptance rule
For a candidate removal \(j\):
1. temporarily remove \(j\);
2. locally refit the relevant window or full scaffold;
3. accept removal if energy regression is within a tolerance and no hard symmetry constraint is violated.

### Group-aware rule
Macro-generators should be pruned as units unless a deliberate split has already occurred.

### Post-prune action
Always perform one refit after pruning.

---

## Suggested Prune Metrics

For each scaffold block \(j\), track:
- magnitude score:
  \[
  M_j = |\theta_j|;
  \]
- sensitivity score:
  \[
  R_j = |g_j^{\mathrm{reprobe}}|;
  \]
- novelty score:
  \[
  N_j^{\mathrm{local}};
  \]
- compiled burden:
  \[
  D_j^{\mathrm{life}};
  \]
- removal regression:
  \[
  \Delta E_j^{\mathrm{remove}}.
  \]

A generic prune priority could be
\[
P_j^{\mathrm{prune}}
=
\frac{a_M M_j + a_R R_j + a_N N_j^{\mathrm{local}}}{1 + a_D D_j^{\mathrm{life}}}
\]
with candidates considered in ascending \(P_j^{\mathrm{prune}}\), but actual acceptance should still rely on direct removal testing.

---

## HH Seed Design

The HH seed should be:
- tiny,
- polaronic,
- noncompeting with ADAPT.

### Target
A 2–6 parameter seed that removes the most obvious HH structure:
- displacement-like modes;
- drag/current-drag leading behavior;
- perhaps one minimal HVA-like coupling block.

### Failure mode to avoid
If the seed becomes too expressive, it competes with the adaptive stage and makes the adaptive stage harder to interpret.

### Design rule
The seed should be a small structured preconditioner, not a second variational algorithm.

---

## Position-Aware Scoring + Refit-Aware Cost = Correct Ranking Objective

A candidate must be ranked by:
1. what it does at the chosen insertion position;
2. after the planned refit;
3. per lifetime cost.

This means the score is not a property of an operator alone.
It is a property of the tuple:
\[
(\text{candidate family}, \text{position}, \text{current scaffold}, \text{planned refit window}, \text{backend}, \text{measurement state}).
\]

This is why a `CandidateContext` object is required in code.

---

## Candidate Feature Object

A generic implementation should make the score consume a single feature bundle.

```python
@dataclass
class CandidateFeatures:
    candidate_id: str
    family_id: str
    position_id: int
    stage_name: str

    g_hat: float
    sigma_hat: float
    F_metric: float

    h_hat: float | None
    b_hat: np.ndarray | None
    H_window: np.ndarray | None
    novelty: float | None

    depth_cost: float
    new_group_cost: float
    new_shot_cost: float
    opt_dim_cost: float
    reuse_count_cost: float

    leakage_penalty: float
    allowed_in_stage: bool
    passes_hard_symmetry_gate: bool

    refit_window_indices: list[int]
    compiled_position_cost_proxy: dict[str, float]
    measurement_cache_stats: dict[str, float]
```

The first implementation may leave some fields as `None`, but the structure should exist.

---

## Suggested Stateful Objects

```python
@dataclass
class ScaffoldState:
    generators: list["GeneratorInstance"]
    angles: np.ndarray
    stage_name: str
    compiled_artifacts: "CompiledScaffoldArtifacts | None"
    measurement_plan: "MeasurementPlan | None"
    optimizer_memory: "OptimizerMemory | None"
    handoff_state_kind: str
    metadata: dict
```

```python
@dataclass
class GeneratorInstance:
    generator_id: str
    family_id: str
    template_id: str
    supports_macro_mode: bool
    symmetry_spec: dict
    compile_metadata: dict
```

```python
@dataclass
class PoolStageConfig:
    stage_name: str
    allowed_families: set[str]
    hard_leakage_cap: float
    novelty_exponent: float
    max_residual_admissions: int
    batch_size_cap: int
```

```python
@dataclass
class ReplayPlan:
    residual_reps: int
    freeze_scaffold_steps: int
    trust_radius_initial: float | None
    trust_radius_growth: float | None
    qn_spsa_refresh_every: int | None
```

```python
@dataclass
class PruneDecision:
    generator_id: str
    accepted: bool
    delta_energy_remove: float
    local_refit_window: list[int]
    rationale: dict
```

---

## Generic Interfaces

### Stage controller
```python
class StageController:
    def current_stage(self, scaffold: ScaffoldState) -> PoolStageConfig: ...
    def maybe_promote_stage(self, scaffold: ScaffoldState, telemetry: dict) -> PoolStageConfig: ...
```

### Insertion policy
```python
class InsertionPolicy:
    def allowed_positions(self, scaffold: ScaffoldState, candidate) -> list[int]: ...
    def should_probe_positions(self, history, scaffold) -> bool: ...
```

### Score engine
```python
class ScoreEngine:
    def cheap_features(self, scaffold, candidate, position, backend_ctx) -> CandidateFeatures: ...
    def full_features(self, scaffold, candidate, position, backend_ctx) -> CandidateFeatures: ...
    def simple_score(self, feat: CandidateFeatures, cfg) -> float: ...
    def full_score(self, feat: CandidateFeatures, cfg) -> float: ...
```

### Compile-cost oracle
```python
class CompileCostOracle:
    def marginal_cost(self, scaffold, candidate, position, backend_ctx) -> dict[str, float]: ...
```

### Novelty oracle
```python
class NoveltyOracle:
    def estimate(self, scaffold, candidate, position, window_ctx, backend_ctx) -> float: ...
```

### Curvature oracle
```python
class CurvatureOracle:
    def estimate_self_curvature(self, scaffold, candidate, position, backend_ctx) -> float: ...
    def estimate_cross_curvature(self, scaffold, candidate, position, window_ctx, backend_ctx) -> np.ndarray: ...
    def estimate_window_hessian(self, scaffold, window_ctx, backend_ctx) -> np.ndarray: ...
```

### Pruner
```python
class Pruner:
    def rank_prune_candidates(self, scaffold, backend_ctx) -> list[PruneDecision]: ...
    def apply_pruning(self, scaffold, decisions) -> ScaffoldState: ...
```

### Replay controller
```python
class ReplayController:
    def build_replay_plan(self, scaffold, runtime_cfg) -> ReplayPlan: ...
    def run_replay(self, scaffold, replay_plan, backend_ctx) -> dict: ...
```

These interfaces can later be bound to actual repo modules.

---

## End-to-End Pseudocode

```python
def adapt_continue_pipeline(problem_ctx, runtime_cfg, backend_ctx):
    scaffold = load_or_initialize_scaffold(problem_ctx, runtime_cfg, backend_ctx)
    stage_ctl = StageController(...)
    insert_ctl = InsertionPolicy(...)
    score_engine = ScoreEngine(...)
    batch_selector = BatchSelector(...)
    pruner = Pruner(...)
    replay_ctl = ReplayController(...)

    while not termination_condition(scaffold, runtime_cfg):
        stage_cfg = stage_ctl.current_stage(scaffold)

        probe_positions = insert_ctl.should_probe_positions(scaffold.metadata["history"], scaffold)
        candidate_records = []

        for candidate in enumerate_stage_pool(stage_cfg, problem_ctx, scaffold):
            for position in insert_ctl.allowed_positions(scaffold, candidate) if probe_positions else [append_position(scaffold)]:
                feat = score_engine.cheap_features(scaffold, candidate, position, backend_ctx)
                s0 = score_engine.simple_score(feat, runtime_cfg.score_cfg)
                candidate_records.append((s0, candidate, position, feat))

        shortlist = shortlist_candidates(candidate_records, runtime_cfg.shortlist_cfg)

        full_records = []
        for _, candidate, position, cheap_feat in shortlist:
            feat = score_engine.full_features(scaffold, candidate, position, backend_ctx)
            s = score_engine.full_score(feat, runtime_cfg.score_cfg) \
                if runtime_cfg.score_cfg.mode == "full_v2" else \
                score_engine.simple_score(feat, runtime_cfg.score_cfg)
            full_records.append((s, candidate, position, feat))

        if stage_cfg.batch_size_cap > 1 and runtime_cfg.enable_batching:
            selected = batch_selector.select(full_records, scaffold, backend_ctx, stage_cfg)
        else:
            selected = [max(full_records, key=lambda x: x[0])]

        if not selected or max(r[0] for r in selected) <= runtime_cfg.min_accept_score:
            stage_cfg = stage_ctl.maybe_promote_stage(scaffold, telemetry_from(full_records, scaffold))
            if stage_cfg.stage_name != scaffold.stage_name:
                scaffold.stage_name = stage_cfg.stage_name
                continue
            else:
                break

        scaffold = admit_selected_candidates(scaffold, selected, runtime_cfg, backend_ctx)
        scaffold = refit_under_current_policy(scaffold, selected, runtime_cfg, backend_ctx)
        scaffold = maybe_periodic_full_refit(scaffold, runtime_cfg, backend_ctx)
        scaffold.metadata["history"].append(telemetry_from_step(scaffold, selected, full_records))

    scaffold = full_prefix_refit(scaffold, runtime_cfg, backend_ctx)
    prune_decisions = pruner.rank_prune_candidates(scaffold, backend_ctx)
    scaffold = pruner.apply_pruning(scaffold, prune_decisions)
    scaffold = post_prune_refit(scaffold, runtime_cfg, backend_ctx)

    replay_plan = replay_ctl.build_replay_plan(scaffold, runtime_cfg)
    replay_result = replay_ctl.run_replay(scaffold, replay_plan, backend_ctx)

    return assemble_artifacts(scaffold, replay_result)
```

---

## Cheap Score Pseudocode

```python
def cheap_score(feat, cfg):
    if not feat.allowed_in_stage:
        return 0.0
    if not feat.passes_hard_symmetry_gate:
        return 0.0
    if feat.leakage_penalty > cfg.leakage_cap:
        return 0.0

    g_lcb = max(abs(feat.g_hat) - cfg.z_alpha * feat.sigma_hat, 0.0)
    if g_lcb <= 0.0:
        return 0.0
    if feat.F_metric <= 0.0:
        return 0.0

    K = (
        1.0
        + cfg.wD * normalize(feat.depth_cost, cfg.depth_ref)
        + cfg.wC * normalize(feat.new_shot_cost, cfg.shot_ref)
        + cfg.wc * normalize(feat.reuse_count_cost, cfg.reuse_ref)
    )

    delta_e = 0.5 * g_lcb * g_lcb / (cfg.lambda_F * feat.F_metric)
    return math.exp(-cfg.eta_L * feat.leakage_penalty) * delta_e / K
```

---

## Full Score Pseudocode

```python
def trust_region_drop(g_lcb, h_eff, F, rho):
    if g_lcb <= 0.0 or F <= 0.0:
        return 0.0
    h_eff = max(h_eff, 0.0)
    alpha_max = rho / (F ** 0.5)
    if h_eff > 0.0:
        alpha_newton = g_lcb / h_eff
        if alpha_newton <= alpha_max:
            return 0.5 * g_lcb * g_lcb / h_eff
    alpha = alpha_max
    return g_lcb * alpha - 0.5 * h_eff * alpha * alpha
```

```python
def full_score(feat, cfg):
    if not feat.allowed_in_stage:
        return 0.0
    if not feat.passes_hard_symmetry_gate:
        return 0.0
    if feat.leakage_penalty > cfg.leakage_cap:
        return 0.0

    g_lcb = max(abs(feat.g_hat) - cfg.z_alpha * feat.sigma_hat, 0.0)
    if g_lcb <= 0.0 or feat.F_metric <= 0.0:
        return 0.0

    novelty = 1.0 if feat.novelty is None else min(max(feat.novelty, 0.0), 1.0)

    if feat.h_hat is None or feat.b_hat is None or feat.H_window is None:
        h_eff = cfg.lambda_F * feat.F_metric
    else:
        Hreg = feat.H_window + cfg.lambda_H * np.eye(feat.H_window.shape[0])
        h_eff = feat.h_hat - feat.b_hat.T @ np.linalg.solve(Hreg, feat.b_hat)

    delta_e = trust_region_drop(g_lcb, h_eff, feat.F_metric, cfg.rho)

    K = (
        1.0
        + cfg.wD * normalize(feat.depth_cost, cfg.depth_ref)
        + cfg.wG * normalize(feat.new_group_cost, cfg.group_ref)
        + cfg.wC * normalize(feat.new_shot_cost, cfg.shot_ref)
        + cfg.wP * normalize(feat.opt_dim_cost, cfg.optdim_ref)
        + cfg.wc * normalize(feat.reuse_count_cost, cfg.reuse_ref)
    )

    return (
        math.exp(-cfg.eta_L * feat.leakage_penalty)
        * (novelty ** cfg.gamma_N)
        * delta_e
        / K
    )
```

---

## Position-Probe Pseudocode

```python
def should_probe_positions(history, cfg):
    if len(history) < cfg.trough_probe_min_depth:
        return False

    recent = history[-cfg.trough_probe_patience:]
    low_drop = all(h["energy_drop_abs"] < cfg.trough_drop_floor for h in recent)
    low_append = recent[-1]["append_best_score"] < cfg.trough_append_score_floor

    return low_drop and low_append
```

```python
def detect_gradient_trough(append_best, alt_best, cfg):
    return alt_best > 0.0 and alt_best >= cfg.trough_ratio * append_best
```

---

## Batch Selection Pseudocode

```python
def greedy_batch_select(ranked_records, compat_oracle, cfg):
    batch = []
    total_score = 0.0

    for record in ranked_records:
        s, cand, pos, feat = record
        if len(batch) >= cfg.batch_size_cap:
            break

        penalty = 0.0
        for existing in batch:
            penalty += compat_oracle.penalty(record, existing)

        if s - penalty > 0.0:
            batch.append(record)
            total_score += s - penalty

    return batch
```

---

## Pruning Pseudocode

```python
def prune_scaffold(scaffold, backend_ctx, cfg):
    candidates = rank_blocks_for_pruning(scaffold, backend_ctx, cfg)

    kept = scaffold
    decisions = []

    for block in candidates:
        trial = remove_block(kept, block)
        trial = local_refit_after_removal(trial, block, backend_ctx, cfg)
        delta_e = trial.energy - kept.energy

        accept = (
            delta_e <= cfg.prune_energy_tolerance
            and preserves_required_symmetry(trial, cfg)
        )

        decisions.append({
            "generator_id": block.generator_id,
            "accepted": accept,
            "delta_energy_remove": delta_e,
        })

        if accept:
            kept = trial

    kept = post_prune_refit(kept, backend_ctx, cfg)
    kept.metadata["prune_decisions"] = decisions
    return kept
```

---

## Replay Pseudocode

```python
def run_scaffolded_replay(scaffold, replay_plan, backend_ctx, cfg):
    params = initialize_replay_params(scaffold, replay_plan, cfg)  # scaffold + zero residual

    # phase 1: residual-only burn-in
    active_mask = residual_only_mask(params, scaffold, replay_plan)
    params = optimize(params, active_mask=active_mask, method=cfg.base_optimizer, backend_ctx=backend_ctx)

    # phase 2: constrained unfreeze
    active_mask = all_params_mask(params)
    trust = replay_plan.trust_radius_initial
    for step in range(replay_plan.unfreeze_steps):
        params = optimize(
            params,
            active_mask=active_mask,
            trust_radius_old=trust,
            method=cfg.base_optimizer,
            backend_ctx=backend_ctx,
        )
        if replay_plan.qn_spsa_refresh_every and step % replay_plan.qn_spsa_refresh_every == 0:
            params = optimize(
                params,
                active_mask=active_mask,
                trust_radius_old=trust,
                method="QN-SPSA",
                backend_ctx=backend_ctx,
                precondition_only=True,
            )
        trust = min(trust * replay_plan.trust_radius_growth, cfg.max_trust_radius)

    # phase 3: full replay
    params = optimize(params, active_mask=active_mask, method=cfg.base_optimizer, backend_ctx=backend_ctx)
    return params
```

---

## Novelty Estimation Notes

A first implementation may not be able to compute exact tangent-span novelty on hardware.

### Acceptable approximation ladder
1. **Exact simulator tangent novelty**
   - Use explicit statevector tangent overlaps.

2. **Approximate novelty from compiled-action tangents**
   - If tangent vectors can be produced cheaply.

3. **Cheap proxy**
   - Family-level novelty proxy,
   - support-overlap proxy,
   - or novelty omitted in `simple_v1`.

### Rule
If novelty is omitted in `simple_v1`, do not fake it with an arbitrary family penalty.
Instead, leave the interface open and rely on stage gating + repeat cost until a better novelty estimate is available.

---

## Curvature Estimation Notes

A first implementation may also lack robust \(h,b,H\) estimates.

### Acceptable approximation ladder
1. **Simulator finite differences**
   - use exact energy evaluations;
2. **Quasi-Newton recycle**
   - inherit approximate inverse-Hessian data from previous optimizer steps;
3. **QN-SPSA / metric proxy**
   - use a metric-aware surrogate in replay;
4. **LM surrogate**
   - use \(h_{\mathrm{eff}} \approx \lambda_F F\) in `simple_v1`.

### Rule
`simple_v1` should not block `full_v2`.
The code must already have an abstraction boundary for curvature estimation.

---

## Cost Normalization and Weights

A clean generic normalization is
\[
\bar x = \frac{x}{x_{\mathrm{ref}}+\epsilon}.
\]

### Reference choice
Use one of:
- fixed budget;
- rolling median over active pool;
- rolling median over accepted candidates.

### Recommended initial weights
\[
w_D = w_G = w_C = 1,
\qquad
w_P = 0 \text{ if every candidate adds one parameter},
\qquad
w_c \in [0.1, 0.5].
\]

### Confidence multiplier
- early screening:
  \[
  z_\alpha \approx 1;
  \]
- late residual enrichment:
  \[
  z_\alpha \in [1.5, 2].
  \]

### Novelty exponent
- early targeted ADAPT:
  \[
  \gamma_N = 1;
  \]
- late residual enrichment:
  \[
  \gamma_N \approx 1/2.
  \]

### Leakage softness
Choose \(L_{\mathrm{soft}}\) and set
\[
\eta_L = \frac{\ln 10}{L_{\mathrm{soft}}},
\]
so that \(L=L_{\mathrm{soft}}\) reduces the score by one order of magnitude.

---

## Lifetime Weighting

Any recurring cost should be lifetime weighted when possible.

If a newly admitted block is expected to persist for \(N_{\mathrm{rem}}\) remaining objective evaluations, then convert per-evaluation cost \(x_{\mathrm{per}}\) to

\[
x_{\mathrm{life}} = N_{\mathrm{rem}} x_{\mathrm{per}}.
\]

This applies most naturally to:
- compiled depth burden;
- measurement burden for repeated evaluations;
- optimizer dimension burden.

The implementation does not need a perfect \(N_{\mathrm{rem}}\) model initially.
A deterministic coarse proxy is sufficient.

---

## Termination and Stage-Transition Logic

The continuation logic should use uncertainty-aware energy-drop tracking.

### Preferred primary signal
Use completed-depth energy-error drop:
\[
\mathrm{drop}(d)
=
\Delta E_{\mathrm{abs}}(d-1)-\Delta E_{\mathrm{abs}}(d),
\]
where \(\Delta E_{\mathrm{abs}}\) is the absolute error to a trusted simulator reference or the best available confidence-aware hardware estimate.

### Secondary signals
- max score;
- max lower-confidence gradient;
- cache growth burden;
- residual-stage persistence failures.

### Do not use
A raw append-position gradient floor should not be the sole stop reason.

### Stage transition rule
The first response to low drop is:
1. probe insertion positions;
2. if still flat, maybe promote to residual enrichment;
3. if residual enrichment is already exhausted, end ADAPT and move to prune/replay.

---

## Telemetry Requirements

Every depth should record enough telemetry to audit the selection.

### Minimum per-depth telemetry
- stage name;
- shortlist candidates;
- allowed positions considered;
- cheap and full scores for shortlisted candidates;
- chosen candidate and position;
- refit window indices;
- lower-confidence gradient;
- metric \(F\);
- curvature proxy level used;
- novelty estimate if available;
- compiled burden terms;
- measurement new-group and new-shot costs;
- cache hit/miss counts;
- batch compatibility penalties if batching used;
- trough-probe trigger and verdict;
- stage-transition trigger if any.

### Pruning telemetry
- prune candidates considered;
- removal regression;
- local refit range;
- acceptance/rejection reason.

### Replay telemetry
- replay seed policy actually used;
- phase transitions;
- trust radius schedule;
- active parameter masks;
- QN-SPSA refresh points;
- optimizer-memory reuse status.

This telemetry is essential for later tuning.

---

## Validation and Mathematical Invariants

The implementation should ship with explicit invariants.

### Score invariants
1. **Rescaling invariance of full score numerator**
   - Under \(A\mapsto \kappa A\), the trust-region predicted drop should remain invariant.

2. **Metric positivity**
   - \(F_{m,p}\ge 0\).

3. **Novelty bounds**
   - \(0\le N_{m,p}\le 1\) after clipping.

4. **Noise suppression**
   - If \(|\hat g|\le z_\alpha \hat \sigma\), the candidate score must be zero.

5. **Homogeneous-pool collapse**
   - If cost, metric, curvature, novelty, and leakage are equal across candidates, ranking should reduce to gradient-based ranking.

### Position invariants
6. If only append is allowed, the selection logic must reduce to append-only scoring.
7. If append is flat but a probed alternate position is not, the trough detector must be able to discover it.

### Window invariants
8. The score’s `refit_window_indices` must exactly match the optimizer’s actual active window after admission.

### Measurement invariants
9. Cache hits may change cost, but not the noiseless numerical value of an observable.
10. Variance-aware shot extension must reduce estimator uncertainty monotonically.

### Pruning invariants
11. Accepted prune operations must satisfy the configured post-refit energy-regression tolerance.
12. Hard symmetry constraints must remain preserved after pruning.

### Replay invariants
13. Residual replay layers must be zero-initialized by default.
14. Provenance-aware replay semantics must not silently regress.
15. Freeze-only burn-in must leave scaffold parameters unchanged during that phase.

---

## Recommended Test Categories

### Unit tests
- trust-region closed form;
- novelty calculation;
- leakage gate;
- score monotonicity in additive costs;
- position-max logic;
- batch compatibility penalty;
- prune accept/reject thresholds.

### Integration tests
- core ADAPT loop with `simple_v1`;
- stage transition after trough probe;
- residual enrichment persistence;
- prune-before-replay;
- replay trust-region schedule;
- measurement cache reuse parity;
- macro-generator retention/splitting behavior.

### Regression tests
- preserve current compiled gradient caching parity;
- preserve current replay provenance behavior;
- preserve current windowed refit behavior;
- preserve repository operator conventions.

---

## Implementation Phases

## Phase 1 — Minimal viable continuation architecture

### Objective
Ship a coherent first version that materially improves the current pipeline without requiring full curvature/QGT machinery.

### Implement
1. pool curriculum stages;
2. tiny HH seed stage hooks;
3. position-probe / trough detection;
4. `simple_v1` score;
5. compile-cost proxy integration;
6. measurement-cache interfaces and cache accounting;
7. prune-before-replay;
8. replay telemetry for freeze/unfreeze and seed provenance.

### Allowed simplifications
- novelty omitted or proxied;
- curvature omitted and replaced by \(\lambda_F F\);
- batching disabled or limited to a trivial compatible pair test;
- measurement reuse limited to grouped Pauli reuse and shot accounting;
- no motif tiling yet;
- no overlap-guided rescue mode yet.

### Deliverable
A stable end-to-end adaptive continuation loop with the right architecture.

---

## Phase 2 — Full shortlist scoring and replay conditioning

### Objective
Upgrade the ranking objective and replay optimizer state.

### Implement
1. shortlist-only `full_v2` score;
2. novelty estimation;
3. Schur-complement curvature or a strong low-rank proxy;
4. QN-SPSA replay refresh;
5. optimizer-memory persistence;
6. batch selection with compatibility penalties.

### Deliverable
A mathematically closer approximation to the desired predicted-drop-per-cost ranking.

---

## Phase 3 — Scaling and advanced hardware integration

### Objective
Add scaling and hardware sophistication.

### Implement
1. motif extraction and tiling;
2. macro-generator/native template library;
3. symmetry expansion / mitigation hooks;
4. overlap-guided rescue mode for simulator diagnostics;
5. improved lifetime cost models.

### Deliverable
A hardware-oriented, scalable continuation framework.

---

## Likely Repo Touchpoints (To Confirm by Audit)

These are likely integration surfaces inferred from current repo documentation; a repo-aware agent should confirm them before editing.

### Likely orchestration touchpoints
- `pipelines/hardcoded/adapt_pipeline.py` for the ADAPT control loop, scoring, pruning hooks, and telemetry.
- `pipelines/hardcoded/hh_vqe_from_adapt_family.py` for scaffolded replay, provenance-aware replay seeds, and replay optimizer phases.
- `src/quantum/operator_pools/` or equivalent HH pool builders for stage-aware pool curriculum and macro-generator metadata.
- `src/quantum/compiled_polynomial.py` and `src/quantum/compiled_ansatz.py` or equivalent compiled-action utilities for executable-scaffold reuse, tangent/metric helper hooks, and cost proxies.
- measurement grouping / reporting / JSON artifact writers for cache accounting and telemetry emission.
- test suites covering ADAPT, replay, windowed refits, and compiled parity.


### Likely “do not rewrite” surfaces
- operator algebra core;
- canonical PauliTerm source;
- JW ladder/operator source of truth.

### Likely new service modules
A clean implementation would likely introduce new service modules or shims for:
- stage control;
- score engine;
- insertion policy;
- compile-cost oracle;
- measurement cache;
- curvature/novelty oracles;
- pruning engine;
- replay controller.

The exact module names should be chosen by the repo-aware agent after audit.

---

## Required Anti-Patterns to Avoid

1. **Do not reopen `full_meta` from depth 0.**
2. **Do not score by raw \(|g|\) alone.**
3. **Do not assume append-only gradients diagnose true convergence.**
4. **Do not multiply unrelated penalty factors in the denominator.**
5. **Do not tile inherited angles across replay layers by default.**
6. **Do not prune during noisy early growth.**
7. **Do not atomize every macro-generator into Pauli factors.**
8. **Do not discard optimizer memory when the ansatz grows.**
9. **Do not rebuild the executable scaffold from abstract operator labels if compiled artifacts can be preserved.**
10. **Do not introduce architectural dependence on unavailable curvature estimates.**
    Build the interfaces first; fill them progressively.

---

## Optional Simulator-Only Rescue Mode

A simulator-only rescue mode may be kept for diagnostics.

### Purpose
When ordinary adaptive growth is clearly trapped in a bad local-minimum structure,
allow an overlap-guided or alternate-objective pass to produce a compact rescue scaffold.

### Rules
- simulator-only by default;
- off the main QPU path;
- used only when ordinary continuation diagnostics indicate failure;
- result should feed back into the normal scaffold representation.

This is a diagnostic valve, not the default control logic.

---

## Motif Tiling for Larger Systems

When scaling in system size \(L\), the implementation should be able to learn operator motifs on smaller systems and lift/tile them.

### Motif extraction
After solving a smaller instance, extract:
- accepted local operator families;
- relative ordering motifs;
- retained macro-generator blocks;
- common parameter sign/magnitude patterns;
- local symmetry metadata.

### Motif library
Store motifs in a transferable representation:
- local support pattern;
- family/template id;
- translation rules;
- admissible boundary behavior.

### Larger-\(L\) use
Use the motif library to:
- seed the pool,
- seed the scaffold,
- initialize replay families.

This is not Phase 1 work, but the operator metadata should be designed so motifs can later be extracted.

---

## Suggested JSON/Artifact Payload Additions

Actual field names are for the repo-aware agent to decide, but the payload should eventually capture:

### Per depth
- `stage_name`
- `positions_considered`
- `selected_position`
- `cheap_score`
- `full_score`
- `g_lcb`
- `F_metric`
- `curvature_mode`
- `novelty`
- `depth_cost`
- `new_group_cost`
- `new_shot_cost`
- `reuse_count_cost`
- `leakage_penalty`
- `refit_window_indices`
- `trough_probe_triggered`
- `trough_detected`
- `batch_selected`

### Pruning
- `prune_candidates`
- `prune_accepted`
- `delta_energy_remove`
- `post_prune_refit`

### Replay
- `replay_phase_history`
- `freeze_steps`
- `trust_radius_schedule`
- `qn_spsa_refresh_steps`
- `optimizer_memory_reused`

This telemetry will be indispensable for later audit and tuning.

---

## Open Integration Questions for the Repo-Aware Agent

These are not blockers to the present design specification, but they must be resolved during audit:

1. Which concrete pool-builder functions currently materialize HH pool families?
2. Where should generator metadata live so macro-generators, symmetry specs, and motif extraction can all reuse it?
3. What compile-cost proxy already exists in the codebase, if any?
4. What measurement grouping/planning utilities already exist, and can they be wrapped rather than replaced?
5. Where can optimizer-memory state be safely persisted without breaking existing JSON contracts?
6. Which test modules currently guard replay provenance and windowed refits?
7. What exact JSON schema changes are acceptable without breaking downstream scripts?
8. Which reporting utilities should surface the new telemetry?
9. Can current compiled-action utilities also support tangent-vector estimation, or is a separate novelty oracle needed?
10. Which existing modules already expose enough state to fingerprint the executable scaffold for cache keys?

These should be answered in the subsequent repo-specific implementation audit.

---

## Final Implementation Rule

The implementation is successful only if it satisfies all of the following:

1. The adaptive path remains a **continuation method**, not a disguised restart.
2. The selected score ranks **predicted useful energy decrease per marginal lifetime cost**.
3. `full_meta` becomes a **controlled residual enrichment stage**, not the default pool.
4. Insertion-position bias is diagnosed before convergence is declared.
5. Measurement reuse is explicit and auditable.
6. Pruning happens before replay.
7. The scaffold preserves compiled/hardware-relevant artifacts.
8. The first implementation may be simplified, but its architecture must already make the advanced score, novelty, curvature, and motif machinery pluggable without a redesign.

---

## Minimal “First Build” Checklist

This is the smallest coherent build that still matches the architecture:

- [ ] Add stage controller with `seed`, `core`, `residual`.
- [ ] Add position-probe trigger and alternative position evaluation.
- [ ] Implement `CandidateFeatures`.
- [ ] Implement `simple_v1` score with \(g^\downarrow\), \(F\), additive costs, stage gate, leakage gate.
- [ ] Add compile-cost oracle proxy.
- [ ] Add measurement-cache interfaces and shot accounting.
- [ ] Add prune-before-replay with post-prune refit.
- [ ] Preserve replay continuation semantics and provenance-aware seed behavior.
- [ ] Emit per-depth telemetry sufficient to audit every admission decision.
- [ ] Keep interfaces open for novelty, full curvature, QN-SPSA refresh, and motif tiling.

If those boxes are checked, the repo will have the correct skeleton and can be upgraded incrementally.

---

## Appendix A — Symbol Dictionary

\(|\psi\rangle\): current state prepared by the executable scaffold.

\(A_m\): candidate Hermitian generator block as parameterized on hardware.

\(p\): insertion position in the current scaffold.

\(\mathcal P_m\): allowed insertion positions for candidate \(m\).

\(\widetilde A_{m,p}\): candidate dressed by the scaffold suffix after position \(p\).

\(\hat g_{m,p}\): estimated zero-initialized gradient.

\(\hat \sigma_{m,p}\): standard deviation of the gradient estimator.

\(g^\downarrow_{m,p}\): lower-confidence gradient magnitude.

\(F_{m,p}\): Fubini–Study metric element / generator variance.

\(W(p)\): actual inherited-parameter window to be unlocked after admission at \(p\).

\(H_{W(p)}\): Hessian or Hessian proxy over the unlocked window.

\(h_{m,p}\): candidate self-curvature.

\(b_{m,p}\): candidate–window cross-curvature vector.

\(\widetilde h_{m,p}\): effective curvature after window relaxation.

\(t_{m,p}\): candidate tangent vector in Hilbert space.

\(N_{m,p}\): novelty, i.e. tangent-norm fraction outside the active window span.

\(D_{m,p}\): compiled depth / entangler burden.

\(G^{\mathrm{new}}_{m,p}\): number of new measurement groups.

\(C^{\mathrm{new}}_{m,p}\): additional shots required after reuse.

\(P_{m,p}\): optimizer dimension burden.

\(c_m\): family repeat/reuse count.

\(L_{m,p}\): symmetry/leakage penalty.

\(\Omega_{\mathrm{stage}}\): currently active pool stage.

\(\rho\): state-space trust radius.

\(z_\alpha\): confidence multiplier.

\(\lambda_H\): Hessian regularization ridge.

\(\lambda_F\): metric-curvature surrogate scale for simplified scoring.

\(\gamma_N\): novelty exponent.

\(\eta_L\): softness of the leakage penalty.

\(N_{\mathrm{rem}}\): estimated remaining objective evaluations after candidate admission.

---

## Appendix B — Immediate Post-Audit Questions for a Coding Agent

1. Where is the current ADAPT candidate-evaluation loop centralized?
2. Where is the replay family and seed policy currently resolved?
3. What object already fingerprints the compiled scaffold?
4. What operator-pool object can be extended with family/template/symmetry metadata?
5. Where should measurement cache state live so it can be shared by both energy and ADAPT steps?
6. Which current refit planner can expose the exact `W(p)` window to the score engine?
7. What current JSON history object can absorb the new telemetry with minimal disruption?

These questions should be answered before repo-level edits begin.

---
