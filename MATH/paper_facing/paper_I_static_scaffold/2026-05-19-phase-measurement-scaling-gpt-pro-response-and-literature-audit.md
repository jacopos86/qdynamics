# Paper I Phase Measurement Scaling: GPT-Pro Response and Literature Audit

Created: 2026-05-19

Target manuscript: `MATH/paper_details/static_adapt_paper_I.tex`

Related handoff: `prompt-exports/2026-05-18-paper-i-phase-measurement-scaling-gpt-pro-handoff.md`

Purpose: preserve the GPT-Pro response supplied by the user, then audit which measurement-scaling claims are literature-backed, which are definition/design claims internal to SNAKE, and which should be softened before use in Paper I.

Scope: Paper I only. This support note does not edit the manuscript.

## 1. GPT-Pro response supplied by the user

Note: the response below is transcribed into Markdown with display math normalized for readability. The claim content and section structure are preserved.

### 1. Terminology Verdict

The recommended terminology is correct. The scalar

\[
h_r=
\left.\frac{d^2}{d\alpha^2}E_r(\alpha;t)\right|_{\alpha=0}
\]

should be called the candidate directional energy curvature. A precise secondary phrase is the candidate-candidate scalar Hessian entry in the augmented local coordinate system. It should not be called simply "the Hessian," because Phase III introduces actual matrix-valued Hessian/refit objects \(H_{W_r}\) and \(R_r\). This distinction is central to the uploaded handoff.

The clean local terminology is

\[
H_r^{\rm local}=\begin{pmatrix}
h_r & c_r^\top\\
c_r & H_{W_r}
\end{pmatrix},
\qquad
R_r=\frac12\left(H_{W_r}+H_{W_r}^\top\right),
\]

where

\[
h_r=\text{candidate directional energy curvature},
\]

\[
c_r=\text{candidate-window mixed energy-curvature vector},
\]

\[
H_{W_r}=\text{window-window energy-Hessian block},
\]

\[
R_r=\text{symmetrized/resolved window Hessian block}.
\]

The word "energy" should remain in the phrase for \(h_r\), because \(F_r,q_r,Q_r\) are geometric Fubini--Study/tangent-Gram quantities, not energy curvatures.

### 2. Phase-by-Phase Scaling Table

Let \(R_k\) be the set of records evaluated in Phase \(k\), with

\[
n_k:=|R_k|,
\qquad
n_0\ge n_1\ge n_2\ge n_3,
\]

and let

\[
m_r:=|W_r|,
\qquad
m:=\max_{r\in R_2\cup R_3}m_r
\]

when a uniform window-size bound is desired.

For an object class

\[
x\in\{g,F,h,q,Q,c,R\},
\]

write

\[
P_a^{(x)}=\text{raw Pauli-term count for object/entry }a,
\]

\[
G_a^{(x)}=\text{number of grouped measurement settings for object/entry }a,
\]

and, for shot accounting,

\[
B_a^{(x)}=\sum_{\ell=1}^{G_a^{(x)}}N_{a,\ell}^{(x)},
\]

where \(N_{a,\ell}^{(x)}\) is the number of shots assigned to group \(\ell\). At target standard error \(\epsilon_x\), a generic variance-limited estimate obeys the schematic law

\[
N_{a,\ell}^{(x)}=O\!\left(\frac{\sigma_{a,\ell}^2}{\epsilon_{a,\ell}^2}\right),
\]

but the actual allocation depends on the estimator, grouping, covariance reuse, and the precision needed by the downstream score.

| Phase | New mathematical objects | Raw object count per survivor \(r\) | Incremental Pauli-term count, no reuse | Incremental grouped-setting burden, no reuse | Expected reuse/caching | Dominant overclaim risk |
|---|---|---:|---|---|---|---|
| 0 | Pilot gradient or resolved gradient proxy \(\widetilde g_r(t)\); cheap metadata features | 1 quantum gradient signal, plus non-measurement metadata | \(\sum_{r\in R_0}P_r^{(g)}\) | \(\sum_{r\in R_0}G_r^{(g)}\), or \(O(n_0G^{(g)})\) | Hamiltonian-term reuse, commutator support reuse, shared Pauli grouping across candidates, lower-confidence early shots | Claiming all cheap selector features are quantum-measured; claiming a fixed gradient-shot multiplier independent of estimator |
| I | Fubini--Study tangent scale \(F_r=\|Q_\psi\tau_r\|^2\) | 1 scalar geometric norm | \(\sum_{r\in R_1}P_r^{(F)}\) | \(\sum_{r\in R_1}G_r^{(F)}\), or \(O(n_1G^{(F)})\) | \(F_r\) may be analytic for special generators; may be cached as a QGT diagonal; may share tangent-overlap circuits with later \(q_r,Q_r\) | Conflating \(F_r\) with energy curvature; assuming \(F_r\) always requires new quantum measurements |
| II | Candidate directional energy curvature \(h_r\); candidate-window tangent overlaps \(q_r\); inherited-window Gram block \(Q_r\) | \(1+m_r+m_r(m_r+1)/2\) | \(\sum_{r\in R_2}\big(P_r^{(h)}+\sum_{i\in W_r}P_{r,i}^{(q)}+\sum_{i\le j\in W_r}P_{r,ij}^{(Q)}\big)\) | \(\sum_{r\in R_2}\big(G_r^{(h)}+\sum_{i\in W_r}G_{r,i}^{(q)}+\sum_{i\le j\in W_r}G_{r,ij}^{(Q)}\big)\), or \(O(n_2[G^{(h)}+mG^{(q)}+m^2G^{(Q)}])\) | \(F_r\) reused from Phase I; \(Q_r\) highly cacheable when windows overlap; \(q_r\) may share tangent-overlap structure; \(h_r\) may share curvature machinery with later Hessian entries | Calling \(h_r\) "the Hessian"; treating \(q_r,Q_r\) as energy-Hessian objects; assuming every \(Q_r\) block is newly measured per candidate |
| III | Mixed energy-curvature vector \(c_r\); symmetrized/resolved refit Hessian block \(R_r\) | \(m_r+m_r(m_r+1)/2\), assuming symmetry/symmetrization | \(\sum_{r\in R_3}\big(\sum_{j\in W_r}P_{r,j}^{(c)}+\sum_{j\le k\in W_r}P_{r,jk}^{(R)}\big)\) | \(\sum_{r\in R_3}\big(\sum_{j\in W_r}G_{r,j}^{(c)}+\sum_{j\le k\in W_r}G_{r,jk}^{(R)}\big)\), or \(O(n_3[mG^{(c)}+m^2G^{(R)}])\) | Reuse \(h_r,F_r,q_r,Q_r\); \(R_r\) can be cached across candidates sharing \(W_r\); Hessian recycling may reduce the \(m^2\) term; only \(c_r\) is strongly candidate-specific | Counting \(h_r^\star,q_r^\star,F_r^\star,N_3\) as separately measured; claiming Phase III is measurement-free; running Phase III over the full candidate pool |

A compact total grouped-setting upper bound, before cross-candidate grouping/reuse, is

\[
\begin{aligned}
\Gamma_{\le {\rm III}}
\lesssim {}&
\sum_{r\in R_0}G_r^{(g)}
+\sum_{r\in R_1}G_r^{(F)}\\
&+\sum_{r\in R_2}
\left(
G_r^{(h)}
+\sum_{i\in W_r}G_{r,i}^{(q)}
+\sum_{i\le j\in W_r}G_{r,ij}^{(Q)}
\right)\\
&+\sum_{r\in R_3}
\left(
\sum_{j\in W_r}G_{r,j}^{(c)}
+\sum_{j\le k\in W_r}G_{r,jk}^{(R)}
\right).
\end{aligned}
\]

Under uniform grouped-cost bounds,

\[
\Gamma_{\le {\rm III}}
=
O\!\left(
 n_0G^{(g)}
+n_1G^{(F)}
+n_2\left[G^{(h)}+mG^{(q)}+m^2G^{(Q)}\right]
+n_3\left[mG^{(c)}+m^2G^{(R)}\right]
\right).
\]

The analogous pre-grouping Pauli-term count is

\[
\Pi_{\le {\rm III}}
=
O\!\left(
 n_0P^{(g)}
+n_1P^{(F)}
+n_2\left[P^{(h)}+mP^{(q)}+m^2P^{(Q)}\right]
+n_3\left[mP^{(c)}+m^2P^{(R)}\right]
\right),
\]

again before cancellations, common support, Pauli grouping, shared circuits, or cached matrix blocks.

Shot cost is not the same object as grouped-setting count. A backend-neutral statement is

\[
S_{\le {\rm III}}
=
\sum_{\ell\in U_{\le {\rm III}}}N_\ell,
\qquad
N_\ell=O\!\left(\frac{\sigma_\ell^2}{\epsilon_\ell^2}\right),
\]

where \(U_{\le {\rm III}}\) is the union of all grouped settings actually measured after grouping and reuse. The tolerances \(\epsilon_\ell\) should be allocated to control uncertainty in the phase scores \(\Delta E_k\), novelty scores \(N_k\), and the damped Schur quantities. Poor conditioning of \(Q_r^+\) or \(R_r^{-1}\) can amplify measurement error, so the shot budget is not determined by object count alone.

### 3. Algebraic vs Quantum-Measured Objects

The distinction is:

> quantum measurement supplies primitive scalar entries; scoring formulas are then classical algebra.

Phase 0 quantum-measured object:

\[
\widetilde g_r(t)
\]

or whatever lower-confidence gradient proxy is used. Cheap cost features, support metadata, commutation metadata, and compiled-circuit burden estimates are generally classical unless the implementation defines them through measured observables.

Phase I quantum-measured or analytically supplied object:

\[
F_r=\|Q_\psi\tau_r\|^2.
\]

The Phase-I score

\[
\Delta E_1(r)=
\max_{|\alpha|\le \rho/\sqrt{F_r}}
\left[\widetilde g_r|\alpha|-\frac12\lambda F_r\alpha^2\right]
\]

is algebraic after \(\widetilde g_r,F_r\) are available.

Phase II quantum-measured or estimator-supplied objects:

\[
h_r,
\qquad
(q_r)_i,\ i\in W_r,
\qquad
(Q_r)_{ij},\ i\le j,\ i,j\in W_r.
\]

The following are algebraic after those entries are available:

\[
N_2(r;t)=1-\frac{q_r^\top Q_r^+q_r}{F_r},
\]

\[
\Delta E_2(r)=
\max_{|\alpha|\le \rho/\sqrt{F_r}}
\left[\widetilde g_r|\alpha|-\frac12h_r\alpha^2\right].
\]

Phase III quantum-measured or estimator-supplied objects:

\[
(c_r)_j=
\left.\frac{\partial^2 E}{\partial\alpha\,\partial(\delta\theta_j)}\right|_0,
\qquad j\in W_r,
\]

\[
(R_r)_{jk},
\qquad j\le k,
\qquad j,k\in W_r.
\]

Equivalently, one may first estimate \(H_{W_r}\) and then form

\[
R_r=\frac12\left(H_{W_r}+H_{W_r}^\top\right)
\]

classically. If the estimator directly targets the symmetrized bilinear form, then \(R_r\) is the primitive measured object. If the estimator produces nonsymmetric noisy estimates of \(H_{jk}\) and \(H_{kj}\), the symmetrization policy must be specified.

The Phase-III starred objects are algebraic:

\[
\delta\theta_W^\star=-\alpha R_r^{-1}c_r,
\]

\[
h_r^\star=h_r-c_r^\top R_r^{-1}c_r,
\]

\[
q_r^\star=q_r-Q_rR_r^{-1}c_r,
\]

\[
F_r^\star
=F_r-2q_r^\top R_r^{-1}c_r
+c_r^\top R_r^{-1}Q_rR_r^{-1}c_r,
\]

\[
N_3(r;t)=1-\frac{(q_r^\star)^\top Q_r^+q_r^\star}{F_r^\star},
\]

\[
\Delta E_3(r)=
\max_{|\alpha|\le \rho/\sqrt{F_r^\star}}
\left[\widetilde g_r|\alpha|-\frac12h_r^\star\alpha^2\right].
\]

Thus Phase III does add quantum-measurement burden through \(c_r\) and \(R_r\), but it does not require separate direct measurement of \(h_r^\star,F_r^\star,q_r^\star,N_3\), or \(\Delta E_3\).

### 4. Shortlisting and Survivor-Count Scaling

The staged selector is viable because the expensive objects are acquired only for survivors:

\[
R_0\supseteq R_1\supseteq R_2\supseteq R_3,
\qquad
n_0\ge n_1\ge n_2\ge n_3.
\]

The critical scaling distinction is not only "Phase III is \(O(m^2)\) per candidate," but rather

\[
\text{Phase III cost}\sim n_3\times O(m^2)
\]

in Hessian-block entries, not

\[
n_0\times O(m^2).
\]

Phase III incurs extra local Hessian/refit-window acquisition because Schur-refit reranking asks how the candidate's quadratic energy response changes after the inherited window \(W_r\) is allowed to relax. That relaxation requires the candidate-window mixed curvature vector \(c_r\) and the window-window curvature block \(R_r\). These are genuine local energy-curvature data beyond the scalar \(h_r\). Applying Phase III to the entire pool would convert the selector into broad Hessian acquisition over many candidate-window pairs, which defeats the staged design. It should therefore be applied only after aggressive shortlisting, with small \(n_3\) and controlled \(m_r\).

Caching can change the practical law. For example, if many candidates share the same inherited window \(W\), the block \(R_W\) is measured once and reused, reducing the apparent repeated cost

\[
\sum_{r\in R_3}\frac{m_r(m_r+1)}{2}G^{(R)}
\]

toward a sum over distinct windows:

\[
\sum_{W\in \mathcal W_3^{\rm distinct}}
\frac{|W|(|W|+1)}{2}G_W^{(R)}.
\]

Similarly, the inherited Gram blocks \(Q_r\) in Phase II may be cached over distinct windows rather than recomputed per candidate.

A more cache-aware schematic expression is

\[
\Gamma_{\rm II}
\approx
\sum_{r\in R_2}G_r^{(h)}
+\sum_{r\in R_2}\sum_{i\in W_r}G_{r,i}^{(q)}
+\sum_{W\in\mathcal W_2^{\rm distinct}}\sum_{i\le j\in W}G_{ij}^{(Q)},
\]

\[
\Gamma_{\rm III}
\approx
\sum_{r\in R_3}\sum_{j\in W_r}G_{r,j}^{(c)}
+\sum_{W\in\mathcal W_3^{\rm distinct}}\sum_{j\le k\in W}G_{jk}^{(R)}.
\]

This form is safer than claiming that every candidate always costs a fresh \(m_r^2\) block.

### 5. Estimator-Model Caveats

For a direct Pauli-derivative or commutator-observable approach, each derivative object is represented as an observable on the current state, for example by expanding a commutator or double-commutator expression into Pauli strings. The entry-count laws above remain the same, but the raw Pauli counts

\[
P^{(g)},P^{(h)},P^{(c)},P^{(R)}
\]

can vary substantially with Hamiltonian locality, generator support, commutator cancellation, and grouping. This approach may allow many quantities to be measured on the same unshifted state, which improves cross-object grouping and covariance reuse. The risk is Pauli-term proliferation in nested derivative observables.

For a parameter-shift Hessian approach, the primitive measurement is often an energy expectation at shifted parameter values. A second derivative or mixed derivative usually requires a constant number of shifted-energy evaluations per Hessian entry, but the constant depends on the generator spectrum and shift rule. The entry-count scaling is still

\[
O(n_2)
\]

for the scalar \(h_r\) terms and

\[
O(n_3[m+m^2])
\]

for \(c_r,R_r\), multiplied by the Hamiltonian measurement burden per shifted circuit. The grouped Hamiltonian settings may be the same across shifted circuits, but the circuits/states are different, so the shot cost multiplies by the number of shifts. This model should not be collapsed into a single universal constant unless the shift rule is specified.

For a finite-difference Hessian approach, \(h_r,c_r,R_r\) are inferred from energy differences at perturbed coordinates. A naive local Hessian over the coordinate set \(\{\alpha\}\cup W_r\) has dimension

\[
d_r=m_r+1
\]

and requires \(O(d_r^2)\) energy-stencil information per local model, before reuse. Finite differences introduce a step-size tradeoff: too small a step amplifies shot noise; too large a step introduces truncation error. The measurement burden is therefore tied to the finite-difference stencil, the Hamiltonian energy-measurement cost, and the chosen accuracy target.

For Hessian recycling, previously measured Hessian entries are reused across candidates, windows, or iterations. Then the effective Phase-III cost may be closer to "new boundary entries" than to a full fresh \(m_r^2\) block per candidate. However, recycling is only defensible with a specified state/update policy, because after ansatz growth and parameter reoptimization the Hessian entries are not literally unchanged. A support note should call this a caching or approximation strategy, not an intrinsic scaling guarantee.

For statevector or noiseless simulation, \(G^{(x)}\), Pauli grouping, and shot counts are development proxies rather than physical measurement costs. Simulation timing may be useful for implementation studies, but it is not a quantum-measurement burden.

### 6. Manuscript-Support Paragraph

A suitable support-note paragraph is:

> The selector is staged, and its measurement burden is controlled by the number of records surviving to each phase. Let \(n_k\) denote the number of candidate records evaluated in Phase \(k\), and let \(m_r=|W_r|\) be the size of the inherited refit window. Phase 0 estimates a pilot gradient signal \(\widetilde g_r\). Phase I adds the Fubini--Study tangent scale \(F_r\). Phase II adds the scalar candidate directional energy curvature \(h_r=d^2E_r/d\alpha^2|_0\), together with the tangent-overlap vector \(q_r\) and inherited Gram block \(Q_r\). Phase III adds local refit-window energy-curvature data, namely the mixed vector \(c_r\) and the symmetrized/resolved window Hessian block \(R_r\). Denoting grouped measurement costs by \(G^{(g)},G^{(F)},G^{(h)},G^{(q)},G^{(Q)},G^{(c)},G^{(R)}\), a backend-agnostic upper-bound form before cross-candidate reuse is
>
> \[
> O\!\left(
> n_0G^{(g)}+n_1G^{(F)}+n_2[G^{(h)}+mG^{(q)}+m^2G^{(Q)}]
> +n_3[mG^{(c)}+m^2G^{(R)}]
> \right).
> \]
>
> This expression counts grouped measurement settings, not a universal shot multiplier. The actual shot cost depends on the derivative estimator, Pauli expansion, grouping rule, variance, target precision, and caching policy. Once \(c_r\), \(R_r\), and the Phase-II geometric quantities are available, the Schur-relaxed quantities \(h_r^\star,q_r^\star,F_r^\star,N_3\), and \(\Delta E_3\) are computed algebraically and are not separate quantum observables.

### 7. What Not To Claim

Do not claim that \(h_r\) is "the Hessian." The accurate statement is

\[
h_r=\text{candidate directional energy curvature}
=\text{candidate-candidate scalar Hessian entry}.
\]

Do not claim that \(F_r,q_r,Q_r\) are energy-Hessian objects. They are Fubini--Study/tangent-Gram geometry objects.

Do not claim that Phase III is measurement-free. It is algebraic only after the additional energy-curvature entries \(c_r\) and \(R_r\) have been acquired.

Do not claim that \(h_r^\star,q_r^\star,F_r^\star,N_3\), or \(\Delta E_3\) require separate direct quantum measurement. They are classical consequences of previously estimated objects.

Do not claim a fixed numeric shot overhead for Phase III unless all of the following are specified:

\[
\text{estimator model},
\text{ Pauli expansion},
\text{ grouping rule},
\text{ shot-allocation rule},
\text{ target precision},
\text{ reuse/caching policy}.
\]

Do not identify grouped-setting count with shot count. Grouped settings count circuit/measurement configurations; shot count also depends on variances and error tolerances.

Do not state that every candidate in Phase II or III necessarily requires a fresh \(Q_r\) or \(R_r\) block. Overlapping windows may allow reuse.

Do not claim the scaling is \(O(n_0m^2)\) unless Phase III is actually applied to the full candidate pool. The staged law is shortlist-dependent:

\[
O(n_3m^2)
\]

for Phase-III block acquisition before caching.

Do not imply that \(R_r^{-1}\) is always a literal stable inverse. The safer statement is that the Schur solve uses a damped, regularized, or resolved inverse on a trusted support.

Do not use statevector/noiseless simulation timings as physical measurement overheads without saying they are simulation proxies.

### 8. Recommended Final Wording for the Support Note

Use this wording:

> In Phase II we denote by
>
> \[
> h_r=\left.\frac{d^2}{d\alpha^2}E_r(\alpha;t)\right|_{\alpha=0}
> \]
>
> the candidate directional energy curvature. Equivalently, \(h_r\) is the candidate-candidate scalar Hessian entry in the local augmented coordinate system, but it is not referred to as "the Hessian" because Phase III uses genuine window Hessian blocks. The Phase-III local quadratic model has block curvature
>
> \[
> \begin{pmatrix}
> h_r & c_r^\top\\
> c_r & H_{W_r}
> \end{pmatrix},
> \qquad
> R_r=\frac12\left(H_{W_r}+H_{W_r}^\top\right),
> \]
>
> where \(c_r\) contains candidate-window mixed energy curvatures and \(R_r\) is the symmetrized/resolved inherited-window Hessian block.
>
> The incremental quantum-measurement burden is shortlist-dependent. With \(n_k\) records entering Phase \(k\), \(m_r=|W_r|\), and grouped measurement costs \(G^{(x)}\) for object class \(x\), a measurement-model-agnostic upper-bound form is
>
> \[
> O\!\left(
> n_0G^{(g)}+n_1G^{(F)}+n_2[G^{(h)}+mG^{(q)}+m^2G^{(Q)}]
> +n_3[mG^{(c)}+m^2G^{(R)}]
> \right),
> \]
>
> before cross-candidate reuse and caching. The corresponding raw Pauli-term expression is obtained by replacing each \(G^{(x)}\) by the relevant pre-grouping Pauli count \(P^{(x)}\). Shot complexity is estimator-dependent and scales with variance and target precision, not merely with object count. Phase III adds measurement burden through \(c_r\) and \(R_r\), but once these entries and the Phase-II quantities \(h_r,F_r,q_r,Q_r\) are available, the Schur-relaxed quantities
>
> \[
> h_r^\star,q_r^\star,F_r^\star,N_3,\Delta E_3
> \]
>
> are computed by classical linear algebra. This is why the Schur-refit stage should be reserved for a small shortlist rather than applied to the full candidate pool.

## 2. Literature support and claim audit

### 2.1 Claims that are strongly supported by existing literature

| Candidate claim | Support status | Literature support | Notes for Paper I |
|---|---|---|---|
| ADAPT-style methods acquire quantum data for gradients/operator selection, so adaptive ansatz construction can trade lower circuit depth for increased measurements. | Strong | Grimsley et al. state that ADAPT-VQE uses necessary gradient measurements and likely increases shot count relative to UCCSD-VQE, while reducing depth/parameters; see Nature Communications 2019. | Use to justify why Phase 0/gradient data is a real measurement surface, not a free classical score. |
| ADAPT gradients can be measured through commutator expressions for each pool element. | Strong | Anastasiou et al. summarize ADAPT-VQE and write the commutator derivative formula; they also note pool choice impacts measurement cost. | Supports treating gradient-object counts as pool/candidate dependent. |
| Fubini--Study/QGT geometry is a state-space metric distinct from ordinary parameter displacement. | Strong | Stokes et al. describe quantum natural gradient as steepest descent with respect to quantum information geometry, the real part of the QGT/Fubini--Study metric. | Supports keeping \(F_r,q_r,Q_r\) in the geometry/tangent-Gram category, not the energy-Hessian category. |
| Pauli measurement grouping reduces measurement burden but grouping count is not the same as shot count. | Strong | Huggins et al. discuss Hamiltonian averaging and optimal measurement distribution across Pauli words; Crawford et al. emphasize finite-sampling error and optimal allocation among commuting collections; Yen/Izmaylov-type work optimizes measurable groups and covariance/allocation. | Supports separating raw Pauli count, grouped-setting count, and shot budget. |
| Shot budget depends on estimator variance and target precision, not just object count. | Strong | Huggins et al. and Crawford et al. both frame measurement burden in terms of finite sampling and allocation; Izmaylov/Yen work explicitly uses variance/covariance-aware measurement allocation. | Use the safer statement: object count is a structural upper bound; shots require estimator/variance/tolerance policy. |
| Parameter-shift derivative costs depend on the shift rule and generator spectrum; higher derivatives do not have one universal constant overhead. | Strong | Wierichs et al. analyze resource requirements for generalized parameter-shift rules and higher-order derivatives. | Supports caveat that \(h_r,c_r,R_r\) measurement cost is estimator-model dependent. |
| Hessian information and Hessian recycling are recognized ways to reduce ADAPT/VQA measurement/runtime cost, but reuse is policy-dependent. | Strong | Ramôa et al. explicitly target reducing measurement costs by recycling Hessian information in adaptive VQAs. | Supports calling reuse/caching a strategy, not a guaranteed intrinsic scaling law. |
| Reusing Pauli measurements and variance-based shot allocation can reduce ADAPT-VQE shot overhead. | Moderate to strong, but preprint-dependent if using Ikhtiarudin 2025. | Ikhtiarudin et al. propose reused Pauli measurements and variance-based shot allocation for ADAPT-VQE. | Use as supporting context, but flag as arXiv/preprint unless accepted status is confirmed. |

### 2.2 Claims that are SNAKE-internal definitions/design claims, not literature claims

| Candidate claim | Status | How to phrase safely |
|---|---|---|
| \(h_r\) is the candidate directional energy curvature. | Definition from the Paper-I local model. | Safe as a notation definition; cite no external literature needed. |
| \(c_r\) and \(R_r\) are Phase-III primitive local energy-curvature entries. | Definition from the Paper-I Schur-refit model. | Safe if introduced as "in our notation" or "we estimate" rather than as a universal ADAPT object. |
| \(h_r^\star,q_r^\star,F_r^\star,N_3,\Delta E_3\) are algebraic after \(h_r,F_r,q_r,Q_r,c_r,R_r\) are available. | Mathematical consequence of the Schur and least-squares definitions. | Safe. No external citation required, but the derivation should be included in the support note or appendix. |
| Phase III scales as \(O(n_3[mG^{(c)}+m^2G^{(R)}])\) before reuse under uniform grouped-cost assumptions. | Paper-I design/analysis claim. | Safe only with "before cross-candidate reuse," "under uniform grouped-cost bounds," and "grouped-setting burden, not shot complexity." |
| Phase III should be applied to a small shortlist. | Algorithm-design consequence. | Safe as an engineering/design implication of the scaling formula. |

### 2.3 Claims to avoid or replace

| Avoid this claim | Replace with this claim |
|---|---|
| "Phase III is measurement-free." | "Phase III adds primitive curvature measurements for \(c_r\) and \(R_r\); the starred quantities are then algebraic." |
| "\(h_r\) is the Hessian." | "\(h_r\) is the candidate directional energy curvature, equivalently the candidate-candidate scalar Hessian entry." |
| "\(F_r,q_r,Q_r\) are Hessian quantities." | "\(F_r,q_r,Q_r\) are Fubini--Study/tangent-Gram geometry quantities." |
| "Grouped setting count equals shot count." | "Grouped setting count is a structural measurement-configuration count; shot count additionally depends on variance, precision, allocation, and reuse." |
| "Phase III costs \(O(n_0m^2)\)." | "If Phase III were applied to the full pool, it would scale like \(O(n_0m^2)\); in the staged selector the intended pre-cache block-acquisition law is \(O(n_3m^2)\)." |
| "Each candidate always gets a fresh \(R_r\) block." | "A fresh \(R_r\) block is an upper-bound model; overlapping windows and Hessian recycling can reduce repeated acquisition." |
| "Statevector runtime measures physical measurement cost." | "Statevector runtime is a simulation proxy and must be separated from grouped-setting and shot budgets." |

## 3. New literature-backed wording candidates for Paper I

### 3.1 Short conservative support paragraph

The staged selector separates primitive quantum-estimated entries from classical score algebra. Phase 0 uses a pilot ADAPT gradient signal, consistent with the standard commutator-gradient measurement surface of ADAPT-VQE. Phase I adds Fubini--Study tangent geometry, which is a state-space metric quantity rather than an energy curvature. Phase II adds the scalar candidate directional energy curvature \(h_r\), the candidate-window tangent-overlap vector \(q_r\), and the inherited tangent-Gram block \(Q_r\). Phase III adds the local refit-window energy-curvature entries \(c_r\) and \(R_r\). Once these primitive entries are estimated, the Schur-relaxed quantities \(h_r^\star,q_r^\star,F_r^\star,N_3\), and \(\Delta E_3\) are computed by classical linear algebra. The structural grouped-setting burden before reuse is therefore shortlist-dependent, with the Phase-III term scaling as \(n_3[mG^{(c)}+m^2G^{(R)}]\) under uniform grouped-cost assumptions, while the actual shot budget remains estimator-, variance-, precision-, grouping-, and caching-dependent.

### 3.2 More manuscript-like wording

Our cost accounting distinguishes three levels: primitive observable entries, grouped measurement configurations, and shots. The staged SNAKE funnel is designed so that Hessian-block acquisition occurs only after aggressive shortlisting. In particular, Phase III does not require direct measurement of the starred Schur quantities; it requires the additional primitive curvature entries \(c_r\) and \(R_r\), after which \(h_r^\star\), \(q_r^\star\), \(F_r^\star\), \(N_3\), and \(\Delta E_3\) are algebraic. Thus, before cross-candidate reuse, the Phase-III grouped-setting contribution is naturally expressed as \(O(n_3[mG^{(c)}+m^2G^{(R)}])\), not as a full-pool \(O(n_0m^2)\) burden and not as a universal shot multiplier.

### 3.3 Terminology insertion

We call

\[
h_r=\left.\frac{d^2}{d\alpha^2}E_r(\alpha;t)\right|_{\alpha=0}
\]

the candidate directional energy curvature. It is the scalar candidate-candidate entry of the local augmented Hessian, but it is not referred to as "the Hessian" because the Phase-III Schur model also uses the candidate-window mixed curvature vector \(c_r\) and the inherited-window Hessian block \(R_r\). This terminology keeps energy curvature separate from the Fubini--Study/tangent-Gram geometry objects \(F_r,q_r,Q_r\).

## 4. Source notes

- Grimsley et al., "An adaptive variational algorithm for exact molecular simulations on a quantum computer," Nature Communications 10, 3007 (2019). DOI: https://doi.org/10.1038/s41467-019-10988-2. Supports canonical ADAPT-VQE, gradient-selected ansatz growth, and increased measurement burden from gradient measurements.
- Anastasiou et al., "Reducing the resources required by ADAPT-VQE using coupled exchange operators and improved subroutines," npj Quantum Information (2025). URL: https://www.nature.com/articles/s41534-025-01039-4. Supports commutator-gradient formula, pool-dependent measurement cost, and ADAPT workflow variants.
- Stokes et al., "Quantum Natural Gradient," Quantum 4, 269 (2020). DOI: https://doi.org/10.22331/q-2020-05-25-269. Supports using the real part of the quantum geometric tensor/Fubini--Study metric as quantum information geometry.
- Huggins et al., "Efficient and noise resilient measurements for quantum chemistry on near-term quantum computers," npj Quantum Information 7, 23 (2021). URL: https://www.nature.com/articles/s41534-020-00341-7. Supports Hamiltonian averaging, Pauli-word measurement, and optimal measurement distribution.
- Crawford et al., "Efficient quantum measurement of Pauli operators in the presence of finite sampling error," Quantum 5, 385 (2021). DOI: https://doi.org/10.22331/q-2021-01-20-385. Supports finite-sampling-aware grouping and the distinction between commuting collections and sampling error.
- Yen, Verteletskyi, and Izmaylov, "Measuring all compatible operators in one series of single-qubit measurements using unitary transformations," Journal of Chemical Theory and Computation 16, 2400--2409 (2020). DOI: https://doi.org/10.1021/acs.jctc.0c00008. Supports compatible-operator measurement and grouping beyond naive Pauli-by-Pauli measurement.
- Izmaylov-group measurement-allocation work, "Deterministic improvements of quantum measurements with grouping of compatible operators, non-local transformations, and covariance estimates," npj Quantum Information 9, 14 (2023). URL: https://www.nature.com/articles/s41534-023-00683-y. Supports covariance/variance-aware measurement allocation and overlapping-group considerations.
- Wierichs et al., "General parameter-shift rules for quantum gradients," Quantum 6, 677 (2022). DOI: https://doi.org/10.22331/q-2022-03-30-677. Supports estimator-dependent derivative and higher-order derivative resource requirements.
- Ramôa et al., "Reducing measurement costs by recycling the Hessian in adaptive variational quantum algorithms," Quantum Science and Technology 10, 015031 (2025). DOI: https://doi.org/10.1088/2058-9565/ad904e. Supports Hessian recycling as a measurement-cost reduction strategy in adaptive VQAs.
- Ikhtiarudin et al., "Shot-Efficient ADAPT-VQE via Reused Pauli Measurements and Variance-Based Shot Allocation," arXiv:2507.16879 (2025). URL: https://arxiv.org/abs/2507.16879. Supports reused Pauli measurements and variance-based shot allocation for ADAPT-VQE, but should be treated as preprint-level support unless a peer-reviewed version is confirmed.

## 5. Recommended decision

Use the GPT-Pro response as a support-note basis, but do not import it into Paper I verbatim. The safest manuscript update is local:

1. Rename \(h_r\) as candidate directional energy curvature.
2. Add one compact measurement-accounting paragraph distinguishing primitive entries, grouped settings, and shots.
3. State Phase III's extra primitive burden as \(c_r\) and \(R_r\), with starred quantities algebraic.
4. Present the scaling as a shortlist-dependent upper-bound form, not a universal physical shot overhead.
5. Place estimator-specific details in support material or an appendix unless the implementation fixes the estimator model.
