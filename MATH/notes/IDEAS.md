Your current §17 already has the right geometric objects and controller separation: McLachlan on a fixed scaffold, miss telemetry (\rho_{\mathrm{miss}}), residual-overlap scout, exact Schur-complement confirm, and checkpoint-local append/prune. The shot problem is not the doctrine; it is where exact geometry is being asked to cover too large a tangent block or too large a candidate pool. The clean reduction is to introduce nested tangent baselines and a Schur-family ladder
[
\underline\Delta ;\le; \widetilde\Delta^{(m)} ;\le; \Delta,
]
with reduced-space miss at scout, compressed Schur confirm on a shortlist, and exact gain only at commit. That keeps everything inside the same local McLachlan geometry. 

I will assume below that regularization makes the baseline solve invertible on the working support. If not, replace inverses by pseudoinverses on the numerical support.

## 1. Short diagnosis of the expensive quantities

Let
[
a:=|A_k| \quad\text{(live active tangent size)},\qquad
M:=|\mathcal R_k^{\mathrm{probe}}| \quad\text{(candidate pool size)},\qquad
q:=\dim(\bar U_r)\quad\text{(candidate block size)}.
]

Then, up to commuting-group reuse:

[
\bar G_A \sim O(a^2),\qquad
\bar f_A \sim O(a),\qquad
{B_{r,A}}*{r=1}^M \sim O(Maq),\qquad
{q_r}*{r=1}^M \sim O(Mq),\qquad
{C_r}_{r=1}^M \sim O(Mq^2).
]

The main bottlenecks are:

1. **Current metric (\bar G)** on the live active basis.
   This is the quadratic checkpoint cost. It is the main base-runtime shot sink.

2. **Candidate cross-blocks (B_r=\Re(\bar T^\dagger \bar U_r))** across the broad pool.
   This is the main append-controller shot sink. Broad-pool exact Schur gains are expensive because they ask for (B_r) against the whole live tangent block.

3. **(q_r)** is usually not the main bottleneck.
   It is only (O(Mq)), and in the staged doctrine below it is better to measure the residualized numerator (w_r) directly.

4. **(C_r)** is often cheap.
   For a single generator direction (\bar u_r=Q_\psi(-i\widetilde A_r\psi)),
   [
   C_r=\Re(\bar u_r^\dagger \bar u_r)
   =\langle \widetilde A_r^2\rangle_\psi-\langle \widetilde A_r\rangle_\psi^2.
   ]
   If (\widetilde A_r^2=I), then
   [
   C_r = 1-\langle \widetilde A_r\rangle_\psi^2.
   ]

5. **Residual overlaps are cheap if measured directly** and expensive if reconstructed by measuring all (B_r) first.
   Your current scout already uses (|\Re(\bar U_r^\dagger \bar r_k)|_2); that is the correct quantity to preserve, but it should be normalized into a Schur lower bound rather than left as a raw overlap. 

6. **Confirm-stage exact gain is not the first bottleneck** if the shortlist is small.
   The right place to spend exact (B_r) shots is shortlist-only confirm/commit, not broad scout.

A side note on pruning: the broad prune score need not require new shots at all once the current (G,f) block is known; its frozen-ablation loss is itself a Schur-complement quantity.

---

## 2. The key doctrine: reduced-space McLachlan is still McLachlan

For any coordinate subset
[
A\subseteq {1,\dots,n_{\theta,k}},
]
define the reduced tangent block
[
\bar T_A := [,\bar \tau_j,]_{j\in A},
\qquad
\bar G_A := \Re(\bar T_A^\dagger \bar T_A),
\qquad
\bar f_A := \Re(\bar T_A^\dagger \bar b_k),
]
[
K_A := \bar G_A+\Lambda_A,
\qquad
K_A\dot\theta_A=\bar f_A,
\qquad
\bar r_A:=\bar T_A\dot\theta_A-\bar b_k.
]

This is not a heuristic change. It is exactly the McLachlan law on the reduced real tangent subspace
[
\mathcal T_A:=\operatorname{span}_{\mathbb R}{\bar\tau_j:j\in A}.
]

Define the reduced miss
[
\epsilon_{\mathrm{proj}}^2(A)
:=
|\bar b_k|^2-\bar f_A^\top \bar G_A^+\bar f_A,
\qquad
\rho_{\mathrm{miss}}(A)
:=
\frac{\epsilon_{\mathrm{proj}}^2(A)}{|\bar b_k|^2+\varepsilon}.
]

If (A\subseteq B), then
[
\mathcal T_A\subseteq \mathcal T_B
\quad\Longrightarrow\quad
\epsilon_{\mathrm{proj}}^2(A)\ge \epsilon_{\mathrm{proj}}^2(B),
\qquad
\rho_{\mathrm{miss}}(A)\ge \rho_{\mathrm{miss}}(B).
]

So:

* (\rho_{\mathrm{miss}}(A)) on a reduced active window is an **exact miss on the reduced tangent space**;
* it is a **conservative upper bound** on the miss of any larger baseline;
* this gives a mathematically meaningful scout miss without breaking the logic.

That answers your question about active suffix windows: yes, miss can be estimated on a reduced active window without breaking the doctrine, because it remains exact McLachlan geometry on that reduced subspace.

---

## 3. The right cheap scout quantity is a Schur lower bound, not a raw overlap

For a candidate block (\bar U_r\in\mathbb C^{d\times q}), relative to a baseline (A), define
[
B_{r,A}:=\Re(\bar T_A^\dagger \bar U_r),
\qquad
C_r:=\Re(\bar U_r^\dagger \bar U_r),
\qquad
q_r:=\Re(\bar U_r^\dagger \bar b_k).
]

The exact reduced-space Schur objects are
[
S_r(A):=C_r+\Lambda_r-B_{r,A}^\top K_A^{-1}B_{r,A},
]
[
w_r(A):=q_r-B_{r,A}^\top K_A^{-1}\bar f_A,
]
[
\Delta_{\mathrm{McL}}(r;A):=w_r(A)^\top S_r(A)^+ w_r(A).
]

Now the crucial identity:
[
\Re(\bar U_r^\dagger \bar r_A)
==============================

# B_{r,A}^\top \dot\theta_A-q_r

-,w_r(A).
]

So the scout residual overlap already in your runtime is not just “related” to the Schur gain; it is **exactly the Schur numerator**:
[
w_r(A) = -\Re(\bar U_r^\dagger \bar r_A).
]

This lets you define a **scout-safe lower bound**
[
\underline\Delta_r(A)
:=
w_r(A)^\top (C_r+\Lambda_r)^+ w_r(A).
]

Since
[
B_{r,A}^\top K_A^{-1}B_{r,A}\succeq 0
\quad\Longrightarrow\quad
S_r(A)\preceq C_r+\Lambda_r,
]
we get, on the common support,
[
\underline\Delta_r(A)\le \Delta_{\mathrm{McL}}(r;A).
]

For a single real candidate direction,
[
\underline\Delta_r(A)
=====================

\frac{\Re(\bar u_r^\dagger \bar r_A)^2}{C_r+\lambda_r}.
]

For a (q)-block, the cheapest block-normalized surrogate is
[
\underline\Delta_r^{\mathrm{tr}}(A)
:=
\frac{|w_r(A)|_2^2}{\operatorname{tr}(C_r+\Lambda_r)}
\le
\underline\Delta_r(A),
]
because (C_r+\Lambda_r\preceq \operatorname{tr}(C_r+\Lambda_r)I).

### Interpretation

* (\underline\Delta_r(A)) is **still in the same Schur family**.
* It uses the **exact residualized numerator**.
* It is a **lower bound** on the exact gain for the same baseline (A).
* It is therefore suitable for broad-pool shortlist construction.

This is the natural replacement for a raw (|\Re(\bar U_r^\dagger \bar r_A)|) scout score.

---

## 4. A clean scout (\to) confirm (\to) commit pipeline

### Stage S0: cheap base geometry on a scout baseline

Choose nested coordinate families
[
A_k^{\mathrm{sc}}
\subseteq
A_k^{\mathrm{cf}}(r)
\subseteq
A_k^{\mathrm{cm}}
\subseteq
{1,\dots,n_{\theta,k}}.
]

Recommended interpretation:

* (A_k^{\mathrm{sc}}): active suffix (W_k^{\mathrm{act}}), possibly plus a very small landmark set of older coordinates;
* (A_k^{\mathrm{cf}}(r)): shortlist-only candidate-local confirm window;
* (A_k^{\mathrm{cm}}): commit baseline, shared across finalists.

Compute exactly on (A_k^{\mathrm{sc}}):
[
\bar G_{A^{\mathrm{sc}}},\quad
\bar f_{A^{\mathrm{sc}}},\quad
K_{A^{\mathrm{sc}}},\quad
\dot\theta_{A^{\mathrm{sc}}},\quad
\bar r_{A^{\mathrm{sc}}},\quad
\rho_{\mathrm{miss}}(A_k^{\mathrm{sc}}).
]

Use
[
\rho_{\mathrm{miss}}(A_k^{\mathrm{sc}})
]
as the **conservative opening signal** for structural search.

This turns the base cost from (O(n_\theta^2)) into (O(|A_k^{\mathrm{sc}}|^2)).

### Stage S1: broad scout over the whole candidate pool

For every (r\in\mathcal R_k^{\mathrm{probe}}), measure only:

* the exact residualized overlap numerator
  [
  w_r(A_k^{\mathrm{sc}})
  ======================

  -\Re(\bar U_r^\dagger \bar r_{A_k^{\mathrm{sc}}}),
  ]
* and the cheap candidate self scale (C_r) or just (\operatorname{tr}(C_r+\Lambda_r)).

Then score geometrically by
[
\Sigma_{\mathrm{scout}}^{\mathrm{geom}}(r;k)
:=
\frac{\underline\Delta_r(A_k^{\mathrm{sc}})}{|\bar b_k|^2+\varepsilon},
]
or, for block-cheap screening,
[
\Sigma_{\mathrm{scout}}^{\mathrm{geom,tr}}(r;k)
:=
\frac{\underline\Delta_r^{\mathrm{tr}}(A_k^{\mathrm{sc}})}{|\bar b_k|^2+\varepsilon}.
]

Then add your existing hardware terms:
[
\Sigma_{\mathrm{scout}}(r;k)
============================

\Sigma_{\mathrm{scout}}^{\mathrm{geom}}(r;k)
+b_{\mathrm{temp}}(r;k)
-\beta_{\mathrm{comp}}C_{\mathrm{comp}}(r;k)
-\beta_{\mathrm{meas}}G_{\mathrm{new}}(r;k)
-\beta_{\mathrm{jump}}J_{\mathrm{jump}}(r;k).
]

This reduces broad append screening from roughly
[
O(M,|A_k^{\mathrm{cm}}|,q)
]
to
[
O(Mq)
]
after the baseline residual is available.

### Stage S2: shortlist-only confirm on a candidate-local block

For shortlisted (r), choose a candidate-local confirm block
[
J_r:=A_k^{\mathrm{cf}}(r),
\qquad
A_k^{\mathrm{sc}}\subseteq J_r\subseteq A_k^{\mathrm{cm}}.
]

Compute exactly on (J_r):
[
\bar G_{J_r},\quad \bar f_{J_r},\quad K_{J_r},\quad \dot\theta_{J_r},\quad \bar r_{J_r}.
]

Then measure the exact confirm numerator
[
w_r(J_r)=-\Re(\bar U_r^\dagger \bar r_{J_r}).
]

At this point you have two confirm options.

#### Option A: exact reduced-space confirm

Measure the full (B_{r,J_r}) and compute
[
\Delta_{\mathrm{McL}}(r;J_r).
]

This is **exact on the reduced baseline (J_r)**.
It is not generally monotone in (J_r), so it is a reranker, not a safety certificate across different windows.

#### Option B: compressed whitened confirm

Diagonalize the current reduced solver matrix:
[
K_{J_r}=V_{J_r}\operatorname{diag}(\kappa_1,\dots,\kappa_{|J_r|})V_{J_r}^\top,
\qquad
\kappa_\ell>0.
]

Define the whitened baseline modes
[
\widetilde v_\ell := \kappa_\ell^{-1/2}\bar T_{J_r}(V_{J_r})*{:\ell},
]
and the candidate couplings to them
[
\beta*{r,\ell}
:=
\Re(\widetilde v_\ell^\dagger \bar U_r)
=======================================

\kappa_\ell^{-1/2}\bigl(V_{J_r}^\top B_{r,J_r}\bigr)_{\ell,:}\in\mathbb R^q.
]

Then
[
B_{r,J_r}^\top K_{J_r}^{-1}B_{r,J_r}
====================================

\sum_{\ell=1}^{|J_r|}\beta_{r,\ell}\beta_{r,\ell}^\top.
]

For (m\le |J_r|), define the truncated denominator
[
\widetilde S_r^{(m)}(J_r)
:=
C_r+\Lambda_r-\sum_{\ell=1}^{m}\beta_{r,\ell}\beta_{r,\ell}^\top,
]
and the compressed confirm gain
[
\widetilde\Delta_r^{(m)}(J_r)
:=
w_r(J_r)^\top
\bigl(\widetilde S_r^{(m)}(J_r)\bigr)^+
w_r(J_r).
]

Because the omitted tail is positive semidefinite,
[
\widetilde S_r^{(m)}(J_r)\succeq S_r(J_r),
]
hence
[
\widetilde\Delta_r^{(0)}(J_r)
=============================

\underline\Delta_r(J_r)
\le
\widetilde\Delta_r^{(1)}(J_r)
\le \cdots \le
\widetilde\Delta_r^{(|J_r|)}(J_r)
=================================

\Delta_{\mathrm{McL}}(r;J_r).
]

This is the clean confirm ladder:

* (m=0): scout lower bound;
* small (m): cheap confirm;
* (m=|J_r|): exact reduced-space confirm.

It is controlled, monotone in (m), and stays in the Schur family.

### Stage S3: final commit on a shared commit baseline

For the final few candidates, evaluate on a **common** baseline (A_k^{\mathrm{cm}}):
[
\Delta_{\mathrm{McL}}(r;A_k^{\mathrm{cm}})
]
or at least its monotone lower-bound sequence
[
\widetilde\Delta_r^{(m)}(A_k^{\mathrm{cm}}).
]

Then commit with your existing rule, but using the commit baseline:
[
\text{append }r_k^\star
\iff
\rho_{\mathrm{miss}}(A_k^{\mathrm{cm}})>\tau_{\mathrm{miss}},
\quad
\frac{\Delta_{\mathrm{McL}}(r_k^\star;A_k^{\mathrm{cm}})}{|\bar b_k|^2+\varepsilon}\ge \tau_{\mathrm{gain}},
\quad
\Delta_{\mathrm{McL}}(r_k^\star;A_k^{\mathrm{cm}})\ge \delta_{\mathrm{append}}.
]

---

## 5. What is exact, what is scout-safe, and what is only confirm-grade

### Exact but expensive

[
\rho_{\mathrm{miss}}(A_k^{\mathrm{cm}}),
\qquad
\Delta_{\mathrm{McL}}(r;A_k^{\mathrm{cm}}),
\qquad
B_{r,A_k^{\mathrm{cm}}}.
]

These are commit-grade quantities.

### Cheap and scout-safe

[
\rho_{\mathrm{miss}}(A_k^{\mathrm{sc}}),
\qquad
\underline\Delta_r(A_k^{\mathrm{sc}}),
\qquad
\underline\Delta_r^{\mathrm{tr}}(A_k^{\mathrm{sc}}).
]

Interpretation:

* (\rho_{\mathrm{miss}}(A_k^{\mathrm{sc}})) is an **upper bound** on larger-baseline miss;
* (\underline\Delta_r(A)) is a **lower bound** on exact gain for the **same** baseline (A).

### Confirm-stage controlled approximants

[
\Delta_{\mathrm{McL}}(r;J_r),
\qquad
\widetilde\Delta_r^{(m)}(J_r).
]

Interpretation:

* (\Delta(r;J_r)) is exact on a reduced baseline but not monotone in (J_r);
* (\widetilde\Delta^{(m)}(J_r)) is a **monotone lower-bound ladder in (m)** for that baseline.

### Biased pilot approximations only

If you want an even cheaper calm-checkpoint pilot, use a frozen metric:
[
\widetilde K_{A,k+1}:=K_{A,k},
\qquad
\widetilde{\dot\theta}*{A,k+1}:=K*{A,k}^{-1}\bar f_{A,k+1}.
]

Then
[
\widetilde{\dot\theta}*{A,k+1}-\dot\theta*{A,k+1}
=================================================

K_{A,k}^{-1}\bigl(K_{A,k+1}-K_{A,k}\bigr)K_{A,k+1}^{-1}\bar f_{A,k+1},
]
so
[
|\widetilde{\dot\theta}*{A,k+1}-\dot\theta*{A,k+1}|*2
\le
|K*{A,k}^{-1}|*2,
|K*{A,k+1}-K_{A,k}|*2,
|K*{A,k+1}^{-1}\bar f_{A,k+1}|_2.
]

This is explicit and usable, but it is **biased**. I would use it only as a calm-checkpoint pilot, never as the final acceptance quantity.

A diagonal metric
[
D_A:=\operatorname{diag}(\bar G_A),
\qquad
\dot\theta_A^{\mathrm{diag}}=(D_A+\Lambda_A)^{-1}\bar f_A
]
is even weaker: it has no clean ordering guarantee and should remain a prefilter of last resort.

---

## 6. Which approximations preserve ordering well enough?

The shortlist-preserving ordering I would actually trust is:

1. **Best scout ordering**
   [
   \zeta_r(A):=
   |(C_r+\Lambda_r)^{-1/2}w_r(A)|_2,
   \qquad
   \zeta_r(A)^2=\underline\Delta_r(A).
   ]
   This is the right broad-pool quantity. It keeps the exact residualized numerator and normalizes candidate scale.

2. **Best confirm ordering**
   [
   \widetilde\Delta_r^{(m)}(J_r)
   ]
   with small (m), because it monotonically restores the Schur redundancy correction.

3. **Best pre-commit ordering**
   [
   \Delta_{\mathrm{McL}}(r;J_r)
   ]
   or, for finalists, exact (\Delta_{\mathrm{McL}}(r;A_k^{\mathrm{cm}})).

I would **not** use raw (q_r), raw (|\Re(\bar U_r^\dagger \bar r)|), or a diagonal-metric solve as the main shortlist surface if the goal is to preserve the McLachlan geometry rather than drift toward an unrelated heuristic.

A safety point:

* if you evaluate (\underline\Delta_r(A_k^{\mathrm{cm}})) on the **same baseline used for commit**, then
  [
  \underline\Delta_r(A_k^{\mathrm{cm}})\ge \delta_{\mathrm{append}}
  \quad\Longrightarrow\quad
  \Delta_{\mathrm{McL}}(r;A_k^{\mathrm{cm}})\ge \delta_{\mathrm{append}}.
  ]
  That is a valid sufficient condition.
* the converse is not true;
* and a lower bound on a smaller scout baseline is not a final certificate for a larger commit baseline.

---

## 7. Reduced active windows and compressed tangent blocks

Yes, both can be used without breaking the logic, but they play different roles.

### Miss on reduced active windows

Safe and clean:
[
\rho_{\mathrm{miss}}(A_k^{\mathrm{sc}})
]
is exact on the reduced tangent space and conservative for larger spaces.

### Candidate gain on reduced windows

Also clean if stated correctly:
[
\Delta_{\mathrm{McL}}(r;J_r)
]
is exact on the reduced baseline (J_r). This is not a heuristic; it is exact Schur geometry on a smaller tangent space.

### Compressed tangent blocks

Also clean:
[
\widetilde\Delta_r^{(m)}(J_r)
]
is a controlled lower bound obtained by retaining only the top (m) whitened modes of the baseline solver matrix. It remains in the same Schur family and is monotone in (m).

So the ADAPT-style analogue is:

* **cheap broad screen**: reduced baseline (A_k^{\mathrm{sc}}), exact residual numerator, zero Schur correction;
* **smaller confirm**: candidate-local (J_r), small number of whitened Schur modes;
* **final commit**: shared commit baseline (A_k^{\mathrm{cm}}), full Schur correction.

That is the clean McLachlan analogue of broad screen (\to) confirm (\to) commit.

---

## 8. What should be cached, and when is it legitimate?

There are three levels.

### 8.1 Exact reuse within one checkpoint

Once a checkpoint’s primitive observable bank is measured, all derived geometric quantities should be treated as algebraic reuse.

Let (x_k\in\mathbb R^{M_k}) be the vector of distinct measured primitive group means at checkpoint (k). Then there exist linear maps
[
\operatorname{vec}(\bar G_{A,k})=L_{G,A}x_k,
\qquad
\bar f_{A,k}=L_{f,A}x_k,
\qquad
\operatorname{vec}(C_{r,k})=L_{C,r}x_k,
]
and, after solving (\dot\theta_{A,k}), a linear functional
[
w_r(A;k)=L_{w,r,A}(\dot\theta_{A,k}),x_k.
]

So once (x_k) exists, scout, confirm, and prune surrogates are all post-processing of the same checkpoint bank.

### 8.2 Exact reuse across zero-initialized append

This is the strongest exact cache invariant.

Let the parent scaffold be (\mathcal O_k), and append (r=(m,p)) to get
[
\mathcal O_k^+ := \mathcal O_k\oplus_p m,
\qquad
\theta_k^+ := \theta_k\oplus_p 0.
]

Then
[
|\psi(\theta_k;\mathcal O_k)\rangle
===================================

|\psi(\theta_k^+;\mathcal O_k^+)\rangle.
]

If (I_p) is the index transport for old coordinates, then for every old coordinate (j),
[
\partial_{\theta_j}|\psi(\theta_k;\mathcal O_k)\rangle
======================================================

\partial_{\theta_{I_p(j)}}|\psi(\theta_k^+;\mathcal O_k^+)\rangle.
]

Hence the old-old projected tangent block transports exactly, and therefore
[
(\bar G_k^+)*{I_p(i)I_p(j)}=(\bar G_k)*{ij},
\qquad
(\bar f_k^+)*{I_p(i)}=(\bar f_k)*i,
\qquad
(K_k^+)*{I_p(i)I_p(j)}=(K_k)*{ij}.
]

So after an accepted zero-init append, only the new row/column quantities
[
B_r,\qquad q_r,\qquad C_r
]
need fresh measurement.

### 8.3 Incremental reuse across nearby checkpoints

If the scaffold and active baseline are unchanged, the observable map is unchanged, so one can reuse previous estimates by increments:
[
x_{k+1}=x_k+\Delta x_{k+1},
\qquad
\widehat x_{k+1}=\widehat x_k+\widehat{\Delta x}_{k+1}.
]

Then
[
\operatorname{vec}(\widehat{\bar G}_{A,k+1})
============================================

# L_{G,A}\widehat x_{k+1}

\operatorname{vec}(\widehat{\bar G}*{A,k})
+
L*{G,A}\widehat{\Delta x}*{k+1},
]
[
\widehat{\bar f}*{A,k+1}
========================

\widehat{\bar f}*{A,k}
+
L*{f,A}\widehat{\Delta x}*{k+1},
\qquad
\widehat C*{r,k+1}
==================

\widehat C_{r,k}
+
L_{C,r}\widehat{\Delta x}_{k+1}.
]

This is the right mathematical formalization of measurement reuse across nearby checkpoints.

If you then freeze (K_A), that becomes a biased pilot surrogate; the formula above tells you exactly where the bias enters.

---

## 9. Prune broad screen can be made shot-free

For scalar coordinate pruning, let (F:=A\setminus{j}). Partition the current unregularized baseline geometry as
[
\bar G_A=
\begin{bmatrix}
\bar G_{jj} & \bar G_{jF}\
\bar G_{Fj} & \bar G_{FF}
\end{bmatrix},
\qquad
\bar f_A=
\begin{bmatrix}
\bar f_j\
\bar f_F
\end{bmatrix}.
]

Then the frozen-ablation fit loss is the scalar Schur complement
[
L_j^{\mathrm{frz}}
==================

\frac{u_j^2}{s_j},
]
with
[
u_j:=\bar f_j-\bar G_{jF}\bar G_{FF}^+\bar f_F,
\qquad
s_j:=\bar G_{jj}-\bar G_{jF}\bar G_{FF}^+\bar G_{Fj}.
]

So prune broad ranking is computational once the current geometry is cached; it should not consume extra measurement shots.

---

## 10. Direct answers to your numbered questions

**1. Main shot bottlenecks?**
(\bar G) and broad-pool (B_r).
Not (\bar b) itself.
(q_r) is moderate; (C_r) is usually cheap.
Residual overlaps are cheap if measured directly.
Exact confirm gain is acceptable if shortlist-only.

**2. Principled approximations?**
Reduced baseline miss (\rho_{\mathrm{miss}}(A));
Schur lower-bound scout gain (\underline\Delta_r(A));
trace-normalized block scout (\underline\Delta_r^{\mathrm{tr}}(A));
candidate-local exact reduced confirm (\Delta_r(J_r));
whitened rank-(m) confirm lower bounds (\widetilde\Delta_r^{(m)}(J_r));
optional frozen-metric pilot (\widetilde K_{A,k+1}=K_{A,k}).

**3. How stage them?**
Scout: (A_k^{\mathrm{sc}},,m=0).
Confirm: (J_r,,m\ll |J_r|).
Commit: shared (A_k^{\mathrm{cm}},,m=|A_k^{\mathrm{cm}}|).

**4. Which preserve ordering?**
Best scout ordering: (\underline\Delta_r(A)).
Best confirm ordering: (\widetilde\Delta_r^{(m)}(J_r)).
Best final ordering: exact (\Delta_r) on shared commit baseline.
Diagonal/frozen metric are not good ordering surfaces.

**5. Can miss and gain be estimated on reduced windows?**
Yes.
Miss: exact and conservative.
Gain: exact on reduced baseline, or lower-bound/compressed on that same baseline.

**6. Clean ADAPT analogue?**
Yes:
reduced-space miss + normalized residual-overlap lower bound (\to) compressed Schur confirm (\to) exact Schur commit.

**7. What cache across checkpoints?**
Primitive moment bank (x_k), current (K_A) factorization, candidate self-blocks (C_r), shortlist numerator measurements (w_r), and exact old-old geometry transport across zero-init append.

**8. Mathematical measurement reuse?**
Use the linear-update formalism for primitive banks
[
x_{k+1}=x_k+\Delta x_{k+1},
]
plus exact transport under append
[
(\bar G_k^+)*{I_p(i)I_p(j)}=(\bar G_k)*{ij},\qquad
(\bar f_k^+)_{I_p(i)}=(\bar f_k)_i.
]

---

## 11. Exact LaTeX patch text

I would add a dedicated subsection immediately after §17.4 and before §17.5.

```latex
\subsection*{17.4A Shot-frugal staged geometry: reduced active baselines, lower-bound scout gains, and compressed confirm}

The live runtime need not remeasure the full checkpoint geometry on the full current scaffold at every stage. Instead, let
\[
A_k^{\mathrm{sc}}
\subseteq
A_k^{\mathrm{cf}}(r)
\subseteq
A_k^{\mathrm{cm}}
\subseteq
\{1,\dots,n_{\theta,k}\}
\]
be nested active-coordinate families used respectively for scout, candidate-local confirm, and commit.

For any such index set \(A\), define the reduced projected tangent matrix
\[
\bar T_{k,A}:=[\,\bar\tau_j\,]_{j\in A},
\qquad
\bar G_{k,A}:=\Re(\bar T_{k,A}^\dagger \bar T_{k,A}),
\qquad
\bar f_{k,A}:=\Re(\bar T_{k,A}^\dagger \bar b_k),
\]
\[
K_{k,A}:=\bar G_{k,A}+\Lambda_{k,A},
\qquad
K_{k,A}\dot\theta_{k,A}=\bar f_{k,A},
\qquad
\bar r_{k,A}:=\bar T_{k,A}\dot\theta_{k,A}-\bar b_k.
\]
This is not a heuristic change of doctrine: it is exactly the McLachlan law on the reduced tangent subspace
\[
\operatorname{span}_{\mathbb R}\{\bar\tau_j:j\in A\}.
\]

The corresponding reduced miss is
\[
\epsilon_{\mathrm{proj},k}^2(A)
:=
\|\bar b_k\|^2-\bar f_{k,A}^\top \bar G_{k,A}^+\bar f_{k,A},
\qquad
\rho_{\mathrm{miss},k}(A)
:=
\frac{\epsilon_{\mathrm{proj},k}^2(A)}{\|\bar b_k\|^2+\varepsilon}.
\]
For \(A\subseteq B\), one has
\[
\epsilon_{\mathrm{proj},k}^2(A)\ge \epsilon_{\mathrm{proj},k}^2(B),
\qquad
\rho_{\mathrm{miss},k}(A)\ge \rho_{\mathrm{miss},k}(B),
\]
because the admissible real tangent space only enlarges. Hence \(\rho_{\mathrm{miss},k}(A_k^{\mathrm{sc}})\) is a conservative scout-side upper bound on the commit miss.

For any candidate-position record \(r\) and any baseline \(A\), define
\[
B_{r,A}:=\Re(\bar T_{k,A}^\dagger \bar U_r),
\qquad
C_r:=\Re(\bar U_r^\dagger \bar U_r),
\qquad
q_r:=\Re(\bar U_r^\dagger \bar b_k),
\]
\[
S_r(A):=C_r+\Lambda_r-B_{r,A}^\top K_{k,A}^{-1}B_{r,A},
\qquad
w_r(A):=q_r-B_{r,A}^\top K_{k,A}^{-1}\bar f_{k,A}.
\]
Equivalently,
\[
w_r(A)=-\Re(\bar U_r^\dagger \bar r_{k,A}).
\]
The exact reduced-space append gain is
\[
\Delta_{\mathrm{McL}}(r;\tau_k,A)
:=
w_r(A)^\top S_r(A)^+ w_r(A).
\]

A scout-safe lower bound is obtained by dropping the redundancy correction in the Schur denominator:
\[
\underline\Delta_{\mathrm{McL}}(r;\tau_k,A)
:=
w_r(A)^\top (C_r+\Lambda_r)^+ w_r(A)
\le
\Delta_{\mathrm{McL}}(r;\tau_k,A),
\]
where the inequality is taken on the support of \(C_r+\Lambda_r\). For a single real candidate direction,
\[
\underline\Delta_{\mathrm{McL}}(r;\tau_k,A)
=
\frac{\Re(\bar u_r^\dagger \bar r_{k,A})^2}{C_r+\lambda_r}.
\]
Accordingly, a shot-frugal scout surface may use
\[
\Sigma_{\mathrm{scout}}^{\mathrm{geom}}(r;k)
:=
\frac{\underline\Delta_{\mathrm{McL}}(r;\tau_k,A_k^{\mathrm{sc}})}{\|\bar b_k\|^2+\varepsilon},
\]
in place of an unnormalized residual-overlap term, while keeping the same hardware penalties as before.

For confirm, let \(J_r:=A_k^{\mathrm{cf}}(r)\) and write the current reduced solver matrix in whitened eigenmodes,
\[
K_{k,J_r}=V_{J_r}\operatorname{diag}(\kappa_1,\dots,\kappa_{|J_r|})V_{J_r}^\top,
\qquad
\kappa_\ell>0.
\]
Define the whitened baseline modes
\[
\widetilde v_\ell:=\kappa_\ell^{-1/2}\bar T_{k,J_r}(V_{J_r})_{:\ell},
\]
and the candidate couplings to these modes by
\[
\beta_{r,\ell}
:=
\Re(\widetilde v_\ell^\dagger \bar U_r)
=
\kappa_\ell^{-1/2}\bigl(V_{J_r}^\top B_{r,J_r}\bigr)_{\ell,:}
\in\mathbb R^{q},
\]
so that
\[
B_{r,J_r}^\top K_{k,J_r}^{-1}B_{r,J_r}
=
\sum_{\ell=1}^{|J_r|}\beta_{r,\ell}\beta_{r,\ell}^\top.
\]
For \(m\le |J_r|\), define the compressed confirm denominator
\[
\widetilde S_r^{(m)}(J_r)
:=
C_r+\Lambda_r-\sum_{\ell=1}^{m}\beta_{r,\ell}\beta_{r,\ell}^\top,
\]
and use the exact residual-overlap numerator \(w_r(J_r)=-\Re(\bar U_r^\dagger \bar r_{k,J_r})\) to form
\[
\widetilde\Delta_{\mathrm{McL}}^{(m)}(r;\tau_k,J_r)
:=
w_r(J_r)^\top\bigl(\widetilde S_r^{(m)}(J_r)\bigr)^+ w_r(J_r).
\]
Then
\[
\underline\Delta_{\mathrm{McL}}(r;\tau_k,J_r)
=
\widetilde\Delta_{\mathrm{McL}}^{(0)}(r;\tau_k,J_r)
\le
\widetilde\Delta_{\mathrm{McL}}^{(1)}(r;\tau_k,J_r)
\le \cdots \le
\widetilde\Delta_{\mathrm{McL}}^{(|J_r|)}(r;\tau_k,J_r)
=
\Delta_{\mathrm{McL}}(r;\tau_k,J_r),
\]
because each omitted tail contribution is positive semidefinite. Hence confirm may measure only the dominant whitened couplings first, enlarge \(m\) only for records that remain competitive, and recover the exact Schur gain only at commit.

At a zero-initialized accepted append \(r=(m,p)\), the child state equals the parent state,
\[
|\psi(\theta_k;\mathcal O_k)\rangle
=
|\psi(\theta_k\oplus_p 0;\mathcal O_k\oplus_p m)\rangle,
\]
and every old tangent vector is transported exactly under the insertion index map \(I_p\). Therefore the old-old blocks of \(\bar G_k\), \(K_k\), and \(\bar f_k\) are reused exactly after append; only the new row/column blocks involving \(r\) require fresh measurement.
```

This is the subsection I would actually add.

If you want the next step, I can turn this into a notation-consistent rewrite that matches your manuscript’s exact symbol style and surrounding prose rhythm.
The clean doctrine is **backbone-first, broad-corrective-later**. That fits your controller-side soft-opening idea much better than literal algebraic “low-order first,” and it matches the manuscript seam where early seeds are broadened later as the scaffold matures.  

**1. Define family order from a structural descriptor, not from a single scalar.**

Use the **macro-family label** as the object of classification, not the record (r=(m,p)). If your implementation already carries a family map like (\mathfrak g:\mathcal G\to\mathcal C), that is the right attachment point. 

A good family descriptor is
[
\Xi(f)=\bigl(\rho_{\mathrm{role}}(f),\ \ell_{\mathrm{phys}}(f),\ r_{\mathrm{exc}}(f),\ a_{\mathrm{agg}}(f),\ s_{\mathrm{sel}}(f),\ b_{\mathrm{breadth}}(f)\bigr),
]
with:

* (\rho_{\mathrm{role}}): seed / backbone / refinement / residual,
* (\ell_{\mathrm{phys}}): physical locality span, meaning onsite, bond-local, neighborhood, or broader,
* (r_{\mathrm{exc}}): excitation rank or conditional rank,
* (a_{\mathrm{agg}}): split-term vs unsplit aggregate vs distributed aggregate,
* (s_{\mathrm{sel}}): sector selectivity, such as spin-resolved or doublon-only,
* (b_{\mathrm{breadth}}): **constructive breadth**, meaning how many qualitatively different missing scaffold effects one admission can absorb.

That last quantity is the most useful intuition. Early-opening families should have **low constructive breadth**: one admission does one small, legible thing. Late-opening families have **high constructive breadth**: one admission can behave like a broad residual fix.

So the right meaning of “lower order” here is not “smaller algebraic degree.” It is **more atomic, more local, more scaffold-defining, less broad**.

Also: use **physical locality** and **excitation structure** before qubit mapping. Raw Pauli weight is a poor primary notion here because it is encoding-dependent.

**2. “Low-order first” is useful only as shorthand.**

As a slogan, it is fine. As doctrine, it is too naive.

What you actually want is:

* open early the families that **build reusable scaffold**,
* open late the families that **mainly close residual error**.

That distinction is sharper than literal algebraic order.

In your HH setting, the split seed family (\mathcal Q) should open before the unsplit (\mathcal G^{(\mathrm{cd})}), even though they represent the same underlying local mechanism, because the split family is the **atomic scaffold form** and the unsplit family is the **aggregate form**. Likewise, (\mathcal G^{(\mathrm{hd})}) can deserve earlier opening than something formally “simpler,” because it builds a physically meaningful bond-local backbone rather than acting as a broad correction. By contrast, (\mathcal A^{\mathrm{dbl}}) is exactly the kind of broad residual family that should open later. That is the key reason scaffold role should dominate literal order.

A good summary is:

[
\text{priority} \approx \text{scaffold role first, structural order second}.
]

**3. A practical family taxonomy for your HH pool**

I would use **four internal classes**.

**Class 0: atomic seed families**
These are split, local, and physically legible.
Default assignment: (\mathcal Q).

This is the clearest early-opening family in your pool. It is the termwise seed form of the local conditional-displacement mechanism, so it is the natural “atom” of scaffold growth.

**Class 1: scaffold / backbone builders**
These establish reusable local or bond structure, but are already more relational than the atomic seed.
Default assignment: (\mathcal G^{(\mathrm{hd})}), and usually (\mathcal A^{\mathrm{sing}}).

(\mathcal G^{(\mathrm{hd})}) is bond-local and scaffold-forming.
(\mathcal A^{\mathrm{sing}}) is the default electronic backbone-adjustment family: earlier than doubles, but not necessarily as early as (\mathcal Q).

**Class 2: local refiners / backbone extension families**
These sharpen or extend an existing scaffold rather than define it from scratch.
Default assignment: (\mathcal G^{(\mathrm{cloud})}), (\mathcal G^{(\mathrm{dd})}), (\mathcal G^{(\mathrm{cd})}).

* (\mathcal G^{(\mathrm{cloud})}): neighborhood extension of local dressing,
* (\mathcal G^{(\mathrm{dd})}): doublon-sector local refinement,
* (\mathcal G^{(\mathrm{cd})}): local aggregate version of a mechanism already seeded by (\mathcal Q).

**Class 3: broad residual / corrective families**
These are expressive closures, not the first scaffold language.
Default assignment: (\mathcal A^{\mathrm{dbl}}).

That is the cleanest default classification.

Two caveats are worth keeping:

* (\mathcal G^{(\mathrm{cloud})}) is the most movable family. If nearest-neighbor cloud formation is core backbone physics in your regime, it can move up to Class 1.
* (\mathcal A^{\mathrm{sing}}) can slide down to Class 2 if your reference already captures the electronic backbone well.

But two placements are very stable:
(\mathcal Q) should be earliest, and (\mathcal A^{\mathrm{dbl}}) should be latest.

There is also a very coarse fallback that is immediately implementable as doctrine: treat the **old (L=2) vocabulary** as the early scaffold language, and treat the **(L=3) additions** as later-opening extensions. In your notation, that means early (\mathfrak F_2), later (\mathfrak F_3\setminus\mathfrak F_2={\mathcal A^{\mathrm{dbl}},\mathcal G^{(\mathrm{cd})}}). That is already a sensible backbone-first policy.

**4. What early-opening and late-opening families should look like**

An **early-opening** family should have most of these properties:

* physically local or single-bond support,
* split or otherwise atomic action,
* clear physical interpretation,
* incremental effect rather than omnibus correction,
* strong **scaffold reuse**, meaning later families refine what it builds rather than replace it,
* modest family breadth, so it does not flood early search with many interchangeable broad fixes,
* good optimizer behavior in the broad sense: admissions tend to accumulate coherently rather than compensate each other.

A **late-opening** family should have most of these properties:

* higher excitation rank, broader span, or more aggregate structure,
* distributed or multi-sector action,
* residual/corrective role rather than scaffold-defining role,
* higher family breadth or many near-substitutes,
* tendency to dominate too early because one admission can absorb several missing scaffold steps at once,
* best interpreted as “close what remains” after the backbone exists.

The important nuance is that **bond-local** can still be early if it is scaffold-building. So “early” does not mean “smallest support no matter what.” It means “constructive and reusable.”

**5. The cleanest controller-side doctrine**

Use a **hybrid map**
[
f \mapsto \bigl(c(f),,o(f)\bigr),
]
where:

* (c(f)) is a **coarse role class**,
* (o(f)) is an optional **within-class order**.

The controller should depend primarily on (c(f)), not on a fine scalar order.

For your default HH taxonomy, I would set
[
c(\mathcal Q)=0,\quad
c(\mathcal G^{(\mathrm{hd})})=1,\quad
c(\mathcal A^{\mathrm{sing}})=1,\quad
c(\mathcal G^{(\mathrm{cloud})})=2,\quad
c(\mathcal G^{(\mathrm{dd})})=2,\quad
c(\mathcal G^{(\mathrm{cd})})=2,\quad
c(\mathcal A^{\mathrm{dbl}})=3.
]

Then add (o(f)) only if you want finer control inside Class 1 or 2.

This is better than a pure order map (f\mapsto o(f)) because there is no honest single total order across all HH families; locality, rank, splitness, and scaffold role do not collapse cleanly to one axis. It is also better than a pure class map with no refinement if you later want small manual distinctions.

So the clean answer to your item 5 is: **hybrid, with class primary and order secondary**.

**6. Practical recommendation**

The doctrine I would actually adopt is:

* **open (\mathcal Q) first;**
* **open (\mathcal G^{(\mathrm{hd})}) and usually (\mathcal A^{\mathrm{sing}}) next;**
* **open (\mathcal G^{(\mathrm{cloud})}), (\mathcal G^{(\mathrm{dd})}), and unsplit (\mathcal G^{(\mathrm{cd})}) after a scaffold exists;**
* **open (\mathcal A^{\mathrm{dbl}}) last.**

So your suspicion is right. The correct prior is not literal **low-order first**. It is:

[
\boxed{\text{atomic-seed / backbone first,\quad broad-corrective later.}}
]

And the most minimal implementable version of that doctrine is even simpler:

[
\boxed{\text{treat } \mathfrak F_2 \text{ as the scaffold vocabulary, and } \mathfrak F_3\setminus\mathfrak F_2 \text{ as later-opening extensions.}}
]

That gives you a family-bias policy that is structural, family-level, implementation-friendly, and independent of (r) and (p).
Because the current controller already emits `site_occupations`, `site_occupations_up`, `site_occupations_dn`, `staggered`, and `doublon` directly from the propagated statevector, this is first a downstream signal-processing problem, not a propagator/controller problem. The object you get is the spectrum of a **driven expectation-value trajectory**, not an equilibrium two-time correlator such as a dynamic structure factor. That distinction matters physically. 

My recommendation in one sentence: use a **windowed Fourier analysis of fluctuation signals** as the default first pass, report **one-sided amplitude spectra** and peak tables, use **staggered or (q=\pi)** modes as the primary density observable for the alternating drive, and treat the current 9-point example as only a coarse sanity check.

## 1. Is a plain post-processing Fourier transform the correct first step?

Yes, with one correction: the right first diagnostic is **not** a bare FFT of the raw arrays as written. It is a Fourier transform of a **preprocessed fluctuation signal** on a clearly chosen time window.

So the first step is:

[
x(t);\to;\text{choose analysis window};\to;\text{remove baseline / drift};\to;\text{apply taper};\to;\text{FFT}
]

That is the mathematically standard route for finite sampled data, and it is exactly appropriate here because the observables are already serialized time series from the propagated state. No repo refactor is justified before this. 

A second, very useful supplement is **harmonic regression / lock-in extraction at the known drive frequency** ( \omega_d ) and a few harmonics. For short driven traces, that can be more informative than generic peak-picking.

## 2. What exactly should be transformed?

For each observable (x_n = x(t_n)), the default object to transform is:

[
\tilde x_n = w_n,[x_n - x_{\text{baseline}}(t_n)]
]

where (w_n) is a tapering window, and (x_{\text{baseline}}) is chosen as follows.

### For per-site occupations (n_j(t))

Do **not** lead with raw (n_j(t)). Remove the trivial density offset first. The clean choice is

[
\delta n_j(t) = n_j(t) - \bar n(t),
\qquad
\bar n(t)=\frac1L\sum_j n_j(t),
]

or, if total particle number is exactly fixed and known,

[
\delta n_j(t)=n_j(t)-N_{\rm e}/L.
]

This projects out the (q=0) common mode. Then, on the chosen analysis window, remove the remaining temporal mean:

[
\delta n_j(t)\to \delta n_j(t)-\langle \delta n_j\rangle_{\rm window}.
]

### For staggered density (m(t)) and pair differences (d_{ij}(t)=n_i(t)-n_j(t))

These already cancel the spatial mean. For spectral content, remove the **window mean**:

[
m(t)\to m(t)-\langle m\rangle_{\rm window},\qquad
d_{ij}(t)\to d_{ij}(t)-\langle d_{ij}\rangle_{\rm window}.
]

### Detrending

Use **constant detrending by default**.
Use **linear detrending only if there is visible secular drift** across the analyzed window, for example from heating, a slow ramp, or controller-induced drift.

That is the right compromise: linear detrending can suppress spurious low-frequency leakage, but it can also erase genuine low-frequency physics. So it should be treated as a robustness check, not the only analysis.

### Windowing

Use a taper unless you have a very specific reason not to. A **Hann window** is the right default. It reduces spectral leakage from the hard finite-time cut.

For strict DC removal under a tapered transform, use the **window-weighted mean**

[
\bar x_w=\frac{\sum_n w_n x_n}{\sum_n w_n}
]

and transform (w_n(x_n-\bar x_w)).

### Practical default

For each signal, store both:

1. the static part: mean / trend,
2. the oscillatory part: windowed FFT of the mean-subtracted signal.

That keeps the zero-frequency physics separate from the AC spectrum.

## 3. What is the right default spectral object?

For these driven deterministic trajectories, the best default is:

### Default

A **one-sided amplitude spectrum** from a tapered DFT / FFT of the chosen window.

For real-valued signals,

[
X_k=\sum_{n=0}^{N-1} w_n \tilde x_n e^{-i2\pi kn/N},
\qquad
\omega_k=\frac{2\pi k}{N\Delta t}.
]

Then report the one-sided amplitude

[
A_k \approx \frac{2|X_k|}{\sum_n w_n}
]

for (k>0) and below Nyquist.

This is the most interpretable object if you want “what oscillation frequencies and amplitudes are present in the observable?”

### Secondary

A **power spectrum** (A_k^2) or a properly normalized periodogram is fine as a secondary plot, especially when comparing relative strength across peaks.

### Not the default here

* **PSD**: mathematically fine, but conceptually less natural for a short deterministic trajectory than an amplitude spectrum.
* **Autocorrelation then FFT**: useful for stationary stochastic processes; not the right lead tool for short driven transients.
* **STFT / spectrogram**: appropriate only when you have enough data and expect time-dependent frequency content.
* **Wavelets**: useful for strongly nonstationary bursts/chirps; not the first pass.

A good driven-system refinement is to add **lock-in amplitudes and phases at (n\omega_d)**. If ( \omega_d ) is known, this is often more physics-facing than a generic FFT.

## 4. What is the physically meaningful “site occupation difference” observable?

There are four distinct objects here.

### Local pair imbalance

[
d_{ij}(t)=n_i(t)-n_j(t)
]

Use this when the physical question is explicitly local: left-vs-right, bond imbalance, edge-vs-center, defect-vs-background.

### Staggered density

[
m(t)=\frac1L\sum_j(-1)^j n_j(t)
]

This is the natural density observable for an alternating drive or bipartite charge pattern. It is the (q=\pi) density mode on a bipartite lattice.

### Even-minus-odd sum

[
\Delta_{\rm eo}(t)=\sum_{j\in {\rm even}}n_j(t)-\sum_{j\in {\rm odd}}n_j(t)
]

This contains the same information as (m(t)), differing only by normalization.

### Full spatial normal modes

The cleanest general object for (L>2) is the spatial mode decomposition

[
\rho_q(t)=\frac1L\sum_j e^{-iqj}[n_j(t)-\bar n(t)].
]

Then analyze the time spectrum of (\rho_q(t)).
This is the correct generalization of “site occupation difference” beyond arbitrary pairwise subtractions:

* (q=0): total density mode, usually trivial,
* (q=\pi): staggered mode,
* other (q): other density patterns.

So the hierarchy I would use is:

* if the drive is alternating: **start with (m(t))**,
* if you care about a specific bond/site contrast: add **(d_{ij}(t))**,
* if (L>2): also compute **(\rho_q(t))**.

In the supplied two-site example, the provided numbers satisfy (n_0(t)=1+m(t)) and (n_1(t)=1-m(t)), so after mean subtraction the spectra of (n_0), (n_1), (m), and (n_0-n_1) are identical up to sign and scale. In that case, separate “per-site” and “staggered” spectra are mostly redundant. 

## 5. For the tiny example (\Delta t=1.25), (T=10), (N=9), what is trustworthy?

Very little is sharply resolvable.

From the supplied sample times, the data are uniformly sampled with (N=9), (\Delta t=1.25), over (t=0,\dots,10). 

### Hard limits

Sampling frequency:
[
f_s=\frac1{\Delta t}=0.8
]

Nyquist:
[
f_{\rm Nyq}=\frac1{2\Delta t}=0.4,
\qquad
\omega_{\rm Nyq}=\frac{\pi}{\Delta t}\approx 2.513
]

DFT bin spacing:
[
\Delta f_{\rm bin}=\frac1{N\Delta t}=\frac1{11.25}\approx 0.0889
]
[
\Delta \omega_{\rm bin}=\frac{2\pi}{N\Delta t}\approx 0.5585
]

Finite-window resolving power is only of order
[
\frac1{T_{\rm obs}}\approx 0.1,
\qquad
\frac{2\pi}{T_{\rm obs}}\approx 0.628,
]
with (T_{\rm obs}=10).

### What that means

After removing DC, you only have four positive-frequency bins:
[
f={0.0889,;0.1778,;0.2667,;0.3556}
]
or
[
\omega={0.5585,;1.117,;1.676,;2.234}.
]

So:

* you cannot resolve narrow peaks,
* you cannot cleanly separate nearby drive / sideband / harmonic lines,
* anything above Nyquist aliases,
* STFT is essentially useless,
* zero-padding gives a prettier curve but no new information.

### What is still defensible

You can extract only a **very coarse statement** such as:

* there is oscillatory content rather than pure monotone drift,
* it lives somewhere in the low-to-mid accessible band,
* one observable is dominated by a slow imbalance mode.

You cannot defend precise claims like:

* “the peak is at ( \omega = 1.31 )”,
* “there is a second harmonic and a phonon sideband”,
* “the linewidth is X”.

If ( \omega_d ) is already known from the run configuration and lies below Nyquist, a direct fit at ( \omega_d ) and (2\omega_d) is more meaningful than a generic FFT on this record.

## 6. What minimum sampling/window conditions should you use for useful HH spectra?

Let

* ( \Omega_{\max} ): highest angular frequency you care about,
* ( \delta\Omega_{\min} ): smallest frequency spacing you want to resolve.

In this setting, ( \Omega_{\max} ) should be chosen from the largest of:

* drive harmonics of interest (n\omega_d),
* phonon frequency ( \omega_{\rm ph} ),
* relevant electronic transition frequencies / gaps,
* any expected sideband locations.

### Bare minimum

[
\Delta t \le \frac{\pi}{\Omega_{\max}}
]
so Nyquist is above the highest frequency of interest.

[
T_{\rm obs} \gtrsim \frac{2\pi}{\delta\Omega_{\min}}
]
to separate features spaced by ( \delta\Omega_{\min} ).

### Practical recommendation

Use at least **8–10 samples per shortest period**, i.e.

[
\Delta t \lesssim \frac{2\pi}{10,\Omega_{\max}}
]

and preferably more if you want weak harmonics or time-frequency analysis.

For the record length, aim for at least **10–20 periods of the slowest oscillation you want to identify cleanly**, or equivalently

[
T_{\rm obs} \gg \frac{2\pi}{\delta\Omega_{\min}}.
]

### Rough target sample counts

For anything beyond a sanity-check spectrum, I would target:

* **(N \sim 128)** as a lower useful floor,
* **(N \sim 256) to (1024)** as the comfortable range.

Nine samples is far below the regime where peaks look trustworthy.

### Driven HH-specific translation

If you want to distinguish:

* the drive line ( \omega_d ),
* harmonics (2\omega_d, 3\omega_d),
* sidebands (n\omega_d \pm \omega_{\rm ph}) or other beat scales,

then:

* choose (\Delta t) from the **largest** of those frequencies,
* choose (T_{\rm obs}) from the **smallest separation** among those lines.

That is the whole sampling theory in one sentence.

## 7. When should you prefer global FFT versus STFT / wavelets?

### Prefer a global FFT when

* the response is approximately stationary or quasi-periodic on the chosen window,
* you have many cycles,
* you care about stable spectral lines.

This is the right tool for a late-time periodic regime.

### Prefer STFT / spectrogram when

* the drive has a ramp, pulse, or finite envelope,
* the response changes qualitatively in time,
* harmonics turn on only during certain intervals,
* controller/manifold events create regime changes,
* heating or drift makes the spectrum time-dependent.

### Prefer wavelets when

* the signal is broadband and transient,
* you expect bursts or chirps,
* you need stronger time localization than STFT provides.

### Current case

With only 9 points, global FFT is the only realistic frequency-domain diagnostic. Time-frequency methods need substantially longer data because each local window must itself contain several periods.

A very practical compromise for future runs is:

* compute a full-window FFT,
* compute a late-time-window FFT,
* only then move to STFT if those disagree because of nonstationarity.

## 8. What should the physics-facing deliverables be?

I would produce four deliverables.

### 1. Time-domain plots

For each analyzed observable:

* raw trace,
* baseline-removed trace,
* marked analysis window.

This prevents spectral plots from floating free of the actual dynamics.

### 2. One-sided amplitude spectra

For:

* selected (\delta n_j(t)),
* (m(t)),
* selected (d_{ij}(t)),
* optionally (\rho_q(t)).

For (L>2), the (\rho_q) spectra are the most compact physical summary.

### 3. Peak table

For each robust peak:

* (f_{\rm peak}) and ( \omega_{\rm peak} ),
* amplitude,
* power if desired,
* phase relative to the drive, if ( \omega_d ) is known,
* a resolution-limited uncertainty, at least of order (2\pi/T_{\rm obs}),
* sensitivity to preprocessing choices.

For deterministic simulation data, “uncertainty” should mostly mean:

* finite-window resolution,
* shift under mean-vs-linear detrend,
* shift under Hann-vs-rectangular window,
* shift under full-trace-vs-late-window analysis.

### 4. Physics comparison

Overlay or tabulate against:

* ( \omega_d ),
* (2\omega_d), (3\omega_d),
* ( \omega_{\rm ph} ),
* expected electronic scales / gaps,
* plausible sidebands ( n\omega_d \pm \omega_{\rm ph} ) or other beat scales.

If the main question is “does the density follow the alternating drive?”, then the most directly physical outputs are:

* (m(t)) spectrum,
* phase of (m) at ( \omega_d ),
* harmonic content of (m).

## 9. Immediate recipe on existing JSON, and better acquisition for future runs

### Immediate post-processing recipe

This is what I would implement now.

1. **Extract time series**

   * Current schema: use `time`, `site_occupations`, `staggered`, and optionally `site_occupations_up/dn`, `doublon`.
   * Legacy schema: fall back to `n_site_*`, `staggered_*`, etc. 

2. **Check sampling**

   * Verify uniform (\Delta t).
   * If not uniform, switch to Lomb–Scargle or a nonuniform Fourier method.

3. **Build observables**

   * (\delta n_j(t)=n_j(t)-\bar n(t)) or (n_j(t)-N_e/L),
   * (m(t)),
   * selected (d_{ij}(t)=n_i(t)-n_j(t)),
   * optionally (\rho_q(t)).

4. **Choose windows**

   * Analyze the full trace.
   * If the trace is longer in other runs, also analyze a late-time window.
   * For the current 9-point trace, use the full window and clearly label the result as coarse only.

5. **Preprocess**

   * subtract window mean,
   * optionally compare with linear detrend,
   * apply Hann window.

6. **Transform**

   * compute one-sided rFFT amplitude spectrum,
   * report both (f) and ( \omega ),
   * do not overinterpret zero-padding.

7. **Peak robustness**

   * keep only peaks stable under:

     * mean-only vs linear detrend,
     * Hann vs rectangular window,
     * full trace vs late-time trace, when available.

8. **Drive-locked supplement**

   * if ( \omega_d ) is known, fit
     [
     x(t)\approx c_0+\sum_{n=1}^{n_{\max}}
     \big[a_n\cos(n\omega_dt)+b_n\sin(n\omega_dt)\big]
     ]
     and report amplitude ( \sqrt{a_n^2+b_n^2} ) and phase.
   * For very short data, this is often more informative than generic FFT bins.

### Better data acquisition recipe

This is the minimal change I would make for future runs.

1. **Keep the propagator/controller unchanged if possible.**
   Increase the **observable emission cadence** first.

2. **Choose output cadence from the highest frequency of interest**
   [
   \Delta t \lesssim \frac{2\pi}{10,\Omega_{\max}}
   ]
   not merely the Nyquist limit.

3. **Choose total horizon from the smallest spacing you need to resolve**
   [
   T_{\rm obs} \gtrsim \frac{2\pi}{\delta\Omega_{\min}}
   ]
   and preferably several times larger.

4. **Aim for (N \ge 128), preferably (256+)** uniformly spaced samples.

5. **If the drive is periodic**

   * record enough late-time evolution to cover (10)–(20) drive periods after transients,
   * if practical, make the analyzed window an integer number of drive periods.

6. **If nonstationarity is expected**

   * record long enough that STFT windows themselves contain several periods,
   * optionally serialize drive phase / controller phase markers to simplify segmentation.

The bottom line is simple: your present pipeline should start with **post-processing only**, and your present 9-point example is **too short and too coarse for resolved spectral claims**. The most defensible current output is a coarse tapered FFT plus a drive-locked harmonic fit. For future runs, the only changes you need first are **denser output cadence** and **longer total observation time**, not a refactor of the propagation logic.
