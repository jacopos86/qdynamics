# Formal manifold warm start for an adaptive variational optimizer

## Scope and notation

This note specifies one reoptimization warm start. It does not define a candidate score. In particular, it does not use candidate Jacobi lookahead, a reconstructed Riemann tensor, mixed candidate--active energy Hessians during broad screening, a future-optionality reward, or qBang's Adam-like momentum.

All tangent spaces are real. For Hilbert vectors,

\[
(u,v)_{\mathbb R}:=\operatorname{Re}\langle u|v\rangle,
\qquad
T^{*_{\mathbb R}}v
=
\bigl((t_1,v)_{\mathbb R},\ldots,(t_d,v)_{\mathbb R}\bigr)^{\mathsf T}.
\]

The provenance words used below have strict meanings:

| Label | Meaning |
|---|---|
| **exact identity** | Follows algebraically from the zero-coordinate restriction, independent of an estimator. |
| **exact-state computed** | Evaluated from statevectors/tangent vectors without sampling error. |
| **measured** | Obtained from a declared quantum observation primitive. |
| **transported** | Pushed between tangent spaces by the selected transport; exact only relative to that selected map. |
| **predicted** | Produced without a new geometric observation, notably by qBroyden. |
| **inferred** | Fixed by secant/Hessian-vector constraints plus a modeling choice. |
| **regularized prior** | Chosen to make an underdetermined initialization usable. |
| **unknown** | Not determined by the available observations. |

The spectral rule is part of the mathematical state. For a positive-semidefinite Gram matrix with eigenvalues \(\lambda_1\geq\cdots\geq\lambda_d\), fix

\[
\tau_G=\max\{\tau_{\rm abs},\tau_{\rm rel}\lambda_1\},
\qquad
\rho=\#\{i:\lambda_i>\tau_G\}.
\]

Then

\[
G_\rho^\dagger
=V_\rho\operatorname{diag}(\lambda_1^{-1},\ldots,\lambda_\rho^{-1})V_\rho^{\mathsf T}.
\]

This is a deterministic numerical quotient, not a statistical rank certificate. The retained/nonretained spectral gap must be stored with the threshold. A curvature-eigenvalue threshold, when needed, is separate from this tangent-rank threshold.

---

## 1. Verdict

A coherent qBroyden-plus-Riemannian-quasi-Newton warm start exists, subject to five qualifications.

First, optimization must occur on a constant-rank active quotient \(V_x=T_x\mathcal M_{\rm act}\), not on the redundant coordinate space with an unqualified inverse. Second, objective curvature must be stored with an explicit type: the primary BFGS branch stores an inverse of the **raised covariant Hessian operator**, while the indefinite SR1 branch stores the direct raised operator; the recycled metric separately maps coordinate covectors into physical tangent vectors. Third, fixed-manifold curvature data must be moved by an explicit tangent transport; identifying parameter arrays at two different physical states is not a transport. Fourth, zero-coordinate ansatz growth exactly preserves the inherited tangent Gram block and pure old coordinate derivatives, but it does **not** generally preserve the enlarged covariant-Hessian block or any block of the enlarged inverse Hessian. Fifth, the published qBroyden recurrence is not an observation-driven QGT secant rule. It is an objective-gradient low-pass preconditioner anchored, at most, by an initial QFIM observation. It is therefore usable only as a labeled predictor with observation-driven correction and reset rules.

The clean representation of the declared resolved tangent model is

\[
T_{x,\rho}=\mathcal E_xL_x,
\qquad
\mathcal E_x^{*}\mathcal E_x=I_\rho,
\qquad
G_{x,\rho}=L_x^{\mathsf T}L_x.
\]

Here \(\mathcal E_x:\mathbb R^\rho\to V_x\) is an orthonormal physical tangent frame and \(L_x:\mathbb R^d\to\mathbb R^\rho\) is the coordinate-to-frame map. Tangent transport and objective inverse curvature live in the \(\mathcal E\)-frame, whose metric is the identity. qBroyden predicts \(L^{\mathsf T}L\), equivalently the singular stretch of \(L\), in a fixed resolved coordinate quotient. It does not update the intrinsic frame metric, which remains \(I_\rho\). This resolves the apparent conflict with \(\nabla g=0\) and prevents double counting.

With \(b=dE\), frame gradient

\[
\bar r=L^{+\mathsf T}b\in\mathbb R^\rho,
\]

and inverse raised-Hessian model \(B\approx\mathcal A^{-1}\) in the orthonormal frame, the unique type-correct step is

\[
\boxed{
z=-B\bar r,
\qquad
p=L^+z=-L^+B L^{+\mathsf T}b.
}
\]

In active circuit coordinates this is equivalently \(p=-B_{\rm coord}M b\) when \(M\approx G^\dagger\) and \(B_{\rm coord}\approx\mathcal A^{-1}\). If the stored matrix instead approximates the inverse covariant form \(h^{-1}:V^*\to V\), the step is \(-h^{-1}b\), with no separate \(M\).

No Riemann tensor belongs in the persistent state. The connection, or a legitimate substitute vector transport, is needed to compare tangent vectors. Curvature of the manifold measures the path dependence of such transport; it is not required to execute one local warm start.

---

## 2. Geometric type system

### 2.1 Active quotient and musical maps

At a constant-rank point, let

\[
T_x:\mathbb R^d\longrightarrow T_x\mathbb P(\mathcal H),
\qquad
T_xe_i=t_i,
\]

where

\[
|t_i\rangle=(I-|\psi\rangle\langle\psi|)\partial_i|\psi\rangle.
\]

The physical active tangent space and coordinate quotient are

\[
V_x=\operatorname{ran}T_x,
\qquad
\mathbb R^d/\ker T_x,
\qquad
\ker T_x=\ker G_x.
\]

The induced metric defines

\[
\flat_x:V_x\to V_x^*,\quad v^{\flat_x}=g_x(v,\cdot),
\qquad
\sharp_x=\flat_x^{-1}:V_x^*\to V_x.
\]

The coordinate differential is \(b\in(\mathbb R^d)^*\). Exact differentiability on the quotient implies

\[
v\in\ker G_x\Longrightarrow b^{\mathsf T}v=0,
\qquad
b\in\operatorname{ran}G_x.
\]

Noise may violate this relationship; projecting \(b\) onto the resolved cotangent range is then a regularization, not an identity.

### 2.2 Gauge-invariant tangent representation

Horizontal statevector tangents acquire the state's global phase. A gauge-invariant embedding uses the density projector

\[
\rho=|\psi\rangle\langle\psi|,
\qquad
\delta\rho(t)=|t\rangle\langle\psi|+|\psi\rangle\langle t|.
\]

For horizontal \(t,u\),

\[
\frac12\operatorname{Tr}\bigl(\delta\rho(t)\delta\rho(u)\bigr)
=\operatorname{Re}\langle t|u\rangle.
\]

Thus the Fubini--Study tangent geometry can be represented by real, gauge-invariant density tangents with Hilbert--Schmidt inner product \(\frac12\operatorname{Tr}(XY)\). An implementation may store statevector tangents after a continuous Pancharatnam gauge choice, but the density-tangent definition is the invariant specification.

Whenever \(\mathcal E_x\) below is a density-tangent frame, \(T_x\) is understood to mean the original horizontal tangent map followed by the isometric embedding \(t\mapsto\delta\rho(t)\). Thus \(T_x\) and \(\mathcal E_x\) always have the same codomain; no formula takes an inner product between a statevector tangent and a density tangent.

Resolve an orthonormal retained frame \(\mathcal E_x:\mathbb R^\rho\to V_{x,\rho}\), let \(\Pi_{x,\rho}=\mathcal E_x\mathcal E_x^*\), and define

\[
T_{x,\rho}=\Pi_{x,\rho}T_x,
\qquad
L_x=\mathcal E_x^*T_x\in\mathbb R^{\rho\times d}.
\]

Then

\[
T_{x,\rho}=\mathcal E_xL_x,
\qquad
G_{x,\rho}=T_{x,\rho}^*T_{x,\rho}=L_x^{\mathsf T}L_x,
\qquad
\Pi_{x,\rho}=\mathcal E_x\mathcal E_x^*
=T_{x,\rho}G_{x,\rho}^\dagger T_{x,\rho}^*.
\]

If the rule discards only exact zero modes, \(T_{x,\rho}=T_x\) and \(G_{x,\rho}=G_x\). If it discards nonzero small modes, these are retained-model identities and \(G_{x,\rho}\neq G_x\); store the discarded residual \(\|G_x-G_{x,\rho}\|\).

Let

\[
b_\rho=P_{\operatorname{row}L_x}b
=L_x^{\mathsf T}L_x^{+\mathsf T}b.
\]

Because \(L_x\) has full row rank, \(L_x^{\mathsf T}\bar r=b_\rho\) has the unique solution

\[
\bar r=L_x^{+\mathsf T}b.
\]

The minimum-Euclidean-norm coordinate representative of a physical frame vector \(z\) is \(p=L_x^+z\). Consequently,

\[
G_{x,\rho}^\dagger=L_x^+L_x^{+\mathsf T},
\qquad
T_{x,\rho}G_{x,\rho}^\dagger b_\rho
=\mathcal E_xL_x^{+\mathsf T}b_\rho
=\mathcal E_xL_x^{+\mathsf T}b.
\]

If nonzero modes were discarded, replacing \(b\) by \(b_\rho\) is part of the declared quotient regularization, not an exact identity for the full differential.

### 2.3 Three different Hessians

The ordinary coordinate Hessian is

\[
Q_{ij}=\partial_i\partial_jE.
\]

It is not a tensor under nonlinear reparameterization. In a local independent active chart, the covariant Hessian is the symmetric bilinear form

\[
h_x(u,v)=\operatorname{Hess}_gE_x(u,v),
\qquad
h_{ij}=Q_{ij}-\Gamma^a{}_{ij}b_a.
\]

This coordinate formula is not defined on the singular redundant \(d\)-coordinate pullback merely by replacing \(G^{-1}\) with \(G^\dagger\); one must first pass to the resolved quotient chart.

Equivalently,

\[
h_x^\flat:V_x\to V_x^*,
\qquad
h_x^\flat(u)=h_x(u,\cdot).
\]

The raised-index Hessian operator is

\[
\mathcal A_x=\sharp_x\circ h_x^\flat:V_x\to V_x.
\]

It is self-adjoint with respect to \(g_x\). The authoritative BFGS object in this note is

\[
B_x:V_x\to V_x,
\qquad
B_x\approx\mathcal A_x^{-1}.
\]

The associated inverse bilinear tensor is

\[
K_x=B_x\sharp_x:V_x^*\to V_x,
\qquad
K_x\approx(h_x^\flat)^{-1}.
\]

Therefore the physical Newton-like tangent is

\[
\eta_N=-B_x\sharp_x b=-K_xb\in V_x.
\]

In a nonorthonormal active coordinate basis, \(B\) is metric-self-adjoint when

\[
GB=B^{\mathsf T}G,
\]

not necessarily when \(B=B^{\mathsf T}\). By contrast, \(K=BG^{-1}\), viewed as a covector-to-vector matrix, is ordinarily symmetric. In the orthonormal \(\mathcal E\)-frame, all three distinctions reduce to ordinary symmetric matrices, but their tensor types remain different.

On a \(\rho\)-dimensional independent active chart, where \(G\succ0\), a nonsingular linear reparameterization \(\theta=S\phi\) gives

\[
b_\phi=S^{\mathsf T}b_\theta,
\quad
G_\phi=S^{\mathsf T}G_\theta S,
\quad
M_\phi=S^{-1}M_\theta S^{-\mathsf T},
\quad
B_\phi=S^{-1}B_\theta S,
\]

and hence

\[
B_\phi M_\phi b_\phi=S^{-1}B_\theta M_\theta b_\theta.
\]

This variance check is the simplest way to detect an illicit extra metric factor.

For a singular full-coordinate pullback, the Moore--Penrose inverse is not congruence-covariant under a general nonorthogonal \(S\). Likewise, \(L^+z\) is the minimum-Euclidean-norm representative in a declared coordinate convention and need not transform tensorially. The physical vector \(\mathcal Ez\) is intrinsic; full redundant-coordinate pseudoinversion is a lift convention. The displayed variance test therefore applies either on the resolved nonsingular quotient chart or under an orthogonal change of the redundant ambient coordinates.

### 2.4 Persistent state

Define a tagged authoritative curvature state

\[
\mathfrak H=
\operatorname{InverseBFGS}(B)
\quad\text{or}\quad
\operatorname{DirectSR1}(A).
\]

Only one branch is authoritative at a time. The smallest useful state is

\[
\boxed{
\mathfrak W=
(x,e,\theta,\mathcal I,\mathcal E,L,b,\mathfrak H,\Delta,\rho,\Sigma,\mathcal C),
\qquad e=E(x).
}
\]

Here \(e\) is the stored current energy; \(\mathcal I\) is the logical-coordinate registry and storage permutation; \(\mathcal E\) is the resolved orthonormal physical frame or a backend frame handle; \(L\) is the coordinate-to-frame map; \(b\) is retained so gradients can be recomputed after a metric correction; \(\mathfrak H\) is either the inverse raised-objective-Hessian model \(B\) or the direct self-adjoint model \(A\), both in the \(\mathcal E\)-frame; \(\Delta\) is the Fubini--Study trust radius; \(\rho\) is active rank; \(\Sigma\) stores provenance, thresholds, innovations, and validity flags; and \(\mathcal C\) is a complete pre-growth checkpoint used for rollback.

The following are derived, not independent state:

\[
T_\rho=\mathcal EL,
\quad
G_\rho=L^{\mathsf T}L,
\quad
M=L^+L^{+\mathsf T},
\quad
\Pi=\mathcal E\mathcal E^*,
\quad
\bar r=L^{+\mathsf T}b.
\]

For qBroyden, it is convenient to cache a reduced inverse metric. If \(Z\in\mathbb R^{d\times\rho}\) is an orthonormal basis for \(\operatorname{row}L\), then

\[
L_R=LZ\in\mathbb R^{\rho\times\rho},
\quad
G_R=L_R^{\mathsf T}L_R\succ0,
\quad
M_R=G_R^{-1},
\quad
M=ZM_RZ^{\mathsf T}.
\]

The cache \((Z,M_R)\) is invalidated whenever resolved rank or resolved coordinate range changes.

---

## 3. Exact ansatz-growth theorem

### 3.1 Statement and inherited tangents

Let

\[
F:\Theta_d\times U\subset\mathbb R^{d+m}\to\mathbb P(\mathcal H),
\qquad
F(\theta,0)=f(\theta)
\]

hold on a neighborhood of \(\theta_\star\), not merely at one point. Assume \(f,F\in C^2\), and work on constant-rank strata. Choose normalized local lifts \(\psi(\theta)\) and \(\Psi(\theta,\alpha)\). Equality as rays means that locally

\[
\Psi(\theta,0)=e^{i\chi(\theta)}\psi(\theta).
\]

Then

\[
\begin{aligned}
\widetilde t_i
&=(I-|\Psi\rangle\langle\Psi|)\partial_{\theta_i}\Psi(\theta,0)\\
&=e^{i\chi}(I-|\psi\rangle\langle\psi|)
\bigl(\partial_i\psi+i(\partial_i\chi)\psi\bigr)\\
&=e^{i\chi}t_i.
\end{aligned}
\]

Thus every inherited **projective** tangent is exact after the canonical phase identification. Raw Hilbert derivatives need not be literally equal because they may differ by a common phase and a vertical term.

Let

\[
T_A=[t_1,\ldots,t_d],
\qquad
D_B=[d_1,\ldots,d_m]
\]

be expressed in a common lift. Then

\[
\boxed{
G^+=
\begin{bmatrix}
G_{AA}&G_{AB}\\
G_{BA}&G_{BB}
\end{bmatrix}
=
\begin{bmatrix}
G^-&\operatorname{Re}(T_A^\dagger D_B)\\
\operatorname{Re}(D_B^\dagger T_A)&\operatorname{Re}(D_B^\dagger D_B)
\end{bmatrix}.
}
\]

This identity is independent of rank. Its quotient interpretation changes when \(G^-\) or \(G^+\) is singular.

### 3.2 Arbitrary circuit insertion, reorder, and sharing

Zero-parameter factors may be inserted at arbitrary circuit positions provided:

1. every inserted factor satisfies \(V_\mu(0)=I\);
2. its zero value is independent of all old coordinates;
3. the ordered subsequence of old circuit factors is unchanged; and
4. the restriction equality holds for all nearby \(\theta\).

Deleting the identity factors then recovers the old circuit as a function of \(\theta\), so every pure old derivative agrees. The insertion location changes \(D_B\), because the new generator is conjugated by the circuit suffix/prefix appropriate to its location, but it does not change \(T_A\). Reordering noncommuting old gates is a physical circuit change, not a coordinate permutation, and generally destroys the restriction equality.

For arbitrary storage order, let \(J_A\in\mathbb R^{(d+m)\times d}\) inject old logical coordinates into their new storage slots and \(J_B\) inject the new coordinates. Then

\[
DF\,J_A=Df,
\qquad
J_A^{\mathsf T}G^+J_A=G^-.
\]

With \(P=[J_A,J_B]\), the canonical block matrix is \(P^{\mathsf T}G_{\rm storage}P\). For an affine injection,

\[
b^-=J_A^{\mathsf T}b^+,
\qquad
Q^-=J_A^{\mathsf T}Q^+J_A.
\]

Equivalently, under storage permutation,

\[
b_{\rm storage}=Pb_{\rm canonical},
\qquad
Q_{\rm storage}=PQ_{\rm canonical}P^{\mathsf T}.
\]

Thus \(b\) transforms as a covector, not by congruence. Equality \(G_{AA}=G^-\) is true only after the explicit logical remapping.

If an old logical parameter controls several gates, its tangent is the sum of the occurrence-level derivatives. That summed tangent is inherited exactly if the tying map is unchanged on \(\alpha=0\). If a tied old parameter is untied into occurrence coordinates \(\beta=L\theta\), only

\[
T_\beta L=T_{\rm old},
\qquad
G_{\rm old}=L^{\mathsf T}G_\beta L
\]

is exact; no principal block of \(G_\beta\) need equal \(G_{\rm old}\). If an old and new gate share a parameter so that no local product chart \((\theta,\alpha)\) exists, the block theorem does not apply. The exact statement is instead the Jacobian pullback of the actual tying map.

### 3.3 Exact projector enlargement

Using the true Moore--Penrose inverse, the inherited real-orthogonal projector is

\[
\Pi_A=T_A(G^-)^\dagger T_A^{*_{\mathbb R}}.
\]

It is self-adjoint and idempotent because

\[
\Pi_A^2
=T_A(G^-)^\dagger G^-(G^-)^\dagger T_A^{*_{\mathbb R}}
=\Pi_A,
\]

and it is the identity on \(\operatorname{ran}T_A\). Define

\[
R_B=(I-\Pi_A)D_B,
\qquad
S_B=R_B^{*_{\mathbb R}}R_B.
\]

Then

\[
\boxed{
S_B=G_{BB}-G_{BA}(G^-)^\dagger G_{AB}.
}
\]

For an exact Gram matrix the generalized-Schur range condition is automatic:

\[
\operatorname{ran}G_{AB}\subseteq\operatorname{ran}G^-.
\]

Indeed, \(z\in\ker G^-\) implies \(T_Az=0\), hence \(z^{\mathsf T}G_{AB}=0\). Separately predicted or independently noise-corrupted blocks need not satisfy this condition. Nor is the range inclusion automatic relative to a **threshold-retained** eigenspace: a discarded small singular direction is not a true kernel direction.

Because \(\operatorname{ran}R_B\perp\operatorname{ran}T_A\),

\[
\boxed{
\Pi_{A\oplus B}
=\Pi_A+R_BS_B^\dagger R_B^{*_{\mathbb R}}
}
\]

is the orthogonal projector onto \(\operatorname{span}_{\mathbb R}(T_A,D_B)\). With an eigendecomposition

\[
S_BV_+=V_+\Lambda_+,
\qquad
\Lambda_+>\tau_S I,
\]

an orthonormal residual frame is

\[
\mathcal E_B=R_BV_+\Lambda_+^{-1/2},
\qquad
q=\operatorname{rank}_{\tau_S}S_B,
\]

and

\[
\operatorname{rank}G^+=\operatorname{rank}G^-+\operatorname{rank}S_B
\]

is the exact algebraic rank identity. For the numerical algorithm define operationally

\[
\rho^+_{\rm declared}
:=\rho^-_{\rm retained}+\operatorname{rank}_{\tau_S}S_B.
\]

A separate thresholded eigendecomposition of the full coordinate Gram matrix \(G^+\) need not give the same count because a nonorthogonal block change can move coordinate-Gram eigenvalues. Disagreement is a rank-instability diagnostic, not a theorem violation. Likewise, replacing \((G^-)^\dagger\) by its truncated version makes \(\Pi_A\) the projector only onto the retained left-singular range, and all residual/range statements then refer to that declared numerical subspace.

For a singleton,

\[
r_\mu=(I-\Pi_A)d_\mu,
\]

and, if \((r_\mu,r_\mu)_{\mathbb R}>\tau_S\),

\[
\Pi_{A\oplus\mu}
=\Pi_A+
\frac{|r_\mu\rangle\langle r_\mu|_{\mathbb R}}
{(r_\mu,r_\mu)_{\mathbb R}}.
\]

Sequential singleton updates with exact reorthogonalization are order independent in exact arithmetic: after each update the projector is the unique projector onto the span processed so far, so the final projector is the unique projector onto \(\operatorname{span}(T_A,D_B)\). The residual basis is order dependent. Finite precision, incomplete reorthogonalization, and threshold crossings restore order dependence. A block SVD/eigendecomposition is preferable for a correlated batch because it resolves collective rank and exposes its spectral gap.

### 3.4 Energy derivatives: exact coordinate identity

The scalar restriction

\[
E^+(\theta,0)=E^-(\theta)
\]

on a neighborhood gives, by ordinary differentiation,

\[
\boxed{
\partial_iE^+(\theta,0)=\partial_iE^-(\theta),
\qquad
\partial_i\partial_jE^+(\theta,0)=\partial_i\partial_jE^-(\theta).
}
\]

For an affine injection, \(Q^-=J_A^{\mathsf T}Q^+J_A\). For a nonlinear tying map \(z=\phi(\theta)\),

\[
Q^-_{ij}
=J^a{}_iQ^+_{ab}J^b{}_j
+b_a^+\,\partial_i\partial_j\phi^a,
\]

which displays the ordinary Hessian's non-tensorial term.

### 3.5 Covariant-Hessian qualification

After resolving redundancy, suppose the old regular manifold \(\mathcal M\) is locally isometrically immersed in the enlarged regular manifold \(\widetilde{\mathcal M}\). For \(X,Y\in T_x\mathcal M\), the Gauss formula is

\[
\widetilde\nabla_XY=\nabla_XY+\mathrm{II}(X,Y),
\qquad
\mathrm{II}(X,Y)\in T_x\widetilde{\mathcal M}\ominus T_x\mathcal M.
\]

Therefore

\[
\begin{aligned}
\operatorname{Hess}_{\widetilde{\mathcal M}}\widetilde E(X,Y)
&=X(Y\widetilde E)-d\widetilde E(\widetilde\nabla_XY)\\
&=\operatorname{Hess}_{\mathcal M}E(X,Y)
-d\widetilde E\bigl(\mathrm{II}(X,Y)\bigr)\\
&=\boxed{
\operatorname{Hess}_{\mathcal M}E(X,Y)
-g\bigl((\operatorname{grad}\widetilde E)^\perp,\mathrm{II}(X,Y)\bigr).
}
\end{aligned}
\]

Equivalently,

\[
h^+_{AA}=h^-_{AA}-g(A_{r_\perp}\,\cdot,\cdot),
\qquad
r_\perp=(\operatorname{grad}\widetilde E)^\perp,
\]

where \(g(A_nX,Y)=g(n,\mathrm{II}(X,Y))\). In coordinates,

\[
h^+_{ij}-h^-_{ij}
=\Gamma^{-k}{}_{ij}b^-_k-\Gamma^{+c}{}_{ij}b^+_c.
\]

Hence the enlarged old--old covariant Hessian equals the old covariant Hessian only if the normal gradient vanishes, the old zero section is totally geodesic, or the particular normal-gradient/second-fundamental-form pairing vanishes. At a point stationary only within the old ansatz,

\[
\operatorname{grad}_{\mathcal M}E=0,
\qquad
(\operatorname{grad}_{\widetilde{\mathcal M}}\widetilde E)^\perp\ne0
\]

is precisely possible, and then

\[
h^-_{AA}=Q_{AA},
\qquad
h^+_{AA}=Q_{AA}-g(r_\perp,\mathrm{II}_{AA})
\]

generally differs. At a stationary point of the enlarged ansatz, the correction vanishes. Identity insertion proves a zero-section restriction; it does not prove that zero section is totally geodesic.

A minimal falsifying example is

\[
\mathcal M=\{(\theta,\theta^2)\}\subset\mathbb R^2,
\qquad
\widetilde E(x,y)=y.
\]

At \(\theta=0\), the restricted objective is \(E(\theta)=\theta^2\), with Hessian \(2\), while the ambient Hessian is zero. The discrepancy is exactly the normal-gradient/second-fundamental-form term.

On a singular coordinate pullback, inserting \(G^\dagger\) into the usual Christoffel formula does not automatically define a Levi--Civita connection on the redundant coordinates. Form \(h\) on a constant-rank quotient chart or in a resolved orthonormal frame.

### 3.6 Why the inverse-Hessian block claim is false

In a common orthonormal tangent frame, write a completed symmetric objective-Hessian operator as

\[
H=
\begin{bmatrix}
A&C\\
C^{\mathsf T}&D
\end{bmatrix}.
\]

If \(A\) and \(S=D-C^{\mathsf T}A^{-1}C\) are invertible,

\[
\boxed{
H^{-1}=
\begin{bmatrix}
A^{-1}+A^{-1}CS^{-1}C^{\mathsf T}A^{-1}&-A^{-1}CS^{-1}\\
-S^{-1}C^{\mathsf T}A^{-1}&S^{-1}
\end{bmatrix}.
}
\]

Thus

\[
(H^{-1})_{AA}
=A^{-1}+A^{-1}CS^{-1}C^{\mathsf T}A^{-1}
\ne A^{-1}
\]

unless the decomposition reduces \(H\), notably \(C=0\), or a special cancellation occurs. Equivalently, if \(D\) is invertible,

\[
(H^{-1})_{AA}=(A-CD^{-1}C^{\mathsf T})^{-1}.
\]

For a singular symmetric matrix, if

\[
H=U_\kappa\Lambda_\kappa U_\kappa^{\mathsf T},
\qquad
U_\kappa=\begin{bmatrix}U_A\\U_B\end{bmatrix},
\]

then

\[
H^\dagger=U_\kappa\Lambda_\kappa^{-1}U_\kappa^{\mathsf T},
\qquad
(H^\dagger)_{AA}=U_A\Lambda_\kappa^{-1}U_A^{\mathsf T}.
\]

This is not generally \((H_{AA})^\dagger\). For example,

\[
H=\begin{bmatrix}1&1\\1&1\end{bmatrix},
\qquad
H^\dagger=\frac14\begin{bmatrix}1&1\\1&1\end{bmatrix},
\qquad
(H_{AA})^\dagger=1.
\]

Replacing every inverse in a Schur formula by a Moore--Penrose inverse is invalid without additional range and orthogonality hypotheses. The exact generic pseudoinverse is obtained from the resolved spectrum of the **completed full operator**. The special decoupled identity is

\[
C=0\Longrightarrow H^\dagger=\operatorname{diag}(A^\dagger,D^\dagger).
\]

Residualizing new tangents makes the metric block diagonal. It does not make \(C=h_{AB}\) vanish.

### 3.7 Honest curvature initializations

Let \(\mathcal E^+=[\mathcal E_A,\mathcal E_B]\) be the inherited plus whitened-residual frame.

If only the inherited inverse-curvature model is available, use

\[
\boxed{
B_0^+=
\begin{bmatrix}
B_A&0\\
0&\beta I_q
\end{bmatrix},
\qquad \beta>0.
}
\]

This is a regularized prior. \(B_A\) is inherited model information; it is not generally an exact block of the new inverse. The mixed prior is zero but the true mixed block is unknown. The residual block is isotropic because the residual frame is orthonormal; \(\beta\) may be chosen from a clipped median scale of \(B_A\), but remains a prior.

If new diagonal covariant curvatures \(q_\mu=h(e_\mu,e_\mu)\) are observed, form a direct model

\[
\widehat A_0^+=
\begin{bmatrix}
\widehat A_A&0\\
0&\operatorname{diag}(q_\mu)
\end{bmatrix}.
\]

Only the \(q_\mu\) are measured. Mixed entries and new--new off-diagonals remain unknown. In positive-definite BFGS mode, spectrally clip this direct prior before inversion; in indefinite mode, retain its signs and use a trust-region SR1 model.

An observed Hessian-vector product \(z=Av\) supplies a linear action constraint. Given a symmetric direct prior \(\widehat A\), let \(r=z-\widehat Av\). The symmetric correction

\[
\Delta A
=\frac{rv^{\mathsf T}+vr^{\mathsf T}}{v^{\mathsf T}v}
-\frac{r^{\mathsf T}v}{(v^{\mathsf T}v)^2}vv^{\mathsf T}
\]

satisfies \((\widehat A+\Delta A)v=z\). When \(r^{\mathsf T}v\) is safely nonzero, the SR1 correction

\[
\Delta A_{\rm SR1}=\frac{rr^{\mathsf T}}{r^{\mathsf T}v}
\]

does as well. An HVP along each new residual basis direction reveals the corresponding full columns, including old--new couplings. Unprobed actions remain inferred or unknown.

The complete growth ledger is therefore:

| Object at growth | Status |
|---|---|
| true \(G^+_{AA}\) | exact identity from \(G^-\) |
| stored predicted \(\widehat G^-\) carried into \(AA\) | reused predictor; its provenance does not improve |
| \(G_{AB},G_{BB}\) | exact-state computed or measured from new tangents; otherwise unknown |
| \(\Pi_A,R_B,S_B,\Pi_{A\oplus B}\) | classical-exact given exact Gram/tangent data |
| true \(Q^+_{AA}\) | exact coordinate restriction |
| true \(h^+_{AA}\) | qualified by the normal-gradient/\(\mathrm{II}\) correction |
| \(h_{AB}\) | unknown unless observed or constrained by HVPs |
| observed \(h_{BB,\mu\mu}\) | measured |
| any block of \((h^+)^{-1}\) or \((\mathcal A^+)^{-1}\) | inferred unless the completed full operator is known and inverted |

The block-diagonal inverse initialization in [Ramôa *et al.*](https://doi.org/10.1088/2058-9565/ad904e) is therefore a positive-definite Euclidean-coordinate prior, exactly as its zero mixed/new information content suggests; it is not an inverse-block theorem.

More precisely, that paper stores an ordinary parameter-coordinate inverse-BFGS matrix \(H_k\approx Q_k^{-1}\) and initializes its \(n\)-parameter ADAPT optimization by

\[
H_0^{(n)}=
\begin{bmatrix}
H_*^{(n-1)}&0\\
0&1
\end{bmatrix}.
\]

Later parameter and coordinate-gradient differences update this prior by Euclidean BFGS. The construction uses no QGT, Levi--Civita connection, horizontal quotient, tangent transport, or explicit Hessian/HVP observation. Its unit residual scale is coordinate-scale dependent; its rigorous algebraic virtue is preservation of positive definiteness, while its measurement saving comes from faster later optimization.

---

## 4. Fixed-manifold transport construction

### 4.1 Comparison

| Map | Domain/codomain | Character | Isometric? | Required information | Status for VQA submanifold | Quasi-Newton suitability |
|---|---|---|---|---|---|---|
| Levi--Civita parallel transport \(P_\gamma\) | \(V_{x_j}\to V_{x_{j+1}}\) | intrinsic, path dependent | yes | connection or tangent-frame evolution along the path | exact if integrated exactly | canonical; strongest theory |
| differentiated-retraction transport \(DR_x(\eta)\) | \(V_x\to V_{R_x(\eta)}\) | tied to chosen retraction; coordinate-free only if retraction is | generally no | derivative/action of retraction | exact derivative of the selected retraction, not parallel transport | theory requires the appropriate adjoint/scaling/locking conditions |
| endpoint polar/Procrustes alignment | \(V_{x_j}\to V_{x_{j+1}}\) | extrinsic, endpoint based | yes by construction at equal rank | two resolved endpoint tangent frames and cross overlaps | exact as a defined partial isometry; approximate to Levi--Civita | coherent experimental transport; do not claim a theorem without checking its hypotheses |

All three require constant active rank for a full isomorphism. Rank deficiency must be resolved before constructing them. A rank change supports at most a partial common-subspace map.

### 4.2 Levi--Civita transport

For a \(C^1\) curve \(\gamma\), parallel transport is defined by

\[
\nabla_{\dot\gamma}V=0,
\qquad
V(0)=v,
\qquad
P_\gamma v=V(1).
\]

Metric compatibility gives

\[
g_{\gamma(1)}(P_\gamma u,P_\gamma v)=g_{\gamma(0)}(u,v).
\]

In a smooth orthonormal density-tangent frame \(\mathcal E(t)\), write \(V(t)=\mathcal E(t)a(t)\). For the induced connection,

\[
\dot a(t)=-\Omega(t)a(t),
\qquad
\Omega(t)=\mathcal E(t)^*\dot{\mathcal E}(t),
\qquad
\Omega^{\mathsf T}=-\Omega,
\]

so

\[
a(1)=\mathcal P\exp\!\left[-\int_0^1\Omega(t)\,dt\right]a(0).
\]

This is exact but requires intermediate tangent-frame or connection data. A same-state metric at the two endpoints is insufficient.

### 4.3 Retraction transport

For a retraction \(R_x\),

\[
\mathcal T^{DR}_\eta\xi=DR_x(\eta)[\xi]
\]

is linear in \(\xi\) and maps into \(V_{R_x(\eta)}\). It is generally nonisometric. If it is full rank, one may polar-isometrize it:

\[
\overline{\mathcal T}_\eta
=\mathcal T^{DR}_\eta
\left[(\mathcal T^{DR}_\eta)^*\mathcal T^{DR}_\eta\right]^{-1/2}.
\]

A coordinate update \(\theta\mapsto\theta+p\) need not define a quotient retraction if two redundant representatives of the same tangent vector produce different endpoint tangents. The backend must specify a horizontal/minimum-norm lift or another quotient-consistent convention.

For a nonisometric invertible map \(L\), the raised operator transports by similarity \(LBL^{-1}\), whereas a contravariant inverse bilinear tensor transports by congruence \(LKL^*\). These agree through the metric identifications only for an isometry. Raw nonisometric transport followed by the ordinary inverse-BFGS formula loses its standard self-adjointness and convergence semantics.

### 4.4 Gauge-invariant endpoint Procrustes transport

Let \(\mathcal E_j,\mathcal E_{j+1}\) be gauge-invariant orthonormal density-tangent frames of equal rank \(\rho\), and form

\[
C_{j+1,j}=\mathcal E_{j+1}^*\mathcal E_j.
\]

If statevector tangents are used instead, an independent phase at the two endpoints changes \(\operatorname{Re}(\mathcal E_{j+1}^\dagger\mathcal E_j)\); a pathwise relative phase convention is mandatory. Density tangents avoid this issue.

Let

\[
C=U\Sigma V^{\mathsf T},
\qquad
Q=UV^{\mathsf T}.
\]

Then \(Q\) solves

\[
\min_{Q^{\mathsf T}Q=I}
\|\mathcal E_{j+1}Q-\mathcal E_j\|_F,
\]

and

\[
\boxed{
\mathcal T_{j+1,j}
=\mathcal E_{j+1}Q\mathcal E_j^*
}
\]

is a partial isometry with

\[
\mathcal T^*\mathcal T=\Pi_j,
\qquad
\mathcal T\mathcal T^*=\Pi_{j+1}.
\]

It is invariant under independent orthogonal frame changes. The singular values \(\sigma_i(C)=\cos\vartheta_i\) are the cosines of principal angles. The polar factor is unique and smooth while \(C\) is nonsingular. If

\[
\sigma_{\min}(C)\leq\tau_{\rm align},
\]

the full transport is invalidated: on a zero-singular subspace \(Q\) depends on arbitrary SVD choices, not merely on a poorly conditioned numerical calculation. A well-separated common-overlap subspace may be transported only as a prior.

This map is **not** Levi--Civita parallel transport in general: it is endpoint based and path independent, whereas parallel transport is path dependent. On a smooth, short, constant-rank accepted curve segment, with a smooth frame choice and nonsingular \(C\), it agrees with parallel transport through first order. Indeed, with \(\mathcal E(t)=\mathcal E(0)+t\dot{\mathcal E}(0)+O(t^2)\), its polar factor cancels the tangential skew term \(\Omega=\mathcal E^*\dot{\mathcal E}\), leaving precisely the first-order parallel derivative. Under those hypotheses the discrepancy is \(O(\|\eta\|^2)\).

The primary exact-state backend should use this density-tangent Procrustes map. It uses only endpoint frames, is exactly isometric as a selected map, preserves symmetric positive-definite curvature models under

\[
B_{j+1}^{\rm tr}=QB_jQ^{\mathsf T},
\]

and avoids pretending that Christoffels or the Riemann tensor were reconstructed. It is suitable for an experimental Riemannian BFGS state, but a superlinear-convergence theorem should not be claimed unless the chosen retraction, transport, and locking hypotheses of that theorem are separately verified. The relevant established constructions are in [Huang, Gallivan, and Absil (2015)](https://doi.org/10.1137/140955483) and [Huang, Absil, and Gallivan (2018)](https://doi.org/10.1137/17M1127582).

A measurement-limited backend must expose at least one of

\[
\texttt{cross\_tangent\_gram}(x_{j+1},x_j)\to\widehat C
\]

or

\[
\texttt{transport\_action}(\eta,v)\to\widehat{\mathcal T_\eta v}.
\]

Two same-state QGT matrices do not determine the relative orientation \(C\). Direct evaluation of density-tangent cross overlaps from two statevectors is a statevector-only primitive unless a hardware overlap circuit and noise model are explicitly supplied. If neither transport interface exists, cross-step Hessian recycling must be damped or reset; qBroyden cannot manufacture a tangent transport.

At zero-coordinate outer growth there is no physical motion, so no Procrustes transport is needed. The old tangent subspace is included identically and only its new residual complement is initialized.

---

## 5. qBroyden metric recycling

### 5.1 The update actually published

The primary source is Fitzek *et al.*, [*Optimizing Variational Quantum Algorithms with qBang*](https://quantum-journal.org/papers/q-2024-04-09-1313/) ([arXiv:2304.13882v2](https://arxiv.org/pdf/2304.13882)). Its equations (9)--(10) and qBroyden algorithm do **not** use a Broyden secant pair. Let

\[
b_k=\nabla_\theta\mathcal L(\theta_k),
\qquad
\epsilon_k=\frac{\epsilon_0}{k+1},
\qquad
a_k=1-\epsilon_k.
\]

The paper denotes its approximate QFIM by \(B_k\); to avoid collision with objective inverse curvature, call it \(\mathsf F_k\). The direct recurrence is

\[
\boxed{
\mathsf F_{k+1}=a_k\mathsf F_k+\epsilon_k b_kb_k^{\mathsf T}.
}
\tag{5.1}
\]

The stored inverse \(K_k=\mathsf F_k^{-1}\) is updated by Sherman--Morrison:

\[
\boxed{
K_{k+1}
=\frac1{a_k}
\left[
K_k-
\frac{\epsilon_kK_kb_kb_k^{\mathsf T}K_k}
{a_k+\epsilon_kb_k^{\mathsf T}K_kb_k}
\right].
}
\tag{5.2}
\]

There is no displacement \(s_k\), gradient difference \(y_k\), or condition \(\mathsf F_{k+1}s_k=y_k\). The rank-one datum is the raw coordinate energy-differential outer product \(b_kb_k^{\mathsf T}\). Calling (5.1) a classical Broyden secant update would be false.

The permitted initialization is a full QFIM, a block-diagonal QFIM approximation, or the identity. Full-QFIM qBroyden observes the QFIM at the first iteration only; later iterations use energy gradients and classical algebra. The paper suggests a possible eventual restart but gives neither a scheduled refresh nor a certification rule. Its advertised cost is therefore one \(O(d^2)\) geometric initialization followed by \(O(d)\) gradient evaluations per step. qBang's Adam-like momentum is a separate state and is not used here.

Unrolling (5.1) gives

\[
\mathsf F_k
=\left(\prod_{j=0}^{k-1}a_j\right)\mathsf F_0
+\sum_{\ell=0}^{k-1}
\left[
\epsilon_\ell\prod_{j=\ell+1}^{k-1}a_j
\right]b_\ell b_\ell^{\mathsf T}.
\]

It is a convex low-pass mixture of the initial matrix and past gradient outer products. Two tests prohibit interpreting it as an exact QGT evolution law:

\[
H\mapsto cH
\Longrightarrow
b_kb_k^{\mathsf T}\mapsto c^2b_kb_k^{\mathsf T},
\qquad
G\mapsto G,
\]

and, at \(b_k=0\),

\[
\mathsf F_{k+1}=a_k\mathsf F_k
\]

even if the true QFIM is constant. Thus qBroyden is an objective-dependent predicted preconditioner, not a measured or statistically calibrated metric.

### 5.2 QFIM-to-Fubini--Study normalization

The paper defines

\[
F_{ij}
=4\operatorname{Re}
\left[
\langle\partial_i\psi|\partial_j\psi\rangle
-\langle\partial_i\psi|\psi\rangle
\langle\psi|\partial_j\psi\rangle
\right]
=4G_{ij}.
\]

Therefore, on a fixed resolved active range,

\[
F^{-1}=\frac14G^{-1},
\qquad
M_G=G^{-1}=4F^{-1}.
\]

To reproduce the paper while storing \(M_k\approx G_k^{-1}\), (5.2) becomes

\[
\boxed{
M_{k+1}
=\frac1{1-\epsilon_k}
\left[
M_k-
\frac{(\epsilon_k/4)M_kb_kb_k^{\mathsf T}M_k}
{1-\epsilon_k+(\epsilon_k/4)b_k^{\mathsf T}M_kb_k}
\right].
}
\tag{5.3}
\]

Equivalently, the direct \(G\)-convention recurrence is

\[
\widehat G_{k+1}
=(1-\epsilon_k)\widehat G_k
+\frac{\epsilon_k}{4}b_kb_k^{\mathsf T}.
\tag{5.4}
\]

Omitting \(1/4\) changes the relative weighting of inherited geometry and gradient innovations; it is not merely a learning-rate convention.

The same conversion applies at initialization. The paper's \(\mathsf F_0=I\) option corresponds to \(\widehat G_0=I/4\) and \(M_{G,0}=4I\). Starting the \(G\)-convention recurrence from \(M_{G,0}=I\) is a permissible but different prior. A measured full or block QFIM must be divided by four before it is treated as a \(G\) anchor.

### 5.3 Symmetry, positivity, and semidefinite metrics

If \(\mathsf F_k\succeq0\) and \(0\leq\epsilon_k\leq1\), (5.1) preserves symmetry and positive semidefiniteness. If \(\mathsf F_0\succ0\) and \(0\leq\epsilon_k<1\), it remains positive definite and

\[
1-\epsilon_k+\epsilon_kb_k^{\mathsf T}\mathsf F_k^{-1}b_k
\geq1-\epsilon_k>0.
\]

The inverse update is then valid and should be implemented in the explicitly symmetric form (5.2), followed by numerical symmetrization.

For a genuinely semidefinite pullback metric, only the direct recurrence is automatically defined. The published inverse formula is not a generic Moore--Penrose update. On a fixed resolved range with

\[
b_k\in\operatorname{ran}\mathsf F_k,
\qquad
\ker\mathsf F_{k+1}=\ker\mathsf F_k,
\]

the same formula holds with \(\mathsf F_k^\dagger\). A range or active-rank change invalidates it. Exact redundancy itself gives \(b\in\operatorname{ran}G\), so a noiseless gradient outer product cannot reveal a genuinely missing tangent direction. Noise outside the range must not be allowed to create a fictitious tangent rank.

Explicitly, under those fixed-kernel hypotheses,

\[
\mathsf F_{k+1}^{\dagger}
=\frac1{1-\epsilon_k}
\left[
\mathsf F_k^\dagger-
\frac{\epsilon_k\mathsf F_k^\dagger b_kb_k^{\mathsf T}\mathsf F_k^\dagger}
{1-\epsilon_k+\epsilon_kb_k^{\mathsf T}\mathsf F_k^\dagger b_k}
\right].
\]

The formal implementation therefore runs (5.3) only on the positive-definite reduced quotient \(G_R\). A full-coordinate ridge is permissible as a regularized preconditioner, but it no longer represents \(G^\dagger\).

The paper's noisy-initial-metric prescription shifts the spectrum when its smallest eigenvalue is negative. As written, a merely singular positive-semidefinite matrix with \(\lambda_{\min}=0\) is not lifted, so that prescription alone does not guarantee invertibility. The formal choices are either the thresholded quotient used here or an explicit strictly positive ridge, with the latter labeled a regularized metric rather than a Moore--Penrose inverse.

### 5.4 Resolution of the parallel-transport tension

Metric compatibility means that in a Levi--Civita-parallel orthonormal frame the physical metric is always \(I_\rho\). It does not mean the pullback matrix in a changing circuit-coordinate basis is constant. The factorization

\[
T_{x,\rho}=\mathcal E_xL_x,
\qquad
G_{x,\rho}=L_x^{\mathsf T}L_x
\]

separates these facts.

The chosen semantics are:

* \(\mathcal E_x\) and the objective-curvature operator are moved by the physical tangent transport;
* qBroyden predicts \(G_R\), hence the singular stretch of \(L_R\), in a fixed resolved logical-coordinate range;
* the left-orthogonal orientation of \(L_R\), which a metric cannot determine, is inherited from the transported frame until a tangent-frame observation corrects it;
* an exact metric/tangent observation overwrites the prediction and reanchors qBroyden.

If, at the old point,

\[
L_R=OH,
\qquad
O^{\mathsf T}O=I,
\qquad
H=(L_R^{\mathsf T}L_R)^{1/2},
\]

and qBroyden predicts \(G_{R,+}\), qBroyden itself has supplied only this reduced metric (or its inverse). After a frame transport with coefficient matrix \(Q_{+,k}\), construct the endpoint map in the transported-frame gauge as

\[
\boxed{
L_{R,+}^{\rm pred}
=Q_{+,k}O_k(G_{R,+}^{\rm pred})^{1/2}.
}
\]

If the endpoint frame is explicitly chosen as the transported old frame, then \(Q_{+,k}=I\) and the simpler formula follows. Until an observed endpoint tangent map supplies a correction, this is a declared transported-frame orientation prior. No qBroyden change is applied to the intrinsic metric \(I_\rho\), and no metric update is applied to the objective-curvature state. Thus the same geometry is not updated twice.

This representation also isolates a limitation: qBroyden plus same-state metrics cannot predict rotation of the physical tangent subspace. That rotation requires transport or cross-frame information.

---

## 6. Riemannian objective-Hessian recycling

### 6.1 Transported secant data

Let an accepted step be

\[
x_+=R_x(\eta),
\qquad
\mathcal T_\eta:V_x\to V_+
\]

with an equal-rank isometric transport. Put every secant object in \(V_+\):

\[
s=\mathcal T_\eta\eta,
\qquad
y=r_+-\mathcal T_\eta r_x,
\qquad
\widetilde B=\mathcal T_\eta B_x\mathcal T_\eta^{-1}.
\tag{6.1}
\]

In the endpoint orthonormal frame selected above,

\[
s=Qz_\eta,
\qquad
y=\bar r_+-Q\bar r_x,
\qquad
\widetilde B=QB_xQ^{\mathsf T}.
\]

For exact parallel transport along a geodesic,

\[
y
=\left[
\int_0^1P_{t\to1}\mathcal A_{\gamma(t)}P_{1\to t}\,dt
\right]s,
\]

so the secant equation is exact for a path-averaged endpoint-represented Hessian operator.

For a general retraction, convergence-backed Riemannian Broyden theory may require an isometric transport satisfying the locking condition

\[
\mathcal T_\xi\xi
=\beta_\xi DR_x(\xi)[\xi],
\qquad
\beta_\xi=
\frac{\|\xi\|_x}{\|DR_x(\xi)[\xi]\|_{R_x(\xi)}},
\]

and then uses

\[
y=\beta_\eta^{-1}r_+-\mathcal T_\eta r_x.
\]

The simpler (6.1) is exact for exponential-map motion with parallel transport and is the declared local approximation for endpoint Procrustes transport. This distinction prevents an experimental construction from borrowing a theorem whose transport assumptions it does not satisfy.

### 6.2 Metric-covariant inverse BFGS

At the endpoint let

\[
\kappa=g_+(s,y)>0,
\qquad
\varrho=\kappa^{-1}.
\]

For vectors \(u,v\), define

\[
(u\otimes v^\flat)w=u\,g_+(v,w).
\]

The inverse Riemannian BFGS update is

\[
\boxed{
B_+
=\left(I-\varrho s\otimes y^\flat\right)
\widetilde B
\left(I-\varrho y\otimes s^\flat\right)
+\varrho s\otimes s^\flat.
}
\tag{6.2}
\]

Its secant identity follows immediately:

\[
\left(I-\varrho y\otimes s^\flat\right)y=0,
\qquad
B_+y=\varrho s\,g(s,y)=s.
\]

Writing \(L=I-\varrho s\otimes y^\flat\), the other factor is \(L^{*_{g_+}}\), so

\[
B_+=L\widetilde BL^{*_{g_+}}+\varrho s\otimes s^\flat
\]

is self-adjoint. If \(\widetilde B\succ_{g_+}0\) and \(g(s,y)>0\), then

\[
g(v,B_+v)
=g(L^{*}v,\widetilde BL^{*}v)
+\varrho g(s,v)^2>0
\]

for every nonzero \(v\). The resulting direction is descent:

\[
dE[-Br]=-g(r,Br)<0.
\]

In a nonorthonormal active basis with endpoint metric \(G_+\), (6.2) is

\[
B_+
=\left(I-\varrho s y^{\mathsf T}G_+\right)
\widetilde B
\left(I-\varrho y s^{\mathsf T}G_+\right)
+\varrho ss^{\mathsf T}G_+.
\]

The direct-Hessian form is

\[
A_+
=\widetilde A
-\frac{\widetilde As\otimes(\widetilde As)^\flat}
{g(s,\widetilde As)}
+\frac{y\otimes y^\flat}{g(s,y)},
\qquad
A_+s=y.
\]

These equations and their transport assumptions follow the Riemannian Broyden-class construction of [Huang, Gallivan, and Absil](https://www.math.fsu.edu/~whuang2/pdf/RBroydenBasic_SIOPT.pdf).

### 6.3 Damping and cautious update

No positive-definite inverse model can satisfy \(B_+y=s\) when \(g(s,y)\leq0\), since positive definiteness would require

\[
g(y,B_+y)=g(y,s)>0.
\]

Let \(\widetilde A=\widetilde B^{-1}\),

\[
\delta=g(s,\widetilde As)>0,
\]

and choose a fixed \(c\in(0,1)\). Powell damping is

\[
\vartheta=
\begin{cases}
1,&g(s,y)\geq c\delta,\\[1mm]
\dfrac{(1-c)\delta}{\delta-g(s,y)},&g(s,y)<c\delta,
\end{cases}
\qquad
\bar y=\vartheta y+(1-\vartheta)\widetilde As.
\]

Then

\[
g(s,\bar y)=\max\{g(s,y),c\delta\}>0.
\]

Applying (6.2) with \(\bar y\) preserves positive definiteness and enforces the damped secant \(B_+\bar y=s\), not the raw secant.

A deterministic conditioning guard should additionally require

\[
\frac{g(s,\bar y)}{\|s\|\,\|\bar y\|}>\tau_{\rm curv}.
\]

Failure means only that this secant is numerically unsuitable; it is not a probability statement. A cautious alternative is to skip the update when normalized positive curvature is too small, as in [Huang, Absil, and Gallivan's nonconvex RBFGS](https://www.math.fsu.edu/~whuang2/pdf/RBFGSNonConvexFull_SIOPT.pdf).

### 6.4 Indefinite SR1 mode

For a direct self-adjoint Hessian model \(\widetilde A\), let

\[
q=y-\widetilde As.
\]

If

\[
|g(q,s)|>\tau_{\rm SR1}\|q\|\,\|s\|,
\]

the Riemannian SR1 update is

\[
\boxed{
A_+=\widetilde A+\frac{q\otimes q^\flat}{g(q,s)},
\qquad
A_+s=y.
}
\tag{6.3}
\]

For an inverse model, with \(u=s-\widetilde By\),

\[
\boxed{
B_+=\widetilde B+\frac{u\otimes u^\flat}{g(u,y)},
\qquad
B_+y=s,
}
\tag{6.4}
\]

provided \(|g(u,y)|>\tau_{\rm SR1}\|u\|\|y\|\). SR1 preserves self-adjointness but not positive definiteness. Therefore the coherent indefinite optimizer is a direct-Hessian Riemannian trust-region SR1 method,

\[
\min_{\|z\|\leq\Delta}
E(x)+\bar r^{\mathsf T}z+\frac12z^{\mathsf T}A z,
\]

solved by a negative-curvature-aware truncated method. This is the setting analyzed in the primary [Riemannian SR1 trust-region paper](https://optimization-online.org/wp-content/uploads/2013/06/3905.pdf). Independently updating direct and inverse SR1 matrices does not keep them mutually inverse; one representation must be authoritative.

### 6.5 Nonisometric transport and rank changes

For an isometry \(Q\), \(Q^*=Q^{-1}\), so \(QBQ^{-1}\) preserves self-adjointness, positivity, and spectrum. For a nonisometric invertible transport \(L\), similarity \(LBL^{-1}\) generally loses endpoint-metric self-adjointness. Congruence \(LBL^*\) preserves positivity but changes the tensor semantics from a raised endomorphism to a contravariant bilinear object. The safe choices are a polar-isometrized transport, a differentiated-retraction/adjoint quasi-Newton construction, or an explicitly approximate prior followed by reset. Raw nonisometric similarity plus ordinary BFGS is not standard Riemannian BFGS.

A smooth manifold has locally constant dimension. A resolved-rank change means either a numerical threshold crossed a closing gap or the iterate reached another stratum. There is no invertible tangent transport across it. Therefore:

\[
\rho_+=\rho
\Longrightarrow
\text{transport and update normally},
\]

whereas

\[
\rho_+\ne\rho
\Longrightarrow
\text{discard the spanning secant, flush history, rebuild the frame, and reinitialize.}
\]

A stably identified common subspace may receive a transported/compressed **prior**, but this does not preserve an exact inverse. If \(J:W\hookrightarrow V\), then \(J^*AJ\) is the exact restriction of a known direct quadratic form, whereas

\[
J^*BJ\ne(J^*AJ)^{-1}
\]

in general. The same warning governs pruning.

### 6.6 Simultaneous metric and Hessian correction

The endpoint gradients in \(y\) must use compatible metric maps. If a metric refresh changes \(L\) before the curvature update, recompute

\[
\bar r=L^{+\mathsf T}b
\]

from the stored differentials. Otherwise the quasi-Newton update learns metric-estimation error as if it were objective curvature. In the orthonormal-frame representation, a small correction of the coordinate stretch \(L\) does not itself change the intrinsic authoritative \(B\) or \(A\); a correction that rotates or changes the physical frame transports that branch by the observed frame alignment. A large metric innovation invalidates \(\mathfrak H\) because its historical secants used materially wrong gradients.

In a resolved full-rank active basis, the same bookkeeping can be expressed by preserving the contravariant model

\[
K_{\rm old}=B_{\rm old}G_{\rm old}^{-1},
\qquad
B_{\rm new}=K_{\rm old}G_{\rm new},
\]

but this merely preserves tensor type; it does not repair secants learned from inaccurate metrics. The orthonormal-frame state avoids this numerical ambiguity.

---

## 7. Unified warm-start transition

### 7.1 Zero-coordinate singleton or batch growth

Write the formal transition as

\[
\mathfrak W_k^+
=\mathcal G_{A\to A\oplus B}
(\mathfrak W_k^-;D_B,\mathcal O_B),
\]

where \(\mathcal O_B\) contains only explicitly available new observations.

1. Save the complete checkpoint \(\mathcal C=\mathfrak W_k^-\).
2. Construct the logical injection matrices \(J_A,J_B\), scatter the old parameters into the new storage order, and set every admitted coordinate to zero.
3. At the unchanged physical state, retain \(\mathcal E_A\). If new tangent columns \(D_B\), or an equivalent tangent-action handle, are available, form the physical \(R_B,S_B\), apply the declared rank rule, and build \(\mathcal E_B\). If only \(G_{AB},G_{BB}\) are available, they determine the Schur residual Gram and an **abstract** residual frame/factor, but not physical \(R_B\) or \(\mathcal E_B\). To claim the implemented growth map exact on its declared quotient, refresh the inherited frame/map first if \((\mathcal E_A,L_A)\) was only predicted or transported. It is exact for the full true tangent space only when the rank rule discards no nonzero modes. Otherwise all identities here concern the retained model, and any predicted inherited input propagates predicted provenance even though the underlying untruncated zero-growth theorem remains exact.
4. Set
   \[
   \mathcal E^+=[\mathcal E_A,\mathcal E_B],
   \qquad
   \rho^+=\rho^-+q.
   \]
5. Form the full coordinate-to-frame map directly from tangent actions,
   \[
   L^+=(\mathcal E^+)^*[T_A,D_B]P^{\mathsf T},
   \qquad
   \widehat G^+_{\rho}=(L^+)^{\mathsf T}L^+.
   \]
   Here \(\widehat G^+_{\rho}\) is the retained Gram model. It equals the true \(G^+\) only when no nonzero tangent mode was discarded.
   If only Gram blocks are available, an arbitrary factor is not admissible because it can rotate the inherited curvature axes. Preserve the inherited rows by defining
   \[
   X=L_A^{+\mathsf T}G_{AB},
   \qquad
   S_B=G_{BB}-X^{\mathsf T}X.
   \]
   Before factoring, require
   \[
   \epsilon_{AB}
   :=\frac{\|\bigl(I-L_A^{\mathsf T}L_A^{+\mathsf T}\bigr)G_{AB}\|_F}
   {\|G_{AB}\|_F+\epsilon}
   \leq\tau_{AB},
   \qquad
   S_B\succeq-\tau_{\rm PSD}I.
   \]
   Exact compatible blocks satisfy both. Failure for measured/predicted blocks triggers an inherited-frame refresh or a declared joint weighted PSD fit; silently projecting \(G_{AB}\) or clipping \(S_B\) would change the observations.
   with \(Y\in\mathbb R^{q\times m}\) built from the retained residual spectrum, so that
   \[
   Y^{\mathsf T}Y
   =S_{B,\tau}
   :=V_+\Lambda_+V_+^{\mathsf T},
   \]
   not the discarded full \(S_B\), and set
   \[
   L^+_{\rm canonical}
   =
   \begin{bmatrix}
   L_A&X\\
   0&Y
   \end{bmatrix},
   \qquad
   L^+_{\rm storage}=L^+_{\rm canonical}P^{\mathsf T}.
   \]
   This factor reproduces the declared retained Gram completion, not discarded nonzero residual modes. Only the residual rows may be orthogonally rotated without also transforming their curvature block. Their physical orientation remains an abstract frame handle until the backend supplies a frame/transport action.
6. Reanchor the qBroyden reduced metric only when enough exact-state/measured information constructs the retained \(G^+,L^+\), and rank. If any necessary block is unobserved, either observe it before reoptimization or supply a declared PSD regularized completion and label the resulting metric **predicted**. A partially unknown Gram matrix cannot define a reoptimization step or an exact anchor, and objective curvature cannot complete it.
7. Assemble \(b^+\). The old entries are exact reuse at the same state if their stored observations use the identical objective and coordinate registry. New entries require a same-state gradient observation unless already available from admission at the identical state.
8. Initialize objective curvature in the adapted orthonormal frame. In inverse-BFGS mode,
   \[
   B_0^+=\operatorname{diag}(B_A,\beta I_q),
   \]
   while in direct-SR1 mode,
   \[
   A_0^+=\operatorname{diag}(A_A,\alpha I_q),
   \]
   before assimilating declared diagonal/HVP observations as in Section 3.7. Mark the old block inherited, mixed block unknown-with-zero-prior, and new block regularized/measured as appropriate.
9. Clear all spanning secant history. No BFGS update occurs at growth because the physical displacement is zero.

If \(q=0\), the admitted coordinates add no physical tangent direction at the growth point. They may be rejected as redundant or retained only as redundant coordinates; in either case they do not enlarge the active quotient.

### 7.2 First inner step after growth

In inverse-BFGS mode compute

\[
\bar r_0=L_{\mathrm{grow},0}^{+\mathsf T}b_0^+,
\qquad
z_0=-B_0^+\bar r_0,
\qquad
p_0=L_{\mathrm{grow},0}^+z_0,
\qquad
\eta_0=\mathcal E_0z_0=T_{\rho,0}p_0,
\]

where \(L_{\mathrm{grow},0}\) is the coordinate-to-frame map at the grown initial point and the superscript \({}+\) denotes the Moore--Penrose inverse. In implementation notation, name this operation `pinv_L`.

In direct-SR1 mode, obtain and accept/reject \(z_0\) through the indefinite trust-region subproblem and its model-reduction ratio; do not substitute a line search, because the negative-curvature step need not be a descent direction. In inverse-BFGS mode, use a line search or an FS trust safeguard. Both branches use the same \(p_0=L^+z_0\), \(\eta_0=\mathcal E_0z_0\) maps. For SPD \(B_0^+\),

\[
b^{\mathsf T}p_0
=\bar r_0^{\mathsf T}z_0
=-\bar r_0^{\mathsf T}B_0^+\bar r_0<0
\]

when \(\bar r_0\neq0\); if \(\bar r_0=0\), this branch has reached its first-order stopping condition.

Globalization may scale or otherwise alter the trial. Let \(p_0^{\rm acc}\) be the coordinate displacement actually accepted. If the backend observes the old-state tangent action, define \(a_0^{\rm obs}\) by

\[
a_0^{\rm obs}=\mathcal E_0^*T_0^{\rm true}p_0^{\rm acc},
\qquad
\eta_0^{\rm acc}=\mathcal E_0a_0^{\rm obs}=T_0^{\rm true}p_0^{\rm acc}.
\]

Otherwise use the model generator \(\widehat a_0=L_0p_0^{\rm acc}\), mark \(s,y\) predicted, and correct/invalidate them on the next tangent observation. Only the observed action is an actual physical tangent.

After acceptance, obtain the endpoint frame or transport action, predict/correct the metric, observe the endpoint differential, form \(s,y\), and perform the first full-space BFGS/SR1 update. This first secant supplies one full-space Hessian-action constraint; a least-change update may populate old--new entries, but one secant does not identify the mixed block. It is inferred, not measured.

### 7.3 Later fixed-manifold steps

For each later inverse-BFGS step, let globalization return the actually accepted coordinate displacement \(p_j^{\rm acc}\), not merely the trial. Let \(a_j\) below denote the backend-observed old-frame tangent action when available, and otherwise the provenance-tagged model \(\widehat a_j=L_jp_j^{\rm acc}\). Then

\[
\begin{aligned}
&z_j=-B_jL_j^{+\mathsf T}b_j,
\qquad
p_j=L_j^+z_j,
\qquad
\eta_j=\mathcal E_jz_j=T_{j,\rho}p_j,\\
&a_j^{\rm sec}=
\begin{cases}
\mathcal E_j^*T_j^{\rm true}p_j^{\rm acc},&\text{observed tangent action},\\
L_jp_j^{\rm acc},&\text{predicted model action},
\end{cases}\\
&x_{j+1}=\text{the endpoint actually accepted by the backend},\\
&Q_j=\operatorname{polar}(\mathcal E_{j+1}^*\mathcal E_j),\\
&\widetilde B_j=Q_jB_jQ_j^{\mathsf T},\\
&s_j=Q_ja_j^{\rm sec},
\qquad
y_j=L_{j+1}^{+\mathsf T}b_{j+1}-Q_jL_j^{+\mathsf T}b_j,\\
&B_{j+1}=\operatorname{damped\_RBFGS}(\widetilde B_j,s_j,y_j),
\end{aligned}
\]

with the locking correction substituted if the backend certifies that version. The second case makes the secant predicted and subject to later correction. If the backend implements the coordinate curve \(f(\theta+p)\), it must declare this as the quotient-lifted retraction \(R_x(\mathcal Ez):=f(\theta+L^+z)\) relative to the resolved model; \(R_x\) is never applied directly to the coordinate array \(p\).

In direct-SR1 mode, transport \(\widetilde A_j=Q_jA_jQ_j^{\mathsf T}\), obtain the accepted \(a_j\) from the trust-region step, use the same \(s_j,y_j\), and apply (6.3). qBroyden predicts only \(G_{R,j+1}\) or \(M_{R,j+1}\); a predicted \(L_{j+1}\) is then constructed in the declared transported-frame gauge of Section 5.4. It supplies neither the endpoint physical frame nor the Procrustes map.

### 7.4 Pruning or coordinate removal

Let \(C:\mathbb R^{d'}\to\mathbb R^d\) inject the surviving coordinates. The same-state restriction below is valid only if the pruned state map obeys the local restriction

\[
f_{\rm pruned}(\phi)
=f_{\rm old}(\theta_0+C\phi)
\]

on a neighborhood of the current \(\phi_\star\), not merely at the single current state. Under that precondition,

\[
T'_\rho=T_\rho C=\mathcal ELC.
\]

Resolve the left range of \(LC\). If \(U_W\) is an orthonormal basis of that range, set

\[
\mathcal E'=\mathcal EU_W,
\qquad
L'=U_W^{\mathsf T}LC,
\qquad
G'_\rho=(L')^{\mathsf T}L'.
\]

The differential restricts as

\[
b'=C^{\mathsf T}b.
\]

This is the exact restriction of the **stored retained model**. It equals the full true pruned tangent/metric only if no surviving nonzero component was previously discarded or a fresh true-tangent evaluation confirms it. In inverse-BFGS mode the curvature compression

\[
B'=U_W^{\mathsf T}BU_W
\]

is only a prior for the reduced problem; it is not the inverse of the restricted objective Hessian. Clear secant history and reanchor the metric. If pruning is accompanied by a refit that moves the physical state, first define the reduced manifold and then transport along the accepted reduced-manifold trajectory. If the removed circuit cannot reproduce the checkpoint state, this is both a manifold change and motion, not simple row/column deletion.

In direct-SR1 mode use \(A'=U_W^{\mathsf T}AU_W\) as the compressed direct-form prior. Even this is merely the restriction of the old ambient-manifold quadratic form. If \(\mathcal N\subset\mathcal M\) is the pruned manifold,

\[
\operatorname{Hess}_{\mathcal N}(E|_{\mathcal N})(X,Y)
=\operatorname{Hess}_{\mathcal M}E(X,Y)
+g\!\left((\operatorname{grad}_{\mathcal M}E)^{\perp_{\mathcal N}},
\mathrm{II}_{\mathcal N\subset\mathcal M}(X,Y)\right),
\]

so the intrinsic reduced Hessian may differ. Deleting a nonzero circuit factor usually violates the same-state embedding and must enter the manifold-change-plus-refit path.

### 7.5 Rollback

Rollback restores the entire checkpoint \(\mathcal C\): physical state, stored energy, parameters, registry, frame, coordinate map, differential, qBroyden anchor/age, objective curvature, rank, trust radius, and metadata. Curvature learned on a failed enlarged manifold is not projected back by default. Such projection would be a new inference requiring validation, not rollback.

### 7.6 Rank gain or loss

At any fixed-manifold step, rank gain/loss triggers:

\[
\text{rebuild }(\mathcal E,L,G,M),
\quad
\text{discard the crossing secant},
\quad
\text{reset qBroyden},
\quad
\text{flush BFGS/SR1 history}.
\]

If the observed old and new frames have a stable common principal-angle subspace, transport/compress the authoritative \(B\) or \(A\) there as a prior and initialize the complement isotropically. If the retained spectral gap has closed or the common alignment is ill-conditioned, set \(B=\beta I\) in inverse-BFGS mode or \(A=\alpha I\) in direct-SR1 mode on the full rebuilt range.

### 7.7 Field-by-field lifecycle

| Field | Zero growth | First post-growth acceptance | Later fixed step | Prune/remove | Rollback | Rank change |
|---|---|---|---|---|---|---|
| \(x\) | unchanged exactly | retracted endpoint | retracted endpoint | unchanged or refit endpoint | checkpoint | current state, new stratum |
| \(\theta,\mathcal I\) | permute old, insert zeros | update all coordinates | update | restrict/remap | checkpoint | retain coordinates, rebuild quotient |
| \(\mathcal E\) | append residual frame | endpoint transport/observation | endpoint transport/observation | restrict left range | checkpoint | rebuild; partial salvage only |
| \(L,G,M\) | exact old block; observe/factor new blocks | qB predict then observe/correct | qB predict or observe | exact restriction if frame exact | checkpoint | invalidate and reanchor |
| \(b,\bar r\) | old differential reused; new observed | observe \(b_+\), recompute \(\bar r_+\) | observe/recompute | observe after any refit | checkpoint | project/remeasure, then recompute |
| \(\mathfrak H=\mathrm{InvBFGS}(B)\) or \(\mathrm{DirectSR1}(A)\) | mode-specific inherited-plus-isotropic prior | first full secant update | transport and update authoritative branch | mode-specific compressed prior | checkpoint | flush/reset or common-subspace prior |
| \(\Delta\) | retain or conservatively shrink | update from acceptance ratio | update | conservatively shrink | checkpoint | shrink/reset |
| \(\Sigma\) | record every block's provenance | record innovations/secant | update diagnostics | mark restriction prior | checkpoint | mark structural invalidation |

---

## 8. Correction, obsolescence, and refresh

The recycled state is a deterministic predictor unless a separate observation-noise model is supplied. Let \(\tau_{\rm soft}\) and \(\tau_{\rm hard}\) be implementation tolerances whose numerical values are declared, not interpreted as confidence levels.

### 8.1 Diagnostics

| Diagnostic | Dimensionless quantity | Supported conclusion | Action |
|---|---|---|---|
| metric quadratic innovation | for probe \(v\), \(\epsilon_G(v)=|v^{\mathsf T}(G_{\rm obs}-\widehat G)v|/(v^{\mathsf T}G_{\rm obs}v+v^{\mathsf T}\widehat Gv+\epsilon)\) | predictor is wrong in the observed direction only | constrained correction; broad metric refresh if many directions fail |
| full metric innovation | \(\|\widehat G_R^{-1/2}(G_{R,\rm obs}-\widehat G_R)\widehat G_R^{-1/2}\|_2\) | relative operator error on the common observed range | reanchor; invalidate curvature if hard |
| pre-update secant residual | inverse mode: \(\|\widetilde By-s\|/(\|s\|+\|\widetilde By\|+\epsilon)\); direct mode: \(\|y-\widetilde As\|/(\|y\|+\|\widetilde As\|+\epsilon)\) | transported curvature predictor is obsolete along this action | BFGS/SR1 correction; reset after repeated hard residuals |
| post-update secant residual | corresponding expression with the updated authoritative branch and used \(y\) | algebraic implementation test | fail closed if not near roundoff/solver tolerance |
| tangent-subspace drift | equal rank: \(\|\Pi_{\rm pred}-\Pi_{\rm obs}\|_2=\sin\vartheta_{\max}=\sqrt{1-\sigma_{\min}(C)^2}\) | predicted transport/subspace rotated materially | correct frame; hard drift invalidates \(\mathfrak H\) |
| retained spectral-gap closure | \(\gamma_G=(\lambda_\rho-\lambda_{\rho+1})/(\lambda_1+\epsilon)\) | numerical quotient rank is unstable | broad refresh, freeze update, no spanning secant |
| active-rank change | \(\rho_{\rm obs}\ne\rho_{\rm pred}\) | no full tangent-space isomorphism exists | structural invalidation/reset |
| reduction agreement | \(\varrho_E=[E(x)-E(x_+)]/[-\bar r^{\mathsf T}z-\tfrac12z^{\mathsf T}Az]\) when denominator \(>0\) | local joint metric/curvature model predicted this accepted direction poorly or well | trust-radius update; rejection for negative realized decrease |
| step-model decrease | BFGS line search: \(\bar r^{\mathsf T}z/(\|\bar r\|\|z\|)\); SR1 trust region: \(\bar r^{\mathsf T}z+\tfrac12z^{\mathsf T}Az<0\) | whether the branch supplied its required local decrease | reject and reset/fallback if its branch condition fails |
| line-search failure | normalized Armijo/Wolfe residuals and backtrack limit | local model or transport is unsuitable; source not uniquely identifiable | shrink, skip secant; refresh after repetition |

A few quadratic metric probes cannot validate unobserved matrix directions. A low secant residual validates the action only on the current secant. A good energy-reduction ratio validates only the tested step. None is a global geometric certificate.

### 8.2 Metric correction

For observed quadratic forms \(v_\ell^{\mathsf T}Gv_\ell=q_\ell\), an honest deterministic correction is the constrained nearest positive-semidefinite problem

\[
G^+
=\arg\min_{X\succeq0}
\|W^{1/2}(X-\widehat G)W^{1/2}\|_F^2
\quad\text{subject to}\quad
v_\ell^{\mathsf T}Xv_\ell=q_\ell.
\]

With a complete observed block, overwrite that block and refactor under the rank rule. With inconsistent noisy constraints, the exact feasibility problem must be replaced by a declared weighted fit; that replacement is not statistically calibrated without a noise model.

After correction, recompute \(L,M,\bar r\) from stored \(b\). If the correction only changes coordinate stretch in a fixed physical frame and is soft, retain the intrinsic authoritative \(B\) or \(A\). If it changes the resolved physical frame, align and transport that branch. If it changes rank, closes the spectral gap, or exceeds the hard innovation threshold, invalidate \(\mathfrak H\) and all secant history.

### 8.3 Refresh levels

**Correction** means enforcing newly observed directional/block data, re-symmetrizing, refactoring, and recomputing the current gradient without discarding the whole state.

**Broad refresh** means observing the full same-state active metric, rebuilding \(\mathcal E,L,G,M\), recomputing \(b\), and retaining \(\mathfrak H\) only if the frame alignment and innovation tests are soft.

**Complete invalidation** means \(B\leftarrow\beta I\) in inverse-BFGS mode or \(A\leftarrow\alpha I\) (or a newly observed direct model) in direct-SR1 mode, together with qBroyden reanchoring, secant-history flush, and a conservatively reduced trust radius. It is required for unresolved rank change, loss of transport invertibility, a nearly right-angle principal direction, failed algebraic postconditions, or repeated rejected/nondecreasing model steps.

### 8.4 Statistical boundary

The exact-state backend yields deterministic numerical validation. A hardware-compatible interface may return estimates, sample counts, and covariance metadata, but the thresholds above remain uncalibrated until an observation model supplies concentration bounds that account for adaptive reuse and multiple diagnostics. Statistically certified rank, innovation, and refresh decisions are therefore unfinished research, not a property of this warm start.

---

## 9. Measurement and information ledger

No line in this table converts a prediction into a measurement.

| Object/action | Newly required primitive | Reused information | Classical-only consequence | What it does **not** determine |
|---|---|---|---|---|
| energy \(E(x)\) | energy evaluation | last value at identical state | acceptance ratio, line search | metric or Hessian |
| coordinate differential \(b\) | energy-gradient observation, often parameter-shift energies | old components at the identical zero-growth state; candidate gradient if already observed there | frame gradient \(L^{+\mathsf T}b\) | mixed Hessian entries |
| inherited \(G_{AA}\) at zero growth | none beyond its previous observation | exact true block identity; stored estimate with unchanged provenance | copy/remap by \(J_A\) | new blocks or improved accuracy |
| \(G_{AB}\) | old--new same-state tangent overlaps/QGT entries | old tangent frame | residualize new directions | energy-Hessian coupling |
| \(G_{BB}\) | new--new same-state tangent overlaps/QGT entries | none except symmetries | residual Gram/rank | objective curvature |
| full same-state frame \(T\) | tangent-state generation or equivalent overlap circuits | prior frame for alignment | factor \(G\), build \(\mathcal E,L,\Pi\) | cross-state transport by itself |
| projector/rank/residual frame | none after Gram/tangent data | \(T_A,G_{AB},G_{BB}\) | eigendecomposition/SVD/pseudoinverse | statistical rank certainty |
| qBroyden prediction | current coordinate gradient only after its anchor | initial/refreshed metric model | equations (5.3)--(5.4) | a new QGT observation or metric secant |
| Riemannian gradient | none beyond \(b,L\) | metric model | \(\bar r=L^{+\mathsf T}b\) | objective Hessian |
| endpoint Procrustes transport | cross-state tangent-frame overlaps | endpoint frames | SVD/polar factor | Levi--Civita path transport in general |
| differentiated-retraction transport | backend transport action/derivative | accepted retraction step | polar-isometrization if full rank | exact parallel transport unless proved |
| BFGS secant | endpoint gradients plus accepted displacement and transport | prior \(B\) | rank-two update | a measured Hessian or untested actions |
| SR1 secant | same as BFGS | prior direct/inverse model | rank-one update | positive definiteness |
| new diagonal objective curvature | directional second derivative along a geodesic/second-order retraction, or equivalent Hessian primitive | none | initialize direct diagonal | mixed or off-diagonal curvature |
| HVP \(Av\) | Hessian-vector-product primitive | direct prior | symmetric least-change/SR1 constraint | unprobed Hessian actions |
| mixed active--new objective curvature | explicit mixed derivative or HVP along new basis directions | none from metric overlaps | assemble measured columns | cannot be inferred from \(G_{AB}\) or a scalar gradient |
| coordinate \(Q_{AA}\) at growth | none if previously measured/stored exactly | scalar restriction identity | remap/copy | enlarged covariant \(h_{AA}\) without connection/\(\mathrm{II}\) data |
| exact \(h^+_{AA}\) from old \(h^-_{AA}\) | normal-gradient and second-fundamental-form/connection data | old \(h^-\) | apply Section 3.5 correction | not free from the restriction identity |
| trust-region ratio | endpoint energy | predicted local model | accept/shrink/grow \(\Delta\) | which of metric or curvature caused an error |
| prune restriction | same-state surviving tangent geometry; new gradient after refit | current frame/model | restrict metric; compress curvature prior | exact reduced inverse Hessian |

For the exact-state backend, same-state and cross-state tangent overlaps may be computed directly from statevectors/density tangents. This is feasibility evidence, not a hardware cost claim. A measurement backend must expose each required overlap/HVP interface explicitly and account for its circuits and shots.

The saving made by metric recycling is reuse of the last geometric anchor plus gradient-only qBroyden prediction on skipped metric steps. The saving made by Hessian recycling is reuse of the transported quasi-Newton operator and its prior secants, reducing the number of subsequent energy/gradient line-search evaluations. This agrees with the actual mechanism in [Ramôa *et al.*](https://arxiv.org/abs/2401.05172): their method reuses a Euclidean-coordinate inverse-BFGS model and does not measure Hessians during the optimization. Neither saving licenses unseen mixed candidate--active Hessian information.

---

## 10. Typed implementation pseudocode

### 10.1 Types and invariants

```text
enum Provenance {
    EXACT_IDENTITY,
    EXACT_STATE_COMPUTED,
    MEASURED,
    TRANSPORTED_APPROX,
    QBROYDEN_PREDICTED,
    SECANT_INFERRED,
    REGULARIZED_PRIOR,
    UNKNOWN,
    INVALID
}

enum GapStatus { STABLE, UNSTABLE, NOT_OBSERVED }
enum Severity { NONE, SOFT, HARD }

struct MetricInnovationStatus {
    observed: bool
    severity: Severity
    value: optional float
}

struct RankRule {
    tau_abs: float
    tau_rel: float
    tau_gap: float
}

struct Metadata {
    provenance_by_field_and_block
    state_id, manifold_id, frame_id, coordinate_registry_id
    rank_rule: RankRule
    gram_eigenvalues: float[d]
    retained_gap: float
    qbroyd_age: int
    qbroyd_epsilon0: float
    tau_curv: float
    tau_align, tau_AB, tau_PSD: float
    tau_cotangent_soft, tau_cotangent_hard: float
    numerical_floor: float
    postcondition_tol, tau_sr1: float
    max_rejections, rejection_count: int
    active_coordinate_range_id
    metric_innovation, secant_residual, max_principal_angle
    valid_metric, valid_transport, valid_curvature: bool
    statistically_calibrated: false
}

union CurvatureState {
    InverseBFGS { B: float[r,r] }    # symmetric SPD inverse raised Hessian
    DirectSR1  { A: float[r,r] }     # symmetric, possibly indefinite direct operator
}

struct WarmState {
    x: PhysicalStateHandle
    energy: float
    theta: float[d]
    registry: LogicalCoordinateRegistry[d]
    E: TangentFrameHandle[r]          # orthonormal density-tangent frame
    L: float[r,d]                     # coordinate -> orthonormal frame
    Z: float[d,r]                     # orthonormal basis of row(L)
    M_R: float[r,r]                   # cached inverse of reduced G; SPD
    b: float[d]                       # coordinate covector, not raised gradient
    curvature: CurvatureState         # exactly one authoritative branch
    trust_radius: float
    rank: int r
    meta: Metadata
    checkpoint: optional WarmState
}

derived:
    T_hat = E * L
    G_hat = L^T * L
    M_hat = L^+ * L^{+T} = Z * M_R * Z^T
    projector = E * E^*
    grad_frame = L^{+T} * b

invariants:
    shape(E) = [ambient_or_handle, r]
    shape(L) = [r,d], rank(L)=r
    E^* E = I_r
    L Z is nonsingular
    M_R = inverse((L Z)^T(L Z))
    b lies in range(L^T) up to declared residual tolerance
    InverseBFGS branch: B=B^T and eig(B)>0
    DirectSR1 branch: A=A^T; indefiniteness permitted
    no secant spans a rank/manifold discontinuity
```

`E` need not be a dense ambient matrix on hardware. It may be a frame handle supporting inner products and transport actions. If it supports neither, curvature transport is unavailable and the authoritative curvature branch must reset after movement.

### 10.2 Exact/refreshed frame factorization

```text
function BUILD_FRAME(state, T_horizontal[d tangent columns], rank_rule)
        -> (E, L, Z, M_R, r, spectrum, gap_status):
    # Convert every column into the same gauge-invariant density-tangent space.
    T = density_tangent_columns(state, T_horizontal)
    # T^*T uses (1/2)Tr(XY), equal to the real FS inner product.
    G_raw = sym(T^* T)                              # [d,d]
    (V, lambda) = eigh_descending(G_raw)
    tau = max(rank_rule.tau_abs, rank_rule.tau_rel * lambda[0])
    keep = {i : lambda[i] > tau}
    r = len(keep)
    gap_status = evaluate_gap(lambda, keep, rank_rule)
    V_r = V[:,keep]                                 # [d,r]
    Lambda_r = diag(lambda[keep])                    # [r,r]
    G_retained = V_r * Lambda_r * V_r^T
    E = T * V_r * Lambda_r^{-1/2}                   # r orthonormal density tangents
    L = E^* T                                       # [r,d]
    Z = orthonormal_basis(row(L))                    # [d,r]
    L_R = L * Z                                     # [r,r]
    M_R = inverse(L_R^T * L_R)
    assert_close(E^*E, I_r)
    assert_close(G_retained, L^T*L)
    discarded_gram_residual = norm(G_raw-G_retained)
    record(discarded_gram_residual)
    return E,L,Z,M_R,r,lambda,gap_status
```

If a retained Gram matrix but no ambient tangent frame is returned, factor \(G_\rho=L^{\mathsf T}L\) and use an abstract frame handle. Mark cross-state Procrustes unavailable until a backend supplies a relative-frame or transport action.

### 10.3 qBroyden prediction on a fixed quotient

```text
function QBROYDEN_PREDICT(W, epsilon0) -> predicted_reduced_metric:
    require W.meta.valid_metric
    require fixed W.rank and fixed column space W.Z
    eps = epsilon0 / (W.meta.qbroyd_age + 1)
    require 0 <= eps < 1

    b_R = W.Z^T * W.b                              # [r]
    range_residual = norm(W.b - W.Z*b_R)/(norm(W.b) + W.meta.numerical_floor)
    if range_residual > W.meta.tau_cotangent_hard:
        return trigger_metric_refresh_or_rank_invalidation()  # terminating path
    else if range_residual > W.meta.tau_cotangent_soft:
        record_projected_cotangent_regularization()
    # Exact translation of published QFIM update into G^{-1} convention.
    denom = 1 - eps + (eps/4) * b_R^T * W.M_R * b_R
    require denom > W.meta.numerical_floor
    M_new = (1/(1-eps)) * (
        W.M_R
        - (eps/4) * W.M_R*b_R*b_R^T*W.M_R / denom
    )
    M_new = sym(M_new)
    require eig(M_new) > 0

    G_R_new = inverse(M_new)
    L_R_old = W.L * W.Z
    O = L_R_old * inverse_spd_sqrt(L_R_old^T*L_R_old)  # left polar factor
    return {
        G_R=G_R_new, M_R=M_new, O_old=O, Z=W.Z,
        qbroyd_age=W.meta.qbroyd_age+1,
        provenance=QBROYDEN_PREDICTED
    }
```

This update assumes the active coordinate range remains fixed. It returns no endpoint frame and no endpoint \(L\). After the physical transport matrix \(Q\) is known, construct

```text
L_pred_endpoint = Q * O * spd_sqrt(G_R_new) * Z^T
```

unless the endpoint frame is gauge-fixed as the transported old frame, in which case `Q=I`. It must not run across growth, pruning, or rank change. A metric observation replaces the prediction, resets qBroyden age, and records innovation before overwrite.

Initialization must also respect units: the paper's \(F_0=I\) prior corresponds to \(\widehat G_0=I/4\) and \(M_{G,0}=4I\). Choosing \(M_{G,0}=I\) is allowed but is a different \(G\)-convention prior. Full or block QFIM anchors are divided by four before entering \(G\) units.

### 10.4 Singleton/batch growth

```text
function GROW(W_minus, admitted_ids[m], growth_geometry, curvature_obs=None) -> W_plus:
    checkpoint = deep_copy(W_minus)
    (J_A, J_B, P, registry_plus) = build_logical_injections(
        W_minus.registry, admitted_ids
    )
    theta_plus = scatter_old_and_insert_zeros(W_minus.theta, J_A, J_B)
    x_plus = W_minus.x                                # exact physical-state identity

    # Fail closed unless every Gram block needed on the retained quotient is
    # observed/computed or supplied by a declared PSD regularized completion.
    require growth_geometry.metric_complete_for_reoptimization

    # Refresh W_minus.E,L first if the implemented transition is to be exact.
    # Otherwise propagate their predicted provenance to every result below.
    if growth_geometry.has_tangent_columns:
        T_A = W_minus.E * W_minus.L                  # exact only if fields are exact
        D_B = density_tangent_columns(
            x_plus, growth_geometry.D_B_horizontal
        )                                             # [ambient,m], same representation as T_A
        Pi_A = W_minus.E * W_minus.E^*
        R_B = (I - Pi_A) * D_B
        S_B = sym(R_B^* * R_B)                       # [m,m]
        (Vq, lambdaq) = retained_eigensystem(S_B, W_minus.meta.rank_rule)
        q = len(lambdaq)
        E_B = R_B * Vq * diag(lambdaq^{-1/2})        # [ambient,q]
        E_plus = concatenate_columns(W_minus.E, E_B) # [ambient,r+q]
        T_plus_canonical = concatenate_columns(T_A, D_B)
        L_plus = E_plus^* * T_plus_canonical * P^T  # [r+q,d+m]
    else:
        G_AB = growth_geometry.G_AB                  # [d,m]
        G_BB = growth_geometry.G_BB                  # [m,m]
        P_cross = W_minus.L^T * pinv(W_minus.L)^T
        cross_residual = norm((I-P_cross)*G_AB)/(norm(G_AB)+W_minus.meta.numerical_floor)
        if cross_residual > W_minus.meta.tau_AB:
            return refresh_old_geometry_or_joint_weighted_PSD_fit_and_restart_growth()
        X = pinv(W_minus.L)^T * G_AB                 # [r,m]
        S_B = sym(G_BB - X^T*X)
        require S_B >= -W_minus.meta.tau_PSD*I or invoke_declared_noise_PSD_fit()
        (Vq, lambdaq) = retained_eigensystem_or_declared_psd_completion(
            S_B, W_minus.meta.rank_rule
        )
        q = len(lambdaq)
        Y = diag(sqrt(lambdaq)) * Vq^T               # [q,m], Y^T Y=S_B(retained)
        L_canonical = block_rows_and_columns(
            top=[W_minus.L, X],
            bottom=[zeros(q,d), Y]
        )                                             # [r+q,d+m]
        L_plus = L_canonical * P^T
        E_B = abstract_orthonormal_residual_frame_handle(q)
        E_plus = concatenate_frames(W_minus.E, E_B)
        require backend_transport_action_for_future_motion or mark curvature_nontransportable

    declared_rank_plus = W_minus.rank + q
    (Z_plus, M_R_plus, rank_plus) = reduced_cache_from_declared_rows(
        L_plus, declared_rank_plus
    )
    compare_with_independent_full_gram_threshold_and_mark_instability_if_disagree()
    b_plus = scatter_old_differential(W_minus.b, J_A)
    b_plus[J_B] = growth_geometry.new_gradient_or_observe_now()

    if W_minus.curvature is InverseBFGS(B_minus):
        beta = clipped_isotropic_inverse_curvature_scale(B_minus)
        curvature_plus = InverseBFGS(
            block_diag(B_minus, beta*I_q)             # PRIOR, not inverse identity
        )
    else if W_minus.curvature is DirectSR1(A_minus):
        alpha = clipped_direct_curvature_scale(A_minus)
        curvature_plus = DirectSR1(
            block_diag(A_minus, alpha*I_q)            # PRIOR; may use signed observed block
        )
    if curvature_obs is not None:
        curvature_plus = assimilate_declared_diagonal_or_HVP_constraints(
            curvature_plus, curvature_obs, W_minus.curvature.tag
        )

    meta = provenance_ledger_for_growth(...)
    meta.qbroyd_age = 0
    clear_all_secant_history(meta)
    return WarmState(x_plus, energy=W_minus.energy, theta_plus, registry_plus,
                     E_plus, L_plus, Z_plus, M_R_plus, b_plus,
                     curvature_plus, ..., checkpoint)
```

If `W_minus.L` was only predicted, `T_A = E*L` is predicted too. The mathematical old-block identity remains true of the actual tangent map, but implementation provenance must remain predicted until refreshed.

### 10.5 One accepted inner step

```text
function INNER_STEP(W, backend) -> W_next_or_rejected:
    grad = pinv(W.L)^T * W.b                         # [r]

    if W.curvature is InverseBFGS(B_old):
        A_model_old = inverse(B_old)
        z_trial = -B_old * grad                      # [r]
        z_trial = radial_cap(z_trial, W.trust_radius)
            # radial cap is an approximate FS trust safeguard, not exact TR solve
        require grad^T*z_trial < 0
    else if W.curvature is DirectSR1(A_old):
        A_model_old = A_old
        z_trial = solve_indefinite_trust_region(A_old, grad, W.trust_radius)
        require grad^T*z_trial + 0.5*z_trial^T*A_old*z_trial < 0

    p_trial = pinv(W.L) * z_trial                    # [d], declared min-norm lift
    eta_trial = W.E * z_trial                        # physical tangent in V_x
    proposal = backend.propose_retraction(
        W.x, W.theta, eta_trial, p_trial
    )
    result = backend.globalize(proposal)
    (accepted, endpoint, energy_new, b_new,
     theta_new, p_acc, a_acc_observed_optional,
     a_acc_provenance_optional, line_data) = result
    if not accepted:
        shrink(W.trust_radius)
        mark_model_failure(W.meta)
        W.meta.rejection_count += 1
        if W.meta.rejection_count >= W.meta.max_rejections:
            broad_refresh_or_invalidate(W)
        return rejected W

    if a_acc_observed_optional exists:
        a_sec = a_acc_observed_optional
        require a_acc_provenance_optional in {MEASURED, EXACT_STATE_COMPUTED}
        secant_displacement_provenance = a_acc_provenance_optional
    else:
        a_sec = W.L * p_acc
        secant_displacement_provenance = provenance(W.L)  # possibly predicted

    # Metric prediction uses b at the old point, as in published qBroyden.
    qpred = QBROYDEN_PREDICT(W, W.meta.qbroyd_epsilon0)
    (G_R_pred, M_R_pred, O_old, Z_pred, q_age_pred) =
        (qpred.G_R, qpred.M_R, qpred.O_old, qpred.Z, qpred.qbroyd_age)

    endpoint_geom = backend.endpoint_geometry(endpoint)
    metric_innovation = MetricInnovationStatus(observed=false, severity=NONE)
    if endpoint_geom.metric_observed:
        (E_new, L_new, Z_new, M_R_new, rank_new,
         spectrum_new, gap_status_new) = BUILD_FRAME(
            endpoint, endpoint_geom.T_new, W.meta.rank_rule
        )
    else:
        require W.meta.valid_metric
        require endpoint_geom.transport_available
        require endpoint_geom.active_coordinate_range_id == W.meta.active_coordinate_range_id
        E_new = endpoint_geom.transported_or_observed_frame_handle
        Z_new, M_R_new, rank_new = W.Z, M_R_pred, W.rank
        spectrum_new, gap_status_new = None, NOT_OBSERVED

    if rank_new != W.rank or gap_status_new is UNSTABLE:
        return RANK_CHANGE_REBUILD(
            W, endpoint, energy_new, theta_new, b_new, endpoint_geom
        )  # accepted endpoint retained; no crossing secant

    (Q, sigma_min_cross) = backend.transport_matrix(W.E, E_new, endpoint, a_sec)
        # exact-state default: polar(E_new^* E_old)
    require close(Q^T*Q, I)
    require sigma_min_cross > W.meta.tau_align

    if endpoint_geom.metric_observed:
        G_pred_full = W.Z * G_R_pred * W.Z^T
        metric_innovation = compare_on_common_range(G_pred_full, L_new^T*L_new)
    else:
        L_new = Q * O_old * spd_sqrt(G_R_pred) * W.Z^T

    if W.curvature is InverseBFGS(B_old):
        B_tilde = Q * B_old * Q^T
    else:
        A_tilde = Q * A_old * Q^T

    grad_old = pinv(W.L)^T * W.b
    grad_new = pinv(L_new)^T * b_new
    s = Q * a_sec
    y = grad_new - Q * grad_old
        # substitute locking-scaled endpoint gradient if backend certifies it

    if metric_innovation.observed and metric_innovation.severity == HARD:
        curvature_new = reset_isotropic_prior_like(W.curvature, rank=W.rank)
        skip secant
    else if W.curvature is InverseBFGS:
        A_tilde_s = solve(B_tilde, s)
        delta = s^T * A_tilde_s
        ybar = powell_damp(y, s, A_tilde_s, delta)
        if fails_curvature_guard(s, ybar, W.meta.tau_curv):
            B_new = B_tilde
        else:
            rho = 1/(s^T*ybar)
            B_new = (I-rho*s*ybar^T)*B_tilde*(I-rho*ybar*s^T) + rho*s*s^T
            B_new = sym(B_new)
            require eig(B_new)>0
            require normalized_norm(B_new*ybar-s) <= W.meta.postcondition_tol
        curvature_new = InverseBFGS(B_new)
    else:
        q = y - A_tilde*s
        if abs(q^T*s) <= W.meta.tau_sr1*norm(q)*norm(s):
            A_new = A_tilde
        else:
            A_new = A_tilde + q*q^T/(q^T*s)
            require normalized_norm(A_new*s-y) <= W.meta.postcondition_tol
        curvature_new = DirectSR1(A_new)

    predicted_drop = -(grad^T*a_sec + 0.5*a_sec^T*A_model_old*a_sec)
    actual_drop = W.energy - energy_new
    ratio = actual_drop/predicted_drop if predicted_drop>0 else invalid
    trust_radius_new = trust_region_policy(W.trust_radius, ratio, line_data)

    meta_new = update_metadata(W.meta, spectrum_new, gap_status_new,
                               secant_displacement_provenance, metric_innovation)
    meta_new.qbroyd_age = 0 if endpoint_geom.metric_observed else q_age_pred
    meta_new.rejection_count = 0
    candidate = WarmState(x=endpoint, energy=energy_new, theta=theta_new,
                          registry=W.registry,
                          E=E_new, L=L_new, Z=Z_new, M_R=M_R_new,
                          b=b_new, curvature=curvature_new,
                          trust_radius=trust_radius_new, rank=W.rank,
                          meta=meta_new, checkpoint=W.checkpoint, ...)
    diagnostics = run_metric_secant_angle_gap_diagnostics(candidate, ...)
    return apply_refresh_policy_if_needed(candidate, diagnostics)
```

### 10.6 Prune, rollback, and rank change

```text
function PRUNE_SAME_STATE(W, survivor_injection C[d,d2]) -> W2:
    require f_pruned(phi)=f_old(theta0+C*phi) on a neighborhood of current phi
    L_survive = W.L * C                               # [r,d2]
    U_W = orthonormal_basis(range(L_survive))         # [r,r2]
    E2 = W.E * U_W
    L2 = U_W^T * L_survive
    b2 = C^T * W.b
    if W.curvature is InverseBFGS(B):
        curvature2 = InverseBFGS(U_W^T*B*U_W)         # PRIOR only
    else if W.curvature is DirectSR1(A):
        curvature2 = DirectSR1(U_W^T*A*U_W)           # direct restriction PRIOR
    rebuild Z2,M_R2; restrict theta,registry; use b2
    reset qBroyden age; clear secants; shrink trust radius
    return W2

function ROLLBACK(W_grown_or_failed) -> WarmState:
    require W_grown_or_failed.checkpoint exists
    return deep_copy(W_grown_or_failed.checkpoint)

function RANK_CHANGE_REBUILD(
        W, accepted_endpoint, energy2, theta2, b2, observed_geometry
    ) -> WarmState:
    (E2,L2,Z2,M_R2,r2,spectrum2,gap2) = BUILD_FRAME(
        accepted_endpoint, observed_geometry.T_new, W.meta.rank_rule
    )
    common = stable_common_subspace(W.E, E2, W.meta.tau_align)
    if W.curvature is InverseBFGS(B):
        beta = clipped_isotropic_inverse_curvature_scale(B)
        if common is stable:
            (U_c, B_common) = transport_and_compress_into_endpoint_frame(B, common)
                # U_c[r2,r_common], U_c^T U_c=I
            P_c = U_c*U_c^T
            B2 = U_c*B_common*U_c^T + beta*(I_r2-P_c)
            curvature2 = InverseBFGS(B2)
        else:
            curvature2 = InverseBFGS(beta*I_r2)
    else if W.curvature is DirectSR1(A):
        alpha = clipped_direct_curvature_scale(A)
        if common is stable:
            (U_c, A_common) = transport_and_compress_into_endpoint_frame(A, common)
            P_c = U_c*U_c^T
            A2 = U_c*A_common*U_c^T + alpha*(I_r2-P_c)
            curvature2 = DirectSR1(A2)
        else:
            curvature2 = DirectSR1(alpha*I_r2)
    clear secants; reset qBroyden; shrink trust radius
    meta2 = structural_rank_change_metadata(spectrum2, gap2, qbroyd_age=0)
    return WarmState(x=accepted_endpoint, energy=energy2, theta=theta2,
                     E=E2,L=L2,Z=Z2,M_R=M_R2,b=b2,
                     curvature=curvature2,rank=r2,meta=meta2,...)
```

---

## 11. Verification identities and falsification tests

Every test below can falsify an incorrect implementation.

1. **Gauge test.** Replace \(\psi(\theta)\) by \(e^{i\chi(\theta)}\psi(\theta)\). Raw derivatives must change, horizontal tangents must acquire only the common phase, density tangents and \(G\) must be invariant.

2. **Arbitrary insertion test.** Insert \(V(\alpha)\) at several circuit positions with \(V(0)=I\). At \(\alpha=0\), phase-aligned projective tangents, density tangents, or their Gram matrices and pure old first/second energy derivatives must match the old circuit after logical remapping. Independently gauged raw statevector finite differences need not match. New tangent columns should generally depend on insertion position.

3. **Old-gate reorder test.** Reorder two noncommuting old factors. The restriction identity should fail; the code must not classify this as a coordinate permutation.

4. **Permutation test.** For a random permutation \(P\), verify
   \[
   G_{\rm storage}=PG_{\rm canonical}P^{\mathsf T},
   \quad
   b_{\rm storage}=Pb_{\rm canonical},
   \quad
   Q_{\rm storage}=PQ_{\rm canonical}P^{\mathsf T}.
   \]

5. **Tied-parameter test.** Let one logical angle drive two gate occurrences. Its tangent must equal the sum of occurrence tangents. On untying, verify \(T_\beta L=T_{\rm tied}\) and \(G_{\rm tied}=L^{\mathsf T}G_\beta L\), while a chosen principal block generally fails to equal \(G_{\rm tied}\).

6. **Redundant-coordinate fixture.** Use \(T=[t,t]\). Then
   \[
   G=\|t\|^2\begin{bmatrix}1&1\\1&1\end{bmatrix},
   \quad
   \operatorname{rank}G=1,
   \quad
   \Pi=\frac{|t\rangle\langle t|_{\mathbb R}}{\|t\|^2}.
   \]
   A valid exact differential must be proportional to \((1,1)\).

7. **Batch projector test.** For correlated \(D_B\), verify symmetry/idempotence of \(\Pi_A+R_BS_B^\dagger R_B^*\), orthogonality of its residual term to \(\Pi_A\), the exact algebraic rank-addition identity, and agreement with the projector formed from a direct SVD of \([T_A,D_B]\). Separately test the declared thresholded residual-rank rule; do not require thresholded eigencount additivity for the full coordinate Gram matrix.

8. **Sequential-order test.** In exact/high-precision arithmetic, all singleton orders with full reorthogonalization must yield the same final projector. Near the threshold, demonstrate and record possible order dependence; the block routine should be the reference.

9. **Covariant-Hessian counterexample.** Use the parabola \((\theta,\theta^2)\subset\mathbb R^2\) and \(\widetilde E(x,y)=y\). The restricted Hessian at zero is \(2\), the ambient Hessian is \(0\), and the second-fundamental correction must account for the difference.

10. **Block inverse test.** For
    \[
    H=\begin{bmatrix}2&1\\1&2\end{bmatrix},
    \qquad
    H^{-1}=\frac13\begin{bmatrix}2&-1\\-1&2\end{bmatrix},
    \]
    verify \((H^{-1})_{11}=2/3\ne1/2=(H_{11})^{-1}\). For the all-ones singular matrix, verify the pseudoinverse upper-left entry \(1/4\ne1\).

11. **Metric-versus-curvature residualization test.** Construct orthogonal tangent directions with a nonzero mixed quadratic objective \(h(e_A,e_B)\). The code must retain a zero **prior** while metadata says the true mixed curvature is unknown.

12. **Procrustes test.** Verify \(Q^{\mathsf T}Q=I\), invariance of the density-tangent physical map under \(\mathcal E_j\mapsto\mathcal E_jO_j\), invariance under independent endpoint state phases, and singular values equal to principal-angle cosines. Force \(C\) singular and require full-curvature transport to fail/reset. On a curved manifold, compare two paths with common endpoints: endpoint Procrustes is the same while Levi--Civita holonomy can differ, falsifying any claim that the two are generally identical.

13. **qBroyden direct/inverse equivalence.** For random SPD \(F\), \(b\), and \(0<\epsilon<1\), update \(F\) by (5.1), invert directly, and compare with (5.2). Repeat in the \(G\) convention and verify the \(1/4\) factors in (5.3)--(5.4).

14. **qBroyden nongeometry tests.** At \(b=0\), verify \(F_+=(1-\epsilon)F\). Rescale the Hamiltonian by \(c\) and verify the rank-one innovation scales by \(c^2\) while an independently computed QGT does not.

15. **RBFGS identities.** For random SPD \(\widetilde B\) and \(s^{\mathsf T}y>0\), verify \(B_+y=s\), \(B_+=B_+^{\mathsf T}\), and positive eigenvalues. In a nonorthonormal coordinate fixture, verify \(G B_+=B_+^{\mathsf T}G\).

16. **Damping test.** Generate \(s^{\mathsf T}y\leq0\). Verify the raw BFGS update is rejected, Powell damping produces \(s^{\mathsf T}\bar y=c\,s^{\mathsf T}\widetilde As>0\), and the damped rather than raw secant is satisfied.

17. **SR1 test.** Verify the direct or inverse secant when the denominator guard passes and verify no update when it fails. Confirm that negative eigenvalues are allowed only under trust-region globalization.

18. **Contraction/type test.** On a nonsingular resolved active chart, under \(\theta=S\phi\), verify that \(-B_\phi M_\phi b_\phi=S^{-1}(-B_\theta M_\theta b_\theta)\), and that applying an extra \(M\) breaks this covariance in general. In a redundant full coordinate system, verify instead that the physical frame step is invariant under the declared quotient isomorphism and demonstrate that minimum-norm coordinate representatives can change under a nonorthogonal reparameterization.

19. **Metric-refresh consistency test.** Change \(L\) at a fixed state, recompute \(\bar r\) from stored \(b\), and verify that forming \(y\) with the stale raised gradient triggers the consistency guard.

20. **Rank-transition test.** Move an eigenvalue across \(\tau_G\). Verify qBroyden, transport secant, and BFGS/SR1 history are invalidated; no rectangular map is silently inverted.

21. **Rollback test.** Grow, take several enlarged-manifold steps, force admission failure, and bitwise/numerically compare every restored checkpoint field. No learned enlarged-manifold curvature may survive unless a separate validated salvage routine is explicitly invoked.

22. **Information-ledger test.** Run growth using only \(G_{AB},G_{BB}\) and scalar gradients. Assert that every mixed objective-Hessian block remains `UNKNOWN` or `REGULARIZED_PRIOR`, never `MEASURED`.

23. **Accepted-displacement test.** Let a line search accept \(\alpha p_{\rm trial}\) with \(\alpha\neq1\), and let a trust-region solver modify its trial. With an observed tangent action verify \(s=Q\mathcal E^*T^{\rm true}p^{\rm acc}\); without one verify \(s=QLp^{\rm acc}\) is tagged predicted. The predicted reduction must use the same accepted secant generator, and neither calculation may use the discarded trial \(z\).

24. **Structured Gram-factor test.** In the Gram-only growth path, verify that
    \[
    L^+_{\rm canonical}=
    \begin{bmatrix}L_A&L_A^{+\mathsf T}G_{AB}\\0&Y\end{bmatrix},
    \qquad Y^{\mathsf T}Y=S_{B,\tau},
    \]
    reproduces the declared retained Gram model, with \(Y^{\mathsf T}Y=S_{B,\tau}\), while preserving the inherited \(L_A/B_A\) or \(L_A/A_A\) axes. It must also record the discarded \(S_B-S_{B,\tau}\). An arbitrary full Gram factor followed by an unchanged inherited curvature block must fail the test.

25. **Pruning-path tests.** Deleting a zero-coordinate factor with a valid local state-map restriction \(f_{\rm pruned}(\phi)=f_{\rm old}(\theta_0+C\phi)\) on a neighborhood must satisfy \(T'=TC\) and \(b'=C^{\mathsf T}b\). Mere equality at one state is insufficient. Deleting a nonzero factor must enter the motion/refit path. For a generic SPD \(A\) and subspace basis \(U\), verify
    \[
    U^{\mathsf T}A^{-1}U\neq(U^{\mathsf T}AU)^{-1}.
    \]

26. **qBroyden frame-gauge test.** With a nontrivial endpoint transport \(Q\), verify that the predicted map is \(L_{R,+}=QO_kG_{R,+}^{1/2}\). Omitting \(Q\) must make the endpoint gradient inconsistent with the transported curvature frame.

---

## 12. Remaining non-equivalences, proposal audit, and final recommendation

### 12.1 Non-equivalences that must remain visible

1. \(Q=\partial^2E\), \(h=\nabla^2E\), and \(\mathcal A=\sharp h\) coincide only in special coordinates/at special points. Their inverse models are not interchangeable.
2. Exact reuse of \(G_{AA}\) at zero growth is an identity. qBroyden evolution away from the anchor is a prediction.
3. Exact reuse of \(Q_{AA}\) does not imply exact reuse of \(h_{AA}\); the normal-gradient/second-fundamental-form term remains.
4. Exact inheritance of a direct old block does not imply inheritance of an inverse old block; Schur couplings alter it.
5. Metric-orthogonal residualization does not imply objective-Hessian decoupling.
6. A gradient-difference secant is inferred objective-curvature action, not a Hessian measurement.
7. A QGT overlap is geometric and does not supply mixed energy curvature.
8. Endpoint Procrustes transport is an isometric selected map, not generally Levi--Civita parallel transport.
9. Same-state metrics do not determine cross-state frame orientation.
10. A changing coordinate pullback metric is compatible with a constant identity metric in a transported orthonormal physical frame.
11. qBroyden and Riemannian BFGS are not algebraically redundant: the former approximates \(\sharp_g\), the latter \(\mathcal A^{-1}\). They are empirically coupled because published qBroyden uses objective gradients.
12. Tangent rank and objective-curvature rank are distinct. A nonsingular tangent metric can coexist with a singular/indefinite objective Hessian.
13. A principal submatrix of an SPD inverse remains SPD after pruning, but it is not generally the inverse of the reduced direct Hessian.
14. Manifold Riemann curvature and objective Hessian curvature are different tensors. Neither reconstructs the other.

### 12.2 Audit of the supplied combined proposal

The proposed persistent Riemann tensor \(\widehat{\mathcal R}\) is unnecessary for the optimizer warm start. Parallel/vector transport needs a connection or a transport action; Riemann curvature quantifies noncommutation/path dependence of covariant derivatives and appears in geodesic-deviation/Jacobi equations. It is not needed to move one quasi-Newton state along one accepted local step.

The block-diagonal inverse-Hessian growth rule survives only as a regularized prior. Its old block is not generally the upper-left block of the enlarged inverse, its mixed zero block is unmeasured, and its new diagonal is a scale choice. Ramôa *et al.* use precisely such a Euclidean-coordinate SPD initialization and then let later BFGS secants learn correlations; their paper does not supply a covariant or inverse-block identity.

The quantities

\[
\nabla\operatorname{Hess}_gE
\]

and

\[
R(J,\dot\gamma)\dot\gamma
\]

are new information. The first is a third covariant derivative of the objective and requires directional variation of Hessian/HVP data plus connection variation. The second requires Riemann-curvature action along a path, hence connection derivatives or equivalent geometric observations. Neither follows from one same-state metric, one objective Hessian model, or recycled scalar gradients. Applying them to possible future candidates is a candidate-specific lookahead selector, not state needed for present reoptimization. It is therefore outside the core optimizer.

qBroyden plus Riemannian BFGS survives scrutiny only with fixed types:

\[
b\in V^*,
\qquad
M\approx\sharp_g:V^*\to V,
\qquad
B\approx\mathcal A^{-1}:V\to V,
\qquad
p=-BMb.
\]

In the recommended orthonormal frame this becomes \(p=L^+[-B L^{+\mathsf T}b]\). qBroyden never updates \(B\); BFGS never updates \(G\). A large correction of one invalidates historical claims made using the other.

The persistent state that survives is therefore

\[
(x,e,\theta,\mathcal I,\mathcal E,L,b,\mathfrak H,\Delta,\rho,\Sigma,\mathcal C),
\]

not a state containing a Riemann tensor, a candidate-utility model, or Adam momentum.

### 12.3 Single implementation recommendation

Implement one experimental feature with the following inseparable choices:

**State representation.** Use a thresholded constant-rank quotient with a gauge-invariant orthonormal density-tangent frame \(\mathcal E\) and coordinate-to-frame map \(L\), so the retained pullback is \(G_\rho=L^{\mathsf T}L\); record any discarded nonzero Gram residual separately. Store the coordinate differential \(b\) and a tagged curvature branch \(\mathfrak H\): normally an SPD inverse raised-objective-Hessian matrix \(B\), or a direct symmetric \(A\) in indefinite SR1 mode. Derive \(M\), \(\Pi\), and the Riemannian gradient rather than storing inconsistent copies.

**Transport.** On the exact-state backend use density-tangent endpoint polar/Procrustes transport. Label it an isometric first-order approximation to Levi--Civita transport. On a measurement backend require either cross-tangent overlaps or a quotient-consistent transport action; otherwise reset objective curvature after motion.

**Metric update.** At initialization and every innovation-triggered refresh, observe/compute the active metric and rebuild \((\mathcal E,L)\). Between anchors, run the published qBroyden inverse recurrence in the resolved quotient with the QFIM-to-\(G\) factor of four, preserving the unobserved polar orientation of \(L\). Treat it as a predicted objective-dependent preconditioner, not as measured QGT evolution. Do not include qBang momentum.

**Objective-Hessian update.** Transport \(B\) by \(QBQ^{\mathsf T}\); form \(s\) and \(y\) in the endpoint frame; use Powell-damped inverse Riemannian BFGS when positive curvature is reliable. If the experiment is intended to retain genuine indefiniteness, switch the authoritative object to a direct Riemannian SR1 trust-region model rather than forcing negative curvature into SPD BFGS.

**Growth map.** At zero-coordinate singleton or batch insertion, reuse only the exact old Gram/coordinate-derivative identities, compute or measure old--new and new--new tangent overlaps, residualize and rank-resolve the new frame, initialize \(B^+=\operatorname{diag}(B,\beta I)\) as a provenance-tagged prior, clear secants, and let the first accepted full-space step infer mixed objective curvature. Arbitrary logical insertion is handled by explicit injection/permutation matrices. Pruning compresses only a prior; rollback restores the complete checkpoint.

**Refresh rule.** Reanchor the metric on directional/full innovation; skip or reset curvature when endpoint metrics are incompatible; completely invalidate qBroyden and quasi-Newton history on rank change, spectral-gap closure, ill-conditioned principal-angle alignment, or repeated failed reductions. Retain no claim of statistical certification until a hardware observation-noise model supplies valid adaptive bounds.

This is the smallest construction that simultaneously preserves exact zero-growth geometry, makes qBroyden and Hessian recycling type-compatible, remains honest about unmeasured mixed curvature, and can be run first on an exact-state backend without hard-coding statevector-only assumptions into the hardware interface.

### Primary sources used

* D. Fitzek, R. S. Jonsson, W. Dobrautz, and C. Schäfer, [*Optimizing Variational Quantum Algorithms with qBang: Efficiently Interweaving Metric and Momentum to Navigate Flat Energy Landscapes*](https://quantum-journal.org/papers/q-2024-04-09-1313/), *Quantum* **8**, 1313 (2024); [arXiv PDF](https://arxiv.org/pdf/2304.13882).
* M. Ramôa, L. P. Santos, N. J. Mayhall, E. Barnes, and S. E. Economou, [*Reducing measurement costs by recycling the Hessian in adaptive variational quantum algorithms*](https://doi.org/10.1088/2058-9565/ad904e), *Quantum Science and Technology* **10**, 015031 (2025); [arXiv](https://arxiv.org/abs/2401.05172).
* W. Huang, K. A. Gallivan, and P.-A. Absil, [*A Broyden Class of Quasi-Newton Methods for Riemannian Optimization*](https://doi.org/10.1137/140955483), *SIAM Journal on Optimization* **25**, 1660--1685 (2015); [author PDF](https://www.math.fsu.edu/~whuang2/pdf/RBroydenBasic_SIOPT.pdf).
* W. Huang, P.-A. Absil, and K. A. Gallivan, [*A Riemannian BFGS Method Without Differentiated Retraction for Nonconvex Optimization Problems*](https://doi.org/10.1137/17M1127582), *SIAM Journal on Optimization* **28**, 470--495 (2018); [author PDF](https://www.math.fsu.edu/~whuang2/pdf/RBFGSNonConvexFull_SIOPT.pdf).
* W. Huang, P.-A. Absil, and K. A. Gallivan, [*A Riemannian Symmetric Rank-One Trust-Region Method*](https://doi.org/10.1007/s10107-014-0765-1), *Mathematical Programming* **150**, 179--216 (2015); [primary manuscript](https://optimization-online.org/wp-content/uploads/2013/06/3905.pdf).
