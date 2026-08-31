# McLachlan Append Linear Algebra Primer

This note is a reading aid for the manuscript's checkpoint-adaptive
McLachlan append section. It focuses on what the matrices mean and why the
append test is written as a supported, whitened Schur-complement calculation.
It is not a general linear algebra review.

## The fixed-manifold solve

Qualitatively, McLachlan dynamics asks for the velocity inside the current
ansatz tangent space that best matches the exact Schrodinger velocity. The
current circuit state is already fixed at checkpoint `k`; only the parameter
velocity is chosen. The columns of the tangent matrix are the directions the
current scaffold can move immediately.

The manuscript uses horizontal tangents, so the physically irrelevant phase
direction has already been projected away:

$$
\bar T_k =
\begin{bmatrix}
\bar\tau_{k,1} & \cdots & \bar\tau_{k,N_k}
\end{bmatrix},
\qquad
\bar b_k = -i(H_k^\sharp-\langle H_k^\sharp\rangle_{\psi_k})|\psi_k\rangle .
$$

The basic least-squares problem is: choose a real vector `v` so that
`\bar T_k v` is close to `\bar b_k`. Regularization adds a penalty for unstable
or expensive coordinate motion:

$$
\mathfrak m_k(v)=\|\bar T_k v-\bar b_k\|_2^2+v^\top\Lambda_k v .
$$

The normal-equation objects are

$$
G_k=\operatorname{Re}(\bar T_k^\dagger\bar T_k),
\qquad
f_k=\operatorname{Re}(\bar T_k^\dagger\bar b_k),
\qquad
K_k=G_k+\Lambda_k .
$$

`G_k` is the tangent metric. It tells which parameter directions are large,
small, or nearly the same direction in state space. `f_k` is the force: it
measures how much the exact Hamiltonian velocity points along each tangent
direction. `K_k` is the matrix actually solved after adding regularization.
The propagated velocity is

$$
\dot\theta_k = K_{k,\tau,\epsilon}^{\oplus} f_k .
$$

## Tangent modes and why tiny modes are discarded

The coordinates in the circuit are not automatically good coordinates in the
tangent space. Two different parameters can produce almost the same state
motion, or one parameter can barely move the state at all. The eigenvectors of
`K_k` are the clean tangent modes seen by the linear solve. Large eigenvalues
are well-resolved modes. Very small eigenvalues are unresolved directions where
tiny force or measurement errors can create huge parameter velocities.

The supported inverse keeps only the modes whose eigenvalues are large enough:

$$
A=V\operatorname{diag}(s_i)V^\top,
\qquad
A_{\tau,\epsilon}^{\oplus}
=V\operatorname{diag}\left(
\frac{\mathbf 1[s_i\ge \tau_{\rm pinv}s_1]}{s_i+\epsilon_{\rm solve}}
\right)V^\top .
$$

This is not saying the discarded coordinate is physically impossible. It says
the current data cannot support inverting that direction. In the controller,
this distinction matters: a poor solve is handled by damping, cutoff, or step
repair, while missing physical expressivity is handled by append.

### Toy example: null or redundant direction

Suppose two coordinates make exactly the same tangent direction:

$$
\bar\tau_1=u,\qquad \bar\tau_2=u .
$$

Then

$$
G=
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
$$

has eigenvalues `2` and `0`. The mode `(1,1)` changes the state. The mode
`(1,-1)` changes one parameter up and the other down but cancels in state
space. Inverting the zero mode would ask the solver to assign arbitrary
opposite velocities to two coordinates with no physical effect. With noise, a
near-zero eigenvalue would be even worse: a tiny force component could be
amplified into a huge parameter step. The supported inverse discards that
mode.

## Expressivity miss

The unregularized residual measures what the current tangent space cannot
represent, independent of the damping used in the implemented solve:

$$
\epsilon_{\rm expr,k}^2
=\min_v\|\bar T_kv-\bar b_k\|_2^2
=\operatorname{Var}_{\psi_k}(H_k^\sharp)-f_k^\top G_{k,\tau}^{\dagger}f_k .
$$

If this miss is small, the current scaffold has enough tangent directions for
the local motion. If it is large, the controller opens the append lane. The
important point is that append decisions use expressivity miss, not just the
realized regularized residual. A bad numerical solve is not automatically a
reason to add new circuit blocks.

## Candidate append geometry

An append candidate `r=(m,p)` inserts zero-valued new coordinates. The state
does not jump, but the tangent plane immediately gains new columns. If the
candidate has primitives `A_{r,a}`, the appended tangent block is

$$
\bar U_{r,k}
=
\left[
-iQ_{\psi_k}\widetilde A_{r,1}^{(p)}|\psi_k\rangle
\ \cdots\
-iQ_{\psi_k}\widetilde A_{r,m_r}^{(p)}|\psi_k\rangle
\right].
$$

The append blocks compare the old tangent columns, the new candidate columns,
and the Hamiltonian velocity:

$$
B_{r,k}=\operatorname{Re}(\bar T_k^\dagger\bar U_{r,k}),
\qquad
C_{r,k}=\operatorname{Re}(\bar U_{r,k}^\dagger\bar U_{r,k}),
\qquad
q_{r,k}=\operatorname{Re}(\bar U_{r,k}^\dagger\bar b_k).
$$

`B_{r,k}` says how much the candidate tangent overlaps the old tangent space.
`C_{r,k}` is the candidate's own metric. `q_{r,k}` is the candidate force.
Large `q` alone is not enough, because the candidate may point in a direction
the old scaffold can already make.

## Schur gain as residual reduction

The Schur gain asks: after the old manifold has already done its best, how much
additional residual can the candidate remove? The old-manifold contribution is
subtracted before scoring the new block:

$$
w_{r,k}=q_{r,k}-B_{r,k}^\top K_{k,\tau,\epsilon}^{\oplus}f_k,
$$

$$
S_{r,k}
=C_{r,k}+\Lambda_r
-B_{r,k}^\top K_{k,\tau,\epsilon}^{\oplus}B_{r,k},
$$

$$
\Delta_{r,k}^{\rm Schur}
=w_{r,k}^\top(S_{r,k})_\tau^\dagger w_{r,k}.
$$

`w_{r,k}` is the force left for the candidate after subtracting what the old
space explains. `S_{r,k}` is the candidate metric after subtracting old-space
redundancy. The gain is therefore not a raw tangent size and not a raw
gradient. It is a local residual-reduction estimate in the part of candidate
space that remains new.

### Toy example: large-looking but redundant append

Let the current old tangent be one normalized direction `u`, and suppose the
Hamiltonian velocity has a large component along it:

$$
\bar T=[u],\qquad \bar b = 10u + e,
\qquad \langle u,e\rangle=0 .
$$

Now consider an append candidate with tangent `\bar U=[10u]`. Its raw force is
large:

$$
q=\langle 10u,\bar b\rangle=100 .
$$

But it is just the old direction scaled up. The old-new overlap and new-new
metric are

$$
B=10,\qquad C=100,\qquad K=1 .
$$

The redundancy-subtracted force is

$$
w=q-B^\top K^{-1}f = 100-10\cdot 10=0 .
$$

So the Schur gain is zero. The candidate looked important because it had a
large raw force, but after the old tangent direction was accounted for, it had
no new residual to remove.

## Whitened coordinates

The whitened form rewrites the same Schur geometry in coordinates where the
retained old modes have unit scale. On retained support,

$$
K_k=V_k\Sigma_k^2V_k^\top .
$$

The old solved force becomes

$$
z_k=\Sigma_k^{-1}V_k^\top f_k .
$$

The candidate overlap with retained old modes becomes

$$
\Gamma_{r,k}=\Sigma_k^{-1}V_k^\top B_{r,k}.
$$

Then the candidate force after subtracting old-manifold content is

$$
\zeta_{r,k}=q_{r,k}-\Gamma_{r,k}^\top z_k,
$$

and the support-truncated candidate Schur matrix is

$$
\widetilde S_{r,k}(J)
=C_{r,k}+\Lambda_r-\Gamma_{r,k;J}^\top\Gamma_{r,k;J}.
$$

The whitened confirmation score is

$$
\widetilde\Delta_{r,k}^{(\lambda_S)}(J)
=\zeta_{r,k}^\top
\left(\widetilde S_{r,k}(J)+\lambda_SI\right)^{-1}
\zeta_{r,k}.
$$

Whitening is useful because different old tangent modes can have wildly
different scales. In unwhitened coordinates, a mode with a large metric value
and a mode with a small metric value are hard to compare. After whitening, the
retained old modes are normalized, so the candidate's overlap with old support
has a direct geometric interpretation.

## Nested retained mode sets

The manuscript uses nested retained mode sets

$$
J_r^{(0)}\subseteq \cdots \subseteq J_r^{(m_{\max})}
$$

as a cheap-to-expensive scout/confirm ladder. These are sets of **old-scaffold
spectral modes**, not sets of new candidate coordinates. First the supported
inverse determines the available trusted mode index set,

$$
I_\tau=\{a:\Sigma_{k,a}^2\ge \tau_{\rm pinv}\Sigma_{k,\max}^2\}.
$$

Each `J_r^{(m)}` is a nested subset of `I_\tau`. In practice the subsets can be
ordered by the strongest retained eigenmodes, by the candidate's largest
whitened overlaps, or by another fixed scout policy. The important manuscript
point is not the exact ordering rule; it is that the supports are nested, use
the same numerator, and use the same positive ridge.

A small `J` checks a cheap portion of the trusted old tangent support. Larger
sets include more retained support and make the old-space correction more
complete. With a common ridge `\lambda_S`, the scores obey

$$
\widetilde\Delta_{r,k}^{(\lambda_S)}(J_r^{(0)})
\le \cdots \le
\widetilde\Delta_{r,k}^{(\lambda_S)}(J_r^{(m_{\max})})
\le \Delta_{r,k}^{\rm Schur,(\lambda_S)} .
$$

Operationally, this lets the controller scout many candidates with a cheap
compressed calculation and reserve the full commit-grade geometry for the few
that survive. The nested sets do not define a different objective; they are a
budget-aware way to approach the same Schur append test.

## Equation map

| Paper equation | Meaning |
|---|---|
| Eq. 42 | Supported inverse `A_{\tau,\epsilon}^{\oplus}`. It inverts only retained spectral modes and damps them by `\epsilon_{\rm solve}`. |
| Eq. 43 | Expressivity miss `\epsilon_{\rm expr,k}^2` and normalized miss `\rho_{\rm expr,k}`. This measures residual motion not captured by the current tangent space. |
| Eq. 49 | Append blocks `B_{r,k}`, `C_{r,k}`, and `q_{r,k}`. These encode old-new overlap, candidate metric, and candidate force. |
| Eq. 50 | Schur gain. This is the candidate's residual reduction after subtracting old-manifold redundancy. |
| Eq. 51 | Whitened append coordinates `z_k`, `\Gamma_{r,k}`, `\zeta_{r,k}`, and `\widetilde S_{r,k}(J)`. These are compressed coordinates for confirming the same Schur geometry. |
| Eq. 52 | Nested retained-support inequality. It justifies using increasingly expensive retained-mode sets as a scout/confirm ladder. |

## Reading checklist

When reading the append section, keep these distinctions separate:

- `G_k` describes the current tangent geometry; `K_k` is the regularized solve
  matrix; `f_k` is the Hamiltonian force projected onto current tangents.
- A tiny eigenvalue is not a useful narrow passage to exploit. It is an
  unresolved mode that can turn noise into unstable velocities.
- Append is state-continuous: zero coordinates are inserted, but new tangent
  directions become available for the next McLachlan step.
- A candidate with a large raw force can still have low value if old tangents
  already explain that force.
- The Schur gain is best read as residual reduction after old-space
  redundancy has been removed.
- Whitened and nested-support formulas are implementation-efficient
  confirmations of the same local geometry, not separate physics.
