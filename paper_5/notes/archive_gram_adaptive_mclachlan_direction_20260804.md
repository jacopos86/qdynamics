# Archive-Gram-guided adaptive multi-coherent McLachlan direction

Status: the first stored-state mixed-tangent pilot is implemented and
numerically validated; an online archive-Gram-guided adaptive controller has
not yet been implemented.

## Why this route is distinct

The multi-coherent state remains the propagated, representable state.  Its
native packet-parameter tangent frame

\[
\left\{
\frac{\partial |\Psi_K\rangle}{\partial\theta_i}
\right\}
\]

supplies the Gram used to determine the McLachlan velocity.  Contraction of the
packet ket produces the archive tuple \(X=(\rho,B,N,A,C)\); these matrices are
physical observables and a comparison interface.  Their 31 independent real
coordinates do not determine an autonomous exact propagation.

Three related routes answer different questions:

\[
\begin{aligned}
\text{packet-capacity continuation}
&\longrightarrow
\text{how much variational capacity is required},\\
\text{archive-Gram-guided packet growth}
&\longrightarrow
\text{when and where the packet manifold must grow},\\
\text{operator-tangent enrichment}
&\longrightarrow
\text{which physical operator directions are required}.
\end{aligned}
\]

In archive-Gram-guided packet growth, coherent packets remain the variational
manifold and archive or mixed operators serve as observer directions.  Direct
operator enrichment instead appends selected operator-generated vectors to the
variational tangent frame.  The methods can be compared or combined after
their separate effects are measured.

## Archive joint Gram as observer-tangent geometry

Define the centered fluctuation-operator column

\[
\mathbf Y=
\left(
\delta b_0,
\delta b_1,
\delta b_0^\dagger,
\delta b_1^\dagger,
\delta\sigma_x,
\delta\sigma_y,
\delta\sigma_z
\right)^{\mathsf T}.
\]

The archive joint moment Gram is

\[
\mathcal G_{ab}
=\langle Y_a^\dagger Y_b\rangle
=
\begin{pmatrix}
\mathcal M_{\rm B}(N,A)&Z(C)\\
Z(C)^\dagger&E(\rho)
\end{pmatrix}_{ab}.
\]

For the local observer tangents

\[
|U_a\rangle=Y_a|\Psi\rangle,
\]

centering gives \(\langle\Psi|U_a\rangle=\langle Y_a\rangle=0\).
They are therefore horizontal with respect to norm and global phase, and

\[
\langle U_a|U_b\rangle=\mathcal G_{ab}.
\]

Consequently, the archive joint Gram is exactly the tangent Gram of this
fluctuation-generated observer frame.  If these observer directions are used
as a variational frame in the separate operator-enrichment route, complex
coefficients \(v_a\) are obtained from

\[
\underset{v}{\operatorname{minimize}}
\left\|
\sum_a v_aY_a|\Psi\rangle
+\mathrm i\bigl(H-\langle H\rangle\bigr)|\Psi\rangle
\right\|_2^2,
\]

which gives

\[
\mathcal Gv=f,
\qquad
f_a=-\mathrm i\left\langle
Y_a^\dagger\bigl(H-\langle H\rangle\bigr)
\right\rangle.
\]

A real-coordinate implementation realifies both \(\mathcal G\) and \(f\).
Smooth Tikhonov filtering should be compared with a hard pseudoinverse cutoff
because the two regularizations alter soft tangent directions differently.
In packet adaptation this observer-frame solve remains diagnostic; the native
packet Gram determines the propagated velocity.

## Why the archive frame is not autonomous

Although \(\mathcal G\) is determined by the retained 31 coordinates, the
force \(f\) and the induced map from tangent coefficients to moment velocities
generally are not.  For Hermitian coordinate observables \(O_\mu\),

\[
\dot x_\mu
=2\operatorname{Re}
\sum_a v_a^*\langle Y_a^\dagger O_\mu\rangle.
\]

When \(O_\mu\) belongs to the electron--phonon correlation block, the required
matrix elements include terms such as

\[
\left\langle
\delta b_{q'}^\dagger\delta b_q\delta\sigma_a
\right\rangle,
\]

which are precisely electron-conditioned two-phonon correlations omitted by
the 31-coordinate closure.  Directly differentiating \(\mathcal G\) similarly
introduces third-order products and centering derivatives.  The identity
between the archive Gram and the tangent Gram is therefore local and useful,
but it does not make the 31-coordinate model closed.

## Direct operator-tangent enrichment

The operator-enrichment route promotes selected operator-generated vectors to
variational tangents.  Its first candidates are mixed conditional
electron--phonon directions.

The cross block \(Z(C)\) changes the geometry between separate bosonic and
electronic tangent directions, but it does not include the genuinely mixed
directions

\[
\delta b_q\delta\sigma_a|\Psi\rangle,
\qquad
\delta b_q^\dagger\delta\sigma_a|\Psi\rangle.
\]

These operators are naturally associated with electron-conditioned phonon
displacement, phase, and branching.  Unlike the entries of \(\mathbf Y\), a
mixed product is not generally centered.  Every candidate must therefore be
horizontalized:

\[
|\widetilde W_\alpha\rangle
=
\left(I-|\Psi\rangle\langle\Psi|\right)
W_\alpha|\Psi\rangle
=
\bigl(W_\alpha-\langle W_\alpha\rangle I\bigr)|\Psi\rangle.
\]

The first enlarged frame should be

\[
\mathbf W=
\left(
\mathbf Y,
\left\{
\delta b_q\delta\sigma_a,
\delta b_q^\dagger\delta\sigma_a
\right\}_{q,\,a\in\{x,y,z\}}
\right),
\]

with Gram and force evaluated from the explicit packet ket.  This avoids
assuming that the missing mixed moments can already be reconstructed from the
31 coordinates.

### Adaptive operator enrichment

Rather than fixing another moment order in advance, begin with the archive
frame and calculate the component of

\[
|R\rangle=-\mathrm i(H-\langle H\rangle)|\Psi\rangle
\]

orthogonal to the current tangent span.  For each horizontalized candidate
\(|\widetilde W_\alpha\rangle\), score the additional residual reduction after
projecting out the current span.  A normalized one-direction score is

\[
s_\alpha
=
\frac{
|\langle\widetilde W_\alpha|R_\perp\rangle|^2
}{
\langle\widetilde W_\alpha|\widetilde W_\alpha\rangle
},
\]

with the actual append decision evaluated through the augmented, regularized
Gram solve.  Append only directions that materially reduce both the
Hilbert-space residual and the induced retained-observable error.  Higher
conditional phonon products should enter the candidate pool only when the
remaining residual requires them.

This operation appends state-space tangent directions, not moment coordinates.
The repeatedly selected operators can subsequently identify candidate hidden
coordinates or additional moments for an autonomous closure.

### First offline operator-frame pilot

At the same stored packet states and times, compare the archive frame, its
mixed extension, and the native packet frame.  Only the within-family
inclusions are assumed:

\[
\mathcal T_Y\subset\mathcal T_W,
\qquad
\mathcal T_{\rm packet}
\subset
\mathcal T_{\rm packet}+\mathcal T_Y
\subset
\mathcal T_{\rm packet}+\mathcal T_W.
\]

The standalone archive and packet spaces are generally nonnested.

Measure:

1. relative McLachlan residual in each space;
2. the induced error in all 31 retained-coordinate velocities, with the
   correlation block reported separately;
3. residual reduction attributable to the mixed conditional directions;
4. Gram singular values, regularization sensitivity, and effective rank;
5. consistency of selected directions across preparations, drive protocols,
   coupling regimes, and time windows; and
6. tangent-frame size and evaluation cost without imposing a scientific
   rejection at an arbitrary fixed cap.

A substantial reduction from \(\mathcal T_Y\) to \(\mathcal T_W\), followed
by only a small remaining gap to \(\mathcal T_{\rm packet}\), would identify
electron-conditioned displacement and branching as the compact physical
mechanism.  Rapid adaptive growth toward the full Hilbert-space tangent rank
would instead provide evidence against a small operator-derived closure.

## Stored-state pilot result

The pilot was evaluated at 41 times, \(t=0,1,\ldots,40\), on each stored
strong-coupling double-pulse trajectory with maximum packet capacities
\(K=6,8,10,12\).  The stored state contains only the interacting relative
phonon mode.  The calculation restored the factored center mode as a centered
vacuum and projected the packet ket and tangent into the spin-exchange-
symmetric dimer sector before constructing the archive frame.  The minimum
fidelity between a raw packet state and its symmetric projection was
\(0.99634\), \(0.99817\), \(0.99967\), and \(0.99842\), respectively.

The explicit operator-vector Gram and the Gram reconstructed from
\((\rho,N,A,C)\) obeyed

\[
\max_{ab}|(\mathcal G_{\rm op}-\mathcal G_{\rm archive})_{ab}|
=\frac{33}{2}\,p_{32}
\]

to within \(2.4\times10^{-15}\), where \(p_{32}\) is the population in the
highest retained relative-mode Fock level.  The finite difference between the
two Grams is therefore exactly the cutoff-space commutator boundary term, not
a lift or contraction error.

The local-site mixed pool contains redundant center and relative combinations.
The six complex symmetry-adapted directions

\[
\delta a_{\rm rel}\delta\sigma_a|\Psi\rangle,
\qquad
\delta a_{\rm rel}^{\dagger}\delta\sigma_a|\Psi\rangle,
\qquad a\in\{x,y,z\},
\]

gave the same projection residual as all twelve local-site mixed products to
within \(4.4\times10^{-16}\), while using six fewer real tangent ranks.

| maximum packets per branch | packet Hilbert residual | packet plus relative-mixed residual | reduction | packet \(C\)-velocity error | enriched \(C\)-velocity error | enriched all-31 error |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | `0.11084` | `0.06607` | `40.4%` | `2.94e-2` | `4.02e-12` | `3.16e-3` |
| 8 | `0.09817` | `0.05203` | `47.0%` | `2.11e-2` | `8.59e-11` | `2.95e-3` |
| 10 | `0.08738` | `0.03901` | `55.4%` | `7.44e-3` | `8.73e-12` | `3.23e-4` |
| 12 | `0.09237` | `0.04045` | `56.2%` | `2.15e-2` | `1.39e-8` | `4.36e-4` |

These are time-RMS same-state errors in the declared scaled coordinates.  By
contrast, adding the archive frame without the mixed products reduced the
packet Hilbert residual by only \(0.14\%\) to \(1.13\%\).  The relative-mode
mixed products therefore supply a distinct missing tangent mechanism rather
than merely restating the archive Gram.  The recurring individual order was
\(\delta a_{\rm rel}\delta\sigma_z\), then its creation partner and the
\(y\) directions, with the \(x\) directions weaker.

The result is local and teacher-evaluated from each current packet ket.  It
does not yet define an autonomous 31-coordinate closure or an online enriched
propagator.  The next closure-specific test should extract the six mixed
coefficients across preparations and drives and determine whether a compact
state variable and reciprocal evolution law predict them in free rollout.

Pilot artifacts:

`output/local_runs/paper_v_archive_mixed_tangent_pilot_cutoff16_20260804_v3/`.

## Mixed-coefficient transfer result

The six complex relative-mode--Pauli directions were next residualized against
the archive frame at 81 stored states over `0 <= t <= 20` for three nearby
preparations under the double pulse, the central preparation under the single
pulse, and the central double-pulse trajectory with `K_max=10`.  At every
sample the factorization

\[
\dot C_{\rm Sch}-\dot C_{31}
=
(\dot C_Y-\dot C_{31})+J_C(X)\eta+r_\perp
\]

had maximum coordinate residual below `2.7e-15`.  Here `dot C_Y` is obtained
from the archive-frame projection, `eta` contains the twelve real coefficients
of the six complex mixed directions after residualization, and `J_C(X)` is
their state-dependent contraction into the fourteen real correlation
velocities.  Thus the six directions are an exact local entrance frame for the
same-state `C` velocity on these packet states.

They are not, however, a fixed source decoder.  Leave-one-preparation-out
linear decoding from `eta` alone had normalized RMS source error `0.774` to
`0.798`.  Adding the retained state and current drive reduced the same-drive
preparation errors to `0.176`--`0.195`, but the distinct-drive test was
decisive.  The single- and double-pulse central paths agree before the second
pulse, where the state-plus-mixed decoder error was `0.159`.  After `t=8`, the
same single-pulse holdout error rose to `1.536`, and it reached `1.711` over
`10 <= t <= 20`.  The `K_max=10` capacity holdout remained close to the
same-drive data (`0.178` over the full interval), so the intervention failure
is not a packet-coordinate-capacity artifact.

The response map itself varies materially: five modes explain 90 percent of
its sampled variance, thirteen explain 99 percent, and twenty-one explain
99.9 percent.  A double-pulse response basis required 35 modes for source NRMS
below `0.1` on the post-intervention single-pulse path and 42 modes for error
below `0.01`.  This PCA count is diagnostic rather than a hidden-state order,
because it is not weighted by reachability and observability, but it rules out
identifying the six coefficients with a six-coordinate instantaneous source.

Decision: stop extending static decoders.  Treat the mixed products as the
leading entrance channels of the unresolved, history-dependent dynamics.  The
next derivation should apply the state-adapted complement of the archive Gram
projection to their Liouvillian images and construct a reciprocal auxiliary
operator realization.  Connected electron--two-phonon and opposite-spin
operators arise in the first commutator layer; packet states can evaluate the
required Grams, initialize their amplitudes, and audit observability without
becoming online inputs.

Artifact:

`output/local_runs/paper_v_mixed_tangent_closure_identifiability_cutoff16_t20_20260804_v2/`.

## Archive-Gram-guided adaptive packet growth

The strongest immediate candidate route propagates an electron-conditioned
multi-coherent ket and uses the archive joint Gram as observer geometry.  The
native packet Gram determines the McLachlan velocity; the archive geometry
tests which physically meaningful fluctuation directions the current packet
chart can realize.

### Native packet McLachlan velocity

Write the normalized packet state as

\[
\begin{aligned}
|\Psi_K(\theta)\rangle
&=
\sum_e\sum_{k=1}^{K_e}
c_{ek}|e\rangle|\boldsymbol\alpha_{ek}\rangle,\\
|\boldsymbol\alpha_{ek}\rangle
&=
\bigotimes_q|\alpha_{ekq}\rangle .
\end{aligned}
\]

The index \(e\) labels an electronic branch, \(k\) labels a coherent packet
within that branch, \(q\) labels a phonon mode, and \(K_e\) is the packet count
in branch \(e\).  The real vector \(\theta\in\mathbb R^{p_K}\) stores the
independent real and imaginary parts of the coefficients and displacements.
Its horizontal tangents are

\[
|\overline T_i\rangle
=
\left(I-|\Psi_K\rangle\langle\Psi_K|\right)
\partial_{\theta_i}|\Psi_K\rangle,
\qquad
i\in\{1,\ldots,p_K\}.
\]

Using the real Hilbert-space inner product
\((u,v)_{\mathbb R}=\operatorname{Re}\langle u|v\rangle\), define

\[
\begin{aligned}
(G_K)_{ij}
&=
\operatorname{Re}
\langle\overline T_i|\overline T_j\rangle,\\
(b_K)_i
&=
\operatorname{Re}
\left\langle
\overline T_i
\middle|
-\mathrm i(H-\langle H\rangle)
\middle|
\Psi_K
\right\rangle,\\
G_K\dot\theta
&=
b_K .
\end{aligned}
\]

Here \(G_K\in\mathbb R^{p_K\times p_K}\) is the native packet tangent Gram and
\(b_K\in\mathbb R^{p_K}\) is the projected Schrödinger force.  A declared
rank-revealing factorization and regularization select the numerically retained
tangent range.

### Realified archive-observer geometry

At the same packet state, reconstruct the centered archive frame

\[
\mathbf Y
=
\left(
\delta b_0,\delta b_1,
\delta b_0^\dagger,\delta b_1^\dagger,
\delta\sigma_x,\delta\sigma_y,\delta\sigma_z
\right)^{\mathsf T},
\qquad
|U_a\rangle=Y_a|\Psi_K\rangle .
\]

Its complex Hermitian Gram is

\[
\mathcal G_K
=
\left(\langle U_a|U_b\rangle\right)_{ab}
=
\langle\mathbf Y^\dagger\mathbf Y\rangle_{\Psi_K}
=
\begin{pmatrix}
\mathcal M_{\rm B}(N,A)&Z(C)\\
Z(C)^\dagger&E(\rho)
\end{pmatrix}
\succeq0 .
\]

The packet coordinates are real, so the packet and observer geometries must be
combined in a real tangent space.  For any complex matrix \(A\), define

\[
\mathscr R(A)
=
\begin{pmatrix}
\operatorname{Re}A&-\operatorname{Im}A\\
\operatorname{Im}A&\operatorname{Re}A
\end{pmatrix}.
\]

The real archive tangent frame and its Gram are

\[
\mathbb U
=
\left(
U_1,\ldots,U_7,
\mathrm iU_1,\ldots,\mathrm iU_7
\right),
\qquad
\mathfrak G_K
=
\mathscr R(\mathcal G_K).
\]

The complex packet--observer cross-Gram and its real form are

\[
\begin{aligned}
(S_K)_{ai}
&=
\langle U_a|\overline T_i\rangle,
\qquad
S_K\in\mathbb C^{7\times p_K},\\
\mathfrak S_K
&=
\begin{pmatrix}
\operatorname{Re}S_K\\
\operatorname{Im}S_K
\end{pmatrix}
\in\mathbb R^{14\times p_K}.
\end{aligned}
\]

The consistent unified real Gram is

\[
\boxed{
\Gamma_K^{\mathbb R}
=
\begin{pmatrix}
G_K&\mathfrak S_K^{\mathsf T}\\
\mathfrak S_K&\mathfrak G_K
\end{pmatrix}
\succeq0 .
}
\]

Adjoint-paired observer directions, state-specific symmetries, and redundant
packet coordinates can create exact or near dependencies.  Rank revelation
and regularization are therefore part of the numerical realization.  Their
appearance does not by itself invalidate the geometry.

With the exact Moore--Penrose inverse on the range of \(G_K\), the generalized
Schur complement is

\[
\boxed{
\mathfrak N_K
=
\mathfrak G_K
-
\mathfrak S_KG_K^+\mathfrak S_K^{\mathsf T}
\succeq0 .
}
\]

This is the Gram of the archive observer tangents after orthogonal projection
away from the packet tangent span.  It identifies fluctuation directions that
the current packet chart cannot instantaneously realize.  A smooth spectral
filter produces a regularized novelty estimate, which must be reported
separately from the exact Moore--Penrose construction.  In the chosen observer
scaling, the scalar \(\operatorname{Tr}\mathfrak N_K\) measures total geometric
novelty, but it cannot determine packet birth because a missing direction may
be dynamically inactive under the current state and drive.

### Archive-Gram rate defect

At the current packet state define the horizontal Schrödinger and McLachlan
velocities

\[
\begin{aligned}
|\dot\Psi_{\rm Sch}\rangle
&=
-\mathrm i(H-\langle H\rangle)|\Psi_K\rangle,\\
|\dot\Psi_{\rm McL}\rangle
&=
\sum_i\dot\theta_i|\overline T_i\rangle .
\end{aligned}
\]

Write each observer as \(Y_a=O_a-\langle O_a\rangle I\).  For either velocity
label \(v\in\{\mathrm{Sch},\mathrm{McL}\}\), the complete derivative is

\[
\begin{aligned}
\left(\dot{\mathcal G}^{(v)}_K\right)_{ab}
={}&
\langle\dot\Psi_v|Y_a^\dagger Y_b|\Psi_K\rangle
+\langle\Psi_K|Y_a^\dagger Y_b|\dot\Psi_v\rangle\\
&+
\left\langle
\dot Y_{a,v}^\dagger Y_b
+Y_a^\dagger\dot Y_{b,v}
\right\rangle,\\
\dot Y_{a,v}
={}&
\partial_tO_a
-
\frac{d}{dt}\langle O_a\rangle_v I,\\
\frac{d}{dt}\langle O_a\rangle_v
={}&
\langle\dot\Psi_v|O_a|\Psi_K\rangle
+\langle\Psi_K|O_a|\dot\Psi_v\rangle
+\langle\partial_tO_a\rangle .
\end{aligned}
\]

For the fixed underlying operators used in the present archive frame,
\(\partial_tO_a=0\).  The remaining centered-operator terms are scalar
multiples of \(\langle Y_a\rangle\) or \(\langle Y_b\rangle\) and therefore
vanish analytically.  Keeping them in the definition and implementation audit
makes the cancellation explicit.  A time-adapted phonon mode, electronic
basis, or observer operator contributes through \(\partial_tO_a\) and need not
cancel.

The complex and realified rate defects are

\[
R_{\mathcal G,K}
=
\dot{\mathcal G}^{(\mathrm{Sch})}_K
-
\dot{\mathcal G}^{(\mathrm{McL})}_K,
\qquad
\mathfrak R_{\mathcal G,K}
=
\mathscr R(R_{\mathcal G,K}).
\]

A regularized Gram-whitened rate defect is

\[
\varepsilon_{\mathcal G,K}^2
=
\left\|
\left(\mathfrak G_K+\lambda_{\mathcal G}I\right)^{-1/2}
\mathfrak R_{\mathcal G,K}
\left(\mathfrak G_K+\lambda_{\mathcal G}I\right)^{-1/2}
\right\|_{\rm F}^2,
\qquad
\lambda_{\mathcal G}>0.
\]

This quantity resolves the defect relative to the current fluctuation scales;
the regularizing \(\lambda_{\mathcal G}I\) limits invariance to unitary archive
frame changes.  A dimensionless relative diagnostic requires an explicitly
declared reference-rate normalization.

The ordinary McLachlan residual is

\[
\varepsilon_{\rm McL,K}^2
=
\left\|
|\dot\Psi_{\rm Sch}\rangle
-
|\dot\Psi_{\rm McL}\rangle
\right\|_2^2 .
\]

Choose positive scales \(s_{\rm McL}\) and \(s_{\mathcal G}\) before candidate
scoring and define

\[
\widetilde\varepsilon_{\rm McL,K}
=
\frac{\varepsilon_{\rm McL,K}}{s_{\rm McL}},
\qquad
\widetilde\varepsilon_{\mathcal G,K}
=
\frac{\varepsilon_{\mathcal G,K}}{s_{\mathcal G}}.
\]

The scales may be fixed from the corresponding Schrödinger velocity and
Gram-rate norms, with declared nonzero floors.  They remain fixed during each
candidate comparison.  The joint local objective is

\[
\mathcal J_K
=
\widetilde\varepsilon_{\rm McL,K}^2
+
\mu\widetilde\varepsilon_{\mathcal G,K}^2,
\qquad
\mu\geq0 .
\]

Both rates are computed from the current packet ket.  This trigger therefore
uses no exact reference trajectory during propagation.

### Candidate search and two-stage packet birth

For a coherent-packet candidate \(\chi\), recompute the regularized augmented
tangent solve and score its reduction of the same objective:

\[
\boxed{
\mathcal S(\chi)
=
\frac{
\mathcal J_K-\mathcal J_{K+\chi}
}{
\Delta P_\chi
}.
}
\]

The positive resource increment \(\Delta P_\chi\) is the declared added
parameter or evaluation cost.  The subtraction makes \(\mathcal S(\chi)\) an
objective reduction per added resource.

Sector-resolved components of \(R_{\mathcal G,K}\) and \(\mathfrak N_K\) guide
candidate generation.  Final selection evaluates candidates across the
allowed electronic sectors and coherent displacements using the actual
\(\mathcal S(\chi)\), dynamically weighted observer coverage, tangent-rank
gain, and conditioning.  No electronic sector is assigned solely from a
heuristic decomposition of the Gram defect.

State-continuous packet admission has two stages.  For

\[
|\Psi_{K+\chi}\rangle
=
|\Psi_K\rangle
+
c_\chi|e_\chi\rangle|\boldsymbol\alpha_\chi\rangle,
\]

the choice \(c_\chi=0\) preserves the ket, while

\[
\partial_{\operatorname{Re}\alpha_{\chi q}}
|\Psi_{K+\chi}\rangle
\propto c_\chi,
\qquad
\partial_{\operatorname{Im}\alpha_{\chi q}}
|\Psi_{K+\chi}\rangle
\propto c_\chi .
\]

At birth, fix \(\boldsymbol\alpha_\chi\) and expose only the horizontalized real
and imaginary coefficient tangents.  After \(|c_\chi|\) exceeds a declared
activation threshold, release the displacement coordinates and perform a
trust-region refit.  Amplitude-weighted displacement coordinates provide a
possible nonsingular alternative.  Adding every packet coordinate at
\(c_\chi=0\) would insert zero tangent columns and reproduce the rank loss that
the adaptive chart is intended to avoid.

### Conditioning, merge, prune, and stopping rules

Candidate acceptance and continued propagation monitor the geometric tangent
rank, regularized retained rank, smallest retained singular value,
parameter-velocity norm, ordinary McLachlan residual, archive-Gram rate defect,
contracted observable errors, and resource cost.

Packets in the same electronic sector may be merged when their coherent-state
overlap is near unity, their independent tangent-rank contribution is
negligible, and a local refit preserves the ket and contracted observables.
Packets from different electronic sectors require equivalence of the complete
electronic--phononic branch states before any analogous merge.  A packet is
pruned only when removal causes negligible increase in \(\mathcal J_K\),
negligible loss of dynamically weighted observer coverage, and no material
observable change after a continuity-preserving refit.

Packet count is a resource axis rather than a theorem about scientific
sufficiency.  The earlier \(r\leq96\) generalized-Hankel gate rejects only that
budgeted global lossless bilinear realization, and the exploratory
\(K\leq6\) ceiling does not define a universal stopping rule.  Adaptive
continuation stops when repeated candidate searches yield negligible marginal
objective reduction per cost, conditioning cannot be repaired through chart
changes or merging, or the packet propagation ceases to improve upon the
declared comparator.

### Mixed conditional observers

The packet route can enlarge its observer geometry to

\[
\mathbf W
=
\left(
\mathbf Y,
\left\{
\delta b_q\delta\sigma_a,
\delta b_q^\dagger\delta\sigma_a
\right\}_{q,\,a\in\{x,y,z\}}
\right).
\]

Because a mixed product \(W_\alpha\) is not generally centered, define its
horizontal observer tangent

\[
|\widetilde W_{\alpha,K}\rangle
=
\left(I-|\Psi_K\rangle\langle\Psi_K|\right)
W_\alpha|\Psi_K\rangle
=
\left(W_\alpha-\langle W_\alpha\rangle I\right)|\Psi_K\rangle .
\]

The packet ket then supplies

\[
\mathcal G^{(W)}_{\alpha\beta}
=
\langle
\widetilde W_{\alpha,K}|\widetilde W_{\beta,K}
\rangle,
\qquad
S^{(W)}_{\alpha i}
=
\langle
\widetilde W_{\alpha,K}|\overline T_i
\rangle .
\]

The same realification, unified-Gram, Schur-novelty, and rate-defect
construction applies to this enlarged observer frame.  Using it as observer
geometry preserves coherent packets as the propagated manifold.  Appending
\(|\widetilde W_{\alpha,K}\rangle\) to the variational tangent frame defines the
separate operator-enrichment route.

## Packet-to-augmented-state compression routes

After each packet propagation step, the state can be contracted into the
archive matrices,

\[
|\Psi_{\rm packet}(t)\rangle
\longrightarrow
X(t)=(\rho,B,N,A,C).
\]

This map is many-to-one.  Different packet states can have the same retained
matrices but different future derivatives, so discarding the packet state
after contraction generally removes the branching and phase memory needed to
repair the archive closure.  The appropriate compression target is therefore

\[
|\Psi_{\rm packet}\rangle
\longrightarrow
(X,\eta),
\]

where \(\eta\) retains only the additional packet information needed to
predict the future missing source and retained observables.

The instantaneous packet velocities \(\dot\theta_i\) are local tangent
coefficients:

\[
\dot\theta(t)=\text{the best packet-coordinate velocity at the current state},
\qquad
\eta(t)=\text{state memory needed to predict later velocities}.
\]

They become candidate memory variables only after satisfying the predictive
test below.  Three related compression constructions remain available.

### Operator-informed compression

Use the adaptive mixed-tangent analysis to identify recurrent physical
directions \(|\widetilde W_r\rangle\).  Candidate components of \(\eta\) can
then be built from their expectation values, packet-spawn history, overlap
amplitudes, and reachable or observable tangent combinations.  Candidate
mechanisms include electron-conditioned relative displacement, branch weight,
relative phase, and interference.

### Direct packet compression

Infer \(\eta\) directly from packet trajectories by requiring \((X,\eta)\) to
predict the missing \(C\)-source and future retained observables.  This is an
alternative to prior operator selection, but it is more vulnerable to
trajectory-specific latent variables and requires stronger preparation and
drive holdouts.

### Periodically refreshed hybrid

Propagate the augmented archive model between selected refresh times.  When an
error or uncertainty indicator activates, lift the reduced state through

\[
\mathcal L:(X,\eta)\longmapsto|\Psi_K\rangle
\]

and perform a short adaptive packet correction before re-encoding
\((X,\eta)\).  The lifting \(\mathcal L\) is generally nonunique.  The refresh
scheme must therefore verify that admissible lifts produce equivalent retained
dynamics over the tested prediction horizon.

For any route, predictive sufficiency requires that, under matched future
drives,

\[
\left\|(X_1,\eta_1)-(X_2,\eta_2)\right\|\leq\epsilon
\quad\Longrightarrow\quad
\sup_{0\leq\tau\leq T_{\rm pred}}
\left\|X_1(t+\tau)-X_2(t+\tau)\right\|
\leq\delta .
\]

The tolerances \(\epsilon,\delta\), prediction horizon \(T_{\rm pred}\), state
metric, preparations, and future-drive class must be declared before the
test.  Reconstructing only the present packet state or missing source does not
establish this future equivalence.

The combined program is

\[
\text{packet state}
\longrightarrow
\text{adaptive mechanism discovery}
\longrightarrow
(X,\eta)
\longrightarrow
\text{autonomous or periodically refreshed closure}.
\]

## Relation to an autonomous closure

The archive-Gram-guided packet propagator, once implemented, would remain
autonomous at the packet-ket level: every native and observer quantity is
evaluated from its current state and drive.  Compression to \((X,\eta)\) is a
subsequent model-discovery problem.  If a small set of packet or operator
mechanisms recurs, define preparation-dependent amplitudes \(\eta\) and their
reciprocal dynamics:

\[
\dot x=f_{31}(t,x)+Lr(\eta),
\qquad
\dot\eta=g(\eta,x,V).
\]

The autonomous model must then be tested in free rollout; accurate
teacher-forced tangent projection is not sufficient.  If no compact
autonomous realization emerges, the adaptive frame remains useful as an
interpretable compressed-wavefunction method and as a diagnostic of why the
archive hierarchy fails.

## Implementation gates

The mathematical specification above has not yet passed the following
implementation or numerical gates.

1. Reconstruct \(\mathcal G_K\) directly from the packet ket and independently
   from the contracted \((\rho,N,A,C)\) blocks, then verify entrywise agreement.
2. Verify \(\Gamma_K^{\mathbb R}\succeq0\), the generalized Schur-complement
   identity, the exact geometric ranks, and the ranks retained by each declared
   regularization.
3. Compare the analytic \(\dot{\mathcal G}^{(v)}_K\) with centered finite
   differences for both velocities, explicitly evaluating the \(\dot Y\) terms
   and their cancellation for the fixed observer frame.
4. At frozen packet states, search candidates across electronic sectors and
   coherent displacements.  Record ordinary residual reduction, archive-Gram
   rate-defect reduction, dynamically weighted observer coverage, tangent-rank
   gain, conditioning, velocity norm, and resource cost.
5. Implement two-stage packet birth, then perform merge and prune ablations
   against otherwise matched propagation.
6. Compare the adaptive method with the locally available fixed-capacity
   continuations:
   output/local_runs/paper_v_packet_derived_closure_source_k6_central_t40_20260804_v2/,
   output/local_runs/paper_v_multi_coherent_capacity_k8_t40_20260804_v1/,
   output/local_runs/paper_v_multi_coherent_capacity_k10_t40_20260804_v1/,
   and
   output/local_runs/paper_v_multi_coherent_capacity_k12_t40_20260804_v1/.
   Their presence supplies comparators; it does not validate the adaptive
   controller.
7. Test multiple preparations and at least one distinct drive.  Score all 31
   retained coordinates, site occupations, electronic, phonon,
   electron--phonon, and total energies, physicality margins, missing-source
   amplitudes, archive-Gram margins, tangent ranks, residuals, and
   computational cost.
8. Freeze the candidate-generation, admission, activation, merge, prune,
   regularization, and stopping policies before any held-out exact-reference
   score.
9. Begin an autonomous \((X,\eta)\) reduction only after the adaptive packet
   parent model passes its held-out propagation gate.

## Required future theory review

A future theory review should audit:

- the correct real/complex McLachlan formulation and gauge treatment;
- symmetry-complete but nonredundant mixed-operator pools;
- regularization and whitening that preserve the physical tangent metric;
- residual-reduction and observability criteria for adaptive selection;
- conditions under which recurrent selected tangents define a finite hidden
  realization; and
- scaling and stopping evidence stated as accuracy--cost curves rather than
  arbitrary hard dimension caps.
