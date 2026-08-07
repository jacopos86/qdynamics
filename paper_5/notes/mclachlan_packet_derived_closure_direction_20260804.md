# McLachlan-packet-derived closure direction

Status: working direction to refine; not an accepted closure or performance
claim.

## Objective

Use the sufficiently accurate multi-coherent McLachlan propagation as a
structured source of the preparation, phase, and phonon-branching information
missing from the 31-coordinate moment equations.  The immediate objective is
not to replace one fitted source trajectory with another.  It is to determine
whether a small reachable and observable subset of the packet dynamics can be
distilled into an autonomous extension of the retained moment state.

## Packet representation and contraction

Write the electron-conditioned multi-coherent state as

\[
|\Psi_{\mathrm{MC}}(t)\rangle
=
\sum_e\sum_{k=1}^{K_e}
c_{ek}(t)|e\rangle|\boldsymbol\alpha_{ek}(t)\rangle,
\qquad
|\boldsymbol\alpha_{ek}\rangle
=
\bigotimes_q|\alpha_{ekq}\rangle .
\]

McLachlan's variational principle determines the packet velocity through

\[
G(\theta)\dot\theta=b(\theta,V),
\qquad
\theta=\{\operatorname{Re}c,\operatorname{Im}c,
\operatorname{Re}\alpha,\operatorname{Im}\alpha\}.
\]

The packet state is contracted into the retained matrices

\[
X_{\mathrm{MC}}(t)=(\rho,B,N,A,C)_{\mathrm{MC}}(t).
\]

For example, with

\[
S_{e'l,ek}
=
\langle\boldsymbol\alpha_{e'l}|\boldsymbol\alpha_{ek}\rangle,
\qquad
O_{ij}^{e'e}=\langle e'|c_j^\dagger c_i|e\rangle,
\]

the connected electron--phonon correlation is

\[
C^q_{ij}
=
\sum_{e,e'}\sum_{k,l}
c_{e'l}^*c_{ek}
O_{ij}^{e'e}\alpha_{ekq}S_{e'l,ek}
-B_q\rho_{ij},
\]

for a normalized packet state.  This equation is the direct map from packet
weights and displacements to the correlation block used by the archive moment
equations.

## Missing-source construction

The packet-derived missing correlation source is

\[
r_{\mathrm{MC}}(t)
=
\dot C_{\mathrm{MC}}(t)
-F_{C,31}\!\left(t,X_{\mathrm{MC}}(t)\right),
\]

where \(F_{C,31}\) is the correlation block of the 31-coordinate moment
velocity.  The McLachlan tangent velocity supplies \(\dot C_{\mathrm{MC}}\)
analytically through the chain rule; finite differencing the sampled
correlation trajectory should not be the primary construction.

The established exact-source audit found that the corresponding
14-real-coordinate source is strongly compressible.  With the frozen scaled
SVD decoder,

\[
z=Q_5D_r^{-1}(r-\bar r),
\qquad
r_5=\bar r+D_rQ_5^{\mathsf T}z,
\]

five source directions explain \(0.99975425\) of the training variance and
reconstruct the three tested exact trajectories with normalized RMS residual
\(0.0156817\).  This is output compression, not evidence that five hidden
dynamical coordinates suffice.  The previous autonomous model with five
amplitudes and five rates worsened the coupled rollout despite reproducing the
teacher-forced source well.

## Proposed connection

The next construction should use the packet dynamics to identify the hidden
state that generates the source:

\[
\theta_{\mathrm{MC}}
\longrightarrow
r_{\mathrm{MC}}
\longrightarrow
z_{\mathrm{MC}}.
\]

Only packet-coordinate combinations that are both excited by preparation or
drive and visible in the five source outputs should be retained.  Denote those
combinations by \(\eta\).  The desired reduced model has the form

\[
\dot x=f_{31}(t,x)+L_Cr(\eta),
\qquad
\dot\eta=g(\eta,x,V),
\]

where \(L_C\) inserts the reconstructed source into the correlation velocity.
The coupling must be reciprocal: the retained state and drive evolve the
hidden coordinates, and the hidden coordinates feed the missing source back
into the retained dynamics.  Preparation-dependent initialization of
\(\eta\) must also be explicit.

The five-dimensional decoder \(Q_5\) may remain useful even when the minimal
causal state \(\eta\) has more than five coordinates.  Reachability and
observability, rather than another output SVD or damped-oscillator fit, should
determine its dimension.

## First mathematical and numerical gates

1. Contract matched McLachlan trajectories into \(X_{\mathrm{MC}}\) and their
   analytic velocities, then construct \(r_{\mathrm{MC}}\).
2. Compare the McLachlan and exact source subspaces using principal angles,
   exact-\(Q_5\) reconstruction error, and held-out source normalized RMS.
3. Check multiple preparations and at least one distinct drive.  Agreement on
   a single trajectory cannot establish a reusable hidden realization.
4. Linearize or otherwise interrogate the packet dynamics about the tested
   trajectories and retain only packet directions that are reachable from the
   declared inputs and observable through the missing source and physical
   observables.
5. Freeze the resulting autonomous reduced model before an exact-reference
   closed-loop score.  Score all retained blocks, site occupations, electronic,
   phonon, electron--phonon, and total energies, physicality margins, and
   preparation/drive holdouts.

## Scaling hypothesis

For \(M\) phonon modes and cutoff \(n_{\max}\), exact Fock-space storage grows
approximately as

\[
D_{\mathrm{exact}}
\sim D_{\mathrm e}(n_{\max}+1)^M,
\]

whereas an electron-conditioned \(K\)-packet state uses approximately

\[
P_{\mathrm{packet}}
\sim O(D_{\mathrm e}KM)
\]

parameters.  A coherent packet can represent a large displacement without
enumerating all occupied Fock levels.  The packet route therefore retains a
potential time and memory advantage if \(K\) remains modest.

McLachlan propagation still requires the tangent solve
\(G\dot\theta=b\).  A dense solve can scale cubically in the number of packet
parameters, so parameter compression alone is not a wall-clock result.  The
scalable implementation must evaluate coherent-state overlaps, Hamiltonian
matrix elements, and tangent matrices analytically without constructing the
truncated Fock ket or its full tangent vectors.

If the distilled hidden state has dimension \(H\ll P_{\mathrm{packet}}\), the
online state \((x,\eta)\in\mathbb R^{31+H}\) could approach the cost of the
archive moment equations while retaining phase and preparation memory.  If
the complete packet state must be propagated, the method remains a compressed
variational replacement rather than a recovered low-dimensional moment
closure.

## Physicality consequence

A normalized packet ket yields representable contracted moments by
construction.  Its electronic and joint moment Gram matrices are therefore
positive semidefinite up to numerical error.  Full packet propagation may not
need the separate minimum-norm representability controller.  A distilled
\((x,\eta)\) model no longer inherits that guarantee automatically and must be
checked, or constrained, in closed-loop propagation.

## Decision conditions

Continue toward a packet-derived autonomous closure if the McLachlan source
agrees with the exact source on held-out preparations and drives, a modest
reachable/observable hidden state reproduces the five source outputs, and the
frozen closed-loop model improves the retained moments and observables without
losing representability.

Treat the packet method only as an offline diagnostic or established
variational replacement if the five-source subspace does not transfer to the
McLachlan trajectories, the required packet or hidden-state dimension grows
rapidly, Fock-space construction remains necessary, or the reduced autonomous
rollout repeats the error amplification of the previous five-mode oscillator.

## Unresolved questions

- Does the McLachlan-derived missing source lie in the exact frozen
  five-direction decoder across preparations and drives?
- Which packet-coordinate combinations are jointly reachable and observable,
  and what is their minimal tested dimension?
- Does that dimension remain small as coupling, propagation horizon, mode
  count, and drive protocol change?
- Can all required packet contractions and tangent matrices be evaluated
  without Fock-space assembly?
- Does the reduced closed-loop model improve accuracy rather than merely
  reconstructing a teacher-forced source?

