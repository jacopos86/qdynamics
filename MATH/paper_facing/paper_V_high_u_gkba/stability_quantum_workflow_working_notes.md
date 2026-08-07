# Paper V Stability and Quantum-Algorithm Working Notes

Status: living research note. This is not manuscript prose, a validated
numerical result, or a claim that the target dynamics are chaotic.

## Paper identity

Paper V concerns equal-time non-Markovian electron--phonon dynamics for quantum
simulation. Its current working title is:

> Regularized Equal-Time Electron--Phonon Dynamics for Quantum Simulation

The immediate scientific problem is the apparent high-interaction instability
of the coupled electron--phonon equations. The longer-term problem is to
construct a defensible quantum or hybrid algorithm for the stabilized,
validated dynamics.

The work therefore has two connected but distinct parts:

1. **Stability workstream:** reduce the matrix equations to a controlled scalar
   model, reproduce the reported initial-condition-dependent instability,
   distinguish numerical divergence from genuine dynamical sensitivity, and
   identify a justified initialization or regularization.
2. **Quantum-algorithm workstream:** encode and propagate the validated
   equations using a quantum or hybrid algorithm. This workstream must not
   assume that an unstable or incorrectly initialized classical equation is a
   valid quantum-simulation target.

The stability workstream comes first. Both workstreams should share one
explicit mathematical model, notation, initial-state contract, invariant set,
and benchmark suite.

## Target equation set

The active Paper V draft transcribes Eqs. (14a)--(14e) from the
Riva--Simoni--Ping source. The dynamical variables are:

- electronic one-body density matrix \(\rho_{12}\);
- phonon covariance \(\delta\rho_{\mathbf q\mathbf q'}\);
- anomalous phonon covariance
  \(\delta\bar\rho_{\mathbf q'\mathbf q}\);
- electron--phonon correlation \(\delta\rho^{\mathbf q}_{12}\);
- coherent phonon amplitude \(B_{\mathbf q}\).

The present algorithmic split is:

- Eqs. (14a)--(14d), covering the electronic, phonon-covariance,
  anomalous-covariance, and electron--phonon-correlation sectors, form the
  proposed quantum-computer target;
- Eq. (14e), the coherent-field equation for \(B_{\mathbf q}\), remains a
  classical update coupled back through the effective electronic Hamiltonian
  \(\tilde h(t)\).

Before implementing a quantum algorithm, the \(L=2\) specialization should
determine the real scalar dimension, nonlinear products, conserved or bounded
quantities, stiffness, admissible initial conditions, and exact mechanism of
the high-\(U\) failure.

## Equation-numbering warning

The equation numbers in the two downloaded working PDFs do not match the
Riva--Simoni--Ping numbering used by the Paper V draft:

- In `Dynamics_on_the_Hubbard_DIMER.pdf`, Eq. (14) is the full
  Liouville--von Neumann density-matrix equation. The relevant scalar
  electron--phonon reductions appear later as Eqs. (78)--(82) and
  Eqs. (87)--(99).
- In `Electron_phonon_interactions___chiral_phonons.pdf`, Eq. (14) defines the
  anomalous two-phonon density matrix
  \(\bar\rho_{\mathbf q'\mathbf q}=\langle b_{\mathbf q}b_{\mathbf q'}\rangle\).
  Its equation of motion is Eq. (15), and the exact but unclosed
  electron--phonon correlation equation is developed in Eqs. (16)--(17).
- Therefore, references to “Eqs. (14a)--(14d)” in this project should mean the
  Riva--Simoni--Ping equal-time system transcribed in the Paper V manuscript,
  not Eq. (14) in either downloaded PDF.

## Evidence for sensitive initial conditions

### Hubbard-dimer document

`/Users/jakestrobel/Downloads/Dynamics_on_the_Hubbard_DIMER.pdf` explicitly
reports strong dependence on initial conditions.

On page 23, the Ehrenfest Hubbard--Holstein dimer is reduced to five real
scalar variables:

\[
\Delta n_{12},\qquad
\operatorname{Re}\rho_{12},\qquad
\operatorname{Im}\rho_{12},\qquad
\Delta\operatorname{Re}B_{12},\qquad
\Delta\operatorname{Im}B_{12}.
\]

The scalar equations, numbered (78)--(82) in that document, are

\[
\begin{aligned}
\Delta\dot n_{12}
  &=4t\,\operatorname{Im}\rho_{12},\\
\operatorname{Re}\dot\rho_{12}
  &=\operatorname{Im}\rho_{12}\,\Delta h^{\mathrm{ext}}
    +2g\,\operatorname{Im}\rho_{12}
      \Delta\operatorname{Re}B_{12},\\
\operatorname{Im}\dot\rho_{12}
  &=-t\Delta n_{12}
    -2g\,\operatorname{Re}\rho_{12}
      \Delta\operatorname{Re}B_{12}
    -\Delta h^{\mathrm{ext}}\operatorname{Re}\rho_{12},\\
\Delta\operatorname{Re}\dot B_{12}
  &=\omega_{\mathrm{ph}}\Delta\operatorname{Im}B_{12},\\
\Delta\operatorname{Im}\dot B_{12}
  &=-\omega_{\mathrm{ph}}\Delta\operatorname{Re}B_{12}
    -2g\Delta n_{12}.
\end{aligned}
\]

The document uses

\[
\gamma=\frac{\omega_{\mathrm{ph}}}{t},
\qquad
\lambda=\frac{2g^2}{\omega_{\mathrm{ph}}t},
\]

and studies \(\gamma=1/2\) with \(\lambda\in[0,2]\).

It reports different stationary initial conditions on the two sides of
\(\lambda=1\):

- for \(\lambda\leq1\),
  \[
  \Delta n_{12}=0,\quad
  \operatorname{Im}\rho_{12}=0,\quad
  \Delta\operatorname{Re}B_{12}=0,\quad
  \Delta\operatorname{Im}B_{12}=0;
  \]
- for \(\lambda>1\),
  \[
  \Delta n_{12}=\pm\sqrt{1-\lambda^{-2}},\quad
  \operatorname{Im}\rho_{12}=0,\quad
  \Delta\operatorname{Re}B_{12}
    =-\frac{2g}{\omega_{\mathrm{ph}}}\Delta n_{12},\quad
  \Delta\operatorname{Im}B_{12}=0.
  \]

In both regimes, \(\operatorname{Re}\rho_{12}\) is constrained by the stated
constant of motion

\[
1=\Delta n_{12}^2
  +4\left[
      \left(\operatorname{Re}\rho_{12}\right)^2
      +\left(\operatorname{Im}\rho_{12}\right)^2
    \right].
\]

The document then extends the scalar model with phonon populations and
electron--phonon correlations in Eqs. (87)--(99), described there as a
Fan--Migdal system.

The most direct reported instability comparison is Figure 34 on page 30:

- parameters: strong external field \(v/t_{\mathrm{hop}}=1\) and strong
  electron--phonon coupling \(\lambda=1.5\);
- unstable initialization: Hartree--Fock ground-state reduced electronic
  density matrix with all other correlation variables initialized to zero;
- stable comparison: the non-Markovian equations initialized from exact
  correlated data;
- reported outcome: exact correlated initial data removes the observed
  divergence, while the uncorrelated Hartree--Fock/zero-correlation
  initialization diverges.

On page 27, the document also proposes subtracting the initial right-hand side
from every equation:

\[
\frac{\mathrm df(t)}{\mathrm dt}
=F(f(t),t)-F(f(0),0).
\]

Figure 35 applies this modification with a Hartree--Fock initial state. The
document describes it as important in the linear limit and as suppressing the
divergence. At present this is a candidate correction, not a validated general
regularization: it changes the vector field and must be justified against the
source equations, equilibrium conditions, conservation laws, and exact
benchmarks.

### Chiral-phonon working document

`/Users/jakestrobel/Downloads/Electron_phonon_interactions___chiral_phonons.pdf`
derives the coupled phonon, anomalous-phonon, and electron--phonon correlation
hierarchy. It explicitly observes that the density-matrix, coherent-phonon,
phonon-covariance, and electron--phonon-correlation equations are not closed
without approximation. It also describes a stationary polaron initialization
obtained by imposing \(\dot B_\lambda(\mathbf q)=0\) and solving the electronic
density and coherent field self-consistently.

This document supports the importance of correlated/self-consistent
initialization and closure choice. In the inspected text, however, it does not
explicitly demonstrate the same initial-condition-driven divergence shown in
the Hubbard-dimer document.

## What is established and what is not

Established from the working documents:

- a scalar Hubbard-dimer reduction exists;
- the scalar equations have coupling-dependent fixed points;
- initializing a strong-coupling calculation at the wrong or uncorrelated
  state is associated with divergence;
- exact correlated initial conditions are reported to remove the divergence;
- subtracting \(F(f(0),0)\) is reported as another stabilizing modification.

Established by the local scalar harness:

- the five-variable Ehrenfest fixed points satisfy Eqs. (78)--(82) and the
  invariant in Eq. (86) to numerical precision;
- the thirteen-variable Eqs. (87)--(99), initialized with the Hartree--Fock
  electronic state and zero correlations, cross a state-magnitude threshold
  of \(10^4\) at \(t\approx130.47\,t_{\mathrm{hop}}^{-1}\) for
  \(\lambda=1.5\), \(\gamma=0.5\), and \(v/t_{\mathrm{hop}}=1\);
- RK4 refinement and converged DOP853, Radau, and BDF integrations agree on
  the failure time;
- the corresponding \(\lambda=0.5\) control remains bounded through
  \(t=140\,t_{\mathrm{hop}}^{-1}\);
- the Hartree--Fock/zero-correlation initial residual is nonzero only in two
  correlation components;
- removing either or both of those initial correlation-source products does
  not remove the driven divergence and instead advances it in this scalar
  transcription.

Not yet established:

- a representability-preserving evolution of the complete 31-real-coordinate
  invariant scalar closure;
- which complete-EOM root satisfies all electronic, bosonic-covariance, and
  higher-moment representability constraints while minimizing the energy;
- a positivity-preserving justification of the
  \(F(f(0),0)\)-subtracted modification for the complete matrix EOM;
- whether any residual instability remains after completing the closure and
  enforcing representability;
- that the trajectories are chaotic.

The word “chaotic” appears informally in the Hubbard-dimer document, but no
Lyapunov exponent, shadowing test, bounded attractor analysis, or
integrator-independent exponential-separation test is supplied. Sensitive
initial conditions alone do not establish chaos.

## Source-faithful audit and regularization update: 2026-07-28

The direct matrix implementation in
`paper_5/src/paper5/stability/matrix_reference.py` transcribes primary
Eqs. (14a)--(14e) for two electronic sites and two local phonon modes, including
the dimer's two-spin factors. The scalar embedding uses

\[
\rho=
\begin{pmatrix}
(1+\Delta n)/2 & \rho_{\mathrm R}+i\rho_{\mathrm I}\\
\rho_{\mathrm R}-i\rho_{\mathrm I} & (1-\Delta n)/2
\end{pmatrix},
\quad
B=(\Delta B/2,-\Delta B/2),
\]

\[
\delta\rho_{\mathrm{ph}}=
\begin{pmatrix}
n_{\mathrm{ph}} & \rho_{\mathrm{ph}}\\
\rho_{\mathrm{ph}} & n_{\mathrm{ph}}
\end{pmatrix},
\qquad
C^{(2)}=-C^{(1)},\qquad \operatorname{tr}C^{(1)}=0,
\]

with the six real correlation coordinates defined by Eqs. (102)--(107) of the
Hubbard-dimer document.

Across 100 seeded random states satisfying basic electronic and retained
phonon positivity, the largest component difference between the direct matrix
RHS projected onto these coordinates and scalar Eqs. (87)--(99) is at
floating-point roundoff. This verifies the signs, conjugations, spin factors,
and normalizations of the 13 retained derivatives.

This does **not** verify equivalence to the complete Eqs. (14a)--(14e). The
13-variable model sets the anomalous phonon covariance
\(\delta\bar\rho\) from Eq. (14c) to zero. A first-order step away from the
Hartree--Fock/zero-correlation state produces a nonzero Eq. (14c) derivative,
so this slice is not invariant under the complete matrix flow.

### Minimal Eq. (14c) extension

The dynamically generated anomalous matrix has the relative-mode form

\[
\delta\bar\rho
=\frac{m_-}{2}
\begin{pmatrix}
1&-1\\
-1&1
\end{pmatrix},
\qquad
m_-=\delta\bar\rho_{00}-\delta\bar\rho_{01}
=m_{\mathrm R}+i m_{\mathrm I}.
\]

Thus Eq. (14c) adds only two real coordinates to the 13D projection:

\[
\dot m_{\mathrm R}
=2\omega_{\mathrm{ph}}m_{\mathrm I}
+8g\,\delta_{\mathrm{Im}},
\qquad
\dot m_{\mathrm I}
=-2\omega_{\mathrm{ph}}m_{\mathrm R}
-8g\,\delta_{\mathrm{Re}}.
\]

The corresponding new Eq. (14d) feedback is

\[
\Delta\dot{\mathrm{Im}}^{+}
\mapsto
\Delta\dot{\mathrm{Im}}^{+}
+2g\,\operatorname{Im}\rho\,m_{\mathrm I},
\qquad
\Delta\dot{\mathrm{Re}}^{+}
\mapsto
\Delta\dot{\mathrm{Re}}^{+}
+2g\,\operatorname{Im}\rho\,m_{\mathrm R}.
\]

Across 100 additional seeded random states, these 15 derivatives agree with
the direct matrix projection to \(2.5\times10^{-16}\). The anomalous sector is
tangent to roundoff.

This still does not close the full matrix equations. The already assumed
relations \(C^{(2)}=-C^{(1)}\) and
\(\operatorname{tr}C^{(1)}=0\) acquire normal derivatives. The maximum such
normal residual in the same audit is approximately \(0.2165\). The 15D system
is therefore a verified projection with Eq. (14c), not a complete
scalarization.

The relevant one-mode bosonic second-moment condition is

\[
n_-(n_-+1)-|m_-|^2\ge0,
\qquad
n_-=n_{\mathrm{ph}}-\rho_{\mathrm{ph}}.
\]

Matched \(\lambda=1.5\), \(\gamma=0.5\), \(v/t_{\mathrm{hop}}=1\) protocols
give:

| 15D protocol | Amplitude threshold | Electronic PSD loss | Normal-phonon PSD loss | Boson uncertainty loss |
|---|---:|---:|---:|---:|
| Hartree--Fock plus zero correlations | \(\approx366.69\) | \(41.50\) | \(33.20\) | \(1.377\) |
| Exact ground-state contractions | \(132.70\) | \(1.019\) | \(129.49\) | \(6.139\) |
| Exact contractions plus Eq. (112) | none through \(400\) | none through \(400\) | none through \(400\) | \(3.682\) |

Adding Eq. (14c) therefore materially delays the HF amplitude threshold, but
physical second-moment failure still occurs much earlier. Eq. (112) keeps the
15D amplitudes bounded through \(t=400\), with maximum magnitude about
\(3.063\), while the bosonic uncertainty margin falls to about \(-5.43\).
This motivated a systematic search for every remaining normal sector.

### Complete 31-scalar invariant closure

Starting from the 15D tangent space, a deterministic invariant-subspace probe
evaluates the matrix RHS on random polynomial states, projects out the current
tangent space, and adds every independent normal direction identified by SVD.
The dimension sequence adds ranks

\[
2,\ 2,\ 2,\ 4,\ 3,\ 2,\ 1,
\]

and closes at 31 real coordinates inside the 44-real-coordinate packed matrix
representation. On a disjoint validation sample, the maximum normal residual
is below \(10^{-14}\).

The 31 coordinates have the explicit count

\[
\underbrace{3}_{\rho=\rho^\dagger,\ {\rm tr}\rho=1}
+\underbrace{4}_{B\in\mathbb C^2}
+\underbrace{4}_{N=N^\dagger}
+\underbrace{6}_{A=A^T}
+\underbrace{14}_{C^{(0)},C^{(1)},\
{\rm tr}C^{(0)}={\rm tr}C^{(1)}}
=31.
\]

This is the first scalar representation in the workflow that is tangent to
every generated direction of the complete two-site Eq. (14a)--(14e) matrix
flow. The 13D and 15D systems remain useful projections, but they are not
closed systems.

The full two-mode bosonic second-moment condition is

\[
\mathcal M_{\rm B}=
\begin{pmatrix}
N^T&A^*\\
A&I+N
\end{pmatrix}\succeq0.
\]

It reduces to \(n(n+1)-|m|^2\ge0\) for one mode.  The initially tested
separate electronic and bosonic marginal conditions give:

| 31D protocol | Amplitude threshold | Electronic PSD loss | Normal-phonon PSD loss | Full boson-moment loss |
|---|---:|---:|---:|---:|
| Hartree--Fock plus zero correlations | \(54.4613\) | \(27.96\) | \(10.68\) | \(1.33\) |
| Exact ground-state contractions | \(141.4056\) | \(1.04\) | \(19.15\) | \(3.69\) |
| Exact contractions plus Eq. (112) | none through \(400\) | none through \(400\) | \(1.49\) | \(1.49\) |

For the residual-subtracted 31D protocol, the maximum magnitude through
\(t=400\) is about \(2.2613\), the minimum electronic eigenvalue remains
positive at about \(0.00776\), but the normal-phonon and full boson-moment
minima reach approximately \(-0.7835\) and \(-1.2996\).

These marginal tests establish that loss of physical representability
precedes amplitude divergence and is initialization-dependent.  They do not
locate the earliest joint electron--phonon failure: the stronger Gram test
introduced below crosses at \(t=0.1607116782\) for the raw exact-contraction
closure, before either marginal matrix fails.  Eq. (112) removes the amplitude
divergence and preserves the electron density but does not preserve even the
bosonic marginal cone.

### First bosonic-boundary flux decomposition

For the 31D exact-contraction plus Eq. (112) protocol at phonon cutoff 16, the
first zero eigenvalue occurs at

\[
t_\partial=1.48695622\,t_{\mathrm{hop}}^{-1}.
\]

At this point the two lowest eigenvalues of \(\mathcal M_{\rm B}\) are
approximately \(0\) and \(0.19436\), so the crossing eigenvalue is simple and
its first derivative is

\[
\dot\lambda_{\min}
=v^\dagger\dot{\mathcal M}_{\rm B}v,
\qquad
\mathcal M_{\rm B}v=0.
\]

The signed equation-group contributions are:

| Contribution | \(v^\dagger\dot{\mathcal M}_{\rm B}^{(k)}v\) |
|---|---:|
| Eq. (14b), correlation source | \(-3.92638757\times10^{-3}\) |
| Eq. (14c), correlation source | \(+2.23204135\times10^{-6}\) |
| Eq. (14d), direct contribution | \(0\) |
| Eq. (112), residual subtraction | \(+1.32861458\times10^{-7}\) |
| Total | \(-3.92402267\times10^{-3}\) |

The two correlation branches in Eq. (14b) each contribute
\(-1.96319379\times10^{-3}\). The free Eq. (14b) rotation is zero for the
degenerate local phonon frequencies, and the free Eq. (14c) rotation projects
to zero at numerical precision. Direct summation reconstructs the full matrix
derivative with zero numerical residual at the reported precision. A centered
finite difference agrees within \(5\times10^{-10}\) under the tested solver
tolerances.

Thus the instantaneous outward flux in this tested regularized protocol comes
from the electron--phonon-correlation source in Eq. (14b). Equation (14c) and
the Eq. (112) correction oppose the crossing weakly. Equation (14d) cannot
contribute directly to this first derivative because
\(\mathcal M_{\rm B}\) depends only on \(N\) and \(A\); it acts indirectly by
creating the correlation field that appears in Eqs. (14b) and (14c).

### History-resolved Eq. (14d) attribution

Along the realized 31D trajectory, Eq. (14d) is linear in its correlation
field \(C\) when \(\rho\), \(N\), \(A\), and the coherent displacement are
prescribed. Write it as

\[
\dot C=L(t)C+\sum_k S_k(t).
\]

The initial correlation and each independent source were propagated through
the same time-ordered homogeneous evolution generated by \(L(t)\). Their sum
reconstructs the realized boundary correlation with error
\(5.2\times10^{-16}\); applying the linear Eq. (14b) flux functional to the
separate histories reconstructs its boundary flux with error
\(1.4\times10^{-16}\).

The causal-history contributions to the Eq. (14b) boundary flux are:

| Correlation history | Eq. (14b) boundary-flux contribution |
|---|---:|
| Eq. (112), correlation-sector subtraction | \(-1.96942674\times10^{-1}\) |
| Eq. (14d), bare Pauli-blocking source | \(+1.93016288\times10^{-1}\) |
| Eq. (14d), first anomalous source | \(-4.60900119\times10^{-4}\) |
| Eq. (14d), second anomalous source | \(+4.60893147\times10^{-4}\) |
| Eq. (14d), normal particle source | \(+2.30146227\times10^{-5}\) |
| Eq. (14d), normal hole source | \(-2.30243186\times10^{-5}\) |
| Propagated initial correlation | \(+1.45595687\times10^{-8}\) |
| Reconstructed Eq. (14b) flux | \(-3.92638757\times10^{-3}\) |

The absolute contributions sum to approximately \(0.39093\), whereas their
signed sum is only \(-0.0039264\): \(98.9956\%\) of their magnitude cancels.
The Eq. (112) correlation history and bare Pauli-blocking history alone
reproduce \(99.99995\%\) of the final signed flux. The two anomalous histories
sum to \(-6.97\times10^{-9}\), and the two normal-occupation histories sum to
\(-9.70\times10^{-9}\). Thus the paired nonlinear sources suspected in
Eq. (14d) are individually active but cancel at this crossing; they are not
the dominant net cause under this protocol.

The exact-contraction initial residual is itself overwhelmingly concentrated
in the correlation sector: its 31-coordinate sector norm is \(0.4096853\),
compared with \(7.66\times10^{-4}\) in the anomalous sector,
\(1.09\times10^{-4}\) in the coherent sector, and numerical zero in the
remaining sectors. Subtracting this large constant correlation residual is
therefore a strong intervention even though the direct Eq. (112) contribution
to \(\dot{\mathcal M}_{\rm B}\) at the crossing is small.

A one-sector ablation confirms the causal interpretation. Full Eq. (112)
crosses the bosonic boundary at \(t=1.48695622\). Retaining Eq. (112) in every
sector except the electron--phonon correlation coordinates delays the crossing
to \(t=3.68307711\), essentially the unregularized exact-contraction crossing
near \(t=3.68207\). Conversely, retaining only the correlation-sector
subtraction gives a crossing at \(t=1.48689983\).

This does not show that Eq. (112) is globally harmful: it still removes the
late amplitude blow-up and preserves electronic positivity in the tested
strong protocol. It shows that applying its full initial-residual subtraction
to Eq. (14d) advances this particular bosonic representability loss. The next
calculation should isolate the correlation-sector correction from the
amplitude-stabilizing correction and test a minimal cone-aware replacement.

The exact-reference implementation in
`paper_5/src/paper5/stability/exact_reference.py` diagonalizes the undriven
two-electron Holstein dimer and contracts its ground state into every matrix
variable. At \(\lambda=1.5\), \(\gamma=0.5\), and local phonon cutoff 16:

- exact ground-state energy:
  \(-3.7753265535\,t_{\mathrm{hop}}\);
- energy reconstructed from the contracted matrix variables:
  the same value to approximately \(10^{-14}\);
- norm of the approximate full-EOM RHS at those exact contractions:
  approximately \(0.3714\), confirming that exact contractions are not an
  exact fixed point of the truncated closure.

### Exact-projected derivative audit: the first defect is in Eq. (14d)

The exact wavefunction propagator now contracts both the moments and their
instantaneous derivatives.  This removes trajectory finite differences from
the comparison.  At each sampled exact state \(X_{\rm ex}(t)\), compare

\[
\dot x_{\rm ex}(t)
=\mathcal R\dot X_{\rm ex}(t)
\quad\hbox{with}\quad
F_{31}\!\left(t,\mathcal R X_{\rm ex}(t)\right).
\]

For the strong protocol at \(t=0.5\), after applying the proposed constant
initial-residual subtraction, the derivative discrepancy has norm
\(4.8511\times10^{-2}\).  Its correlation-coordinate norm is also
\(4.8511\times10^{-2}\), whereas the combined discrepancy in
\(\rho,B,N,A\) is below \(7\times10^{-6}\).  Thus the retained equations for
\(\rho,B,N,A\) give the exact derivatives when supplied with the exact
retained moments at this time.  The first wrong derivative is \(\dot C\); the
later errors in \(\rho,N,A\), the outward bosonic-boundary flux, and the
amplitude growth are downstream of that correlation error.

The error is not a phonon-cutoff artifact.  At \(t=0.5\), the
residual-subtracted derivative-defect norms for cutoffs 12, 16, and 20 are
approximately \(0.04610\), \(0.04851\), and \(0.04860\), respectively.  Nor
is the joint cone barrier its source: the defect is present when the archive
RHS is evaluated directly on the exact trajectory, before any barrier
correction is applied.

The coupling scan identifies the approximation order.  With
\(\gamma=0.5\), the defect increases from \(5.18\times10^{-5}\) at
\(\lambda=0.05\) to \(2.42\times10^{-3}\) at \(\lambda=0.5\) and
\(4.85\times10^{-2}\) at \(\lambda=1.5\).  Over the weak-coupling points its
growth is approximately \(g^{3.3}\).  This is consistent with the source's
stated construction of Eq. (14d): four-body
\(\langle c^\dagger c b^\dagger b\rangle\) correlators are factorized and
higher correlations are discarded when the hierarchy is truncated at second
order in \(g\).

For the dimer, write \(O_{ij}=c_j^\dagger c_i\) and
\(\delta X_r=\delta b_r+\delta b_r^\dagger\).  The exact interaction part of
\(\dot C^q_{ij}\) contains the mixed expectation

\[
\left\langle
 [O_{ij},n_{r\uparrow}]\,\delta X_r\,\delta b_q
\right\rangle .
\]

Equation (14d) replaces this object by products constructed from
\(\rho,N,A\).  The algebra-preserving product is

\[
Q^{qr}_{ij,\mathrm{fact}}
=\langle[O_{ij},n_{r\uparrow}]\rangle
 \langle\delta X_r\,\delta b_q\rangle,
\qquad
K^{qr}_{ij}=Q^{qr}_{ij}-Q^{qr}_{ij,\mathrm{fact}}.
\]

This definition matters because separately factorizing the two products inside
the commutator need not preserve the exact Pauli identity
\([n_i,n_r]=0\).  The source-faithful audit therefore separates three model
terms.  Between \(t=0\) and \(t=0.5\), the change in the genuine connected
\(K\) velocity has norm \(0.048149\), the fixed-sector Pauli-algebra repair has
norm \(0.007873\), and the omitted opposite-spin covariance has norm
\(0.005737\).  The finite-cutoff commutator remainder is only
\(8.68\times10^{-6}\).  These vectors partially cancel, so their norms must not
be added.  The earlier value \(0.04797\) grouped the Pauli-algebra term into an
effective ``archive \(K\)'' and is superseded by this operator-level split.  The
genuine evolving mixed remainder remains the largest individual source, while
a missing dimer spin factor is not the leading explanation.

This separates two meanings of “correction”:

1. The joint energy-neutral matrix barrier is a **representability
   correction**.  It changes the instantaneous velocity only when needed to
   keep \(\rho\), \(I-\rho\), and the full bosonic moment matrix inside their
   positive-semidefinite sets.  It does not reconstruct the discarded \(K\)
   dynamics.  The earlier \(t=1000\) result combined this barrier with a
   constant initial-residual subtraction, so that run establishes the
   stability of the composite regularization rather than the barrier alone.
2. An **accuracy correction** must preserve the exact fixed-sector Pauli
   algebra and supply information absent from \(x\in\mathbb R^{31}\).  The
   systematic route retains the complete next-order moment set containing
   \(K\) and the cross-spin covariance, derives its EOM, and closes one level
   higher.  A constant subtraction of the initial Eq. (14d) residual cancels
   the missing contribution only at \(t=0\); it cannot follow its driven
   evolution and is not a replacement for the higher moments.

Consequently, an autonomous 31-coordinate law cannot in general be made
exact by retuning coefficients: distinct full quantum states can share the
same retained 31 moments while having different \(K\), and therefore different
\(\dot C\).  A barrier-stabilized 31D trajectory must therefore be labeled as
a regularized second-order closure.  Exact strong-coupling agreement requires
the higher-moment extension rather than a stronger barrier.

### Why the first joint barrier failed, and the corrected solve

The first joint controller allowed correction velocities only in the three
independent entries of \(\rho\) and the ten entries of \((N,A)\).  It imposed
the two electronic matrix inequalities, the bosonic moment-matrix inequality,
and zero instantaneous correction-energy flux.  On the raw archive closure,
that restricted problem becomes infeasible near \(t=28.57\): a nonequilibrium
velocity can satisfy the three positive-semidefinite barriers, or it can be
energy-neutral within those thirteen controls, but at that state it cannot do
both.  Raising the optimizer iteration or constraint limit does not repair an
empty feasible set.

Allowing all 31 velocity coordinates removes that optimization infeasibility.
However, it is not yet the right physical correction.  It keeps \(\rho\) and
the bosonic moment matrix separately positive but leaves the cross-correlation
\(C\) unconstrained.  That run was stopped at its \(t=320\) checkpoint after
a stronger necessary representability test showed that it had already left
the physical joint moment set.

For a fixed spin, use the operator list

\[
Y=(\delta b_0,\delta b_1,\delta b_0^\dagger,\delta b_1^\dagger,
\delta\sigma_x,\delta\sigma_y,\delta\sigma_z).
\]

The Gram matrix \(\mathcal G_{ab}=\langle Y_a^\dagger Y_b\rangle\) is

\[
\mathcal G=
\begin{pmatrix}
\mathcal M_{\rm B}&Z(C)\\
Z(C)^\dagger&E(\rho)
\end{pmatrix}\succeq0,
\qquad
Z(C)=\begin{pmatrix}c^*\\c\end{pmatrix},
\qquad
c_{qa}=\operatorname{Tr}(C^q\sigma_a),
\]

where
\(E_{ab}(\rho)=\operatorname{Tr}(\rho\sigma_a\sigma_b)
-\operatorname{Tr}(\rho\sigma_a)\operatorname{Tr}(\rho\sigma_b)\).
This one matrix combines the retained bosonic, electronic, and
electron--phonon fluctuations.  Its positive semidefiniteness follows
directly from
\(z^\dagger\mathcal Gz=\langle(\sum_a z_aY_a)^\dagger
(\sum_bz_bY_b)\rangle\ge0\).

On the raw exact-contraction closure, \(\mathcal G\) first reaches zero at

\[
t=0.1607116782.
\]

At that time the separate electronic eigenvalues are approximately
\((0.10380,0.89620)\), and the smallest bosonic-moment eigenvalue is
\(1.8890\times10^{-3}\).  Thus the mixed correlation constraint fails much
earlier than either separate cone.  The exact truncated-Hamiltonian
trajectory keeps \(\mathcal G\succeq0\).  The closure also violates the exact
fixed-particle-number identity \(\operatorname{Tr}C^q=0\): at the initial
exact contractions its undriven RHS has
\(\operatorname{Im}\operatorname{Tr}\dot C^q=-0.0593937\).

The corrected controller therefore optimizes
\(u\in\mathbb R^{31}\) against the joint matrix itself:

\[
\begin{aligned}
\min_{u\in\mathbb R^{31}}\quad &\tfrac12\lVert u\rVert_2^2,\\
\text{subject to}\quad
&\dot\rho+\Delta\dot\rho(u)+
  \beta(\rho-h_\star I)\succeq0,\\
&-\dot\rho-\Delta\dot\rho(u)+
  \beta(I-\rho-h_\star I)\succeq0,\\
&\dot{\mathcal G}+\Delta\dot{\mathcal G}(u)
  +\beta(\mathcal G-h_\star I)\succeq0,\\
&\dot{\operatorname{Tr}C^q}
  +\Delta\dot{\operatorname{Tr}C^q}(u)=0,\\
&\nabla_x E(x)^{\mathsf T}u=0.
\end{aligned}
\]

At \(t=0\), the minimum-norm correction has norm \(0.059506\), changes the
joint barrier minimum from \(-2.9476\times10^{-3}\) to numerical zero,
cancels the correlation-trace velocity to roundoff, and contributes less than
\(3\times10^{-20}\) instantaneous energy flux.  Fixed-step RK4 refinements
\(\Delta t=0.01\) and \(0.005\) agree at \(t=2\) within
\(1.2\times10^{-6}\) in state-vector norm.  The finer trajectory keeps
\(\lambda_{\min}(\mathcal G)\ge2.0633\times10^{-5}\), while the raw closure
has already crossed the joint boundary.  The checkpointed raw-closure run of
this controller completed \(t=1000\) with fixed RK4 step \(0.01\): all
\(100{,}000\) steps and \(400{,}000\) corrected RHS evaluations finished
without a nonconverged solve,
\(\lambda_{\min}(\mathcal G)\ge2.06367\times10^{-5}\),
\(\lambda_{\min}(M_{\rm B})\ge2.06469\times10^{-5}\),
\(0.0239455\le\lambda(\rho)\le0.976055\),
\(\max|x_j|=2.09860\), and
\(\max|\operatorname{Tr}C^q|=8.78\times10^{-17}\).  The maximum post-pulse
energy drift from \(t=4\) was \(5.03\times10^{-6}\), and the maximum
correction-energy flux was \(2.64\times10^{-16}\).  The run summary and
checkpoint are under
`output/local_runs/paper_v_joint_moment_barrier_t1000_local_20260801_v1/`.
The Euclidean minimum is defined in the chosen 31-coordinate scaling; a
weighted metric would define a different smallest intervention.

### Exact/raw/corrected driven-horizon analysis and closure decision

The five-part diagnostic under
`output/local_runs/paper_v_electron_phonon_analysis_20260801_v3/` compares
the exact truncated Hamiltonian, the unmodified 31-coordinate closure, and
the joint-Gram-corrected closure from (t=0) through (t=4).  The baseline
uses ((\lambda,\gamma,V)=(1.5,0.5,1)), phonon cutoff 16, fixed corrected
RK4 step (0.01), and the same exact ground-state contractions for all three
trajectories.  Exact data are used only after propagation for errors and
certificates; the controller never queries them.

The raw and corrected physicality certificates separate stability from
accuracy:

| trajectory | (min\lambda(\mathcal G)) | (min\lambda(M_{\rm B})) | electronic eigenvalue range | (max|\operatorname{Tr}C^q|) |
|---|---:|---:|---:|---:|
| exact cutoff 16 | (2.8524\times10^{-5}) | (2.8549\times10^{-5}) | ([0.08338,0.91662]) | (1.06\times10^{-15}) |
| raw 31D closure | (-0.51727) | (-0.14777) | ([-0.02230,1.02230]) | (7.57\times10^{-2}) |
| joint barrier | (2.0637\times10^{-5}) | (2.0647\times10^{-5}) | ([0.02395,0.97605]) | (1.14\times10^{-16}) |

The (0.01)-spaced trajectory first samples the raw joint-cone exit at
(t=0.17), consistent with the independently located root
(t=0.1607116782).  The corrected trajectory remains representable over the
whole driven horizon.  Its block errors do not, however, follow the exact
dynamics:

| block | raw RMS error / exact dynamic RMS | corrected RMS error / exact dynamic RMS | corrected maximum Frobenius error |
|---|---:|---:|---:|
| (ho) | 0.6072 | 0.7613 | 0.1868 |
| (B) | 0.1076 | 0.2089 | 0.1153 |
| (N) | 6.5530 | 8.5389 | 0.3978 |
| (A) | 4.8673 | 4.6735 | 0.8951 |
| (C) | 2.5749 | 2.1250 | 0.2551 |

Thus the barrier lowers the (C) and (A) errors but increases the
(ho,B,N) errors.  It selects the nearest admissible velocity in the
chosen coordinate metric; it is not an estimator of the omitted exact
velocity.

The controller decomposition makes the mechanism explicit.  After removing
the minimum-norm energy- and trace-equality correction, the integrated
squared norm of the additional cone action is distributed as

[
  C:69.18\%,\qquad \rho:23.37\%,\qquad A:4.52\%,\qquad
  B:2.14\%,\qquad N:0.80\%.
]

The weakest joint-barrier eigenmode is (62.97\%) electronic and
(37.03\%) bosonic on the correction-weighted average.  Its largest
individual weights are (delta\sigma_z) (34.11%),
(delta\sigma_y) (24.51%), and the two phonon-annihilation entries
(30.24% together).  The electronic lower and upper barriers remain inward;
the joint Gram barrier is the active sector at every baseline sample.

On the exact trajectory, the residual-subtracted derivative defect is
dominated by (C) at 400 of 401 samples.  Its (C)-block RMS norm is
0.16669 and its maximum is 0.27431; every other block has maximum defect
below (2.32\times10^{-4}), while the (ho) defect is at roundoff.
At samples where the (C) defect exceeds 1% of its maximum, the additional
cone correction has mean cosine 0.374 with the negative defect and positive
alignment at every sample.  It nevertheless supplies only 0.0956 of the
missing-defect direction on average.  The barrier therefore partially
opposes the consequence of discarding (K), without reconstructing (K)'s
velocity contribution.

The discrepancy is numerically resolved.  Refining the corrected step from
0.01 to 0.005 changes (C) by at most (1.30\times10^{-6}).  Raising the
exact phonon cutoff from 16 to 20 changes (C) by at most
(3.19\times10^{-4}), whereas the corrected (C) error reaches 0.2551.
The model error is therefore 799 times the larger numerical floor.  The
cutoff-12-to-16 (C) difference is (1.24\times10^{-2}), confirming that
12, 16, and 20 are a convergence study and that 12 should not replace 16 as
the working cutoff.

A 12-point Cartesian check used
(lambda\in\{0.5,1.5\}),
(gamma\in\{0.25,0.5,1\}), and
(V\in\{0.5,1\}), with cutoff 16 and step 0.02.  The raw joint Gram matrix
became negative at every point.  The corrected Gram matrix stayed
nonnegative to the declared numerical tolerance at every point.  For weak
coupling, the corrected (C)-error ratio ranges from 0.324 to 0.792 and the
maximum correction norm from 0.0165 to 0.0715.  For strong coupling, those
ranges are 1.812 to 4.698 and 0.312 to 0.574.  All 12 corrected (C)-error
ratios exceed the predeclared 0.1 materiality threshold.

The predeclared closure-extension rule required both a (C)-error ratio
above 0.1 and a model error more than ten times the numerical floor.  The
measured values are 2.125 and 799, respectively.  The decision is therefore
to retain the mixed moment (K^{qr}_{ij}) at the next closure level, enforce the
fixed-sector Pauli algebra exactly, and include the opposite-spin covariance.
The equations of motion must first determine the complete same-order moment
set; the seven quantities visible in the instantaneous defect are not by
themselves an autonomous state.  The existing joint Gram barrier remains the
admissibility condition for an enlarged model; it does not substitute for the
missing-moment dynamics.

### Exact mixed-moment attribution and the next hierarchy level

The reusable exact-reference audit evaluates the operator identity

\[
\dot C_{\rm ex}
=\dot C_{31}
+\Delta\dot C_K
+\Delta\dot C_{\rm P}
+\Delta\dot C_{\uparrow\downarrow}
+\Delta\dot C_{\rm cut}.
\]

Here \(\Delta\dot C_K=-ig\sum_r(K^{qr})\),
\(\Delta\dot C_{\rm P}\) restores the exact one-up-electron Pauli algebra,
\(\Delta\dot C_{\uparrow\downarrow}\) is the connected covariance with the
opposite-spin site occupation, and \(\Delta\dot C_{\rm cut}\) measures the
failure of the finite oscillator matrix to obey the infinite-dimensional
canonical commutator exactly.  This identity is checked directly against the
Schrödinger derivative at every sampled time; the exact trajectory remains a
reporting input and is not used by an autonomous controller.

For the strong baseline over \(0\leq t\leq4\), the residual-subtracted
correlation-velocity errors are

| supplied source terms | RMS \(\ell_2\) norm | maximum \(\ell_2\) norm |
|---|---:|---:|
| archive Eq. (14d) | 0.166690 | 0.274310 |
| archive + exact \(K\) | 0.089345 | 0.137864 |
| archive + exact \(K\) + Pauli repair | 0.026160 | 0.038268 |
| archive + exact \(K\) + Pauli repair + opposite spin | \(2.53\times10^{-5}\) | \(3.47\times10^{-5}\) |

Thus the three physical model terms reduce the RMS defect by a factor of about
\(6.58\times10^3\); the remaining curve is the measured cutoff commutator
remainder.  The separate steps are not monotone because the omitted source
vectors partially cancel in the archive equation.

The dimer symmetry makes the instantaneous tensors much smaller than their raw
array shapes suggest.  Let \(b_+=(b_0+b_1)/\sqrt2\) and
\(b_-=(b_0-b_1)/\sqrt2\).  Fixed total electron number drives \(b_+\) only as
an independent coherent oscillator, so its centered fluctuations decouple.
With \(s_0=+1\) and \(s_1=-1\), the infinite-cutoff relations are

\[
K^{1r}=-K^{0r},\qquad K^{q0}=K^{q1},\qquad
K^{qr}_{00}=K^{qr}_{11}=0.
\]

Only \(K_{01}\) and \(K_{10}\) remain complex, giving four real coordinates.
For the opposite-spin covariance
\(D^q_{ij}=\operatorname{Cov}(O_{ij},n_{q\downarrow})\),

\[
D^1=-D^0,\qquad D^0=(D^0)^\dagger,\qquad \operatorname{tr}D^0=0,
\]

so the sampled block has three real coordinates.  The small violations of the
\(K\) mode relations converge away with phonon cutoff: the maximum residuals
for cutoffs 12, 16, and 20 are respectively about
\(1.8\times10^{-3}\), \(9.6\times10^{-5}\), and
\(1.1\times10^{-6}\).  The corresponding cutoff-velocity changes are
\(2.74\times10^{-4}\), \(8.38\times10^{-6}\), and
\(8.51\times10^{-8}\).

These seven sampled coordinates do **not** form a closed 38-coordinate ODE.
Writing the electronic variables as Pauli components makes the obstruction
explicit.  Hopping rotates
\(D_{\mu z}=\operatorname{Cov}(\sigma_\mu^\uparrow,
\sigma_z^\downarrow)\) into \(D_{\mu y}\), so autonomous propagation requires
the rest of the spin-symmetric cross-spin covariance matrix.  The
electron--phonon interaction then introduces connected two-electron/one-phonon
moments.  Likewise, differentiating \(K\) introduces one-electron/three-phonon
and two-electron/one-phonon connected moments.  Appending only \(K\) and
\(D_{\mu z}\) would therefore hide a new factorization at the very line where
the previous factorization failed.

The next source-faithful candidate is a symmetry-adapted third-cumulant
closure: retain the complete same-order relative-mode moment set, factorize
the next cumulant order explicitly, and then apply the joint Gram barrier as a
separate representability safeguard.  The exact audit provides the red test:
the enlarged autonomous RHS must reduce the converged \(C\)-velocity defect
without consulting the exact trajectory online.

The retrievable run is
`output/local_runs/paper_v_mixed_moment_attribution_20260801_v1/`; its runtime
manifest verifies the source, summary, trajectory, and attribution-plot hashes.

### Complete third-order moment closure: lower equations pass, terminal closure fails

The next candidate was derived directly from the fixed one-up/one-down
Holstein Hamiltonian rather than by appending selected tensors to the archive
state.  With

\[
b_\pm=\frac{b_0\pm b_1}{\sqrt2},\qquad
x=\frac{b_-+b_-^\dagger}{\sqrt2},\qquad
p=\frac{b_--b_-^\dagger}{i\sqrt2},
\]

the centered fluctuations of the \(+\) mode decouple and its coherent
amplitude obeys

\[
\dot B_+=-i(\omega B_++\sqrt2g).
\]

The interacting Hamiltonian, up to an irrelevant constant, is

\[
H_-=-t(\sigma_x^\uparrow+\sigma_x^\downarrow)
+\frac{V(t)}2(\sigma_z^\uparrow+\sigma_z^\downarrow)
+\frac{\omega}{2}(x^2+p^2)
+g(\sigma_z^\uparrow+\sigma_z^\downarrow)x.
\]

The autonomous state retains every spin-exchange-symmetric Hermitian Weyl
moment

\[
\left\langle
\sigma_\mu^\uparrow\sigma_\nu^\downarrow
\,\mathcal W(x^a p^b)
\right\rangle,
\qquad
\mathbf 1_{\mu\ne I}+\mathbf 1_{\nu\ne I}+a+b\leq3.
\]

There are 5 degree-one, 15 degree-two, and 25 degree-three moments.  Together
with the two real coordinates of \(B_+\), this gives 47 real coordinates.  The
Pauli products are reduced exactly, bosonic products use the Weyl/Moyal
algebra, and only a generated degree-four moment is approximated: its
connected fourth cumulant is set to zero and its raw value is reconstructed
from retained cumulants of orders one through three.

An exact-reference adapter contracts the truncated wavefunction and its
Schrödinger derivative into precisely the same 47 coordinates.  It is an
offline oracle and is absent from the autonomous right-hand side.  In the
decoupled \(g=0\) control, the maximum component derivative defect through
\(t=2\) is \(3.13\times10^{-16}\), verifying the transformed Hamiltonian,
Pauli algebra, Weyl convention, and coordinate map together.

For the strong baseline over \(0\leq t\leq4\), using cutoff 20, the
exact-state derivative audit gives

| retained block | defect RMS \(\ell_2\) norm | exact velocity RMS \(\ell_2\) norm | ratio |
|---|---:|---:|---:|
| degree one | \(2.25\times10^{-7}\) | \(5.8077\times10^{-1}\) | \(3.87\times10^{-7}\) |
| degree two | \(8.11\times10^{-6}\) | \(6.7717\times10^{-1}\) | \(1.20\times10^{-5}\) |
| degree three | \(9.1692\times10^{-1}\) | \(1.2557\) | \(7.3020\times10^{-1}\) |

The degree-one and degree-two discrepancies are the finite-cutoff canonical-
commutator floor.  Their convergence is explicit at \(t=0\) and \(t=0.5\):
the maximum degree-two defect falls from \(2.063\times10^{-2}\) at cutoff 12
to \(7.059\times10^{-4}\) at 16 and \(7.943\times10^{-6}\) at 20.  The
degree-three defect instead remains about \(0.30\) at \(t=0.5\), so it is a
closure error rather than a cutoff artifact.

Projecting the exact-state 47-coordinate derivatives back to
\((\rho,B,N,A,C)\) confirms why the enlarged state was the correct diagnostic
move.  The \(C\)-velocity defect has RMS \(1.01\times10^{-6}\), compared with
\(1.6669\times10^{-1}\) for the archive closure.  Thus the complete
third-order state supplies the missing lower-level dynamics.  Its own
terminal equations are the new bottleneck: the zero-fourth-cumulant rule has
relative RMS defect \(0.7302\), above the declared \(0.1\) gate, and produces
an undriven initial residual of \(0.21147\) in the exact ground-state moments.

Raw 47-coordinate propagation and joint-Gram barrier adaptation are therefore
deferred.  A barrier could keep a trajectory representable but could not make
this inaccurate terminal velocity source-faithful.  The next hierarchy test
should retain all degree-four moments (35 additional symmetry-adapted
coordinates) and measure the degree-four terminal defect from a declared
fifth-cumulant rule before any long propagation.

The retrievable audit is
`output/local_runs/paper_v_third_cumulant_gate_20260801_v1/`; its manifest
hashes the implementation, exact adapter, summary, trajectory arrays, and
derivative-gate plot.

### Complete fourth-order hierarchy: the defect moves to the fifth cumulant

The next gate retains all 35 symmetry-adapted degree-four moments in addition
to the 47-coordinate state above.  The resulting state has

\[
2+(5+15+25+35)=82
\]

real coordinates.  Its Pauli and Weyl products are still evaluated exactly;
only the degree-five moments generated when differentiating the degree-four
block are reconstructed by setting the connected fifth cumulant to zero.
Thus this is a direct test of the proposed next closure, not a long-time
propagation or a representability correction.

The implementation first passes two structural controls.  Its maximum
component derivative error in the decoupled \(g=0\) problem is
\(4.07\times10^{-16}\).  If its degree-four coordinates are initialized from
the third-order zero-cumulant reconstruction, its first 47 derivatives agree
with the previous implementation to roundoff.  These controls verify that the
82-coordinate result is a genuine hierarchy extension rather than an equation
or coordinate-ordering change.

For the strong baseline over \(0\leq t\leq4\), at phonon cutoff 20, the
exact-state derivative audit gives

| retained block | defect RMS \(\ell_2\) norm | exact velocity RMS \(\ell_2\) norm | ratio |
|---|---:|---:|---:|
| degree one | \(2.25\times10^{-7}\) | \(5.8077\times10^{-1}\) | \(3.87\times10^{-7}\) |
| degree two | \(8.11\times10^{-6}\) | \(6.7717\times10^{-1}\) | \(1.20\times10^{-5}\) |
| degree three | \(6.84\times10^{-6}\) | \(1.2557\) | \(5.45\times10^{-6}\) |
| degree four | \(3.3465\) | \(1.5729\) | \(2.1276\) |

Retaining degree four therefore does exactly what the hierarchy predicts for
the lower equations: the former degree-three ratio falls from \(0.7302\) to
the cutoff floor, \(5.45\times10^{-6}\).  The approximation has not converged,
however.  The newly exposed terminal rule---zero connected fifth
cumulant---has relative RMS defect \(2.13\), exceeding the declared \(0.1\)
gate by more than a factor of twenty and exceeding the previous terminal
relative defect.

This failure is not caused by the finite phonon cutoff.  At \(t=0\) and
\(t=0.5\), increasing the cutoff from 12 to 16 to 20 reduces the maximum
degree-three defect from \(7.02\times10^{-3}\) to
\(1.96\times10^{-4}\) to \(1.86\times10^{-6}\).  Over the same sequence, the
degree-four terminal defect approaches the nonzero values \(3.50\), \(3.74\),
and \(3.74\).  The largest failed components combine an electronic
\(x\)- or \(y\)-Pauli factor with three powers of the relative phonon
displacement, especially \(\langle\sigma_y x^3\rangle\) and
\(\langle\sigma_x x^3\rangle\).  The exact ground-state coordinates already
give a degree-four residual norm of \(3.7404\) before the pulse begins.

Raw 82-coordinate propagation and barrier adaptation are therefore deferred.
A barrier would alter the inaccurate terminal velocity but would not repair
its missing fifth-order electron--phonon correlations.  Blindly adding the 45
degree-five moments would merely move the untested factorization to the sixth
cumulant.  The next scientific decision should instead compare physically
adapted terminal closures---for example, conditional displaced-Gaussian or
polaron-frame cumulants---against the same exact-state derivative gate.

The retrievable audit is
`output/local_runs/paper_v_fourth_cumulant_gate_20260801_v1/`; its manifest
hashes the generalized hierarchy implementation, exact adapter, summary,
trajectory arrays, and derivative-gate plot.

### Electron-conditioned Gaussian closure: useful structure, insufficient accuracy

The first physically adapted closure reconstructs an electronic moment matrix
for each relative-mode Weyl monomial,

\[
\Gamma_{ab}
=\operatorname{Tr}_{\rm ph}
\left[\varrho_{\rm ep}\,\mathcal W(x^a p^b)\right].
\]

The retained matrices through phonon degree two determine Hermitian
electron-conditioned displacement operators \(D_x,D_p\) and covariance
operators \(V_{xx},V_{xp},V_{pp}\).  Writing
\(J_Q(M)=(QM+MQ)/2\), the Gaussian recurrence has the form

\[
\widehat\Gamma_{a+1,b}
=J_{D_x}(\widehat\Gamma_{ab})
+aJ_{V_{xx}}(\widehat\Gamma_{a-1,b})
+bJ_{V_{xp}}(\widehat\Gamma_{a,b-1}),
\]

with the analogous \(p\)-recurrence.  Averaging the two paths for mixed
monomials preserves Hermiticity.  This construction allows the relative
phonon displacement and squeezing to depend on the electronic state while
using only the 82 retained coordinates.  The exact wavefunction remains an
offline validation oracle.

The electronic density has support rank three along the strong reference
trajectory, as required by spin-exchange symmetry.  The Jordan solves remain
well resolved: their maximum relative residual is
\(6.76\times10^{-14}\), and the minimum electronic eigenvalue differs from
zero only by \(1.67\times10^{-16}\).  The decoupled \(g=0\) control has maximum
component derivative error \(4.07\times10^{-16}\).

At cutoff 20 over \(0\leq t\leq4\), the closure changes the two relevant gates
as follows:

| gate | zero fifth cumulant | conditioned Gaussian |
|---|---:|---:|
| required fifth-moment relative RMS defect | \(0.3389\) | \(0.2446\) |
| degree-four derivative relative RMS defect | \(2.1276\) | \(1.5975\) |

Electronic conditioning therefore removes about one quarter of each defect.
It does not approach the declared derivative threshold \(0.1\).  At
\(t=0\) and \(t=0.5\), the adapted derivative ratio converges from \(2.271\)
at cutoff 12 to \(2.163\) at 16 and \(2.157\) at 20, establishing that the
remaining error is not a cutoff effect.  Even an offline exact-fitted scalar
blend of the zero-cumulant and conditioned-Gaussian velocities has optimal
weight \(0.594\) and relative defect \(0.912\); that diagnostic is excluded
from autonomous evolution.

The dominant missed input is
\(\langle\sigma_x x^4\rangle\), and the largest resulting velocity error is in
the derivative of \(\langle\sigma_y x^3\rangle\).  A single displaced and
squeezed transition kernel per electronic sector therefore captures part of
the polaronic displacement but misses the conditional phonon non-Gaussianity
that controls these coherences.  Raw 82-coordinate propagation remains
deferred.  The next representation test should measure whether the exact
electron-conditioned phonon blocks require multiple Gaussian packets or a
larger non-Gaussian basis before another autonomous closure is constructed.

The retrievable audit is
`output/local_runs/paper_v_conditional_closure_gate_20260801_v1/`; its manifest
hashes the closure, hierarchy, exact adapter, summary, trajectory arrays, and
comparison plot.

### Exact conditional packet geometry

The failed conditioned-Gaussian closure motivates a representation test before
another moment closure.  The exact local phonon amplitudes were transformed to
the center and relative modes with the finite-cutoff beam-splitter isometry.
For each of the four electronic site configurations, tracing the center mode
leaves an almost pure relative-mode state.  At cutoff 20 over
(0\leq t\leq4), the minimum center--relative factorization is
(0.999999879) and the minimum conditional purity is (0.999999757).
Spin-exchanged electronic configurations agree in Frobenius norm to better
than (5\times10^{-14}).

Each conditional block was fitted first by one displaced-squeezed Gaussian
and then by the optimal span of two coherent states.  The strongest results
are

| diagnostic | result |
|---|---:|
| largest relative-entropy non-Gaussianity | (0.59555) |
| worst one-Gaussian infidelity | (5.884\times10^{-2}) |
| worst two-coherent infidelity | (1.117\times10^{-2}) |
| largest two-packet fidelity gain | (5.341\times10^{-2}) |
| largest resolved Husimi-peak count | 1 |

The phonon state is therefore not splitting into two well-separated classical
lobes.  It remains a single broad phase-space feature with non-Gaussian shape;
two overlapping coherent packets provide a useful compression basis.  At the
decisive times (t=0,2,4), cutoff 16 agrees with cutoff 20 to
(8.32\times10^{-4}) or better in every reported scalar metric, and the
minimum conditional-state fidelity between those cutoffs is (0.9999978).
Cutoff 12 is retained as the coarser control.

This compression gate does not define dynamics.  In particular, fitting
packet centers from the exact state at every time would leak the exact oracle
into the model.  The result instead authorizes a separate autonomous
wavefunction-parameter test.

The retrievable packet audit is
`output/local_runs/paper_v_conditional_packet_gate_20260801_v1/`.

### Autonomous electron-conditioned multi-coherent representation

The physically adapted state is the joint electron--relative-mode ket

\[
|\psi(\theta)\rangle
=\sum_{s=0}^{3}\sum_{k=1}^{K}
c_{sk}|s\rangle|\alpha_{sk}\rangle .
\]

The center oscillator remains in its exactly decoupled displaced state.  The
real parameter velocity is obtained autonomously by minimizing the
McLachlan residual between the ansatz tangent (J(\theta)\dot\theta) and the
phase-fixed Schrödinger velocity.  Exact wavefunctions are used only to fit
offline gate states and score their moment and velocity defects; they are not
inputs to the autonomous right-hand side.

A direct comparison of packet counts at cutoff 20 over (0\leq t\leq4)
gives

| packets per electronic configuration | minimum state fidelity | maximum nonstationary exact-tangent defect | autonomous 82-coordinate derivative RMS defect |
|---:|---:|---:|---:|
| 2 | (0.993594) | (0.30477) | (0.59210) |
| 3 | (0.999064) | (0.19630) | (0.12670) |
| 4 | (0.999881) | (0.04890) | (0.03138) |

Thus state fidelity alone would have selected too small a model: two packets
compress well but fail the velocity gate, and three packets still exceed the
declared (0.1) thresholds.  Four packets are the smallest tested family that
passes both.  Its cutoff-16 and cutoff-20 derivative RMS defects are
(0.03325) and (0.03128), respectively.  The decoupled one-packet control
has derivative RMS defect (9.00\times10^{-6}).

The first four-packet implementation used a hard relative singular-value
cutoff in the tangent pseudoinverse.  It passed the local gate and reached
(t=4.5), but the next half-unit segment exceeded 30 seconds.  The hard cutoff
changes tangent rank discontinuously when a singular value crosses the
threshold, so the adaptive ODE solver encountered a discontinuous parameter
velocity.  The last hard-cutoff checkpoint remained physical and had exact
state fidelity (0.989745), but this is a coordinate-stiffness failure, not a
long-horizon result.

The corrected tangent solve uses smooth Tikhonov filtering.  If
(J_{\mathbb R}=U\operatorname{diag}(s_i)V^{\mathsf T}) and (b) is the
realified Schrödinger velocity, then

\[
\dot\theta
=V\operatorname{diag}\!\left(
\frac{s_i}{s_i^2+(\eta s_{\max})^2}
\right)U^{\mathsf T}b,
\qquad \eta=0.003.
\]

This is a smooth minimum-residual coordinate metric; it does not project or
overwrite the physical ket.  Its formal four-packet gate has maximum exact
tangent defect (0.04890), 82-coordinate derivative RMS defect (0.03138),
and maximum parameter-speed norm (4.96).

The no-feedback autonomous trajectory then gives

| horizon | final state fidelity | 82-coordinate error | maximum norm drift | minimum electronic eigenvalue | minimum relative uncertainty margin |
|---:|---:|---:|---:|---:|---:|
| (t=4) | (0.999022) | (0.01783) | (1.69\times10^{-5}) | (0.08356) | (0.35556) |
| (t=20) | (0.918291) | (0.26669) | (5.54\times10^{-4}) | (0.03212) | (0.35557) |

All 40 half-unit segments through (t=20) complete, with at most 713 right-
hand-side evaluations per segment.  The smooth regulator therefore fixes the
parameter-coordinate freeze and the wavefunction representation preserves
physicality by construction.  Quantitative accuracy nevertheless degrades:
the maximum 82-coordinate error through (t=20) is (0.30325).  Four packets
are a validated short/intermediate-time representation, not a long-horizon
replacement.  A (t=140) or (t=1000) claim is not authorized.

The next representation study, if pursued, should compare adaptive packet
spawning or a larger (K) with a gauge-fixed/orthogonalized tangent metric.
The alternative dimer-specific baseline is direct relative-mode Fock
propagation, which is representability preserving but does not provide a
portable moment closure.

Retrievable artifacts are:

- `output/local_runs/paper_v_multi_coherent_tikhonov_gate_20260801_v1/`;
- `output/local_runs/paper_v_multi_coherent_tikhonov_trajectory_20260801_v1/`;
- `output/local_runs/paper_v_multi_coherent_tikhonov_segmented_t20_20260801_v1/`.

### Earliest failure is positivity, not amplitude growth

For the pinned 13-scalar Hartree--Fock/zero-correlation trajectory:

- the electronic 1-RDM first loses positive semidefiniteness at
  \(t\approx15.5177\);
- the state magnitude does not reach \(10^4\) until
  \(t\approx130.47\).

The amplitude blow-up is therefore downstream of an earlier exit from the
basic physical density-matrix domain. This is stronger evidence for closure or
initialization failure than the late-time Jacobian growth by itself.

### Matched scalar initialization results at \(\lambda=1.5\)

| Protocol | Initial residual | Amplitude outcome | First electronic PSD loss |
|---|---:|---:|---:|
| Hartree--Fock plus zero correlations | \(0.3423\) | threshold at \(130.47\) | \(15.5177\) |
| Exact-seed-connected stationary root | \(<10^{-14}\) | bounded through \(400\) | \(30.5557\) |
| Exact ground-state contractions | \(0.3282\) | threshold at \(154.25\) | \(0.9909\) |
| Exact contractions plus Eq. (112) residual subtraction | zero by construction | bounded through \(400\), \(\max|x_i|\approx1.8592\) | none through \(400\) |

For the last protocol, the minimum electronic eigenvalue through \(t=400\) is
approximately \(0.0252\), and the minimum retained phonon-density eigenvalue is
approximately \(2.9\times10^{-5}\). Thus Eq. (112) is a successful
strong-coupling scalar diagnostic regularization.

It is not yet a full solution. When Eq. (14c) and all matrix coordinates are
propagated, residual subtraction keeps the strong case amplitude-bounded and
the electronic 1-RDM positive through \(t=140\), but the phonon fluctuation
matrix develops a negative eigenvalue. The next task is therefore a
bosonic-boundary flux decomposition followed by a representability-preserving
correction, not a chaos calculation.

## Leading Eq. (14d) nonlinearity hypothesis

The primary source is G. Riva, J. Simoni, and Y. Ping,
[*Open-quantum-system theory of non-Markovian electron--phonon
dynamics*](https://arxiv.org/abs/2606.22233), arXiv:2606.22233. Equation (14d)
contains two especially strong nonlinear terms:

\[
\mathcal N_1
=
-i\sum_{\mathbf q'}\sum_{34}
g_{34}^{\mathbf q'+}
\left(\delta_{\mathbf q\mathbf q'}+\delta\rho_{\mathbf q\mathbf q'}\right)
\left(\delta_{14}-\rho_{14}\right)\rho_{32},
\]

\[
\mathcal N_2
=
+i\sum_{\mathbf q'}\sum_{34}
g_{34}^{\mathbf q'+}
\delta\rho_{\mathbf q\mathbf q'}
\left(\delta_{32}-\rho_{32}\right)\rho_{14}.
\]

These are the leading candidate causes of the high-interaction instability.
Unlike the linear free-evolution term and the bilinear
anomalous-covariance terms in Eq. (14d), \(\mathcal N_1\) and
\(\mathcal N_2\) contain products of:

- the electron--phonon coupling \(g\);
- the evolving phonon covariance \(\delta\rho_{\mathbf q\mathbf q'}\);
- electronic particle/hole factors such as
  \(\delta_{14}-\rho_{14}\) and \(\delta_{32}-\rho_{32}\);
- another electronic density-matrix factor.

They therefore contain quadratic and cubic dependence on the evolving state.
They also participate in a closed feedback loop:

\[
\delta\rho^{\mathbf q}
\longrightarrow
\left(\dot\rho,\delta\dot\rho\right)
\longrightarrow
\left(\rho,\delta\rho\right)
\longrightarrow
\left(\mathcal N_1,\mathcal N_2\right)
\longrightarrow
\delta\dot\rho^{\mathbf q}.
\]

The source paper states that coupling Eq. (14d) back to Eq. (14a) produces the
Fan--Migdal self-energy, while coupling it to Eq. (14b) produces the phonon
polarization at the random-phase-approximation level. The same feedback that
creates the desired non-Markovian physics could amplify an inconsistent
initial residual, a sign/index error, a closure error, or a genuinely unstable
mode.

The source derivation closes the hierarchy by truncating at second order in
the electron--phonon matrix element and decoupling electron and phonon fields
in four-body correlators. This makes closure consistency and the initial
correlations part of the hypothesis; large nonlinear terms alone do not prove
that the mathematical equations are pathological.

The source paper also records a crucial initialization contract: its Holstein
dimer simulations start from a root of the full EOM right-hand side and choose
the root with the lowest total energy. This is materially different from
starting with a Hartree--Fock electronic density and setting every correlation
variable to zero.

### Required causal ablation

After reproducing the full instability, define diagnostic scale factors
\(\alpha_1\) and \(\alpha_2\):

\[
\delta\dot\rho^{\mathbf q}
=
\mathcal L
+\alpha_1\mathcal N_1
+\alpha_2\mathcal N_2
+\mathcal A,
\]

where \(\mathcal L\) contains the free/commutator contribution and
\(\mathcal A\) contains the anomalous-covariance terms. At minimum compare:

| Diagnostic | \(\alpha_1\) | \(\alpha_2\) |
|---|---:|---:|
| Full source equation | 1 | 1 |
| Remove only \(\mathcal N_1\) | 0 | 1 |
| Remove only \(\mathcal N_2\) | 1 | 0 |
| Remove both nonlinear terms | 0 | 0 |

These are diagnostic ablations, not candidate physical equations. For every
case, record:

- \(\lVert\mathcal N_1\rVert\) and
  \(\lVert\mathcal N_2\rVert\) separately;
- their vector sum and cancellation ratio;
- each term's contribution to the local Jacobian;
- the largest real part of the fixed-point Jacobian eigenvalues;
- the initial residual and the failure time;
- conservation, positivity, and solver-convergence diagnostics.

Large opposing values of \(\mathcal N_1\) and \(\mathcal N_2\) would indicate
sensitive cancellation and possible roundoff/discretization amplification.
A large positive Jacobian eigenvalue that converges under solver refinement
would instead indicate a genuine local dynamical instability. The terms are
causally implicated only if the controlled ablations change the reproducible
failure while the remaining numerical and physical contracts are held fixed.

## Closed-form prospects

The complete coupled Eqs. (14a)--(14e) are nonlinear and, under a
time-dependent drive, nonautonomous. A general elementary closed-form
trajectory is therefore unlikely. There are nevertheless useful exact formal
solutions and analytically tractable reductions.

Define

\[
D_{\mathbf q\mathbf q'}=\delta\rho_{\mathbf q\mathbf q'},
\qquad
\bar D_{\mathbf q'\mathbf q}
  =\delta\bar\rho_{\mathbf q'\mathbf q},
\qquad
X_{\mathbf q}=\delta\rho^{\mathbf q}.
\]

### Equation (14e)

For a given electronic trajectory, the coherent-phonon equation is a linear
forced oscillator with the exact integral solution

\[
B_{\mathbf q}(t)
=e^{-i\omega_{\mathbf q}(t-t_0)}B_{\mathbf q}(t_0)
-i\int_{t_0}^{t}
e^{-i\omega_{\mathbf q}(t-s)}
\sum_{12}g_{12}^{\mathbf q+}\rho_{12}(s)\,\mathrm ds.
\]

This is a genuine closed integral expression, but it remains self-consistent
because \(B\) re-enters \(\tilde h\) and thereby changes \(\rho\).

### Equations (14b) and (14c)

For a given electron--phonon-correlation trajectory \(X_{\mathbf q}(t)\), the
ordinary and anomalous phonon covariances have exact forced-oscillator forms:

\[
D_{\mathbf q\mathbf q'}(t)
=e^{-i(\omega_{\mathbf q}-\omega_{\mathbf q'})(t-t_0)}
D_{\mathbf q\mathbf q'}(t_0)
+\int_{t_0}^{t}
e^{-i(\omega_{\mathbf q}-\omega_{\mathbf q'})(t-s)}
S_D[X(s)]\,\mathrm ds,
\]

\[
\bar D_{\mathbf q'\mathbf q}(t)
=e^{-i(\omega_{\mathbf q'}+\omega_{\mathbf q})(t-t_0)}
\bar D_{\mathbf q'\mathbf q}(t_0)
+\int_{t_0}^{t}
e^{-i(\omega_{\mathbf q'}+\omega_{\mathbf q})(t-s)}
S_{\bar D}[X(s)]\,\mathrm ds,
\]

where \(S_D\) and \(S_{\bar D}\) are the source terms written explicitly in
Eqs. (14b) and (14c).

### Equation (14d)

For prescribed \(\rho(t)\), \(D(t)\), \(\bar D(t)\), and \(B(t)\),
Eq. (14d) is linear in its own variable \(X_{\mathbf q}\). Write

\[
\dot X_{\mathbf q}
=\mathcal L_{\mathbf q}(t)X_{\mathbf q}
+S_{\mathbf q}[\rho(t),D(t),\bar D(t)],
\]

where \(\mathcal L_{\mathbf q}\) contains the commutator with
\(\tilde h(t)\) and the free phonon frequency, while \(S_{\mathbf q}\)
contains \(\mathcal N_1\), \(\mathcal N_2\), and the
anomalous-covariance terms. Its exact formal solution is

\[
X_{\mathbf q}(t)
=\mathcal U_{\mathbf q}(t,t_0)X_{\mathbf q}(t_0)
+\int_{t_0}^{t}
\mathcal U_{\mathbf q}(t,s)
S_{\mathbf q}[\rho(s),D(s),\bar D(s)]\,\mathrm ds,
\]

with the time-ordered propagator

\[
\mathcal U_{\mathbf q}(t,s)
=\mathcal T\exp\left[
\int_s^t\mathcal L_{\mathbf q}(\tau)\,\mathrm d\tau
\right].
\]

The primary paper explicitly states that it solves Eqs. (14d) and (14e)
analytically and substitutes them into Eqs. (14a) and (14b). This produces the
Fan--Migdal, RPA-polarization, and Ehrenfest memory contributions. The result
is a formal closed memory equation, not generally an elementary time-domain
solution.

### Equation (14a)

For prescribed \(B(t)\) and \(X_{\mathbf q}(t)\), Eq. (14a) is a linear
matrix equation with a time-ordered commutator propagator plus a collision
integral. Its feedback through Eqs. (14d) and (14e) makes the simultaneous
system nonlinear.

### Most useful analytic goal

The most useful closed-form program is:

1. derive the exact real \(L=2\) scalar reduction of Eqs. (14a)--(14e);
2. solve \(F(x_\ast,0)=0\) algebraically for every fixed-point branch;
3. evaluate the exact Jacobian
   \[
   J_\ast=\left.\frac{\partial F}{\partial x}\right|_{x_\ast};
   \]
4. obtain its characteristic polynomial and stability boundaries;
5. derive special-limit solutions for \(g=0\), coherent-only dynamics,
   linearized weak coupling, diagonal/Markovian reductions, and any autonomous
   symmetry-reduced case.

The linearized dynamics

\[
\delta x(t)=e^{J_\ast(t-t_0)}\delta x(t_0)
\]

is closed form and directly tests local sensitivity. Symbolic assistance can
be valuable for the scalarization, fixed-point elimination, Jacobian,
factorization, and special limits, but every result should be checked by
substitution into the source equations and by numerical residual tests.

## Current divergence diagnosis

A pinned local reproduction harness now exists. It rules out coarse RK4 time
stepping as the explanation for the scalar failure, but it does not yet prove
the physical cause or the source-faithfulness of the scalar transcription.
The current evidence supports a ranked working diagnosis:

The explicit divergence example in the Hubbard-dimer working PDF varies the
electron--phonon coupling \(\lambda\), with the clearest comparison at
\(\lambda=1.5\). It does not by itself establish the cause of a divergence
controlled by a separate Hubbard interaction \(U\). If the target failure is
specifically high-\(U\), the scalar reduction must state exactly where \(U\)
enters \(\tilde h\), the initial-state solve, or the closure before treating
the two observations as the same instability.

1. **Inconsistent initial state or wrong fixed-point branch.** The primary
   paper initializes the dimer at a root of the full EOM right-hand side and
   chooses the root with minimum total energy. The unstable working example
   instead uses a Hartree--Fock electronic density with all correlation
   variables set to zero. These data need not satisfy
   \(F(x_0,0)=0\).
2. **Strong-coupling fixed-point bifurcation.** In the scalar Ehrenfest system,
   the stationary branch changes at \(\lambda=1\). For \(\lambda>1\), the
   stable candidates have
   \(\Delta n=\pm\sqrt{1-\lambda^{-2}}\) and a nonzero coherent displacement.
   Continuing the symmetric weak-coupling initialization into this regime can
   place the trajectory on an unstable stationary branch.
3. **Driven nonlinear feedback and cancellation.** A nonzero initial residual
   in the correlation sector is returned to \(\rho\) and \(D\) and fed back
   into the correlation equations. However, zeroing both initially active
   source products makes \(F(x_0,0)=0\) and still produces a pulse-triggered
   divergence. The complete nonlinear feedback network, not the initial
   residual alone, is therefore implicated.
4. **Closure breakdown or physical-admissibility loss.** The source equations
   close a correlation hierarchy by a second-order truncation and a
   four-body-factorization approximation. Strong coupling may expose
   positivity, conservation, or consistency failures of that closure.
5. **Numerical stiffness.** The dynamics become increasingly stiff near the
   failure, but RK4 refinement and adaptive nonstiff and stiff solvers agree on
   the same threshold-crossing time. A fixed-step RK4 artifact is therefore
   strongly disfavored for the transcribed scalar ODE.
6. **Incomplete projections, followed by complete scalar closure.**
   Component-wise matrix parity excludes a sign, conjugation, spin-factor, or
   normalization error in the 13 retained derivatives. Adding Eq. (14c)
   produces a 15D projection, while iterative tangency closure requires 31
   real coordinates. The 31D result matches the complete matrix flow and shows
   that bosonic representability fails even when no generated direction is
   discarded.

The first decisive calculation is not a long trajectory. It is:

\[
r_0=F(x_0,0),
\qquad
J_0=\left.\frac{\partial F}{\partial x}\right|_{x_0}.
\]

For the Hartree--Fock/zero-correlation state and each energy-minimizing fixed
point, record \(\lVert r_0\rVert\), its decomposition by equation and term, and
the eigenvalues of \(J_0\). This separates:

- a state that is not a fixed point;
- a fixed point with a linearly unstable mode;
- a locally stable but numerically stiff state;
- a residual dominated specifically by
  \(\mathcal N_1\) or \(\mathcal N_2\).

The strongest present inference is that the strong pulse moves the approximate
equal-time closure out of its bosonic representability cone before any late
amplitude growth. The initial state and root branch materially affect the
later threshold, but the initial residual is not the sole cause. It is now
established that the complete two-site matrix flow exhibits this physicality
failure under the tested protocols. It is not established that Eq. (14d)
alone is responsible or that the behavior is chaotic.

## Local scalar reproduction: 2026-07-28

The reusable implementation is under `paper_5/src/paper5/stability/`. The
red-capable command is:

```bash
cd paper_5
PYTHONPATH=src python3 -m paper5.stability.reproduce \
  --lambda-ep 1.5 \
  --gamma 0.5 \
  --drive 1.0 \
  --time-step 0.01 \
  --final-time 140 \
  --expect-bounded
```

It exits nonzero because the boundedness expectation fails.

### Baseline result

For a failure threshold
\(\max_i|x_i|=10^4\):

| Integrator | Settings | Failure time |
|---|---|---:|
| RK4 | \(\Delta t=0.02\) | \(130.48\) |
| RK4 | \(\Delta t=0.01\) | \(130.47\) |
| RK4 | \(\Delta t=0.005\) | \(130.47\) |
| DOP853 | relative tolerance \(10^{-10}\), absolute tolerance \(10^{-12}\) | \(130.4674620\) |
| Radau | relative tolerance \(10^{-10}\), absolute tolerance \(10^{-12}\) | \(130.4674620\) |
| BDF | relative tolerance \(10^{-10}\), absolute tolerance \(10^{-12}\) | \(130.4673189\) |

The first threshold-crossing component is
\(\Delta\operatorname{Im}^{-}\). The weak-coupling control
\(\lambda=0.5\) remains bounded through \(t=140\), with maximum absolute state
component \(0.6044\).

### Initial residual and local Jacobian

For the Hartree--Fock electronic state with all correlation variables zero:

\[
\lVert F(x_0,0)\rVert_2=0.3423265984.
\]

The only nonzero components are

\[
\dot{\delta\operatorname{Im}}(0)
=-\frac{g}{4}
=-0.1530931089,
\]

\[
\Delta\dot{\operatorname{Im}}^{-}(0)
=-\frac{g}{2}
=-0.3061862178.
\]

The maximum real part of the instantaneous Jacobian eigenvalues at \(x_0\) is
numerically zero, approximately \(2.8\times10^{-16}\). Because \(x_0\) is not
a fixed point, this is not a fixed-point stability certificate. Along the
driven trajectory, the maximum instantaneous real part grows to approximately
\(1.39\) at \(t=128\), \(5.19\) at \(t=130\), and \(14.39\) at \(t=130.4\).
These local values diagnose rapidly increasing instability but are not
Lyapunov exponents.

### Initial-source ablations

The source products in scalar Eqs. (95) and (97) are diagnostic projections of
the strong correlation feedback. They have not yet been proven to correspond
one-to-one with \(\mathcal N_1\) and \(\mathcal N_2\) in the primary matrix
equation.

| Eq. (95) source scale | Eq. (97) source scale | Initial residual norm | Failure time |
|---:|---:|---:|---:|
| 1 | 1 | \(0.3423\) | \(130.47\) |
| 0 | 1 | \(0.3062\) | \(60.43\) |
| 1 | 0 | \(0.1531\) | \(69.67\) |
| 0 | 0 | \(0\) | \(59.92\) |

The physical pair delays failure relative to all three ablations. This is
evidence for stabilizing cancellation or balance between coupled terms, not
evidence that either source product alone causes the divergence.

Near \(t=130\), the largest scalar right-hand-side contributions include both
coherent-displacement/correlation products and electronic/phonon population
products. No single addend dominates every failing equation. The result is
therefore a network-level nonlinear feedback failure in the current scalar
model.

### Preliminary fixed-point finding

An unconstrained least-squares search finds many algebraic roots of
\(F(x_\ast,0)=0\), including roots with negative electronic or phonon
eigenvalues and roots whose reported energy falls below the exact ground-state
energy. Those are direct evidence that elementary root solving is not enough.

The maintained workflow now starts from exact ground-state contractions and
selects the locally connected stationary scalar root. At
\(\lambda=1.5,\gamma=0.5\), it has residual below \(10^{-14}\), positive
electronic and retained phonon eigenvalues, and remains amplitude-bounded
through \(t=400\). It still loses electronic positivity at \(t\approx30.56\).
This source-connected root is a reproducible candidate, not a proof of the
global minimum over unresolved higher-moment representability constraints.

## Reproduction protocol

The first deliverable is now a deterministic scalar instability harness, not a
quantum circuit.

The initial baseline should reproduce the five-variable Ehrenfest system,
because it supplies explicit equations, invariants, fixed points, and parameter
definitions. The next stage should add the Fan--Migdal correlation variables
from Eqs. (87)--(99). The full Paper V \(L=2\) reduction should follow only
after the baseline transcription is verified.

The reproduction matrix should include:

| Axis | Initial cases |
|---|---|
| Coupling | \(\lambda=0.5,\ 1.0,\ 1.5\), with \(\gamma=1/2\) |
| Drive | weak \(v=10^{-3}\) and strong \(v/t_{\mathrm{hop}}=1\) |
| Initial state | coupling-appropriate fixed point; Hartree--Fock electronic state plus zero correlations; exact correlated state when available |
| Vector field | unmodified source equations; initial-residual-subtracted candidate |
| Integration | fixed-step RK4 refinement plus at least one high-order adaptive nonstiff method and one stiff method |

Every run should record:

- time step or solver tolerances;
- solver method and precision;
- first non-finite time, if any;
- maximum state-vector magnitude;
- electronic trace, Hermiticity error, and positivity violation;
- the scalar invariant above when applicable;
- total-energy drift when the energy functional is available;
- residual norm at the proposed initial state,
  \(\lVert F(f(0),0)\rVert\);
- separation between trajectories initialized at \(f_0\) and
  \(f_0+\varepsilon\delta f\).

A useful red-capable reproduction criterion is:

> For a pinned strong-coupling configuration, the
> Hartree--Fock/zero-correlation initialization crosses a declared physical or
> numerical failure threshold reproducibly, while the corresponding
> correlated or coupling-appropriate stationary initialization does not.

The threshold and time interval must be declared before interpreting results.
A solver exception by itself is not sufficient; the harness must identify the
physical or numerical quantity that fails.

## Structured cone correction result

The direct correction is implemented in
`paper_5/src/paper5/stability/cone_correction.py`.  Its ten real control
coordinates are exactly the normal and anomalous moment velocities:

\[
y=
(\Delta\dot N_{00},\Delta\dot N_{11},
\operatorname{Re}\Delta\dot N_{01},
\operatorname{Im}\Delta\dot N_{01},
\operatorname{Re}\Delta\dot A_{00},
\operatorname{Im}\Delta\dot A_{00},
\operatorname{Re}\Delta\dot A_{11},
\operatorname{Im}\Delta\dot A_{11},
\operatorname{Re}\Delta\dot A_{01},
\operatorname{Im}\Delta\dot A_{01}).
\]

The structured lift \(\mathcal S(y)\) changes only closed-state derivative
indices 7--16.  The electronic, coherent-displacement, and correlation
velocities are unchanged directly.

A one-minimum-eigenvector correction repairs the first simple boundary mode,
but stalls near \(t\approx3.60\), where the two lowest bosonic eigenvalues
become nearly degenerate.  The ordered minimum eigenvector then switches
between modes and produces numerical chattering.  The implemented controller
therefore solves the complete matrix barrier

\[
\begin{aligned}
\min_{y\in\mathbb R^{10}}\quad&\frac12\lVert y\rVert_2^2,\\
\text{subject to}\quad&
D+\mathcal S(y)
+\beta(\mathcal M_{\mathrm B}-h_\star I)
-\kappa I\succeq0,
\end{aligned}
\]

with \(h_\star=10^{-5}\), \(\beta=5\), and \(\kappa=0\).  Constraint
generation adds the half-space induced by every violated eigenvector, solves
the accumulated Euclidean projection, and repeats until the complete
\(4\times4\) barrier matrix is positive semidefinite to tolerance.

For a correction confined to \(N\) and \(A\),

\[
\Delta\dot E
=\omega_{\mathrm{ph}}
(\Delta\dot N_{00}+\Delta\dot N_{11}).
\]

The energy-neutral realization optimizes in
\(\Delta\dot N_{00}+\Delta\dot N_{11}=0\).  This equality removes the direct
energy injection exactly.

Pinned \(t=20\) results use \(\lambda=1.5\), \(\gamma=0.5\), drive amplitude
1, exact ground-state contractions at phonon cutoff 16, DOP853,
\(\mathrm{rtol}=10^{-9}\), \(\mathrm{atol}=10^{-11}\), and maximum step
0.05:

| Protocol | Minimum \(\lambda(\mathcal M_{\mathrm B})\) | Minimum \(\lambda(\rho)\) | Maximum \(|x_i|\) | Maximum post-pulse drift from \(E(4)\) |
|---|---:|---:|---:|---:|
| Eq. (112), no cone correction | \(-3.05033\times10^{-1}\) | \(5.51945\times10^{-2}\) | 1.97874 | \(2.41557\times10^{-6}\) |
| Full matrix barrier | \(2.89314\times10^{-5}\) | \(5.51655\times10^{-2}\) | 1.89301 | \(6.54778\times10^{-2}\) |
| Energy-neutral matrix barrier | \(2.89314\times10^{-5}\) | \(4.88418\times10^{-2}\) | 1.97408 | \(2.53904\times10^{-6}\) |

The unconstrained barrier is active on 36.16 percent of sampled times and has
maximum correction norm 0.42120.  The energy-neutral barrier is active on
55.11 percent of sampled times, has maximum norm 0.59424, and converges for
every sampled subproblem.  The short-time result therefore establishes
structured cone controllability with simultaneous direct energy neutrality.
Its frequent activity remains a closure-adequacy diagnostic.

The next calculation extends the energy-neutral run through the late
amplitude-validation horizon, refines step/tolerance/cutoff settings, and
compares selected observables with exact driven-dimer propagation.  A
correlation-level Eq. (14d) controller remains the causal alternative because
it acts on the history channel identified by the boundary decomposition.

## Separating instability from chaos

Only after the divergence is reproducible should the project test for chaos.
The following distinctions must be preserved:

1. **Discretization instability:** the failure time or trajectory changes
   materially under time-step, tolerance, method, or precision refinement.
2. **Unstable or inconsistent initial state:** the initial residual is nonzero
   or the state violates stationary, positivity, normalization, or closure
   constraints.
3. **Dynamical instability:** a converged trajectory departs exponentially
   from an unstable fixed point but may remain deterministic and nonchaotic.
4. **Finite-time blow-up:** a norm grows without bound with a
   solver-independent scaling law.
5. **Chaos:** nearby bounded, converged trajectories show a robust positive
   largest Lyapunov exponent over an adequate time window.

For a chaos claim, at minimum:

- first establish integrator and precision convergence;
- perturb admissible initial conditions, not constraint-violating states;
- estimate a finite-time and asymptotic largest Lyapunov exponent with periodic
  perturbation renormalization;
- verify that exponential separation is not merely departure along one
  linearly unstable mode;
- confirm bounded long-time motion or otherwise state that the behavior is
  blow-up rather than chaos.

## Strange-attractor hypothesis

A strange attractor is a candidate interpretation of the bounded irregular
behavior, but it is a stronger claim than sensitive initial conditions or
chaos. The working hypothesis is:

> In some strong-coupling regime, the nonlinear feedback through
> \(\mathcal N_1\) and \(\mathcal N_2\) in Eq. (14d) may produce a bounded,
> attracting invariant set with a positive largest Lyapunov exponent and
> fractal geometry.

This hypothesis must remain separate from the instability-reproduction task.
A divergent trajectory cannot establish a strange attractor because an
attractor must be bounded. A visibly irregular trajectory also cannot
distinguish a strange attractor from a quasiperiodic torus, a long transient, a
noisy numerical orbit, or a conservative chaotic region.

There is an additional structural issue. The source paper reports conservation
of total energy after the external pump. If the complete post-pulse scalar
flow also preserves phase-space volume, then a conventional dissipative
strange attractor is excluded; the system could instead contain a chaotic
invariant set or chaotic sea on a constant-energy surface. Energy conservation
alone does not decide this question because the closed approximate EOM need not
be Hamiltonian or volume preserving. The phase-space divergence and Lyapunov
spectrum must be measured.

### Minimum strange-attractor test

Apply the test only after obtaining bounded, solver-converged trajectories:

1. Turn off the explicit pulse after its declared end time and analyze the
   resulting autonomous dynamics. If a periodic drive remains, augment the
   state with its phase before discussing an invariant set.
2. Construct the exact real scalar state and its constraint manifold. Remove
   redundant Hermitian-conjugate components before estimating dimensions.
3. Integrate the tangent equation
   \[
   \dot{\delta x}=J_F(x(t))\,\delta x
   \]
   alongside the trajectory and compute the full Lyapunov spectrum using
   periodic QR reorthonormalization.
4. Require a robust positive largest Lyapunov exponent under time-step,
   tolerance, precision, and trajectory-length refinement.
5. Test attraction by starting several admissible states within a proposed
   basin and checking whether their long-time invariant statistics and
   geometric support agree even though their pointwise trajectories separate.
6. Estimate phase-space contraction from
   \[
   \nabla\!\cdot F(x)
   \quad\text{and}\quad
   \sum_i\Lambda_i.
   \]
   A conventional dissipative attractor should have negative long-time volume
   contraction, equivalently a negative sum of Lyapunov exponents.
7. Estimate a noninteger invariant-set dimension using at least the
   Kaplan--Yorke dimension from the Lyapunov spectrum and, if the sampling is
   adequate, an independent correlation-dimension estimate.
8. Inspect a Poincaré section or recurrence plot to distinguish fractal
   structure from a limit cycle or quasiperiodic torus.
9. Repeat the analysis for the full source equation and the controlled
   \((\alpha_1,\alpha_2)\) ablations to determine whether either nonlinear term
   creates, destroys, or merely shifts the candidate set.

Evidence should be classified conservatively:

| Observation | Supported interpretation |
|---|---|
| Step-size-dependent divergence | numerical instability |
| Converged unbounded growth | dynamical blow-up |
| Bounded irregular motion without Lyapunov analysis | candidate complex dynamics |
| Positive largest Lyapunov exponent, no attraction test | chaos |
| Positive exponent plus basin attraction and fractal dimension | strange-attractor evidence |
| Positive exponent with no phase-space contraction | conservative chaotic set more likely than a conventional attractor |

## Workflow boundary for the quantum algorithm

The quantum-algorithm workstream may proceed conceptually in parallel, but no
algorithm should be benchmarked as a solution to the physical problem until
the stability workstream fixes:

- the exact \(L=2\) scalar/vector state;
- the admissible initial-state manifold;
- whether \(F(f(0),0)=0\) is required and how it is enforced;
- closure and regularization choices;
- physical invariants and error metrics;
- a stable classical reference trajectory.

Candidate quantum approaches currently mentioned in the Paper V draft include
McLachlan-projected dynamics, variational residual minimization, pVQD-style
time-step matching, variational linear solvers inside implicit steps, and
hybrid classical evaluation of nonlinear contractions. These remain candidate
routes, not selected solutions.

## Open questions

- Which specific term or sector of the Paper V Eqs. (14a)--(14d) first becomes
  unstable in the \(L=2\), high-\(U\) calculation?
- Does the instability occur for a state that is stationary under the
  approximate closed equations, or only when exact/Hartree--Fock data are
  inserted into an inconsistent closure?
- Is the initial residual dominated by the electron--phonon correlation
  equation, the phonon covariance, the anomalous covariance, or the electronic
  equation?
- Do \(\mathcal N_1\) and \(\mathcal N_2\) become individually large and
  cancellation-sensitive before the observed failure?
- Which of \(\mathcal N_1\), \(\mathcal N_2\), or their feedback through
  Eqs. (14a) and (14b) controls the unstable Jacobian mode?
- Is any bounded irregular regime attracting, or is it a conservative chaotic
  invariant set, a quasiperiodic torus, or a long transient?
- Does the post-pulse scalar flow contract phase-space volume, and what is the
  sign of the sum of its Lyapunov exponents?
- Can the exact correlated initial data used in the Hubbard-dimer document be
  reconstructed reproducibly from a ground-state solve or energy
  minimization?
- Does subtracting \(F(f(0),0)\) restore a missing correlation counterterm, or
  does it merely alter the intended dynamics?
- What is the exact mapping between the Hubbard-dimer scalar variables and
  every component of the Riva--Simoni--Ping Eqs. (14a)--(14d)?
- After numerical convergence and admissible initialization, is there any
  positive-Lyapunov regime left to classify as chaos?
- Does adaptive coherent-packet spawning keep the exact-state and 82-moment
  errors controlled through (t=20) without reintroducing exact-reference
  feedback?
- Can a gauge-fixed or orthogonalized packet tangent replace the chosen
  Euclidean Tikhonov metric while retaining smooth parameter velocities?

## Working sources

- Active Paper V draft:
  `MATH/paper_details/paper_V_high_u_gkba.tex`
- Paper V support workspace:
  `MATH/paper_facing/paper_V_high_u_gkba/`
- Paper V code and exploratory workspace:
  `paper_5/`
- Hubbard-dimer working PDF:
  `/Users/jakestrobel/Downloads/Dynamics_on_the_Hubbard_DIMER.pdf`
- Electron--phonon/chiral-phonon working PDF:
  `/Users/jakestrobel/Downloads/Electron_phonon_interactions___chiral_phonons.pdf`
