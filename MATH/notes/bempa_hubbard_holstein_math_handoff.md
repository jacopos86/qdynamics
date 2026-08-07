# Agent Handoff: Mathematical/Physical Use of BEMPA for the Hubbard--Holstein Repo

Created: 2026-05-09  
Source paper: Sina Bahrami and Nicolas Sawaya, "Particle-conserving quantum circuit ansatz with applications in variational simulation of bosonic systems," arXiv:2402.18768.  
Source verified: arXiv page, accessed 2026-05-09. The arXiv abstract describes BEMPA as a binary encoded multilevel particles circuit ansatz that preserves particle count by construction and demonstrates VQE for the Bose--Hubbard Hamiltonian.

Repo context: `MATH/Math.md` describes a Hubbard--Holstein-oriented mathematical stack with binary/unary boson encodings, HH Hamiltonian substitution, polaronic operator families, staged ADAPT selection, reduced tangent novelty, pruning/recovery, and projective McLachlan real-time control.

This handoff is for a mathematics/physics agent. Do not begin with code. First formalize the operator algebra, symmetry claims, sector restrictions, and benchmark contracts. Only after the identities below are explicit should implementation details be considered.

## 1. Central takeaway

The BEMPA paper gives a compact binary-encoded ansatz for bosonic systems with a conserved total particle number. Its main mathematical value for the repo is not merely a pair of new circuits. It is a symmetry-native generator construction:

\[
[A_m,\widehat N_{\mathrm{tot}}]=0
\quad\Longrightarrow\quad
\widehat N_{\mathrm{tot}}\,e^{-i\theta A_m}|\psi_0\rangle
= N_0\,e^{-i\theta A_m}|\psi_0\rangle
\]

whenever

\[
\widehat N_{\mathrm{tot}}|\psi_0\rangle=N_0|\psi_0\rangle.
\]

Thus BEMPA replaces the penalty philosophy

\[
H_{\mathrm{eff}}=H+\eta(\widehat N_{\mathrm{tot}}-N_0)^2
\]

by a hard variational-manifold restriction. This is directly relevant to the repo's ADAPT mathematics because the repo already scores candidate generators by tangent geometry, novelty, cost, and symmetry gates. BEMPA gives a clean bosonic \(U(1)\)-preserving operator subpool for compact binary boson registers.

The crucial caveat is physical: the Hubbard--Holstein phonon sector does not generally conserve phonon number because

\[
H_g=g\sum_i x_i(\hat n_i-\bar n I),
\qquad
x_i=b_i+b_i^\dagger,
\]

and \(x_i\) changes phonon occupation. Therefore BEMPA must not be blindly imposed as a hard phonon-number-preserving ansatz for the production HH ground-state manifold. It is directly applicable to the repo's number-conserving Bose--Hubbard route, to any \(U(1)\) boson transport benchmark, and as inspiration for binary-carry-aware phonon transition generators in HH.

## 2. Register convention to use before any derivation

The repo's mathematical convention is that qubit \(q_0\) is the rightmost printed bit and the least-significant bit. Therefore every BEMPA formula from the paper must be re-indexed into the repo convention before being treated as canonical.

For a bosonic mode \(i\) with local binary cutoff dimension \(d_i\le 2^q\), define qubits

\[
(i,k),\qquad k=0,1,\ldots,q-1,
\]

where \(k\) is the binary significance and the bit contributes \(2^k\) to occupation. The local binary occupation operator is

\[
\widehat n_i^{\mathrm{bin}}
=\sum_{k=0}^{q-1}2^k\widehat z_{i,k},
\qquad
\widehat z_{i,k}:=\frac{I-Z_{i,k}}{2}.
\]

The total conserved boson number for a pure Bose--Hubbard or other \(U(1)\) boson problem is

\[
\widehat N_B
=\sum_i\widehat n_i^{\mathrm{bin}}
=\sum_i\sum_{k=0}^{q-1}2^k\frac{I-Z_{i,k}}{2}.
\]

If the local dimension is not a power of two, then the binary register contains invalid states. In that case, symmetry preservation of \(\widehat N_B\) is not enough. The agent must also define a validity projector

\[
\Pi_{\mathrm{valid}}
=\sum_{0\le n_i\le d_i-1\ \forall i}|\mathbf n\rangle\langle \mathbf n|
\]

and check whether proposed generators preserve \(\operatorname{im}\Pi_{\mathrm{valid}}\). For the first BEMPA benchmark, use power-of-two local dimensions only.

## 3. BEMPA primitive generators

### 3.1 Same-significance exchange generator \(G_A\)

For two mode-bits \((i,k)\) and \((j,k)\) in the same significance block, define

\[
G_A^{(i,j;k)}
:=i|01\rangle\langle10|-i|10\rangle\langle01|.
\]

With the two-qubit order matching the paper's convention, its Pauli form is

\[
G_A
=\frac12(X_0Y_1-Y_0X_1).
\]

In repo-indexed form, use

\[
G_A^{(i,j;k)}
=\frac12\left(X_{i,k}Y_{j,k}-Y_{i,k}X_{j,k}\right),
\]

after confirming the sign convention against the chosen ordered exponential

\[
U_A(\theta)=e^{-i\theta G_A}.
\]

The particle-number proof is immediate because \(G_A\) only mixes \(|01\rangle\) and \(|10\rangle\), and both states have one occupied bit of the same weight \(2^k\):

\[
[ G_A^{(i,j;k)},\widehat N_B]=0.
\]

This generator is a compact-encoding analogue of same-level boson transport.

### 3.2 Binary-carry transfer generator \(G_B\)

The \(A\) generators alone cannot connect all states with a fixed total boson number. They preserve the count inside each significance block separately. BEMPA adds a carry-transfer generator that trades two lower-significance bits for one higher-significance bit.

For two lower-significance bits \((i,k-1)\) and \((j,k-1)\) and one higher-significance bit \((\ell,k)\), define the abstract carry transition

\[
|1_{i,k-1}1_{j,k-1}0_{\ell,k}\rangle
\longleftrightarrow
|0_{i,k-1}0_{j,k-1}1_{\ell,k}\rangle.
\]

Both sides have the same weighted occupation:

\[
2^{k-1}+2^{k-1}=2^k.
\]

The Hermitian generator is

\[
G_B^{(i,j\to\ell;k)}
:=i|001\rangle\langle110|-i|110\rangle\langle001|,
\]

where the displayed basis order must be mapped carefully onto the repo's qubit order. In the paper's qubit convention, the Pauli representation is

\[
G_B
=\frac14\left(
X_0X_1Y_2
-X_0Y_1X_2
-Y_0X_1X_2
-Y_0Y_1Y_2
\right).
\]

Thus the repo-indexed version should be stored symbolically as

\[
G_B^{(a,b,c)}
=\frac14\left(
X_aX_bY_c
-X_aY_bX_c
-Y_aX_bX_c
-Y_aY_bY_c
\right),
\]

provided \((a,b,c)\) are ordered consistently with the \(|001\rangle\leftrightarrow |110\rangle\) convention. The agent must explicitly verify this by applying the Pauli polynomial to the two active basis states before accepting the sign/order.

Again,

\[
[G_B^{(i,j\to\ell;k)},\widehat N_B]=0.
\]

The terms in \(G_B\) commute pairwise, so the exponential can be treated exactly as a product of Pauli rotations:

\[
e^{-i\alpha G_B}
=\prod_{r=1}^{4}e^{-i\alpha c_rP_r},
\qquad
G_B=\sum_{r=1}^{4}c_rP_r,
\qquad
[P_r,P_s]=0.
\]

This exact-commuting structure is important for the repo's measurement-cost, tangent-cost, and McLachlan-metric accounting.

## 4. Direct repo improvement: add a \(U(1)\) boson-transport pool

For a number-conserving bosonic benchmark such as Bose--Hubbard, define the candidate pool

\[
\mathcal G_{\mathrm{BEMPA}}
:=
\mathcal G_A\cup\mathcal G_B,
\]

with

\[
\mathcal G_A
:=\{G_A^{(i,j;k)}:i<j,\ 0\le k<q\},
\]

and

\[
\mathcal G_B
:=\{G_B^{(i,j\to\ell;k)}:
0<k<q,\ i,j,\ell\ \text{admissible mode labels}\}.
\]

The admissibility gate should be mathematical, not empirical:

\[
\Gamma_{\mathrm{sym}}^{B}(m)
:=\mathbf 1\left([G_m,\widehat N_B]=0\right).
\]

For this pool,

\[
\Gamma_{\mathrm{sym}}^{B}(m)=1
\quad\forall m\in\mathcal G_{\mathrm{BEMPA}}.
\]

The symmetry leakage observable for a candidate state is

\[
\mathcal L_N(\psi)
:=\langle\psi|(\widehat N_B-N_0)^2|\psi\rangle.
\]

For a BEMPA-only manifold initialized in the correct sector,

\[
\mathcal L_N(\psi(\theta))=0
\]

up to numerical and measurement error. This lets the repo replace a penalty objective with a hard sector-conserving manifold for the Bose--Hubbard route.

## 5. Do not confuse Bose--Hubbard bosons with Hubbard--Holstein phonons

The repo's HH Hamiltonian has the symbolic master form

\[
H_{\mathrm{HH}}(t)
=
H_{\mathrm{Hub}}
+
\omega_0\sum_i\left(n_{b,i}+\frac12I\right)
+g\sum_i x_i(n_i-\bar n I)
+\sum_i v_i(t)n_i.
\]

The phonon displacement operator is

\[
x_i=b_i+b_i^\dagger,
\]

and therefore

\[
[x_i,\widehat N_{\mathrm{ph}}]\ne0,
\qquad
[H_{\mathrm{HH}},\widehat N_{\mathrm{ph}}]\ne0
\]

in general. Consequently, a hard BEMPA phonon-number-preserving HH ansatz would exclude physical phonon-displacement amplitude and is not a correct production HH ansatz.

Correct use cases are:

1. Use BEMPA directly for number-conserving bosonic benchmarks, especially Bose--Hubbard.
2. Use BEMPA as a compact binary carry-transport subpool when a model has actual bosonic \(U(1)\) conservation.
3. Use BEMPA's carry logic to design HH transition-resolved phonon generators, but do not force \([G,\widehat N_{\mathrm{ph}}]=0\) for the HH phonon displacement channels.

## 6. HH-relevant extension: binary transition-resolved phonon generators

The repo already uses the phonon primitives

\[
x_i=b_i+b_i^\dagger,
\qquad
P_i=i(b_i^\dagger-b_i),
\]

and polaronic families such as

\[
\tilde n_iP_i,
\qquad
D_iP_i,
\qquad
T_{ij}^{(+)}(P_i-P_j),
\qquad
J_{ij}^{(-)}(x_i-x_j).
\]

For \(n_{\mathrm{ph,max}}>1\), binary phonon transitions involve carry structure. BEMPA suggests resolving the phonon ladder into binary transition operators.

For local binary states \(|n\rangle_i\mapsto|\operatorname{bin}(n)\rangle_i\), define

\[
C_{i,n}^{(+)}
:=
|n+1\rangle_i\langle n|+|n\rangle_i\langle n+1|,
\]

and

\[
C_{i,n}^{(-)}
:=
i\left(|n+1\rangle_i\langle n|-|n\rangle_i\langle n+1|\right).
\]

Then

\[
x_i
=\sum_{n=0}^{d-2}\sqrt{n+1}\,C_{i,n}^{(+)},
\]

and

\[
P_i
=\sum_{n=0}^{d-2}\sqrt{n+1}\,C_{i,n}^{(-)}.
\]

Each transition operator has an exact Pauli expansion from the single-bit identity

\[
|a\rangle\langle b|
=
\begin{cases}
(I+Z)/2,&a=b=0,\\
(I-Z)/2,&a=b=1,\\
(X+iY)/2,&a=0,\ b=1,\\
(X-iY)/2,&a=1,\ b=0.
\end{cases}
\]

Thus the agent should define HH carry-aware child families

\[
G_{i,n}^{(\mathrm{cd})}
:=\tilde n_i C_{i,n}^{(-)},
\]

\[
G_{i,n}^{(\mathrm{dd})}
:=D_i C_{i,n}^{(-)},
\]

\[
G_{ij,n}^{(\mathrm{hd})}
:=T_{ij}^{(+)}\left(C_{i,n}^{(-)}-C_{j,n}^{(-)}\right),
\]

and, if needed,

\[
G_{ij,n}^{(\mathrm{od})}
:=J_{ij}^{(-)}\left(C_{i,n}^{(+)}-C_{j,n}^{(+)}\right).
\]

These are not BEMPA generators because they do not conserve phonon number. They are BEMPA-inspired binary-carry-resolved HH generators. They can improve the repo when \(n_{\mathrm{ph,max}}\ge2\) because they expose which phonon-number transitions are geometrically useful instead of treating \(x_i\) or \(P_i\) as a monolithic encoded object.

For the current \(n_{\mathrm{ph,max}}=1\) HH line, \(d=2\) and \(q=1\), so

\[
C_{i,0}^{(+)}=X_{i,0},
\qquad
C_{i,0}^{(-)}=Y_{i,0}.
\]

Therefore the BEMPA-inspired refinement is mostly invisible at the present two-level phonon cutoff. Its value appears at \(n_{\mathrm{ph,max}}\ge2\).

## 7. ADAPT selector integration at the mathematical level

The repo's staged selector uses candidate-position records

\[
r=(m,p),
\]

horizontal tangents

\[
\tau_r=Q_\psi(-i\widetilde A_r|\psi\rangle),
\]

and reduced novelty / trust-region scores in later phases. BEMPA integrates naturally as an additional structured generator family.

For a BEMPA candidate \(A_m\), the gradient is

\[
g_m
:=\left.\partial_\alpha
\langle\psi|e^{i\alpha A_m}He^{-i\alpha A_m}|\psi\rangle
\right|_{\alpha=0}
=-i\langle\psi|[H,A_m]|\psi\rangle.
\]

The Fubini--Study tangent norm is

\[
F_m
=\langle\tau_m,\tau_m\rangle
=\langle A_m^2\rangle_\psi-\langle A_m\rangle_\psi^2.
\]

For the number-conserving boson route, add a symmetry certificate to the candidate metadata:

\[
\chi_N(m):=\mathbf 1([A_m,\widehat N_B]=0).
\]

For \(m\in\mathcal G_{\mathrm{BEMPA}}\), \(\chi_N(m)=1\) exactly.

For HH transition-resolved phonon candidates, do not demand \(\chi_N=1\) with respect to phonon number. Instead track electron-sector symmetry and physical transition support.

## 8. Carry-connectivity diagnostic

A-only BEMPA is mathematically incomplete because it preserves the separate bit-count in each significance block. The repo should treat \(G_B\) as a bridge/carry class, not as an expendable duplicate of \(G_A\).

For a fixed total boson number \(N_0\), define the sector graph

\[
\mathfrak G_{N_0}
=(V_{N_0},E_A\cup E_B),
\]

where

\[
V_{N_0}
:=\left\{\mathbf n=(n_1,\ldots,n_L):0\le n_i<d,\ \sum_i n_i=N_0\right\}.
\]

Edges in \(E_A\) connect states related by one same-significance exchange. Edges in \(E_B\) connect states related by one binary carry transfer

\[
2^{k-1}+2^{k-1}\leftrightarrow 2^k.
\]

The agent should verify:

\[
\mathfrak G_{N_0}^{A}:=(V_{N_0},E_A)
\]

is generally disconnected, while

\[
\mathfrak G_{N_0}^{A\cup B}:=(V_{N_0},E_A\cup E_B)
\]

is the intended connected candidate graph for the fixed-number sector, subject to cutoff and boundary restrictions.

This diagnostic should feed the pruning theory: removing all \(B\)-type carry generators may fragment the reachable manifold even if local energy seems temporarily recoverable. Treat \(B\)-type coordinates as possible noncommuting bridge or carry-bridge coordinates in the prune-recoverability ladder.

## 9. Gray conversion as measurement/dynamics mathematics

BEMPA state preparation uses standard binary because the \(A/B\) carry structure is defined in binary significance blocks. The paper then suggests optionally converting the prepared state to Gray encoding before measurement or later Hamiltonian simulation. The mathematical reason is that Gray-encoded Hamiltonians often have lower Pauli weight and better grouped-measurement behavior.

For binary bits \(b_{q-1},\ldots,b_0\), define Gray bits by

\[
g_{q-1}=b_{q-1},
\qquad
g_j=b_{j+1}\oplus b_j,
\quad 0\le j<q-1.
\]

The conversion is a Clifford map. At the symbolic level, define a unitary \(U_{\mathrm{bin}\to\mathrm{Gray}}\) satisfying

\[
U_{\mathrm{bin}\to\mathrm{Gray}}|b\rangle=|g(b)\rangle.
\]

For measurement, transform observables as

\[
H_{\mathrm{Gray}}
=U_{\mathrm{bin}\to\mathrm{Gray}}H_{\mathrm{bin}}U_{\mathrm{bin}\to\mathrm{Gray}}^\dagger.
\]

Then compare Pauli-weight distributions and grouped-measurement costs.

Use the paper's shot-count reduction diagnostic. For

\[
H=\sum_i a_iP_i,
\]

the ungrouped estimate is

\[
N_{\mathrm{ungrouped}}
=\frac{1}{\epsilon^2}\left(\sum_i |a_i|\sqrt{\operatorname{Var}(P_i)}\right)^2.
\]

For commuting partitions \(k\) with terms \(a_{k\ell}P_{k\ell}\), the paper's approximate grouped-shot reduction is

\[
\widehat R
:=
\left[
\frac{
\sum_k\sum_{\ell=1}^{m_k}|a_{k\ell}|
}{
\sum_k\sqrt{\sum_{\ell=1}^{m_k}|a_{k\ell}|^2}
}
\right]^2.
\]

The repo's measurement-cost term should be allowed to compare binary and terminal Gray surfaces. For \(q=1\), Gray equals binary, so no benefit is expected. Do not spend effort on Gray conversion for the present \(n_{\mathrm{ph,max}}=1\) HH runs.

## 10. McLachlan dynamics interpretation

The repo's Chapter 17A real-time law uses projective McLachlan geometry:

\[
G_{\mu\nu}=\Re\langle\bar\tau_\mu,\bar\tau_\nu\rangle,
\qquad
f_\mu=\Re\langle\bar\tau_\mu,\bar b\rangle,
\qquad
K\dot\theta=f.
\]

For a number-conserving boson system, adding BEMPA candidates at a checkpoint changes the tangent plane without leaving the fixed-\(N\) sector. A BEMPA append candidate \(r\) has new tangent block

\[
\bar U_r=\left[Q_\psi(-i\widetilde A_{r,a}|\psi\rangle)\right]_a.
\]

The same Schur append-gain geometry already in the repo applies, but the candidate set is symmetry-native:

\[
[A_{r,a},\widehat N_B]=0.
\]

Therefore, for a Bose--Hubbard dynamics benchmark, BEMPA append should be preferred over generic Pauli append whenever the structural miss is caused by missing fixed-number boson transport.

For HH dynamics, BEMPA append is only appropriate for a genuinely number-conserving bosonic subsystem. For phonon displacement dynamics, use the transition-resolved \(C_{i,n}^{(\pm)}\) families instead.

## 11. Benchmark contracts the agent should propose

### 11.1 Pure Bose--Hubbard fixed-number ADAPT benchmark

Use

\[
H_{\mathrm{BH}}
=-t\sum_{\langle i,j\rangle}(b_i^\dagger b_j+b_j^\dagger b_i)
+\omega_0\sum_i n_i
+\frac{U}{2}\sum_i n_i(n_i-I)
+\sum_i v_i n_i.
\]

Initialize in a fixed nonzero particle sector:

\[
|\phi_{N_0}\rangle\in\ker(\widehat N_B-N_0I).
\]

Compare

\[
\mathcal G_{\mathrm{BEMPA}}=\mathcal G_A\cup\mathcal G_B
\]

against penalty-based generic Pauli or hardware-efficient pools using

\[
|E_{\mathrm{ADAPT}}-E_{\mathrm{exact},N_0}|,
\qquad
\langle(\widehat N_B-N_0)^2\rangle,
\qquad
\text{logical depth},
\qquad
\text{two-qubit burden},
\qquad
\widehat R_{\mathrm{binary}},\ \widehat R_{\mathrm{Gray}}.
\]

Use \(d=4\) or \(d=8\) first. The present \(d=2\) case is too small to expose the carry value of \(G_B\).

### 11.2 HH transition-resolved phonon benchmark

For HH with \(n_{\mathrm{ph,max}}\ge2\), compare the monolithic polaronic pool

\[
\{\tilde n_iP_i,\ D_iP_i,\ T_{ij}^{(+)}(P_i-P_j),\ldots\}
\]

with the transition-resolved pool

\[
\{\tilde n_i C_{i,n}^{(-)},\ D_iC_{i,n}^{(-)},\ T_{ij}^{(+)}(C_{i,n}^{(-)}-C_{j,n}^{(-)}),\ldots\}.
\]

The purpose is not to conserve phonon number. The purpose is to expose binary carry-resolved phonon transitions to the ADAPT geometry so that the selector can learn which \(n\leftrightarrow n+1\) channels are useful.

Primary metrics:

\[
|E_{\mathrm{ADAPT}}^{(N_b^{\mathrm{work}})}-E_{\mathrm{exact}}^{(N_b^{\mathrm{work}})}|,
\]

\[
|E_{\mathrm{ADAPT}}^{(N_b^{\mathrm{work}})}-E_{\mathrm{exact}}^{(N_b^{\mathrm{eval}})}|,
\]

fidelity to exact sector state where available, scaffold length, tangent novelty behavior, and measurement-group cost.

### 11.3 Carry-fragmentation ablation

Run three pools in a pure \(U(1)\) boson benchmark:

\[
\mathcal G_A,
\qquad
\mathcal G_B,
\qquad
\mathcal G_A\cup\mathcal G_B.
\]

Expected mathematical result:

\[
\mathcal G_A\ \text{alone is generally sector-fragmented},
\qquad
\mathcal G_A\cup\mathcal G_B\ \text{is the intended connected fixed-}N\text{ pool}.
\]

The report should explicitly show a sector graph or reachable-basis count, not only final energy.

## 12. Where to amend the mathematical manuscript

Suggested manuscript-level insertions for a later pass:

1. After the binary/unary encoding section, add a subsection titled `Binary significant-figure blocks and BEMPA U(1) boson transport`.
2. After the Bose--Hubbard reference subsection, add `BEMPA fixed-number generator pool` with \(G_A\), \(G_B\), and \([G,\widehat N_B]=0\) proofs.
3. In the HH polaronic-operator section, add `binary transition-resolved phonon ladder decomposition` with \(C_{i,n}^{(\pm)}\) and the expansions of \(x_i\) and \(P_i\).
4. In the adaptive-selection section, add BEMPA as a symmetry-native pool family and carry-bridge metadata class.
5. In the measurement-cost section, add terminal binary-to-Gray observable transformation and \(\widehat R\) grouped-shot comparison.
6. In the prune-recoverability section, add a note that B-type carry coordinates are possible sector-connectivity bridges and should not be deleted merely because their amplitudes are small.
7. In the real-time McLachlan section, add BEMPA append candidates for number-conserving boson dynamics and transition-resolved phonon append candidates for HH dynamics.

## 13. Mathematical checks before any implementation

The agent should verify the following identities on paper or with exact symbolic algebra.

First, same-significance exchange:

\[
\left[\frac12(X_{i,k}Y_{j,k}-Y_{i,k}X_{j,k}),
2^k\frac{I-Z_{i,k}}2+2^k\frac{I-Z_{j,k}}2
\right]=0.
\]

Second, carry transfer:

\[
\left[G_B^{(i,j\to\ell;k)},
2^{k-1}\frac{I-Z_{i,k-1}}2
+2^{k-1}\frac{I-Z_{j,k-1}}2
+2^k\frac{I-Z_{\ell,k}}2
\right]=0.
\]

Third, the Pauli expansion of \(G_B\) actually maps the chosen repo-ordered basis states with the intended sign.

Fourth, the four Pauli words in \(G_B\) commute pairwise.

Fifth, for \(d=2\):

\[
C_{i,0}^{(+)}=X_{i,0},
\qquad
C_{i,0}^{(-)}=Y_{i,0},
\]

so the transition-resolved HH pool reduces to the existing one-qubit phonon-quadrature language.

Sixth, for \(d=4\) and higher, transition-resolved \(C_{i,n}^{(\pm)}\) contains multi-qubit carry strings and is therefore a genuinely new compact-binary phonon refinement.

Seventh, if \(d\) is not a power of two, every candidate must be tested for validity-subspace preservation:

\[
[G,\Pi_{\mathrm{valid}}]=0
\]

or else the benchmark must explicitly project / penalize invalid code states.

## 14. Non-goals and failure modes

Do not hard-constrain HH phonon number. That would remove the physics of the Holstein displacement.

Do not treat BEMPA as a universal replacement for the repo's polaronic HH pools. It is a \(U(1)\) boson-transport construction. Its HH value is mainly in binary carry-aware transition resolution and in number-conserving benchmark routes.

Do not report only energy. For BEMPA, always report sector leakage

\[
\langle(\widehat N_B-N_0)^2\rangle.
\]

Do not use \(A\) gates alone as a final pool except as an ablation. \(A\) gates do not generally connect the full fixed-number sector in binary encoding.

Do not use Gray encoding during BEMPA generator placement unless the generator algebra has been rederived in Gray coordinates. The paper's construction is binary-first and Gray-after.

Do not allow pruning to erase all carry-transfer directions before the reachable-sector graph has been checked.

## 15. Minimal final deliverable for the next agent

The next math/physics agent should produce a manuscript patch or note with these objects explicitly defined:

\[
\widehat N_B
=\sum_i\sum_k2^k\frac{I-Z_{i,k}}2,
\]

\[
G_A^{(i,j;k)}
=\frac12(X_{i,k}Y_{j,k}-Y_{i,k}X_{j,k}),
\]

\[
G_B^{(i,j\to\ell;k)}
=\frac14\left(
X_aX_bY_c-X_aY_bX_c-Y_aX_bX_c-Y_aY_bY_c
\right),
\]

with explicit basis-order verification for \((a,b,c)\),

\[
[G_A,\widehat N_B]=0,
\qquad
[G_B,\widehat N_B]=0,
\]

\[
\mathcal G_{\mathrm{BEMPA}}=\mathcal G_A\cup\mathcal G_B,
\]

\[
C_{i,n}^{(+)}=|n+1\rangle_i\langle n|+|n\rangle_i\langle n+1|,
\]

\[
C_{i,n}^{(-)}=i(|n+1\rangle_i\langle n|-|n\rangle_i\langle n+1|),
\]

\[
x_i=\sum_{n=0}^{d-2}\sqrt{n+1}C_{i,n}^{(+)},
\qquad
P_i=\sum_{n=0}^{d-2}\sqrt{n+1}C_{i,n}^{(-)},
\]

and the explicit warning

\[
[H_{\mathrm{HH}},\widehat N_{\mathrm{ph}}]\ne0
\]

for the physical HH phonon sector.

The most valuable repo improvement is therefore two-layered:

\[
\text{Bose--Hubbard / }U(1)\text{ bosons: hard BEMPA sector-preserving ADAPT pool},
\]

and

\[
\text{Hubbard--Holstein phonons: BEMPA-inspired binary carry-resolved ladder-transition pool}.
\]

