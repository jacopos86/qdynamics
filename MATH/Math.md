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
- $\mathcal C_{\mathrm{cheap}}$: the cheap-pass capped candidate set retained after the first Phase-3 gradient screen.
- $\mathcal S_2,\mathcal S_3$: the Phase-2 and Phase-3 shortlists that survive cheap screening before full reranking.
- $\mathcal C_{\mathrm{split}}(m)$: the family of split children generated from macro generator $m$.
- $\mathcal O_*$: the finally selected adaptive scaffold, i.e. the ordered generator sequence admitted by the adaptive stage.
- $\mathcal M_{\mathrm{warm}},\mathcal M_{\mathrm{refine}},\mathcal M_{\mathrm{replay}}$: the warm-start, optional refine, and replay manifolds.
- $\mathcal B$: a portable handoff/state bundle carrying manifest, amplitudes, energies, and continuation metadata.
- $\mathcal C$: continuation payload / provenance payload attached to a handoff bundle.
- $\Gamma_{\mathrm{stage}}$: stage-admissibility gate.
- $\Gamma_{\mathrm{sym}}$: symmetry-admissibility gate.
- $\mathfrak b$: a beam branch state.
- $\mathfrak F_c$: live frontier at beam round $c$.
- $\mathfrak T_c$: terminal-branch pool at beam round $c$.
- $\mathfrak W_{\mathrm{final}}$: final union of surviving frontier and terminal branches.
- $J_b$: cumulative branch objective tuple used to rank or prune branch $b$.

### 1.2.2 Quick parameter glossary for adaptive and continuation sections

The symbols below are the main scalar controls that appear in the adaptive-selection, split-aware, and beam-style formulas.

- $\lambda_{\mathrm{2q}}$: weight on two-qubit proxy burden.
- $\lambda_d$: weight on depth-style proxy burden.
- $\lambda_\theta$: weight on marginal parameter-count burden.
- $\lambda_F$: cheap-stage normalization weight multiplying the tangent metric term.
- $\lambda_H$: regularization strength for inherited-window Hessian blocks.
- $\rho$: trust-region radius scale used in one-dimensional local improvement models.
- $N_{\mathrm{cheap}}^{\max}$: maximum cheap-pass size before shortlist construction.
- $N_{\max}$: maximum shortlist size.
- $f_{\mathrm{short}}$: shortlist fraction used in Phase 3.
- $w_R$: weight on remaining-depth or remaining-evaluations burden.
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
- $\tau_{\mathrm{miss}}$: projected-miss threshold used in manifold-growth logic.
- $B_{\mathrm{child}}$: beam child cap per parent branch-expansion round.
- $B_{\mathrm{live}}$: live-frontier beam width retained after pruning.
- $B_{\mathrm{term}}$: cap on stored terminal branches.
- $\Delta k_{\mathrm{cp}}$: checkpoint advance before evaluating branch continuation.
- $\Delta k_{\mathrm{probe}}$: probe advance used for local branch scoring after checkpointing.
- $C$: final beam round index / number of retained beam-update rounds.

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

## 11.1 Escalating phase order for adaptive selection

Rather than scoring the full master operator pool at every adaptive depth, the selector works through a nested sequence of candidate universes
$$
\mathcal G^{(3)} \subseteq \mathcal G^{(2)} \subseteq \mathcal G^{(1)} \subseteq \mathcal G,
$$
where $\mathcal G$ is the full expensive master pool.

The **runtime** order is:

- **Phase 3 (primary / preferred):** search a highly restricted, heavily filtered subset of the master pool, typically the operators judged most physically plausible or most likely to matter locally,
- **Phase 2 (first fallback):** if Phase 3 yields no admissible continuation signal, relax the filter and search a broader subset,
- **Phase 1 (final fallback):** if Phase 2 also fails, remove the filter and scan the broad expensive pool so that no viable gradient is missed before declaring local exhaustion.

That is what the current HH continuation logic does. But for understanding the construction, it is clearer to define the surfaces in the reverse conceptual order:

- **Phase 1** gives the broad base selector,
- **Phase 2** adds shortlist-and-rerank refinement on top of that base,
- **Phase 3** is the highest-tier selector we actually run now, built on the lower two surfaces.

So the exposition below is written in the pedagogical order **Phase 1 $\to$ Phase 2 $\to$ Phase 3**, even though the runtime tries them in the order **Phase 3 $\to$ Phase 2 $\to$ Phase 1**.

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

### 11.3.1 Core signal, append position, refit window, and cheap stage

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

This section describes the broader symbolic highest-tier selector surface: position-aware candidate records, inherited refit windows, and split-aware reranking. This is the surface currently used in runtime policy, but it is easiest to read only after the Phase-1 and Phase-2 constructions are already in place.

### 11.4.1 Core signal, append position, refit window, and cheap stage

The Phase-3 selector is most naturally written over candidate-position pairs
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

The full Phase-3 available pool is
$$
\mathcal P_{\mathrm{avail}}^{(3)}
=
\left\{
r=(m,p):
m\in\mathcal G^{(3)}_{\mathrm{avail}},
\ p\in\mathcal P_m
\right\}.
$$
Let
$$
N_{\mathrm{cheap}}
=
\min\!\left\{
N_{\mathrm{cheap}}^{\max},
\left|\mathcal P_{\mathrm{avail}}^{(3)}\right|
\right\}
$$
be the active cheap-pass size, where $N_{\mathrm{cheap}}^{\max}$ is the Phase-3 cheap-pass cap. Then the cheap universe is the gradient-ranked capped subset
$$
\mathcal C_{\mathrm{cheap}}
=
\operatorname{Top}_{N_{\mathrm{cheap}}}\!\left(
\mathcal P_{\mathrm{avail}}^{(3)};
g_{\mathrm{lcb}}
\right),
$$
that is, the set of the top $N_{\mathrm{cheap}}$ records in $\mathcal P_{\mathrm{avail}}^{(3)}$ ranked by the lower-confidence gradient signal $g_{\mathrm{lcb}}(r)$ defined below.

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
K_{\mathrm{cheap}}(r)
&=1+w_R\bar R_{\mathrm{rem}}(r)+w_D\bar D_{\mathrm{life}}(r)\\
&\quad +w_G\bar G_{\mathrm{new}}(r)+w_C\bar C_{\mathrm{new}}(r)+w_c\bar c(r).
\end{aligned}
$$
Yes—here is the full term expansion you asked for, in symbolic form:

$$
K_{\mathrm{cheap}}(r)=1+w_R\bar R_{\mathrm{rem}}(r)+w_D\bar D_{\mathrm{life}}(r)+w_G\bar G_{\mathrm{new}}(r)+w_C\bar C_{\mathrm{new}}(r)+w_c\bar c(r)
$$
with
$$
\bar D_{\mathrm{life}}(r)=\frac{D_{\mathrm{raw}}(r)}{\text{depth\_ref}},\quad
D_{\mathrm{raw}}(r)=\sum_{\ell\in\mathcal L_r}\!\left[2\max(\mathrm{wt}(\ell)-1,0)+\frac12(2\,\#_{X,Y}(\ell)+1)\right]+|p_{\text{app}}-p|,
$$
where $\mathcal L_r$ is the active Pauli-label set for the candidate term (fallback is $\max(1,\text{candidate\_term\_count})+|p_{\text{app}}-p|$).

$$
\bar G_{\mathrm{new}}(r)=\frac{G_{\mathrm{new}}(r)}{\text{group\_ref}},\quad G_{\mathrm{new}}(r)=|\mathcal G_{\mathrm{new}}(r)|,
$$
$$
\bar C_{\mathrm{new}}(r)=\frac{C_{\mathrm{new}}(r)}{\text{shot\_ref}},\quad C_{\mathrm{new}}(r)=G_{\mathrm{new}}(r)\cdot n_{\text{shots/group}},
$$
$$
\bar c(r)=\frac{c(r)}{\text{reuse\_ref}},\quad c(r)=\text{groups\_new}(r),
$$
$$
\bar R_{\mathrm{rem}}(r)=N_{\mathrm{rem}}\!\left(\bar D_{\mathrm{life}}+\bar G_{\mathrm{new}}+\bar C_{\mathrm{new}}+\bar c+\bar P_{\mathrm{opt}}\right)\mathbf{1}_{\text{lifetime mode}},\quad
N_{\mathrm{rem}}=\max(1,d_{\max}-d+1),\quad \bar P_{\mathrm{opt}}=\frac{|W(p)|}{\text{optdim\_ref}}.
$$

Qualitatively: $\bar D_{\mathrm{life}\!}$ is circuit/compile burden at the candidate record, $\bar G_{\mathrm{new}}$ and $\bar C_{\mathrm{new}}$ are measurement-group and shot burden of the new term, $\bar c$ is the reuse penalty, and $\bar R_{\mathrm{rem}}$ is the remaining-depth lifetime discount/multiplier applied only when lifetime-cost mode is on.
So the cheap Phase-3 score is not only symbolically but explicitly
$$
\begin{aligned}
S_{3,\mathrm{cheap}}(r)
&=\Gamma_{\mathrm{stage}}(r)\Gamma_{\mathrm{sym}}(r)\frac{g_{\mathrm{lcb}}(r)^2}{2\lambda_FF_{\mathrm{raw}}(r)}\frac{1}{K_{\mathrm{cheap}}(r)+\varepsilon}\\
&=\Gamma_{\mathrm{stage}}(r)\Gamma_{\mathrm{sym}}(r)
\frac{\max\{|2\Re\langle \psi|H|d_r\rangle|-z_\alpha\sigma_r,0\}^2}{2\lambda_F\langle d_r|Q_\psi|d_r\rangle\,(K_{\mathrm{cheap}}(r)+\varepsilon)}.
\end{aligned}
$$
This stage is meant to be wide relative to the final shortlist but still inexpensive relative to a full local-model rebuild: it keeps only the strongest cheap gradient proposals before any more expensive proving by reduced-path geometry, novelty, and rerank score is carried out.

**Math**

$$
K_{\mathrm{cheap}}(r)
=
1
+
w_R\,\bar R_{\mathrm{rem}}(r)
+
w_D\,\bar D_{\mathrm{life}}(r)
+
w_G\,\bar G_{\mathrm{new}}(r)
+
w_C\,\bar C_{\mathrm{new}}(r)
+
w_c\,\bar c(r),
\qquad r=(m,p).
$$

$$
\bar D_{\mathrm{life}}(r)=\frac{D_{\mathrm{raw}}(r)}{\mathrm{depth}_{\mathrm{ref}}},
\qquad
D_{\mathrm{raw}}(r)
=
\sum_{\ell\in\mathcal L_r}
\left[
2\max(\mathrm{wt}(\ell)-1,0)
+
\frac12\!\left(2\,\#_{X,Y}(\ell)+1\right)
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
\qquad
C_{\mathrm{new}}(r)=G_{\mathrm{new}}(r)\,n_{\mathrm{shots/group}},
\qquad
c(r)=\mathrm{groups}_{\mathrm{new}}(r),
$$

$$
\bar R_{\mathrm{rem}}(r)
=
N_{\mathrm{rem}}
\left(
\bar D_{\mathrm{life}}+\bar G_{\mathrm{new}}+\bar C_{\mathrm{new}}+\bar c+\bar P_{\mathrm{opt}}
\right)\mathbf 1_{\mathrm{lifetime}},
\qquad
N_{\mathrm{rem}}=\max(1,d_{\max}-d+1),
\qquad
\bar P_{\mathrm{opt}}=\frac{|W(p)|}{\mathrm{optdim}_{\mathrm{ref}}}.
$$

From first principles, $K_{\mathrm{cheap}}(r)$ is a dimensionless burden functional: the numerator of $S_{3,\mathrm{cheap}}(r)$ estimates local benefit, and $K_{\mathrm{cheap}}(r)$ discounts that benefit by how expensive it is to admit $r$ now and, through $\bar R_{\mathrm{rem}}$, how expensive that decision is likely to remain over the rest of the run. The $2\max(\mathrm{wt}(\ell)-1,0)$ piece is a two-qubit-style burden proxy, the $\frac12(2\,\#_{X,Y}(\ell)+1)$ piece is a basis-change / one-qubit rotation proxy, and $|p_{\mathrm{app}}-p|$ penalizes inserting far from the current append location.

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
where the first arrow is a cheap gradient screen and the later arrows perform the more expensive local proving and reranking.

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
K_{\mathrm{full}}(r)
&=1+w_R\bar R_{\mathrm{rem}}(r)+w_D\bar D_{\mathrm{life}}(r)\\
&\quad +w_G\bar G_{\mathrm{new}}(r)+w_C\bar C_{\mathrm{new}}(r)+w_c\bar c(r).
\end{aligned}
$$
So the base rerank score closes to
$$
\begin{aligned}
S_{3,\mathrm{base}}(r)
&=\Gamma_{\mathrm{stage}}(m)\Gamma_{\mathrm{sym}}(r)\,\nu_r^{\gamma_N}\frac{\Delta E_{\mathrm{TR}}(r)}{K_{\mathrm{full}}(r)+\varepsilon}\\
&=\Gamma_{\mathrm{stage}}(m)\Gamma_{\mathrm{sym}}(r)
\left[\operatorname{clip}_{[0,1]}\!\left(1-\frac{(q_r^{\mathrm{red}})^\top(Q_r+\varepsilon_{\mathrm{nov}}I)^{-1}q_r^{\mathrm{red}}}{F_r^{\mathrm{red}}}\right)\right]^{\gamma_N}
\frac{\Delta E_{\mathrm{TR}}(r)}{K_{\mathrm{full}}(r)+\varepsilon}.
\end{aligned}
$$
The augmented selector score is therefore
$$
\begin{aligned}
S_{3,\mathrm{aug}}(r)
&=S_{3,\mathrm{base}}(r)+\beta_{\mathrm{split}}\Sigma(r)+\beta_{\mathrm{motif}}M(r)+\beta_{\mathrm{sym}}Y(r)-\beta_{\mathrm{dup}}D_{\mathrm{dup}}(r)\\
&=\Gamma_{\mathrm{stage}}(m)\Gamma_{\mathrm{sym}}(r)
\left[\operatorname{clip}_{[0,1]}\!\left(1-\frac{(q_r^{\mathrm{red}})^\top(Q_r+\varepsilon_{\mathrm{nov}}I)^{-1}q_r^{\mathrm{red}}}{F_r^{\mathrm{red}}}\right)\right]^{\gamma_N}
\frac{\Delta E_{\mathrm{TR}}(r)}{K_{\mathrm{full}}(r)+\varepsilon}\\
&\qquad+\beta_{\mathrm{split}}\Sigma(r)+\beta_{\mathrm{motif}}M(r)+\beta_{\mathrm{sym}}Y(r)-\beta_{\mathrm{dup}}D_{\mathrm{dup}}(r).
\end{aligned}
$$
Here $\Sigma$ measures the gain available from selective macro-splitting, $M$ measures motif compatibility or transferable local pattern alignment, $Y$ measures symmetry quality or mitigation value, and $D_{\mathrm{dup}}$ penalizes near-duplicate continuation directions. If a shortlisted macro generator $m$ admits a split family
$$
\mathcal C_{\mathrm{split}}(m)=\{m_1,\dots,m_{K_m}\},
$$
then each split child inherits the same position $p$ and therefore defines a child record $r_j=(m_j,p)$, with replacement rule
$$
r\leadsto r_j^*\quad\text{if}\quad \max_j S_{3,\mathrm{aug}}(r_j) > S_{3,\mathrm{aug}}(r)+\tau_{\mathrm{split}}.
$$

### 11.4.3 Beam-adapt branch state, pruning, and effective selector

The beam-adaptive Phase-3 state closes to
$$
\begin{aligned}
\mathfrak b
&=\bigl(\operatorname{id}_b,\pi_b,\mathcal O_b,\theta_b,\Theta_b,\Psi_b,\Xi_b,k_b,J_b,\mathcal M_b^{\mathrm{opt}},\sigma_b,\tau_b\bigr),\\
\mathcal O_b&=(m_{b,1},\dots,m_{b,|\mathcal O_b|}),
\qquad
\Theta_b=(\theta_b^{(0)},\dots,\theta_b^{(k_b)}),
\qquad
\Psi_b=(|\psi_b^{(0)}\rangle,\dots,|\psi_b^{(k_b)}\rangle),\\
\Xi_b&=(\mu_b^{(0)},\dots,\mu_b^{(k_b)}),
\qquad
J_b=\bigl(J_b^{\epsilon},J_b^{\mathrm{rt}},J_b^{\mathrm{res}},J_b^{\mathrm{model}},J_b^{\mathrm{damp}},J_b^{\mathrm{cons}}\bigr).
\end{aligned}
$$
These entries record branch id, parent, selected scaffold, current parameters, retained parameter/state/telemetry histories, current time index, cumulative cost, optimizer memory, status, and terminal reason.

At beam round $c$, with live frontier $\mathfrak F_c$ and terminal pool $\mathfrak T_c$, the checkpoint, proposal, materialization, and probe surfaces compress to
$$
\begin{aligned}
k_{\mathrm{cp}}(b)
&=\min\{k_b+\Delta k_{\mathrm{cp}},N_t-1\},
\qquad
\mathcal P(b)=\{\mathrm{stay}\}\ \text{or}\ \{\mathrm{stay},a_{b,1},\dots,a_{b,K_b}\},
\qquad
K_b\le B_{\mathrm{child}}-1,\\
\mathcal T(\mathfrak b,\mathrm{stay})
&=\mathfrak b,
\qquad
\mathcal T(\mathfrak b,m)=\mathfrak b'\ \text{with}\ \mathcal O_{b'}=(\mathcal O_b,m),\ \theta_{b'}=(\theta_b,0),\\
k_{\mathrm{pr}}(b)
&=\min\{k_{\mathrm{cp}}(b)+\Delta k_{\mathrm{probe}},N_t-1\},\\
\Delta J_{b'}
&=\bigl(\Delta J_{b'}^{\epsilon},\Delta J_{b'}^{\mathrm{rt}},\Delta J_{b'}^{\mathrm{res}},\Delta J_{b'}^{\mathrm{model}},\Delta J_{b'}^{\mathrm{damp}},\Delta J_{b'}^{\mathrm{cons}}\bigr),\\
\Delta J_{b'}^{\epsilon}
&=\int_{t_{\mathrm{cp}}}^{t_{\mathrm{pr}}}\epsilon_{\mathrm{step}}^2(t)\,dt,
\qquad
\Delta J_{b'}^{\mathrm{model}}=\int_{t_{\mathrm{cp}}}^{t_{\mathrm{pr}}}\epsilon_{\mathrm{proj}}^2(t)\,dt,\\
\Delta J_{b'}^{\mathrm{damp}}
&=\int_{t_{\mathrm{cp}}}^{t_{\mathrm{pr}}}\max\!\bigl(\epsilon_{\mathrm{step}}^2(t)-\epsilon_{\mathrm{proj}}^2(t),0\bigr)\,dt,
\qquad
J_{b'}=J_b+\Delta J_{b'}.
\end{aligned}
$$
The remaining entries $\Delta J_{b'}^{\mathrm{rt}}$, $\Delta J_{b'}^{\mathrm{res}}$, and $\Delta J_{b'}^{\mathrm{cons}}$ are supplied by the corresponding runtime, resource, and conservation densities.

Children, fingerprints, prune keys, and the beam update then close to
$$
\begin{aligned}
\mathfrak C_b^{\mathrm{front}}
&=\{\mathfrak b' : \mathfrak b'=\mathcal T(b,r),\ r\in\mathcal P(b),\ \sigma_{b'}=\mathrm{frontier}\},\\
\mathfrak C_b^{\mathrm{term}}
&=\{\mathfrak b' : \mathfrak b'=\mathcal T(b,r),\ r\in\mathcal P(b),\ \sigma_{b'}=\mathrm{terminal}\},\\
\mathrm{fp}(\mathfrak b)
&=\bigl(k_b,\operatorname{labels}(\mathcal O_b),\operatorname{round}_{10}(\theta_b)\bigr),\\
\kappa_{\mathrm{prune}}(\mathfrak b)
&=\bigl(J_b^{\mathrm{rt}},J_b^{\mathrm{res}},J_b^{\mathrm{model}},J_b^{\mathrm{damp}},J_b^{\mathrm{cons}},|\mathcal O_b|,k_b,\operatorname{labels}(\mathcal O_b),\operatorname{round}_{10}(\theta_b),\operatorname{id}_b\bigr),\\
\operatorname{Dedup}(\mathcal X)
&=\{\text{best branch per fingerprint under }\kappa_{\mathrm{prune}}\},\\
\operatorname{Prune}(\mathcal X;C)
&=\text{lowest-}C\text{ branches in }\operatorname{Dedup}(\mathcal X)\text{ by }\kappa_{\mathrm{prune}},\\
\mathfrak F_{c+1}
&=\operatorname{Prune}\!\left(\bigcup_{b\in\mathfrak F_c}\mathfrak C_b^{\mathrm{front}};B_{\mathrm{live}}\right),\\
\mathfrak T_{c+1}
&=\operatorname{Prune}\!\left(\mathfrak T_c\cup\bigcup_{b\in\mathfrak F_c}\mathfrak C_b^{\mathrm{term}};B_{\mathrm{term}}\right),\\
\mathfrak W_{\mathrm{final}}
&=\mathfrak F_C\cup\mathfrak T_C,
\qquad
\mathfrak b_*=\arg\min_{\mathfrak b\in\mathfrak W_{\mathrm{final}}}\kappa_{\mathrm{prune}}(\mathfrak b).
\end{aligned}
$$
So the effective beam selector itself closes to
$$
\begin{aligned}
\mathfrak b_*
&=\arg\min_{\mathfrak b\in\mathfrak F_C\cup\mathfrak T_C}\kappa_{\mathrm{prune}}(\mathfrak b)\\
&=\arg\min_{\mathfrak b\in\mathfrak F_C\cup\mathfrak T_C}\bigl(J_b^{\mathrm{rt}},J_b^{\mathrm{res}},J_b^{\mathrm{model}},J_b^{\mathrm{damp}},J_b^{\mathrm{cons}},|\mathcal O_b|,k_b,\operatorname{labels}(\mathcal O_b),\operatorname{round}_{10}(\theta_b),\operatorname{id}_b\bigr),
\end{aligned}
$$
after recursively generating children by $\mathcal T$ and accumulating $J_{b'}=J_b+\Delta J_{b'}$.

An implementation-aligned restricted append-only projected-dynamics version of the highest-tier surface is written later in §17.19.

## 11.5 HH pool and replay-family surfaces

A denser symbolic staged decomposition is
$$
\begin{aligned}
\mathcal M_{\mathrm{warm}}
&=\{\,U_{\mathrm{warm}}(\theta_{\mathrm{w}})|\phi_0\rangle:\theta_{\mathrm{w}}\in\mathbb R^{m_{\mathrm{w}}}\,\},\\
\mathcal M_{\mathrm{refine}}
&=\{\,U_{\mathrm{refine}}(\theta_{\mathrm{r}})U_{\mathrm{warm}}(\theta_{\mathrm{w}}^*)|\phi_0\rangle:\theta_{\mathrm{r}}\in\mathbb R^{m_{\mathrm{r}}}\,\},\\
\Omega_{\mathrm{HH}}^{(3)}&\subseteq\Omega_{\mathrm{HH}}^{(2)}\subseteq\Omega_{\mathrm{HH}}^{(1)},
\qquad
\mathcal G_{\mathrm{adapt}}^{(k)}=\{\tau_m\}_{m\in\Omega_{\mathrm{HH}}^{(k)}},\\
\mathcal M_{\mathrm{replay}}(\mathcal O_*)
&=\{\,U_{\mathrm{replay}}(\varphi;\mathcal O_*)|\phi_0\rangle:\varphi\in\mathbb R^{m_{\mathrm{rep}}}\,\},
\end{aligned}
$$
so the stage handoff closes as
$$
\begin{aligned}
(\theta_{\mathrm{w}}^*,|\psi_{\mathrm{warm}}\rangle)
&\in\mathcal M_{\mathrm{warm}}
\mapsto
(\theta_{\mathrm{r}}^*,|\psi_{\mathrm{refine}}\rangle)
\in\mathcal M_{\mathrm{refine}}\\
&\mapsto
(\mathcal O_*,\theta_*^{\mathrm{adapt}},|\psi_*\rangle),
\qquad
\mathcal O_*=(\tau_{m_1},\dots,\tau_{m_d}),\ \tau_{m_j}\in\mathcal G_{\mathrm{adapt}}^{(3)}\\
&\mapsto
(\varphi^{(0)},|\psi_{\mathrm{replay}}^{(0)}\rangle)
=\mathcal H_{\mathrm{replay}}(|\psi_*\rangle,\theta_*^{\mathrm{adapt}},\mathcal O_*,\mathcal C_*)
\in\mathcal M_{\mathrm{replay}}(\mathcal O_*).
\end{aligned}
$$
Warm and refine are fixed-structure manifolds, the adaptive stage draws from the nested HH pools $\Omega_{\mathrm{HH}}^{(3)}\subseteq\Omega_{\mathrm{HH}}^{(2)}\subseteq\Omega_{\mathrm{HH}}^{(1)}$, and replay is a controlled continuation around the selected scaffold $\mathcal O_*$. 

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

# 14. Continuation, Handoff, and Replay Contract

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
- $\mathcal P$ is replay-policy metadata.

## 14.3 Replay contract

Let $\mathcal H_{A\to B}$ be a handoff map from source manifold $\mathcal M_A$ to target manifold $\mathcal M_B$:
$$
\mathcal H_{A\to B}: (|\psi_A\rangle,\theta_A,\mathcal C_A) \mapsto (|\psi_B^{(0)}\rangle,\theta_B^{(0)},\mathcal C_B).
$$
A common replay policy is:

1. preserve inherited scaffold parameters,
2. initialize residual parameters at zero,
3. optionally freeze inherited coordinates for a burn-in stage,
4. then unfreeze and refit in a controlled window.

## 14.4 Symmetry and tier-three payload note

Symmetry information belongs in the handoff if later stages will use it for validation, postselection, or projector-renormalized estimation. Mathematically, the carried object is a projector or a list of commuting symmetry generators.

## 14.5 Optional staged seed-refine insertion and provenance

A staged chain may include an optional refine map between warm and adaptive stages:
$$
|\psi_{\mathrm{warm}}\rangle
\xrightarrow{\mathcal R}
|\psi_{\mathrm{refine}}\rangle
\xrightarrow{\mathcal A}
|\psi_{\mathrm{adapt}}\rangle.
$$
If no refine stage is inserted, then one simply sets
$$
|\psi_{\mathrm{refine}}\rangle = |\psi_{\mathrm{warm}}\rangle.
$$

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

7. **Append-only current branch growth.** The clean story allows arbitrary insertion positions and multi-parameter blocks. The helper layer now supports position-aware insertion and position-centered post-insert refit windows,
   $$
   \mathcal T(\mathfrak b,m,p)=\mathfrak b',
   \qquad
   \mathcal O_{b'}=\mathcal O_b\oplus_p m,
   \qquad
   \theta_{b'}=\theta_b\oplus_p 0,
   $$
   but the live selector still evaluates only $p=|\theta|$. So the implemented active runtime surface remains zero-initialized **single-parameter append** growth rather than full anywhere-insertion selection.

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
12. define continuation, handoff, and replay on top of those manifolds,
13. define benchmark and noise-validation contracts relative to the same exact targets.

This is the point of the symbolic duplicate: the same mathematical skeleton remains, but implementation labels are stripped away so the document can serve as a clean theory-facing reference.

# Appendix A. Spec-only or not-yet-formalized surfaces kept out of the main body

## A.1 Broader continuation-theory items

Several topics belong naturally in a later symbolic pass rather than in the present core manuscript:

- multi-source motif transfer and motif tiling,
- richer split-event formalism,
- more detailed rescue dynamics,
- exact handoff-policy closure beyond the present abstract maps,
- measured-geometry and backend-specific closure beyond the exact symbolic statevector surface.

## A.2 Drive-amplitude comparison path

A drive-amplitude comparison path can be expressed symbolically by comparing several amplitudes $A_0,A_1,\dots$ at fixed Hamiltonian parameters and examining
$$
\Delta E_A(t),\qquad \Delta D_A(t),\qquad F_A(t)
$$
across the amplitude family. The formal structure is clear even if the exact report layout is deferred.

## A.3 Appendix rule

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
