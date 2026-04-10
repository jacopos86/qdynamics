---
title: "Hubbard-Holstein Mathematical Implementation (Current Linear Substitution-First Form)"
author: "Jake Skyler Strobel (repo-grounded revision)"
date: "March 21, 2026"
geometry: margin=0.8in
fontsize: 10pt
---

# 1. Parameter Manifest and Reader Contract

> Naming note: `MATH/archaic_repo_math.md` is the archival repo-oriented manuscript. `MATH/Math.md` is the markdown authoring source for manuscript sync, and `MATH/Math.tex` is the generated typesetting twin used to build `MATH/Math.pdf`. Regenerate the synchronized TeX/PDF pair with `python MATH/build_math_from_md.py` or `MATH/build_math.sh`.

This manuscript is a present-tense, self-contained mathematical description of the implemented Hubbard and Hubbard-Holstein (HH) stack in this repository. It keeps the same linear style as the older manuscript,

1. primitives first,
2. composite operators second,
3. explicit substitutions third,
4. fully substituted forms last,

but it removes older future-tense framing and replaces it with the currently implemented operator, variational, drive, PAOP, continuation, and handoff surfaces.

## 1.1 Required parameter manifest

- Model family: `Hubbard` / `Hubbard-Holstein`.
- Lattice size: `L` or `dims`.
- Fermion ordering: `blocked` or `interleaved`.
- Boundary condition: `open` or `periodic`.
- Core fermion parameters: `t` (or `J` on HH surfaces), `U`, `dv` / site potentials.
- Core phonon parameters: `omega0`, `g` / `g_ep`, `n_ph_max`, `boson_encoding`.
- Variational family when relevant:
  - `hh_hva` = HH layerwise,
  - `hh_hva_tw` = HH Pauli-termwise,
  - `hh_hva_ptw` = HH physical-termwise,
  - ADAPT / explicit-family surfaces including `hva`, `full_meta`, `paop_*`,
    `uccsd_paop_lf_full`, `uccsd_otimes_paop_lf_std`,
    `uccsd_otimes_paop_lf2_std`, `uccsd_otimes_paop_bond_disp_std`,
    optional logical-pair `*_seq2p`, and `full_hamiltonian`.
- Optimizer/runtime parameters when relevant:
  - VQE optimizer,
  - SPSA schedule parameters,
  - energy backend,
  - ADAPT state backend,
  - staged continuation mode,
  - drive waveform controls,
  - propagator choice.

## 1.2 Reader contract

This document intentionally favors explicit substitution over compressed meta-definition.

- If a primitive exists, it is written first.
- If a composite operator is formed from primitives, those primitives are substituted into the composite.
- If a later formula can be reduced by inserting an earlier primitive explicitly, the insertion is shown.
- When a runtime surface has different defaults from a core builder surface, the difference is named explicitly instead of being hidden behind a single “repo default”.

## 1.3 Non-negotiable repository conventions

### 1.3.1 Internal Pauli alphabet

Internally the Pauli alphabet is always
$$
\{e,x,y,z\},
$$
with `e` as identity.

### 1.3.2 Pauli-word and qubit ordering

Pauli words and computational-basis labels are written left-to-right as
$$
q_{N_q-1}\cdots q_1 q_0,
$$
with qubit `q_0` the rightmost character and also the least-significant bit in basis-index arithmetic.

### 1.3.3 Canonical algebra sources

- Canonical `PauliTerm`: `src/quantum/qubitization_module.py`
- Canonical `PauliPolynomial`: `src/quantum/pauli_polynomial_class.py`
- Canonical JW ladder primitives:
  - `fermion_plus_operator(...)`
  - `fermion_minus_operator(...)`
- Canonical number operator surface:
  - `jw_number_operator(...)` in `src/quantum/hubbard_latex_python_pairs.py`

### 1.3.4 Surface-specific defaults

This manuscript does **not** pretend there is one universal default surface.

- Core builder functions in `src/quantum/hubbard_latex_python_pairs.py` often default to
  - `indexing="interleaved"`,
  - `pbc=True`.
- Hardcoded pipeline CLIs use different defaults, most notably
  - `ordering="blocked"`,
  - `boundary="periodic"`.
- `pipelines/hardcoded/adapt_pipeline.py` currently exposes HH `--boson-encoding binary` only.
- `pipelines/hardcoded/hubbard_pipeline.py` exposes a broader boson-encoding surface.

Whenever a formula depends on the surface, the surface is named locally.

## 1.4 Canonical code anchors

- `AGENTS.md`
- `README.md`
- `src/quantum/qubitization_module.py`
- `src/quantum/pauli_polynomial_class.py`
- `src/quantum/hubbard_latex_python_pairs.py`
- `src/quantum/hartree_fock_reference_state.py`
- `src/quantum/vqe_latex_python_pairs.py`
- `src/quantum/operator_pools/polaron_paop.py`
- `src/quantum/ed_hubbard_holstein.py`
- `src/quantum/spsa_optimizer.py`
- `src/quantum/drives_time_potential.py`
- `pipelines/hardcoded/hubbard_pipeline.py`
- `pipelines/hardcoded/adapt_pipeline.py`
- `pipelines/hardcoded/hh_continuation_stage_control.py`
- `pipelines/hardcoded/hh_continuation_scoring.py`
- `pipelines/hardcoded/handoff_state_bundle.py`
- `pipelines/exact_bench/cross_check_suite.py`
- `pipelines/exact_bench/hh_noise_hardware_validation.py`

# 2. Ordering, Indexing, and Register Layout

## 2.1 Site, spin, and mode indices

The site index is
$$
i\in\{0,1,\dots,L-1\},
$$
and the spin label is stored as
$$
\sigma\in\{\uparrow,\downarrow\}\equiv\{0,1\}.
$$

The fermion mode index is the JW qubit index.

### 2.1.1 Interleaved ordering

The interleaved map is
$$
p(i,\sigma)=2i+\sigma.
$$
So
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

These are the two cases implemented by `mode_index(...)`.

## 2.2 Pauli-word placement and basis-index extraction

If a Pauli letter acts on qubit `q`, then in a printed word of length `N_q` it sits at string position
$$
\operatorname{pos}(q)=N_q-1-q.
$$

If the computational-basis index is `k`, then the occupation bit on qubit `q` is
$$
b_q(k)=\left\lfloor\frac{k}{2^q}\right\rfloor \bmod 2=((k\gg q)\&1).
$$

So the printed bitstring and the integer basis index obey the same rightmost-`q_0` convention.

## 2.3 Full HH register layout

The fermion register uses
$$
N_{\mathrm{ferm}}=2L
$$
qubits.

If the local phonon cutoff is `n_ph_max`, then the local Hilbert dimension is
$$
d=n_{\mathrm{ph,max}}+1.
$$

The phonon qubits per site are
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

In qubit-index order the register is
$$
[\text{fermion qubits}\;|\;\text{site-0 phonon qubits}\;|\;\text{site-1 phonon qubits}\;|\;\cdots].
$$

In printed bitstring order `q_(N_q-1)...q_0`, the high-index phonon blocks appear on the left, so the displayed HH basis label is read as
$$
[\text{site-(L-1) phonons}\;|\;\cdots\;|\;\text{site-0 phonons}\;|\;\text{fermions}].
$$

Implemented surfaces:

- `src/quantum/hubbard_latex_python_pairs.py`
  - `mode_index`
  - `boson_qubits_per_site`
  - `phonon_qubit_indices_for_site`
- `src/quantum/hartree_fock_reference_state.py`
  - `bitstring_qn1_to_q0`
  - `hubbard_holstein_reference_state`

# 3. Fermionic Primitives and Direct Substitution

## 3.1 Jordan-Wigner ladder primitives

For mode `p`, the creation operator implemented by `fermion_plus_operator("JW", N_q, p)` is
$$
\hat c_p^{\dagger}
=
\frac{1}{2}\,e_{N_q-1}\cdots e_{p+1}x_p z_{p-1}\cdots z_0
-
\frac{i}{2}\,e_{N_q-1}\cdots e_{p+1}y_p z_{p-1}\cdots z_0.
$$

The annihilation operator implemented by `fermion_minus_operator("JW", N_q, p)` is
$$
\hat c_p
=
\frac{1}{2}\,e_{N_q-1}\cdots e_{p+1}x_p z_{p-1}\cdots z_0
+
\frac{i}{2}\,e_{N_q-1}\cdots e_{p+1}y_p z_{p-1}\cdots z_0.
$$

Equivalently, in operator notation,
$$
\hat c_p^{\dagger}=\frac{1}{2}(X_p-iY_p)\prod_{r=0}^{p-1}Z_r,
\qquad
\hat c_p=\frac{1}{2}(X_p+iY_p)\prod_{r=0}^{p-1}Z_r,
$$
but the repository’s printed words always follow the explicit `q_(N_q-1)...q_0` ordering above.

## 3.2 Number primitive

The implemented number operator is
$$
\hat n_p=\hat c_p^{\dagger}\hat c_p=\frac{I-Z_p}{2}.
$$

This is exactly the formula returned by `jw_number_operator(...)`.

## 3.3 Site densities and doublon operator

If the site `i` uses fermion modes
$$
p_{i\uparrow}=p(i,\uparrow),
\qquad
p_{i\downarrow}=p(i,\downarrow),
$$
then
$$
\hat n_{i\uparrow}=\frac{I-Z_{p_{i\uparrow}}}{2},
\qquad
\hat n_{i\downarrow}=\frac{I-Z_{p_{i\downarrow}}}{2}.
$$

So the full site density is
$$
\hat n_i=\hat n_{i\uparrow}+\hat n_{i\downarrow}
=\frac{I-Z_{p_{i\uparrow}}}{2}+\frac{I-Z_{p_{i\downarrow}}}{2}
=I-\frac{1}{2}\bigl(Z_{p_{i\uparrow}}+Z_{p_{i\downarrow}}\bigr).
$$

The onsite doublon operator is
$$
\hat d_i=\hat n_{i\uparrow}\hat n_{i\downarrow}
=\frac{I-Z_{p_{i\uparrow}}}{2}\cdot\frac{I-Z_{p_{i\downarrow}}}{2}
=\frac{1}{4}\Bigl(I-Z_{p_{i\uparrow}}-Z_{p_{i\downarrow}}+Z_{p_{i\uparrow}}Z_{p_{i\downarrow}}\Bigr).
$$

## 3.4 Worked ordering substitutions

For `L=3`, the two repository orderings become

### 3.4.1 Interleaved
$$
\begin{aligned}
p(0,\uparrow)&=0, & p(0,\downarrow)&=1,\\
p(1,\uparrow)&=2, & p(1,\downarrow)&=3,\\
p(2,\uparrow)&=4, & p(2,\downarrow)&=5.
\end{aligned}
$$

### 3.4.2 Blocked
$$
\begin{aligned}
p(0,\uparrow)&=0, & p(0,\downarrow)&=3,\\
p(1,\uparrow)&=1, & p(1,\downarrow)&=4,\\
p(2,\uparrow)&=2, & p(2,\downarrow)&=5.
\end{aligned}
$$

Implemented surfaces:

- `src/quantum/pauli_polynomial_class.py`
- `src/quantum/hubbard_latex_python_pairs.py`
- `src/quantum/qubitization_module.py`

# 4. Boson Primitives and Encodings

## 4.1 Local boson Hilbert space

At site `i`, the truncated phonon space has basis
$$
\{|0\rangle_i,|1\rangle_i,\dots,|n_{\mathrm{ph,max}}\rangle_i\}.
$$

The local annihilation operator is
$$
\hat b_i=\sum_{n=1}^{n_{\mathrm{ph,max}}}\sqrt{n}\,|n-1\rangle_i\langle n|,
$$
the creation operator is
$$
\hat b_i^{\dagger}=\sum_{n=0}^{n_{\mathrm{ph,max}}-1}\sqrt{n+1}\,|n+1\rangle_i\langle n|,
$$
the number operator is
$$
\hat n_{b,i}=\hat b_i^{\dagger}\hat b_i=\sum_{n=0}^{n_{\mathrm{ph,max}}} n\,|n\rangle_i\langle n|,
$$
and the displacement operator is
$$
\hat x_i=\hat b_i+\hat b_i^{\dagger}
=\sum_{n=0}^{n_{\mathrm{ph,max}}-1}\sqrt{n+1}\Bigl(|n+1\rangle_i\langle n|+|n\rangle_i\langle n+1|\Bigr).
$$

## 4.2 Binary encoding

In binary encoding,
$$
q_{\mathrm{pb}}=\max\{1,\lceil\log_2(n_{\mathrm{ph,max}}+1)\rceil\}.
$$

For site `i`, the phonon qubit block starts at
$$
q_{\mathrm{base}}(i)=2L+i\,q_{\mathrm{pb}}.
$$

So the local phonon qubits are
$$
q_{\mathrm{base}}(i),q_{\mathrm{base}}(i)+1,\dots,q_{\mathrm{base}}(i)+q_{\mathrm{pb}}-1.
$$

The binary implementation pads the local `d\times d` oscillator matrix into the `2^{q_pb}` qubit space and then decomposes it in the local Pauli basis:
$$
\hat M_i^{\mathrm{pad}}=
\sum_{\alpha\in\{I,X,Y,Z\}^{\otimes q_{\mathrm{pb}}}}
\frac{1}{2^{q_{\mathrm{pb}}}}
\operatorname{Tr}\bigl(P_{\alpha}^{\dagger}\hat M_i^{\mathrm{pad}}\bigr)
P_{\alpha},
$$
with `M = b, b†, n, x`.

That explicit Pauli decomposition is what `boson_local_operator_pauli_decomp(...)` and `boson_operator(...)` implement.

## 4.3 Unary encoding

In unary encoding,
$$
q_{\mathrm{pb}}=n_{\mathrm{ph,max}}+1.
$$

The qubit corresponding to local level `n` at site `i` is
$$
q(i,n)=2L+i(n_{\mathrm{ph,max}}+1)+n.
$$

The unary one-hot projector is
$$
\hat n_{i,n}=|1\rangle\langle 1|_{q(i,n)}=\frac{I-Z_{q(i,n)}}{2}.
$$

Therefore the unary phonon number operator is already explicit in single-qubit `Z` primitives:
$$
\hat n_{b,i}=
\sum_{n=0}^{n_{\mathrm{ph,max}}}n\,\hat n_{i,n}
=
\sum_{n=0}^{n_{\mathrm{ph,max}}}n\,\frac{I-Z_{q(i,n)}}{2}.
$$

Define
$$
\sigma_q^{+}=\frac{X_q+iY_q}{2},
\qquad
\sigma_q^{-}=\frac{X_q-iY_q}{2}.
$$
Then the unary ladder operators are
$$
\hat b_i^{\dagger}=
\sum_{n=0}^{n_{\mathrm{ph,max}}-1}
\sqrt{n+1}\,\sigma_{q(i,n)}^{+}\sigma_{q(i,n+1)}^{-},
$$
$$
\hat b_i=
\sum_{n=0}^{n_{\mathrm{ph,max}}-1}
\sqrt{n+1}\,\sigma_{q(i,n)}^{-}\sigma_{q(i,n+1)}^{+}.
$$

So the unary displacement operator is
$$
\hat x_i=\hat b_i+\hat b_i^{\dagger}
=\frac{1}{2}
\sum_{n=0}^{n_{\mathrm{ph,max}}-1}
\sqrt{n+1}\Bigl(
X_{q(i,n)}X_{q(i,n+1)}+Y_{q(i,n)}Y_{q(i,n+1)}
\Bigr).
$$

## 4.4 Phonon vacuum

### 4.4.1 Binary vacuum

The binary phonon vacuum is the all-zero phonon register:
$$
|\mathrm{vac}_{\mathrm{ph}}\rangle_{\mathrm{binary}}=|0\cdots 0\rangle.
$$

### 4.4.2 Unary vacuum

The unary phonon vacuum is one-hot at `n=0` for each site, so each site block contributes
$$
0^{q_{\mathrm{pb}}-1}1
$$
in printed `q_(N_q-1)...q_0` order.

Implemented surfaces:

- `src/quantum/hubbard_latex_python_pairs.py`
  - `boson_qubits_per_site`
  - `phonon_qubit_indices_for_site`
  - `boson_operator`
  - `boson_number_operator`
  - `boson_displacement_operator`
- `src/quantum/hartree_fock_reference_state.py`
  - `_phonon_vacuum_bitstring`

# 5. Hubbard Hamiltonian by Explicit Substitution

## 5.1 Kinetic term

The Hubbard hopping term is
$$
\hat H_t=-J\sum_{\langle i,j\rangle,\sigma}
\left(
\hat c_{i\sigma}^{\dagger}\hat c_{j\sigma}+
\hat c_{j\sigma}^{\dagger}\hat c_{i\sigma}
\right).
$$

Now substitute
$$
p_i=p(i,\sigma),
\qquad
p_j=p(j,\sigma),
\qquad
p_< = \min\{p_i,p_j\},
\qquad
p_> = \max\{p_i,p_j\}.
$$

After inserting the JW ladder primitives and collecting the Hermitian pair, the repository’s hopping generator becomes
$$
\hat c_{p_i}^{\dagger}\hat c_{p_j}+\hat c_{p_j}^{\dagger}\hat c_{p_i}
=
\frac{1}{2}\Bigl(
X_{p_>}Z_{p_>-1}\cdots Z_{p_<+1}X_{p_<}
+
Y_{p_>}Z_{p_>-1}\cdots Z_{p_<+1}Y_{p_<}
\Bigr).
$$

So the fully substituted kinetic term is
$$
\hat H_t
=
-\frac{J}{2}
\sum_{\langle i,j\rangle,\sigma}
\Bigl(
X_{p_>(i,j,\sigma)}Z\cdots ZX_{p_<(i,j,\sigma)}
+
Y_{p_>(i,j,\sigma)}Z\cdots ZY_{p_<(i,j,\sigma)}
\Bigr).
$$

## 5.2 Onsite interaction

The onsite interaction starts as
$$
\hat H_U=U\sum_i \hat n_{i\uparrow}\hat n_{i\downarrow}.
$$

Substitute the number primitives:
$$
\hat H_U
=
U\sum_i
\frac{I-Z_{p_{i\uparrow}}}{2}
\frac{I-Z_{p_{i\downarrow}}}{2}.
$$

Multiply explicitly:
$$
\hat H_U
=
\frac{U}{4}\sum_i
\Bigl(
I-Z_{p_{i\uparrow}}-Z_{p_{i\downarrow}}+Z_{p_{i\uparrow}}Z_{p_{i\downarrow}}
\Bigr).
$$

## 5.3 Static potential term

The static site-potential term is
$$
\hat H_v=-\sum_{i,\sigma}v_i\hat n_{i\sigma}.
$$

Substitute `\hat n_{i\sigma}=(I-Z_{p(i,\sigma)})/2`:
$$
\hat H_v
=
-\sum_{i,\sigma}v_i\frac{I-Z_{p(i,\sigma)}}{2}
=
-\frac{1}{2}\sum_{i,\sigma}v_i I
+\frac{1}{2}\sum_{i,\sigma}v_i Z_{p(i,\sigma)}.
$$

If the potential is uniform, `v_i=dv`, then this becomes
$$
\hat H_v
=-L\,dv\,I+\frac{dv}{2}\sum_{i,\sigma}Z_{p(i,\sigma)}.
$$

## 5.4 Fully substituted Hubbard Hamiltonian

The full Hubbard Hamiltonian is therefore
$$
\hat H_{\mathrm{Hub}}
=
-\frac{J}{2}
\sum_{\langle i,j\rangle,\sigma}
\Bigl(
X_{p_>}Z\cdots ZX_{p_<}+Y_{p_>}Z\cdots ZY_{p_<}
\Bigr)
+\frac{U}{4}\sum_i
\Bigl(
I-Z_{p_{i\uparrow}}-Z_{p_{i\downarrow}}+Z_{p_{i\uparrow}}Z_{p_{i\downarrow}}
\Bigr)
-\frac{1}{2}\sum_{i,\sigma}v_i I
+\frac{1}{2}\sum_{i,\sigma}v_i Z_{p(i,\sigma)}.
$$

Implemented surfaces:

- `src/quantum/hubbard_latex_python_pairs.py`
  - `build_hubbard_kinetic`
  - `build_hubbard_onsite`
  - `build_hubbard_potential`
  - `build_hubbard_hamiltonian`
- `src/quantum/vqe_latex_python_pairs.py`
  - `hubbard_hop_term`
  - `hubbard_onsite_term`
  - `hubbard_potential_term`

# 6. Hubbard-Holstein Hamiltonian by Explicit Substitution

## 6.1 Phonon energy

The phonon energy is
$$
\hat H_{\mathrm{ph}}=\omega_0\sum_i\left(\hat n_{b,i}+\frac{1}{2}I\right).
$$

### 6.1.1 Unary explicit form

Insert the unary number operator:
$$
\hat H_{\mathrm{ph}}^{\mathrm{unary}}
=
\omega_0\sum_i
\left(
\sum_{n=0}^{n_{\mathrm{ph,max}}}
 n\,\frac{I-Z_{q(i,n)}}{2}
+\frac{1}{2}I
\right).
$$

### 6.1.2 Binary explicit form

Insert the binary Pauli decomposition of `\hat n_{b,i}`:
$$
\hat n_{b,i}
=
\sum_{\alpha}
\frac{1}{2^{q_{\mathrm{pb}}}}
\operatorname{Tr}\bigl(P_{\alpha}^{\dagger}\hat n_{b,i}^{\mathrm{pad}}\bigr)
P_{i,\alpha}.
$$
So the binary phonon energy is
$$
\hat H_{\mathrm{ph}}^{\mathrm{binary}}
=
\omega_0\sum_i
\left(
\sum_{\alpha}
\frac{1}{2^{q_{\mathrm{pb}}}}
\operatorname{Tr}\bigl(P_{\alpha}^{\dagger}\hat n_{b,i}^{\mathrm{pad}}\bigr)
P_{i,\alpha}
+\frac{1}{2}I
\right).
$$

## 6.2 Electron-phonon coupling

The Holstein coupling starts as
$$
\hat H_g=g\sum_i \hat x_i(\hat n_i-I).
$$

Now substitute the explicit site density from Section 3.3:
$$
\hat n_i-I
=
\left(I-\frac{1}{2}(Z_{p_{i\uparrow}}+Z_{p_{i\downarrow}})\right)-I
=-\frac{1}{2}(Z_{p_{i\uparrow}}+Z_{p_{i\downarrow}}).
$$

So the coupling becomes
$$
\hat H_g
=
-\frac{g}{2}\sum_i
\hat x_i\bigl(Z_{p_{i\uparrow}}+Z_{p_{i\downarrow}}\bigr).
$$

This is the most important substitution in the HH layer: the fermion density shift `\hat n_i-I` is not left abstract; it is reduced all the way to explicit `Z` operators.

### 6.2.1 Unary explicit form

Insert the unary displacement operator:
$$
\hat x_i
=
\frac{1}{2}
\sum_{n=0}^{n_{\mathrm{ph,max}}-1}
\sqrt{n+1}
\Bigl(
X_{q(i,n)}X_{q(i,n+1)}+Y_{q(i,n)}Y_{q(i,n+1)}
\Bigr).
$$
Therefore
$$
\hat H_g^{\mathrm{unary}}
=
-\frac{g}{4}
\sum_i
\left[
\sum_{n=0}^{n_{\mathrm{ph,max}}-1}
\sqrt{n+1}
\Bigl(
X_{q(i,n)}X_{q(i,n+1)}+Y_{q(i,n)}Y_{q(i,n+1)}
\Bigr)
\right]
\bigl(Z_{p_{i\uparrow}}+Z_{p_{i\downarrow}}\bigr).
$$

### 6.2.2 Binary explicit form

Insert the binary Pauli decomposition of `\hat x_i`:
$$
\hat x_i
=
\sum_{\alpha}
\frac{1}{2^{q_{\mathrm{pb}}}}
\operatorname{Tr}\bigl(P_{\alpha}^{\dagger}\hat x_i^{\mathrm{pad}}\bigr)
P_{i,\alpha}.
$$
Then
$$
\hat H_g^{\mathrm{binary}}
=
-\frac{g}{2}
\sum_i
\left[
\sum_{\alpha}
\frac{1}{2^{q_{\mathrm{pb}}}}
\operatorname{Tr}\bigl(P_{\alpha}^{\dagger}\hat x_i^{\mathrm{pad}}\bigr)
P_{i,\alpha}
\right]
\bigl(Z_{p_{i\uparrow}}+Z_{p_{i\downarrow}}\bigr).
$$

## 6.3 Time-dependent density drive

The implemented HH drive builder uses the physical increment
$$
\delta v_i(t)=v_i(t)-v_{0,i}
$$
and constructs
$$
\hat H_{\mathrm{drive}}(t)=\sum_{i,\sigma}\delta v_i(t)\hat n_{i\sigma}.
$$

Now substitute the number operator explicitly:
$$
\hat H_{\mathrm{drive}}(t)
=
\sum_{i,\sigma}\delta v_i(t)\frac{I-Z_{p(i,\sigma)}}{2}
=
\frac{1}{2}\sum_{i,\sigma}\delta v_i(t)I
-\frac{1}{2}\sum_{i,\sigma}\delta v_i(t)Z_{p(i,\sigma)}.
$$

Because there are two spins per site, the identity contribution can also be written as
$$
\frac{1}{2}\sum_{i,\sigma}\delta v_i(t)I
=\sum_i \delta v_i(t)I.
$$

The implementation route in `build_hubbard_holstein_drive(...)` is explicit:

- `build_hubbard_potential(...)` assumes `H_v=-\sum_{i,\sigma} v_i n_{i\sigma}`,
- therefore the HH drive builder passes `v_for_existing(i)=-\delta v_i(t)`.

### 6.3.1 Implemented density-drive waveform surface

The runtime density-drive helper in `src/quantum/drives_time_potential.py` uses the scalar waveform
$$
f(\tau)=A\sin(\omega\tau+\phi)\exp\!\left(-\frac{\tau^2}{2\bar t^2}\right),
$$
with the sampled time
$$
\tau=t+t_0.
$$
So the site-resolved potential surface is
$$
v_i(t)=s_i f(t+t_0),
$$
where the spatial weights are
$$
s_i\in\{(-1)^i,\,[+1,-1]\text{ for }L=2,\,\text{custom user weights}\}.
$$

The runtime Pauli-coefficient map is then
$$
\Delta c[Z_{p(i,\sigma)}](t)=-\frac{1}{2}v_i(t),
$$
and, when requested,
$$
\Delta c[I](t)=\sum_i v_i(t).
$$

## 6.4 Two implemented HH assembly surfaces

There are two distinct implemented HH assembly surfaces and this manuscript states both instead of pretending they are identical.

### 6.4.1 Core Hamiltonian builder surface

`build_hubbard_holstein_hamiltonian(...)` assembles
$$
\hat H_{\mathrm{HH,core}}(t)=\hat H_t+\hat H_U+\hat H_{\mathrm{ph}}+\hat H_g+\hat H_{\mathrm{drive}}(t),
$$
with the Hubbard part built using `v=None`.

### 6.4.2 Variational HH ansatz surface

The HH ansatz classes in `src/quantum/vqe_latex_python_pairs.py` expose the more general grouped surface
$$
\hat H_{\mathrm{HH,ansatz}}(t)=\hat H_t+\hat H_U+\hat H_v+\hat H_{\mathrm{ph}}+\hat H_g+\hat H_{\mathrm{drive}}(t),
$$
where the static fermion potential `\hat H_v` may be included separately.

## 6.5 Fully substituted HH master form

Combining the substitutions above gives the primitive-closed HH expression
$$
\begin{aligned}
\hat H_{\mathrm{HH}}(t)
=&-\frac{J}{2}
\sum_{\langle i,j\rangle,\sigma}
\Bigl(
X_{p_>}Z\cdots ZX_{p_<}+Y_{p_>}Z\cdots ZY_{p_<}
\Bigr)\\
&+\frac{U}{4}\sum_i
\Bigl(
I-Z_{p_{i\uparrow}}-Z_{p_{i\downarrow}}+Z_{p_{i\uparrow}}Z_{p_{i\downarrow}}
\Bigr)\\
&-\frac{1}{2}\sum_{i,\sigma}v_i I
+\frac{1}{2}\sum_{i,\sigma}v_i Z_{p(i,\sigma)}\\
&+\omega_0\sum_i\left(\hat n_{b,i}+\frac{1}{2}I\right)\\
&-\frac{g}{2}\sum_i \hat x_i\bigl(Z_{p_{i\uparrow}}+Z_{p_{i\downarrow}}\bigr)\\
&+\frac{1}{2}\sum_{i,\sigma}\delta v_i(t)I
-\frac{1}{2}\sum_{i,\sigma}\delta v_i(t)Z_{p(i,\sigma)}.
\end{aligned}
$$

At this point only `\hat n_{b,i}` and `\hat x_i` remain as boson primitives, and those are already given explicitly in binary and unary form in Section 4.

Implemented surfaces:

- `src/quantum/hubbard_latex_python_pairs.py`
  - `build_holstein_phonon_energy`
  - `build_holstein_coupling`
  - `build_hubbard_holstein_drive`
  - `build_hubbard_holstein_hamiltonian`
- `src/quantum/drives_time_potential.py`
  - `GaussianSinusoidSitePotential`
  - `DensityDriveTemplate`
  - `TimeDependentOnsiteDensityDrive`
  - `build_gaussian_sinusoid_density_drive`

# 7. Reference States and Exact Sector ED

## 7.1 Fermionic Hartree-Fock determinant

The repository’s Hartree-Fock reference fills the first `n_alpha` up-spin orbitals and first `n_beta` down-spin orbitals in the chosen ordering.

If the occupied qubit set is `Q_occ`, then the HF bitstring in `q_(N_q-1)...q_0` order is the string with
$$
\text{bit}(q)=1 \iff q\in Q_{\mathrm{occ}}.
$$

The corresponding basis index is
$$
k_{\mathrm{HF}}=\sum_{q\in Q_{\mathrm{occ}}}2^q.
$$

The half-filled default used by the HH reference helper is
$$
(N_{\uparrow},N_{\downarrow})=
\left(\left\lceil\frac{L}{2}\right\rceil,\left\lfloor\frac{L}{2}\right\rfloor\right).
$$

## 7.2 Hubbard-Holstein reference state

The HH reference state is
$$
|\psi_{\mathrm{ref}}^{\mathrm{HH}}\rangle
=|\mathrm{vac}_{\mathrm{ph}}\rangle\otimes|\Phi_{\mathrm{HF}}\rangle.
$$

The printed basis label is built exactly as
$$
\texttt{full\_bitstring}
=
\texttt{phonon\_vacuum\_bitstring}
+
\texttt{hf\_fermion\_bitstring}.
$$

So the phonon block is displayed on the left and the fermion block on the right.

## 7.3 Exact HH sector basis

The exact ED surface in `src/quantum/ed_hubbard_holstein.py` is independent of the Pauli-polynomial builder. It constructs a basis of physical states
$$
|\text{fermion bits}; n_0,n_1,\dots,n_{L-1}\rangle
$$
subject to

- fixed fermion sector `(N_up, N_dn)`,
- unrestricted local phonon occupations inside the truncation `0\le n_i\le n_ph_max`.

Binary and unary encodings change only the map from a physical state to the computational basis index:

### 7.3.1 Binary index map

If the fermion basis bits form the integer `f_bits`, then
$$
\text{index}_{\mathrm{binary}}
=f_{\mathrm{bits}}+\sum_{i=0}^{L-1} n_i\,2^{2L+i q_{\mathrm{pb}}}.
$$
This is the compact register interpretation implemented by `encode_state_to_qubit_index(...)`.

### 7.3.2 Unary index map

If the same physical state uses unary phonon encoding, then the computational-basis index is built by setting one-hot qubits `q(i,n_i)` in addition to the fermion bits. The physical matrix elements are unchanged; only the encoded index changes.

## 7.4 Exact HH matrix elements

The exact ED builder constructs
$$
\hat H_{\mathrm{HH}}=\hat H_t+\hat H_U+\hat H_{\mathrm{ph}}+\hat H_g+\hat H_{\mathrm{drive}}
$$
directly in the occupation basis.

### 7.4.1 Diagonal elements

For a basis state with fermion occupations `n_{i\uparrow}, n_{i\downarrow}` and phonon occupations `n_i^{\mathrm{ph}}`, the diagonal contribution is
$$
\sum_i
\Bigl[
U n_{i\uparrow}n_{i\downarrow}
+\delta v_i(n_{i\uparrow}+n_{i\downarrow})
+\omega_0 n_i^{\mathrm{ph}}
\Bigr]
+
\frac{\omega_0L}{2}
$$
when `include_zero_point=True`.

### 7.4.2 Hopping matrix elements

If `\hat c_p^{\dagger}\hat c_q` is allowed on the fermion bitstring, then the off-diagonal matrix element is
$$
-J\times(\text{fermionic sign from JW parity}).
$$
The helper `_apply_cdag_c(...)` computes the sign explicitly by counting occupied qubits below the action site.

### 7.4.3 Electron-phonon matrix elements

If the site density gives the prefactor
$$
g(n_i-1),
$$
then `\hat x_i=\hat b_i+\hat b_i^{\dagger}` contributes
$$
\langle n_i+1|\hat x_i|n_i\rangle=\sqrt{n_i+1},
\qquad
\langle n_i-1|\hat x_i|n_i\rangle=\sqrt{n_i}.
$$
So the ED builder inserts off-diagonal matrix elements
$$
g(n_i-1)\sqrt{n_i+1}
\quad\text{or}\quad
g(n_i-1)\sqrt{n_i}
$$
for phonon raising or lowering respectively.

Implemented surfaces:

- `src/quantum/hartree_fock_reference_state.py`
- `src/quantum/ed_hubbard_holstein.py`

# 8. Statevector Action, Expectation, and Exponential Primitives

## 8.1 Basis-state primitive

The statevector primitive `basis_state(N_q, bitstring)` places amplitude `1` at
$$
\text{index}=\operatorname{int}(\texttt{bitstring},2),
$$
with the bitstring already in `q_(N_q-1)...q_0` order.

## 8.2 Explicit Pauli-word action

Let the Pauli word be
$$
P=\sigma_{N_q-1}\cdots\sigma_1\sigma_0,
\qquad
\sigma_q\in\{e,x,y,z\}.
$$

Define the bit-flip set
$$
F(P)=\{q\mid \sigma_q\in\{x,y\}\},
$$
and the phase on basis index `k` with bits `b_q(k)` as
$$
\phi_P(k)=
\prod_{q:\sigma_q=z}(-1)^{b_q(k)}
\prod_{q:\sigma_q=y} i\,(-1)^{b_q(k)}.
$$

Then the implemented Pauli action is
$$
P|k\rangle=\phi_P(k)\,|k\oplus \chi_{F(P)}\rangle,
$$
so on amplitudes,
$$
(P\psi)_{k\oplus\chi_{F(P)}}=\phi_P(k)\,\psi_k.
$$

This is exactly what `apply_pauli_string(...)` does by looping over qubits and applying `x`, `y`, and `z` cases with the repository’s rightmost-`q_0` convention.

## 8.3 Expectation values

If
$$
\hat H=\sum_j h_j P_j,
$$
then the variational energy is
$$
E(\psi)=\langle\psi|\hat H|\psi\rangle
=\sum_j h_j\langle\psi|P_j|\psi\rangle.
$$

`expval_pauli_polynomial(...)` evaluates this term by term.

The compiled one-apply backend instead computes
$$
E(\psi)=\operatorname{Re}\langle\psi|\hat H\psi\rangle,
$$
after compiling the polynomial action once.

## 8.4 Pauli rotations

For a single Pauli word `P` with `P^2=I`, the implemented rotation primitive is
$$
R_P(\varphi)=\exp\!\left(-i\frac{\varphi}{2}P\right)
=\cos\left(\frac{\varphi}{2}\right)I
-i\sin\left(\frac{\varphi}{2}\right)P.
$$

So the updated state is
$$
R_P(\varphi)|\psi\rangle
=
\cos\left(\frac{\varphi}{2}\right)|\psi\rangle
-i\sin\left(\frac{\varphi}{2}\right)P|\psi\rangle.
$$

## 8.5 Exponential of a Pauli polynomial

If
$$
\hat H=\sum_j h_j P_j,
$$
then the code applies the first-order ordered product
$$
\exp(-i\theta \hat H)|\psi\rangle
\approx
\prod_j \exp(-i\theta h_j P_j)|\psi\rangle.
$$

Because `apply_pauli_rotation(...)` expects the angle `\varphi` in
$$
\exp\!\left(-i\frac{\varphi}{2}P\right),
$$
`apply_exp_pauli_polynomial(...)` uses
$$
\varphi_j=2\theta h_j.
$$

So the implemented ordered update is
$$
|\psi_{\mathrm{out}}\rangle
\approx
\prod_j
\left[
\cos(\theta h_j)I-i\sin(\theta h_j)P_j
\right]
|\psi_{\mathrm{in}}\rangle,
$$
after optionally dropping identity terms and sorting the Pauli words deterministically.

## 8.6 Exact sector energy target

The exact HH target used by the hardcoded VQE surfaces is a sector-filtered exact energy, not an unrestricted full-Hilbert minimum. Only the fermionic particle numbers are fixed; the phonon qubits are left unconstrained. This is what `exact_ground_energy_sector_hh(...)` and the ED basis logic provide.

Write the total Hilbert space as
$$
\mathcal H = \mathcal H_{\mathrm{ferm}} \otimes \mathcal H_{\mathrm{ph}},
$$
with qubit layout
$$
[\,2L\ \text{fermion qubits}\mid L\cdot q_{\mathrm{pb}}\ \text{phonon qubits}\,].
$$

Let $\mathcal I_\alpha$ and $\mathcal I_\beta$ denote the fermion-qubit index sets for spin-up and spin-down. In the two supported orderings,
$$
\mathcal I_\alpha =
\begin{cases}
\{0,\dots,L-1\}, & \texttt{blocked},\\
\{0,2,\dots,2L-2\}, & \texttt{interleaved},
\end{cases}
\qquad
\mathcal I_\beta =
\begin{cases}
\{L,\dots,2L-1\}, & \texttt{blocked},\\
\{1,3,\dots,2L-1\}, & \texttt{interleaved}.
\end{cases}
$$

Using the qubit number operator
$$
\hat n_q = \frac{I-Z_q}{2},
$$
define the fermion-number operators
$$
\hat N_\alpha = \sum_{q\in\mathcal I_\alpha}\hat n_q,
\qquad
\hat N_\beta = \sum_{q\in\mathcal I_\beta}\hat n_q.
$$

For target counts $(N_\alpha,N_\beta)$, the HH sector is
$$
\mathcal H_{N_\alpha,N_\beta}
=
\left\{\,|\psi\rangle\in\mathcal H:
\hat N_\alpha|\psi\rangle=N_\alpha|\psi\rangle,\ 
\hat N_\beta|\psi\rangle=N_\beta|\psi\rangle
\,\right\},
$$
with no restriction on the phonon occupation.

Equivalently, in the computational basis $|f,p\rangle$ with fermion bits
$f=(f_q)$ and phonon bits $p$, the projector onto this sector is
$$
\hat\Pi_{N_\alpha,N_\beta}
=
\sum_{f,p}
\delta_{\sum_{q\in\mathcal I_\alpha} f_q,\;N_\alpha}
\,
\delta_{\sum_{q\in\mathcal I_\beta} f_q,\;N_\beta}
\,
|f,p\rangle\langle f,p|.
$$

So for an arbitrary state
$$
|\psi\rangle=\sum_{f,p} c_{f,p}|f,p\rangle,
$$
the sector-filtered component is
$$
\hat\Pi_{N_\alpha,N_\beta}|\psi\rangle
=
\sum_{f,p}
\delta_{\sum_{q\in\mathcal I_\alpha} f_q,\;N_\alpha}
\,
\delta_{\sum_{q\in\mathcal I_\beta} f_q,\;N_\beta}
\,
c_{f,p}|f,p\rangle.
$$

If normalization is needed, use
$$
|\psi_{N_\alpha,N_\beta}\rangle
=
\frac{\hat\Pi_{N_\alpha,N_\beta}|\psi\rangle}
{\sqrt{\langle\psi|\hat\Pi_{N_\alpha,N_\beta}|\psi\rangle}}.
$$

The exact sector energy target is then the lowest eigenvalue of the projected Hamiltonian,
$$
E_0^{(N_\alpha,N_\beta)}
=
\min_{\substack{|\psi\rangle\in\mathcal H_{N_\alpha,N_\beta}\\ \langle\psi|\psi\rangle=1}}
\langle\psi|\hat H|\psi\rangle
=
\lambda_{\min}\!\left(\hat\Pi_{N_\alpha,N_\beta}\hat H\hat\Pi_{N_\alpha,N_\beta}\right).
$$

In matrix form, the implementation builds the full Hamiltonian matrix $M$,
keeps only the basis indices satisfying the two fermion-count constraints,
forms the submatrix
$$
M_{\mathrm{sector}} = M[\mathcal B_{N_\alpha,N_\beta},\mathcal B_{N_\alpha,N_\beta}],
$$
and returns
$$
E_0^{(N_\alpha,N_\beta)}=\lambda_{\min}(M_{\mathrm{sector}}).
$$

Implemented surfaces:

- `src/quantum/vqe_latex_python_pairs.py`
  - `basis_state`
  - `apply_pauli_string`
  - `expval_pauli_polynomial`
  - `expval_pauli_polynomial_one_apply`
  - `apply_pauli_rotation`
  - `apply_exp_pauli_polynomial`
  - `exact_ground_energy_sector_hh`

# 9. Current Hardcoded Fixed VQE Ansatz Families

The user-facing fixed ansatz choices in `pipelines/hardcoded/hubbard_pipeline.py` are

- `uccsd`,
- `hva`,
- `hh_hva`,
- `hh_hva_tw`,
- `hh_hva_ptw`.

The ADAPT pool/replay-family surfaces are documented separately in Sections 10 and 13.

## 9.1 Hubbard layerwise ansatz: `hva`

For the static Hubbard problem, the fixed layerwise ansatz groups the Hamiltonian into

1. hopping,
2. onsite interaction,
3. static potential if present.

Write
$$
\hat H_{\mathrm{Hub}}=\hat H_t+\hat H_U+\hat H_v,
$$
with
$$
\hat H_t=-t\sum_{\langle i,j\rangle,\sigma}\left(\hat c_{i\sigma}^{\dagger}\hat c_{j\sigma}+\hat c_{j\sigma}^{\dagger}\hat c_{i\sigma}\right),
$$
$$
\hat H_U=U\sum_i\hat n_{i\uparrow}\hat n_{i\downarrow},
$$
$$
\hat H_v=\sum_{i,\sigma} v_i \hat n_{i\sigma}.
$$

For layer `\ell`, if
$$
\hat H_{\mathrm{group}}=\sum_{P\in\mathrm{split}(\hat H_{\mathrm{group}})} h_P P,
$$
then the implemented Hubbard layerwise unitary is
$$
\hat U_{\mathrm{Hub\text{-}layer}}^{(\ell)}
=
\prod_{P\in\mathrm{split}(\hat H_t)} e^{-i\theta_t^{(\ell)}h_PP}
\prod_{P\in\mathrm{split}(\hat H_U)} e^{-i\theta_U^{(\ell)}h_PP}
\prod_{P\in\mathrm{split}(\hat H_v)} e^{-i\theta_v^{(\ell)}h_PP}.
$$

So all split Pauli terms inside one physical group share one parameter per layer.

## 9.2 Hubbard hardcoded UCCSD surface: `uccsd`

Relative to the Hartree-Fock occupied/virtual partition, the primitive Hermitian single- and double-excitation generators are
$$
G_{i\to a}^{(\sigma)}
=
i\left(\hat c_{a\sigma}^{\dagger}\hat c_{i\sigma}-\hat c_{i\sigma}^{\dagger}\hat c_{a\sigma}\right),
$$
$$
G_{ij\to ab}^{(\sigma,\sigma')}
=
i\left(
\hat c_{a\sigma}^{\dagger}\hat c_{b\sigma'}^{\dagger}\hat c_{j\sigma'}\hat c_{i\sigma}
-
\hat c_{i\sigma}^{\dagger}\hat c_{j\sigma'}^{\dagger}\hat c_{b\sigma'}\hat c_{a\sigma}
\right).
$$

If `\mathcal S` and `\mathcal D` denote the generated single and double excitation sets, then the fixed hardcoded CLI surface `--vqe-ansatz uccsd` dispatches to the **layerwise** form
$$
\hat U_{\mathrm{UCCSD\text{-}layer}}^{(\ell)}
=
\prod_{G\in\mathcal S} e^{-i\theta_s^{(\ell)}G}
\prod_{G\in\mathcal D} e^{-i\theta_d^{(\ell)}G}.
$$

So the live fixed VQE surface shares one parameter across all singles and one across all doubles per layer.

The repo also contains the lower-level base class `HardcodedUCCSDAnsatz`,
$$
\hat U_{\mathrm{UCCSD\text{-}base}}(\theta)=\prod_k e^{-i\theta_k G_k},
$$
with one independent parameter per excitation generator. ADAPT pool builders reuse those base generators, but the fixed hardcoded VQE CLI chooses the shared-parameter layerwise surface above.

## 9.3 HH layerwise ansatz: `hh_hva`

The layerwise HH ansatz groups the Hamiltonian into physical sectors:

1. hopping,
2. onsite interaction,
3. static potential if present,
4. phonon energy,
5. electron-phonon coupling,
6. drive if present.

For layer `\ell`, write the split Pauli terms of each physical group as
$$
\hat H_{\mathrm{group}}=\sum_{P\in\mathrm{split}(\hat H_{\mathrm{group}})} h_P P.
$$
Then the implemented layerwise unitary is
$$
\hat U_{\mathrm{layer}}^{(\ell)}
=
\prod_{P\in\mathrm{split}(\hat H_t)} e^{-i\theta_t^{(\ell)}h_PP}
\prod_{P\in\mathrm{split}(\hat H_U)} e^{-i\theta_U^{(\ell)}h_PP}
\prod_{P\in\mathrm{split}(\hat H_v)} e^{-i\theta_v^{(\ell)}h_PP}
\prod_{P\in\mathrm{split}(\hat H_{\mathrm{ph}})} e^{-i\theta_{\mathrm{ph}}^{(\ell)}h_PP}
\prod_{P\in\mathrm{split}(\hat H_g)} e^{-i\theta_g^{(\ell)}h_PP}
\prod_{P\in\mathrm{split}(\hat H_{\mathrm{drive}})} e^{-i\theta_{\mathrm{drive}}^{(\ell)}h_PP}.
$$

The crucial point is that **all split Pauli terms inside one physical group share one parameter per layer**.

## 9.4 HH Pauli-termwise ansatz: `hh_hva_tw`

The Pauli-termwise HH ansatz removes that sharing. Every single split Pauli term gets its own parameter:
$$
\hat U_{\mathrm{tw}}^{(\ell)}
=
\prod_{P\in\mathrm{split}(\hat H_t)} e^{-i\theta_{t,P}^{(\ell)}h_PP}
\prod_{P\in\mathrm{split}(\hat H_U)} e^{-i\theta_{U,P}^{(\ell)}h_PP}
\prod_{P\in\mathrm{split}(\hat H_v)} e^{-i\theta_{v,P}^{(\ell)}h_PP}
\prod_{P\in\mathrm{split}(\hat H_{\mathrm{ph}})} e^{-i\theta_{{\mathrm{ph}},P}^{(\ell)}h_PP}
\prod_{P\in\mathrm{split}(\hat H_g)} e^{-i\theta_{g,P}^{(\ell)}h_PP}
\prod_{P\in\mathrm{split}(\hat H_{\mathrm{drive}})} e^{-i\theta_{{\mathrm{drive}},P}^{(\ell)}h_PP}.
$$

This is more expressive, but the split single-Pauli factors need not preserve the fermion sector individually.

## 9.5 HH physical-termwise ansatz: `hh_hva_ptw`

The physical-termwise HH ansatz keeps one parameter per physical generator before Pauli splitting.

Its Hamiltonian contract is
$$
\hat H_{\mathrm{HH}}(t)=\hat H_t+\hat H_U+\hat H_{\mathrm{ph}}+\hat H_g+\hat H_{\mathrm{drive}}(t),
$$
with
$$
\hat H_t=-J\sum_{\langle i,j\rangle,\sigma}\left(\hat c_{i\sigma}^{\dagger}\hat c_{j\sigma}+\hat c_{j\sigma}^{\dagger}\hat c_{i\sigma}\right),
$$
$$
\hat H_U=U\sum_i\hat n_{i\uparrow}\hat n_{i\downarrow},
$$
$$
\hat H_{\mathrm{ph}}=\omega_0\sum_i\left(\hat n_{b,i}+\frac12\right),
$$
$$
\hat H_g=g\sum_i \hat x_i\bigl(\hat n_i-\mathbb 1\bigr),\qquad \hat n_i=\hat n_{i\uparrow}+\hat n_{i\downarrow},\ \hat x_i=b_i+b_i^{\dagger},
$$
$$
\hat H_{\mathrm{drive}}(t)=\sum_{i,\sigma}\bigl(v_i(t)-v_{0,i}\bigr)\hat n_{i\sigma}.
$$

For one layer,
$$
\hat U_{\mathrm{ptw}}^{(\ell)}=
\prod_{\langle i,j\rangle,\sigma}
 e^{-i\theta_{ij\sigma}^{(\ell)}\hat H_{ij\sigma}^{(t)}}
\prod_i e^{-i\theta_{U,i}^{(\ell)}\hat H_i^{(U)}}
\prod_{i,\sigma} e^{-i\theta_{v,i\sigma}^{(\ell)}\hat H_{i\sigma}^{(v)}}
\prod_i e^{-i\theta_{\mathrm{ph},i}^{(\ell)}\omega_0\hat n_{b,i}}
\prod_i e^{-i\theta_{g,i}^{(\ell)} g\hat x_i(\hat n_i-I)}
\prod_{i,\sigma} e^{-i\theta_{d,i\sigma}^{(\ell)}\delta v_i(t)\hat n_{i\sigma}}.
$$

Here
$$
\hat H_{ij\sigma}^{(t)}=-J\left(\hat c_{i\sigma}^{\dagger}\hat c_{j\sigma}+\hat c_{j\sigma}^{\dagger}\hat c_{i\sigma}\right),
$$
$$
\hat H_i^{(U)}=U\hat n_{i\uparrow}\hat n_{i\downarrow},
$$
$$
\hat H_{i\sigma}^{(v)}=-v_i\hat n_{i\sigma}.
$$

In the implemented HH PTW path (`HubbardHolsteinPhysicalTermwiseAnsatz`), generator selection is done on physical sectors (`H_t`, `H_U`, `H_{\mathrm{ph}}`, `H_g`, optional `H_{\mathrm{drive}}`), then each sector is instantiated through the same `hubbard_latex_python_pairs.py` builders that immediately materialize mapped Pauli polynomials via `hubbard_*_term`, `jw_number_operator`, `boson_*` helpers. That means selection is **before Pauli-term splitting**, but each selected generator is already in mapped Pauli form by construction.

This ansatz is sector-preserving in fermion space because each physical generator preserves fermion number before Pauli splitting.

## 9.6 Reference-state and ansatz dispatch in the hardcoded pipeline

For `problem="hubbard"`, the pipeline uses the Hartree-Fock basis state
$$
|\psi_{\mathrm{ref}}\rangle=|\Phi_{\mathrm{HF}}\rangle.
$$

For `problem="hh"`, it uses
$$
|\psi_{\mathrm{ref}}\rangle=|\mathrm{vac}_{\mathrm{ph}}\rangle\otimes|\Phi_{\mathrm{HF}}\rangle
$$
via `hubbard_holstein_reference_state(...)`.

The live dispatch is

- `uccsd` -> `HardcodedUCCSDLayerwiseAnsatz`,
- `hva` -> `HubbardLayerwiseAnsatz`,
- `hh_hva` -> `HubbardHolsteinLayerwiseAnsatz`,
- `hh_hva_tw` -> `HubbardHolsteinTermwiseAnsatz`,
- `hh_hva_ptw` -> `HubbardHolsteinPhysicalTermwiseAnsatz`.

Implemented surfaces:

- `src/quantum/vqe_latex_python_pairs.py`
- `pipelines/hardcoded/hubbard_pipeline.py`

# 10. PAOP Families, ADAPT Selection, and Staged Continuation

## 10.1 Primitive PAOP ingredients

The PAOP layer is implemented in `src/quantum/operator_pools/polaron_paop.py`.

### 10.1.1 Shifted density

If the total electron count is `N_e`, then the mean density used by the pool builder is
$$
\bar n = \frac{N_e}{L}
$$
when `N_e>0`, and the shifted density is
$$
\tilde n_i=\hat n_i-\bar n I.
$$

Insert the explicit site density:
$$
\tilde n_i
=
\left(I-\frac{1}{2}(Z_{p_{i\uparrow}}+Z_{p_{i\downarrow}})\right)-\bar n I
=(1-\bar n)I-\frac{1}{2}(Z_{p_{i\uparrow}}+Z_{p_{i\downarrow}}).
$$

At half filling $N_e=L$, $\bar n=1$, so
$$
\tilde n_i=-\frac{1}{2}(Z_{p_{i\uparrow}}+Z_{p_{i\downarrow}}).
$$
This is therefore the default algebra used in HH runs when `num_particles` is set to half filling.

### 10.1.2 Phonon primitives \(P_i\) and \(x_i\)

The momentum-like primitive is
$$
P_i=i(\hat b_i^{\dagger}-\hat b_i).
$$

The displacement primitive is
$$
x_i=\hat b_i+\hat b_i^\dagger.
$$

In code, these are built from phonon ladder operators on the site’s phonon register inside
`src/quantum/operator_pools/polaron_paop.py`:

- `p_i(site) = i*(b_i^† - b_i)` gives \(P_i\),
- `x_i(site)` multiplies `boson_displacement_operator(..., which="x")` so \(x_i=b_i+b_i^\dagger\).

### 10.1.3 Local squeeze primitive

The newly added squeeze-like phonon primitive used by the probe families is
$$
S_i=i\left((\hat b_i^{\dagger})^2-\hat b_i^2\right).
$$

In code this is built as
$$
S_i=i\,\hat b_i^{\dagger}\hat b_i^{\dagger}-i\,\hat b_i\hat b_i,
$$
with intermediate polynomial products cleaned by the same PAOP pruning / real-coefficient rules used for the other generators.

### 10.1.4 Doublon primitive

The local doublon primitive is the explicit operator from Section 3.3:
$$
\hat d_i=\frac{1}{4}\Bigl(I-Z_{p_{i\uparrow}}-Z_{p_{i\downarrow}}+Z_{p_{i\uparrow}}Z_{p_{i\downarrow}}\Bigr).
$$

### 10.1.5 Even hopping channel

The even hopping channel is
$$
K_{ij}=\sum_{\sigma}\left(
\hat c_{i\sigma}^{\dagger}\hat c_{j\sigma}+
\hat c_{j\sigma}^{\dagger}\hat c_{i\sigma}
\right).
$$

### 10.1.6 Odd current channel

The odd current channel is
$$
J_{ij}=i\sum_{\sigma}\left(
\hat c_{i\sigma}^{\dagger}\hat c_{j\sigma}-
\hat c_{j\sigma}^{\dagger}\hat c_{i\sigma}
\right).
$$

For one spin channel with mode indices `p_< < p_>`, the implemented JW current primitive is
$$
J_{p_<p_>}
=
\frac{1}{2}
\Bigl(
X_{p_>}Z_{p_>-1}\cdots Z_{p_<+1}Y_{p_<}
-
Y_{p_>}Z_{p_>-1}\cdots Z_{p_<+1}X_{p_<}
\Bigr).
$$

The code sums that object over both spins.

## 10.2 Implemented PAOP channels

The implemented channel families are not abstract labels only; they are explicit algebraic generators.

### 10.2.1 Conditional displacement

$$
\mathcal O_{\mathrm{disp},i}=\tilde n_i P_i.
$$

### 10.2.2 Legacy doublon dressing

$$
\mathcal O_{\mathrm{dbl},i}=\tilde n_i \hat d_i.
$$

### 10.2.3 Dressed hopping drag

$$
\mathcal O_{\mathrm{hopdrag},ij}=K_{ij}(P_i-P_j).
$$

### 10.2.4 Odd-channel drag

$$
\mathcal O_{\mathrm{curdrag},ij}=J_{ij}(P_i-P_j).
$$

### 10.2.5 Second-order even channel

$$
\mathcal O_{\mathrm{hop2},ij}=K_{ij}(P_i-P_j)^2.
$$

In the implementation, this family drops terms that are identity on every phonon qubit after the polynomial is formed, so the retained generator keeps explicit phonon support.

### 10.2.6 Third-order odd current channel

The newly added LF third-order odd channel is
$$
\mathcal O_{\mathrm{curdrag3},ij}=J_{ij}(P_i-P_j)^3.
$$

The concrete builder label is
- `paop_curdrag3(i,j)`.

### 10.2.7 Fourth-order even hopping channel

The newly added LF fourth-order even channel is
$$
\mathcal O_{\mathrm{hop4},ij}=K_{ij}(P_i-P_j)^4.
$$

As with `hop2`, the implementation drops terms that are identity on all phonon qubits after the product is expanded and cleaned.

The concrete builder label is
- `paop_hop4(i,j)`.

### 10.2.8 Bond-conditioned displacement and squeeze channels

The newly added bond-conditioned channels are
$$
\mathcal O_{\mathrm{bond\_disp},ij}=K_{ij}(P_i+P_j),
$$
$$
\mathcal O_{\mathrm{hop\_sq},ij}=K_{ij}(S_i+S_j),
$$
$$
\mathcal O_{\mathrm{pair\_sq},ij}=i\left(\hat b_i^{\dagger}\hat b_j^{\dagger}-\hat b_i\hat b_j\right).
$$

The concrete builder labels are
- `paop_bond_disp(i,j)`,
- `paop_hop_sq(i,j)`,
- `paop_pair_sq(i,j)`.

### 10.2.9 Local squeeze channels

The newly added squeeze-family channels are
$$
\mathcal O_{\mathrm{sq},i}=S_i,
\qquad
\mathcal O_{\mathrm{dens\_sq},i}=\tilde n_i S_i.
$$

The concrete builder labels are
- `paop_sq(site=i)`,
- `paop_dens_sq(site=i)`.

### 10.2.10 Extended cloud channels

For cloud radius `R`, the implemented extended cloud channels are
$$
\mathcal O_{\mathrm{cloud\_p},i\to j}=\tilde n_i P_j,
\qquad
\mathcal O_{\mathrm{cloud\_x},i\to j}=\tilde n_i x_j,
\qquad
\mathcal O_{\mathrm{cloud\_sq},i\to j}=\tilde n_i S_j,
$$
with the distance gate `\operatorname{dist}(i,j)\le R`.

The concrete operator labels emitted by the builder are
- `paop_cloud_p(site=i->phonon=j)`,
- `paop_cloud_x(site=i->phonon=j)`,
- `paop_cloud_sq(site=i->phonon=j)`.

For `paop_full`, `paop_lf_full`, and `paop_sq_full`, effective radius is forced to at least 1 (`--paop-r=0` maps to nearest-neighbor cloud support).

### 10.2.11 Doublon-translation and doublon-squeeze channels

The implemented doublon-conditioned phonon channels are
$$
\mathcal O_{\mathrm{dbl\_p},i\to j}=\hat d_i P_j,
\qquad
\mathcal O_{\mathrm{dbl\_x},i\to j}=\hat d_i x_j,
\qquad
\mathcal O_{\mathrm{dbl\_sq},i\to j}=\hat d_i S_j.
$$

The concrete labels are
- `paop_dbl_p(site=i->phonon=j)`,
- `paop_dbl_x(site=i->phonon=j)`,
- `paop_dbl_sq(site=i->phonon=j)`.

## 10.3 Pool-family map

The current PAOP-family map is:

- `paop` = `paop_std`
- `paop_lf` = `paop_lf_std`
- `paop_min` = `disp`
- `paop_std` = `disp + hopdrag`
- `paop_full` = `disp + doublon + hopdrag + cloud_p + cloud_x`
- `paop_lf_std` = `disp + hopdrag + curdrag`
- `paop_lf2_std` = `disp + hopdrag + curdrag + hop2`
- `paop_lf3_std` = `paop_lf2_std + curdrag3`  *(experimental / offline-probe-only)*
- `paop_lf4_std` = `paop_lf3_std + hop4`  *(experimental / offline-probe-only)*
- `paop_lf_full` = `disp + hopdrag + curdrag + hop2 + cloud_p + cloud_x + dbl_p + dbl_x`
- `paop_sq_std` = `disp + hopdrag + curdrag + sq + dens_sq`  *(experimental / offline-probe-only)*
- `paop_sq_full` = `paop_sq_std + cloud_sq + dbl_sq`  *(experimental / offline-probe-only)*
- `paop_bond_disp_std` = `disp + hopdrag + curdrag + bond_disp`  *(experimental / offline-probe-only)*
- `paop_hop_sq_std` = `disp + hopdrag + curdrag + hop_sq`  *(experimental / offline-probe-only)*
- `paop_pair_sq_std` = `disp + hopdrag + curdrag + pair_sq`  *(experimental / offline-probe-only)*

When one of these PAOP families is selected on the HH ADAPT path and `g_{ep}\neq 0`, the live pool builder returns the deduplicated merge
$$
\mathcal P_{\mathrm{HH,selected}}
=
\mathcal P_{\mathrm{hva}}
\cup
\mathcal P_{\mathrm{hh\_termwise\_augmented}}
\cup
\mathcal P_{\mathrm{selected\_paop\_family}}.
$$
If `g_{ep}=0`, the implementation returns the selected PAOP family without the HH augmentation merge.

The built-in broad family presets are
$$
\mathcal P_{\mathrm{uccsd\_paop\_lf\_full}}
=
\mathcal P_{\mathrm{uccsd\_lifted}}
\cup
\mathcal P_{\mathrm{paop\_lf\_full}},
$$
$$
\mathcal P_{\mathrm{full\_meta}}
=
\mathcal P_{\mathrm{uccsd\_lifted}}
\cup
\mathcal P_{\mathrm{hva}}
\cup
\mathcal P_{\mathrm{hh\_termwise\_augmented}}
\cup
\mathcal P_{\mathrm{paop\_full}}
\cup
\mathcal P_{\mathrm{paop\_lf\_full}},
$$
with the `hh_termwise_augmented` block active when `g_{ep}\neq 0`.

### 10.3.1 Standalone VLF/SQ macro families

Separate from `polaron_paop.py`, the repo now exposes grouped macro-generator families in `src/quantum/operator_pools/vlf_sq.py`.

Define the shell-resolved VLF macro generator
$$
G_r^{\mathrm{VLF}}=\sum_{\operatorname{dist}(i,j)=r}\tilde n_i P_j,
$$
and the global squeeze generators
$$
G^{\mathrm{SQ}}=\sum_i S_i=\frac12\sum_i\bigl(x_iP_i+P_i x_i\bigr),
$$
$$
G_{\mathrm{SQ}}^{(n)}=\sum_i \tilde n_i S_i.
$$

The implemented family map is

- `vlf_only` = `{G_r^{\mathrm{VLF}}}` over the allowed shells,
- `sq_only` = `{G^{\mathrm{SQ}}}`,
- `vlf_sq` = `{G_r^{\mathrm{VLF}}} \cup \{G^{\mathrm{SQ}}\}`,
- `sq_dens_only` = `{G_{\mathrm{SQ}}^{(n)}}`,
- `vlf_sq_dens` = `{G_r^{\mathrm{VLF}}} \cup \{G^{\mathrm{SQ}}, G_{\mathrm{SQ}}^{(n)}\}`.

These are emitted as grouped macro generators such as
- `vlf_only:vlf_shell(r=0)`,
- `vlf_sq:sq_global`,
- `vlf_sq_dens:dens_sq_global`.

Unlike the PAOP families above, the VLF/SQ families are returned directly as macro families and the implementation rejects `--paop-split-paulis` for them.

The implementation then optionally applies

- pruning,
- normalization (`none`, `fro`, `maxcoeff`),
- signature deduplication.

## 10.4 Legacy fallback selector

The ADAPT signal remains
$$
g_m^{(n)}=i\langle\psi^{(n)}|[\hat H,A_m]|\psi^{(n)}\rangle,
$$
and on the compiled production path this is evaluated as
$$
g_m^{(n)}=2\,\Im\langle H\psi^{(n)}\mid A_m\psi^{(n)}\rangle,
$$
with
$$
|\psi^{(n)}\rangle=
\exp(-i\theta_n A_{m_n})\cdots \exp(-i\theta_1 A_{m_1})|\psi_{\mathrm{ref}}\rangle.
$$

If staged continuation scoring is disabled, the selector reduces to the legacy rule
$$
m_{\star}=\arg\max_{m\in\mathcal P_{\mathrm{avail}}}|g_m^{(n)}|.
$$

If repeated operator reuse is allowed, the live code applies the repeat-biased surrogate
$$
\widetilde s_m^{\mathrm{repeat}}
=\frac{|g_m^{(n)}|}{1+\beta\,c_m},
\qquad
\beta=1.5,
$$
where `c_m` is the prior selection count of pool element `m`. The chosen legacy operator is then
$$
m_{\star}=\arg\max_{m\in\mathcal P_{\mathrm{avail}}}\widetilde s_m^{\mathrm{repeat}}.
$$

## 10.5 `phase1_v1` as a self-contained selector

### 10.5.1 Core signal, probe set, insertion position, and refit window

At ADAPT depth `n`, the ordered scaffold state is
$$
|\psi^{(n)}\rangle=
\exp(-i\theta_n A_{m_n})\cdots \exp(-i\theta_1 A_{m_1})|\psi_{\mathrm{ref}}\rangle.
$$

For a candidate generator `A_m`, the compiled production gradient is
$$
g_m^{(n)}=
i\langle\psi^{(n)}|[H,A_m]|\psi^{(n)}\rangle
=
2\,\Im\langle H\psi^{(n)}\mid A_m\psi^{(n)}\rangle.
$$

The `phase1_v1` selector first ranks the available pool by descending gradient magnitude and keeps the cheap candidate universe
$$
\mathcal C_{64}
=
\text{top-}\min\{64,|\mathcal P_{\mathrm{avail}}|\}
\text{ candidates ranked by }|g_m^{(n)}|.
$$

The default append position is
$$
p_{\mathrm{app}}=n.
$$

But the live `phase1_v1` controller does not force admission at append. When probing is enabled, it evaluates the ordered probe list obtained by concatenating

1. the append slot `p_{\mathrm{app}}`,
2. the left boundary `0`,
3. the current active-window indices `W_{\mathrm{probe}}`,

then deduplicating in first-occurrence order and truncating after
$$
M_{\mathrm{probe}}=\texttt{phase1\_probe\_max\_positions}
$$
distinct entries. If probing is disabled, the evaluated position list collapses to `[p_{\mathrm{app}}]`.

For a tentative insertion position `p`, define the hypothetical admitted parameter vector
$$
\theta^{(p)}=\operatorname{insert}(\theta,p,0).
$$
The active reoptimization window is then built from `\theta^{(p)}` by

1. keeping the newest `w_{\mathrm{eff}}^{(p)}=\min\{w,n+1\}` indices,
2. then optionally adding the top-`k` older indices ranked by descending `|\theta_j^{(p)}|`, tie-breaking by ascending index.

So the active refit set is
$$
W_{\mathrm{refit}}(p)=W_{\mathrm{newest}}^{(p)}\cup W_{\mathrm{top}|\theta^{(p)}|}^{(p)}.
$$

In particular, if the reoptimization policy is `full`, or `windowed` with a maximally large window and enough carry slots so that
$$
W_{\mathrm{probe}}=\{0,1,\dots,n-1\},
$$
and if `M_{\mathrm{probe}}\ge n+1`, then the probed insertion positions cover the whole scaffold,
$$
\{0,1,\dots,n\}.
$$
so a dressed PAOP / ADAPT generator is not restricted to append-only placement.

### 10.5.2 Gates, burdens, and feature ingredients

For each candidate-position pair `(m,p)` in the cheap scan, the feature build records

- signed and absolute gradients `g_m`, `|g_m|`,
- lower-confidence gradient `g_{\mathrm{lcb}}`,
- raw tangent surrogate `F_{\mathrm{raw}}`,
- compile-cost proxy,
- grouped-measurement bookkeeping,
- candidate family / stage metadata,
- symmetry metadata,
- predicted refit window.

The stage gate is
$$
\Gamma_{\mathrm{stage}}(m)=
\begin{cases}
1, & \text{stage = residual},\\
1, & \text{stage = core/seed and } m \notin \mathcal P_{\mathrm{residual}},\\
0, & \text{stage = core/seed and } m \in \mathcal P_{\mathrm{residual}}.
\end{cases}
$$

The hard symmetry gate is
$$
\Gamma_{\mathrm{sym}}(m)=
\begin{cases}
0, & \text{candidate symmetry spec has } \texttt{hard\_guard=true},\\
1, & \text{otherwise}.
\end{cases}
$$

If either gate is closed, the record is not admissible.

The compile-cost proxy used inside the score is
$$
D_{\mathrm{proxy}}
=
n_{\mathrm{new\,pauli}}
+n_{\mathrm{rot}}
+|p_{\mathrm{app}}-p|
+|W_{\mathrm{refit}}(p)|.
$$

The grouped-measurement bookkeeping tracks
$$
G_{\mathrm{new}},\qquad S_{\mathrm{new}},\qquad R_{\mathrm{reuse}},
$$
for new groups, new nominal-shot burden, and grouped-reuse miss count.

The lower-confidence gradient is
$$
g_{\mathrm{lcb}}=\max\{|g|-z_{\alpha}\sigma,0\}.
$$

In the current live `phase1_v1` implementation, the feature builder takes
$$
\sigma=0,
\qquad
F_{\mathrm{raw}}=|g_m^{(n)}|,
$$
so the active cheap-path lower-confidence gradient is simply
$$
g_{\mathrm{lcb}}=|g_m^{(n)}|.
$$

### 10.5.3 Cheap score

The cheap score used by `phase1_v1` is
$$
S_{\mathrm{simple}}
=
\mathbf 1(m\in\Omega_{\mathrm{stage}})
\frac{g_{\mathrm{lcb}}^2}{2\lambda_F F_{\mathrm{raw}}}
\frac{1}{K_{\mathrm{screen}}},
$$
with
$$
K_{\mathrm{screen}}
=
1+w_D\bar D_{\mathrm{life}}+w_G\bar G_{\mathrm{new}}+w_C\bar C_{\mathrm{new}}+w_c\bar c.
$$

Here

- `D_life` is the compiled burden proxy, optionally lifetime-weighted,
- `G_new` is new grouped-measurement burden,
- `C_new` is new nominal shot burden,
- `c` is the same-family streak burden.

The live defaults are

- `lambda_F = 1.0`,
- `lambda_compile = 0.05`,
- `lambda_measure = 0.02`,
- `lambda_leak = 0.0`,
- `z_alpha = 0.0`.

So the live `phase1_v1` cheap screen uses

- `g_{\mathrm{lcb}}=|g|`,
- no active leakage penalty,
- no active additive motif bonus term.

### 10.5.4 Position probing, trough detection, and effective selector

`phase1_v1` does not force append-only behavior. Outside the residual stage, it probes alternate insertion positions when one of the implemented triggers fires:

- drop plateau,
- `eps_grad` plus flat finite-angle behavior,
- repeated-family flatness.

If probing is enabled, the allowed candidate positions are
$$
\mathcal P_{\mathrm{probe}}
=
\{p_{\mathrm{app}},0\}\cup W_{\mathrm{refit}}(p_{\mathrm{app}}),
$$
truncated to the configured maximum number of probe positions.

The score `S_{\mathrm{simple}}` is then evaluated over
$$
m\in\mathcal C_{64},
\qquad
p\in\mathcal P_{\mathrm{probe}}.
$$

The trough detector compares the best append score against the best non-append score. A trough is declared when
$$
g_{\mathrm{lcb}}^{\mathrm{nonappend}}>0
$$
and either
$$
S_{\mathrm{nonappend}}\ge \mu\,S_{\mathrm{append}},
$$
or
$$
S_{\mathrm{append}}<\tau_{\mathrm{append}}
\quad\text{and}\quad
S_{\mathrm{nonappend}}\ge \tau_{\mathrm{append}},
$$
where `\mu` is the configured probe-margin ratio and `\tau_{\mathrm{append}}` is the append-admit threshold.

So the effective `phase1_v1` selector is:

- build cheap candidate-position records,
- evaluate `S_{\mathrm{simple}}`,
- admit the best surviving record.

## 10.6 `phase2_v1` as a self-contained selector

### 10.6.1 Core signal, probe set, insertion position, refit window, and cheap stage

At ADAPT depth `n`, the ordered scaffold state is
$$
|\psi^{(n)}\rangle=
\exp(-i\theta_n A_{m_n})\cdots \exp(-i\theta_1 A_{m_1})|\psi_{\mathrm{ref}}\rangle.
$$

For a candidate generator `A_m`, the compiled production gradient is
$$
g_m^{(n)}=
i\langle\psi^{(n)}|[H,A_m]|\psi^{(n)}\rangle
=
2\,\Im\langle H\psi^{(n)}\mid A_m\psi^{(n)}\rangle.
$$

The cheap candidate universe is
$$
\mathcal C_{64}
=
\text{top-}\min\{64,|\mathcal P_{\mathrm{avail}}|\}
\text{ candidates ranked by }|g_m^{(n)}|.
$$

The default append position is
$$
p_{\mathrm{app}}=n.
$$

When probing is enabled, `phase2_v1` uses the same ordered probe list as `phase1_v1`: start from the append slot `p_{\mathrm{app}}`, prepend the left boundary `0`, append the current active-window indices `W_{\mathrm{probe}}`, then deduplicate in first-occurrence order and truncate after
$$
M_{\mathrm{probe}}=\texttt{phase1\_probe\_max\_positions}
$$
distinct entries. If probing is disabled, only `p_{\mathrm{app}}` is evaluated.

For a tentative insertion position `p`, define
$$
\theta^{(p)}=\operatorname{insert}(\theta,p,0).
$$
Then the active reoptimization window is
$$
W_{\mathrm{refit}}(p)=W_{\mathrm{newest}}^{(p)}\cup W_{\mathrm{top}|\theta^{(p)}|}^{(p)},
$$
where `W_{\mathrm{newest}}^{(p)}` keeps the newest `w_{\mathrm{eff}}^{(p)}=\min\{w,n+1\}` indices and `W_{\mathrm{top}|\theta^{(p)}|}^{(p)}` optionally adds the top-`k` older indices ranked by descending `|\theta_j^{(p)}|`.

Hence, with a full or maximally large window and sufficiently large `phase1_probe_max_positions`, the cheap `phase2_v1` scan can also evaluate every insertion position `0,\dots,n` rather than append only.

The stage gate is
$$
\Gamma_{\mathrm{stage}}(m)=
\begin{cases}
1, & \text{stage = residual},\\
1, & \text{stage = core/seed and } m \notin \mathcal P_{\mathrm{residual}},\\
0, & \text{stage = core/seed and } m \in \mathcal P_{\mathrm{residual}}.
\end{cases}
$$

The hard symmetry gate is
$$
\Gamma_{\mathrm{sym}}(m)=
\begin{cases}
0, & \text{candidate symmetry spec has } \texttt{hard\_guard=true},\\
1, & \text{otherwise}.
\end{cases}
$$

The compile-cost proxy is
$$
D_{\mathrm{proxy}}
=
n_{\mathrm{new\,pauli}}
+n_{\mathrm{rot}}
+|p_{\mathrm{app}}-p|
+|W_{\mathrm{refit}}(p)|.
$$

The grouped-measurement bookkeeping tracks
$$
G_{\mathrm{new}},\qquad S_{\mathrm{new}},\qquad R_{\mathrm{reuse}}.
$$

The lower-confidence gradient is
$$
g_{\mathrm{lcb}}=\max\{|g|-z_{\alpha}\sigma,0\},
$$
and the cheap-stage burden is
$$
K_{\mathrm{screen}}
=
1+w_D\bar D_{\mathrm{life}}+w_G\bar G_{\mathrm{new}}+w_C\bar C_{\mathrm{new}}+w_c\bar c.
$$

The cheap score is
$$
S_{\mathrm{simple}}
=
\mathbf 1(m\in\Omega_{\mathrm{stage}})
\frac{g_{\mathrm{lcb}}^2}{2\lambda_F F_{\mathrm{raw}}}
\frac{1}{K_{\mathrm{screen}}}.
$$

In the current live cheap path for `phase2_v1`, the feature builder takes
$$
\sigma=0,
\qquad
F_{\mathrm{raw}}=|g_m^{(n)}|,
$$
so the active cheap-path lower-confidence gradient is
$$
g_{\mathrm{lcb}}=|g_m^{(n)}|.
$$

### 10.6.2 Shortlist rule

After the cheap pass, `phase2_v1` keeps the shortlist
$$
N_{\mathrm{short}}
=
\min\left\{
N,\,
N_{\max},\,
\left\lceil f_{\mathrm{short}}N\right\rceil
\right\},
$$
where

- `N` is the number of cheap records,
- `N_{\max}=\texttt{shortlist\_size}`,
- `f_{\mathrm{short}}=\texttt{shortlist\_fraction}`.

The retained shortlist is ranked by

1. descending cheap score,
2. descending `simple_score`,
3. ascending pool index,
4. ascending insertion position.

### 10.6.3 Reduced-path geometry, novelty, curvature, and trust-region drop

For each shortlisted record, `phase2_v1` rebuilds an exact local derivative and geometry model at the proposed insertion position:

- recompute the gradient on the current state,
- build the exact inherited-window parameter tangents of the ordered scaffold,
- build the exact inherited-window Hessian block,
- build the candidate self-curvature and mixed candidate-window Hessian vector.

Let `W_m` denote the inherited window that would actually be refit if candidate `m` were admitted. Let `t_m` be the horizontal append tangent and `\{t_j\}_{j\in W_m}` be the exact horizontal tangents of the current scaffold parameters in that window. Then
$$
F_{\mathrm{raw}}=\langle t_m,t_m\rangle,
\qquad
(Q_m)_{jk}=\operatorname{Re}\langle t_j,t_k\rangle,
\qquad
(q_m)_j=\operatorname{Re}\langle t_j,t_m\rangle.
$$

The exact local quadratic model over append parameter `\alpha` and inherited-window displacement `\delta\theta_{W_m}` is
$$
E(\alpha,\delta\theta_{W_m})
\approx
E_0
+g_m\alpha
+\frac12 h_m\alpha^2
+\alpha\,b_m^\top\delta\theta_{W_m}
+\frac12\delta\theta_{W_m}^\top H_{W_m}\delta\theta_{W_m}.
$$

The inherited-window Hessian block is symmetrized and regularized as
$$
M_m
=
\frac12(H_{W_m}+H_{W_m}^{\top})+\lambda_H I,
$$
with ridge growth until the linear solve is numerically stable.

The reduced curvature is
$$
\widetilde h_m=h_m-b_m^\top M_m^{-1}b_m.
$$

The reduced-path metric is
$$
F_m^{\mathrm{red}}
=
F_{\mathrm{raw}}
-2q_m^\top M_m^{-1}b_m
+b_m^\top M_m^{-1}Q_mM_m^{-1}b_m.
$$

The reduced-path overlap is
$$
q_m^{\mathrm{red}}=q_m-Q_mM_m^{-1}b_m.
$$

The implemented novelty is the Tikhonov-regularized reduced-path novelty
$$
\nu_m
=
\operatorname{clip}_{[0,1]}
\left(
1-
\frac{(q_m^{\mathrm{red}})^\top(Q_m+\varepsilon_{\mathrm{nov}}I)^{-1}q_m^{\mathrm{red}}}{F_m^{\mathrm{red}}}
\right).
$$

The trust-region drop is
$$
\Delta E_{\mathrm{TR}}
=
\max_{|\alpha|\le \rho/\sqrt{F_m^{\mathrm{red}}}}
\left[
g_{\mathrm{lcb}}|\alpha|-\frac12\max(\widetilde h_m,0)\alpha^2
\right].
$$

Equivalently,
$$
\Delta E_{\mathrm{TR}}
=
\begin{cases}
\dfrac{g_{\mathrm{lcb}}^2}{2\max(\widetilde h_m,0)},
& \max(\widetilde h_m,0)>0\ \text{and}\ \dfrac{g_{\mathrm{lcb}}}{\max(\widetilde h_m,0)}\le \dfrac{\rho}{\sqrt{F_m^{\mathrm{red}}}},\\[1.2ex]
\dfrac{\rho g_{\mathrm{lcb}}}{\sqrt{F_m^{\mathrm{red}}}}
-\dfrac{\rho^2\max(\widetilde h_m,0)}{2F_m^{\mathrm{red}}},
& \text{otherwise.}
\end{cases}
$$

### 10.6.4 Full rerank score and effective selector

The full rerank burden is
$$
K
=
1+w_D\bar D_{\mathrm{life}}+w_G\bar G_{\mathrm{new}}+w_C\bar C_{\mathrm{new}}+w_c\bar c.
$$

The full rerank score is
$$
S_{\mathrm{full}}
=
\mathbf 1(m\in\Omega_{\mathrm{stage}})
\,\nu_m^{\gamma_N}
\,\frac{\Delta E_{\mathrm{TR}}}{K}.
$$

The live defaults are

- `lambda_F = 1.0`,
- `lambda_H = 10^{-6}`,
- `rho = 0.25`,
- `gamma_N = 1.0`,
- `wD = 0.2`,
- `wG = 0.15`,
- `wC = 0.15`,
- `wc = 0.1`,
- `shortlist_fraction = 0.2`,
- `shortlist_size = 12`,
- `batch_target_size = 2`,
- `batch_size_cap = 3`,
- `batch_near_degenerate_ratio = 0.9`.

So the effective `phase2_v1` selector is:

- cheap-rank candidate-position pairs by `S_{\mathrm{simple}}`,
- keep the configured shortlist,
- rerank that shortlist by `S_{\mathrm{full}}`,
- admit the top full-score record.

The code still exposes batch-related configuration knobs and a compatibility-penalty helper, but the current live branch admits a singleton batch only:

- `phase2_selected_records` is set to the top full-score record,
- `greedy_batch_select(...)` is presently a singleton pass-through helper,
- `phase2_last_batch_selected` remains false in the active branch.

## 10.7 `phase3_v1` as a self-contained selector

### 10.7.1 Core signal, probe set, insertion position, refit window, and cheap stage

At ADAPT depth `n`, the ordered scaffold state is
$$
|\psi^{(n)}\rangle=
\exp(-i\theta_n A_{m_n})\cdots \exp(-i\theta_1 A_{m_1})|\psi_{\mathrm{ref}}\rangle.
$$

For a candidate generator `A_m`, the compiled production gradient is
$$
g_m^{(n)}=
i\langle\psi^{(n)}|[H,A_m]|\psi^{(n)}\rangle
=
2\,\Im\langle H\psi^{(n)}\mid A_m\psi^{(n)}\rangle.
$$

The cheap candidate universe is
$$
\mathcal C_{64}
=
\text{top-}\min\{64,|\mathcal P_{\mathrm{avail}}|\}
\text{ candidates ranked by }|g_m^{(n)}|.
$$

The default append position is
$$
p_{\mathrm{app}}=n.
$$

When probing is enabled, `phase3_v1` uses the same ordered probe list as `phase1_v1`: start from the append slot `p_{\mathrm{app}}`, prepend the left boundary `0`, append the current active-window indices `W_{\mathrm{probe}}`, then deduplicate in first-occurrence order and truncate after
$$
M_{\mathrm{probe}}=\texttt{phase1\_probe\_max\_positions}
$$
distinct entries. If probing is disabled, only `p_{\mathrm{app}}` is evaluated.

For a tentative insertion position `p`, define
$$
\theta^{(p)}=\operatorname{insert}(\theta,p,0).
$$
Then the active reoptimization window is
$$
W_{\mathrm{refit}}(p)=W_{\mathrm{newest}}^{(p)}\cup W_{\mathrm{top}|\theta^{(p)}|}^{(p)},
$$
where `W_{\mathrm{newest}}^{(p)}` keeps the newest `w_{\mathrm{eff}}^{(p)}=\min\{w,n+1\}` indices and `W_{\mathrm{top}|\theta^{(p)}|}^{(p)}` optionally adds the top-`k` older indices ranked by descending `|\theta_j^{(p)}|`.

Hence, with a full or maximally large window and sufficiently large `phase1_probe_max_positions`, the cheap `phase3_v1` scan can also evaluate every insertion position `0,\dots,n` rather than append only.

The stage gate is
$$
\Gamma_{\mathrm{stage}}(m)=
\begin{cases}
1, & \text{stage = residual},\\
1, & \text{stage = core/seed and } m \notin \mathcal P_{\mathrm{residual}},\\
0, & \text{stage = core/seed and } m \in \mathcal P_{\mathrm{residual}}.
\end{cases}
$$

The hard symmetry gate is
$$
\Gamma_{\mathrm{sym}}(m)=
\begin{cases}
0, & \text{candidate symmetry spec has } \texttt{hard\_guard=true},\\
1, & \text{otherwise}.
\end{cases}
$$

The compile-cost proxy is
$$
D_{\mathrm{proxy}}
=
n_{\mathrm{new\,pauli}}
+n_{\mathrm{rot}}
+|p_{\mathrm{app}}-p|
+|W_{\mathrm{refit}}(p)|.
$$

The grouped-measurement bookkeeping tracks
$$
G_{\mathrm{new}},\qquad S_{\mathrm{new}},\qquad R_{\mathrm{reuse}}.
$$

The lower-confidence gradient is
$$
g_{\mathrm{lcb}}=\max\{|g|-z_{\alpha}\sigma,0\},
$$
and the cheap-stage burden is
$$
K_{\mathrm{screen}}
=
1+w_D\bar D_{\mathrm{life}}+w_G\bar G_{\mathrm{new}}+w_C\bar C_{\mathrm{new}}+w_c\bar c.
$$

The cheap score is
$$
S_{\mathrm{simple}}
=
\mathbf 1(m\in\Omega_{\mathrm{stage}})
\frac{g_{\mathrm{lcb}}^2}{2\lambda_F F_{\mathrm{raw}}}
\frac{1}{K_{\mathrm{screen}}}.
$$

In the current live cheap path for `phase3_v1`, the feature builder takes
$$
\sigma=0,
\qquad
F_{\mathrm{raw}}=|g_m^{(n)}|,
$$
so the active cheap-path lower-confidence gradient is
$$
g_{\mathrm{lcb}}=|g_m^{(n)}|.
$$

### 10.7.2 Shortlist, reduced-path geometry, novelty, and full rerank score

After the cheap pass, `phase3_v1` keeps the shortlist
$$
N_{\mathrm{short}}
=
\min\left\{
N,\,
N_{\max},\,
\left\lceil f_{\mathrm{short}}N\right\rceil
\right\},
$$
where

- `N` is the number of cheap records,
- `N_{\max}=\texttt{shortlist\_size}`,
- `f_{\mathrm{short}}=\texttt{shortlist\_fraction}`.

For each shortlisted record, `phase3_v1` rebuilds an exact local derivative and geometry model at the proposed insertion position.

Let `W_m` denote the inherited window that would actually be refit if candidate `m` were admitted. Let `t_m` be the horizontal append tangent and `\{t_j\}_{j\in W_m}` the exact horizontal tangents of the current scaffold parameters in that window. Then
$$
F_{\mathrm{raw}}=\langle t_m,t_m\rangle,
\qquad
(Q_m)_{jk}=\operatorname{Re}\langle t_j,t_k\rangle,
\qquad
(q_m)_j=\operatorname{Re}\langle t_j,t_m\rangle.
$$

The local quadratic model is
$$
E(\alpha,\delta\theta_{W_m})
\approx
E_0
+g_m\alpha
+\frac12 h_m\alpha^2
+\alpha\,b_m^\top\delta\theta_{W_m}
+\frac12\delta\theta_{W_m}^\top H_{W_m}\delta\theta_{W_m}.
$$

The positive-definite inherited-window model is
$$
M_m
=
\frac12(H_{W_m}+H_{W_m}^{\top})+\lambda_H I,
$$
with ridge growth until the solve is stable.

The reduced curvature is
$$
\widetilde h_m=h_m-b_m^\top M_m^{-1}b_m.
$$

The reduced-path metric is
$$
F_m^{\mathrm{red}}
=
F_{\mathrm{raw}}
-2q_m^\top M_m^{-1}b_m
+b_m^\top M_m^{-1}Q_mM_m^{-1}b_m.
$$

The reduced-path overlap is
$$
q_m^{\mathrm{red}}=q_m-Q_mM_m^{-1}b_m.
$$

The implemented novelty is
$$
\nu_m
=
\operatorname{clip}_{[0,1]}
\left(
1-
\frac{(q_m^{\mathrm{red}})^\top(Q_m+\varepsilon_{\mathrm{nov}}I)^{-1}q_m^{\mathrm{red}}}{F_m^{\mathrm{red}}}
\right).
$$

The trust-region drop is
$$
\Delta E_{\mathrm{TR}}
=
\max_{|\alpha|\le \rho/\sqrt{F_m^{\mathrm{red}}}}
\left[
g_{\mathrm{lcb}}|\alpha|-\frac12\max(\widetilde h_m,0)\alpha^2
\right].
$$

Equivalently,
$$
\Delta E_{\mathrm{TR}}
=
\begin{cases}
\dfrac{g_{\mathrm{lcb}}^2}{2\max(\widetilde h_m,0)},
& \max(\widetilde h_m,0)>0\ \text{and}\ \dfrac{g_{\mathrm{lcb}}}{\max(\widetilde h_m,0)}\le \dfrac{\rho}{\sqrt{F_m^{\mathrm{red}}}},\\[1.2ex]
\dfrac{\rho g_{\mathrm{lcb}}}{\sqrt{F_m^{\mathrm{red}}}}
-\dfrac{\rho^2\max(\widetilde h_m,0)}{2F_m^{\mathrm{red}}},
& \text{otherwise.}
\end{cases}
$$

The full rerank burden is
$$
K
=
1+w_D\bar D_{\mathrm{life}}+w_G\bar G_{\mathrm{new}}+w_C\bar C_{\mathrm{new}}+w_c\bar c.
$$

The full rerank score is
$$
S_{\mathrm{full}}
=
\mathbf 1(m\in\Omega_{\mathrm{stage}})
\,\nu_m^{\gamma_N}
\,\frac{\Delta E_{\mathrm{TR}}}{K}.
$$

### 10.7.3 Additional runtime surfaces and effective selector

The live `phase3_v1` surface also carries the following runtime surfaces:

1. **Motif seeding**. If `phase3_motif_source_json` is supplied, the code loads a motif library and may insert transferable tiled generators before the main ADAPT loop.

2. **Runtime split on shortlisted macro generators**. If
$$
\texttt{phase3\_runtime\_split\_mode}=\texttt{shortlist\_pauli\_children\_v1},
$$
and a shortlisted candidate is a macro generator, the code builds child generators
$$
\mathcal C(m)=\{m_1,\ldots,m_K\},
$$
with one split child per serialized Pauli term.

Each child is audited against the same required particle-number / spin-sector symmetry intent. So at the child-atom level the implementation computes a runtime symmetry gate
$$
\Gamma_{\mathrm{sym}}(m_r)\in\{0,1\},
$$
and updates the child metadata accordingly:

- if `\Gamma_{\mathrm{sym}}(m_r)=0`, the child metadata is marked symmetry-violating and `\texttt{hard\_guard}=\texttt{true}`;
- if `\Gamma_{\mathrm{sym}}(m_r)=1`, the child remains symmetry-admissible.

The code then builds candidate child sets
$$
\mathcal A(m)=\left\{S\subseteq \mathcal C(m):\Gamma_{\mathrm{sym}}(S)=1\right\},
$$
where the gate is re-evaluated on the combined child-set polynomial itself. Only members of `\mathcal A(m)` survive as admissible runtime-split replacements. Unsafe child atoms may still appear in telemetry/provenance, but they do not survive admission as symmetry-safe replacements.

The full rerank logic therefore compares the parent against symmetry-safe split representatives — singleton child atoms when safe, and multi-child `child_set[...]` combinations when the combined set preserves the required symmetry.

3. **Lifetime-weighted burden**. If
$$
\texttt{phase3\_lifetime\_cost\_mode}=\texttt{phase3\_v1},
$$
the depth burden is weighted by
$$
R_{\mathrm{rem}}=\max\{1,\texttt{max\_depth}-\texttt{current\_depth}+1\}.
$$

4. **Symmetry metadata / verification hooks**. The exposed mode surface is
$$
\texttt{phase3\_symmetry\_mitigation\_mode}\in
\{\texttt{off},\texttt{verify\_only},\texttt{postselect\_diag\_v1},\texttt{projector\_renorm\_v1}\}.
$$

5. **Rescue diagnostics**. If `phase3_enable_rescue=true`, the code may record rescue attempts and outcomes in `rescue_history`.

The live defaults are

- `lambda_F = 1.0`,
- `lambda_H = 10^{-6}`,
- `rho = 0.25`,
- `gamma_N = 1.0`,
- `wD = 0.2`,
- `wG = 0.15`,
- `wC = 0.15`,
- `wc = 0.1`,
- `shortlist_fraction = 0.2`,
- `shortlist_size = 12`,
- singleton batch admission in the active branch.

So the effective `phase3_v1` selector is:

- cheap-rank candidate-position pairs by `S_{\mathrm{simple}}`,
- keep the configured shortlist,
- rerank the shortlist by `S_{\mathrm{full}}`,
- optionally runtime-split shortlisted macros,
- admit the top surviving full-score record,
- carry motif/runtime-split/rescue/symmetry/lifetime metadata as part of the live phase-3 surface.

## 10.8 Current HH pool and replay-family surfaces

The active HH ADAPT surface in `pipelines/hardcoded/adapt_pipeline.py` supports

- `hva`,
- `full_meta`,
- `uccsd_paop_lf_full`,
- `uccsd_otimes_paop_lf_std`, `uccsd_otimes_paop_lf2_std`, `uccsd_otimes_paop_bond_disp_std`,
- `uccsd_otimes_paop_lf_std_seq2p`, `uccsd_otimes_paop_lf2_std_seq2p`, `uccsd_otimes_paop_bond_disp_std_seq2p`,
- `paop`, `paop_min`, `paop_std`, `paop_full`,
- `paop_lf`, `paop_lf_std`, `paop_lf2_std`, `paop_lf3_std`, `paop_lf4_std`, `paop_lf_full`,
- `paop_sq_std`, `paop_sq_full`,
- `paop_bond_disp_std`, `paop_hop_sq_std`, `paop_pair_sq_std`,
- `vlf_only`, `sq_only`, `vlf_sq`, `sq_dens_only`, `vlf_sq_dens`,
- `full_hamiltonian`.

The replay-side family resolver in `hh_vqe_from_adapt_family.py` additionally accepts the compatibility aliases

- `pool_a`,
- `pool_b`,

which are replay-only legacy mappings rather than direct `adapt_pipeline.py` pool keys.

For the new HH product-family surfaces, the canonical generator is
$$
G_{a\mu}=F_a M_\mu,
$$
where

- $F_a$ is one lifted fermionic UCCSD excitation factor,
- $M_\mu$ is one boson-only phonon motif constructed directly from the phonon
  primitives used by the PAOP families,
- the allowed v1 motif families are `paop_lf_std`, `paop_lf2_std`, and
  `paop_bond_disp_std`,
- admissible pairs are locality-filtered against the excitation support before
  canonicalization and deduplication.

So the product-family pool is
$$
\mathcal P_{\mathrm{uccsd}\otimes\mathrm{motif}}
=
\left\{
F_a M_\mu \;:\; (a,\mu)\in \mathcal C_{\mathrm{local}}
\right\},
$$
with $\mathcal C_{\mathrm{local}}$ the code-level compatibility relation induced
by excitation support and motif support.

This is distinct from the older additive union
$$
\mathcal P_{\mathrm{uccsd\_paop\_lf\_full}}
=
\mathcal P_{\mathrm{uccsd\_lifted}}\cup \mathcal P_{\mathrm{paop\_lf\_full}},
$$
which remains unchanged.

For the `_seq2p` variants, one logical pair $(F_a,M_\mu)$ is exposed with two
parameters via the ordered factorization
$$
U_{a\mu}(\alpha,\beta)=e^{-i\alpha F_a}e^{-i\beta M_\mu},
$$
rather than the single-parameter product exponential
$$
U_{a\mu}(\theta)=e^{-i\theta(F_a M_\mu)}.
$$
These logical-pair variants are additive opt-in surfaces and do not change the
staged `phase3_v1` default core.

For staged HH continuation modes `phase1_v1`, `phase2_v1`, and `phase3_v1`, the code enforces

- no direct `full_meta` request as the depth-0 core pool,
- default narrow core pool `paop_lf_std`,
- residual pool `full_meta`.

So the staged pool is
$$
\mathcal P_{\mathrm{staged}}=\mathcal P_{\mathrm{core}}\cup \mathcal P_{\mathrm{residual}},
$$
with
$$
\mathcal P_{\mathrm{core}}=\mathcal P_{\mathrm{paop\_lf\_std}},
\qquad
\mathcal P_{\mathrm{residual}}=\mathcal P_{\mathrm{full\_meta}}\backslash \mathcal P_{\mathrm{core}}.
$$

## 10.9 Stage controller

The implemented stage controller in `hh_continuation_stage_control.py` uses the stage chain
$$
\texttt{seed} \rightarrow \texttt{core} \rightarrow \texttt{residual}.
$$

The `seed` stage is entered first in staged HH continuation and then resolves immediately:

- `seed -> core` after optional motif seeding completes,
- if no seed block is inserted, the code records `seed_skipped_or_empty` and still moves to `core`,
- `core -> residual` when the drop plateau patience is hit **and** no trough is detected,
- `residual` remains `residual` while open.

Position probing is enabled when one of the implemented triggers fires:

- drop plateau,
- `eps_grad` + finite-angle flatness,
- repeated-family flatness.

The live controller defaults are

- `plateau_patience = 2`,
- `probe_margin_ratio = 1.0`,
- `max_probe_positions = 6`,
- `append_admit_threshold = 0.05`,
- `family_repeat_patience = 2`.

These are defaults, not hard structural limits. In the current implementation, the probed positions are `append`, `0`, and the active reoptimization window indices. So if the active window is made maximally large and `phase1_probe_max_positions` is also made large enough, the probe set spans the full scaffold.

That full-scaffold regime is already present in the March 21, 2026 artifact comparisons

- `artifacts/reports/pareto_lean_l2_vs_heavy_L2_ecut1_comparison.md`,
- `artifacts/reports/pareto_lean_vs_heavy_L3_comparison.md`,

which both record

- `adapt_window_size = 999999`,
- `adapt_window_topk = 999999`,
- `phase1_probe_max_positions = 999999`,

so insertion is effectively allowed at any scaffold position when probing is triggered.

Implemented surfaces:

- `src/quantum/operator_pools/polaron_paop.py`
- `pipelines/hardcoded/adapt_pipeline.py`
- `pipelines/hardcoded/hh_continuation_stage_control.py`
- `pipelines/hardcoded/hh_continuation_scoring.py`

# 11. SPSA and Optimizer Semantics

## 11.1 SPSA schedules

The implemented SPSA schedules are
$$
c_k=\frac{c}{(k+1)^\gamma},
\qquad
a_k=\frac{a}{(A+k+1)^\alpha}.
$$

## 11.2 Two-point stochastic gradient

At iteration `k`, with random Rademacher vector `\Delta_k`, the objective is sampled at
$$
y_+=f(x_k+c_k\Delta_k),
\qquad
y_-=f(x_k-c_k\Delta_k),
$$
and the implemented gradient estimate is
$$
\hat g_k=\frac{y_+-y_-}{2c_k}\,\Delta_k.
$$

## 11.3 Update and projection

The parameter update is
$$
x_{k+1}=x_k-a_k\hat g_k.
$$

If clipping projection is enabled, then clipping is applied

1. before evaluating `x_+` and `x_-`,
2. after the parameter update.

## 11.4 Repeat aggregation

If `eval_repeats > 1`, then the same objective point is evaluated multiple times and aggregated by
$$
\operatorname{mean}\quad\text{or}\quad\operatorname{median}.
$$

## 11.5 Return policy

The implemented return policy is

- if `avg_last > 0`, return the Polyak-style average of the last `avg_last` iterates and evaluate it once more;
- if `avg_last = 0`, return the best observed sampled point among the evaluated `x_+` and `x_-` points.

## 11.6 Surface note

`pipelines/hardcoded/adapt_pipeline.py` currently defaults the HH ADAPT inner optimizer to `SPSA`, while non-ADAPT hardcoded VQE surfaces may also use deterministic optimizers such as `COBYLA` or `SLSQP`.

Implemented surfaces:

- `src/quantum/spsa_optimizer.py`
- `pipelines/hardcoded/adapt_pipeline.py`
- `pipelines/hardcoded/hubbard_pipeline.py`

# 12. Drive, Exact Propagation, and Propagator Semantics

## 12.1 Static drive labels and time-dependent coefficients

The implemented density-drive template precomputes static `Z` labels for each `(site, spin)` pair:
$$
(i,\sigma)\mapsto Z_{p(i,\sigma)}.
$$
Only the coefficients are time dependent.

So the drive term is always of the form
$$
\hat H_{\mathrm{drive}}(t)=\sum_{\lambda\in\mathcal L_{\mathrm{drive}}} c_{\lambda}(t) P_{\lambda},
$$
with a fixed label set `\mathcal L_{\mathrm{drive}}` and time-varying coefficients `c_{\lambda}(t)`.

## 12.2 Reference coefficient map

For the density-drive helper,
$$
\Delta c[Z_{p(i,\sigma)}](t)=-\frac{1}{2}v_i(t),
$$
and optionally
$$
\Delta c[I](t)=\sum_i v_i(t).
$$

This is the direct runtime version of the substituted drive Hamiltonian from Section 6.3.

## 12.3 Propagator surfaces

The non-ADAPT hardcoded pipeline exposes propagators including

- `suzuki2`,
- `piecewise_exact`,
- `cfqm4`,
- `cfqm6`.

For CFQM surfaces, the repository policy is explicit:

- CFQM ignores midpoint/left/right `drive-time-sampling`,
- CFQM uses its own fixed scheme nodes `c_j`.

So if the macro-step grid is `t_n`, a CFQM stage samples at
$$
t_n+c_j\Delta t,
$$
not at left, midpoint, or right rule points chosen by a legacy sampler flag.

Code anchor: `pipelines/hardcoded/hubbard_pipeline.py`

## 12.4 Reference propagator versus reported trajectory propagator

The hardcoded pipeline now uses an explicit split between

1. the **reported trajectory propagator** selected by `--propagator`, and
2. the **reference / exact branch propagator** used for fidelity and exact-observable comparison.

Write
$$
N_{\mathrm{macro}}=\texttt{trotter\_steps},
\qquad
M_{\mathrm{ref}}=\max\{1,\texttt{exact\_steps\_multiplier}\},
\qquad
N_{\mathrm{ref}}=M_{\mathrm{ref}}N_{\mathrm{macro}}.
$$

### 12.4.1 Static reference branch

If the drive is disabled, the reference branch is the exact eigendecomposition evolution
$$
\hat H_{\mathrm{static}}=V\Lambda V^{\dagger},
\qquad
|\psi_{\mathrm{ref}}(t)\rangle
=
V e^{-i\Lambda t}V^{\dagger}|\psi(0)\rangle.
$$

### 12.4.2 Drive-enabled reference branch

If the drive is enabled, the reference branch is no longer built from one static eigendecomposition. The code switches to a piecewise-constant exact reference,
$$
|\psi_{\mathrm{ref}}(t)\rangle
=
\hat U_{\mathrm{piecewise}}^{(\mathrm{ref})}(t)|\psi(0)\rangle,
$$
with
$$
\hat U_{\mathrm{piecewise}}^{(\mathrm{ref})}(t)
\approx
\prod_{r=0}^{N_{\mathrm{ref}}-1}
\exp\!\left[-i\,\Delta t_{\mathrm{ref}}\,\hat H\!\left(t_r^{\star}\right)\right],
\qquad
\Delta t_{\mathrm{ref}}=\frac{t}{N_{\mathrm{ref}}},
$$
where `t_r^*` follows the selected time-sampling rule for the piecewise reference path.

So the implemented refinement law is explicit:
$$
N_{\mathrm{ref}}=\texttt{exact\_steps\_multiplier}\times \texttt{trotter\_steps}.
$$

This multiplier changes the reference path only. It does **not** change the macro-step count of `cfqm4` or `cfqm6`.

Code anchor: `pipelines/hardcoded/hubbard_pipeline.py`

## 12.5 CFQM macro-step mathematics

For CFQM propagation the hardcoded pipeline loads a scheme
$$
\{(a_m,c_m)\}_{m=1}^{M}
$$
from `get_cfqm_scheme(...)` and validates it with `validate_scheme(...)`.

For one macro-step of size `\Delta t`, the implemented CFQM mathematical surface is
$$
\hat U_{\mathrm{CFQM}}(\Delta t;t_n)
\approx
\prod_{m=1}^{M}
\exp\!\left[
-i\,a_m\Delta t\,\hat H(t_n+c_m\Delta t)
\right].
$$

The exposed scheme ids are
$$
\texttt{cfqm4}\equiv \texttt{CF4:2},
\qquad
\texttt{cfqm6}\equiv \texttt{CF6:5Opt}.
$$

The stage exponential backend is then chosen from
$$
\texttt{cfqm\_stage\_exp}\in
\{\texttt{expm\_multiply\_sparse},\ \texttt{dense\_expm},\ \texttt{pauli\_suzuki2}\}.
$$

The code also states two runtime semantics directly:

- midpoint/left/right sampling is ignored on CFQM paths because the sample nodes are the fixed `c_m`;
- `pauli_suzuki2` changes the stage exponential implementation and therefore collapses the overall method to a second-order inner product-formula realization.

Code anchors:

- `src/quantum/time_propagation/cfqm_schemes.py`
- `src/quantum/time_propagation/cfqm_propagator.py`
- `pipelines/hardcoded/hubbard_pipeline.py`

## 12.6 Filtered exact manifold and subspace fidelity

The hardcoded pipeline does not compare a propagated trial state only against a single exact vector. It constructs a filtered exact manifold at `t=0` by selecting every exact state in the target sector whose energy satisfies
$$
E\le E_0+\varepsilon_{\mathrm{subspace}},
\qquad
\varepsilon_{\mathrm{subspace}}=\texttt{fidelity\_subspace\_energy\_tol}.
$$

If the resulting orthonormal basis at `t=0` is
$$
V_0=[|v_1(0)\rangle,\dots,|v_r(0)\rangle],
$$
then each basis vector is propagated by the same reference propagator used for the exact branch, producing
$$
V(t)=[|v_1(t)\rangle,\dots,|v_r(t)\rangle].
$$

After re-orthonormalization the time-dependent projector is
$$
\hat P_{\mathrm{gs,filtered}}(t)=V(t)V(t)^{\dagger}.
$$

The reported projected fidelities are therefore
$$
F_{\mathrm{legacy}}(t)
=
\langle \psi_{\mathrm{legacy,trot}}(t)|
\hat P_{\mathrm{gs,filtered}}(t)
|\psi_{\mathrm{legacy,trot}}(t)\rangle,
$$
$$
F_{\mathrm{paop}}(t)
=
\langle \psi_{\mathrm{paop,trot}}(t)|
\hat P_{\mathrm{gs,filtered}}(t)
|\psi_{\mathrm{paop,trot}}(t)\rangle,
$$
$$
F_{\mathrm{hva}}(t)
=
\langle \psi_{\mathrm{hva,trot}}(t)|
\hat P_{\mathrm{gs,filtered}}(t)
|\psi_{\mathrm{hva,trot}}(t)\rangle.
$$

The compatibility key `fidelity` follows the currently selected legacy branch, while the branch-resolved keys

- `fidelity_paop_trotter`,
- `fidelity_hva_trotter`

remain explicit in the payload.

Code anchor: `pipelines/hardcoded/hubbard_pipeline.py`

## 12.7 Branch-resolved observable contracts

The hardcoded pipeline now propagates several explicit branch states instead of only one ansatz branch:

1. filtered exact-sector ground-state reference,
2. exact propagation from the PAOP / ADAPT handoff state,
3. Trotter or CFQM propagation from the PAOP / ADAPT handoff state,
4. exact propagation from the hardcoded VQE branch,
5. Trotter or CFQM propagation from the hardcoded VQE branch.

Write these as
$$
|\psi_{\mathrm{gs,exact}}(t)\rangle,\ 
|\psi_{\mathrm{paop,exact}}(t)\rangle,\ 
|\psi_{\mathrm{paop,prop}}(t)\rangle,\ 
|\psi_{\mathrm{hva,exact}}(t)\rangle,\ 
|\psi_{\mathrm{hva,prop}}(t)\rangle.
$$

The static-energy observable on any branch `b` is
$$
E_{\mathrm{static}}^{(b)}(t)
=
\langle \psi_b(t)|\hat H_{\mathrm{static}}|\psi_b(t)\rangle.
$$

If the drive is enabled, the total instantaneous energy on branch `b` is
$$
E_{\mathrm{total}}^{(b)}(t)
=
\left\langle \psi_b(t)\left|
\hat H_{\mathrm{static}}
+\hat H_{\mathrm{drive}}(t_{\mathrm{drive}})
\right|\psi_b(t)\right\rangle,
\qquad
t_{\mathrm{drive}}=\texttt{drive\_t0}+t.
$$

The site-resolved densities and doublon observables are evaluated branchwise as
$$
n_{i\uparrow}^{(b)}(t)=\langle \psi_b(t)|\hat n_{i\uparrow}|\psi_b(t)\rangle,
\qquad
n_{i\downarrow}^{(b)}(t)=\langle \psi_b(t)|\hat n_{i\downarrow}|\psi_b(t)\rangle,
$$
$$
n_i^{(b)}(t)=n_{i\uparrow}^{(b)}(t)+n_{i\downarrow}^{(b)}(t),
\qquad
D^{(b)}(t)=\sum_i\langle \psi_b(t)|\hat n_{i\uparrow}\hat n_{i\downarrow}|\psi_b(t)\rangle.
$$

When the drive is disabled, the code path reduces to
$$
E_{\mathrm{total}}^{(b)}(t)=E_{\mathrm{static}}^{(b)}(t).
$$

Code anchor: `pipelines/hardcoded/hubbard_pipeline.py`

## 12.8 `adapt_json` import and handoff contract

The hardcoded pipeline exposes an explicit import surface
$$
\texttt{initial\_state\_source}=\texttt{adapt\_json},
$$
with the required input path
$$
\texttt{adapt\_input\_json}.
$$

The imported statevector is the amplitude payload written by the handoff bundle path, so the mathematical initial condition is
$$
|\psi_{\mathrm{paop}}(0)\rangle = |\psi_{\mathrm{imported\ adapt}}(0)\rangle.
$$

The hardcoded pipeline also performs a physics-setting comparison between the current run arguments and the imported JSON metadata. If the mismatch set is
$$
\mathcal M
=
\{\text{field}\mid \text{current setting} \neq \text{imported setting}\},
$$
then:

- if `adapt_strict_match = true` and `\mathcal M\neq \varnothing`, execution stops with an error;
- if `adapt_strict_match = false`, the mismatches are retained as provenance and the import continues.

The imported-handoff payload therefore has two simultaneous meanings:

1. it defines the propagated PAOP branch initial state,
2. it defines an auditable metadata match or mismatch contract through fields such as
   - `metadata_match_passed`,
   - `metadata_mismatches`,
   - `pool_type`,
   - `ansatz_depth`,
   - `energy`,
   - `abs_delta_e`.

Code anchors:

- `pipelines/hardcoded/hubbard_pipeline.py`
- `pipelines/hardcoded/handoff_state_bundle.py`

## 12.9 Exposed phase-3 continuation surfaces

The hardcoded pipeline now exposes several continuation-era control surfaces even when this manuscript does not upgrade them into stronger mathematical claims than the code presently supports.

The exposed continuation-mode surface is
$$
\texttt{adapt\_continuation\_mode}\in
\{\texttt{legacy},\texttt{phase1\_v1},\texttt{phase2\_v1},\texttt{phase3\_v1}\}.
$$

The exposed symmetry-mitigation mode names are
$$
\texttt{phase3\_symmetry\_mitigation\_mode}\in
\{\texttt{off},\texttt{verify\_only},\texttt{postselect\_diag\_v1},\texttt{projector\_renorm\_v1}\}.
$$

The exposed runtime split surface is
$$
\texttt{phase3\_runtime\_split\_mode}\in
\{\texttt{off},\texttt{shortlist\_pauli\_children\_v1}\}.
$$

Additional phase-3-adjacent control surfaces include

- `phase3_motif_source_json`,
- `phase3_enable_rescue`,
- `phase3_lifetime_cost_mode`.

This manuscript therefore treats those names as **implemented runtime surfaces**. It does **not** promote every broader continuation-theory claim from the design notes into a present-tense statement unless a code anchor is present in the main implementation.

Implemented surfaces:

- `src/quantum/drives_time_potential.py`
- `src/quantum/time_propagation/cfqm_propagator.py`
- `src/quantum/time_propagation/cfqm_schemes.py`
- `pipelines/hardcoded/hubbard_pipeline.py`

# 13. Implemented Continuation, Handoff, and Replay Contract

## 13.1 Handoff state bundle

The canonical non-interactive handoff surface is `pipelines/hardcoded/handoff_state_bundle.py`.

The written payload contains

- the HH settings manifest,
- the ADAPT/VQE energy summary,
- the normalized statevector amplitudes in `q_(N_q-1)...q_0` order,
- the exact sector energy,
- the optional continuation block.

The amplitude dictionary is
$$
\{\texttt{bitstring}_{q_{N_q-1}\cdots q_0} \mapsto (\Re a,\Im a)\},
$$
with all amplitudes below the chosen cutoff omitted.

## 13.2 Continuation block

The continuation payload is no longer hypothetical. The implemented continuation block may contain

- `mode`,
- `scaffold`,
- `optimizer_memory`,
- `selected_generator_metadata`,
- `generator_split_events`,
- `runtime_split_summary`,
- `motif_library`,
- `motif_usage`,
- `symmetry_mitigation`,
- `rescue_history`,
- `replay_contract`,
- optionally `replay_contract_hint` as a compatibility hint written by bundle tooling.

A schematic shape is

```yaml
continuation:
  mode: phase1_v1 | phase2_v1 | phase3_v1 | legacy
  scaffold: {...}
  optimizer_memory: {...}
  selected_generator_metadata: [...]
  generator_split_events: [...]
  runtime_split_summary: {...}
  motif_library: {...}
  motif_usage: {...}
  symmetry_mitigation: {...}
  rescue_history: [...]
  replay_contract: {...}
  replay_contract_hint: {...}   # compatibility-only hint when present
```

This is the implemented successor to the older continuation discussion.

## 13.3 Replay contract

The replay-side consumer is the continuation family/replay stack under

- `pipelines/hardcoded/hh_vqe_from_adapt_family.py`,
- `pipelines/hardcoded/hh_continuation_generators.py`,
- `pipelines/hardcoded/hh_continuation_motifs.py`,
- `pipelines/hardcoded/hh_continuation_symmetry.py`.

The mathematical meaning of the replay payload is straightforward:

- the statevector amplitudes define the initial state,
- the continuation block defines how the selected generators were staged, split, scored, reused, and annotated,
- the replay consumer reconstructs a compatible continuation trajectory from those stored objects.

Modern replay treats
$$
\texttt{continuation.replay\_contract}
$$
as the authoritative contract when present. The implemented v2 shape is

```yaml
continuation:
  replay_contract:
    contract_version: 2
    continuation_mode: legacy | phase1_v1 | phase2_v1 | phase3_v1
    generator_family:
      requested: match_adapt | <canonical_hh_family>
      resolved: <canonical_hh_family>
      resolution_source: <string>
      fallback_family: <canonical_hh_family> | null
      fallback_used: true | false
    seed_policy_requested: auto | scaffold_plus_zero | residual_only | tile_adapt
    seed_policy_resolved: scaffold_plus_zero | residual_only | tile_adapt
    handoff_state_kind: prepared_state | reference_state
    provenance_source: contract | explicit | inferred_source | ambiguous
    adapt_depth: <int>
    reps: <int>
    derived_num_parameters_formula: adapt_depth * reps
    derived_num_parameters: <int>
    replay_block_source: adapt_vqe.operators
    seed_source: adapt_vqe.optimal_point
```

The live auto-resolution rule is

- `auto -> residual_only` for `handoff_state_kind = prepared_state`,
- `auto -> scaffold_plus_zero` for `handoff_state_kind = reference_state`.

So `replay_contract` now carries the authoritative family resolution, seed-policy resolution, handoff-state semantics, and replay provenance. When `replay_contract` is present, any remaining `replay_contract_hint` should be read as compatibility metadata rather than the primary contract.

## 13.4 Symmetry and phase-3 payload note

The raw staged ADAPT CLI already exposes phase-3 symmetry, runtime-split, motif, and rescue surfaces. This manuscript therefore states them as implemented payload/data-contract surfaces without overstating raw staged-ADAPT enforcement beyond what the current code actually does.

In particular, the current mainline repository does document and enforce the runtime-split symmetry contract, but it does **not** expose a separate active `beam` source surface in the tracked mainline code. Historical beam-labelled artifacts should therefore be read as provenance from earlier workflows, not as a distinct current CLI contract. The current invariant stated in this manuscript is the stronger one that the live code and tests enforce now: any admitted runtime-split child or `child_set[...]` replacement must satisfy the required particle-number / spin-sector symmetry gate, while violating child atoms are tagged and hard-guarded in metadata rather than admitted.

## 13.5 Optional staged seed-refine insertion and provenance

Code anchors:

- `pipelines/hardcoded/hh_staged_workflow.py`
- `pipelines/hardcoded/hh_vqe_from_adapt_family.py`
- `pipelines/hardcoded/handoff_state_bundle.py`

The default staged HH chain remains
$$
|\psi_{\mathrm{HF}}\rangle
\xrightarrow{U_{\mathrm{warm}}(\theta)}
|\psi_{\mathrm{warm}}\rangle
\xrightarrow{\mathrm{ADAPT}_{\mathrm{phase3\_v1}}}
|\psi_{\mathrm{ADAPT}}\rangle
\xrightarrow{\mathrm{replay}_{\mathrm{match\_adapt}}}
|\psi_{\mathrm{final}}\rangle,
$$
with $U_{\mathrm{warm}}$ built from `hh_hva_ptw` by default and `hh_hva` kept as
an override-only alternative.

When `--seed-refine-family` is enabled, the workflow inserts one explicit-family
conventional VQE stage:
$$
|\psi_{\mathrm{HF}}\rangle
\xrightarrow{U_{\mathrm{warm}}(\theta)}
|\psi_{\mathrm{warm}}\rangle
\xrightarrow{U_{\mathrm{family}}(\vartheta)}
|\psi_{\mathrm{refined}}\rangle
\xrightarrow{\mathrm{ADAPT}_{\mathrm{phase3\_v1}}}
|\psi_{\mathrm{ADAPT}}\rangle
\xrightarrow{\mathrm{replay}_{\mathrm{match\_adapt}}}
|\psi_{\mathrm{final}}\rangle.
$$

The family stage is explicit-only:

- it materializes the requested family directly,
- it does not use `match_adapt`,
- it does not auto-fallback to `full_meta`,
- failure aborts before ADAPT rather than silently skipping ahead.

When a refine stage is present, the handoff bundle may include additive
provenance
```yaml
seed_provenance:
  warm_ansatz: hh_hva_ptw | hh_hva
  refine_family: <explicit_family>
  refine_family_kind: explicit_family | <future-kind>
  refine_paop_motif_families: [<motif_family>, ...]
  refine_reps: <int>
```
Missing `seed_provenance` remains a valid legacy state and means “no refine
stage”. Replay family resolution ignores this block; it remains provenance only.

# 14. Cross-Check Suite and Exact-Benchmark Contracts

## 14.1 Trial matrix by problem family

Code anchor: `pipelines/exact_bench/cross_check_suite.py`

The cross-check suite implements an exact-benchmark matrix over ansatz families and ADAPT modes.

For pure Hubbard the implemented trial set is
$$
\mathcal T_{\mathrm{Hub}}
=
\{
\text{HVA-Layerwise},
\text{UCCSD-Layerwise},
\text{ADAPT(UCCSD)},
\text{ADAPT(full\_H)}
\}.
$$

For Hubbard-Holstein the implemented trial set is
$$
\mathcal T_{\mathrm{HH}}
=
\{
\text{HH-Termwise},
\text{HH-Layerwise},
\text{ADAPT(full\_H)}
\}.
$$

This remains the default HH matrix. When the opt-in seed surface is enabled via
`--hh-seed-refine-surface`, the suite additively appends the conventional seed
comparison arms
$$
\mathcal T_{\mathrm{HH}}^{\mathrm{seed}}
=
\mathcal T_{\mathrm{HH}}
\cup
\{
\text{HH-PhysicalTermwise},
\text{HH-PhysicalTermwise}+\texttt{uccsd\_paop\_lf\_full},
\text{HH-PhysicalTermwise}+\texttt{uccsd\_otimes\_paop\_lf\_std},
\text{HH-PhysicalTermwise}+\texttt{uccsd\_otimes\_paop\_lf2\_std},
\text{HH-PhysicalTermwise}+\texttt{uccsd\_otimes\_paop\_bond\_disp\_std}
\}.
$$

On the smallest-system comparison only, the suite additionally includes the
reduced arm
$$
\text{HH-Layerwise}+\texttt{uccsd\_otimes\_paop\_lf\_std}.
$$

Each trial uses the same Hamiltonian instance for the chosen problem and then compares its variational energy against the same sector-filtered exact reference.

## 14.2 Auto-scaled parameter resolution

Code anchor: `pipelines/exact_bench/cross_check_suite.py`

The suite resolves its runtime parameters by an explicit override-or-table rule. If `x` is a configurable parameter such as

- `vqe_reps`,
- `vqe_restarts`,
- `vqe_maxiter`,
- `vqe_method`,
- `trotter_steps`,
- `num_times`,
- `t_final`,

then the resolved value is
$$
x_{\mathrm{resolved}}
=
\begin{cases}
x_{\mathrm{CLI}}, & x_{\mathrm{CLI}}\neq \varnothing,\\
x_{\mathrm{table}}, & x_{\mathrm{CLI}}=\varnothing.
\end{cases}
$$

For HH runs the suite also enforces the method contract
$$
\texttt{vqe\_method}=\texttt{SPSA},
$$
so HH cross-checks do not silently drop onto legacy deterministic optimizer settings.

The three table sources are separated in code:

- Hubbard trajectory / VQE defaults from `_get_hubbard_params(L)`,
- HH enrichment from `_get_hh_params(L,n_{\mathrm{ph,max}})`,
- ADAPT defaults from `_get_adapt_params(L)`.

## 14.3 Exact target and shared reference-state construction

Code anchor: `pipelines/exact_bench/cross_check_suite.py`

The cross-check suite constructs one Hamiltonian matrix
$$
\hat H \longrightarrow H_{\mathrm{mat}},
$$
then computes the exact sector-filtered target
$$
E_{\mathrm{exact,filtered}}
=
\min_{\psi\in \mathcal S_{\mathrm{sector}}}
\langle \psi|\hat H|\psi\rangle.
$$

For HH, the common reference state is
$$
|\psi_{\mathrm{ref}}^{\mathrm{HH}}\rangle
=
|\mathrm{vac}_{\mathrm{ph}}\rangle\otimes|\Phi_{\mathrm{HF}}\rangle.
$$

For pure Hubbard, the common reference state is the HF basis state
$$
|\psi_{\mathrm{ref}}^{\mathrm{Hub}}\rangle
=
|\Phi_{\mathrm{HF}}\rangle.
$$

So every variational trial begins from the same physical reference within a given problem instance, and only the ansatz family or ADAPT pool changes.

## 14.4 Per-trial energy and trajectory semantics

Code anchor: `pipelines/exact_bench/cross_check_suite.py`

If trial `\tau` returns variational state `|\psi_{\tau}\rangle`, then its benchmark energy gap is
$$
\Delta E_{\tau}
=
E_{\tau}-E_{\mathrm{exact,filtered}},
\qquad
|\Delta E_{\tau}|=\texttt{delta\_E\_abs}.
$$

After VQE or ADAPT state preparation, every trial is propagated on the same common grid:
$$
t_n\in \mathrm{linspace}(0,t_{\mathrm{final}},N_t),
\qquad
N_t=\texttt{num\_times}.
$$

The propagated trajectory of each trial then stores the same observable families:

- projected fidelity,
- exact and Trotter energy,
- site-0 spin occupations,
- total doublon.

So the trajectory comparison is controlled: the ansatz family changes, but the Hamiltonian, reference target, and propagation grid do not.

## 14.5 JSON and PDF artifact contracts

Code anchor: `pipelines/exact_bench/cross_check_suite.py`

The machine-readable payload begins with
$$
\{
\texttt{cross\_check\_suite}: \texttt{true},
\texttt{parameters}: \cdots,
\texttt{exact\_ground\_energy}: E_{\mathrm{exact,filtered}},
\texttt{trials}: [\cdots]
\}.
$$

For each trial, the summary contract includes

- `energy`,
- `exact_energy`,
- `delta_e`,
- `delta_E_abs`,
- `num_parameters`,
- `nfev`,
- `runtime_s`,
- ADAPT depth and stop-reason fields when the method kind is ADAPT.

The PDF contract is not only a raw command dump. It contains

1. a parameter manifest,
2. an executive summary,
3. a scoreboard table,
4. per-trial trajectory pages,
5. overlay appendix pages,
6. the executed command page.

So the cross-check suite is mathematically a multi-trial map
$$
\tau\mapsto
\left(
E_{\tau},
\Delta E_{\tau},
\{\mathcal O_{\tau}(t_n)\}_{n=0}^{N_t-1}
\right),
$$
with a fixed exact target and fixed time grid.

Implemented surfaces:

- `pipelines/exact_bench/cross_check_suite.py`

## 14.6 HH seed-surface preset and proxy sidecars

For the current opt-in HH seed benchmark preset, the suite runs the four-point
set
$$
(L,g)\in \{2,3\}\times\{0.8,1.2\},
$$
with shared HH choices
$$
t=1,\qquad U=4,\qquad \omega_0=1,\qquad n_{\mathrm{ph,max}}=1,
$$
and the repository HH boundary/ordering convention.

For a warm-only baseline trial $\tau_0$ and a refined seed trial $\tau$, define
the absolute-energy improvement
$$
I(\tau\mid \tau_0)
=
|\Delta E_{\tau_0}|-|\Delta E_{\tau}|.
$$

Let the additive preparation-cost deltas be
$$
\Delta c_{\tau}=c_{\tau}-c_{\tau_0},\qquad
\Delta d_{\tau}=d_{\tau}-d_{\tau_0},\qquad
\Delta s_{\tau}=s_{\tau}-s_{\tau_0},
$$
where $(c_\tau,d_\tau,s_\tau)$ denote the explicit `cx_proxy`,
`depth_proxy`, and `sq_proxy` values persisted through the proxy sidecars.

Then the comparison views are
$$
\eta_{\tau}^{(\mathrm{cx})}
=
\frac{I(\tau\mid\tau_0)}{\max(\Delta c_{\tau},1)},
\qquad
\eta_{\tau}^{(\mathrm{depth})}
=
\frac{I(\tau\mid\tau_0)}{\max(\Delta d_{\tau},1)},
\qquad
\eta_{\tau}^{(\mathrm{sq})}
=
\frac{I(\tau\mid\tau_0)}{\max(\Delta s_{\tau},1)}.
$$

The decision rule uses $\eta_{\tau}^{(\mathrm{cx})}$ as the primary ranking
metric, with the depth- and single-qubit-normalized views retained as
secondaries. The proxy formulas themselves remain sourced from
`docs/reports/qiskit_circuit_report.py` through
`compute_sweep_proxy_cost(...)`, `compute_time_dynamics_proxy_cost(...)`, and
the additive `ProxyMetricRow` sidecar schema.

# 15. Noise Validation, Symmetry-Mitigation, and Legacy-Parity Contracts

## 15.1 Filtered versus full exact targets

Code anchor: `pipelines/exact_bench/hh_noise_hardware_validation.py`

The noise-validation path stores two distinct exact energies:
$$
E_{\mathrm{exact,filtered}}
=
\min_{\psi\in \mathcal S_{\mathrm{sector}}}\langle \psi|\hat H|\psi\rangle,
\qquad
E_{\mathrm{exact,full}}
=
\min \mathrm{spec}(H_{\mathrm{mat}}).
$$

The filtered energy is the primary VQE comparison target in HH validation, while the full-Hilbert minimum is retained separately as provenance.

## 15.2 Noisy-minus-ideal observable deltas

Code anchor: `pipelines/exact_bench/hh_noise_hardware_validation.py`

For each trajectory observable `\mathcal O(t)` on the validation grid, the payload records a noisy value, an ideal value, and a difference field
$$
\Delta \mathcal O(t)
=
\mathcal O_{\mathrm{noisy}}(t)-\mathcal O_{\mathrm{ideal}}(t).
$$

In particular the validation runner explicitly stores
$$
\Delta E_{\mathrm{static}}(t)
=
E_{\mathrm{static,trotter}}^{\mathrm{noisy}}(t)
-
E_{\mathrm{static,trotter}}^{\mathrm{ideal}}(t),
$$
$$
\Delta D(t)
=
D_{\mathrm{trotter}}^{\mathrm{noisy}}(t)
-
D_{\mathrm{trotter}}^{\mathrm{ideal}}(t),
$$
with companion stderr fields carried in the payload and plotted as uncertainty bands in the PDF.

At the VQE level the same noisy-minus-ideal logic is retained:
$$
\Delta E_{\mathrm{noise}}
=
E_{\mathrm{noisy}}-E_{\mathrm{ideal\ reference}}.
$$

## 15.3 Legacy parity gate

Code anchor: `pipelines/exact_bench/hh_noise_hardware_validation.py`

The legacy parity helper compares the new ideal trajectory against a locked legacy reference trajectory.

For an observable `\mathcal O` on grid points `t_n`, the code computes
$$
\delta_{\mathcal O}^{\max}
=
\max_n
\left|
\mathcal O_{\mathrm{new,ideal}}(t_n)
-
\mathcal O_{\mathrm{legacy}}(t_n)
\right|,
$$
$$
\delta_{\mathcal O}^{\mathrm{mean}}
=
\frac{1}{N_t}\sum_n
\left|
\mathcal O_{\mathrm{new,ideal}}(t_n)
-
\mathcal O_{\mathrm{legacy}}(t_n)
\right|,
$$
$$
\delta_{\mathcal O}^{\mathrm{final}}
=
\left|
\mathcal O_{\mathrm{new,ideal}}(t_{N_t-1})
-
\mathcal O_{\mathrm{legacy}}(t_{N_t-1})
\right|.
$$

The pass condition is conjunctive:
$$
\texttt{passed}_{\mathcal O}
\iff
\bigl(\texttt{time\_grid\_match}=\mathrm{true}\bigr)
\land
\bigl(\delta_{\mathcal O}^{\max}\le \texttt{tolerance}\bigr).
$$

The global verdict is
$$
\texttt{passed\_all}
\iff
\texttt{time\_grid\_match}
\land
\bigwedge_{\mathcal O\in\mathcal O_{\mathrm{cmp}}}
\texttt{passed}_{\mathcal O}.
$$

This is stricter than a pointwise visual comparison: an unequal time grid already fails the parity contract.

## 15.4 Same-anchor paired comparison semantics

Code anchor: `pipelines/exact_bench/hh_noise_hardware_validation.py`

The paired-anchor validation path evaluates one common workload / observable pair across several execution modes while retaining shared-anchor evidence.

Abstractly it evaluates a fixed tuple
$$
\bigl(\hat U,\hat O,\text{lock family}\bigr)
$$
across several mode labels `m`, producing rows
$$
r_m=
\left(
\text{mode}_m,\,
\text{executor}_m,\,
\text{noise kind}_m,\,
\mu_m,\,
\sigma_m,\,
\text{layout/provenance hashes}_m
\right).
$$

The comparability payload then records whether these rows share anchor evidence such as

- backend identity,
- snapshot hash,
- layout hash,
- used physical qubits,
- used physical edges,
- circuit-structure hash,
- runtime-execution-bundle relationships.

So the same-anchor block is mathematically a controlled multi-mode comparison of one fixed workload rather than a loose comparison of different circuits.

## 15.5 Implemented symmetry-mitigation mode surface

Code anchor: `pipelines/exact_bench/hh_noise_hardware_validation.py`

The validation runner exposes the symmetry-mitigation mode names
$$
\texttt{symmetry\_mitigation\_mode}\in
\{
\texttt{off},
\texttt{verify\_only},
\texttt{postselect\_diag\_v1},
\texttt{projector\_renorm\_v1}
\}.
$$

These modes are normalized into the noisy-oracle configuration and, separately, into the ideal-reference symmetry-mitigation configuration.

This manuscript therefore states the implemented contract conservatively:

1. the mode names are live runtime inputs,
2. the chosen mode is retained in the validation provenance,
3. the PDF and JSON surfaces report the mitigation configuration,
4. the main-body mathematics here does not claim more internal estimator algebra than the code surfaces explicitly expose in this audit.

## 15.6 Manifest and provenance hashes

Code anchor: `pipelines/exact_bench/hh_noise_hardware_validation.py`

The validation payload tracks provenance through explicit hash and lock fields rather than only through human-readable text.

The recorded identifiers include

- `resolved_noise_spec_hash`,
- `snapshot_hash`,
- `layout_hash`,
- `transpile_hash`,
- `circuit_structure_hash`,
- `noise_artifact_hash`,
- `layout_anchor_source`,
- `runtime_execution_bundle`.

So the validation artifact is mathematically a tuple
$$
\mathcal V=
\bigl(
\text{settings},
\text{filtered/full exact targets},
\text{noisy/ideal observables},
\text{legacy parity verdict},
\text{paired-anchor evidence},
\text{provenance hashes}
\bigr),
$$
not just a trajectory dump.

Implemented surfaces:

- `pipelines/exact_bench/hh_noise_hardware_validation.py`

# 16. Final Primitive-Closed Summary

The full repository-aligned substitution chain is now linear and closed:

1. choose the fermion ordering `p(i,\sigma)`,
2. write the JW ladder primitives `\hat c_p^{\dagger}`, `\hat c_p`,
3. reduce `\hat n_p` to `(I-Z_p)/2`,
4. substitute those primitives into `\hat H_t`, `\hat H_U`, `\hat H_v`,
5. write `\hat b_i`, `\hat b_i^{\dagger}`, `\hat n_{b,i}`, `\hat x_i`,
6. substitute them into `\hat H_{\mathrm{ph}}` and `\hat H_g`,
7. reduce the drive term to explicit identity-plus-`Z` coefficients,
8. propagate those operators into the statevector primitives,
9. build the current hardcoded HH ansatz families from those same operators,
10. build PAOP and staged continuation objects from those same explicit primitives,
11. define the reported propagator, reference branch, filtered exact manifold, and branch-resolved observables on top of those same operators,
12. serialize the resulting continuation and state handoff with the implemented payload contract,
13. benchmark multiple conventional and ADAPT trial families against the same sector-filtered exact target on a common propagation grid,
14. validate noisy, ideal, and legacy trajectories with explicit parity gates and provenance hashes.

This is exactly the manuscript shape the current repository supports: linear, substitution-first, implementation-backed, and extended through the current propagation, benchmark, and validation contracts without promoting speculative future design claims into the main body.

# Appendix A. Spec-only or not-located surfaces kept out of the main body

This appendix records material that appears in `MATH/IMPLEMENT_SOON.md` or `MATH/IMPLEMENT_NEXT.md` but is **not** promoted into the main manuscript as an implemented claim unless a direct code anchor was located in the scoped audit above.

## A.1 Broader continuation-theory items

The two implementation-spec documents describe a broader continuation architecture, including items such as

- larger live selective macro-splitting frameworks,
- richer motif transfer and multi-source tiling,
- broader active symmetry-mitigation architecture,
- extended replay / rescue / continuation scoring theory.

Status in this manuscript:

- these ideas are **not** used as main-body source material;
- only runtime surfaces actually located in code, such as exposed mode names and handoff payloads, are promoted into present-tense chapters above.

## A.2 Amplitude-comparison compare path

The drive-amplitude comparison compare-pipeline path is documented in `pipelines/run_guide.md`, including PDF and metrics-artifact descriptions.

Status in this audit:

- **not located in current scoped code search** across the inspected `pipelines/hardcoded/`, `pipelines/exact_bench/`, and `src/quantum/` surfaces;
- therefore it remains appendix-only here and is not stated as a present-tense implemented contract in the main manuscript.

## A.3 Appendix rule

The operational rule for this manuscript is therefore:

1. main body = code-backed present-tense mathematics and payload/report contracts,
2. appendix = spec-only, design-note, or not-located material that should not be mistaken for implemented behavior in this checkout.

# Addendum (2026-04-07). Former `MATH/Math.md` main-body sections displaced by the canonical manuscript rebase

## Former Math.md §11

# 11. Adaptive Selection and Staged Continuation

## 11.1 Cumulative phase construction for adaptive selection

Rather than treating Phase 1, Phase 2, and Phase 3 as three unrelated selector passes tried in reverse order, the current HH continuation logic is best read cumulatively. We write
$$
\mathcal G^{(3)} \subseteq \mathcal G^{(2)} \subseteq \mathcal G^{(1)} \subseteq \mathcal G,
$$
where $\mathcal G$ is the full expensive master pool, but the later phases reuse the earlier scoring and refinement machinery rather than replacing it with an unrelated fallback pass.

In that cumulative reading:

- **Phase 1** provides the broad cheap-screen surface over candidate-position records,
- **Phase 2** reuses the Phase-1 records and adds shortlist-and-full-rerank refinement,
- **Phase 3** reuses the Phase-1 and Phase-2 machinery and adds the highest-level augmentations used in current HH mainline.

Separately, the runtime controller may broaden the currently exposed candidate family through controller-stage transitions such as `seed/core $\to$ residual`; that controller-stage evolution should not be confused with the Phase-1/2/3 sophistication labels.

So the exposition below is written in the same constructive order **Phase 1 $\to$ Phase 2 $\to$ Phase 3** that the cumulative selector surface itself follows.

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

This broad score is intentionally early-stage: it privileges immediate drop estimates when the controller still has long useful runway. Later phases keep the candidate-position language but no longer value raw $\underline{\delta E}$ in the same dominant way; the later cheap/full surfaces progressively downweight immediate-drop dominance as the useful horizon contracts.

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

Phase 2 should be read as an extension of the Phase-1 cheap screen: it keeps the same candidate-position language, but adds shortlist formation and a richer reranking model on top of the earlier score. More precisely, Phase 1 does not hand Phase 2 only the single best record from §11.2.4; it hands forward a retained family of candidate-position records, and the explicit retained set is the shortlist $\mathcal S_2$ defined below by threshold or Top-$K$ on $S_1$.

### 11.3.1 Core signal, append position, refit window, and cheap screen

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
If $\pi_g(g,p)=g$ denotes projection onto generator identity, then the generators actually carried forward by the shortlist are $\pi_g(\mathcal S_2)$. We therefore do not write $\mathcal G^{(2)}=\mathcal S_2$: $\mathcal G^{(2)}$ lives in generator space, whereas $\mathcal S_2$ is a state-dependent shortlist of candidate-position records.

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

This section describes the full symbolic Phase-3 selector surface: position-aware candidate records, inherited refit windows, split-aware reranking, and post-admission local simplification. In current HH mainline, Phase 3 is not a separate reverse-order fallback pass; it reuses the earlier cheap-screen and shortlist/full-rerank machinery, then adds the highest-level augmentations used in runtime policy.

The formulas in §§11.4.1--11.4.4 are the corrected canonical Phase-3 target contract. Current repo code may still lag some details of this contract; that implementation lag is stated explicitly later in §19.4 rather than folded back into the symbolic definitions here.

### 11.4.1 Core signal, append position, refit window, and cheap screen

The cumulative Phase-3 selector is most naturally written over candidate-position pairs
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

The raw candidate-position universe entering the cumulative Phase-3 selector is
$$
\mathcal P_{\mathrm{avail}}^{(3)}
=
\left\{
r=(m,p):
m\in\mathcal G^{(3)}_{\mathrm{avail}},
\ p\in\mathcal P_m
\right\}.
$$

Let $t$ denote the current selector step, let $D_{\mathrm{left}}(t)$ be the remaining controller depth budget, let $\widehat N_{\mathrm{rem}}(t)$ be a heuristic forecast of useful remaining admissions, and define the effective useful horizon
$$
H_t=\min\!\bigl(D_{\mathrm{left}}(t),\widehat N_{\mathrm{rem}}(t)\bigr).
$$
One useful controller-level estimator writes
$$
S_t(k)=\prod_{j=1}^{k}\bigl(1-h_t(j)\bigr),
\qquad
\widehat N_{\mathrm{rem}}(t)=\sum_{k=1}^{D_{\mathrm{left}}(t)} S_t(k)\,Q_t(k),
$$
where $S_t(k)$ is a survival factor for still having a useful admission opportunity $k$ selector steps ahead, $Q_t(k)$ is the conditional usefulness of that future opportunity, and $h_t(j)$ is the corresponding controller hazard. A compact controller-level realization is
$$
u_{\mathrm{pat}}(t)=\operatorname{clip}\!\left(\max\!\left(\frac{\mathrm{plateau\_streak}(t)}{P},\frac{\mathrm{low\_streak}(t)}{L}\right),0,1\right),
$$
$$
u_{\mathrm{front}}(t)=
\left(1-\frac{s_1^{\mathrm{cheap}}(t)-s_2^{\mathrm{cheap}}(t)}{s_1^{\mathrm{cheap}}(t)+\varepsilon}\right)
\left(1+\frac{s_1^{\mathrm{cheap}}(t)}{c_s\,s_{\mathrm{ref}}^{\mathrm{cheap}}(t)+\varepsilon}\right)^{-1},
$$
$$
h_t(j)=\sigma\!\Bigl(\beta_0+\beta_{\mathrm{pat}}\nu_{\mathrm{pat}}(t)+\beta_{\mathrm{front}}\nu_{\mathrm{front}}(t)+\eta_h(j-1)\Bigr),
$$
$$
m_t=\operatorname{EWMA}_{s\in W_t^+}\!\bigl(\log \Delta_{\mathrm{adm}}(s)\bigr),
\qquad
\rho_t=\operatorname{clip}\!\left(\frac{\operatorname{EWMA}_{\mathrm{recent}}(\Delta_{\mathrm{adm}})}{\operatorname{EWMA}_{\mathrm{older}}(\Delta_{\mathrm{adm}})+\varepsilon},\rho_{\min},1\right),
$$
$$
\gamma_t=[-\log \rho_t]_+,
\qquad
s_t=\max\!\Bigl(s_{\min},\operatorname{MAD}_{s\in W_t^+}\!\bigl(\log \Delta_{\mathrm{adm}}(s)\bigr)\Bigr),
$$
$$
Q_t(k)=\Phi\!\left(\frac{m_t-\gamma_t(k-1)-\log \tau_{\mathrm{use},t}}{s_t}\right).
$$
If optimistic and pessimistic runway envelopes
$$
\widehat N_{\mathrm{rem}}^{\mathrm{lo}}(t)\le \widehat N_{\mathrm{rem}}(t)\le \widehat N_{\mathrm{rem}}^{\mathrm{hi}}(t)
$$
are also tracked, one convenient band construction is
$$
\operatorname{logit} h_t^{\mathrm{pess}}(j)=\operatorname{logit} h_t(j)+\Delta_h,
\qquad
\operatorname{logit} h_t^{\mathrm{opt}}(j)=\operatorname{logit} h_t(j)-\Delta_h,
$$
$$
m_t^{\mathrm{pess}}=m_t-\Delta_m,\quad s_t^{\mathrm{pess}}=s_t+\Delta_s,
\qquad
m_t^{\mathrm{opt}}=m_t+\Delta_m,\quad s_t^{\mathrm{opt}}=\max(s_{\min},s_t-\Delta_s),
$$
$$
\widehat N_{\mathrm{rem}}^{\mathrm{lo}}(t)=\sum_{k=1}^{D_{\mathrm{left}}(t)} S_t^{\mathrm{pess}}(k)\,Q_t^{\mathrm{pess}}(k),
\qquad
\widehat N_{\mathrm{rem}}^{\mathrm{hi}}(t)=\sum_{k=1}^{D_{\mathrm{left}}(t)} S_t^{\mathrm{opt}}(k)\,Q_t^{\mathrm{opt}}(k),
$$
and then a compact confidence and saturation summary is
$$
C_t=\frac{\widehat N_{\mathrm{rem}}^{\mathrm{lo}}(t)}{\widehat N_{\mathrm{rem}}^{\mathrm{hi}}(t)+\varepsilon},
\qquad
W_{\mathrm{sat}}(t)=1-\frac{\widehat N_{\mathrm{rem}}(t)}{D_{\mathrm{left}}(t)+\varepsilon}.
$$
The corresponding selector mode and saturation regime may be written as
$$
\pi_t=
\begin{cases}
\mathrm{fast},&\widehat N_{\mathrm{rem}}^{\mathrm{lo}}(t)>n_{\mathrm{fast}},\\
\mathrm{high\mbox{-}fidelity},&\widehat N_{\mathrm{rem}}^{\mathrm{hi}}(t)\le n_{\mathrm{slow}},\\
\mathrm{mixed},&\text{otherwise},
\end{cases}
$$
$$
\mathrm{regime}_{\mathrm{sat}}(t)=
\begin{cases}
\mathrm{hungry},&W_{\mathrm{sat}}(t)<\tau_{\mathrm{off}},\\
\mathrm{plateau},&W_{\mathrm{sat}}(t)>\tau_{\mathrm{on}},\\
\mathrm{watch},&\text{otherwise}.
\end{cases}
$$
The quantity $\widehat N_{\mathrm{rem}}(t)$ is controller telemetry rather than a physical observable: it summarizes recent selector progress and remaining useful runway, but it is not itself a hard admissibility certificate.

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
K_{\mathrm{cheap}}(r;t)
&=1+w_R\bar R_{\mathrm{rem}}(r;t)+w_D\bar D_{\mathrm{life}}(r)\\
&\quad +w_G\bar G_{\mathrm{new}}(r)+w_C\bar C_{\mathrm{new}}(r)+w_c\bar c(r).
\end{aligned}
$$
with
$$
p_{\mathrm{app}}:=|\mathcal O|
$$
for the current append slot, and
$$
\bar D_{\mathrm{life}}(r)=\frac{D_{\mathrm{raw}}(r)}{\mathrm{depth}_{\mathrm{ref}}},
\qquad
D_{\mathrm{raw}}(r)
=
\sum_{\ell\in\mathcal L_r}
\left[
2\max(\mathrm{wt}(\ell)-1,0)
+
\frac12\bigl(2\,\#_{X,Y}(\ell)+1\bigr)
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
$$
$$
C_{\mathrm{new}}(r)=G_{\mathrm{new}}(r)\,n_{\mathrm{shots/group}},
\qquad
c(r)=\mathrm{groups}_{\mathrm{new}}(r),
\qquad
\bar P_{\mathrm{opt}}(r)=\frac{|W(p)|}{\mathrm{optdim}_{\mathrm{ref}}},
$$
$$
\bar R_{\mathrm{rem}}(r;t)
=
H_t\Bigl(
\bar D_{\mathrm{life}}(r)+\bar G_{\mathrm{new}}(r)+\bar C_{\mathrm{new}}(r)+\bar c(r)+\bar P_{\mathrm{opt}}(r)
\Bigr)\mathbf 1_{\mathrm{lifetime}}.
$$
Here $\bar D_{\mathrm{life}}$ is circuit/compile burden at the candidate record, $\bar G_{\mathrm{new}}$ and $\bar C_{\mathrm{new}}$ are measurement-group and shot burden of the new term, $\bar c$ is the reuse penalty, and $\bar R_{\mathrm{rem}}$ is the horizon-weighted lifetime multiplier applied only when lifetime-cost mode is on. The offset $|p_{\mathrm{app}}-p|$ is a local insertion-displacement penalty; it measures how far the proposal sits from the append slot and should not be conflated with expected remaining depth.

To interpolate between a wide early-stage gradient screen and a later-stage cost-aware screen, define
$$
\lambda_t=\frac{H_t}{H_t+c_\lambda},
\qquad c_\lambda>0,
$$
$$
K_{\mathrm{coarse}}(t)=K_{\min}+\left\lfloor (K_{\max}-K_{\min})\frac{\widehat N_{\mathrm{rem}}^{\mathrm{hi}}(t)}{D_{\mathrm{left}}(t)+\varepsilon}\right\rfloor,
$$
and
$$
\begin{aligned}
S_{3,\nabla}(r)
&=
\Gamma_{\mathrm{stage}}(r)\Gamma_{\mathrm{sym}}(r)\,
\frac{g_{\mathrm{lcb}}(r)^2}{2\lambda_FF_{\mathrm{raw}}(r)},\\
S_{3,\mathrm{late}}(r;t)
&=
\Gamma_{\mathrm{stage}}(r)\Gamma_{\mathrm{sym}}(r)\,
\frac{g_{\mathrm{lcb}}(r)^2}{2\lambda_FF_{\mathrm{raw}}(r)}
\frac{1}{K_{\mathrm{cheap}}(r;t)+\varepsilon},\\
S_{3,\mathrm{cheap}}(r;t)
&=
\lambda_t\,S_{3,\nabla}(r)
+
\bigl(1-\lambda_t\bigr)S_{3,\mathrm{late}}(r;t).
\end{aligned}
$$
Then the active coarse-shortlist size is
$$
N_{\mathrm{cheap}}(t)
=
\min\!\left\{
\left|\mathcal P_{\mathrm{avail}}^{(3)}\right|,
N_{\mathrm{cheap}}^{\max},
K_{\mathrm{coarse}}(t)
\right\},
$$
and the Phase-3 cheap universe is the full-pool cheap-score shortlist
$$
\mathcal C_{\mathrm{cheap}}
=
\operatorname{Top}_{N_{\mathrm{cheap}}(t)}\!\left(
\mathcal P_{\mathrm{avail}}^{(3)};
S_{3,\mathrm{cheap}}(\cdot;t)
\right).
$$
So the cumulative Phase-3 cheap screen is applied over the full admissible candidate-position pool, not behind a separate upstream $g_{\mathrm{lcb}}$-only cap. The gates $\Gamma_{\mathrm{stage}}$ and $\Gamma_{\mathrm{sym}}$ determine admissibility, while $S_{3,\mathrm{cheap}}$ ranks only within that admissible pool. The horizon $H_t$ smooths cheap-stage width and burden, but it is not itself a new hard gate and it is not a theorem-level invariant of the Hamiltonian.

The term $S_{3,\nabla}(r)$ is the direct continuation of the early drop-dominant surface $S_1$. It matters most when useful runway is still long. The late-stage factor $S_{3,\mathrm{late}}(r;t)$ matters more once that runway contracts.

From first principles, $K_{\mathrm{cheap}}(r;t)$ is a dimensionless burden functional: the numerator of $S_{3,\mathrm{cheap}}(r;t)$ estimates local utility, and $K_{\mathrm{cheap}}(r;t)$ discounts that utility by how expensive it is to admit $r$ now and, through $\bar R_{\mathrm{rem}}(r;t)$, how expensive that decision is expected to remain over the useful horizon of the current run. The $2\max(\mathrm{wt}(\ell)-1,0)$ piece is a two-qubit-style burden proxy, the $\frac12(2\,\#_{X,Y}(\ell)+1)$ piece is a basis-change / one-qubit rotation proxy, and $|p_{\mathrm{app}}-p|$ penalizes inserting far from the current append location.

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
where the first arrow is the horizon-smoothed cheap ranking over the full admissible pool, the middle arrow forms the internal shortlist, and the later arrows perform the richer reranking and Phase-3 augmentation built on top of the earlier surfaces. The horizon forecast enters here only as a controller-side burden/width modulator; it does not turn shortlist membership into a theorem of global optimality.

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
K_{\mathrm{full}}(r;t)
&=1+w_R\bar R_{\mathrm{rem}}(r;t)+w_D\bar D_{\mathrm{life}}(r)\\
&\quad +w_G\bar G_{\mathrm{new}}(r)+w_C\bar C_{\mathrm{new}}(r)+w_c\bar c(r).
\end{aligned}
$$
So the base rerank score closes to
$$
\begin{aligned}
S_{3,\mathrm{base}}(r;t)
&=\Gamma_{\mathrm{stage}}(m)\Gamma_{\mathrm{sym}}(r)\,\nu_r^{\gamma_N}\frac{\Delta E_{\mathrm{TR}}(r)}{K_{\mathrm{full}}(r;t)+\varepsilon}\\
&=\Gamma_{\mathrm{stage}}(m)\Gamma_{\mathrm{sym}}(r)
\left[\operatorname{clip}_{[0,1]}\!\left(1-\frac{(q_r^{\mathrm{red}})^\top(Q_r+\varepsilon_{\mathrm{nov}}I)^{-1}q_r^{\mathrm{red}}}{F_r^{\mathrm{red}}}\right)\right]^{\gamma_N}
\frac{\Delta E_{\mathrm{TR}}(r)}{K_{\mathrm{full}}(r;t)+\varepsilon}.
\end{aligned}
$$
The augmented selector score is therefore
$$
\begin{aligned}
S_{3,\mathrm{aug}}(r;t)
&=S_{3,\mathrm{base}}(r;t)+\beta_{\mathrm{split}}\Sigma(r)+\beta_{\mathrm{motif}}M(r)+\beta_{\mathrm{sym}}Y(r)-\beta_{\mathrm{dup}}D_{\mathrm{dup}}(r)\\
&=\Gamma_{\mathrm{stage}}(m)\Gamma_{\mathrm{sym}}(r)
\left[\operatorname{clip}_{[0,1]}\!\left(1-\frac{(q_r^{\mathrm{red}})^\top(Q_r+\varepsilon_{\mathrm{nov}}I)^{-1}q_r^{\mathrm{red}}}{F_r^{\mathrm{red}}}\right)\right]^{\gamma_N}
\frac{\Delta E_{\mathrm{TR}}(r)}{K_{\mathrm{full}}(r;t)+\varepsilon}\\
&\qquad+\beta_{\mathrm{split}}\Sigma(r)+\beta_{\mathrm{motif}}M(r)+\beta_{\mathrm{sym}}Y(r)-\beta_{\mathrm{dup}}D_{\mathrm{dup}}(r).
\end{aligned}
$$
Here $\Sigma$ measures the gain available from selective macro-splitting, $M$ measures motif compatibility or transferable local pattern alignment, $Y$ measures symmetry quality or mitigation value, and $D_{\mathrm{dup}}$ penalizes near-duplicate continuation directions.^[These four addends are intentionally higher-level selector covariates. They should be read as implementation-facing heuristic modifiers layered on top of the primitive geometric score $S_{3,\mathrm{base}}$, not as unique primitive invariants with a single repo-independent closed form.] If a shortlisted macro generator $m$ admits a split family
$$
\mathcal C_{\mathrm{split}}(m)=\{m_1,\dots,m_{K_m}\},
$$
then each split child inherits the same position $p$ and therefore defines a child record $r_j=(m_j,p)$, with replacement rule
$$
r\leadsto r_j^*\quad\text{if}\quad \max_j S_{3,\mathrm{aug}}(r_j) > S_{3,\mathrm{aug}}(r)+\tau_{\mathrm{split}}.
$$

$$
\mathcal S_{3}=\{r_{1},\dots,r_{N}\},
\qquad
s(r)=S_{3,\mathrm{aug}}(r)\ \text{or}\ S_{3,\mathrm{base}}(r),
$$
and let the shortlist be sorted so that
$$
s(r_{1})\ge s(r_{2})\ge \cdots \ge s(r_{N}).
$$
A batch selector does not choose a single winner; it chooses a set
$$
\mathcal B_{\mathrm{batch}}\subseteq \mathcal S_{3},
\qquad
1\le |\mathcal B_{\mathrm{batch}}|\le B_{\mathrm{cap}},
$$
by starting from the top record and then greedily adding near-degenerate compatible records.

$$
r\in\mathcal B_{\mathrm{batch}}
\quad\Longrightarrow\quad
s(r)\ge \eta_{\mathrm{nd}}\,s(r_{\max}),
$$
where $r_{\max}$ is the top shortlisted record and $0<\eta_{\mathrm{nd}}\le 1$ is the near-degeneracy ratio. In other words, only candidates whose score is sufficiently close to the best score are even allowed to compete for the same batch.

Then define a baseline pairwise incompatibility penalty
$$
\Pi_0(r,r')
=
w_{\mathrm{ov}}\,\Omega(r,r')
+
w_{\mathrm{comm}}\,\Xi(r,r')
+
w_{\mathrm{curv}}\,\mathcal K(r,r')
+
w_{\mathrm{sched}}\,\mathcal A(r,r')
+
w_{\mathrm{meas}}\,\mathcal M(r,r'),
$$
with
$$
\Omega(r,r')=\frac{|\operatorname{supp}(r)\cap \operatorname{supp}(r')|}{|\operatorname{supp}(r)\cup \operatorname{supp}(r')|},
\qquad
\Xi(r,r')=
\begin{cases}
0,&\text{if the candidate generators commute},\\
1,&\text{otherwise},
\end{cases}
$$
$$
\mathcal K(r,r')=\text{cross-curvature / tangent-overlap proxy},
\qquad
\mathcal A(r,r')=\frac{|W_r\cap W_{r'}|}{|W_r\cup W_{r'}|},
\qquad
\mathcal M(r,r')=1-\text{measurement-overlap}(r,r').
$$
^[Batch note: $\mathcal K$ and $\mathcal M$ are proxy penalties rather than primitive-closed observables. On paper they are best interpreted as controller-supplied overlap costs used to discourage mutually awkward co-admissions.]

The greedy admission rule is then
$$
r\ \text{is appended to}\ \mathcal B_{\mathrm{batch}}
\quad\Longrightarrow\quad
s(r)-\sum_{r'\in\mathcal B_{\mathrm{batch}}}\Pi_t(r,r')>0,
$$
together with
$$
|\mathcal B_{\mathrm{batch}}|<B_{\mathrm{target}}(t)
\quad\text{or at worst}\quad
|\mathcal B_{\mathrm{batch}}|<B_{\mathrm{cap}}.
$$
Here the batch target should itself depend on useful remaining runway. A canonical controller-level rule is
$$
B_{\mathrm{target}}(t)
=
B_{\min}
+
\left\lfloor
(B_{\max}-B_{\min})
\frac{\widehat N_{\mathrm{rem}}(t)}{D_{\mathrm{left}}(t)+\varepsilon}\,
C_t
\right\rfloor,
$$
where $B_{\min}$ and $B_{\max}$ are controller lower/upper batch-width bounds. So batch growth is largest only when useful runway is still long and the runway estimate is confident. The scheduler-scaled compatibility penalty is
$$
\Pi_t(r,r')=\gamma_{\mathrm{compat}}(t)\,\Pi_0(r,r'),
\qquad
\gamma_{\mathrm{compat}}(t)=\gamma_{\min}+(\gamma_{\max}-\gamma_{\min})W_{\mathrm{sat}}(t).
$$
Here $\gamma_{\min}$ and $\gamma_{\max}$ are controller lower/upper compatibility-pressure bounds.
Thus both the allowed batch width and the effective pairwise compatibility pressure depend on expected useful depth remaining; batch is deliberately more permissive when the selector still expects several useful admissions and tighter when saturation pressure is high.
So batch selection is a greedy set-building rule:
$$
\mathcal B_{\mathrm{batch},0}=\varnothing,
\qquad
\mathcal B_{\mathrm{batch},k+1}=
\mathcal B_{\mathrm{batch},k}\cup\{r\}
\ \text{only if }\
r\ \text{is near-degenerate and still has positive net value after penalties.}
$$

Qualitatively, batch means “several shortlisted candidates are so similarly good that we admit more than one at the same ADAPT depth, provided they do not clash too much.” It is not beam: beam keeps multiple alternative scaffolds alive, while batch commits multiple compatible admissions into one scaffold immediately. It is also not a joint global optimizer over subsets; it is a greedy compatibility-filtered multi-admission rule.

### 11.4.3 Post-admission local simplification: permissibility, expendability, and gain-retention safety

After Phase 3 admits its winning payload, the selector surface does not terminate immediately. Let $\mathcal A_d^\star$ denote the payload admitted at adaptive depth $d$: either a singleton $\{r^\star\}$ or a greedy batch $\mathcal B_{\mathrm{batch}}^\star$. Let
$$
(\mathcal O_d,\theta_d,E_d)
$$
be the scaffold state immediately before that admission, and let
$$
(\mathcal O_d^+,\theta_d^+,E_d^+)
$$
be the post-insertion state after the ordinary local reoptimization on the admitted scaffold has completed. The admitted-step gain is
$$
\Delta_{\mathrm{adm}}(\mathcal A_d^\star)=E_d-E_d^+.
$$

The local simplification pass is guarded first by a prune-permissibility gate
$$
\Gamma_{\mathrm{prune}}^{\mathrm{perm}}(\mathcal O_d^+,\mathcal A_d^\star)\in\{0,1\},
$$
which decides whether any post-admission removal attempt is allowed at all. A canonical controller-facing form is
$$
\Gamma_{\mathrm{prune}}^{\mathrm{perm}}(t)
=
\mathbf 1[\text{accepted admission at }t]\,
\mathbf 1[\text{rollback-safe}]\,
\mathbf 1\!\Bigl[
W_{\mathrm{sat}}(t)\ge \tau_{\mathrm{hi}}
\ \vee\
\bigl(W_{\mathrm{sat}}(t)\ge \tau_{\mathrm{med}}
\wedge
\Delta_{\mathrm{adm}}(\mathcal A_d^\star)\le c_{\mathrm{weak}}\Delta_{\mathrm{scr}}(t)\bigr)
\Bigr],
$$
with screening scale
$$
\Delta_{\mathrm{scr}}(t)=\max\!\Bigl(\tau_{\mathrm{use},t},\operatorname{EWMA}(\Delta_{\mathrm{adm}}^+),\Delta_{\mathrm{num}}\Bigr).
$$
Here $\tau_{\mathrm{hi}},\tau_{\mathrm{med}},c_{\mathrm{weak}},\tau_{\mathrm{use},t}$, and $\Delta_{\mathrm{num}}$ are controller thresholds or numerical safety scales.
So the permissibility gate depends on global saturation/runway state and local step quality, but it is not itself a ranking score.

If $\Gamma_{\mathrm{prune}}^{\mathrm{perm}}=1$, define the locally expendable set
$$
\mathcal J_{\mathrm{exp}}(\mathcal A_d^\star)\subseteq\{1,\dots,|\mathcal O_d^+|\},
$$
obtained from the mature eligible set
$$
\mathcal M_t
=
\Bigl\{
j:
a_j(t)\ge a_{\min},\ 
j\notin\mathcal P_{\mathrm{protect}}(t),\ 
c_j^{\mathrm{cool}}(t)=0
\Bigr\},
$$
after excluding coordinates introduced by the just-admitted payload. Thus prune permissibility and local expendability remain separate notions: the first asks whether simplification may be attempted, while the second asks which previously active coordinates are locally weakest once admission has already succeeded.

If a cheap local prescreen is desired, define
$$
\theta_{\mathrm{ref}}(t)=\max\!\Bigl(\theta_{\mathrm{abs}},c_\theta\,\operatorname{median}_{j\in\mathcal M_t}|\operatorname{wrap}(\theta_j(t))|\Bigr),
$$
$$
S_\theta(j,t)=\left[1+\left(\frac{|\operatorname{wrap}(\theta_j(t))|}{\theta_{\mathrm{ref}}(t)+\varepsilon_\theta}\right)^{p_\theta}\right]^{-1},
$$
and let
$$
\mathcal P_{\mathrm{probe}}(t)=
\begin{cases}
\operatorname{Top}_{K_p}\bigl(\mathcal M_t;S_\theta(\cdot,t)\bigr),&\text{if angle prescreen is enabled},\\
\mathcal M_t,&\text{otherwise}.
\end{cases}
$$
This prescreen is computational only. It is not the authoritative local prune score.

For each $j\in\mathcal P_{\mathrm{probe}}(t)$, let $E_{-j}^{\mathrm{frz}}$ denote the frozen-ablation energy obtained by removing coordinate $j$ from the post-admission scaffold while holding all remaining post-admission coordinates fixed at their values in $\theta_d^+$. The local frozen-ablation loss is then
$$
L_j^{\mathrm{frz}}(\mathcal A_d^\star)=E_{-j}^{\mathrm{frz}}-E_d^+.
$$
Coordinates with smaller $L_j^{\mathrm{frz}}$ are more expendable, so the canonical local prune ranking is
$$
X_j(t)
=
\left[
1+\frac{L_j^{\mathrm{frz}}(\mathcal A_d^\star)}{\kappa_\delta\Delta_{\mathrm{scr}}(t)+\varepsilon}
\right]^{-1},
$$
and the live prune candidate set is
$$
\mathcal C_{\mathrm{exp}}(t)=\{j\in\mathcal P_{\mathrm{probe}}(t):X_j(t)\ge \tau_{\mathrm{stale}}\}.
$$
A weak age/rank regularizer may then be written as
$$
B_j(t)=1+\lambda_{\mathrm{age}}\bigl(2\varrho_j(t)-1\bigr),
\qquad 0\le \lambda_{\mathrm{age}}\le 0.1,
$$
$$
R_j(t)=X_j(t)\,B_j(t),
\qquad
j_t^\star=\arg\max_{j\in\mathcal C_{\mathrm{exp}}(t)}R_j(t).
$$
Thus $X_j(t)$ is the authoritative local expendability score, while $B_j(t)$ is only a weak policy bias. The frozen-ablation loss is therefore a local, solver-relative expendability score; it is not a theorem that the corresponding coordinate is globally redundant in the full optimization landscape. The global saturation variable $W_{\mathrm{sat}}(t)$ belongs in $\Gamma_{\mathrm{prune}}^{\mathrm{perm}}(t)$, not in $X_j(t)$ or $R_j(t)$ themselves.

Before the actual remove-refit trial, take the exact rollback snapshot
$$
\Xi_t^{\mathrm{trial}}
=
\bigl(
\mathcal O_d^+,\theta_d^+,E_d^+,\mathcal M_{\mathrm{opt}}^+,\text{local prune metadata}
\bigr).
$$
Each ranked removal attempt is then tested by a local remove-refit trial on the reduced scaffold. If $E_t^{(-j_t^\star)}$ is the post-refit energy of the scaffold with $j_t^\star$ removed, then the retained admitted-step gain is
$$
\Delta_{\mathrm{keep}}^{(j_t^\star)}=E_d-E_t^{(-j_t^\star)}.
$$
The canonical Phase-3 prune accept rule requires both a numerical safety guard and gain retention:
$$
E_t^{(-j_t^\star)}\le E_d^+ + \Delta_{\mathrm{safe}},
\qquad
\Delta_{\mathrm{keep}}^{(j_t^\star)}
\ge
\eta_{\mathrm{ret}}\,\Delta_{\mathrm{adm}}(\mathcal A_d^\star).
$$
In repo-facing notation, with
$$
E_t^-:=E_d,\qquad E_t:=E_d^+,\qquad \delta_t:=\Delta_{\mathrm{adm}}(\mathcal A_d^\star),\qquad \beta_{\mathrm{keep}}:=\eta_{\mathrm{ret}},
$$
the same accept rule is
$$
E_t^{(-j_t^\star)}
\le
E_t^- - \beta_{\mathrm{keep}}\delta_t + \Delta_{\mathrm{safe}}.
$$
The first form shows explicit retained gain. The second is the concise repo-facing energy inequality. If a trial fails, the state is restored exactly to $\Xi_t^{\mathrm{trial}}$.

So the post-admission local simplification contract is
$$
\text{admit }\mathcal A_d^\star
\;\longrightarrow\;
\text{local reopt}
\;\longrightarrow\;
\text{snapshot }\Xi_t^{\mathrm{trial}}
\;\longrightarrow\;
\begin{cases}
\text{keep that state},&\Gamma_{\mathrm{prune}}^{\mathrm{perm}}=0,\\
\text{rank } \mathcal C_{\mathrm{exp}}(t) \text{ by }R_j(t),&\Gamma_{\mathrm{prune}}^{\mathrm{perm}}=1,
\end{cases}
$$
followed, when pruning is permitted, by sequential remove-refit trials with exact rollback on every rejected trial. A failed local prune attempt restores the last committed post-admission state; prune rejection never invalidates the original admission event itself. In particular, prune permissibility is not a local expendability score, frozen ablation is not a global redundancy theorem, and global progress or stopping logic remain controller-level surfaces rather than consequences of the local prune ranking alone.

### 11.4.4 Beam-adapt over scaffold alternatives: branch state, pruning, and effective selector

In this section beam-adapt is **purely structural**: each branch carries an alternative ADAPT scaffold together with its locally refit amplitudes. There is no time checkpoint, probe rollout, or projected-dynamics integral in this section. Those belong to the separate time-dynamics beam surface developed in §17.

Within each branch, the local cheap-screen / shortlist chain reuses §§11.4.1--11.4.2: the branch-local cheap universe is obtained by ranking the full branch-local admissible record set with $S_{3,\mathrm{cheap}}$, not by inserting a separate upstream gradient-only cap. The branch-level frontier-prune operators introduced below remain beam-management surfaces and are distinct from the post-admission scaffold-simplification prune contract of §11.4.3.

The cleanest way to read a live scaffold branch is
$$
\mathfrak b
=
(\text{identity/history};\ \text{current scaffold and amplitudes};\ \text{current energy/depth};\ \text{cumulative selector totals/status}).
$$
More explicitly, write
$$
\begin{aligned}
\mathfrak b
\!&=\bigl(\operatorname{id}_b,\pi_b,\mathcal H_b,\mathcal O_b,\theta_b,E_b,d_b,S_b^{\mathrm{cum}},K_b^{\mathrm{cum}},\sigma_b,\tau_b\bigr),\\
\mathcal H_b&=(r_{b,1},\dots,r_{b,d_b}),
\qquad
r_{b,j}=(m_{b,j}^{\mathrm{adm}},p_{b,j}^{\mathrm{adm}}),\\
\mathcal O_b&=(m_{b,1},\dots,m_{b,|\mathcal O_b|}),
\qquad
E_b=E(\theta_b;\mathcal O_b),\\
S_b^{\mathrm{cum}}&=\sum_{j=1}^{d_b}s_{b,j},
\qquad
K_b^{\mathrm{cum}}=\sum_{j=1}^{d_b}\kappa_{b,j}.
\end{aligned}
$$
Here $\operatorname{id}_b$ and $\pi_b$ are the branch and parent identifiers, $\mathcal H_b$ is the ordered history of admitted records, $\mathcal O_b$ is the scaffold currently carried by branch $b$, $\theta_b$ is the refit parameter vector on that scaffold, $E_b$ is the current variational energy, $d_b$ is the branch ADAPT depth, $s_{b,j}$ and $\kappa_{b,j}$ are the selector score and burden increment contributed by the $j$th admitted record in $\mathcal H_b$, and $(\sigma_b,\tau_b)$ record live/terminal status and termination label.

If beam search is entered from an incumbent scaffold $(\mathcal O^{(0)},\theta^{(0)})$, the root branch is
$$
\begin{aligned}
\mathfrak b_{\mathrm{root}}
=
\bigl(
\operatorname{id}_0,\varnothing,\mathcal H^{(0)},\mathcal O^{(0)},\theta^{(0)},E(\theta^{(0)};\mathcal O^{(0)}),d_0,S_{\mathrm{root}}^{\mathrm{cum}},K_{\mathrm{root}}^{\mathrm{cum}},\mathrm{frontier},\varnothing
\bigr),
\end{aligned}
$$
with
$$
d_0=|\mathcal H^{(0)}|,
\qquad
S_{\mathrm{root}}^{\mathrm{cum}}=\sum_{j=1}^{d_0}s_j^{(0)},
\qquad
K_{\mathrm{root}}^{\mathrm{cum}}=\sum_{j=1}^{d_0}\kappa_j^{(0)}.
$$
If the beam is started from a fresh scaffold, then $\mathcal H^{(0)}=\varnothing$, $d_0=0$, and both cumulative sums vanish.

One beam round is best read as four structural steps.

**(i) Recompute the branch-local cumulative Phase-3 screening chain.**  
Each live branch $\mathfrak b\in\mathfrak F_c$ induces its own raw candidate-position surface. Let
$$
\eta_b\in\{\mathrm{core/seed},\mathrm{residual}\}
$$
be the branch-local stage label, and let the branch-local enabled generator family be
$$
\mathcal G_{\eta_b}^{(3)}(\mathfrak b)
=
\begin{cases}
\mathcal G_{\mathrm{core/seed}}^{(3)}(\mathfrak b),&\eta_b=\mathrm{core/seed},\\
\mathcal G_{\mathrm{residual}}^{(3)}(\mathfrak b),&\eta_b=\mathrm{residual}.
\end{cases}
$$
Let $\mathcal P_{\mathrm{residual}}(\mathfrak b)\subseteq\mathcal G^{(3)}$ denote the residual-only generator family on branch $b$.^[Implementation note: this is a controller-defined subset of the Phase-3 pool rather than a primitive Hamiltonian object. In practice it is decided from the branch's current stage/state and can depend on runtime policy rather than only on algebraic data.] Then the controller-stage gate and the branch-local available Phase-3 generator family are
$$
\Gamma_{\mathrm{stage},b}(m)
=
\begin{cases}
1,&\eta_b=\mathrm{residual},\\
1,&\eta_b=\mathrm{core/seed}\ \text{and}\ m\notin\mathcal P_{\mathrm{residual}}(\mathfrak b),\\
0,&\eta_b=\mathrm{core/seed}\ \text{and}\ m\in\mathcal P_{\mathrm{residual}}(\mathfrak b),
\end{cases}
\qquad
\mathcal G_{\mathrm{avail}}^{(3)}(\mathfrak b)
=
\{\,m\in\mathcal G^{(3)}:\Gamma_{\mathrm{stage},b}(m)=1\,\}.
$$
In the branch-local beam surface, the optional transient `seed` stage is absorbed into the label $\mathrm{core/seed}$ and resolves immediately before persistent beam branching. A convenient closed stage-label transition rule consistent with the stage-controller surface of §11.6 is^[Reader note: $\eta_b$, $n_d^{\mathrm{small}}$, $d_{\min}$, $M$, $\tau_{\mathrm{drop}}$, and $\chi_b^{\mathrm{trough}}$ are best read as controller-state quantities. The manuscript gives a mathematically usable surface for them, but they are still runtime decision variables/signals rather than primitive observables derived directly from the Hamiltonian.] 
$$
\eta_{\mathrm{root}}\in\{\mathrm{core/seed},\mathrm{residual}\},
\qquad
\eta_{b'}=
\begin{cases}
\mathrm{residual},&\eta_b=\mathrm{residual},\\
\mathrm{residual},&\eta_b=\mathrm{core/seed},\ d_{b'}\ge d_{\min},\ n_{d_{b'}}^{\mathrm{small}}=M,\ \chi_{b'}^{\mathrm{trough}}=0,\\
\mathrm{core/seed},&\text{otherwise},
\end{cases}
$$
where $\chi_{b'}^{\mathrm{trough}}\in\{0,1\}$ is the branch-local trough indicator. Thus $\eta_{\mathrm{root}}$ is inherited from the incoming controller state, once a branch reaches residual stage it remains there, and the core/seed $\to$ residual transition occurs only after a branch-local plateau-patience hit without trough detection.
Now let
$$
n_b=|\mathcal O_b|,
\qquad
p_{\mathrm{app}}(\mathfrak b)=n_b,
\qquad
W_{\mathrm{act}}(\mathfrak b)\subseteq\{0,\dots,n_b-1\}
$$
be the current scaffold length, append slot, and active position-probe window on branch $b$. The ordered probe list is
$$
\mathcal L_{\mathrm{probe}}(\mathfrak b)
:=
\bigl(p_{\mathrm{app}}(\mathfrak b),\,0,\,W_{\mathrm{act}}(\mathfrak b)\bigr),
$$
where the entries of $W_{\mathrm{act}}(\mathfrak b)$ are read in their native order. The distinct retained probe positions are^[Algorithmic note: $\operatorname{Dedup}_{\mathrm{ord}}$ means stable order-preserving deduplication of the probe list, and $\operatorname{Head}_{M_{\mathrm{probe}}}$ means truncation to the first $M_{\mathrm{probe}}$ surviving entries. These are algorithmic selection operators rather than primitive algebraic maps.]
$$
\mathcal P_{\mathrm{probe}}(\mathfrak b)
=
\operatorname{Head}_{M_{\mathrm{probe}}}
\Bigl(
\operatorname{Dedup}_{\mathrm{ord}}\bigl(\mathcal L_{\mathrm{probe}}(\mathfrak b)\bigr)
\Bigr)
\subseteq
\{0,\dots,n_b\}.
$$
For each candidate-position record $r=(m,p)$, let the branch-local symmetry gate be
$$
\Gamma_{\mathrm{sym},b}(m,p)
=
\begin{cases}
0,&\text{the symmetry audit for }(m,p;\mathcal O_b)\text{ returns a hard violation},\\
1,&\text{otherwise}.
\end{cases}
$$
Hence the branch-local admissible insertion-position set for generator $m$ is
$$
\mathcal P_m(\mathfrak b)
=
\{\,p\in\mathcal P_{\mathrm{probe}}(\mathfrak b):\Gamma_{\mathrm{sym},b}(m,p)=1\,\},
$$
and the resulting raw candidate-position surface is
$$
\mathcal R_b^{\mathrm{raw}}
=
\{\,r=(m,p):m\in\mathcal G_{\mathrm{avail}}^{(3)}(\mathfrak b),\ p\in\mathcal P_m(\mathfrak b)\,\},
$$
so the generator stage gate and the position-level symmetry gate are already absorbed into $\mathcal R_b^{\mathrm{raw}}$. In the maximally wide limit,
$$
W_{\mathrm{act}}(\mathfrak b)=\{0,\dots,n_b-1\},
\qquad
M_{\mathrm{probe}}\ge n_b+1
\quad\Longrightarrow\quad
\mathcal P_{\mathrm{probe}}(\mathfrak b)=\{0,\dots,n_b\},
$$
and the branch can test every insertion position on its current scaffold. The local screening flow is then
$$
\begin{aligned}
N_{\mathrm{cheap}}(\mathfrak b)
&=
\min\!\left\{N_{\mathrm{cheap}}^{\max},|\mathcal R_b^{\mathrm{raw}}|,K_{\mathrm{coarse}}(\mathfrak b)\right\},\\
\mathcal C_{\mathrm{cheap}}(\mathfrak b)
&=
\operatorname{Top}_{N_{\mathrm{cheap}}(\mathfrak b)}\!\left(\mathcal R_b^{\mathrm{raw}};S_{3,\mathrm{cheap}}\right),\\
N_{\mathrm{short}}(\mathfrak b)
&=
\min\!\left\{|\mathcal C_{\mathrm{cheap}}(\mathfrak b)|,N_{\max},\left\lceil f_{\mathrm{short}}|\mathcal C_{\mathrm{cheap}}(\mathfrak b)|\right\rceil\right\},\\
\mathcal S_3(\mathfrak b)
&=
\operatorname{Top}_{N_{\mathrm{short}}(\mathfrak b)}\!\left(\mathcal C_{\mathrm{cheap}}(\mathfrak b);S_{3,\mathrm{cheap}}\right).
\end{aligned}
$$
where
$$
H_b=\min\!\bigl(D_{\mathrm{left}}(\mathfrak b),\widehat N_{\mathrm{rem}}(\mathfrak b)\bigr)
$$
is the branch-local useful horizon. The same controller layer may also induce
$$
K_{\mathrm{coarse}}(\mathfrak b)
=
K_{\min}+\left\lfloor (K_{\max}-K_{\min})\frac{\widehat N_{\mathrm{rem}}^{\mathrm{hi}}(\mathfrak b)}{D_{\mathrm{left}}(\mathfrak b)+\varepsilon}\right\rfloor,
$$
$$
B_{\mathrm{target}}(\mathfrak b)
=
B_{\min}+\left\lfloor (B_{\max}-B_{\min})\frac{\widehat N_{\mathrm{rem}}(\mathfrak b)}{D_{\mathrm{left}}(\mathfrak b)+\varepsilon}C(\mathfrak b)\right\rfloor,
$$
$$
\Pi_{\mathfrak b}(r,r')=\gamma_{\mathrm{compat}}(\mathfrak b)\,\Pi_0(r,r'),
\qquad
\gamma_{\mathrm{compat}}(\mathfrak b)=\gamma_{\min}+(\gamma_{\max}-\gamma_{\min})W_{\mathrm{sat}}(\mathfrak b),
$$
and mode/regime labels
$$
\pi(\mathfrak b)\in\{\mathrm{fast},\mathrm{mixed},\mathrm{high\mbox{-}fidelity}\},
\qquad
\mathrm{regime}_{\mathrm{sat}}(\mathfrak b)\in\{\mathrm{hungry},\mathrm{watch},\mathrm{plateau}\}.
$$
Using the same cumulative chain from §§11.4.1--11.4.2 — first the reused cheap screen, then the shortlist/full rerank, and finally the Phase-3 augmentations — this branch-local screen is now evaluated on the current branch state $(\mathcal O_b,\theta_b)$ with
$$
\Gamma_{\mathrm{stage}}(m)=\Gamma_{\mathrm{stage},b}(m),
\qquad
\Gamma_{\mathrm{sym}}(r)=\Gamma_{\mathrm{sym},b}(m,p)
\quad\text{for }r=(m,p).
$$
Also set
$$
S_{3,\mathrm{cheap}}(r;\mathfrak b):=s_{b,\mathrm{cheap}}(r),
\qquad
S_{3,\mathrm{base}}(r;\mathfrak b):=s_{b,\mathrm{base}}(r),
$$
so the branch-local shortlist operators are using the same score names as §§11.4.1--11.4.2.
Denote the resulting local selector by
$$
s_b(r)=S_{3,\mathrm{aug}}(r;\mathfrak b)
\quad\text{or}\quad
s_b(r)=S_{3,\mathrm{base}}(r;\mathfrak b).
$$
More explicitly, the branch-local score flow is
$$
\begin{aligned}
s_{b,\mathrm{cheap}}(r)
&=
\Gamma_{\mathrm{stage},b}(m)\Gamma_{\mathrm{sym},b}(m,p)
\frac{\max\!\left\{\left|2\Re\langle\psi_b|H|d_r^{(b)}\rangle\right|-z_\alpha\sigma_r(\mathfrak b),0\right\}^2}{2\lambda_F\langle d_r^{(b)}|Q_{\psi_b}|d_r^{(b)}\rangle\,(K_{\mathrm{cheap}}(r;\mathfrak b)+\varepsilon)},\\
|\psi_b\rangle
&=
U(\theta_b;\mathcal O_b)|\phi_0\rangle,
\qquad
Q_{\psi_b}=I-|\psi_b\rangle\langle\psi_b|,\\
|d_r^{(b)}\rangle
&=
-i\widetilde A_r^{(b)}|\psi_b\rangle,
\qquad
t_r^{(b)}=Q_{\psi_b}|d_r^{(b)}\rangle,
\qquad
F_{\mathrm{raw}}(r;\mathfrak b)=\langle d_r^{(b)}|Q_{\psi_b}|d_r^{(b)}\rangle,\\
t_j^{(b)}
&=
Q_{\psi_b}\partial_{\theta_j}|\psi_b\rangle,
\qquad
(Q_r(\mathfrak b))_{jk}=\Re\langle t_j^{(b)},t_k^{(b)}\rangle,
\qquad
(q_r(\mathfrak b))_j=\Re\langle t_j^{(b)},t_r^{(b)}\rangle,\\
\widetilde h_r(\mathfrak b)
&=
h_r(\mathfrak b)-b_r(\mathfrak b)^\top M_r(\mathfrak b)^{-1}b_r(\mathfrak b),\\
F_r^{\mathrm{red}}(\mathfrak b)
&=
F_{\mathrm{raw}}(r;\mathfrak b)
-2q_r(\mathfrak b)^\top M_r(\mathfrak b)^{-1}b_r(\mathfrak b)
+b_r(\mathfrak b)^\top M_r(\mathfrak b)^{-1}Q_r(\mathfrak b)M_r(\mathfrak b)^{-1}b_r(\mathfrak b),\\
q_r^{\mathrm{red}}(\mathfrak b)
&=
q_r(\mathfrak b)-Q_r(\mathfrak b)M_r(\mathfrak b)^{-1}b_r(\mathfrak b),\\
\nu_r(\mathfrak b)
&=
\operatorname{clip}_{[0,1]}\!\left(
1-\frac{(q_r^{\mathrm{red}}(\mathfrak b))^\top(Q_r(\mathfrak b)+\varepsilon_{\mathrm{nov}}I)^{-1}q_r^{\mathrm{red}}(\mathfrak b)}{F_r^{\mathrm{red}}(\mathfrak b)}
\right),\\
\Delta E_{\mathrm{TR}}(r;\mathfrak b)
&=
\max_{|\alpha|\le \rho/\sqrt{F_r^{\mathrm{red}}(\mathfrak b)}}
\left[
g_{\mathrm{lcb}}(r;\mathfrak b)|\alpha|-\frac12\max(\widetilde h_r(\mathfrak b),0)\alpha^2
\right],\\
s_{b,\mathrm{base}}(r)
&=
\Gamma_{\mathrm{stage},b}(m)\Gamma_{\mathrm{sym},b}(m,p)
\left[
\operatorname{clip}_{[0,1]}\!\left(
1-\frac{(q_r^{\mathrm{red}}(\mathfrak b))^\top(Q_r(\mathfrak b)+\varepsilon_{\mathrm{nov}}I)^{-1}q_r^{\mathrm{red}}(\mathfrak b)}{F_r^{\mathrm{red}}(\mathfrak b)}
\right)
\right]^{\gamma_N}\\
&\qquad\times
\frac{
\max_{|\alpha|\le \rho/\sqrt{F_r^{\mathrm{red}}(\mathfrak b)}}
\left[
\max\!\left\{\left|2\Re\langle\psi_b|H|d_r^{(b)}\rangle\right|-z_\alpha\sigma_r(\mathfrak b),0\right\}|\alpha|
-\frac12\max(\widetilde h_r(\mathfrak b),0)\alpha^2
\right]
}{
K_{\mathrm{full}}(r;\mathfrak b)+\varepsilon
},\\
s_b(r)
&=
s_{b,\mathrm{base}}(r)
+\beta_{\mathrm{split}}\Sigma(r;\mathfrak b)
+\beta_{\mathrm{motif}}M(r;\mathfrak b)
+\beta_{\mathrm{sym}}Y(r;\mathfrak b)
-\beta_{\mathrm{dup}}D_{\mathrm{dup}}(r;\mathfrak b).
\end{aligned}
$$
So the branch-local screening/rerank chain closes as
$$
\mathcal R_b^{\mathrm{raw}}
\xrightarrow{\ s_{b,\mathrm{cheap}}\ }
\mathcal C_{\mathrm{cheap}}(\mathfrak b)
\xrightarrow{\ s_{b,\mathrm{cheap}}\ }
\mathcal S_3(\mathfrak b)
\xrightarrow{\ s_b\ }
\mathcal A_b^{\mathrm{raw}}
\xrightarrow{\operatorname{Top}_{K_b}}
\mathcal A_b.
$$
For $r=(m,p)\in\mathcal S_3(\mathfrak b)$, let the runtime-split child family be
$$
\mathcal C_{\mathrm{split}}(m)=\{m_1,\dots,m_{K_m}\}.
$$
For a child subset $S\subseteq\mathcal C_{\mathrm{split}}(m)$, define the combined child-set polynomial
$$
m_S:=\sum_{u\in S}u.
$$
For any real-coefficient Pauli polynomial
$$
m=\sum_{w}c_wP_w
$$
written in the internal $e/x/y/z$ word alphabet, define the canonical polynomial signature
$$
\operatorname{Sig}_{\mathrm{poly}}(m;\tau_{\mathrm{sig}})
:=
\operatorname{sort}
\left\{
\bigl(w,\operatorname{rnd}_{12}(c_w)\bigr):
|c_w|>\tau_{\mathrm{sig}}
\right\},
\qquad
\operatorname{rnd}_{12}(x):=10^{-12}\operatorname{round}(10^{12}x).
$$
Thus $\operatorname{Dedup}_{\mathrm{sig}}$ retains one representative from each common value of $\operatorname{Sig}_{\mathrm{poly}}(\cdot;\tau_{\mathrm{sig}})$ after collecting like terms. Each singleton child is re-audited at the same insertion position $p$, and the symmetry-safe combined child-set family is
$$
\mathcal A_{\mathrm{split}}(m,p;\mathfrak b)
=
\operatorname{Dedup}_{\mathrm{sig}}
\Bigl[
\{\,S\subseteq\mathcal C_{\mathrm{split}}(m):\Gamma_{\mathrm{sym},b}(m_S,p)=1,\ m_S\not\equiv m\,\}
\Bigr].
$$
Write
$$
r_j=(m_j,p),
\qquad
r_S=(m_S,p)\quad (|S|\ge 2),
$$
for the singleton and child-set representatives, where $r_S$ denotes the combined child-set polynomial placed at the same position $p$. Then the full split-aware promotion universe of the parent record is
$$
\mathcal U_b(r)
=
\{r\}
\cup
\{\,r_j:\Gamma_{\mathrm{sym},b}(m_j,p)=1\,\}
\cup
\{\,r_S:S\in\mathcal A_{\mathrm{split}}(m,p;\mathfrak b),\ |S|\ge 2\,\}.
$$
Unsafe child atoms and unsafe child sets are excluded from $\mathcal U_b(r)$ by the symmetry gate and therefore may remain in provenance only, not in the admissible promotion surface. Now define
$$
r_b^{\star}(r)
=
\arg\max_{u\in\mathcal U_b(r)} s_b(u),
$$
and then the split-aware promoted representative is
$$
\widehat r_b(r)
=
\begin{cases}
r_b^{\star}(r),&\text{if }s_b\!\bigl(r_b^{\star}(r)\bigr)>s_b(r)+\tau_{\mathrm{split}},\\
r,&\text{otherwise}.
\end{cases}
$$
The actual raw admission set passed to the beam is therefore
$$
\mathcal A_b^{\mathrm{raw}}
=
\{\,\widehat r_b(r):r\in\mathcal S_3(\mathfrak b),\ s_b(\widehat r_b(r))>0\,\},
$$
and the retained candidate-admission set is
$$
\mathcal A_b
=
\operatorname{Top}_{K_b}\!\left(\mathcal A_b^{\mathrm{raw}};s_b\right),
\qquad
K_b\le B_{\mathrm{child}}-1.
$$
^[Beam-width note: $\operatorname{Top}_{K_b}$ is the score-ordered child selector on branch $b$. The inequality fixes only an admissible cap; the exact runtime choice of $K_b$ remains a controller policy decision unless specified separately.]

**(ii) Materialize either stop or one new admission.**  
The proposal family is
$$
\mathcal Q(b)=\{\mathrm{stop}\}\cup\mathcal A_b.
$$
The symbol $\mathrm{stop}$ means “terminate this scaffold as-is,” while each $r=(m,p)\in\mathcal A_b$ means “insert one new generator into this branch and refit.” If
$$
\mathcal O_b=(m_{b,1},\dots,m_{b,n_b}),
\qquad
0\le p\le n_b,
$$
then the scaffold insertion map is
$$
\operatorname{Insert}(\mathcal O_b,m,p)
=
(m_{b,1},\dots,m_{b,p},m,m_{b,p+1},\dots,m_{b,n_b}),
$$
and the insertion index transport is
$$
I_p(j)
=
\begin{cases}
j,&1\le j\le p,\\
j+1,&p<j\le n_b.
\end{cases}
$$
The zero-lifted parameter vector is therefore
$$
\theta_b^{\uparrow(r)}
=
(\theta_{b,1},\dots,\theta_{b,p},0,\theta_{b,p+1},\dots,\theta_{b,n_b}).
$$
Equivalently,
$$
(\theta_b^{\uparrow(r)})_{I_p(j)}=\theta_{b,j},
\qquad
(\theta_b^{\uparrow(r)})_{p+1}=0.
$$
For newest-window cap $w_{\mathrm{new}}$ and amplitude-carry cap $k_{|\theta|}$, let
$$
w_{\mathrm{eff}}^{(p)}=\min\{w_{\mathrm{new}},n_b+1\},
\qquad
W_{\mathrm{new}}^{(p)}=\{n_b+2-w_{\mathrm{eff}}^{(p)},\dots,n_b+1\},
$$
$$
W_{|\theta|}^{(p)}
=
\operatorname{Top}_{k_{|\theta|}}
\Bigl(
\{1,\dots,n_b+1\}\backslash W_{\mathrm{new}}^{(p)};
\bigl|(\theta_b^{\uparrow(r)})_j\bigr|
\Bigr).
$$
Hence the inherited refit window on the inserted scaffold is
$$
W_r=W(m,p)=W_{\mathrm{new}}^{(p)}\cup W_{|\theta|}^{(p)}.
$$
Then the branch-local refit map is
$$
\operatorname{Refit}_{W_r}(\theta_b;r)
:=
\arg\min_{\vartheta}
\left\{
E(\vartheta;\operatorname{Insert}(\mathcal O_b,m,p))
:\ 
\vartheta_j=(\theta_b^{\uparrow(r)})_j\ \forall j\notin W_r
\right\}.
$$
The transfer map is
$$
\mathcal T(\mathfrak b,\mathrm{stop})=\mathfrak b'
\quad\text{with}\quad
\mathcal H_{b'}=\mathcal H_b,\ 
\mathcal O_{b'}=\mathcal O_b,\ 
\theta_{b'}=\theta_b,\ 
\ E_{b'}=E_b,\ 
\ d_{b'}=d_b,\ 
\ S_{b'}^{\mathrm{cum}}=S_b^{\mathrm{cum}},\ 
\ K_{b'}^{\mathrm{cum}}=K_b^{\mathrm{cum}},\ 
\sigma_{b'}=\mathrm{terminal},
\qquad
\tau_{b'}=
\begin{cases}
\mathrm{empty},&\mathcal A_b=\varnothing,\\
\mathrm{stop},&\mathcal A_b\neq\varnothing,
\end{cases}
$$
and for $r=(m,p)$,
$$
\mathcal T(\mathfrak b,r)=\mathfrak b'
\quad\text{with}\quad
\mathcal O_{b'}=\operatorname{Insert}(\mathcal O_b,m,p),
\qquad
\theta_{b'}=\operatorname{Refit}_{W_r}(\theta_b;r),
$$
$$
\mathcal H_{b'}=(\mathcal H_b,r),
\qquad
E_{b'}=E(\theta_{b'};\mathcal O_{b'}),
\qquad
d_{b'}=d_b+1,
\qquad
S_{b'}^{\mathrm{cum}}=S_b^{\mathrm{cum}}+s_b(r),
\qquad
K_{b'}^{\mathrm{cum}}=K_b^{\mathrm{cum}}+K_{\mathrm{full}}(r),
\qquad
\sigma_{b'}=
\begin{cases}
\mathrm{terminal},&d_{b'}\ge d_{\max}\ \text{or}\ \mathcal A_{b'}=\varnothing,\\
\mathrm{frontier},&d_{b'}<d_{\max}\ \text{and}\ \mathcal A_{b'}\neq\varnothing,
\end{cases}
\qquad
\tau_{b'}=
\begin{cases}
\mathrm{depth\_cap},&d_{b'}\ge d_{\max},\\
\mathrm{empty},&d_{b'}<d_{\max}\ \text{and}\ \mathcal A_{b'}=\varnothing,\\
\varnothing,&d_{b'}<d_{\max}\ \text{and}\ \mathcal A_{b'}\neq\varnothing,
\end{cases}
$$
where $\mathcal A_{b'}$ is recomputed from step (i) on the child branch itself. So a non-stop child is exactly one more ADAPT admission applied to the parent scaffold, followed by the inherited-window refit already defined above. If $\mathcal A_b=\varnothing$, then $\mathcal Q(b)=\{\mathrm{stop}\}$ and the branch contributes only a terminal child.

**(iii) Classify, deduplicate, and prune the scaffold children.**  
The frontier and terminal descendants of branch $b$ are
$$
\mathfrak C_b^{\mathrm{front}}
=
\{\mathfrak b' : \mathfrak b'=\mathcal T(\mathfrak b,r),\ r\in\mathcal A_b,\ \sigma_{b'}=\mathrm{frontier}\},
$$
$$
\mathfrak C_b^{\mathrm{term}}
=
\{\mathfrak b' : \mathfrak b'=\mathcal T(\mathfrak b,q),\ q\in\mathcal Q(b),\ \sigma_{b'}=\mathrm{terminal}\}.
$$
Two branches are treated as duplicates when they represent the same scaffold and essentially the same refit point, for example under the fingerprint
$$
\operatorname{rnd}_{10}(x):=10^{-10}\operatorname{round}(10^{10}x),
\qquad
\operatorname{round}_{10}(\theta_b)
:=
\bigl(\operatorname{rnd}_{10}(\theta_{b,1}),\dots,\operatorname{rnd}_{10}(\theta_{b,n_b})\bigr),
$$
$$
\mathrm{fp}(\mathfrak b)
=
\bigl(d_b,\operatorname{labels}(\mathcal O_b),\operatorname{round}_{10}(\theta_b)\bigr).
$$
Among near-equivalent branches, keep the one with best lexicographic prune key
$$
\kappa_{\mathrm{prune}}(\mathfrak b)
=
\bigl(E_b,-S_b^{\mathrm{cum}},K_b^{\mathrm{cum}},|\mathcal O_b|,\operatorname{labels}(\mathcal O_b),\operatorname{round}_{10}(\theta_b),\operatorname{id}_b\bigr).
$$
This means that, within the symbolic scaffold beam, lower current energy is primary; stronger cumulative admitted selector value is secondary; lower accumulated burden and then smaller/simpler equivalent scaffolds break the remaining ties.

The beam operators are therefore
$$
\operatorname{Dedup}(\mathcal X)
=
\{\text{best branch per fingerprint under }\kappa_{\mathrm{prune}}\},
\qquad
\operatorname{Prune}(\mathcal X;B)
=
\text{lowest-}B\text{ branches in }\operatorname{Dedup}(\mathcal X)\text{ by }\kappa_{\mathrm{prune}}.
$$
Hence the live and terminal pools update by
$$
\mathfrak F_{c+1}
=
\operatorname{Prune}\!\left(\bigcup_{b\in\mathfrak F_c}\mathfrak C_b^{\mathrm{front}};B_{\mathrm{live}}\right),
\qquad
\mathfrak T_{c+1}
=
\operatorname{Prune}\!\left(\mathfrak T_c\cup\bigcup_{b\in\mathfrak F_c}\mathfrak C_b^{\mathrm{term}};B_{\mathrm{term}}\right).
$$

**(iv) Choose the final scaffold branch.**  
After $C$ scaffold-beam rounds, the surviving candidate set is
$$
\mathfrak W_{\mathrm{final}}=\mathfrak F_C\cup\mathfrak T_C,
$$
and the effective beam selector is
$$
\mathfrak b_*
=
\arg\min_{\mathfrak b\in\mathfrak W_{\mathrm{final}}}\kappa_{\mathrm{prune}}(\mathfrak b).
$$
Equivalently,
$$
\mathfrak b_*
=
\arg\min_{\mathfrak b\in\mathfrak F_C\cup\mathfrak T_C}
\bigl(E_b,-S_b^{\mathrm{cum}},K_b^{\mathrm{cum}},|\mathcal O_b|,\operatorname{labels}(\mathcal O_b),\operatorname{round}_{10}(\theta_b),\operatorname{id}_b\bigr).
$$

A compact scaffold-beam algorithm block is therefore
$$
\begin{aligned}
\text{A: }&
\mathfrak F_0=\{\mathfrak b_{\mathrm{root}}\},
\qquad
\mathfrak T_0=\varnothing,\\
\text{B: }&
\mathcal R_b^{\mathrm{raw}}
\to
\mathcal C_{\mathrm{cheap}}(\mathfrak b)
\to
\mathcal S_3(\mathfrak b)
\to
\mathcal A_b
\qquad(\mathfrak b\in\mathfrak F_c),\\
\text{C: }&
\mathcal Q(b)=\{\mathrm{stop}\}\cup\mathcal A_b,
\qquad
\mathfrak C_b=\{\mathcal T(\mathfrak b,q):q\in\mathcal Q(b)\},\\
\text{D: }&
\mathfrak C_b^{\mathrm{front}}
=
\{\mathfrak b'\in\mathfrak C_b:\sigma_{b'}=\mathrm{frontier}\},
\qquad
\mathfrak C_b^{\mathrm{term}}
=
\{\mathfrak b'\in\mathfrak C_b:\sigma_{b'}=\mathrm{terminal}\},\\
\text{E: }&
\mathfrak F_{c+1}
=
\operatorname{Prune}\!\left(\bigcup_{b\in\mathfrak F_c}\mathfrak C_b^{\mathrm{front}};B_{\mathrm{live}}\right),
\qquad
\mathfrak T_{c+1}
=
\operatorname{Prune}\!\left(\mathfrak T_c\cup\bigcup_{b\in\mathfrak F_c}\mathfrak C_b^{\mathrm{term}};B_{\mathrm{term}}\right),\\
\text{F: }&
\mathfrak W_{\mathrm{final}}=\mathfrak F_C\cup\mathfrak T_C,
\qquad
\mathfrak b_*=\arg\min_{\mathfrak b\in\mathfrak W_{\mathrm{final}}}\kappa_{\mathrm{prune}}(\mathfrak b).
\end{aligned}
$$

In words: each round keeps several scaffold alternatives alive, recomputes the local cumulative Phase-3 screening chain on each surviving scaffold, branches by admitting one additional generator or stopping, deduplicates nearly equivalent descendants, prunes back to the beam budgets, and finally returns the best surviving scaffold under the lexicographic selector key.

The separate time-dynamics / projected-dynamics beam surface, with checkpoint advances, probe rollouts, and integrated residual objectives, is developed later in §17.

## 11.5 Active HH pool surface and selected scaffold

The active HH adaptive surface is carried by the nested pool family
$$
\Omega_{\mathrm{HH}}^{(3)}\subseteq\Omega_{\mathrm{HH}}^{(2)}\subseteq\Omega_{\mathrm{HH}}^{(1)},
\qquad
\mathcal G_{\mathrm{adapt}}^{(k)}=\{\tau_m\}_{m\in\Omega_{\mathrm{HH}}^{(k)}}.
$$
The adaptive output is the selected scaffold
$$
\mathcal O_*=(\tau_{m_1},\dots,\tau_{m_d}),
\qquad
\tau_{m_j}\in\mathcal G_{\mathrm{adapt}}^{(3)},
$$
together with its fitted amplitude vector and resulting state,
$$
(\mathcal O_*,\theta_*^{\mathrm{adapt}},|\psi_*\rangle),
\qquad
|\psi_*\rangle=U(\theta_*^{\mathrm{adapt}};\mathcal O_*)|\phi_0\rangle.
$$
Thus the active HH scaffold manifold selected by ADAPT may be written as
$$
\mathcal M_{\mathrm{scaf}}(\mathcal O_*)
=
\{\,U(\vartheta;\mathcal O_*)|\phi_0\rangle:\vartheta\in\mathbb R^{|\mathcal O_*|}\,\}.
$$
The obsolete historical warm-start $\to$ refine $\to$ ADAPT $\to$ replay chain is no longer part of the mainline symbolic flow and is recorded only in Appendix A.2.

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

## Former Math.md §17

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

7. **Position-aware current branch growth.** The clean story allows arbitrary insertion positions and multi-parameter blocks. The helper layer now supports position-aware insertion and position-centered post-insert refit windows,
   $$
   \mathcal T(\mathfrak b,m,p)=\mathfrak b',
   \qquad
   \mathcal O_{b'}=\mathcal O_b\oplus_p m,
   \qquad
   \theta_{b'}=\theta_b\oplus_p 0,
   $$
   and the active probe set is built from the append slot, the left boundary, and the active reoptimization window. Hence, when the active window is made maximally large and the probe-position cap is also made large enough, the implementation can evaluate the full insertion set $p\in\{0,\dots,|\theta|\}$ rather than append only. The live surface still remains zero-initialized **single-parameter** growth, but not append-only growth.

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
