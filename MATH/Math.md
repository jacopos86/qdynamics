---
title: "Hubbard-Holstein Mathematical Implementation (Current Linear Substitution-First Form)"
author: "Jake Skyler Strobel (repo-grounded revision)"
date: "March 9, 2026"
geometry: margin=0.8in
fontsize: 10pt
---

# 1. Parameter Manifest and Reader Contract

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
  - ADAPT pools including `hva`, `full_meta`, `paop_*`, `uccsd_paop_lf_full`, `full_hamiltonian`.
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

The exact HH target used by the hardcoded VQE surfaces is a sector-filtered exact energy, not an unrestricted full-Hilbert minimum. This is what `exact_ground_energy_sector_hh(...)` and the ED basis logic provide.

Implemented surfaces:

- `src/quantum/vqe_latex_python_pairs.py`
  - `basis_state`
  - `apply_pauli_string`
  - `expval_pauli_polynomial`
  - `expval_pauli_polynomial_one_apply`
  - `apply_pauli_rotation`
  - `apply_exp_pauli_polynomial`
  - `exact_ground_energy_sector_hh`

# 9. Current Hardcoded HH Ansatz Families

## 9.1 HH layerwise ansatz: `hh_hva`

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

## 9.2 HH Pauli-termwise ansatz: `hh_hva_tw`

The Pauli-termwise ansatz removes that sharing. Every single split Pauli term gets its own parameter:
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

## 9.3 HH physical-termwise ansatz: `hh_hva_ptw`

The physical-termwise HH ansatz keeps one parameter per physical generator before Pauli splitting.

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

This ansatz is sector-preserving in fermion space because each physical generator preserves fermion number before Pauli splitting.

## 9.4 Reference-state dispatch in the hardcoded pipeline

For `problem="hh"`, `pipelines/hardcoded/hubbard_pipeline.py` constructs
$$
|\psi_{\mathrm{ref}}\rangle=|\mathrm{vac}_{\mathrm{ph}}\rangle\otimes|\Phi_{\mathrm{HF}}\rangle
$$
with `hubbard_holstein_reference_state(...)` and then dispatches

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

### 10.1.2 Phonon momentum-like primitive

The PAOP `P_i` primitive is
$$
P_i=i(\hat b_i^{\dagger}-\hat b_i).
$$

### 10.1.3 Doublon primitive

The local doublon primitive is the explicit operator from Section 3.3:
$$
\hat d_i=\frac{1}{4}\Bigl(I-Z_{p_{i\uparrow}}-Z_{p_{i\downarrow}}+Z_{p_{i\uparrow}}Z_{p_{i\downarrow}}\Bigr).
$$

### 10.1.4 Even hopping channel

The even hopping channel is
$$
K_{ij}=\sum_{\sigma}\left(
\hat c_{i\sigma}^{\dagger}\hat c_{j\sigma}+
\hat c_{j\sigma}^{\dagger}\hat c_{i\sigma}
\right).
$$

### 10.1.5 Odd current channel

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

### 10.2.6 Extended cloud channels

For cloud radius `R`, the implemented extended cloud channels are
$$
\mathcal O_{\mathrm{cloud\_p},i\to j}=\tilde n_i P_j,
\qquad
\mathcal O_{\mathrm{cloud\_x},i\to j}=\tilde n_i x_j,
$$
with the distance gate `\operatorname{dist}(i,j)\le R`.

### 10.2.7 Doublon-translation channels

The implemented doublon-translation channels are
$$
\mathcal O_{\mathrm{dbl\_p},i\to j}=\hat d_i P_j,
\qquad
\mathcal O_{\mathrm{dbl\_x},i\to j}=\hat d_i x_j.
$$

## 10.3 Pool-family map

The current pool-family map is:

- `paop` = `paop_std`
- `paop_lf` = `paop_lf_std`
- `paop_min` = `disp`
- `paop_std` = `disp + hopdrag`
- `paop_full` = `disp + doublon + hopdrag + cloud_p + cloud_x`
- `paop_lf_std` = `disp + hopdrag + curdrag`
- `paop_lf2_std` = `disp + hopdrag + curdrag + hop2`
- `paop_lf_full` = `disp + hopdrag + curdrag + hop2 + cloud_p + cloud_x + dbl_p + dbl_x`

The implementation then optionally applies

- pruning,
- normalization (`none`, `fro`, `maxcoeff`),
- split-into-single-Pauli children,
- signature deduplication.

## 10.4 ADAPT selection signal

The ADAPT signal remains
$$
g_m^{(n)}=i\langle\psi^{(n)}|[\hat H,A_m]|\psi^{(n)}\rangle.
$$

This document keeps that primitive, but the implemented HH stack now adds pool construction, staged continuation, shortlist scoring, and handoff provenance on top of it.

## 10.5 Current HH pool composition rules

The active HH ADAPT surface in `pipelines/hardcoded/adapt_pipeline.py` supports

- `hva`,
- `full_meta`,
- `uccsd_paop_lf_full`,
- `paop`, `paop_min`, `paop_std`, `paop_full`,
- `paop_lf`, `paop_lf_std`, `paop_lf2_std`, `paop_lf_full`,
- `full_hamiltonian`.

For staged HH continuation modes `phase1_v1`, `phase2_v1`, and `phase3_v1`, the code enforces

- no `full_meta` at depth `0`,
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
\mathcal P_{\mathrm{residual}}=\mathcal P_{\mathrm{full\_meta}}\setminus \mathcal P_{\mathrm{core}}.
$$

## 10.6 Implemented continuation scoring

The older continuation discussion is replaced here by the actual implemented score surfaces.

### 10.6.1 `simple_v1`

If the stage and leakage gates are open, the simple score is
$$
S_{\mathrm{simple}}
=
|g|
+\lambda_F F
-\lambda_{\mathrm{compile}} C_{\mathrm{proxy}}
-\lambda_{\mathrm{measure}}(G_{\mathrm{new}}+S_{\mathrm{new}}+R_{\mathrm{reuse}})
-\lambda_{\mathrm{leak}}L.
$$

Here

- `F` is the metric proxy,
- `C_proxy` is the compiled-position cost proxy,
- `G_new` is new measurement groups,
- `S_new` is new shots,
- `R_reuse` is grouped-reuse count cost,
- `L` is the leakage penalty.

### 10.6.2 Trust-region drop proxy

The full score first builds the lower-confidence gradient
$$
g_{\mathrm{lcb}}=\max\{ |g|-z_{\alpha}\sigma,\,0\}.
$$

Then it constructs the trust-region drop proxy
$$
\Delta E_{\mathrm{TR}}=
\begin{cases}
0, & g_{\mathrm{lcb}}\le 0 \text{ or } F\le 0,\\
\frac{1}{2}\frac{g_{\mathrm{lcb}}^2}{h_{\mathrm{eff}}}, & h_{\mathrm{eff}}>0 \text{ and } \frac{g_{\mathrm{lcb}}}{h_{\mathrm{eff}}}\le \frac{\rho}{\sqrt F},\\
g_{\mathrm{lcb}}\alpha_{\max}-\frac{1}{2}h_{\mathrm{eff}}\alpha_{\max}^2,
& \alpha_{\max}=\frac{\rho}{\sqrt F}, \text{ otherwise.}
\end{cases}
$$

### 10.6.3 `full_v2`

The implemented full score is
$$
S_{\mathrm{full}}
=
\exp(-\eta_L L)
\,\mathrm{novelty}^{\gamma_N}
\,\frac{\Delta E_{\mathrm{TR}}}{K}
+w_{\mathrm{motif}}\,B_{\mathrm{motif}},
$$
with
$$
K=
1
+w_D\frac{D}{D_{\mathrm{ref}}}
+w_G\frac{G_{\mathrm{new}}}{G_{\mathrm{ref}}}
+w_C\frac{S_{\mathrm{new}}}{S_{\mathrm{ref}}}
+w_P\frac{P_{\mathrm{opt}}}{P_{\mathrm{ref}}}
+w_c\frac{R_{\mathrm{reuse}}}{R_{\mathrm{ref}}}
+w_{\mathrm{life}}\,K_{\mathrm{life}}.
$$

This is the implemented “useful predicted drop divided by burden” surface, not a future plan.

## 10.7 Stage controller

The implemented stage controller in `hh_continuation_stage_control.py` uses the stage chain
$$
\texttt{seed} \rightarrow \texttt{core} \rightarrow \texttt{residual}.
$$

The transition rule is explicit:

- `seed -> core` after the seed step completes,
- `core -> residual` when the drop plateau patience is hit **and** no trough is detected,
- `residual` remains `residual` while open.

Position probing is enabled when one of the implemented triggers fires:

- drop plateau,
- `eps_grad` + finite-angle flatness,
- repeated-family flatness.

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
- `motif_library`,
- `motif_usage`,
- `symmetry_mitigation`,
- `rescue_history`,
- `replay_contract_hint`.

A schematic shape is

```yaml
continuation:
  mode: phase1_v1 | phase2_v1 | phase3_v1 | legacy
  scaffold: {...}
  optimizer_memory: {...}
  selected_generator_metadata: [...]
  generator_split_events: [...]
  motif_library: {...}
  motif_usage: {...}
  symmetry_mitigation: {...}
  rescue_history: [...]
  replay_contract_hint: {...}
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

## 13.4 Symmetry note

The raw staged ADAPT CLI already exposes phase-3 symmetry mode names, but on that raw path they are still primarily metadata/telemetry hooks. This manuscript therefore states the continuation/symmetry payload as an implemented data contract without overstating raw staged-ADAPT enforcement beyond what the code currently does.

# 14. Final Primitive-Closed Summary

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
11. serialize the resulting continuation and state handoff with the implemented payload contract.

This is exactly the manuscript shape the current repository supports: linear, substitution-first, and implementation-backed, without a prospective future layer.
