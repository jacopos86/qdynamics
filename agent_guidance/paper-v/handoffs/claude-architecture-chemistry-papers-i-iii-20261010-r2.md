# Claude architecture reply, revision 2: molecular electronic structure into Papers I–III

Date: 2026-10-10. Revision 2 of
`agent_guidance/paper-v/handoffs/claude-architecture-chemistry-papers-i-iii-20261010.md`
("r1"). r1 was committed at `3e5d4179` (SHA-256
`7bbb051db9acecbbedcc9c5133261fa842bd76d27e8bb9dbeca49c62cc9cc4fc`) and is not
edited here. r1's status line (lines 7–8, "nothing was committed") is stale for
that reason.

This revision answers Codex coordinator message
`chemistry-source-audit-20261010-010`. It reconciles the three independent
analyses and their refutations: A1 per-word leakage, A2 QSE conditions, and A3
scope and size contradictions. Their source is
`scratchpad/chem_r2_analyses.json` from the coordinator, with artifacts under
`output/local_runs/paper_v_chem_verify_20261010/<key>/` and `<key>-refute/`.

It also takes account of the upstream candidate audit
`agent_guidance/shared/handoffs/chemistry-upstream-candidate-audit-20261010.md`
(branch `codex/chemistry-papers-i-iii-groundwork-20261010`, commit `f9f4709e`,
SHA-256 `e878a547df8641303579ac7c4ac365de94e278c6906208264306e71f2c5567d1`).

**Scope.** r2 amends only what §0 lists. Everything else in r1 stands by
reference: the input contract (r1 §1), the Paper I entry analysis (r1 §2.1),
the impact numbers (r1 §5.2), and the baseline-failure verdict (r1 §5.3).
Where r2 and r1 disagree, r2 governs. Reconciliation with the three analyses,
including every disagreement, is in §5.

| Anchor | Value |
|---|---|
| Source | worktree `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`, HEAD `c1c1f9cc`. `git diff 0de9685a HEAD -- src pipelines test` is empty, so every `path:line` in r1 and r2 holds at both commits |
| My measurements | `output/local_runs/paper_v_chem_arch_20261010/` (gitignored), with the scripts that wrote each JSON. Each process ran in under one minute with `/Users/lordskyler/Holstein_test/.venv/bin/python`, except `ap_leakage_probe_v3.py`, which I stopped at 18 minutes with no output |
| Analyses' measurements | `output/local_runs/paper_v_chem_verify_20261010/` |
| "Synthetic 8-qubit system" | random real closed-shell integrals, 4 spatial orbitals, (N_alpha, N_beta) = (2,2), sector dimension 36. h is symmetric. g_pqrs = Σ_L B^L_pq B^L_rs with symmetric B^L, which gives exact 8-fold symmetry. Seeds: 11 (my runs) and 20261010 (A1's runs). This is not a molecule |
| "H2 writer payload" | `_write_molecular_problem_json` (`test/test_static_adapt_builders.py:88-125`): 2 spatial orbitals, (1,1), 4 qubits. A second copy sits at `test/test_adapt_vqe_integration.py:136-176` |
| Leakage metric (one definition throughout) | out-of-sector population 1 − ‖Π_(Na,Nb)ψ‖²/‖ψ‖², the SIAM definition (`pipelines/time_dynamics/diagnostics/siam_offline_scoring.py:611-612`). r1 §6.2 increment 1 used a norm; that is withdrawn |
| Upstream | unselected. Per the candidate audit's §7 matrix, no candidate supplies the local RA, AP or QSE contract. Nothing here selects, clones, installs, imports or adopts one |

---

## 0. What r2 amends

| r1 passage | r2 disposition |
|---|---|
| Lines 7–8 status | stale: r1 is committed at `3e5d4179` |
| Decision 0.1 (lines 29–34); site 4 effect "infeasible above about 13 qubits" (line 231); §6.0 "Nothing above 12 qubits" (line 418); §7 cap row (line 617) | §3.1. The cap is decided on the Claude side as a memory limit, and running above it is the user's scope. The two 16-qubit tests become 12-qubit guard tests. Paper III has no lifting increment. The copy counts are now measured |
| Decision 0.2 placement wording "after the existing ones" (lines 43–44, 336, 372–373) | §3.2: replaced by the invariant that actually holds, a default-false gate |
| Decision 0.3, consequence (b) (lines 54–58); §3.2 coordinates row (line 258); §5.1 grouped-generator row (line 323) | §1: (b) restated. The Paper II gate is an existing repository rule (logical-shared coordinates), not a choice |
| §4.1 ground sources (line 282 onward); §4.2 rows (lines 304–310); §6.3 increment 1 test (lines 588–591) | §2: conditions stated; certificate scope corrected; non-Hermitian pricing corrected |
| §2.1 line 167 "one added branch at each of sites 1–4" (site 2 = `_h2o_application_active`) | §3.2: `_h2o_application_active` is not edited. The new predicate is called beside each of its calls (`engine.py:1454`, `:2680`, `:5801`). r1's `:1461` is inside the H2O branch, not the call |
| §5.2 "not edited" verdicts contradicted by §6 Work lines; every shared-module write | §3.2 |
| §6.0 fixture (lines 407–411); §6.0 scope-out "`src/quantum/**` operator core" (lines 414–416) | §3.3. The H2 writer is an importable read-only helper, but its integrals are defective (N2). One seeded random-integral generator is owned by Paper I. The operator core is three files (`src/quantum/AGENTS.md:31-37`) |
| §6 increments and the "Files to edit" line (619) | superseded by §4 and §6 |

## 0.1 Negative results (stated first)

- **N1. A UCCSD-only support does not move under Paper II.** Every
  Jordan–Wigner word of a molecular UCCSD generator has an odd number of Y
  letters. Measured, `y_parity.json`: 12 of 12 words at 4 qubits, 160 of 160
  at 8, 828 of 828 at 12, and 2,688 of 2,688 at 16. For a real reference and a
  real H, each tangent −iP_kψ is therefore real while b̄ = −i(H−E)ψ is
  imaginary, so f = 0 and θ̇ = 0 under per-word and logical-shared coordinates
  alike.
  - My run (synthetic 8-qubit system, 6 single-excitation generators at common
    angle 0.1, append runner, Euler, t = 0 to 0.2): energy constant at
    −13.225633492505308, McLachlan residual ratio 0.9999999999999971, no
    insertion.
  - A1, independently: max|f| = 0.0 at 4 qubits (12 per-word, 3 logical
    coordinates) and at 8 qubits (160, 26), with ‖b̄‖² = 0.413 and 1.149.
  - A1's refuter: max|f| = 0.0 also for an independent random real
    number-conserving H; a complex H gives max|f| = 0.69 (per-word) and 1.39
    (logical).
  - A molecular Paper II support or pool therefore needs generators whose words
    have an even number of Y letters. Which ones is GPT A2's question.
  - The molecular `full_meta` pool lists its 160 UCCSD words first. The first
    even-Y word among its 520 deduplicated words is at index 160, from
    `ham_block::molecular_one_body` (`full_meta_order.json`). The structural
    prefix caps (typed default 8, production 128) therefore never offer one.
  - Without the cap, my 8-qubit generalized-exchange run did not finish its
    first configuration in 18 minutes and was stopped.
- **N2. The in-repository H2 writer payload is internally inconsistent.**
  - **Symmetry:** (00|11) = 0.181287518779 but (11|00) = 0.663472101, a
    (pq|rs) = (rs|pq) residual of 0.482 Hartree; every other symmetry residual
    is 0.
  - **Likely copy error:** the 0.181287518779 value equals the fixture's own
    (01|01). The JW builder sums both index orders, so H is Hermitian (dense
    residual 0.0) and uses the average.
  - **Effect on the spectrum (A2 refuter):** E_0 and E_3 are unchanged, but E_1
    and E_2 shift by −0.2410923 Hartree against the symmetric tensor.
    As written: E_1 = −0.7722149, E_2 = −0.4096399. Symmetric tensor:
    [−1.1359419, −0.5311227, −0.1685476, 0.4812186], which an independent
    Slater–Condon build matches to 4.4e-16.
  - **HF energy:** its `hf_energy` (−1.1166843871) differs from ⟨HF|H|HF⟩ of its
    own integrals (−1.115357102683) by −1.327e-3 Hartree.
  - **Why tests miss it:** a ground-energy check or a Hermitian interlacing
    test cannot see this.
- **N3. 4 qubits cannot witness per-word leakage.** At (1,1) in 2 orbitals,
  every UCCSD word maps the 4-dimensional sector into itself (12 of 12).
  Measured exactly 0.0 by me and by A1 at every deviation size, for both
  references, and in every AP run.
- **N4. Per-word coordinates leak in Paper II's own drivers at canonical
  numerics.** At 8 qubits, A1 measured:
  - up to 8.4e-2 by t = 2 with fixed support (RK4, ridge 1e-7);
  - 3.9e-1 through the generalized-exchange route;
  - at most 4.9e-32 in every logical-shared run.

  §1 gives the mechanism and the rule.

---

## 1. Correction 1 — per-word AP coordinates

### 1.1 Decision 0.3 restated

r1 consequence (a) stands: the Paper I molecular candidate is the whole macro
generator, and raw single-word candidates are inadmissible.

r1 consequence (b) is **withdrawn** and replaced by A1's corrected statement:

> Per-word execution equals grouped execution only when every word of
> generator k carries the same runtime coordinate (θ_{k,j} = θ_k, i.e.
> rotation angle c_{k,j}θ_k). Measured: norm difference 0.0 at θ = 0.2, at 4
> and 8 qubits; out-of-sector population 5.4e-34 (me) and 9.1e-34 (A1) at 8
> qubits.
>
> Paper II's default coordinates are one per Pauli word
> (`ap_append_from_adapt_artifact.py:155`; CLI default `:1772`; fixed runner
> `ap_fixed_from_adapt_artifact.py:63`) and span a larger manifold. Of the 160
> per-word directions at 8 qubits:
> - 46 commute with (N̂_alpha, N̂_beta) at the operator level (62 with N̂ alone);
> - 56 commute with the (2,2) sector projector, and only these leak nothing
>   from in-sector states. They include all 8 words of each same-spin double,
>   because flipping all four alpha qubits sends N_alpha = 2 to 2.
>
> At 4 qubits all 12 commute with the (1,1) projector.

### 1.2 Sector safety by lane

| Lane | Coordinates | Sector safety |
|---|---|---|
| Paper I | one angle per macro generator (r1 §2.1) | **Safe; keep.** All words of a generator commute and share one X/Y flip support, so exp(−iθG) conserves N_alpha and N_beta at every θ (5.4e-34 at 8 qubits) |
| Paper II | one per Pauli word by default, varied independently by the McLachlan solve | **Not safe by construction**, and leaking in practice (§1.4). A single word such as X_pY_q maps \|00⟩ to i\|11⟩ on its pair |

### 1.3 Leakage of the per-word product (static)

Definitions:
- φ_k = c_kθ̄ + δ_k is the rotation angle of word k, with δ_k i.i.d. uniform
  on [−δ, δ]. Words are applied as exp(−iφ_kP_k) in the AP runtime order
  (`build_parameter_layout`, `src/quantum/ansatz_parameterization.py:111`;
  `apply_exp_term`, `src/quantum/pauli_actions.py:320`).
- The table reports mean / max out-of-sector population over 32 draws (mine).
- p is the least-squares slope of log10(mean) against log10(δ).
- An AP runtime coordinate deviation Δθ rotates word k by |c_k|Δθ, where
  |c_k| = 1/2 for single-excitation words and 1/8 for double-excitation words.

Synthetic 8-qubit system (26 generators, 160 words):

| Reference | θ̄ | δ = 1e-3 | δ = 1e-2 | δ = 1e-1 | p |
|---|---|---|---|---|---|
| Hartree–Fock | 0 | 7.13e-10 / 1.89e-9 | 8.44e-6 / 2.52e-5 | 5.73e-2 / 1.00e-1 | 3.95 |
| Hartree–Fock | +0.2 | 1.45e-5 / 2.09e-5 | 1.33e-3 / 2.31e-3 | 1.27e-1 / 2.07e-1 | 1.97 |
| Hartree–Fock | −0.2 | 9.09e-6 / 1.44e-5 | 9.16e-4 / 1.61e-3 | 1.04e-1 / 1.75e-1 | 2.03 |
| random in-sector | 0 | 2.34e-5 / 3.96e-5 | 2.40e-3 / 3.96e-3 | 1.90e-1 / 2.82e-1 | 1.95 |
| random in-sector | +0.2 | 2.42e-5 / 4.13e-5 | 2.76e-3 / 4.95e-3 | 1.86e-1 / 3.09e-1 | 1.94 |
| random in-sector | −0.2 | 2.45e-5 / 3.54e-5 | 2.59e-3 / 4.04e-3 | 1.84e-1 / 2.86e-1 | 1.94 |

A1 and its refuter add the following (A1 seed 20261010, θ̄ = 0.3, 20 draws):

- **Leading order.** L = δᵀMδ + O(δ³), so the mean out-of-sector population is
  (δ²/3)·tr M for i.i.d. uniform spreads. A1's summary prefactor 2δ²/3 was
  refuted; (δ²/3)·tr M is the corrected form.
  - HF, rotation-angle convention: tr M = 40.80, rank 28 of 160. Predicted
    mean at δ = 1e-3 is 1.360e-5; measured 1.376e-5 (A1) and 1.426e-5
    (refuter).
  - Random in-sector state: rank 56.
  - At 4 qubits, M = 0.
- **AP runtime-coordinate convention** (θ_j = 0.3 + δ_j): HF mean 4.06e-7,
  4.19e-5 and 3.96e-3 at δ = 1e-3, 1e-2 and 1e-1 (p 1.99).
- **Fourth order on HF at θ̄ = 0.** On the unrotated Hartree–Fock state every
  occupied–virtual pair a word flips is (1,0), so the first-order amplitude
  vanishes (A1 p = 3.88; refuter 3.94).

### 1.4 Leakage in Paper II's drivers (dynamic)

All at 8 qubits, per-checkpoint out-of-sector population, with the numerics
named per row.

| Run (who) | Coordinates | Max leakage |
|---|---|---|
| UCCSD-only support, append runner, Euler, t ≤ 0.2 (me) | per-word | 7.8e-37 (frozen, N1) |
| Fixed support of 404 words: 6 UCCSD generators at 0.1 plus the two `ham_block` generators at 0, of which 392 words are even-Y. Euler, dt 0.02, ridge 1e-7, t ≤ 0.2 (me, `ap_fixed_per_word_leakage.json`) | per-word | 4.1e-9 at t = 0.2 (3.7e-11 at 0.02, 6.8e-11 at 0.10, 1.3e-9 at 0.16), roughly tenfold per 0.04 after t = 0.1. Also: ‖Δθ‖ = 2.80; energy drift +6.3e-3 Hartree, which the exact undriven flow conserves |
| Case B (A1): HF reference, 26 UCCSD generators at 0.3 plus 8 single-word Z_p generators; RK4, ridge 1e-7, dt 0.05, t ≤ 2; fixed support | per-word / logical | 3.3e-22 / 4.8e-33 |
| Case B, generalized-exchange append route | per-word / logical | 2.40e-2 / 1.07e-32 (detail below) |
| Case C (A1): random complex in-sector reference, a diagnostic and non-physical case; UCCSD at 0.3; fixed support | per-word / logical | 8.38e-2 (RK4) and 0.65 (Euler) / 1.43e-32 |
| Case C, append route | per-word / logical | 3.91e-1 / 4.87e-32 |
| Ridge scan, case C fixed per-word (A1) | ridge λ = 1e-7 / 1e-5 / 1e-3 / 1e-1 | RK4: 8.4e-2 / 1.4e-3 / 1.4e-14 / 3.6e-32 |

Detail for the case-B append row: 14 of 14 per-word insertions were a lone
half of an alpha single (e.g. X_2Y_1), and leakage began right after the
first one. Under logical-shared all 11 insertions were whole generators.

**Two mechanisms.**

1. **Ridge-amplified saddle.** On the common-angle locus K_LC = 0 and f_L = 0
   exactly (‖P_L K P_C‖ = 7.6e-16, ‖P_L f‖ = 4.8e-16). So in exact arithmetic
   the velocity has no number-changing component. The locus is nevertheless a
   saddle of the ridge-regularized flow:
   - The Jacobian of θ̇_L has real eigenvalues up to +1.38e5 / +1.38e3 / +13.5
     at λ = 1e-7 / 1e-5 / 1e-3.
   - The amplifier is the 58 K-null leaking directions (58 of 160 K
     eigenvalues are below 1e-8, all in the leaking block; condition number
     6.0e6).
   - At λ = 0 (pseudoinverse truncation only) the largest real eigenvalue is
     +3.46, and RK4 leaks 1.4e-25 by t = 2.
   - Raising the ridge is not a general fix. With an independent random real
     number-conserving H, λ = 1e-3 still leaks 5.9e-3 (rate +36.7).
2. **Single-word insertion.** Under per-word coordinates, the append route
   splits every pool generator into single-word candidates
   (`support_atoms.py:146-190`).

My fixed-support run grows at a lower rate than case C. That is consistent
with the refuter's finding that the leakage depends on the state, from 3e-22
to 8e-2.

### 1.5 The gate is an existing repository rule

r1 line 258 left the coordinate choice to GPT A2. The repository already makes
it.

**Static audit.** The shared static sector audit sets
`requires_logical_shared = grouped_preserves and not componentwise_preserves`
(`pipelines/static_adapt/sector_invariants.py:627-630`). This is True for 26 of
26 molecular UCCSD generators at 8 qubits and 3 of 3 at 4
(`A1-perword-leakage/static_audit_check.json`).

**Lanes that already obey it.**
- The static lane resolves to `logical_shared` when the audit says so
  (`pipelines/static_adapt/adapt_pipeline.py:12457-12463`, `:17929-17937`).
- Paper II already enforces `logical_shared` for SIAM: runner guard
  `ap_append_from_adapt_artifact.py:2478-2481`, and the SIAM runs in
  `paper_ii_runs.py:1382,1513`. The SIAM precedent ran Euler with ridge 1e-6
  (`paper_ii_runs.py:1375`).

**Gate for molecular UCCSD seeds in Paper II:**
1. `parameterization_mode = logical_shared`, with `theta_logical` in the seed
   (`state.py:915-937`). Under logical-shared, append atoms are whole
   generators (`support_atoms.py:110-141`). The structural cache admits
   `termwise_product` terms under logical-shared
   (`structural_cache.py:72-79,106-120`), and every molecular UCCSD term is
   `termwise_product`.
2. `per_pauli_term` fails closed when the seed's generator audit reports
   `requires_logical_shared_parameterization`.
3. Every checkpoint reports the population leakage of the Leakage-metric
   anchor, with ceiling 1e-8. That is the SIAM ceiling
   (`siam_offline_scoring.py:92`, check `:619-620`). It is a fail-closed
   diagnostic, never a controller input. A reusable auditor exists:
   `FixedCountSectorStateAuditor` (`sector_invariants.py:756`).

The per-word option is **not admissible at the canonical ridge**.

**Facts for the implementation:**
- **The generic seed loader carries `theta_logical`.** Read from code, not
  run. `_resolve_theta_logical` (`runtime_loader.py:595-615`, called at `:1157`,
  passed to the runtime input at `:1342-1345`) returns
  `adapt_vqe.logical_optimal_point` when present. Otherwise it returns the
  block mean of the runtime angles (`project_runtime_theta_block_mean`,
  `src/quantum/ansatz_parameterization.py:185`), which equals the logical angle
  exactly on the common-angle locus.
- **A candidate seam for the fail-closed rule:** the legal-subspace append
  guard. `legal_subspace_append_guard.no_pauli_split_parent_labels`
  (`runtime_loader.py:1894-1970`) makes `support_atoms.py:105,148` refuse to
  split a marked parent. It is applied only on the HH loader path
  (`runtime_loader.py:2039`); non-HH seeds take the generic loader
  (`:2086-2095`) and never receive it.

**Question for the Paper II lane (not a finding).** HH Paper II production runs
use `per_pauli_term` without a leakage report. `paper_ii_runs.py` passes
`--parameterization-mode logical_shared` only in the SIAM commands
(`:1382,1513`). Do the HH fermionic words leak under the same mechanism? r1
line 258's analogy to HH words was not checked.

**GPT A2 keeps two questions:** which even-Y generators make a molecular
support move (N1), and pool order versus structural cap.

**Replacement rows for r1:**
- **Line 258:** "One per Pauli word by default; molecular UCCSD runs
  logical-shared by the existing sector-audit rule. Per-word leaks up to 8.4e-2
  (fixed support) and 3.9e-1 (exchange route) at 8 qubits, against ≤ 4.9e-32
  for logical-shared. Per-checkpoint population leakage ≤ 1e-8. A UCCSD-only
  pool freezes the flow (N1)."
- **Line 323:** "Absent for spin-orbital UCCSD under logical-shared
  coordinates. Under per-word coordinates the sector is not preserved."

---

## 2. Correction 2 — QSE bounds and the conditions they need

Notation:
- H_s is H restricted to the (N_alpha, N_beta) sector, with eigenvalues
  E_0 ≤ E_1 ≤ … counted with multiplicity.
- θ_1 ≤ θ_2 ≤ … are the eigenvalues returned by
  `solve_qse_generalized_eigenproblem` (`pipelines/qse_spectra/core.py:843-937`)
  on the retained subspace W.
- f = |⟨E_0|ψ⟩|² is the ground fidelity of the Q0 root ψ.
- ΔE_ref = ⟨ψ|H|ψ⟩ − E_0.

The statements, as A2 derived and its refuter checked:

> **(A)** If W lies in the sector, θ_k ≥ E_{k−1}.
>
> **(B)** If in addition ψ is an exact sector eigenvector at E_0 (degenerate
> allowed) and Q0 = 1 − |ψ⟩⟨ψ|, then θ_k ≥ E_k.
>
> **(C)** For any other ψ, θ_k ≥ E_k − (E_k − E_0)(1 − f), and when E_1 > E_0
> also θ_k ≥ E_k − ΔE_ref (E_k − E_0)/(E_1 − E_0).
>
> Each holds up to roughly ε_mach·κ(S_retained)·‖H‖.

r2's draft used tan²(a) = (1−f)/f in place of (1 − f). That form is valid but
loose, because a unit v ⊥ ψ has |⟨E_0|v⟩|² ≤ 1 − f (§5, D4).

### The five conditions

**(i) Sector.**
- *Requirement:* every record B_i commutes with N̂_alpha and N̂_beta
  separately, and ψ has zero (N_alpha, N_beta) leakage, so that W lies in the
  sector.
- *What the code does:* it enforces neither. `sector_projection` must be
  `identity` (`core.py:83-84`), and `sector_label` is metadata
  (`core.py:628-631,691-693`). Root leakage is checked for SIAM imports
  (≤ 1e-10, `siam_static_adapter.py:74,339-344`), but the HH import checks
  only the norm (`paper_iii_static_qse_cell.py:228-230`).
- *Measured:*
  - 4 qubits (A2): single-qubit X records put θ_2 − E_2 = −0.3575 and
    θ_3 − E_3 = −0.9181 Hartree, and the code reported no failure.
  - 8 qubits (me): the 160 single words at the exact ground put 9 Ritz values
    below the sector E_0; θ_1 = −21.506 is the full-register minimum. The
    full-register min–max bound still holds (min_k(θ_k − λ_k) = −8.9e-14).
- *Gate:*
  - the provider asserts [B_i, N̂_alpha] = [B_i, N̂_beta] = 0 for every record;
  - it records ψ's sector leakage for every ground source.

**(ii) Exact-ground complement and index matching.**
- *What Q0 removes:* the prepared ψ, not |E_0⟩. Q0 subtracts ψ⟨ψ|raw⟩
  (`core.py:663-664`), with ψ the normalized prepared state (`core.py:762`).
- *Index matching:* (B) holds only for an exact eigenvector, and even then the
  exact ground is **necessary but not sufficient**. A support can skip sector
  roots below E_ν. Root k may be compared with E_k only when the support
  reaches every sector eigenvector below it, and no code verifies that.
- *Measured* (A2 and refuter, 4 qubits, symmetric H2 tensor):
  - **Exact ground, all three UCCSD generators:** max|θ_k − E_k| = 4.4e-16,
    κ = 1.578. This is an equality, because the support spans the
    3-dimensional complement.
  - **Exact ground, subset {sing(α), dbl}:** θ_1 − E_1 = +0.2219 and
    θ_2 − E_2 = +0.6498.
  - **HF root** (f = 0.98727): θ_3 − E_3 = −2.0585e-2 Hartree = ΔE_ref, and the
    fidelity bound (C) is tight to 2e-16.
  - **Certificate scope.** `certify_fixed_pool_support`
    (`method_completion.py:121-147,209-217`) still reports
    `target_window_certified = true`, with residual 7.47e-17, on that HF root.
    The certificate bounds θ_ν against an eigenvalue of Q0HQ0 on the
    complement of ψ, not against E_ν. With the exact ground it was also
    certified when the support skipped roots:
    - pool {Gd}, R = 1: θ_1 = E_3, residual 1.1e-16;
    - spin-adapted singlet pool, R = 2: θ_1 = E_2, residual 1.5e-16.
  - **8 qubits (me, `qse_fidelity_bound_recheck.json`):**
    - HF root (f = 0.732): 3 of the lowest 6 θ_k fall below E_k, the worst
      θ_6 − E_6 = −0.694 Hartree. The margins of the surviving bounds are
      +0.149 for (C) and +0.0104 for (A).
    - Roots ψ_0 + εξ (f = 1/(1+ε²), ε = 1e-3, 1e-2, 1e-1): no violations;
      (C) holds with margins 9.2e-4, 1.06e-3 and 2.27e-2.
- *Which grounds carry (B):*
  - The exact-sector diagnostic fixture (HH: `paper_iii_static_qse_cell.py:208-217`
    via the dense sector eigh, `paper_iii_qse_regime_frontier_sweep.py:141-156`;
    SIAM: `siam_static_adapter.py:278-297,407-424`) carries (B) to eigensolver
    accuracy. So does the SIAM exact benchmark seed (`:427-449`).
  - This holds provided the vector is an eigenvector of the pencil's own qubit
    H. That is test-checked only for SIAM `siam_nb2_fixture`
    (`test/test_paper_iii_siam_static_adapter.py:111-122`). It is unchecked
    when HH loads from the reference store, which has no Hamiltonian content
    digest and no eigen-residual check on load
    (`paper_iii_problem.py:153-162,224-231`).
  - An imported prepared state (ADAPT) carries only (A) and (C).
- *The lane's actual guard:* the post-run `window_error_gate` of 1e-6 against
  dense exact references (`paper_iii_static_qse_cell.py:78`, check `:858`).
  - Rank < R and solver errors become `None` in the growth trace
    (`paper_iii_growth_trace_campaign.py:141,146`), and qualification fails
    them through this gate (`:858-859`).
  - The gate needs the dense reference spectrum, so above the 12-qubit cap
    nothing in the lane detects a θ_k < E_k or a skipped root (§3.1).
- *Gate:*
  - the molecular provider defines its diagnostic ground as the exact sector
    eigenvector and asserts ‖(H − E_0)ψ‖ ≤ 1e-10;
  - every bound statement names its ground source and uses (B) only for an
    exact one;
  - index matching is asserted against the dense sector spectrum, never
    inferred from the certificate.

**(iii) Completed pencil and conditioning.**
- *Construction:* S = Φ†Φ and H = Φ†HΦ with Φ_i = Q0 B_i ψ (raw-projected,
  `core.py:796-803`).
- *Completion:* Löwdin canonical orthogonalization.
  - It raises if min eig(S) < −max(1e-12, 1e-9·s_max) (`core.py:875-881`).
  - It keeps s ≥ max(1e-12, 1e-10·s_max) (`core.py:886-893`; defaults
    `core.py:149-160`), so the retained κ ≤ 1e10.
  - `retained_rank` and `discarded_rank` are reported (`core.py:930-931`), and
    QSEResult keeps every raw overlap eigenvalue. The qualification record
    keeps only ranks, min/max and condition (`paper_iii_static_qse_cell.py:893-901`).
  - `overlap_condition_gate` = 1e14 (`:82`, `:866`) can never bind under the
    default cutoffs.
- *Measured, 4 qubits (A2 and refuter):* {Gsa, Gd, c·Gsb}:

  | Scale c | κ | θ_1 − E_1 |
  |---|---|---|
  | 1e-4 | 1.05e8 | −2.08e-9 |
  | (not recorded) | 1.17e9 | +2.83e-8 |
  | 1.2e-5 | 7.31e9 | −1.83e-7 |
  | 1e-5 | pruned to rank 2 | the E_1 root is lost |

  A deliberately singular 6-record pencil of rank 2 lost 4 directions and
  reproduced the nonsingular Ritz values to 2e-16.
- *Measured, 8 qubits (me):* adding an exact duplicate record gives discarded
  rank 1 and Ritz values unchanged to 1e-14. The 160 single words give
  retained 63, discarded 97, κ 8.5e5.
- *Gate:*
  - a bound test at tolerance τ also asserts κ ≤ τ/(ε_mach·‖H‖);
  - a 1e-10 tolerance needs κ well below about 1e7 at this ‖H‖, so the
    provider test asserts κ ≤ 1e2;
  - under raw-projected vectors the discarded set depends on record scale, so
    record coefficients are fixed by name (v).
- *Monotonicity:* "adding a record cannot raise a Ritz value" holds only for
  nested retained subspaces. The selector's frame cutoff is relative only
  (`record_selection.py:540`). It agrees with the solver's
  max(1e-12, 1e-10·s_max) only when s_max ≥ 1e-2.

**(iv) Cost weighting.**
- *How cost enters:* only in ranking. The score is
  utility / max(0.05, K_i/K_max)^α with α = 1 (`record_selection.py:793-797`,
  floor `:82`; locked config `selection_policy.py:34-46`). K_max is fixed per
  selection call over the initial eligible pool (`record_selection.py:512`).
- *Consequence:* cost changes which subspace is selected, never the Ritz values
  of a given subspace, so (i)–(iii) are unaffected.
- *Properties to state:*
  - Every record with K_i < 0.05·K_max shares one denominator.
  - The Ky Fan gain is clipped at 0 (`:880`) and padded during bootstrap
    (`:853-864`).
  - K_i is the `two_qubit_only_v1` Marrakesh graph-span tail-insertion price
    of exp(−iθB_i) (`compiled_costs.py:61-72,169-215`), not a measurement cost.
- *Correction to r1 line 305:* the oracle does **not** leave a non-Hermitian
  record unpriced. It silently prices a†_1a_0 at 8.0 two-qubit gates against
  4.0 for the Hermitian generator (4 qubits, A2 refuter), and that price has
  no circuit meaning.
- *Hermiticity and the pencil:* the pencil itself accepts non-Hermitian records
  (`core.py:466-473`). Hermitian-only records span {(T − T†)ψ} ⊂ {Tψ, T†ψ}, so
  the bounds stay valid but are not the tightest available.
- *Gate:*
  - `compiled_costs` is mandatory, with the explicitly passed preset;
  - the provider asserts B_i = B_i† before costing.

**(v) Provenance.**
- *Three mutually exclusive ground sources* (`siam_static_adapter.py:569-595`):
  - an imported artifact, validated for norm, leakage and quench overlap
    (`:315-370`);
  - the exact-sector benchmark seed, `paper_facing: true` (`:427-449`);
  - the diagnostic fixture, `diagnostic_only: true` (`:407-424`).

  r1 §4.1 named only two.
- *What exists today:*
  - Exact sources record `ground_energy`; imported sources record neither
    ⟨ψ|H|ψ⟩ nor ‖(H − ⟨H⟩)ψ‖.
  - `pool_digest` hashes ordered names only, truncated to 24 hex
    (`paper_iii_problem.py:129-132`; `siam_static_adapter.py:272-275`).
  - R is `len(cell.target_roots)` (`paper_iii_static_qse_cell.py:182`). The
    reference count is R+1 (`paper_iii_problem.py:220`), and the provider
    default R = 6 (`paper_iii_problem.py:176`; `siam_static_adapter.py:554`)
    is an HH/SIAM constant.
- *Gate:* the molecular provider records, for every source:
  - ⟨ψ|H|ψ⟩, ‖(H − ⟨ψ|H|ψ⟩)ψ‖, and sector leakage;
  - an operator-content digest, and the S0 integral digest (§4.1);
  - R declared per molecule (r1 input item 17).

**Spin.** Within an (N_alpha, N_beta) sector the E_k include all M_S = 0
states. On H2 the M_S = 0 triplet is E_1 = −0.53112. Labelling a root by spin
needs r1 input item 18 or spin-adapted records.

**Memory.** r1 decision 0.1's "Paper III's QSE core is matrix-free
(`core.py:777`)" stands: H is compiled once and applied per column
(`core.py:777-783,799-801`). It does, however, hold the m Q0-projected basis
vectors densely, about 2 × 16 × 2^n × m bytes at peak (`core.py:796`;
`method_completion.py:94-99` stacks them again).

**Replacement for the r1 §6.3 increment-1 test (lines 588–591):** see III-1 in
§4.4.

---

## 3. Correction 3 — scope and size contradictions

### 3.1 The 12-qubit cap

**Ownership.** The cap is **decided on the Claude side as a memory limit**
(coordinator ruling, consistent with r1 decision 0.1 and coordination record
lines 315–316). **Running anything above 12 qubits is the user's scope**
(§6). r1 line 617, which gave the cap itself to the user, is withdrawn.

**What the cap covers.** No molecular fixture or run above 12 qubits enters
any path.
- The two 16-qubit tests of r1 (§6.1 increment 3 at lines 503–505, §6.2
  increment 2 at lines 549–551) become **12-qubit guard tests in which the
  dense builder is patched to raise** (I-3 and II-2 in §4).
- A 16-qubit run is a **named user-scope step**, U-16 in §4.5, not part of any
  dispatched increment.
- r1 line 418 "Nothing above 12 qubits" stands, now consistent.

**What the cap does not cover.**
- *Paper III* has no lifting increment at all. Its `window_error_gate`
  (§2 (ii)) depends on the dense reference spectrum, so above 12 qubits the
  lane would have no guard against θ_k < E_k or skipped roots.
- *Paper II:* the matrix-free option alone serves a geometry evaluation. An AP
  run also needs the option set where the runner builds the provider
  (`ap_append_from_adapt_artifact.py:174`), which II-2 now includes.
- *Paper I:* I-3 removes the dense exact target for the molecular application
  at every size (§3.2).

**Measured dense cost.** These copy counts replace r1's single-copy figure and
r1 line 231 ("infeasible above about 13 qubits", which is withdrawn as
unattributed). One copy is 16·4^n bytes: 0.27 GB at 12 qubits, 4.3 GB at 14,
69 GB at 16 (decimal GB). Copy counts were measured with tracemalloc at 10
qubits (A3 refuter, `A3-scope-size-contradictions-refute/results.json`):

| Path | Peak copies of 16·4^n bytes | At 14 qubits |
|---|---|---|
| Paper I `hamiltonian_matrix` build | 3 | 12.9 GB |
| Paper II undriven, first geometry evaluation / later | 3 / 2 | 12.9 / 8.6 GB |
| Paper II driven, first / later | 4 / 3 | 17.2 / 12.9 GB |

**Measured dense time** (random integrals):
- `hamiltonian_matrix` took 9.2 s at 10 qubits (876 Pauli terms).
- The dense exact sector energy took 256.0 s at 12 qubits (1,819 Pauli terms),
  with about 0.9 GB RSS.
- So a 12-qubit molecular RA run that resolves the dense exact target pays
  about 256 s before any controller round, and Paper II's first geometry
  evaluation at 12 qubits pays the same build. The JW polynomial build itself
  is small: 0.50 s at 8 qubits, 3.46 s at 12 (mine).
- The 18 GB figure in `agent_guidance/shared/memory-budget.md:15` is "the prior
  … planning allowance", which that file treats as context, not a rule.

### 3.2 Writes to shared modules, and r1's "not edited" verdicts

The invariant that actually holds for every edited existing function is a
**default-false gate**. The new molecular predicate is false for every
existing request, so each existing request reaches the same branch with the
same arguments, and every Hubbard–Holstein digest is unchanged. Placement
wording ("after the existing ones") is dropped: it cannot hold in
`_parent_pool_spec`, where a molecular branch would have to sit before the
`_require_supported_singleton_problem` call at `pools.py:581`.

| Module (impact verdict) | r1 increment that wrote it | r2 disposition |
|---|---|---|
| `ra_adapt/pools.py` (`_parent_pool_spec` LOW; `_receipt` MEDIUM, 9 callers) | §6.1-2 | **Not written.** The only callers of `build_parent_template_inventory` and `build_executable_macro_pool` outside `pools.py` are adapter methods (`adapters.py:135,144`), and the molecular adapter overrides both. It imports `_receipt` read-only, as `adapters.py:162` does. If I-2b finds the runtime still reaches `_parent_pool_spec`, the branch is keyed on the family between `:580` and `:581` under the default-false gate (A3 refuter); `pool_key` validation at `:542-554` already admits `uccsd` |
| `ra_adapt/adapters.py` (sibling class) | §6.1-2 | **Moved** to new `ra_adapt/molecular_application.py`, the pattern of `l3_page12.py` and `pure_hubbard_noise_page12.py` |
| `ra_adapt/engine.py` (`run_ra_adapt`, `build_resolved_ra_protocol` CRITICAL; `_repaired_route_contract` HIGH; `_h2o_application_active` LOW) | §6.1-2 | **Gated increment I-2b.** Default-false branches in `run_ra_adapt` (`:5742-5755`), `_repaired_route_contract` and `build_resolved_ra_protocol`. The new predicate is called beside each `_h2o_application_active` call (`:1454`, `:2680`, `:5801`). `_h2o_application_active` itself is not edited: it returns False for any non-H2O family (`:520-522`). The named-application source-lock check at `:1826` is extended to the molecular application in the same increment |
| `ra_adapt/contracts.py` (`_adapter_from_mapping` LOW, 2 callers) | missed in r1 | **Gated, joins I-2b:** one branch (`contracts.py:1393-1401`). The protocol round trip otherwise rejects the id ("Macro adapter id drifted."). `RAAdaptRequest` (`:247`) is not edited; the gradient tolerance in Hartree is a required keyword of the molecular request builder, as in `build_h2o_ra_request` (`h2o_application.py:128-133`) |
| `sr_snake_route_profile.py` (HIGH) | §6.1-2 | **Moved** to `molecular_application.py` (import of `canonical_sr_snake_no_prune_symmetric_cost_v1_contract` only). If validation resolves the new profile by id (`validate_sr_route_profile_contract`, `:4116-4150`), registration joins I-2b |
| `src/quantum/vqe_latex_python_pairs.py` (`exact_ground_energy_sector` HIGH) | §6.1-3 ("beside" it) | **Moved** to new `pipelines/static_adapt/builders/sector_sparse_exact.py`, outside `src/quantum/` |
| `builders/problem_setup.py` (`_exact_gs_energy_for_problem` MEDIUM) | §6.1-3 | **Not written.** `resolve_problem_context` takes `exact_energy_impl` and the molecular resolver calls it (`problem_registry.py:804-860`). The molecular application injects the sparse helper at every size, which also removes the 256 s dense build at 12 qubits |
| `ap_mclachlan/hamiltonian.py` (`TimeDependentHamiltonian` HIGH, 20 files; frozen, `:21-32`) | §6.2-2 | **Gated, II-2:** one defaulted attribute (dense by default) plus one additive method applying H_static ψ + c(t)·Dψ through compiled Pauli actions (`src/quantum/compiled_polynomial.py:40,85`). The factory is not edited; the runner sets the attribute with `dataclasses.replace`. This replaces my draft's sibling class (§5, D11) |
| `ap_mclachlan/geometry_eval.py` (`evaluate_mclachlan_geometry` HIGH, 9 callers) | §6.2-2 | **Gated, II-2:** one branch on the attribute; `:72-77` stays the dense default |
| `runners/ap_append_from_adapt_artifact.py` (`run_append_ap_mclachlan_from_runtime_input`: graph LOW, 1 caller; grep finds 8 files including 6 tests) | not in r1 | **Gated, II-2:** set the attribute where the provider is built (`:174`) from a named runner option. **Gated, II-3b:** the logical-shared fail-closed rule and the per-checkpoint leakage report, beside the SIAM guard (`:2478-2481`) |
| `time_dynamics/run_lock.py` (`build_run_lock` LOW) | §6.2-3 | **Gated, II-3a**, unchanged |
| `adapters/drive_terms.py` (HIGH), `builders/problem_registry.py` (`_MOLECULAR_CAPABILITIES`, `:1586`, one reference `:1744`) | §6.2-4 | **Gated and blocked (II-4).** `problem_registry.py` was missing from r1's file list |
| `exact_bench/paper_iii_static_qse_cell.py` (`_resolve_problem` LOW) | §6.3-2 | **Gated, III-2**, unchanged |
| `qse_spectra/static_campaign.py` (`StaticQSEPhysicsKey` MEDIUM) | §6.3-2 | **Moved:** the constructor goes into the new provider module |
| `ap_mclachlan/state.py`, `structural_cache.py`, `qse_spectra/core.py`, `paper_iii_problem.py`, `sector_invariants.py`, `psi4_adapter.py` | none | not written |

### 3.3 Files more than one lane would write

| Item | r1 | r2 owner |
|---|---|---|
| H2 writer | "every lane uses it" | no write at all. It is an importable test helper; cross-test imports are established (e.g. `test/test_ra_adapt_refactor_parity.py:27`). Because of N2, positive tests use an in-memory symmetric copy, (00\|11) := (11\|00), never the writer as written |
| Random closed-shell integrals | "in the test itself" (three copies) | **one seeded generator, owned by Paper I** (S0): `test_support/molecular_closed_shell_fixture_factory.py` (the `test_support/h2o_linear_fd_fixture_factory.py` pattern) |
| Integral content digest | II-3 and III-2 each implied one | **S0** (`pipelines/contracts/molecular_input.py`); II and III import it |
| Sector-restricted eigenpairs | I-3 (energy) and III-1 (spectrum) | **Paper I (I-3)**: `pipelines/static_adapt/builders/sector_sparse_exact.py` returns the lowest R eigenpairs of the (N_alpha, N_beta) block, with vectors embedded in the full register, because Paper III needs the ground vector. Until it lands, III-1 keeps its own sector-index code at ≤ 12 qubits. `_sector_spectrum` hard-codes the HH (1,1) qubits (`paper_iii_qse_regime_frontier_sweep.py:141-155`) and is not reusable |
| `_MOLECULAR_CAPABILITIES` | II-4 | Paper II only; Papers I and III read it |

### 3.4 Time bounds

**Definition.** Every bound in §4 limits **agent wall time from the start of
the increment to its commit**, per the coordination record's rule that every
local run and every handoff's first increment is bounded at 20 minutes. Each
increment also states its test-command runtime (measured or expected),
separately.

r1's "about 20 minutes" (five increments) is replaced by explicit bounds.

**Measured test runtimes:**

| Suite | Result | Runtime |
|---|---|---|
| `test/test_ra_adapt_facade.py` | 67 tests: 66 passed, 1 failed (`FULL_SUITE_BASELINE_20260831_failed.txt:547`) | 200.34 s at `a23ff642` (refuter, run concurrently with another job; slowest test 45.85 s) |
| Paper I batch | 62 passed, 2 failed | 72.8 s |
| Paper II batch | 69 passed, 5 failed | 37.6 s |
| Paper III batch | 51 passed | 7.3 s |
| Parity | 2 passed | 9.4 s |

**Corrections to r1 §6.1 increment 2:**
- Its expected "62 passed / 2 known reds" came from a command that omitted
  `test/test_hamiltonian_adapter_capabilities.py`, which holds both reds; that
  command would show 58/0.
- Its CRITICAL `run_ra_adapt` edit had no test catching a change to the
  existing branches (HH L=3, Hubbard L ∈ {2,4}, H2O).
- I-2b fixes both (§4.2).

### 3.5 Cross-lane contract writes need the user

r1 §6.1 increment 4 wrote the Paper IV lane contract
(`agent_guidance/molecular-vibronic/AGENTS.md:22-25`). The root contract carries
the same stale facade sentence (`AGENTS.md:94-96`, "remains executable only
for the Paper-I Hubbard--Holstein `L=2` problem"). Both are cross-lane or root
writes and are listed for the user (§6). I-4 edits only
`agent_guidance/static-adapt/AGENTS.md:79-80`.

---

## 4. Revised increment list (supersedes r1 §6.1–6.3)

**Order.** S0 first, on the Paper I branch. Papers II and III branch from the
commit that lands S0; they do not wait for Paper I's I-2. Each chain proceeds
autonomously, commits at every increment boundary, and stops on any stop
condition.

**Shared rules.** r1 §6.0 stands except the fixture and scope-out lines
amended in §0.

**Baselines** at `0de9685a`, unchanged at `c1c1f9cc`:
- Paper I batch: 62 passed / 2 failed
  (`FULL_SUITE_BASELINE_20260831_failed.txt:356-357`), plus 2 parity passed;
- Paper II batch: 69 / 5 (`:575-577,591-592`);
- Paper III batch: 51 / 0;
- facade suite: 66 / 1 (`:547`).

### 4.1 S0 — content-bound closed-shell input validation and provenance (first seam)

**Bound:** under 15 minutes of agent wall time. The new tests are expected to
run in under 10 s.

**Goal.** One pure validator turns a raw closed-shell JSON payload into a
receipt, or fails with named reason codes. It runs before any loader default
applies, it is characterized on named in-repository fixtures, and it is
independent of the upstream choice.

**Files (new only; no existing file written):**
- `pipelines/contracts/molecular_input.py`. Pure Python with no NumPy and no
  `src` imports, as `pipelines/contracts/__init__.py:1-6` permits ("pure
  validators, and small deterministic helpers"). It contains the validator,
  the receipt dataclass and the digests.
- `test_support/molecular_closed_shell_fixture_factory.py`: the seeded random
  real closed-shell generator, with exact 8-fold ERI symmetry (construction as
  in `output/local_runs/paper_v_chem_arch_20261010/chem_fixtures.py`), plus a
  helper that adds provenance fields. Owned by Paper I; Papers II and III
  import it.
- `test/test_molecular_input_contract.py`.

**Checks.** Each failure has a named reason code and is evaluated on the raw
mapping:

1. **Required keys, no defaults:** `n_spatial_orbitals`, `n_alpha`, `n_beta`,
   `reference`, `multiplicity`, `charge`, `one_body_integrals_mo`,
   `two_body_integrals_mo`, `nuclear_repulsion_energy`, and `hf_energy`.
   A missing scalar is `missing_scalar_energy`; a missing HF energy is
   `missing_hf_energy`. These are the two silent-0.0 gaps of
   `psi4_adapter.py:122-123`, now hard failures.
2. **Provenance (r1 input item 11):**
   - `geometry_spec`; `geometry_units` ∈ {`angstrom`, `bohr`}; `basis`;
     `scf_type`; `point_group`;
   - `active_orbitals`, `frozen_core_orbitals`, `deleted_virtual_orbitals`:
     disjoint index lists, with `len(active_orbitals) == n_spatial_orbitals`;
   - `energy_units == "hartree"`.

   A missing field fails as `missing_provenance:<field>`.
3. **Closed shell:**
   - `reference == "rhf"` and `multiplicity == 1`;
   - `n_alpha == n_beta`, else `open_shell`;
   - 0 ≤ n_alpha ≤ m;
   - `n_spin_orbitals == 2m` when present.
4. **Real and finite:**
   - every entry is a JSON number that is not a bool, string or `[re, im]`
     pair (`non_real_integral`);
   - every entry is finite (`non_finite_integral`);
   - shapes are (m,m) and (m,m,m,m) (`bad_shape`).
5. **Symmetry**, with named parameter `symmetry_atol` (default 1e-10 Hartree,
   recorded):
   - h = hᵀ (`one_body_asymmetric`);
   - (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq), reason `eri_symmetry:<relation>`.
6. **HF witness**, with named parameter `hf_energy_atol` (default 1e-8 Hartree,
   recorded). The check is |hf_energy − E_HF| ≤ `hf_energy_atol`, with
   E_HF = E_scalar + 2Σ_{i<n} h_ii + Σ_{i,j<n}[2(ii|jj) − (ij|ji)] over the
   first n = n_alpha orbitals (r1 input item 8). This is a closed form on the
   payload values: no Hamiltonian, loader or state is built (§5, D1). Failure
   reason: `hf_energy_inconsistent`.

**Receipt.**
- physics digest: sha256 of canonical JSON of (m, n_alpha, n_beta, scalar, h, g);
- provenance digest: sha256 of canonical JSON of the item-11 fields;
- sha256 of the exact bytes read;
- the measured symmetry residuals and the HF witness value.

**Expected results.**
- **H2 writer payload as written** (4 qubits): fails with exactly
  - the seven missing provenance fields (`geometry_units`, `scf_type`,
    `point_group`, `active_orbitals`, `frozen_core_orbitals`,
    `deleted_virtual_orbitals`, `energy_units`);
  - `eri_symmetry:(rs|pq)`, residual 0.48218458222 Hartree;
  - `hf_energy_inconsistent`, difference −1.3273e-3 Hartree (N2).
- **Same payload with provenance supplied:** still fails on the last two.
- **Same payload with `nuclear_repulsion_energy` or `hf_energy` deleted:**
  fails `missing_scalar_energy` or `missing_hf_energy`. The test also pins,
  without editing the loader, that `restricted_closed_shell_problem_from_jsonable`
  returns 0.0 for those fields.
- **In-memory symmetric copy** ((00|11) := (11|00), `hf_energy` := E_HF,
  provenance supplied): passes.
- **Synthetic 8-qubit payload** from the fixture factory (provenance and
  `hf_energy = E_HF`):
  - passes;
  - its physics digest is unchanged under key reordering and changes when one
    symmetric ERI quadruple moves by 1e-9;
  - E_HF equals ⟨HF|H|HF⟩ from `build_restricted_closed_shell_molecular_hamiltonian`
    to 1e-12 (tests may import `src`).
- **Negative controls:** an `[re, im]` entry gives `non_real_integral`; a NaN
  gives `non_finite_integral`; n_alpha ≠ n_beta gives `open_shell`.

**Commands.**

```sh
/Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q test/test_molecular_input_contract.py
```

Then the three lane baseline batches.

**Stop.** Stop and report, without editing any fixture, if the H2 writer
payload passes a check listed as failing, or if any baseline count moves.

### 4.2 Paper I chain

- **I-1 (under 15 min; tests only).**
  - Work: r1 §6.1 increment 1, plus (a) an assertion that per-word execution
    at a common angle equals grouped execution (norm difference 0.0 at 4 and
    8 qubits, θ = 0.2), and (b) a recorded facade-suite count and time.
  - Expected: facade 66 passed / 1 failed (`:547`), about 200 s.
- **I-2a (under 20 min; new files only).**
  - Work: `pipelines/static_adapt/ra_adapt/molecular_application.py` holds:
    - the predicate;
    - the named macro adapter (`application_family_key` set;
      `parent_inventory` and `executable_pool` overridden; built from
      `build_molecular_uccsd_pool`; fails closed on any grouped or execution
      violation in `audit_candidate_pool_sector_contract`,
      `sector_invariants.py:656`);
    - an S0 receipt requirement;
    - the parent route contract;
    - the request builder (required `gradient_tolerance` in Hartree);
    - problem resolution with `exact_energy_impl` from I-3 once it lands.
  - Tests go in `test/test_ra_adapt_problem_family_admissibility.py`:
    - the 8- and 12-qubit synthetic pools (26 and 117 generators, zero
      violations);
    - contract determinism;
    - the H2 writer payload is rejected through S0.
  - Stop: on any need to write `pools.py`, `adapters.py`,
    `sr_snake_route_profile.py` or `engine.py`, or if the SR runtime rebuilds
    its pool from execution settings rather than from the adapter.
- **I-2b (under 20 min; gated: `engine.py` CRITICAL, `contracts.py` LOW).**
  - Work: the default-false branches of §3.2.
  - Tests:
    - a two-round fresh-start run on the synthetic 8-qubit system;
    - `request=None` rejected;
    - the single-word adapter rejected on the molecular problem;
    - the named adapter rejected on the HH problem.
  - Command: I-1's command plus `test/test_ra_adapt_h2o_application.py`,
    `test/test_hamiltonian_adapter_capabilities.py`,
    `test/test_static_adapt_builders.py` and `test/test_ra_adapt_facade.py`.
  - Expected:
    - the Paper I batch 62 passed / 2 failed, unchanged;
    - parity 2 passed;
    - the facade suite 66 / 1, unchanged — this is the test that catches any
      change to the existing `run_ra_adapt` branches;
    - the I-1 digests unchanged;
    - the new tests pass.
  - Test runtime is about 300 s, inside the 20-minute bound.
- **I-3 (under 20 min; new file only).**
  - Work: `pipelines/static_adapt/builders/sector_sparse_exact.py`, giving the
    lowest R eigenpairs of the sector block (vectors embedded in the full
    register) through the `exact_energy_impl` signature
    (`problem_registry.py:852-860`).
  - Tests:
    - equals dense to 1e-10 at 4 and 8 qubits;
    - a **12-qubit guard test** with `hamiltonian_matrix` patched to raise;
    - injected by `molecular_application.py` at every size.
  - Not written: `problem_setup.py`, `vqe_latex_python_pairs.py`.
- **I-4 (under 10 min).** `agent_guidance/static-adapt/AGENTS.md:79-80` only.

### 4.3 Paper II chain

- **II-1 (under 15 min; tests only).** This replaces r1 §6.2 increment 1,
  whose two numbers are identically 0 on the 4-qubit fixture (N3 and N1).
  - Test file `test/test_ap_mclachlan_molecular_runtime.py`, on the synthetic
    8-qubit system:
    - the static audit reports `requires_logical_shared` for 26 of 26
      generators;
    - common-angle seed parity in both modes to 1e-10;
    - the drive-rejection message;
    - the freeze pin (N1): over 11 checkpoints, energy change 0 to 1e-12 and
      residual ratio 1 to 1e-12;
    - the §1.3 table at fixed seeds: p in 2 ± 0.2 for random states, 4 ± 0.3
      for HF at θ̄ = 0, and leakage exactly 0 at 4 qubits;
    - a fixed-support moving run (support with even-Y words): logical-shared
      leakage ≤ 1e-30 asserted; per-word leakage recorded, with no threshold.
  - No uncapped generalized-exchange run (one exceeded 18 minutes).
  - Expected test runtime: under 60 s.
- **II-2 (under 20 min; gated: `hamiltonian.py` HIGH, `geometry_eval.py` HIGH,
  runner `:174`).**
  - Work: the defaulted provider attribute and method, the guarded geometry
    branch, and the runner setting (§3.2).
  - Tests:
    - b̄, K, f and V agree to 1e-12 with dense on the HH L=2, n_ph_max=1 seed
      and on the synthetic system;
    - `test_ap_mclachlan_route_parity.py` unchanged;
    - a **12-qubit guard test**: one geometry evaluation with
      `TimeDependentHamiltonian.matrix_at` and `hamiltonian_matrix` patched to
      raise, at most 8 runtime coordinates.
- **II-3a (under 20 min; gated: `run_lock.py`).** Seed contract and run-lock
  digest (r1 §6.2 increment 3), with the digest imported from S0.
- **II-3b (under 20 min; gated: runner).**
  - Work: the logical-shared fail-closed rule for seeds whose audit reports
    `requires_logical_shared_parameterization`, and the per-checkpoint
    population-leakage report with ceiling 1e-8 (`FixedCountSectorStateAuditor`).
  - Tests: per-word on a molecular seed fails closed; logical-shared passes and
    reports ≤ 1e-30.
- **II-4 (blocked).** The dipole drive (`drive_terms.py`,
  `problem_registry.py` `_MOLECULAR_CAPABILITIES`) and the even-Y pool
  composition, both waiting on A1 and GPT A2.

### 4.4 Paper III chain

- **III-1 (under 20 min; new file only).**
  - Work: `pipelines/qse_spectra/molecular_static_adapter.py`, holding:
    - the provider and the physics-key constructor;
    - the operator-content digest and the S0 integral digest;
    - three ground sources, with ⟨ψ|H|ψ⟩, residual and leakage recorded for
      each;
    - explicit R;
    - mandatory `compiled_costs` with `two_qubit_only_v1`;
    - commutator and Hermiticity assertions per record;
    - its own sector-index code at ≤ 12 qubits until I-3 lands.
  - Tests go in `test/test_paper_iii_molecular_static_adapter.py`. They
    replace r1 lines 588–591. On the 4-qubit symmetric H2 copy (sector
    dimension 4), with ψ the exact sector eigenvector at E_0, every record
    asserted to commute with N̂_alpha and N̂_beta, and E_k the same sector's
    eigenvalues counted with multiplicity:
    - with all three UCCSD generators: `retained_rank` 3,
      `overlap_condition_estimate` ≤ 1e2, and |θ_k − E_k| ≤ 1e-12, which is
      an equality because the support spans the complement;
    - with {sing(α), dbl}: θ_1 − E_1 ≥ 0.2;
    - with the HF root: θ_k ≥ E_{k−1} − 1e-10 and the fidelity bound (C)
      asserted, and the certificate reports certified while θ_3 < E_3. That
      pins the certificate's scope (§2 (ii)), not a defect to fix;
    - X records rejected by the commutator check;
    - a†_1a_0 rejected by the Hermiticity check before costing;
    - the as-written H2 payload rejected through S0.

    The synthetic 8-qubit repeat uses the §2 measurements.
- **III-2 (under 15 min; gated: `_resolve_problem` LOW).** One
  `kind == "molecular"` branch (`paper_iii_static_qse_cell.py:184-193`).
- **III-3 (blocked).** Observables and the scientific pool (A1, GPT A2).

### 4.5 U-16 — a named user-scope step (not dispatched)

One 16-qubit instance through the I-3 helper, and one through the II-2 option,
once the user brings sizes above 12 qubits into scope. Measured prototypes
(A3 refuter, random integrals, (4,4), sector dimension 4,900):
- sector-sparse ground energy: 14.4 s;
- JW Hamiltonian build: 18.4 s, 5,793 terms (mine: 12.5 s);
- UCCSD pool build: 1.6 s, 360 generators and 2,688 words.

A full-support 16-qubit tangent block would be 2.82 GB.

---

## 5. Reconciliation with the three analyses

**Accepted without disagreement** (each corrected statement is used in §1–§4):

- **A1:** findings 1, 3–9 and the measurements, with the refuter's
  corrections: the 56 projector-level directions; the mechanism split; the
  saddle and the 1/λ rate; that raising the ridge is not a general fix; the
  HH-only append guard; and the vacuous r1 increment.
- **A2:** findings 0–5 and 7–12, with the refuter's corrections: degenerate
  grounds allowed; ψ-leakage as a hypothesis; the certificate cannot verify
  index matching; the window gate; the reference-store caveat; the oracle's
  non-Hermitian price; and the 2×-dense memory note.
- **A3:** findings 0, 2, 5–10, 12–13, 15, 17–22 and 24–28, with the refuted
  items (1, 3, 4, 11, 14, 16) replaced by their corrected statements.

**Disagreements and refinements**, each with the line relied on:

| # | Item | My reading | Line relied on |
|---|---|---|---|
| D1 | A3 refuter V15: "the HF witness belongs in an implementation module, not `pipelines/contracts`" | I keep it in `pipelines/contracts/molecular_input.py` as a closed form on payload values. The refuter's objection is to an ⟨HF\|H\|HF⟩ check *through the JW build*, which would need `molecular_hamiltonian.py` and a statevector. The closed form needs neither. The JW equivalence is asserted in the test, which may import `src` | `pipelines/contracts/__init__.py:1-6` (pure validators allowed); `pipelines/contracts/problem.py:1-7` (excludes Hamiltonian construction, chemistry loaders and NumPy state construction, none of which the closed form performs) |
| D2 | A3 refuter V8: the minimal fix is a private helper in `problem_setup.py` | Not adopted. The coordinator ruled "move it to a new file", and the `exact_energy_impl` injection means `problem_setup.py` needs no edit | `problem_registry.py:804-860` |
| D3 | A3 refuter V0: both cap resolutions fit the memory rationale | Recorded. The coordinator ruled resolution (i) (§3.1) | coordination record lines 315–316 |
| D4 | My draft's tan²a bound | **I was loose.** For v ⊥ ψ, \|⟨E_0\|v⟩\|² ≤ ‖(1−\|ψ⟩⟨ψ\|)E_0‖² = 1 − f, so (C) holds with (1 − f), not (1 − f)/f. My numbers are recomputed: HF margin +0.149, not +0.410 | A2 F2; `qse_fidelity_bound_recheck.json` |
| D5 | The coordinator's premise "a diagnostic-fixture ground is not an exact one" | Not supported. The diagnostic fixture is the dense sector eigenvector (A2 F7, not refuted). The caveats are the unchecked HH reference store and SIAM nb2-only testing (refuter V7). My draft's "exact only to its eigensolver" stands, with those caveats | `paper_iii_static_qse_cell.py:208-217`; `paper_iii_problem.py:153-162,224-231` |
| D6 | The coordinator's summary "ridge 1e-3 brings it to 1.4e-14 — so the instability is numerical" | Refined per the A1 refuter. It is a saddle of the ridge-regularized *flow* (rate ∝ 1/λ), not an integrator artifact: λ = 0 gives 1.4e-25, and λ = 1e-3 still leaks 5.9e-3 with another H. Raising the ridge is not a fix; logical-shared is | `refute_dynamics.json`, `ridge_zero_check.json` |
| D7 | A3 F14 "no test imports another test module" | Refuted by V14, adopted: 7 such imports exist | `test/test_ra_adapt_refactor_parity.py:27` |
| D8 | A1 F5 "r1 line 258 is wrong" | Agreed on substance, refined per V5. Line 258 did say leakage must be reported; its error is deferring a choice the repository already makes (§1.5) | `sector_invariants.py:627-630` |
| D9 | A3 F1 "Paper II needs per-call threading" | Refuted by V1/V11, adopted: a provider attribute reaches all 9 call sites; only the runner construction site needs the setting | `ap_append_from_adapt_artifact.py:174` |
| D10 | A2 F7's "the diagnostic fixture carries the guarantee" read as "ψ = exact ⇒ index matching" | Refuted by V6 and adopted: exact ground is necessary, not sufficient (certified at θ_1 = E_3 and at θ_1 = E_2) | `method_completion.py:167-210` |
| D11 | My draft's sibling provider class (`matrix_free_hamiltonian.py`) | Replaced by the defaulted attribute that the coordinator and the A3 refuter (V11) specified. Both reach every call site; the attribute avoids a second provider type for the 20 importers | `hamiltonian.py:21-32` |
| D12 | My draft's S0 location `src/quantum/chemistry/` | Moved to `pipelines/contracts/` (A3 F15 / V15), avoiding the r1 scope-out dispute (A3 V8) | `src/quantum/AGENTS.md:31-37` |

My own measurements that the analyses did not contain, and that agree with
them:
- the UCCSD odd-Y count at 12 and 16 qubits;
- the `full_meta` order and the cap interaction (N1);
- the 18-minute uncapped exchange run;
- the 404-word fixed-support per-word run (4.1e-9 by t = 0.2);
- the 8-qubit QSE sector, HF and conditioning cases (§2).

---

## 6. Decisions for the user, and files

1. **Scope above 12 qubits.** Whether any molecule above 12 qubits (U-16 and
   beyond) is in Program A scope. The cap itself is decided on the Claude side
   as a memory limit (§3.1).
   - Recommendation: not now. Above 12 qubits Paper III also loses its only
     index and bound guard.
2. **Cross-lane contract corrections.** Authorize correcting the stale facade
   sentence in `agent_guidance/molecular-vibronic/AGENTS.md:22-25` (Paper IV
   lane) and in the root `AGENTS.md:94-96`. Neither is written by any lane
   handoff.
   - Recommendation: yes, citing `engine.py:5742-5755`.

**Not the user's:**
- the Paper II coordinates: an existing repository rule (§1.5);
- the even-Y pool composition, and pool order against the structural cap:
  GPT A2;
- the H2 fixture: kept as a negative control; positive tests use an in-memory
  symmetric copy.

**Files to edit** (r2 itself: this file only).

By Codex, per §4:
- **New:**
  - `pipelines/contracts/molecular_input.py`
  - `test_support/molecular_closed_shell_fixture_factory.py`
  - `pipelines/static_adapt/ra_adapt/molecular_application.py`
  - `pipelines/static_adapt/builders/sector_sparse_exact.py`
  - `pipelines/qse_spectra/molecular_static_adapter.py`
  - `test/test_molecular_input_contract.py`
  - `test/test_ra_adapt_problem_family_admissibility.py`
  - `test/test_ap_mclachlan_molecular_runtime.py`
  - `test/test_paper_iii_molecular_static_adapter.py`
- **Gated:**
  - `pipelines/static_adapt/ra_adapt/engine.py`
  - `pipelines/static_adapt/ra_adapt/contracts.py`
  - `pipelines/time_dynamics/ap_mclachlan/hamiltonian.py`
  - `pipelines/time_dynamics/ap_mclachlan/geometry_eval.py`
  - `pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py`
  - `pipelines/time_dynamics/run_lock.py`
  - `pipelines/exact_bench/paper_iii_static_qse_cell.py`
  - `agent_guidance/static-adapt/AGENTS.md`
- **Blocked:**
  - `pipelines/time_dynamics/adapters/drive_terms.py`
  - `pipelines/static_adapt/builders/problem_registry.py`

## 7. Corrections to the coordination record (`CODEX-COORDINATION-20261010.md`)

**Lines 344–346.**
- *Present text:* "every failure is pre-recorded in … One of them, the facade
  digest, is a stale source lock".
- *What is wrong:* the facade red is not one of the counted failures.
  - The counted baselines' 7 failures are:
    - Paper I: `test_hamiltonian_adapter_capabilities.py` × 2
      (`FULL_SUITE_BASELINE_20260831_failed.txt:356-357`);
    - Paper II: `test_realtime_drive_terms.py` × 3 (`:575-577`) and
      `test_scaffold_runtime_loader.py` × 2 (`:591-592`).
  - The facade red (`:547`, `test_ra_adapt_facade.py:2009`) sits in a file
    that no counted batch runs.
- *Corrected sentence:* "Every failure in these batches is pre-recorded in
  `docs/reports/FULL_SUITE_BASELINE_20260831_failed.txt`:
  - Paper I: `test_hamiltonian_adapter_capabilities.py` twice (`:356-357`);
  - Paper II: `test_realtime_drive_terms.py` three times (`:575-577`) and
    `test_scaffold_runtime_loader.py` twice (`:591-592`).

  Separately, the facade suite `test/test_ra_adapt_facade.py`, which none of
  these batches runs, has one known red, the frozen ordinary-v1 route digest
  (`:547`). It is a stale source lock whose re-pin is the user's decision.
  Measured at `a23ff642`: 66 passed and 1 failed of 67, in 200 s."

**Lines 349–351.**
- *Present text:* "each first increment a tests-only pin under 15 minutes that
  records the current Hubbard-Holstein protocol and route digests".
- *What is wrong:*
  - Only Paper I's increment 1 fits that description.
  - Paper II's increment 1 is a tests-only characterization of the molecular
    runtime path. It records no HH digest, and as written in r1 its two
    numbers are identically 0.
  - Paper III's increment 1 writes a source module, is bounded at 20 minutes,
    and records no digest.
  - Under r2, the first increment of all is S0.
- *Corrected sentence:* "Dispatch: revision 2
  (`…-20261010-r2.md`, §4) supersedes the reply's section 6 increments.
  - The first increment is S0, closed-shell input validation and provenance
    (new files only, under 15 minutes), on the Paper I branch. Papers II and
    III branch from the commit that lands S0.
  - Paper I's increment 1 is a tests-only pin, under 15 minutes, of the
    Hubbard–Holstein protocol and route digests (`fc25b5c2…`, `0312875b…` at
    `0de9685a`).
  - Paper II's increment 1 is a tests-only characterization of the molecular
    runtime path on the synthetic 8-qubit system.
  - Paper III's increment 1 adds the provider module (under 20 minutes)."

**Also stale in the same record** (A3, not refuted):
- Lines 328–329: "added after the existing branches". It should read "added
  behind a default-false gate (false for every existing request)".
- Lines 351–352: "Papers II and III follow once the Paper I seam is in". This
  is covered by the corrected dispatch sentence above.
- Lines 315–317: "until the matrix-free increments land". It should read "the
  cap is a memory limit decided on the Claude side; running above 12 qubits
  is the user's scope".

## 8. Unmeasured

- 16-qubit runtimes through the to-be-written helper and option. Only
  prototypes exist (§4.5).
- Peak memory and time of a 16-qubit matrix-free geometry evaluation.
- Paper III's 12-qubit full-register reference build: time and peak copies
  (`paper_iii_qse_regime_frontier_sweep.py:103-110`, 4,096 columns).
- Whether Paper I builds the exact reference state on every molecular run at
  12 qubits (`problem_setup.py:983` admits dimension 4,096). By source search
  it is not reached: the SR runtime's only caller,
  `_resolve_required_exact_initial_state` (`adapt_pipeline.py:30457-30469`),
  has no non-test caller. No run has measured it.
- Per-word leakage at 12 qubits and with real molecular integrals.
- The split of case C's 3.9e-1 between the two mechanisms.
- Stability of per-word case B beyond t = 2.
- Logical-shared leakage when grouped candidates have non-commuting words.
- The generic seed-loader `theta_logical` path, which was read from code and
  not run.
- Whether the SR runtime takes its pool from the adapter or rebuilds it from
  execution settings (an I-2a stop condition).
- Whether `_MOLECULAR_CAPABILITIES` enters the Paper I protocol digest or the
  Paper III physics key.
- Whether I-2b fits 20 minutes of agent wall time.
- Runtimes of every new test.
- The Ky Fan selector on a molecular pool, and the effect of the
  selector-versus-solver cutoff mismatch (s_max < 1e-2).
- A degenerate molecular sector ground (tested only on a random
  8-dimensional Hermitian).
- Shot-noise (estimated) pencils.
- S² assignment of roots beyond H2.
- The H2 writer against a reference FCI: no pyscf, psi4 or openfermion in the
  venv.
- My uncapped generalized-exchange run, stopped at 18 minutes.
