# Disposition of review 026 and reconciliation 023 against chemistry r2

*Revision 2026-10-18* (`chemistry-disposition-scope-20261018-002`): two
statements in row 026-1 qualified in place, with one consequential §3 row. The
committed version `5554f2c0` (SHA-256 `516b4ff0…`, 547 lines) stays in
history. The final section records this revision's disposition.

Author: the existing Claude chemistry architecture author (r1, r2). Requested
by Codex coordinator message `codex-to-claude-20261010-013`, paragraphs 2–3.

**Text being amended.** r2 at commit `c943ae8a`, file
`agent_guidance/paper-v/handoffs/claude-architecture-chemistry-papers-i-iii-20261010-r2.md`
(1,062 lines). Its SHA-256 is
`f609dfd8e4648b27a82095f13d26f10bfbd5cfd330ff2e3eb85f6b33289114d0`, verified
both in the working file and via `git show c943ae8a:<path>`. Every "r2 line"
below refers to those bytes.

**Inputs** (all verified with `shasum -a 256` before reading):

| Input | SHA-256 |
|---|---|
| Root review 026, `/Users/lordskyler/Documents/ChatGPT/Holstein/campaign-20261010/chemistry-committed-r2-review-20261011-026.md` | `53beb3c8da105d2152fca51d2f04f493e2247efb7170b3285346747f37a658ed` |
| Codex reconciliation 023, `agent_guidance/shared/handoffs/chemistry-r2-execution-reconciliation-20261010.md` (Codex worktree, commit `90e9f9e4`, 395 lines) | `b605ba975efb36ee75b40699eecd1d4d98f3a390452434f7556da9478ffd278c` |
| Mathematical contract (commit `07346ca3`) | `5addf0334832aa664cf402969686361e51d8bd467d197727b6a4690b0d8815f7` |
| Increment A report (commit `393e335a`) | `37bc322ddad6115eda5b84746d7462fc909310b877041741c4f23d47bbf7a097` |
| Release 021 | `0e8a7892462cc869e4946aede8d877fd57f36af1fb2b7cbcd9acd3d55c8a20ac` |
| Message 013 | `b078b9a4df55ceb73e2c5ed3e8f7b695b89fd774ba7b2856cda045bedbcaea73` |

The companion review of A is
`agent_guidance/paper-v/handoffs/REVIEW-chemistry-strict-intake-claude-20261010.md`.

**Note on 023's line numbers.** 023 reviewed the earlier untracked 620-line
draft (`3c02b918…`), not the committed r2, so its r2 line numbers refer to
that draft. Where the committed r2 already made a 023 correction, the row says
so.

**Dispositions.** Every finding is accepted: 13 outright and 2 with an
amendment. None is contested. **I do not write an r3** (§4): this file, applied
to the committed r2 bytes, is the current architecture.

---

## 1. Review 026 (seven findings, finding 7 split into five rows)

### 026-1 · Operator commutator counts

- **Disposition:** accept with amendment.
- **Finding.** The 46 and 62 counts cannot be counts of individual Pauli words.
  A word that flips f > 0 qubits has N̂P|vac⟩ = fP|vac⟩ but PN̂|vac⟩ = 0, so it
  does not commute with N̂. The 56 projector-safe count needs a source/mask
  inventory.
- **r2 lines changed:** 127–134 (§1.1 quoted block); 890–891 (§5, "the 56
  projector-level directions").
- **Replacement for lines 127–134:**

> Paper II's default coordinates are one per Pauli word
> (`ap_append_from_adapt_artifact.py:155`; CLI default `:1772`; fixed runner
> `ap_fixed_from_adapt_artifact.py:63`), and they span a larger manifold.
>
> **No individual word with X/Y support commutes with the total count N̂, and
> none commutes simultaneously with both N̂_alpha and N̂_beta** (026; qualified by
> `chemistry-disposition-scope-20261018-002`).
>
> - A word that flips f > 0 qubits has N̂P|vac⟩ = fP|vac⟩ while PN̂|vac⟩ = 0.
> - If it flips qubits of both species, the same argument, applied to each
>   species' vacuum, gives [P, N̂_alpha] ≠ 0 and [P, N̂_beta] ≠ 0.
> - A word acting on one species only, P = P_alpha ⊗ I_beta, **does** commute
>   with the unaffected count: [P, N̂_beta] = 0. Every word of an alpha single
>   and of the αα double is of this kind.
>
> The counts that follow are dimensions over a restricted domain. For each
> generator k, take the m_k words of that generator. They commute mutually and
> share one X/Y flip mask. The domain is the real per-word variations
> δ ∈ R^{m_k}, i.e. the block combinations D_k = Σ_j δ_j P_{k,j}, restricted to
> those with the stated commutator equal to zero. The table sums these over the
> 26 generators (160 words) at 4 spatial orbitals, (2,2):
>
> | Generator kind (count × words) | [D, N̂_α] = [D, N̂_β] = 0 | [D, N̂] = 0 | [D, Π_(2,2)] = 0 | single words with [P, Π_(2,2)] = 0 |
> |---|---|---|---|---|
> | single α (4 × 2) | 1 each | 1 | 1 | 0 |
> | single β (4 × 2) | 1 each | 1 | 1 | 0 |
> | double αα (1 × 8) | 3 | 3 | 8 | 8 |
> | double ββ (1 × 8) | 3 | 3 | 8 | 8 |
> | double αβ (16 × 8) | 2 each | 3 | 2 | 0 |
> | **total (26 generators, 160 words)** | **46** | **62** | **56** | **16** |
>
> Here Π_(2,2) is the full-register projector onto the (N_alpha, N_beta) = (2,2)
> sector.
>
> - Each generator's common-angle direction c_k lies in its
>   (N̂_α, N̂_β)-commuting subspace (residual 0.0).
> - The 16 projector-commuting single words are exactly the words of the two
>   same-spin doubles. Their flip mask covers all four orbitals of one spin
>   block, which maps N_σ = 2 to 2 and N_σ ≠ 2 to a value ≠ 2. A single or an αβ
>   double flips two orbitals of a block, which moves some sector states out.
> - At 2 spatial orbitals, (1,1), the totals are 4, 5 and 12; all 12 words
>   commute with Π_(1,1).
> - **Source and status.** The totals 46, 62 and 56 (and 4, 5 and 12 at
>   4 qubits) are **reported numerical nullities of full-register operator
>   matrices**, taken over the generator-internal domains above at declared
>   NumPy rank thresholds.
>   - Full-register numerical scope is not an exact or certified rank proof.
>   - No exact-rank computation was made, and none is requested.
>   - **Operator level (46, 62):** A1
>     `output/local_runs/paper_v_chem_verify_20261010/A1-perword-leakage/leakage_measure.py:158-167`
>     (SHA-256 `de95ed8a2858ed749e9dcb5ac1a2328afc3fbdd2e295392c225871ca0acfc322`,
>     verified). It takes a NumPy SVD of the stacked commutator columns with rank
>     threshold `1e-9 * max(1, s.max())`.
>   - **All three (46, 62, 56):** A1 refuter
>     `output/local_runs/paper_v_chem_verify_20261010/A1-perword-leakage-refute/refute_static.py:161-175`
>     (SHA-256 `ea52cc31a06de4218b598a2a4948d94b67aa9c4c201aee583d1a85de9543be10`,
>     verified). It uses `numpy.linalg.matrix_rank(..., tol=1e-9)`; the projector
>     level uses the full-register 0/1 sector projector (`:158`).
>   - **The single-word count 16 (12 at 4 qubits) is a separate, discrete
>     count.** A word counts when every in-sector basis index XOR its flip mask
>     stays in the sector (`refute_static.py:152-154`, key
>     `words_mapping_sector_into_itself`). It is not a nullity.
>   - Per-kind output: `refute_static.json` key `conserving_dims_by_kind`.
>   - I read these artifacts and verified the two script hashes; I did not
>     rerun them.
> - These dimensions describe which per-word deviations leak. They support no
>   admission.

- **Replacement for lines 890–891:** "the 56 projector-level dimensions (16 of
  them single words) with the domain stated in §1.1".

### 026-2 · A is the sole selected strict intake

- **Disposition:** accept.
- **r2 lines changed:**
  - 581 (§3.3 digest owner)
  - 626–629 (§4 order "S0 first")
  - 641–741 (§4.1 S0, entire)
  - 757 and 765 (I-2a "S0 receipt", "rejected through S0")
  - 828–829 (II-3a "digest imported from S0")
  - 845 and 866 (III-1 "S0 integral digest", "rejected through S0")
  - 478 ((v) gate)
  - 905 and 916 (§5 D1, D12)
  - 952–953 and 957 (§6 new files)
  - 1008–1013 (§7 dispatch sentence)
- **Replacement for lines 641–741:**

> §4.1 is withdrawn in full. S0 is recorded as a proposal superseded by
> increment A: `resolve_molecular_diagnostic_intake`
> (`pipelines/static_adapt/builders/molecular_diagnostic_intake.py:168`,
> commit `393e335a`) is the sole selected diagnostic intake.
>
> A is not a parallel implementation, and no second validator is proposed. A
> mapping-only validator cannot establish the consumed-byte hash; that requires
> A's byte-consumption interface (one read, expected SHA-256, `:192-195`).
>
> What S0 asked for and A does not do is a candidate for a separately
> versioned extension of A's own boundary (a receipt schema after
> `molecular_diagnostic_intake_v1`), unselected:
> - required provenance fields: geometry units, SCF type, point group, and
>   active/frozen/deleted orbital lists;
> - named reason codes;
> - separate symmetry and HF tolerances;
> - a physics digest distinct from A's tensor and semantic digests.
>
> A's single-snapshot read, strict rejection, raw-byte and operator binding,
> and ordinary-resolver semantics are preserved. A's witness remains distinct
> from downstream enforcement.

- **Replacement for every other listed line:**
  - "S0 receipt/digest" reads: "A's receipt
    `runtime_data["molecular_diagnostic_intake_receipt"]`, with its
    `raw_sha256`, `tensor_sha256`, `semantic_sha256` and `operator_sha256`".
  - "rejected through S0" reads: "rejected by A".
  - Lines 626–629 read: "No increment is selected. A is complete. Any next
    release names its own order".
  - §5 D1 and D12 are withdrawn as moot.
  - The seeded random-integral generator
    (`test_support/molecular_closed_shell_fixture_factory.py`) is no longer an
    S0 file. It stays an optional test helper for a later release, unselected.

### 026-3 · Action parity and interface gates before B

- **Disposition:** accept.
- **Finding.** `compile_polynomial_action` merges equal labels and only then
  drops |c| ≤ 1e-15 (`src/quantum/compiled_polynomial.py:55-69`, verified).
  `hamiltonian_matrix` drops each raw term with |c| < 1e-12 before summing
  (`src/quantum/vqe_latex_python_pairs.py:2300-2304`, verified). The two
  semantics differ even with no time dependence.
- **r2 lines changed:**
  - 518–520 (§3.1 Paper II bullet)
  - 566–568 (§3.2 rows for `hamiltonian.py`, `geometry_eval.py`, runner)
  - 817–827 (II-2)
  - 913 and 915 (§5 D9, D11)
  - 964–966 (§6 gated files)
- **Replacement:**

> r2's design is withdrawn: a defaulted `TimeDependentHamiltonian` attribute,
> an additive method through `compile_polynomial_action`, and a runner setting
> at `:174`. It is replaced by 023's proposed interface, which is likewise
> unselected:
>
> ```python
> # new pipelines/time_dynamics/ap_mclachlan/matrix_free_hamiltonian.py
> class MatrixFreeHamiltonian:
>     @classmethod
>     def from_provider(cls, provider: TimeDependentHamiltonian, *, nq: int): ...
>     @property
>     def drive_enabled(self) -> bool: ...
>     def apply_at(self, time: float, psi: np.ndarray) -> np.ndarray: ...
> # evaluate_mclachlan_geometry gains hamiltonian_action_mode="dense" (default);
> # the only other mode, "compiled_pauli_diagnostic_v1", requires that sibling explicitly.
> ```
>
> **Write scope.** Exactly three paths: the new module, `geometry_eval.py`, and
> the new `test/test_ap_mclachlan_hamiltonian_action.py`. No `hasattr`
> discovery, no runner or CLI switch, no lock or default change.
>
> **Semantics** (023):
> - Target cached `matrix_at` (`hamiltonian.py:50-69,89-118`).
> - Apply the raw cutoff before merging: discard |c| < 1e-12 and keep
>   equality, per raw term, with duplicates and order preserved. Read-only
>   Pauli primitives (`src/quantum/pauli_actions.py:232`) may be reused;
>   `compile_polynomial_action` is not the semantic boundary.
> - Take an immutable owned snapshot of the ordered raw static records and the
>   **unscaled** drive records: words, finite coefficients, explicit `nq`. A
>   later mutation must not change the action.
> - Compute A_static ψ + c(t)·A_drive ψ, with no rethresholding of scaled drive
>   terms.
> - Keep identity and scalar terms; do not normalize Hψ.
> - Validate time, envelope, coefficients and vector sizes.
>
> **Gates.**
> - Cold-cache test: `TimeDependentHamiltonian.matrix_at` and
>   `hamiltonian_matrix` patched to raise.
> - Dense parity of h_ψ, E, K, f and V, with unchanged tangents.
> - Selector omitted → dense path and metadata unchanged; unknown mode fails;
>   wrong sibling type fails.
>
> **Open, unselected:** runner exposure (`ap_append_from_adapt_artifact.py:174`),
> run-lock and artifact propagation, and any full AP run. r2's D9 observation
> stands only as the open question for runner exposure: a call keyword reaches
> an AP run only by being threaded through the 9 `evaluate_mclachlan_geometry`
> call sites. It is not an objection to the diagnostic.

### 026-4 · Conservation of the executed unitary

- **Disposition:** accept.
- **r2 lines changed:**
  - 224–258 (§1.5 heading, audit paragraph, gate items 1–2)
  - 285–291 (replacement rows)
  - 813–814 (II-1 even-Y run)
  - 830–835 (II-3b)
  - 942 (§6 "an existing repository rule")
- **Replacement for lines 224–258:**

> **1.5 What the repository already decides, and what it does not**
>
> **The admissibility predicate.** Admissibility of an executed generator is
> `execution_preserves_fixed_counts` for its declared execution mode
> (`pipelines/static_adapt/sector_invariants.py:613-626`, reported at `:651`).
> It is not `requires_logical_shared_parameterization` (`:627-630`, reported at
> `:652`). The latter says that per-word coordinates would break a conserving
> generator; it does not say that shared coordinates conserve.
>
> **The audit condition is sufficient, not necessary.** For `termwise_product`,
> the audit requires that the components conserve individually, or that the
> sum conserves and all components mutually commute. 023's pair-ordered product
> conserves although its components do not all commute.
>
> **UCCSD.** Both flags are True for spin-orbital UCCSD, and logical-shared
> coordinates conserve (measured ≤ 4.9e-32 at 8 qubits; r2 §1.4).
>
> **Even-Y aggregates.** For an aggregate such as
> `ham_block::molecular_one_body` executed `termwise_product` under one shared
> angle, conservation is not established. 023's six-qubit witness: H = J01 + J12
> on index 9 at θ = 0.4, executed in the interleaved order
> [eeeexx, eeexxe, eeeeyy, eeeyye] with each coefficient +1/2, leaked
> 2.3624146274315547e-4 into index 15 (amplitude −0.015370148429444508i). The
> pair order [eeeexx, eeeeyy, eeexxe, eeeyye] leaked nothing.
>
> **The one family with a finite-product proof** is 023's intact same-spin
> Hermitian hopping pair:
> J_pq = a_p†a_q + a_q†a_p = ½(X_p Z… X_q + Y_p Z… Y_q), p > q.
> - Its two words commute and both have even Y parity.
> - [J_pq, N̂_α] = [J_pq, N̂_β] = 0.
> - With one shared coordinate per pair and the coefficient signs preserved,
>   the executed product is exactly exp(−iθJ_pq).
> - This is a mathematical candidate, not a pool or default.
>
> **What existing guards do not establish.**
> - The SIAM runner guard (`ap_append_from_adapt_artifact.py:2478-2481`) is
>   family-specific and is not a molecular guard.
> - The static lane's resolution (`adapt_pipeline.py:12457-12463`) is not
>   reached by Paper II.
> - Checkpoint leakage monitoring is a diagnostic. It proves nothing between
>   checkpoints and does not restore a sector-preserving ansatz.
>
> **Unselected:** a molecular monitoring policy, pools, and lane admission.

- **Replacement for lines 285–291:**
  - Line 258 row: "One per Pauli word by default. Molecular UCCSD needs shared
    coordinates; admissibility is `execution_preserves_fixed_counts` per
    executed generator. Per-word leaked up to 8.4e-2 (fixed support, A1 case C)
    and 3.9e-1 (exchange route) at 8 qubits, against ≤ 4.9e-32 for
    logical-shared UCCSD. A UCCSD-only pool freezes the flow (N1)."
  - Line 323 row: "Absent for spin-orbital UCCSD under logical-shared
    coordinates. Not established for noncommuting shared aggregates (023's
    six-qubit witness)."
- **Replacement for lines 813–814:** "a fixed-support run whose even-Y part is
  intact hopping pairs J_pq. It first asserts `execution_preserves_fixed_counts`
  for every executed generator, then asserts logical-shared leakage ≤ 1e-30.
  No `ham_block` aggregate under shared coordinates."
- **Replacement for lines 830–835:** "fail closed unless every executed
  generator reports `execution_preserves_fixed_counts` for the requested
  parameterization (a generator reporting `requires_logical_shared_parameterization`
  fails under per-word coordinates). The per-checkpoint population-leakage
  report is a diagnostic and never a guard on dynamics between checkpoints."
- **Replacement for line 942:** "the Paper II coordinates follow the sector
  audit (§1.5); the even-Y generator family is GPT A2's".

### 026-5 · Ritz lower bounds versus root identification and completion

- **Disposition:** accept.
- **r2 lines changed:** 345–348 ((ii) index matching); 361–364 (certificate);
  395–396 ((ii) gate); 914 (§5 D10).
- **Replacement for lines 345–348:**

> *Index matching.* Let ψ be the exact sector ground, and let the retained
> space V ⊂ S ∩ ψ^⊥ have rank r. Then θ_k ≥ E_k for k = 1…r (one-based θ,
> zero-based E), **whether or not V contains E_1…E_{k−1}**.
>
> Containing the lower eigenvectors matters only for equality and ordered
> accuracy, i.e. for reading θ_k as an approximation to E_k. It does not matter
> for the bound. A support that misses lower roots gives θ_k far above E_k.
> Measured on 4 qubits: θ_1 = E_3 with the pool {Gd}. That is an accuracy
> failure, not a bound violation.

- **Added after line 364** (restored from 023):

> **Physical residual.** For normalized Ψ ⊥ ψ,
> ‖(H − θ)Ψ‖² = ‖Q0(H − θ)Ψ‖² + |⟨ψ, HΨ⟩|².
> The completion and selection diagnostics project with Q0
> (`method_completion.py:121`, `record_selection.py:605`), so they omit the
> second term. That term is bounded by the seed's full residual, but it is not
> zero for an approximate seed.
>
> **Joint window.** The completion selector tests unused records one at a time
> (`method_completion.py:164`). 023's counterexample: H = 0 ⊕ 2 ⊕ [[3, −2], [−2, 3]],
> with e1 selected and e2, e3 unused. Each unused record passes the individual
> pressure test at energy 3, while their joint root is 1 < 2.
>
> **Scope of the flag.** A small projected residual bounds the distance to an
> eigenvalue of the compression. `target_window_certified` does not certify
> the lowest joint physical window.

- **Replacement for lines 395–396:** "ordered accuracy (θ_k as an
  approximation to E_k) is checked offline against the dense sector spectrum,
  as a reporting diagnostic. It is never inferred from the certificate and
  never enters selection."
- **Replacement for line 914 (D10):** "Exact ground: necessary for (B), not
  sufficient for ordered accuracy. The certificate passed at θ_1 = E_3 and at
  θ_1 = E_2; this does not restrict the bound."

### 026-6 · Qualify numerical spectral bounds and bind their inputs

- **Disposition:** accept.
- **r2 lines changed:** 316; 391–392; 424–427; 446–457 ((iv)); 476–479 ((v)
  gate).
- **Replacement for line 316:** "Each is an exact-arithmetic statement. The
  observed floating-point deviation grows with κ(S_retained) (§2 (iii)). The
  estimate ε_mach·κ·‖H‖ is a heuristic with no specified backward-error
  constants; it is not a certified spectral enclosure."
- **Replacement for lines 391–392:**

> The molecular provider's diagnostic ground is a **numerical** sector
> eigenvector, recorded with its residual ‖(H − E_0)ψ‖ and its eigensolver.
> Treating it as the exact ground is a recorded assumption, distinct from an
> externally certified exact ground.
>
> A ground-angle or fidelity claim for any other ψ needs independently
> qualified ground identification, fidelity and separation provenance: values
> E1_lower, ρ_upper and η_upper with δ_lower = E1_lower − ρ_upper > 0, giving
> tan α ≤ η_upper/δ_lower (023). Without them the angle is recorded as
> unknown. A ground gap alone, or an excited Ritz upper value, is not
> E1_lower.

- **Replacement for lines 424–427:** "*Heuristic test guard, not a
  certificate.* A bound test at tolerance τ also bounds κ; the provider tests
  at 1e-12 use κ ≤ 1e2. Motivation, 4 qubits: −2.08e-9 at κ 1.05e8 and
  −1.83e-7 at κ 7.31e9."
- **Replacement for lines 446–457** (adds acquisition cost):

> K_i prices exp(−iθB_i) as a unitary tail insertion
> (`compiled_costs.py:64,81`). It is not the cost of acquiring the projected
> moments (b_i = ⟨B_i⟩):
> - S_ij = ⟨B_i†B_j⟩ − b_i*b_j;
> - H_ij = ⟨B_i†HB_j⟩ − b_i*⟨HB_j⟩ − ⟨B_i†H⟩b_j + b_i*b_j⟨H⟩.
>
> A measured-cost claim must enumerate those moments, with grouping, settings,
> precision and shot policy (023). None is implemented. The coefficient- and
> phase-blind raw-word catalog (`paper_iii_qse_measurement_cost.py:20,156`)
> omits the ⟨B_i⟩ information: 023's B = X, H = Z case needs ⟨X⟩.
>
> Hermitian B_i is a proposed molecular-provider restriction, not a
> requirement of the pencil (`core.py:466-473`). The oracle silently prices a
> non-Hermitian record (8.0 against 4.0 two-qubit gates, 4 qubits), and that
> price has no circuit meaning.

- **Replacement for lines 476–479:** "the molecular provider binds, by content
  and not by names or an ordered-name digest:
  - the normalized reference amplitudes, with an explicit phase and
    normalization policy;
  - the ordered operator widths, words and complex coefficients;
  - the H identity;
  - record cleanup (`core.py:390`), Q0 projection, column normalization,
    overlap cutoff and retained rank;
  - the actual cost policy;
  - R declared per molecule;
  - A's integral digests;
  - any reporting reference, separately, with a reporting-only role (023)."

### 026-7a · `exact_energy_impl` is scalar

- **Disposition:** accept.
- **r2 lines changed:** 582 (§3.3); 787–791 (I-3).
- **Replacement:** "Two APIs:
  1. a new eigenpair helper returning the lowest R eigenvalues and
     full-register embedded eigenvectors of the (N_α, N_β) block;
  2. a scalar wrapper with the `exact_energy_impl` signature, returning one
     float. The resolver coerces the result to float
     (`problem_registry.py:852-860`).

  The scalar contract is not widened. Both are unselected."

### 026-7b · Dense construction is distinct from diagonalization

- **Disposition:** accept.
- **r2 lines changed:** 540–543; 565 ("removes the 256 s dense build").
- **Replacement:** "The 256.0 s was measured on `exact_ground_energy_sector` at
  12 qubits. That function builds the full-register dense matrix and then
  diagonalizes the 400-dimensional sector block; the split between the two was
  not measured. Paper II's first geometry evaluation pays the dense
  construction (`hamiltonian.py:50-56`), not a sector eigensolve."

### 026-7c · A family predicate is not proof of a default-false gate

- **Disposition:** accept.
- **r2 lines changed:** 549–552 (§3.2); 1022–1023 (§7).
- **Replacement:** "Each edited existing function gains a branch entered only
  on an explicit opt-in identity: the named molecular adapter's `adapter_id`
  and its application lane, together with the family key. The family key alone
  does not establish that the branch is unreachable for existing requests.
  Legacy validation runs unchanged. For existing requests, only what the pins
  test is established: the facade suite (66/1), the parity file and the I-1
  digests."

### 026-7d · The 12-qubit scope is not a memory cap or a selection

- **Disposition:** accept.
- **r2 lines changed:** 500–503; 505–512; 873–882 (§4.5 U-16); 930–932;
  1026–1028.
- **Replacement:**

> The 12-qubit limit is a scope statement for r2's proposals. It coincides
> with the cap A implements for its diagnostic
> (`molecular_diagnostic_intake.py:213`, 2m ≤ 12).
>
> - It is not a campaign memory cap, and it selects no implementation or
>   execution.
> - The 18 GB figure is historical planning context
>   (`agent_guidance/shared/memory-budget.md:15`), not a mandatory cap.
> - r2's 16-qubit exceptions are removed: §4.5 U-16 is withdrawn, its prototype
>   timings included, as a plan.
> - Whether anything above 12 qubits enters Program A scope remains the user's
>   question.

### 026-7e · A symmetrized H2 copy is an artificial fixture

- **Disposition:** accept.
- **r2 lines changed:** 87–91 (N2); 349; 482; 579; 719–721; 852–853; 945–946.
- **Replacement:** "Any symmetrized copy of the H2 writer ((00|11) := (11|00))
  is an **artificial diagnostic fixture**. Its agreement with an independent
  Slater–Condon build checks only internal tensor and JW consistency; it does
  not establish corrected physical H2 data. The symmetrized copy is withdrawn
  as a positive fixture. Positive tests use A's synthetic fixture (contract
  §5.4, independently rechecked in the companion review §3) or seeded synthetic
  integrals. The 4-qubit numbers quoted at lines 349 and 482 are labelled
  'artificial symmetrized H2 copy'."

---

## 2. Reconciliation 023 (eight findings)

**How the eight were enumerated.** Seven are the rows of 023's own disposition
table ("r2 location | Reconciled scope"). The eighth is its completion-limit
material in "Ground identification, Ritz claims and completion limits".

| # | 023 finding | Disposition | r2 line(s) | Replacement text |
|---|---|---|---|---|
| 023-1 | N1's frozen flow is conditional on a real prepared state, a real H and an odd-Y-only executed tangent manifold | accept with amendment | 56–62 | "N1 holds under three conditions: (1) a real prepared state; (2) a real Hamiltonian with real Pauli coefficients and real parameters; (3) executed generators whose words all have odd Y parity. Then every executed exponential is real orthogonal, the actual tangents are real, b̄ = −i(H−E)ψ is imaginary and f = Re(T†b̄) = 0. The existing homogeneous ridge/inverse policy keeps θ̇ = 0. Spin-orbital UCCSD meets condition (3): odd Y in 12/12, 160/160, 828/828 and 2,688/2,688 words at 4/8/12/16 qubits. A complex state, a complex H or another tangent manifold falls outside the argument. Even-Y directions can leave the real manifold (023's synthetic H = J01 on index 9 gives K = f = 1), but neither adequate dynamics nor sector safety follows from adding them." |
| 023-2 | Eight qubits is the smallest *tested* per-word leakage witness, not the smallest possible; there is a six-qubit witness | accept | 96–99 (N3); the committed r2 makes no "smallest" claim | Append to N3: "Six qubits suffice. At 3 spatial orbitals, (1,1), take ψ = \|001100⟩ (index 12: alpha orbital 2, beta orbital 0). Each word of the alpha 0→1 single G01 = −½ eeeexy + ½ eeeeyx maps index 12 to i × index 15 = \|001111⟩, which is in sector (3,1). An independently varied word leaks sin²(θ/2); the tied pair cancels on this ψ (023, 'Six-qubit leakage'). The 4-qubit (1,1) result is a property of that fixture, not a conservation theorem." |
| 023-3 | Shared coordinates need a proof for the finite executed product; structural eligibility is insufficient. Intact hopping pairs are the candidate; an interleaved J01 + J12 product is the counterexample | accept | 224–258, 285–291, 813–814, 830–835 | As 026-4 above |
| 023-4 | A ground angle from a residual needs independently qualified separation (δ_lower = E1_lower − ρ_upper > 0; tan α ≤ η_upper/δ_lower); the sin² bound replaces tan² | accept with amendment. The committed r2 already replaced tan² with (1 − f) (§5 D4) and removed the unspecified-gap residual-angle claim | 313–314 ((C)); 391–392 | (C): "If ψ ∈ S has ground fidelity f, then θ_k ≥ E_k − (1 − f)(E_k − E_0), with sin²α = 1 − f. Using it requires f, or a bound on it, from an independent qualification: tan α ≤ η_upper/δ_lower with δ_lower > 0. The Eckart form θ_k ≥ E_k − ΔE_ref(E_k − E_0)/(E_1 − E_0) uses the exact E_0 and E_1 and is a reporting relation, not an online certificate. 023's gates are the test cases: H = diag(0,2) with ψ = (√3/2, ½), saturating sin² at 3/2; ψ = (0,1), where admission fails; H = diag(0,1,10), where the excited Ritz value is rejected as E1_lower." Lines 391–392: as 026-6 |
| 023-5 | Retained-rank and nesting, projected acquisition and content binding qualify the spectral and cost claims | accept | 430–433 (monotonicity); 446–457; 476–479 | Monotonicity, append: "023's counterexample: H = diag(0,1,2,3), ψ = e0, basis [e1] against [e1, 10⁶e2]. The relative overlap cutoff discards e1, and the retained root moves from 1 to 2. Record the actual overlap cutoff, normalization policy and retained rank, not only the record count." Acquisition and content: as 026-6 |
| 023-6 | Sibling action ownership is a future proposal; implicit `hasattr` dispatch is rejected | accept | 518–520, 566–568, 817–827, 913, 915, 964–966 | As 026-3 above |
| 023-7 | Existing A owns intake; richer S0 metadata is a future extension of A, not a second validator | accept | 581, 626–629, 641–741, 757, 765, 828–829, 845, 866, 478, 905, 916, 952–953, 957, 1008–1013 | As 026-2 above. The tolerance rows of 023's A table also stand: one `atol = 1e-12` governs four checks, builder pruning is separately fixed at 1e-12, and r2's 1e-10/1e-8 defaults are unselected |
| 023-8 | Completion limits: the physical-residual decomposition and the joint-window counterexample | accept | after 364; 395–396 | As 026-5 above |

---

## 3. Points named in message 013, and where each is settled

| 013 point | Row |
|---|---|
| The 46/160 and 62/160 counts are impossible for single X/Y words; identify the restricted domain | 026-1: the reported numerical nullities (46, 62, 56) of block-internal coefficient combinations within each generator's commuting block. No X/Y word commutes with N̂ or with both spin counts at once; a one-species word commutes with the other species' count. Separately, 16 single words map the (2,2) sector into itself (a discrete flip-mask count) |
| Restore A's sole intake ownership; S0 is a superseded proposal | 026-2, 023-7 |
| Reconcile the action/runner proposal with 023's sibling selector, raw-before-merge cutoff, immutable snapshot, unscaled drive, scalar, validation and cold-cache gates | 026-3, 023-6 |
| Shared logical angles do not prove a noncommuting executed product conserves; cite the six-qubit witness and the hopping-pair proof | 026-4, 023-2, 023-3 |
| Exact-ground projection gives θ_k ≥ E_k without lower-root coverage; coverage matters for equality and ordered accuracy | 026-5 |
| Restore the joint-window and physical-residual limitations | 026-5, 023-8 |
| Full provenance and acquisition-cost binding | 026-6, 023-5 |
| Qualify the numerical-conditioning heuristics | 026-6 |
| The eigenpair helper needs a separate API from the scalar exact-energy contract | 026-7a |
| Dense construction is distinct from diagonalization | 026-7b |
| The 12-qubit scope removes the 16-qubit exceptions and selects no implementation or memory cap | 026-7d |

---

## 4. Interface disposition, and why there is no r3

| r2 proposed interface | Status | Replaced by / condition |
|---|---|---|
| S0 validator `pipelines/contracts/molecular_input.py` (r2 §4.1) | **withdrawn** | A's `resolve_molecular_diagnostic_intake` and `MolecularDiagnosticDeclaration`, `393e335a`, the sole intake |
| Richer provenance, reason codes, separate tolerances, physics digest | **open**, unselected | Only as a versioned extension of A's boundary |
| `test_support/molecular_closed_shell_fixture_factory.py` | **open**, unselected | An optional test helper; A's synthetic fixture is the selected positive witness |
| Symmetrized H2 copy as a positive fixture | **withdrawn** | A's synthetic fixture; any symmetrized copy is artificial (026-7e) |
| Paper I `ra_adapt/molecular_application.py`: predicate, named macro adapter, parent route contract, request builder with required Hartree `gradient_tolerance` | **stands as a proposal**, unselected | Amended: explicit opt-in identity (026-7c); consumes A's receipt (026-2) |
| Paper I gated branches in `engine.py` and `contracts.py` (I-2b) | **stands as a proposal**, unselected | Pins: the facade suite, the parity file, the I-1 digests |
| Sector eigenpair helper `pipelines/static_adapt/builders/sector_sparse_exact.py` "through `exact_energy_impl`" | **amended**, open, unselected | A separate eigenpair API plus a scalar wrapper (026-7a) |
| Paper II defaulted provider attribute, additive method and runner setting (r2 §3.2, II-2) | **withdrawn** | 023's sibling `MatrixFreeHamiltonian` plus `hamiltonian_action_mode`, three-path scope, raw-before-merge cutoff, unscaled-drive snapshot, cold-cache gates (026-3); itself unselected |
| Runner exposure at `ap_append_from_adapt_artifact.py:174`; run-lock and artifact propagation of the action mode | **open**, unselected | Needs its own release; the call-site threading question is recorded |
| II-3a run-lock content digest | **stands as a proposal**, unselected | Binds A's digests instead of S0's |
| II-3b fail-closed rule keyed on `requires_logical_shared_parameterization` | **amended**, open, unselected | Keyed on `execution_preserves_fixed_counts` per executed generator; leakage report is diagnostic only (026-4) |
| II-1 even-Y test support | **amended**, unselected | Intact hopping pairs J_pq with the audit flag asserted first; no shared `ham_block` aggregate |
| Paper III `pipelines/qse_spectra/molecular_static_adapter.py` and the cell branch (III-1, III-2) | **stands as a proposal**, unselected | Amended: conditions and bindings of 026-5/026-6/023-5/023-8; Hermitian B_i is a provider restriction |
| U-16 (r2 §4.5) | **withdrawn** | Scope above 12 qubits stays the user's question (026-7d) |
| I-4, `agent_guidance/static-adapt/AGENTS.md:79-80` | **stands**, unselected | Paper IV and root contract sentences stay the user's (r2 §3.5) |
| r2 §6 "Files to edit" | **void** until a release names paths | — |

**Choice: no r3.**

- **The changes are material:** a validator withdrawn, an interface replaced,
  one mathematical restriction reversed, and an increment order dissolved.
- **But r2 is itself an amendment-by-reference document** (it amends r1).
  An r3 in the same form would restate the replacement texts above verbatim.
- **No increment is selected,** so no execution path reads r2 §4. The next
  bounded release will be written from the agreed interface, not from r2's
  increment list.

The current architecture is therefore the committed r2 bytes (`f609dfd8…`) with
every replacement in §1–§2 applied. r2 itself is unchanged.

**Coordination-record note.** r2 §7's corrected dispatch sentence (lines
1009–1019) is superseded by 026-2. The record should read: "Increment A
(`393e335a`) is the completed sole diagnostic intake. No further chemistry
increment is selected."

## 5. What remains the user's

1. Whether any molecule above 12 qubits is in Program A scope (026-7d).
2. Whether to authorize correcting the stale facade sentences in
   `agent_guidance/molecular-vibronic/AGENTS.md:22-25` and root
   `AGENTS.md:94-96` (r2 §3.5).

Nothing in this disposition selects increment B, a pool, a provider, a
monitoring policy, a reference-selection policy, a lane, an upstream
repository, a molecule or evidence.

## 6. Disposition of chemistry-disposition-scope-20261018-002

Message: `/Users/lordskyler/Documents/ChatGPT/Holstein/campaign-20261010/chemistry-disposition-scope-followup-20261018-002.md`
(SHA-256 `38178bc695da645441673d0aa7016119e31cebf3d73e6ff8f0273ff947042e3a`,
17 lines, verified).

Edits are text only. Nothing was rerun, no source or test was edited, and no
agent was started.

| # | Correction | Disposition | Lines touched (this revision) |
|---|---|---|---|
| 1 | "No individual word with X/Y support commutes with N̂_alpha, N̂_beta or N̂" is false for an unaffected species: P = P_alpha ⊗ I_beta has [P, N̂_beta] = 0 | **accepted** | 60–69. Restated as: no X/Y word commutes with N̂; none commutes with both spin counts at once; a one-species word commutes with the other species' count. Consequential edit in the §3 summary row, line 511, which said "commute with the sector projector only" |
| 2 | "Exact full-register matrix nullities" overclaims the computation | **accepted** | 98–120. 46/62/56 (and 4/5/12) are relabelled as reported numerical full-register operator-matrix nullities over the generator-internal domains, with each rank threshold cited. Script hashes verified: `leakage_measure.py` `de95ed8a…c322` (lines 158–167, SVD threshold `1e-9*max(1, s.max())`) and `refute_static.py` `ea52cc31…be10` (lines 161–175, `matrix_rank(..., tol=1e-9)`). The single-word count 16 is kept separate as a discrete flip-mask count (`refute_static.py:152-154`) |

Header revision note: lines 3–6. Every other row and interface disposition is
unchanged.

Files to edit: None.
