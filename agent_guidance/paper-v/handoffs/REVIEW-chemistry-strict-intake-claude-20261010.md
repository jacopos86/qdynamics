# Review: strict molecular diagnostic intake, increment A (`393e335a`)

Reviewer: the existing Claude chemistry architecture author (author of r1 and
r2), at the request of Codex coordinator message `codex-to-claude-20261010-013`,
paragraphs 1–3. This is a read-only review of the committed diff. It is
independent of root's review and of the reviewer cited in 013.

**Verdict.** I found no implementation defect in the four-file diff. The
diagnostic does what contract `07346ca3` §5 and release 021 require.

I found three **test-sensitivity gaps** (T1–T3). In each, the implementation
is correct, but a specific regression would still pass the 78 new tests. They
do not block A. They are listed for whoever next edits the test file; nothing
here asks for that edit.

**What does not follow.** No physical provider admission, no lane admission
(Papers I–III), no upstream repository, no molecule and no evidence adoption
follows from A or from this review. Increment B (the opt-in Hamiltonian action)
remains unselected.

## 0. Inputs and hash verifications

Every hash below was computed with `shasum -a 256` before reading. All match
the values in 013.

| Artifact | SHA-256 (verified) | How |
|---|---|---|
| Mathematical contract, `agent_guidance/shared/handoffs/chemistry-mathematical-integration-contract-20261010.md` (743 lines) | `5addf0334832aa664cf402969686361e51d8bd467d197727b6a4690b0d8815f7` | working file and `git show 07346ca3:<path>` |
| Increment A report, `…/chemistry-strict-intake-increment-20261010.md` (226 lines) | `37bc322ddad6115eda5b84746d7462fc909310b877041741c4f23d47bbf7a097` | working file and `git show 393e335a:<path>` |
| 023 reconciliation, `…/chemistry-r2-execution-reconciliation-20261010.md` (395 lines) | `b605ba975efb36ee75b40699eecd1d4d98f3a390452434f7556da9478ffd278c` | working file and `git show 90e9f9e4:<path>` |
| Release 021, `/Users/lordskyler/Documents/ChatGPT/Holstein/campaign-20261010/chemistry-strict-intake-release-20261010-021.md` (48 lines) | `0e8a7892462cc869e4946aede8d877fd57f36af1fb2b7cbcd9acd3d55c8a20ac` | file |
| Root review 026, `…/chemistry-committed-r2-review-20261011-026.md` (33 lines) | `53beb3c8da105d2152fca51d2f04f493e2247efb7170b3285346747f37a658ed` | file |
| Message 013, `…/codex-to-claude-20261010-013.md` | `b078b9a4df55ceb73e2c5ed3e8f7b695b89fd774ba7b2856cda045bedbcaea73` | file |
| r2 under review in 026 | `f609dfd8e4648b27a82095f13d26f10bfbd5cfd330ff2e3eb85f6b33289114d0` | working file and `git show c943ae8a:<path>`, this worktree |
| A's intake module at `393e335a` | `b57529efe78e9cdc1bdda5f42c2c739c41a47d6e42726e7665230844a2c50874` | `git show`; matches A report table line 174 |
| A's registry after extraction | `89af85b1143bf69dd93bcd9af3564df29445e8b2f34b31ce7c4212217e1e8e4c` | `git show`; matches line 175 |
| A's new tests | `f510e496dc6ceab14f31fa7bacb2968c35d4aad971e179dceec48a9f428511dd` | `git show`; matches line 176 |
| Molecular builder `src/quantum/chemistry/molecular_hamiltonian.py` | `403ea2978f2f452709f239b5f8e30999fdbe95fdc942a01c54342d615582e7e6` | at `393e335a` **and in this worktree** (identical) |
| Legacy test file `test/test_static_adapt_builders.py` | `fd1d3162fad1e9f2f29bee62023fc805fbc8a995b6eed6a9533a5f134cc0f843` | at `393e335a` and in this worktree (identical) |
| Shared parser `src/quantum/chemistry/psi4_adapter.py` | `de1e37eb84e501aefcfd47494bd2d435501b6de11250c8e0a1164a68cf5f14b0` | at `393e335a` and in this worktree (identical) |

The Codex worktree was used read-only: `git log`, `git show`, `git diff`, and
reads only. Nothing ran there.

**The A test file was not run.** It does not import in this worktree:
`pipelines.static_adapt.builders.molecular_diagnostic_intake` and the
extracted `_resolve_molecular_problem_context_from_problem` exist only on the
Codex branch. Copying them in would mean writing source files here, which this
task excludes. The review therefore rests on reading every changed line, plus
one independent recomputation (§3).

## 1. The four changed files

`git diff --numstat 07346ca3 393e335a` (the parent of A is `07346ca3`):

| Path | Added | Deleted |
|---|---|---|
| `agent_guidance/shared/handoffs/chemistry-strict-intake-increment-20261010.md` | 226 | 0 |
| `pipelines/static_adapt/builders/molecular_diagnostic_intake.py` | 287 | 0 |
| `pipelines/static_adapt/builders/problem_registry.py` | 19 | 0 |
| `test/test_molecular_diagnostic_intake.py` | 460 | 0 |

**Status: verified.** The diff is exactly the release-021 write scope: three
code and test paths plus the owned report. No shared parser, quantum core,
loader, serializer, CLI or default file changed.

## 2. Legacy H2 compatibility witness

| Check | Status | Evidence |
|---|---|---|
| The writer `_write_molecular_problem_json` is byte-for-byte unchanged | **verified** | `test/test_static_adapt_builders.py` SHA-256 `fd1d3162…` is identical at `393e335a`, at `07346ca3` (contract §5.4 line 474), and in this worktree |
| Its defects are documented | **verified** | Contract §5.4, lines 469–474, records the pair-exchange defect `0.48218458222099997` and the determinant-energy discrepancy `0.0013272844169998077` Hartree. Both agree with my r2 N2: `fixture_probe.json` gives `(rs\|pq)` residual 0.48218458222099997 and `hf_energy − ⟨HF\|H\|HF⟩` = −1.3272844170000297e-3, a dense-matrix evaluation that agrees with the contract's closed form to 2.2e-16 |
| Strict path rejects it before JW construction | **verified** (by reading) | `test_molecular_diagnostic_intake.py:137-153` extracts the payload literal by AST without executing the writer, patches the registry's `build_restricted_closed_shell_molecular_hamiltonian` to fail, and expects `ValueError`. Control flow: the first failing check is ERI pair exchange, `molecular_diagnostic_intake.py:241-248`, before the parse at `:257` and the construction at `:258` |
| Ordinary route still accepts it | **verified** (by reading; gate not rerun) | The gate includes `test/test_adapt_vqe_integration.py:527-563`. That test resolves the writer's payload through the ordinary `resolve_problem_context` and builds the molecular `full_meta` pool |
| The rejection reason is pinned | **gap T3** | The test asserts only `ValueError`. It does not name which check fired. The HF defect alone is exercised on the synthetic fixture (`:158`, `hf_energy = −1.2`), but never on the legacy payload |

## 3. Synthetic positive fixture: independent recomputation

I transcribed the fixture from contract §5.4 (lines 479–483), not from the Codex
test file. I rebuilt it in this worktree in the dict insertion order of
`_payload` (`test_molecular_diagnostic_intake.py:26-44`) and checked it with
this worktree's builder, whose bytes are identical to the pin.

Script: `output/local_runs/paper_v_chem_arch_20261010/strict_intake_fixture_recheck.py`.
Output: `strict_intake_fixture_recheck.json`. Run time under 1 s.

| Quantity | Value | Status |
|---|---|---|
| JSON bytes, `json.dumps(payload)` | 468 bytes, SHA-256 `583832f7a6a2d3570ae64e2b1bcf17b88ed85dee8684634d18a4367f0202a9a6`, equal to A report line 179 | **verified** |
| Eightfold ERI symmetry: max \|g_pqrs − g_π(pqrs)\| over all 7 nontrivial permutations of the real chemists' group | 0.0 for every permutation | **verified** |
| h symmetry residual | 0.0 | **verified** |
| Occupied-prefix determinant energy E_det = E_scalar + 2Σ_{i<k} h_ii + Σ_{i,j<k}(2g_iijj − g_ijji), k = n_alpha = 1 | −1.2999999999999998 by naive sum, against the declared −1.3: difference 2.2e-16. A sums with `math.fsum` (`molecular_diagnostic_intake.py:250`) | **verified** |
| JW Hamiltonian | 15 Pauli terms; dense Hermiticity residual 0.0 | **verified** |
| Blocked-sector block on basis indices [5, 9, 6, 10], from independent 2×2 Pauli matrices, against the contract's analytic matrix (§5.4, lines 485–495) | max deviation 4.4e-16 | **verified** |
| Sector roots against 0.1 − √785/20, −0.15, −0.05, 0.1 + √785/20 | max deviation 2.2e-16; roots [−1.3008925726121896, −0.1499999999999999, −0.04999999999999992, 1.50089257261219] | **verified** |
| Coupling from the sector to the other 12 basis states | 0.0 | **verified** |

The fixture is content-valid as a synthetic diagnostic. It is not a molecule,
and A's report says so (line 184).

## 4. HF witness and the two tolerances

| Check | Status | Evidence |
|---|---|---|
| The HF check is an **input-tensor determinant-energy witness** | **verified** | It is computed from the validated `h`, `g` and scalar (`:249-255`) before the shared parser (`:257`) and before JW construction (`:258`), and it compares against the declared `hf_energy` (`:253-254`). It is labelled as such in the docstring (`:177-178`), in the receipt qualification (`:285`) and in the report (lines 55–62). It does not involve the mapped operator, so it says nothing about SCF stationarity or mapped parity |
| The diagnostic `atol` is distinct from builder pruning | **verified** | `atol` (default 1e-12) is validated at `:180-182`. It governs the scalar/core sum (`:237`), the symmetry checks (`:247`), the HF witness (`:253`) and the mapped imaginary parts (`:150`). The builder's pruning is the unchanged default `tol = 1e-12` of `build_restricted_closed_shell_molecular_hamiltonian`, called with no `tol` argument by the extracted helper (`problem_registry.py:845-852` at `393e335a`). `_BUILDER_TOL` (`:35`) is only *recorded* in the receipt (`:159`, `:261`, `:278`) |
| A test discriminates the two tolerances | **verified** (by reading) | `test_molecular_diagnostic_intake.py:375-389` sets (00\|00) = 7.5e-13 with `atol = 1e-14`. The witness accepts, while the builder prunes 0.5 × 7.5e-13 ≤ 1e-12, so the mapped HF diagonal is −1.7 and differs from `hf_energy` by more than `atol`. Lines 392–407 do the same for a scalar of 0.5e-12. If intake passed `atol` to the builder, or validated against the mapped HF value, these tests would fail |
| The recorded pruning value tracks the builder | **plausible-unverified (P1)** | `_BUILDER_TOL = 1.0e-12` is a hardcoded copy of the builder's default, not read from it. If the builder default changed, the recorded `fixed_pruning_tolerance` would be stale. Mitigation already present: `builder_source_file_sha256` (`:263`) would change, and `operator_sha256` binds the actual output (`:162-165`) |
| One `atol` governs four distinct checks | **plausible-unverified (P2)** | A single absolute 1e-12 (symmetry, scalar sum, HF witness, mapped imaginary part) suits these exact synthetic values. Integrals from a real SCF transformation may be symmetric only to rounding well above 1e-12. A caller can raise `atol`, but only for all four at once. 023's table records this as a future versioned policy, not a defect of A |

## 5. What the snapshot receipt covers and does not cover

**Covers** (`molecular_diagnostic_intake.py:259-287`; encoding `:111-130`):

- exact consumed bytes (`raw_sha256`, `:192-195`). The expected hash is
  validated as 64 hex characters (`:183-184`) and compared case-insensitively
  (`:194`).
- the payload location, which must be unique (`:94-108`, `:200`).
- `tensor_sha256`: shapes plus C-order little-endian float64 `h` and `g`, with
  signed zeros canonicalized (`:116-119`, `:267-270`).
- `semantic_sha256`: the tensor digest, counts, `ordering`, `reference`,
  `geometry_spec`, `basis`, every declaration field, scalar, `hf_energy`,
  `atol`, builder name, builder source-file hash and pruning policy
  (`:271-278`).
- `operator_sha256`: width, word convention, mapping, pruning policy, builder
  identity, and every returned term in actual return order, with exact real
  and imaginary float64 bytes (`:133-165`). It binds what the builder actually
  produced, not a reconstruction.
- the measured defects, `checks` (`:241-255`, `:266`).
- explicit qualification labels: `binding = snapshot_only_no_downstream_enforcement`
  and `builder_source_identity = observed_source_file_hash` (`:285`).

**Does not cover** (all stated by A: report lines 106–116; module docstring
`:1-5`):

- **Mutation after return.** `replace(context, runtime_data=…)` (`:287`)
  returns a context that shares the mutable Hamiltonian polynomial and the
  parsed problem arrays. Nothing re-verifies `operator_sha256` later.
- **Propagation.** A Paper II seed rebuilds its `ProblemRequest` from
  `settings` (`pipelines/scaffold/runtime_loader.py:1375-1381`) and resolves it
  through the ordinary resolver (`:1056-1057`). That rereads the path and
  carries no receipt. Run locks and protocol digests do not bind A's digests.
- **Execution identity of the builder.** The source hash is the file on disk
  at intake time, not proof that the imported function ran those bytes.
- **Physical truth of the declarations:** units, orthonormality,
  occupied-prefix order, and the nuclear/core decomposition.
- **Presence of provenance.** `geometry_spec` and `basis` are hashed but not
  required. When absent, the shared parser's defaults `""` and `"unknown"` are
  hashed (`psi4_adapter.py:114-115`). A missing `charge` hashes as `"0"`
  (`:272`), the same as an explicit 0. This matches the contract, which does
  not require these fields; 023 lists them as future admission requirements.
- **Lazy closures.** The reference-state and exact-energy closures are lazy
  and unbound by the receipt.

**Status: verified** against the contract's snapshot language (§5.3,
lines 455–465). Nothing in A claims enforcement.

## 6. Ordinary resolver path

**Status: verified, structurally and behaviourally.**

- **Structurally:** the registry diff has 19 insertions and **0 deletions**.
  The ordinary `_resolve_molecular_problem_context`
  (`problem_registry.py:804-822` at `393e335a`) still checks ordering first
  (`:811-814`), then loads the path (`:815`), then delegates with both
  overrides (`:816-822`). The construction body from `:835` onward consists of
  the original lines, unedited. The family registration still names the
  wrapper (`:1764`).
- **Behaviourally:**

  | Test | What it proves |
  |---|---|
  | `test_ordinary_ordering_error_precedes_path_read` (`test_molecular_diagnostic_intake.py:431-435`) | The ordering error is raised before the path is read; an absent path still raises `FileNotFoundError` |
  | `test_ordinary_override_and_exact_closure_remain_lazy` (`:438-460`) | The Hamiltonian override is used, the exact-energy callback is lazy and receives the same kwargs, and the `request.num_sites` message is unchanged |
  | `test/test_adapt_vqe_integration.py:527-563` | The ordinary route accepts the legacy writer and builds the molecular `full_meta` pool |

- **Not part of this proof:** the other existing gate test,
  `test/test_static_adapt_builders.py::test_build_hamiltonian_blocks_pool_matches_molecular_grouped_builders`.
  It calls `load_restricted_closed_shell_problem_from_json` and the
  primitive-pool builder directly, not the resolver. It characterizes the
  builder.
- **No golden comparison** of the ordinary-path Hamiltonian or context against
  pre-A bytes exists. Byte-identity rests on the zero-deletion diff, which is
  sufficient here.

## 7. Are the 78 new cases meaningful?

**Count.** I counted the parametrizations by hand and get 78. With the two
existing characterizations that gives the report's 80.

| Test (lines) | Cases | Fails under this defect | Status |
|---|---|---|---|
| `test_analytic_sector_block` (103–134) | 1 | Wrong ERI index lookup, prefactor, spin ordering or scalar insertion: block or roots deviate from the analytic matrix by more than 1e-12. It uses independent Pauli matrices (`:86-100`), not the production dense builder. It also fails if intake triggers the exact solve (`:104-107`) | meaningful |
| `test_legacy_h2_fixture_rejected` (137–153) | 1 | Strict path accepts the legacy payload, or the rejection happens after JW | meaningful; reason not pinned (T3) |
| `test_invalid_payload_fails_before_jw` (156–187) | 32 | Each leaf case would fail if its check were removed. The following are isolated: string, complex-string and 2^53+1 entries in h (the shared parser alone would coerce "0.5" and lose 2^53+1); multiplicity 3; charge 0.5; `uhf`; asymmetric h; n_beta = 0; the fractional and boolean counts; `n_spin_orbitals` 4.5 | mostly meaningful; T1, T2 below |
| `test_broken_eri_symmetry_fails_before_jw` (190–201) | 3 | Removing any of the three generating symmetry checks. The occupied (00\|00) is untouched, so the HF witness cannot backstop | meaningful |
| `test_invalid_declaration_fails_before_jw` (204–217) | 9 | Any convention, scalar-semantics or core-policy check removed | meaningful |
| `test_invalid_request_fails_before_jw` (220–229) | 3 | Family, ordering and `num_sites` checks | meaningful |
| `test_snapshot_syntax_and_identity_fail_before_jw` (232–253) | 5 | Duplicate keys at the root and nested: the duplicated value equals the original, so only duplicate detection rejects. Ambiguous location: the shared unwrapper alone would silently pick `problem`. Malformed and mismatched hash | meaningful |
| `test_invalid_tolerance_fails_before_jw` (256–263) | 3 | Negative, NaN or infinite `atol` | meaningful |
| `test_supported_payload_locations` (266–277) | 4 | A wrong location label, or a rejected supported wrapper | meaningful |
| `test_snapshot_survives_path_change` (280–298) | 1 | A second read of the path, or construction from a reread | meaningful |
| `test_raw_and_tensor_hashes_have_distinct_semantics` (301–312) | 1 | Raw digest insensitive to formatting, or content digests sensitive to it, or insensitive to h[1,1] | meaningful |
| `test_signed_zero_has_canonical_content_binding` (315–324) | 1 | −0.0 not canonicalized | meaningful |
| `test_operator_hash_binds_actual_builder_output` (327–342) | 1 | Operator digest computed from tensors rather than from the returned polynomial | meaningful |
| `test_operator_hash_binds_actual_returned_term_order` (345–363) | 1 | Sorting or merging before hashing | meaningful |
| `test_contracted_scalar_is_declared_once` (366–372) | 1 | Intake adds the declared core scalar again: H[5,5] would move from −1.3 | meaningful |
| `test_diagnostic_tolerance_does_not_change_fixed_builder_pruning` (375–389) and `test_zero_and_near_pruned_scalar_follow_input_tensor_witness` (392–407) | 1 + 2 | `atol` leaking into builder pruning, or the witness being evaluated on the mapped operator (§4) | meaningful |
| `test_actual_mapped_operator_is_validated` (410–428) | 6 | Width, fractional width (the regression A's report says it reproduced), nonfinite, imaginary, empty and non-JW checks | meaningful |
| Ordinary path (431–460) | 2 | §6 | meaningful |

**Test-sensitivity gaps.** All are correct by reading, non-blocking, and not
fixed here.

- **T1. Silent-0.0 regression undetected** (`test_molecular_diagnostic_intake.py:157`).
  - The two deletion cases, missing `nuclear_repulsion_energy` and missing
    `hf_energy`, are rejected today by the strict `KeyError` path (`:202-212`).
  - But if that path regressed to the shared parser's silent defaults
    (`psi4_adapter.py:122-123`, the gap r1 and r2 flagged), both cases would
    still be rejected:
    - a missing scalar would become 0.0, which disagrees with the declared
      0.3 (`:237`);
    - a missing `hf_energy` would become 0.0, which disagrees with E_det = −1.3
      (`:253`).
  - **A discriminating version:**
    - delete the scalar while declaring `nuclear_scalar_hartree = 0.0`, with
      `hf_energy = −1.6`, which is consistent with a zero scalar;
    - delete `hf_energy` from a payload whose E_det is exactly 0 (scalar 1.6,
      declared 1.6);
    - assert the message "Missing strict molecular field" in both cases.
- **T2. Over-cap case does not isolate the cap** (`:162`).
  - `("n_spatial_orbitals", 7)` is rejected by the cap (`:213`). Without the
    cap, the `request.num_sites` check (`:217`) and the shape checks
    (`:239-240`) would still reject it.
  - The 2m ≤ 12 cap therefore has no isolating test.
  - Separately, `("n_alpha", 3)` (`:165`) is rejected by n_alpha ≠ n_beta, not
    by the capacity bound n_alpha ≤ m. A case with n_alpha = n_beta = 3 at m = 2
    would isolate it. Without the bound it would reach `h[2,2]` and raise
    `IndexError`, not `ValueError`.
  - **A discriminating version:** a consistent m = 7 payload (7×7 h, 7^4 g)
    with `request.num_sites = 7`, asserting the cap message; and an
    equal-count over-capacity case.
- **T3. Legacy rejection reason not pinned** (`:151-153`). Asserting the ERI
  symmetry message would tie the test to the documented defect instead of to
  any `ValueError`.

**Other checks that were not isolated.** These are backstopped by other checks
and harmless: `("hf_energy", True)` (the HF witness also rejects 1.0) and
`("nuclear_repulsion_energy", 0.4)` (the declaration and witness both reject).

## 8. Statement of scope

A is a direct-Python opt-in diagnostic that returns the existing
`ResolvedProblemContext` with a snapshot receipt. A does **not**, and nothing
in this review does:
- admit a physical provider;
- admit a Paper I, II or III lane;
- select an upstream repository or a molecule;
- adopt a pool, seed, reference or default;
- qualify evidence.

Increment B and every r2 increment remain unselected; see the companion
disposition,
`agent_guidance/paper-v/handoffs/DISPOSITION-chemistry-026-023-claude-20261010.md`.

Files to edit: None.
