# Strict molecular diagnostic intake: selected increment A

Message acknowledged and executed: `chemistry-strict-intake-release-20261010-021`.
Date: 2026-10-10 (America/Chicago); execution began 2026-10-11 00:11:14 UTC.
Status: selected increment A implemented and scoped gate passed. Increment B,
upstream selection and chemistry admission remain unselected.

## 1. Locked authorization and source

| Anchor | Value |
|---|---|
| Release | `/Users/lordskyler/Documents/ChatGPT/Holstein/campaign-20261010/chemistry-strict-intake-release-20261010-021.md` |
| Release SHA256 | `0e8a7892462cc869e4946aede8d877fd57f36af1fb2b7cbcd9acd3d55c8a20ac` |
| Locked contract | [chemistry-mathematical-integration-contract-20261010.md](chemistry-mathematical-integration-contract-20261010.md) |
| Contract commit | `07346ca3c6180be5353b4b4f3875e03e839623dc` |
| Contract SHA256, preserved | `5addf0334832aa664cf402969686361e51d8bd467d197727b6a4690b0d8815f7` |
| Executable baseline | `a264ca2354a455c503d45dd8363d6973fbd6421e` |
| Initial HEAD, clean | `07346ca3c6180be5353b4b4f3875e03e839623dc` |
| Checkout | `/Users/lordskyler/Holstein_test.worktrees/codex-chemistry-papers-i-iii-groundwork-20261010` |
| Branch | `codex/chemistry-papers-i-iii-groundwork-20261010` |

The complete release was read and its SHA256 verified before editing. The
selected write scope contains three code/test files and this additionally
authorized owned report. Root remains the Claude relay owner. No self-relay,
other-chat message, push, merge, UI, authentication, CHTC, shared-worklog,
manuscript, evidence or canonical-setting action was performed.

## 2. Implemented boundary

New `pipelines/static_adapt/builders/molecular_diagnostic_intake.py` supplies
`MolecularDiagnosticDeclaration` and the contract's direct-Python opt-in
`resolve_molecular_diagnostic_intake(request, *, expected_raw_sha256,
declaration, atol=1e-12)` interface. It consumes the input path once through
`Path.read_bytes`, checks the exact byte digest, parses those same bytes,
validates before the JW builder, and passes the parsed problem directly into
the private context constructor. It never invokes the path resolver again.

The strict checks reject malformed/mismatched digests, duplicate JSON keys at
any nesting level, nonfinite constants or numeric leaves, competing supported
payload locations, booleans/string/fractional count coercions, lossy real
integer-to-float64 tensor values, invalid conventions, unsupported ordering,
excess spin occupation and registers above twelve qubits. Real tensors require
their exact declared shapes, symmetric one-body entries and the three
generating real chemists' ERI symmetries. A mapped operator requires JW,
integer width `2m`, correct `e/x/y/z` words, finite coefficients and real
Hermiticity within the recorded diagnostic tolerance.

The explicit scalar slot may be zero. Declared nuclear and contracted-core
components must be finite, consistent with the declared one-body policy and
sum to that slot within diagnostic tolerance. The existing builder inserts
the slot once; intake neither recontracts core contributions nor adds them
again. Units, orthonormality, occupied-prefix ordering and nuclear/core
decomposition are supplied assumptions, not physically established facts.

The HF comparison is labeled an **input-tensor determinant-energy witness**:

\[
 E_{\mathrm{det}}=E_{\mathrm{scalar}}+
 2\sum_{i<k}h_{ii}+\sum_{i,j<k}(2g_{iijj}-g_{ijji}),\qquad k=n_\alpha.
\]

It proves neither SCF stationarity nor independent mapped parity near pruning.
Diagnostic `atol` is separate from the existing fixed builder policy,
`abs(coefficient)<=1e-12` pruning before and after aggregation. Intake does
not change that builder, choose a pool, solve a reference or qualify a seed.

`pipelines/static_adapt/builders/problem_registry.py` changes by nineteen
added lines only. The ordinary `_resolve_molecular_problem_context` retains
its blocked-ordering check and path load, then forwards the parsed problem
and both overrides to new `_resolve_molecular_problem_context_from_problem`.
AST comparison confirmed that its remaining constructor body, lazy reference
and exact-energy closures are unchanged. Family registration and dynamic
dispatch remain unchanged. No facade/CLI export, request field, shared parser,
quantum core, serializer, generic loader or default was edited.

## 3. Receipt encoding and limits

The returned context retains its existing runtime-data entries and gains
`runtime_data["molecular_diagnostic_intake_receipt"]`. Schema is
`molecular_diagnostic_intake_v1`; encoding is
`length_delimited_le_float64_v1`. Every hash-field UTF-8 name and value has
its own uint64 big-endian length prefix. Tensors use explicit shapes and
C-order little-endian float64 bytes. Scalars and operator real/imaginary
coefficients use little-endian float64 bytes. Signed zeros are deliberately
canonicalized to positive zero; this policy is recorded.

- `raw_sha256` binds exact consumed bytes, including formatting. The receipt
  also records the supplied normalized expected hash and selected nested
  payload location.
- `tensor_sha256` binds explicit shapes and numerical tensor bytes.
- `semantic_sha256` binds tensor contents, scalar/HF values, counts,
  ordering, metadata, declarations, diagnostic tolerance and builder policy.
- `operator_sha256` binds actual mapped width, JW/word convention, fixed
  pruning policy, source snapshot identity, and **actual returned term order**
  with exact coefficient representation under the signed-zero policy. No
  operator reconstruction, sorting, rethresholding or imaginary projection
  occurs while hashing.

The operator-order policy deliberately binds the consumed executable term
sequence, rather than only an unordered algebraic coefficient map. Reversing
the returned list leaves the analytic dense operator equal but changes this
digest. Formatting changes alter raw identity while preserving tensor,
semantic and operator identities. A changed actual builder result changes
the operator digest even when input bytes and tensors stay the same.

Builder identity contains the observed source-file SHA256 and fixed policy.
The receipt labels that hash a source snapshot: it does not attest that an
already imported Python function executed those exact source bytes. The
operator digest binds the resulting operator independently of that limitation.

This receipt is a snapshot, not downstream enforcement. The context still
contains mutable dictionaries, arrays and operators, and the generic loader
does not propagate the new receipt automatically. An entirely empty/pruned
mapped polynomial is rejected explicitly because its existing representation
has no intrinsic operator width; no width is fabricated from the state.
Zero scalar is separately supported and tested with a nonempty Hamiltonian.

## 4. Mathematical and compatibility verification

The new test file contains 78 cases. The positive witness is the independent
synthetic four-qubit fixture from contract section 5.4, using blocked-sector
indices `[5,9,6,10]`. Tests construct Pauli matrices independently and compare
the mapped sector block with the prescribed analytic matrix and roots
`[0.1-sqrt(785)/20,-0.15,-0.05,0.1+sqrt(785)/20]`, at absolute `1e-12` and
zero relative tolerance. The tests also check absence of couplings outside
that sector and leave exact-reference closures lazy.

The legacy H2-labelled writer remains byte-for-byte unchanged. Its previously
identified ERI symmetry violation causes strict rejection before JW. Ordinary
route compatibility is checked separately; rejection does not rewrite its
fixture or imply new canonical chemistry admission.

Additional witnesses cover all supported wrapper locations; early rejection
before a monkeypatched fail-on-call JW builder; path replacement immediately
after the single read; distinct raw/content hash semantics; actual builder
coefficient and term-order binding; contracted-core scalar insertion once;
explicit zero scalar; mapped width/JW/nonfinite/Hermiticity checks; and ordinary
wrapper error order, Hamiltonian override and lazy exact callback behavior.

Near-pruning tests accept the input-tensor witness with an occupied ERI
`7.5e-13` at diagnostic `atol=1e-14`, and scalar `0.5e-12` at `atol=1e-15`.
The fixed builder prunes those tiny contributions, so the independently
computed mapped HF value differs beyond diagnostic tolerance. These passing
tests discriminate the input witness from a claim of mapped parity and prove
that diagnostic tolerance does not change builder pruning.

Review found and corrected a fail-closed width gap: `int(term.nqubit())`
would accept `4.5` as `4`. The validator now uses the strict integer check.
The fractional-width regression was run against the former coercion restored
**in process memory only**, with no file edit: it failed as expected,
`DID NOT RAISE ValueError` (`1 failed in 0.32s`). The actual implementation
passes that regression in the completed combined gate.

Required gate, executed in the scoped checkout:

```sh
PYTHONDONTWRITEBYTECODE=1 PYDEPHASING_TESTING=1 /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q \
  test/test_molecular_diagnostic_intake.py \
  test/test_static_adapt_builders.py::test_build_hamiltonian_blocks_pool_matches_molecular_grouped_builders \
  test/test_adapt_vqe_integration.py::TestAdaptResolvedProblemContext::test_molecular_full_meta_pool_is_problem_local_mega_pool
```

Final result: **80 passed in 2.76s**, exit zero: 78 new cases and both existing
characterizations. The earlier partial gate passed 70 cases in 1.44s; the
test author also ran 74 cases in 0.40s before final additions. These are
development receipts, not substitute final gates. The three documented
facade/runtime-loader baseline failures remain outside scope and were not
repaired, waived or broadly rerun.

## 5. Source/input identities, resource and graph receipts

| Artifact | SHA256 |
|---|---|
| New intake module | `b57529efe78e9cdc1bdda5f42c2c739c41a47d6e42726e7665230844a2c50874` |
| Registry after private extraction | `89af85b1143bf69dd93bcd9af3564df29445e8b2f34b31ce7c4212217e1e8e4c` |
| New tests | `f510e496dc6ceab14f31fa7bacb2968c35d4aad971e179dceec48a9f428511dd` |
| Unchanged molecular builder | `403ea2978f2f452709f239b5f8e30999fdbe95fdc942a01c54342d615582e7e6` |
| Unchanged legacy test file | `fd1d3162fad1e9f2f29bee62023fc805fbc8a995b6eed6a9533a5f134cc0f843` |
| Positive synthetic JSON bytes (468 bytes) | `583832f7a6a2d3570ae64e2b1bcf17b88ed85dee8684634d18a4367f0202a9a6` |

The synthetic input hash is reproducible as
`json.dumps(_payload()).encode("utf-8")` from the new test file with default
JSON formatting and its preserved dictionary insertion order. It is a
synthetic diagnostic, not a molecule or benchmark adoption. This report's
SHA256 and the scoped commit ID are returned after committing; the report
does not attempt to contain its own hash.

Fresh headroom check observed 24 GiB physical memory, 54% memory-pressure
system-free percentage, 202,919 free 16-KiB pages and substantial compression.
The source files had ordinary resident file flags, not dataless/compressed.
Only tiny four-qubit tests ran. Three delegated agents handled extraction,
tests and read-only review; root implemented intake and integrated the result.
There was no substantial launch, RAM-guard wrapper or invented memory cap.
The twenty-minute end-to-end bound includes preparation, review and commit;
actual final elapsed time is returned with the commit receipt.

Fresh GitNexus impact queries used explicit repository
`/Users/lordskyler/Holstein_test`. The main index is older (`d1436ab1`), and
the new worktree/files are unindexed. The molecular callback, new targets and
report returned UNKNOWN/zero indexed impact, not LOW or a safety proof.
Source inspection ratified the callback registration/dispatch and private
helper construction. This release explicitly requires treating unindexed
zeros as unseen and checking source; no shared index refresh or analyzer
build was added to the bounded scope. Graph impact/change gates from the
work skill were applied with that release-specific qualification; there is
no claim of an index-current execution-flow proof. Final explicit-worktree
`detect_changes` and staged diff inspection precede the scoped commit.

Preserved earlier Git blobs: internal map
`5a13a287e8185957c3bb80553211863aa0f17763`, candidate audit
`86e1521040b100840a7ac05d5a82b49f481e0f84`, locked contract
`802810b430ac8d5b3d5e6987cfc1cfe1c840ab55`. No changes occur in shared
quantum code, contracts, loaders, existing test fixtures or canonical settings.

## 6. Completion and next action

Increment A meets its selected scoped gate and mathematical/snapshot
obligations. Changed paths are exactly the new intake module, nineteen-line
registry extraction, new test file and this owned report. The coordinator
can relay this scoped commit to the existing Claude architecture author for
independent review. No implementation of B, molecule/upstream/lane admission,
pool/default adoption, seed qualification, downstream enforcement, evidence
promotion, push or merge follows automatically.

Files to edit: None. The selected implementation is complete; the coordinator
owns the independent-review relay and any later selection.
