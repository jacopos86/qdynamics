# Doron bridge: disposition of independent review findings — corrections 002 (2026-10-19)

Message ID acknowledged: `doron-bridge-review-corrections-20261019-002`.
Packet: `/Users/lordskyler/Documents/Codex/2026-10-18/task-4/doron-bridge-correction-request-20261019-002.md`,
37,669 bytes, SHA-256 `2d11faa130ce184e5ab2fcc66889bf4dd1f3b7113e4e679ed45878376990e036` (verified, read in full).
Embedded review: `/Users/lordskyler/Documents/ChatGPT/Holstein/verification/doron-bridge-20261019/report.md`,
189 lines, 30,747 bytes, SHA-256 `7f071731558dd99acd405efda24f8414f33763c68f9eecd3b020c31ae760f50e`
(verified, read in full, together with `probes.txt`, `diagnostic-probes.txt`, `final-checks.txt`,
`closure.json`, `cli-failed.command.json` and `probe-inputs/contradictory.FCIDUMP`).
Author: Claude (Fable 5.1), same thread and isolated worktree. Recheck: the existing Codex 6.1
Ultra reviewer `01a15508-8cf9-75e0-80c0-9ef5807e0575`, arranged by the coordinator. Nothing here
is self-certified as independently accepted.

## 1. Ownership and base (checked before any edit)

| item | value |
|---|---|
| Worktree / branch | `/Users/lordskyler/Holstein_test.worktrees/claude-doron-bridge-20261019`, `claude/doron-chemistry-quantum-bridge-20261019` |
| HEAD at preflight | `40e6f6bc3bb4fd47a30e4dfd42f01a3b3212fdc6`; `git status --short` empty (no user or uncommitted edits to preserve; the uncommitted `methods.py` the review saw during its 9fb14d3b inspection was the data-role rename later committed as `82fe6bc3`) |
| Preserved | `82fe6bc3` (GPU receipts, first data-role rename) and `40e6f6bc`; no reset, no broad staging, no edit outside this worktree |
| Correction commits | `404454ac` (002-A: findings 2, 3, 6), `16e12761` (002-B: findings 1, 4, 5), plus the 002-C commit recorded below (receipts, this disposition, main-handoff amendments) |
| Runtimes | repository venv `/Users/lordskyler/Holstein_test/.venv/bin/python` (3.12.14; no torch/h5py) and the previously approved isolated venv `/Users/lordskyler/.venvs/doron-bridge-gpu/bin/python` (torch 2.14.1 MPS, h5py 3.16.0). Nothing installed or updated. No cluster, scheduler, credential, campaign or CHTC action. |
| Graph checks | worktree indexed in place; `detect_changes`: 002-A 41 touched symbols / 0 affected processes / low; 002-B 13 / 0 / low; 002-C below |

What the first review did not cover and what changed since: the review pinned `9fb14d3b`. `82fe6bc3` added MPS receipts and renamed the file-input data role; it did not address findings 1–5 and only partially addressed finding 6 (a string rename keyed on the schema label). All six findings were therefore open at `40e6f6bc` and are dispositioned below against the current code.

## 2. Findings, dispositions and exact changes

| # | finding (P2) | disposition | code | regression |
|---|---|---|---|---|
| 1 | `logical_shared` is not sector-safe for the `full_meta` pool | **Corrected by explicit, labelled restriction; pool unchanged.** `run_ap_mclachlan` audits every candidate with the repository's `pipelines.static_adapt.sector_invariants.audit_generator_sector_contract` and offers only generators whose *executed* unitary preserves `(n_up, n_dn)` under the chosen parameterization (`logical_shared` → `execution_preserves_fixed_counts`; `per_pauli_term` → `components_individually_preserve_fixed_counts`). `candidate_sector_filter="none"` keeps the unaudited pool for leakage regressions. The measured leakage gate is unchanged. Seed selection is `auto` over eligible generators (preference `ham_block` > `ham_term` > UCC) or an explicit label that must be eligible. | `methods.py`: `_sector_safe_candidates`, `run_ap_mclachlan(candidate_sector_filter, seed_term_label="auto")`, receipt fields `sector_audit`, `executed_parameterization_sector_safe`, `candidate_count_offered`, `seed_selection`, `trajectory_moved`, `theta_max_abs`; `run_bridge(apm_candidate_filter)`; CLI `--apm-candidate-filter`. Module docstring corrected (no universal claim). | `test_finding1_logical_shared_alone_leaks_and_the_executed_audit_filter_restricts` (review case: random6, cap 128, `mclachlan_l2`, cut 1e-6 → unfiltered `failed`, leakage **1.236228e-4** = review's 0.0001236228263; filtered `cpu_verified`, leakage 0), `test_finding1_per_pauli_term_leaks_without_the_filter_and_is_restricted_with_it` (random8 per-word unfiltered 3.3e-5 → `failed`; filtered `cpu_verified`; explicit non-eligible seed refused). The old test named "logical_shared conserves" is removed. |
| 2 | malformed `nuclear_repulsion` truncated | **Corrected.** `_scalar_dataset`: only a real, finite, zero-dimensional number is accepted; shape `(1,)`, `(2,)`, text, boolean, nonfinite and complex-with-imaginary inputs are rejected with codes `nuclear_repulsion_shape`, `nuclear_repulsion_type`, `nuclear_repulsion_nonfinite`, `complex_tensor`. Length-one compatibility is deliberately not offered. | `qdynamics_export.py` | `test_finding2_malformed_nuclear_scalar_is_rejected_before_extraction` (8 cases incl. the review's `[0.3, nan]`), `test_finding2_zero_dimensional_numpy_scalar_is_the_only_accepted_shape` |
| 3 | contradictory FCIDUMP entries overwrite silently | **Corrected.** Entries are keyed by their canonical eightfold representative; a repeated key differing by more than `duplicate_atol` (default 0.0, exact repetition only) raises `fcidump_conflict`; identical repeats are counted in `repeated_identical_entries`; `unique_*_entries` count distinct canonical keys; `duplicate_atol` is recorded in `FcidumpData` and provenance. | `qdynamics_export.py`: `_canonical_two_body_key`, `_canonical_one_body_key`, `read_fcidump(duplicate_atol=0.0)` | `test_finding3_review_contradictory_fcidump_is_rejected` (the review's probe text, sha256 `3b6a6ce3…` asserted in-test), `test_finding3_symmetry_equivalent_conflicts_and_identical_repeats` ((22\|11) vs (11\|22), h(2,1) vs h(1,2), identical repeat accepted and counted, tolerance-admitted repeat) |
| 4 | exact-reference clock wrong for nonzero initial time | **Corrected.** The static zero-drive reference is propagated by elapsed time from the first checkpoint; receipts carry `initial_time`, `elapsed_times_from_initial_checkpoint`, `exact_reference_clock`. Non-increasing or single-point grids are rejected. No drive protocol exists in this runner (none is configurable), so no physical time origin was altered. | `methods.py` `run_ap_mclachlan` | `test_finding4_static_zero_drive_reference_is_translation_invariant` ((0,.02,.04) vs (1,1.02,1.04): identical energies and fidelities to 1e-12; both initial fidelities > 1−1e-12) |
| 5 | CLI exit 0 for no calculation / failed calculation | **Corrected.** `normalize_methods` rejects unknown/empty requests (usage error). Exit policy, printed with every run and stored in the bundle: **0** all requested methods `cpu_verified`/`gpu_verified`; **1** any `failed` (receipts kept); **2** usage error or strict-intake refusal; **3** nothing failed but a requested method was `skipped_unsupported` or `blocked` (never GPU evidence). | `methods.py`: `KNOWN_METHODS`, `normalize_methods`, `exit_code_for_statuses`, `EXIT_*`; `__main__.py` | `test_finding5_method_requests_are_validated_and_exit_codes_follow_statuses`; new `test/test_chemistry_bridge_cli.py` (5 subprocess tests: misspelled → 2, empty → 2, verified → 0, CUDA unavailable → 3, per-word APM `failed` → 1 with receipt kept, intake refusal → 2) |
| 6 | schema label promoted to producer-export evidence | **Corrected (the `82fe6bc3` rename was a string change only).** Provenance now separates `container_schema` (set only by the loader that produced the bytes; a caller mapping cannot claim a file container), `claimed_producer` (a recorded claim), `origin` ∈ {`synthetic`, `simulated`, `unknown`, `attested`} declared by the caller and defaulting to `unknown`, and `origin_attestation` (required for `attested`, must name the attester and cover the loaded file's sha256; in-memory mappings cannot be attested). Caller flags `simulation=True` / `producer_generated=False` force a non-producer origin. `producer_origin_established` is always `False` (no mechanism exists). Receipts carry `data_role = origin:container` and the separate fields. CLI: `export-qdynamics --declared-origin {unknown,synthetic,simulated} --claimed-producer`; `export-synthetic` declares `synthetic`. | `qdynamics_export.py`: `_origin_record`, `qdynamics_matrix_elements_from_mapping(declared_origin, declared_producer, attestation)`, `load_qdynamics_matrix_elements_h5(...)`, `FcidumpSidecar.origin`; `methods.py` `resolve_bridge_context`, `BridgeContext.input_identity` | `test_finding6_schema_label_and_hash_never_establish_producer_origin` (the review's probe mapping → `simulated:in_memory_mapping`; undeclared → `unknown`; claimed producer recorded as a claim; `attested` refused without / with a mismatching attestation; unknown origin word refused); `test_h5_loader_is_explicit_about_its_dependency` extended (file → `unknown` until declared; attestation hash must match) |

Scope qualifications retained from the review, now stated in code or receipts: QSE pencil parity is not a physical spectral certificate (`basis_spans_sector` recorded; roots from the CPU core); diagnostic APM `cpu_verified` is not accuracy acceptance (no drift/fidelity gate; `trajectory_moved` recorded); full RA-ADAPT/SNAKE stays facade-blocked (`full_route_status=blocked`); GPU APM unsupported by construction; mps float64 refused by the device; no CUDA.

**Scientific consequence of finding 1, reported, not decided:** after the executed-unitary audit the repository `full_meta` molecular pool offers UCC generators (odd-Y; safe under `logical_shared` only) and Z-only Hamiltonian words; the Hamiltonian blocks (even-Y, real) fail because their words do not mutually commute, and single X/Y words fail outright. From a real HF reference the McLachlan force of odd-Y generators vanishes, so the audited trajectory does not move (`trajectory_moved=False`) and its fidelity trace measures the exact state's departure from HF (contract4 0.99997516 at t=0.1; random6 0.99937 at t=0.02, 0.9845 at t=0.1). Which executed-safe pool a molecular Paper II lane should use (e.g. the same-spin hopping-pair candidates of the r2 reconciliation, or grouped-exact execution of the blocks, which the APM runtime does not currently admit) is unselected and is Jake's decision.

## 3. Tests and receipts (exact commands)

```sh
cd /Users/lordskyler/Holstein_test.worktrees/claude-doron-bridge-20261019
export PYTHONPATH=. STATIC_ADAPT_HH_POOL_CACHE=off PYTHONDONTWRITEBYTECODE=1 PYDEPHASING_TESTING=1
# repository venv:   66 passed, 18 skipped in 27.37 s (skips: torch not importable)
/Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q -p no:randomly -rs test/test_chemistry_bridge_export.py test/test_chemistry_bridge_backend.py test/test_chemistry_bridge_methods.py test/test_chemistry_bridge_cli.py test/test_chemistry_bridge_gpu_parity.py
# GPU venv:          75 passed, 9 skipped in 30.86 s (skips: mps float64 refused by the device; cuda absent)
/Users/lordskyler/.venvs/doron-bridge-gpu/bin/python -m pytest -q -p no:randomly -rs test/test_chemistry_bridge_export.py test/test_chemistry_bridge_backend.py test/test_chemistry_bridge_methods.py test/test_chemistry_bridge_cli.py test/test_chemistry_bridge_gpu_parity.py
# strict-intake gate (unchanged): 80 passed
/Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q -p no:randomly test/test_molecular_diagnostic_intake.py "test/test_static_adapt_builders.py::test_build_hamiltonian_blocks_pool_matches_molecular_grouped_builders" "test/test_adapt_vqe_integration.py::TestAdaptResolvedProblemContext::test_molecular_full_meta_pool_is_problem_local_mega_pool"
```

Per-file counts (repository venv): export 41, backend 10, methods 8, cli 5, gpu_parity 2 passed / 18 skipped.

Receipts regenerated under the corrected code in the GPU venv (the exact argv, start/end UTC, wall time, return code, stdout/stderr tails of all 22 commands are in `doron-bridge-receipts-20261019-002/command_log.json`; every file's SHA-256 is listed in §5). Earlier receipts in `doron-bridge-receipts-20261019/` are the `82fe6bc3` evidence and are untouched.

| category | witness | result |
|---|---|---|
| synthetic NumPy (CPU, all five methods, filtered APM) | contract4 / random6 / random6_two_occupied(fc=1) | exit 0; all `cpu_verified`; `data_role=synthetic:in_memory_mapping`; APM `trajectory_moved=False`, leakage 0; eligible generators 14/19, 29/127, 13/31 |
| torch CPU float64 control | contract4 / random6 / random8 | all `cpu_verified`, `exercised_device=cpu`; HF-witness, exact-target, QSE-pencil and gradient parity 0 … 5.3e-15 |
| actual Metal, `torch:mps:float32` | contract4 / random6 / random8 | all `gpu_verified`, `exercised_device=mps`; parity: HF 1.9e-7 / 2.7e-7 / 1.6e-6, exact E0 1.9e-7 / 4.6e-9 / 3.6e-7, QSE H pencil 1.9e-7 / 2.7e-7 / 1.6e-6, gradients 1.5e-9 / 8.3e-8 / 9.2e-7 (float32 tolerances 5e-5 scalar / 2e-4 matrix) |
| actual HDF5 reading (synthetic file in the producer schema, h5py 3.16.0) | `synthetic_random6_ele.h5`, sha256 `e6fe97b0debd8e40033ea79a60be7fed14672d54a0102692cfce31593b7047b9` | `export-qdynamics --declared-origin synthetic` → `data_role=synthetic:qdynamics_matrix_elements_ele_h5`, four methods `gpu_verified` on MPS; the same file undeclared → `origin=unknown` (manifest `synthetic_ele_h5_undeclared.manifest.json`) |
| CLI exit semantics (review probes) | contract4 / random8 | `--methods qes` → 2 (no receipt written); `--methods ""` → 2; `--backend torch:cuda:float64` → 3 with `skipped_unsupported` receipt; per-word unfiltered APM on random8 → 1 with `failed` receipt (leakage 3.3e-5) |
| review leakage case | random6, logical_shared, cap 128, `mclachlan_l2`, cut 1e-6 | unfiltered `failed`, leakage 1.236228e-4, support 2→6; filtered `cpu_verified`, leakage 0, 29/127 eligible, support 1 (`review_case_random6_logical_shared.json`) |
| REAL Doron-produced input | — | **none exists on this machine; blocked** (unchanged) |

CUDA, float64 GPU and GPU APM: none claimed; mps float64 is refused by the device.

## 4. Whole-increment clocks (actual machine UTC, `date -u`; no reset)

| increment | start | end | wall | commit |
|---|---|---|---|---|
| packet hash verification and full read | ≈ 17:05 | 17:09:01 | ≈ 4 min, not clocked | — |
| 002-A findings 2, 3, 6 | 2026-10-19T17:09:01Z | 2026-10-19T17:15:42Z | 401 s | `404454ac` |
| 002-B findings 1, 4, 5 (+ full suites in both venvs) | 2026-10-19T17:15:42Z | 2026-10-19T17:22:05Z | 383 s | `16e12761` |
| 002-C receipts regeneration, this disposition, main-handoff amendments, graph check, commit | 2026-10-19T17:22:05Z | recorded in the 002-C commit message and appended below | | |

The `.6NZ` suffix in the `404454ac` message is a literal from an unsupported `date` directive; the clock has whole-second precision.

## 5. Receipt and document identities (SHA-256)

Computed at commit time by `shasum -a 256` over `agent_guidance/shared/handoffs/doron-bridge-receipts-20261019-002/*.json`; the list is appended below in the 002-C record so that this file does not try to contain its own hash.

## 6. Remaining blockers and unselected decisions (unchanged in kind)

1. No pinned Doron-produced `_ele.h5` export with declarations (geometry/units, basis, SCF thresholds, bohr constant, method/spin) → real-data integration blocked; `origin` can only become `attested` with a named attestation covering the file hash.
2. Intended system, basis, active space (full H2O/STO-3G is 14 qubits; the declared frozen-core contraction is offered, not chosen), lane ownership, data-license statement.
3. Paper I: typed facade rejects the family; a named lane-owned application is a separate, gated, CRITICAL-impact increment.
4. Paper II: an executed-safe even-Y molecular pool (or an admitted grouped-exact execution) is unselected; a dipole drive is a separate input.
5. Paper III: R, sectors, observables, retained-frame/conditioning/physical-residual qualification.
6. GPU: Metal float32 only; no CUDA, no float64 GPU, no GPU APM.

Files edited in 002: `pipelines/chemistry_bridge/{qdynamics_export.py, methods.py, __main__.py}`, `test/test_chemistry_bridge_{export,backend,methods,gpu_parity}.py`, new `test/test_chemistry_bridge_cli.py`, this file, `doron-chemistry-quantum-bridge-20261019.md` (§4 Paper II row, §9), new `doron-bridge-receipts-20261019-002/`.

## 7. 002-C record (appended after the commit)

| item | value |
|---|---|
| 002-C commit | `f0892355beec17dc8c4ffae836e42c7ccdc0c0ee` |
| 002-C clock | start 2026-10-19T17:22:05Z, end 2026-10-19T17:26:55Z |
| Correction commits, in order | `404454ac` → `16e12761` → `f0892355beec17dc8c4ffae836e42c7ccdc0c0ee` (→ this record commit) |

Receipt and manifest SHA-256 (`shasum -a 256`), as committed in `f0892355beec17dc8c4ffae836e42c7ccdc0c0ee`:

```text
a61ebf8a36d2b4ca293336c1d05cba7f7fb5adda71e3c4d5af58c544c8f28d28  doron-bridge-receipts-20261019-002/cli_apm_per_pauli_term_unfiltered_exit1.json
24a289a842f63b5b04ab47e52b57a132b2c682e6ae5e1ad905f878479c71ec46  doron-bridge-receipts-20261019-002/cli_cuda_unavailable_exit3.json
8e2c20488725fcba6d2b354cbc055420a46ba2aa61c4bdf7fad1b24419cd9fba  doron-bridge-receipts-20261019-002/command_log.json
c781e13830a442256fd0a6f71926868fa45dd88af90d96ac346ffd768d08b71c  doron-bridge-receipts-20261019-002/contract4_numpy_float64_all_methods.json
05a233114cfa7a16a2e071aacffa5d1bb527ff19a17e27defc9e1d53706895fa  doron-bridge-receipts-20261019-002/contract4_torch_cpu_float64.json
d9ee8bb80d3d405e86637f8ff995e4bd2cde2f8a07d746a07bbbdf9726a0a7d5  doron-bridge-receipts-20261019-002/contract4_torch_mps_float32.json
129ff176749c2f6093f323c754fd8f195278d0ee3696539b51771b11da58899c  doron-bridge-receipts-20261019-002/probe_backends_gpu_venv.json
29d63e81687b976aafa2880f12305646127a5952894c380b06b84a346777b287  doron-bridge-receipts-20261019-002/random6_numpy_float64_all_methods.json
fdc5576fcb1630471fcad057cdb86e3dbd459563dd15539d69cef8a1da5e3ba0  doron-bridge-receipts-20261019-002/random6_torch_cpu_float64.json
579a33c8dcc1782835a86558f8bd05e1a7d2e281cd106fffe89ffa5dd60c518a  doron-bridge-receipts-20261019-002/random6_torch_mps_float32.json
b6fec127db381e74416341b34d15c3a1198f26ded35b9213277c63af95444342  doron-bridge-receipts-20261019-002/random6_two_occupied_numpy_float64_all_methods.json
ac08a8fb134fad3c8e4dda3b363dcba3b25b974a35416e8c923ba945929c7a23  doron-bridge-receipts-20261019-002/random8_torch_cpu_float64.json
7370da11c6b00755f1b0ea08c5d4b1bb2fb5065711b090747ba68623c3e11b9c  doron-bridge-receipts-20261019-002/random8_torch_mps_float32.json
e9468d5c590528ce7db574f86e0b0cc446fbeb9c2c80d3ae4b66e74257e68802  doron-bridge-receipts-20261019-002/review_case_random6_logical_shared.json
884575d56f451c2897008f4648051f3698b1aca8c9006f3a540171a8e0089fd1  doron-bridge-receipts-20261019-002/synthetic_ele_h5_declared_synthetic.manifest.json
8d7eef246e06950c4cbfb39babf909c4367ab44f361c33d3c2c07ffd3ca3abf7  doron-bridge-receipts-20261019-002/synthetic_ele_h5_declared_synthetic_torch_mps_float32.json
3d79f3f4a97de2da42af9f7dac8e465418fa303e2b41c2bf6d669712a6fd9557  doron-bridge-receipts-20261019-002/synthetic_ele_h5_undeclared.manifest.json
```

Main handoff `doron-chemistry-quantum-bridge-20261019.md` at `f0892355beec17dc8c4ffae836e42c7ccdc0c0ee`: sha256 855a2aadc92af509e14d82634286fad1fedf53e8cb71c0b9d9d04872ffe80091

## 8. Recheck 1 response — increment 002-D (2026-10-19)

The existing reviewer rechecked HEAD `00a9c844` and reported: findings 1–5 **resolved** for the reported failures, item g (Metal receipts) **supported**, finding 6 **partially resolved**, and one low-severity CLI discrepancy. The statement "all six corrected" in §2 was therefore overstated at `00a9c844`; the residuals are closed here and finding 6 is now claimed complete *pending the reviewer's recheck*, not self-certified.

| residual (reviewer) | disposition in 002-D | regression |
|---|---|---|
| **[P2] FCIDUMP public API**: `FcidumpSidecar(origin="attested")` was accepted without an attestation and propagated to `attested:fcidump_text` with `origin_attestation=None`. | **Closed.** `FcidumpSidecar` gains `attestation`; `origin="attested"` requires `{"attested_by": <who>, "text_sha256": <64-hex>}` and `bridge_problem_from_fcidump` refuses an attestation whose `text_sha256` differs from the parsed text's sha256 (`origin_attestation`); an attestation without `origin="attested"` is refused; the record is carried into provenance and receipts; `producer_origin_established` stays `False`. | `test_finding6_fcidump_attested_origin_requires_an_attestation_over_the_text` (refused without attestation, refused with wrong hash, accepted with matching hash and carried end to end: `attested:fcidump_text`; undeclared stays `unknown`) |
| **Exposed loader hooks**: `qdynamics_matrix_elements_from_mapping(_container_schema=…, _file_sha256=…)` let a caller fabricate a file container and self-attest without reading a file. | **Closed.** The public function's signature is now `(mapping, *, source, declared_origin, declared_producer)`: no container, no file hash, no attestation (an in-memory mapping can never be attested). The loader and the mapping path share a private constructor `_matrix_elements_from_payload` that refuses a file container without the loader-computed sha256 and a file sha256 without a file container (`container_claim`). | `test_finding6_public_mapping_api_cannot_name_a_file_container_or_attest` (signature asserted; the forged-container and attestation calls raise `TypeError`; the private constructor refuses both inconsistent combinations) |
| **CLI**: a nonexistent manifest escaped as an uncaught `FileNotFoundError` (exit 1, no receipt), contradicting the usage-error/failed-check distinction. | **Closed.** `run` and `export-qdynamics` classify `FileNotFoundError`/`OSError`/`KeyError`/`ValueError` as usage errors (exit 2, no receipt); the docstring says so. | `test_cli_exit_two_when_the_manifest_is_missing` (run → 2 with `FileNotFoundError` named, no out dir; export → 2) |

Reviewer's scope statements retained: GPU evidence supports kernel/pencil parity only; authoritative QSE roots remain CPU; APM remains diagnostic; a useful moving molecular APM pool, real Doron data, named RA application, dipoles, QSE roots/sectors, CUDA/float64 GPU/GPU APM all remain blocked or unselected as in §6.

Gates after 002-D (same commands as §3): repository venv **69 passed, 18 skipped**; GPU venv **78 passed, 9 skipped**; intake gate **80 passed**. Probe receipts for the three residuals (attested FCIDUMP without/with wrong/with matching hash, public-API forgery attempts, missing manifest and missing HDF5 exits) are in `doron-bridge-receipts-20261019-002/recheck1_residual_probes.json`.

002-D clock: start 2026-10-19T18:17:58Z; end and commit recorded below after the commit.
