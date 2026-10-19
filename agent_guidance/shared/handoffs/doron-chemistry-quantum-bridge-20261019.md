# Doron chemistry–quantum bridge: implementation and verification handoff (2026-10-19)

Thread: "Doron chemistry–quantum bridge" (message
`doron-chemistry-quantum-bridge-20261019-001`, Jake). Author: Claude (Fable 5.1).
Verifier: a separate Codex 6.1 Ultra thread arranged by the coordinator from
this document. Nothing here is self-certified as independently verified.

## 0. Negative results, stated first

1. **No Doron-produced data file exists on this machine or in any checkout.**
   Bounded searches of `~/Documents/ChatGPT`, `~/Downloads`, `~/Desktop`,
   `Holstein_test` and every worktree for `*fcidump*`, `*doron*`, `*.h5`,
   `*.hdf5`, `*molden*` found nothing. The only Doron-attributed numbers in the
   repository are the scalar slide row in
   `pipelines/exact_bench/audit_h2o_cross_code_reference.py:25-43`
   (`EXTERNAL_BENCHMARK`, E_SCF −74.9630231385 Ha and three frequencies): a
   scalar energy, not a state or a Hamiltonian. **Real-data acceptance is
   therefore blocked**, and every method result below is on labelled synthetic
   witnesses.
2. **Doron's actual export format is not FCIDUMP.** At the pinned commit
   (§2) the PySCF driver writes only `<prefix>_ele.h5` with *interleaved
   spin-orbital* `h1`/`h2` (chemists' Coulomb, not antisymmetrized). The
   2026-10-14 plan's FCIDUMP-first reader (B2) would not ingest it. The bridge
   adapter targets the HDF5 schema; an FCIDUMP reader is kept as a secondary,
   declaration-bound path.
3. **The Paper-I RA-ADAPT/SNAKE route rejects the electronic molecular family
   at HEAD.** `pipelines/static_adapt/ra_adapt/engine.py:5742-5757` admits only
   HH L∈{2,3}, named Hubbard applications and `molecular_vibronic_h2o_linear_fd`;
   the bridge calls the facade and records its verdict (`"locked to the
   canonical Hubbard--Holstein L=2 problem"`). The historical static
   `adapt_pipeline` CLI is retired at HEAD (`adapt_pipeline.py:30520` raises).
   Paper I status is **blocked**; the bridge delivers the repository pool plan
   and the ADAPT selection gradients only.
4. **GPU execution happened only after the §7 install, in an isolated venv,
   on Apple Metal (`torch:mps:float32`).** The repository venv
   (`/Users/lordskyler/Holstein_test/.venv`) still has no torch/h5py and its
   GPU cases skip with the recorded reason. In the isolated venv the MPS cases
   ran and passed at float32 tolerances (§5.5); `mps:float64` is refused by the
   device itself and `cuda` does not exist on this machine. No GPU result is
   float64, and no CUDA parity exists.
5. **Full H2O/STO-3G (7 orbitals, 14 qubits) exceeds the strict intake's
   12-qubit scope cap.** The bridge offers a *declared* frozen-core contraction
   (`scalar_semantics="nuclear_plus_frozen_core"`, `already_contracted`) so a
   full-space Doron export can enter as CAS(8e,6o); whether that is the
   intended problem is the user's call (§8).
6. **The reader workflow's baseline reds are confirmed, not new:** the
   unchanged-route sample (§5.3) reports 3 failures, all missing-fixture
   failures that predate this branch (LiH JSON and vibronic-H2 fixture absent at
   `d1436ab1`, as recorded in the 2026-10-14 plan).

## 1. Anchors

| field | value |
|---|---|
| Worktree | `/Users/lordskyler/Holstein_test.worktrees/claude-doron-bridge-20261019` |
| Branch | `claude/doron-chemistry-quantum-bridge-20261019` |
| Base | `main` at `d1436ab12978fa45e9f767132558069d98b6dee9` (2026-10-19) |
| Reconciled intake | `2e876e7d892dd87b0c3e4b000ba80ad443f4349d` = cherry-pick (`-x`) of `393e335a65405481e29280210d3e714acd4ce9ed` from `codex/chemistry-papers-i-iii-groundwork-20261010` (same patch-id); adds `pipelines/static_adapt/builders/molecular_diagnostic_intake.py`, 19 lines in `problem_registry.py`, `test/test_molecular_diagnostic_intake.py`, and the Codex increment report |
| Not applied | `d81c5481` (H2 fixture repair, unmerged, user decision 2026-10-14 §8 item 5); the Codex docs-only commits; any Paper V handoff |
| Python | `/Users/lordskyler/Holstein_test/.venv/bin/python` (3.12.14; numpy 2.3.5, scipy 1.16.3, qiskit 2.3.1); run from the worktree with `PYTHONPATH=.` (no editable install; `pipelines` resolves to the worktree) |
| Machine | Apple M4 Pro, 14 CPU cores, 24 GiB, Metal 4 GPU (20 cores); memory-pressure free 35% at start; no RAM guard used; every run here is seconds on 4–8 qubits |
| Intake gate | `80 passed in 2.26s` (unchanged from the Codex report) |
| Bridge gate | `47 passed, 18 skipped in 13.79s` (§5.1) |
| GitNexus | main index at `d1436ab1` is older than this worktree; the worktree was indexed in place before `detect_changes` (§5.4). No existing function was edited by this branch beyond the cherry-picked extraction; the CRITICAL/HIGH upstream counts for `apply_compiled_polynomial` (231), `compute_qse_spectra` (84) and `run_append_mclachlan_trajectory` (12) are callers of symbols the bridge only *calls* |

## 2. Doron upstream evidence (primary source, read-only)

Read over the GitHub REST/raw API; nothing cloned, installed, imported or
executed. Fetched files are in the session scratchpad only, treated as data.

| item | value |
|---|---|
| Profile | `github.com/doronwu` owns six repositories: `qdynamics` (fork, main `c2b942f0` only), `shci` (fork, default `old-doron`), `SHCI-SZ` (created 2026-10-12), `pyqmc` (fork), `West` (fork), `MoireLattice` |
| Producer | `https://github.com/jacopos86/qdynamics`, branch `doron-chemistry`, tip `e5ec5bc45d02169e77ce93b05126c9804a03cdb6`, author `doronwu <dwu269@wisc.edu>`, 2026-08-26T21:02:57Z, message `tested-eph`; repository license GPL-3.0; sibling branch `doron-pyscf-archive` `10e9e242` |
| Writer | `src/chemistry/pyscf/read_write_pyscf_results.py::write_matrix_elements_h5` (blob `3bf71a4d0d2c`) |
| Driver | `src/chemistry/pyscf/main_pyscf.py` (blob `594293ae21d9`): runs SCF, refuses to write on non-convergence, writes `<prefix>_ele.h5`; optional `_vib.h5`, `_eph.h5` |
| Integrals | `src/chemistry/pyscf/electron_solver.py::PySCFDriver.get_matrix_elements` (blob `8400fc64c5ac`): `h_mo = C^T (T+V) C`, `eri_mo = ao2mo.kernel(mol, mo, compact=False)` (chemists' (pq|rs)); `h1[0::2,0::2]=h1[1::2,1::2]=h_mo`; `h2[s::2,s::2,s'::2,s'::2]=eri_mo` |
| Scalar | `nuclear_repulsion = mol.energy_nuc()` (nuclear only; the producer has no active-space or frozen-core option) |
| Metadata attrs | `mol_str, basis, method, charge, spin, unit, xc, e_scf, converged, nelec, nmo, ms2, e_converg(1e-12), d_converg(1e-8), max_iter, isotope_masses`, plus `h1_convention='spin_orbital_interleaved'`, `h2_convention='chemist_spin_orbital_coulomb'`, `h2_antisymmetrized=False` |
| FCIDUMP | `input_parser.py` accepts `write_fcidump`/`fcidump_tol` but `main_pyscf.py` never writes one; the Psi4 driver writes the same `ele.h5` schema |
| Unresolved | which system (H2O at the inferred G3? H2 at 0.7414 Å? a solid?), basis (STO-3G is inferred from an energy match), the commit that produced the 2026-07-14 slide, whether `doronwu` is the slide's author, and whether his outputs may be committed under `test_support/` (GPL-3.0 code; data license unstated) |

## 3. What was implemented (all new files, opt-in; no canonical default changed)

```
pipelines/chemistry_bridge/__init__.py            package note, BRIDGE_SCHEMA_VERSION
pipelines/chemistry_bridge/qdynamics_export.py    Doron _ele.h5 schema -> strict-intake document (+ FCIDUMP secondary path, frozen core)
pipelines/chemistry_bridge/array_backend.py       NumpyBackend / TorchBackend with live-device reports; no silent fallback
pipelines/chemistry_bridge/polynomial_action.py   matrix-free Pauli-polynomial action on a backend; sector matrix; ADAPT gradients; QSE pencil
pipelines/chemistry_bridge/methods.py             strict intake -> mapping witness, exact reference, QSE, ADAPT screening (+facade gate), AP-McLachlan; receipts
pipelines/chemistry_bridge/synthetic_witnesses.py labelled synthetic fixtures in the producer's schema
pipelines/chemistry_bridge/__main__.py            CLI: probe-backends | export-qdynamics | export-synthetic | run
test/test_chemistry_bridge_export.py              29 tests   conventions, negative controls, frozen core, FCIDUMP, h5py path
test/test_chemistry_bridge_backend.py             10 tests   kernel/dense/independent-kron parity, sector, status rule, spec parsing
test/test_chemistry_bridge_methods.py              6 tests   end-to-end on synthetic witnesses; leakage witness; unsupported backend
test/test_chemistry_bridge_gpu_parity.py          20 cases   2 run without torch; 18 skip with the backend's own reason
```

### 3.1 Conventions, as implemented and checked

| topic | rule in the bridge | where checked |
|---|---|---|
| Spin-orbital order | producer interleaved `(a0,b0,a1,b1,…)` → spatial tensors extracted from the alpha block; the repository consumes blocked alpha-then-beta JW order (`molecular_hamiltonian.py`) | `spatial_tensors_from_interleaved`: cross-spin `h1` blocks zero, alpha==beta block, the four allowed `h2` placements equal, every spin-forbidden `h2` entry zero (codes `h1_cross_spin_nonzero`, `h1_spin_blocks_differ`, `h2_spin_block_placement`, `h2_spin_forbidden_nonzero`) |
| Two-body notation | chemists' `(pq|rs)`, real, bound by the producer's `h2_convention` and `h2_antisymmetrized=False` metadata; symmetry is checked (three generators of the eightfold group) but is **necessary, not sufficient** (a physicists' tensor of a degenerate system can pass) | codes `h2_convention`, `h2_antisymmetrized`, `eri_symmetry`; test `test_physicists_ordered_two_body_tensor_fails_symmetry_for_generic_tensors` |
| Prefactors | the repository builder's `H = E I + Σ h_pq a†a + ½ Σ (pr|qs) a†_p a†_q a_s a_r` is reused unchanged; the HF witness `E_det = E + 2Σh_ii + Σ(2g_iijj − g_ijji)` equals `e_scf` | `closed_shell_determinant_energy`; strict intake re-checks at `atol=1e-12`; mapped HF expectation equals the witness to 2e-16 (§5.2) |
| Scalar / core | nuclear-only by default; declared `nuclear_plus_frozen_core` with `already_contracted` one-body when `FrozenCoreReduction(n)` is requested; contraction `E_core = E_nuc + 2Σ_c h_cc + Σ_cd (2(cc|dd)−(cd|dc))`, `h'_pq = h_pq + Σ_c (2(pq|cc)−(pc|cq))` | parity with `generate_h2o_linear_fd_fixture._freeze_core_active_tensors` (1e-12) and a spectrum identity: reduced (1,1) sector == full Hamiltonian restricted to core-doubly-occupied determinants (1e-10) |
| Units / frame | hartree; orthonormal canonical MOs; occupied prefix verified from `mo_occ` when present (else recorded as unverified) | code `occupied_prefix`; provenance `assumptions_not_verified_here` |
| Reference / sector | RHF determinant (`hartree_fock_statevector`), fixed `(n_alpha, n_beta)` sector; the action preserves the sector (leakage ≤ 7e-18) | `run_mapping_witness`, `sector_matrix` |
| Root identity | the exact reference is the lowest eigenvalue of the sector block assembled by the bridge action and diagonalized on CPU; it equals the repository's dense `exact_ground_energy_sector` to ≤ 5.3e-15 | `run_exact_reference` |
| QSE pencil | mirrors `build_qse_matrices` (projection none/q0, normalized/raw_projected, Hermitian symmetrization); Ritz values compared with exact roots only when the retained basis spans the sector (`basis_spans_sector`) | `qse_pencil`, `run_qse` |
| Executed-unitary conservation | AP-McLachlan default `parameterization_mode="logical_shared"`; measured: per-Pauli-word coordinates leak particle number on the 8-qubit witness (3.3e-5 at t=0.02, 0.12 at t=0.10) while logical_shared stays ≤ 1e-16 | `run_ap_mclachlan` leakage check; test `test_per_pauli_term_coordinates_can_leak…` |
| Provenance chain | HDF5 file sha256 + dataset sha256 → bridge JSON (`raw_sha256` in the manifest) → strict intake receipt (`raw/tensor/semantic/operator_sha256`) → every method receipt (`inputs`) | `write_bridge_problem_json`, `BridgeContext.input_identity`; editing the JSON fails the intake (`test_editing_the_problem_document_fails_the_strict_intake`) |

### 3.2 Backend and status rules

* `BackendReport.exercised_device` is read from a live tensor; a request that
  lands elsewhere raises `BackendUnavailable` at construction and, if it ever
  reached a receipt, `verified_status` returns `failed` (never `cpu_verified`).
* `gpu_verified` requires `exercised_device in {cuda, mps}` **and** passing
  parity checks at `parity_tolerances(real_dtype)` (float64: 1e-11 vector/scalar,
  1e-10 matrix; float32: 5e-5 vector/scalar, 2e-4 matrix).
* `TorchBackend` stores complex vectors as real `(2, N)` tensors so `mps`
  (no float64, partial complex kernels) is exercised with float32 real
  arithmetic; `float64` on `mps` is reported unavailable with the device's own
  error text.
* The CPU float64 action reproduces the repository kernel
  (`apply_compiled_polynomial`), the dense `hamiltonian_matrix`, and an
  independent Kronecker construction written in the test, to 2.6e-16.

## 4. Lane status (precise)

| lane | route executed | status | GPU |
|---|---|---|---|
| Shared input | strict intake A (`resolve_molecular_diagnostic_intake`) from the bridge document; mapping witness and exact sector reference on the backend | CPU verified on 4 synthetic witnesses (HF defect 0.0); a producer-schema `_ele.h5` file went through the real loader (§5.5) | mapping witness and exact reference **gpu_verified** on `torch:mps:float32` |
| Paper III (QSE) | `pipelines.qse_spectra.core.compute_qse_spectra` on the HF reference with identity + UCC generators | **cpu_verified**; Ritz roots equal exact sector roots when the basis spans the sector (4 qubits: 4.4e-16; 6 qubits: 2.9e-15); backend pencil parity ≤ 4.9e-15 | **gpu_verified** on `torch:mps:float32` (pencil parity ≤ 1.6e-6 abs on the 8-qubit witness, §5.5); skipped in the repository venv (no torch) |
| Paper II (AP-McLachlan) | `run_append_mclachlan_trajectory` (generalized exchange, zero drive, rk4, logical_shared, repository `full_meta` molecular pool, seed `ham_block::molecular_one_body` at θ=0) | **cpu_verified**; 4-qubit witness: fidelity vs exact sector propagation ≥ 0.9999999999992 to t=0.1, energy drift 8.7e-8, leakage 0; 6-qubit: fidelity 0.99937 to t=0.02 | **unsupported by construction** (dense numpy geometry; not attempted) |
| Paper I (RA-ADAPT/SNAKE) | `resolve_pool_plan` (phase3_v1, `full_meta`) + ADAPT selection gradients `g_k = 2 Im⟨Hψ|A_kψ⟩`; `run_ra_adapt` called once to record the gate | gradients **cpu_verified** (parity with the repository formula 0.0); full route **blocked** by the facade gate (named lane-owned application required: r2 I-2b) | gradients **gpu_verified** on `torch:mps:float32` (parity ≤ 9.2e-7 over a 388-generator pool); the blocked route has no GPU meaning |
| Real Doron data | — | **blocked**: no pinned `_ele.h5` export, no declarations (geometry/basis/units/thresholds), no data license statement | — |

## 5. Evidence

### 5.1 Test gate (worktree, 2026-10-19)

```sh
cd /Users/lordskyler/Holstein_test.worktrees/claude-doron-bridge-20261019
PYTHONPATH=. STATIC_ADAPT_HH_POOL_CACHE=off PYTHONDONTWRITEBYTECODE=1 PYDEPHASING_TESTING=1 \
/Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q -p no:randomly \
  test/test_chemistry_bridge_export.py test/test_chemistry_bridge_backend.py \
  test/test_chemistry_bridge_methods.py test/test_chemistry_bridge_gpu_parity.py
# 47 passed, 18 skipped in 13.79s   (export 29, backend 10, methods 6, gpu_parity 2 passed / 18 skipped)
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYDEPHASING_TESTING=1 /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q -p no:randomly \
  test/test_molecular_diagnostic_intake.py \
  "test/test_static_adapt_builders.py::test_build_hamiltonian_blocks_pool_matches_molecular_grouped_builders" \
  "test/test_adapt_vqe_integration.py::TestAdaptResolvedProblemContext::test_molecular_full_meta_pool_is_problem_local_mega_pool"
# 80 passed in 2.26s
```

Skip reasons are the backend's own (`torch:mps:float32 unsupported here: torch
is not importable in this environment`); none is a silent pass.

Same suite in the isolated GPU venv (§7, executed 2026-10-19):

```sh
PYTHONPATH=. STATIC_ADAPT_HH_POOL_CACHE=off PYTHONDONTWRITEBYTECODE=1 PYDEPHASING_TESTING=1 \
/Users/lordskyler/.venvs/doron-bridge-gpu/bin/python -m pytest -q -p no:randomly -rs \
  test/test_chemistry_bridge_export.py test/test_chemistry_bridge_backend.py \
  test/test_chemistry_bridge_methods.py test/test_chemistry_bridge_gpu_parity.py
# 56 passed, 9 skipped in 21.64s
# skipped: torch:mps:float64 (mps rejected dtype float64: TypeError ... MPS framework doesn't support float64),
#          torch:cuda:float64 and torch:cuda:float32 (torch.cuda.is_available() is False)
```

The 9 extra passes are the `torch:cpu:float64`, `torch:cpu:float32` and
`torch:mps:float32` parity cases plus the h5py round-trip branch of
`test_h5_loader_is_explicit_about_its_dependency`.

### 5.2 Receipts on the four synthetic witnesses (CPU, numpy float64)

```sh
S=/tmp/bridge_runs   # any scratch directory
for spec in contract4:0 random6:0 random6_two_occupied:1 random8:0; do k=${spec%%:*}; fc=${spec##*:}
  PYTHONPATH=. /Users/lordskyler/Holstein_test/.venv/bin/python -m pipelines.chemistry_bridge export-synthetic --kind $k --frozen-core $fc --out $S/$k/problem.json
  PYTHONPATH=. STATIC_ADAPT_HH_POOL_CACHE=off /Users/lordskyler/Holstein_test/.venv/bin/python -m pipelines.chemistry_bridge run --manifest $S/$k/problem.manifest.json --out-dir $S/$k/out --backend numpy:float64
done
```

| witness | qubits, sector | HF witness defect (intake) | mapped HF − witness | exact E0 (bridge) vs repository | QSE Ritz − exact (spans?) | ADAPT grad parity, pool | APM (to t=0.10): drift, min fidelity, leakage |
|---|---|---|---|---|---|---|---|
| contract4 (contract §5.4) | 4, (1,1) | 0.0 | 2.2e-16 | −1.3008925726121896, Δ 0.0; roots = `0.1∓√785/20, −0.15, −0.05` | 4.4e-16 (yes, 4/4) | 0.0, 19 | 8.7e-8, 0.99999999999919, 0 |
| random6 | 6, (1,1) | 0.0 | 2.2e-16 | −1.750065895928, Δ 3.1e-15 (dim 9) | 2.9e-15 (yes, 9/9) | 0.0, 127 | 1.9e-4, 0.98450, 6e-20 (0.99937 to t=0.02) |
| random6_two_occupied, frozen core 1 | 4, (1,1) | 0.0 | 0.0 | 7.394922940563, Δ 0.0 (dim 4) | 2.8e-14 (yes, 4/4) | 0.0, 31 | 0.26, 0.9973, 0 |
| random8 | 8, (2,2) | 0.0 | 0.0 | −14.764431491473, Δ 5.3e-15 (dim 36) | 2.32 (no: 27/36, upper bounds) | 0.0, 388 | 1.83, 0.757, 7e-17 (large synthetic coefficients; coarse grid) |

Facade gate recorded on every witness:
`admitted=False`, message `The ordinary Paper-I RA-ADAPT facade is locked to the
canonical Hubbard--Holstein L=2 problem. Other families and sizes require an
explicitly named lane-owned application.`

Each run writes `out/bridge_receipts.json` with the intake receipt (raw, tensor,
semantic, operator SHA-256), the backend report (library, version, requested
and exercised device, dtype) and per-method checks.

### 5.3 Unchanged-route regression sample

```sh
PYTHONPATH=. STATIC_ADAPT_HH_POOL_CACHE=off PYTHONDONTWRITEBYTECODE=1 PYDEPHASING_TESTING=1 \
/Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q -p no:randomly -rf \
  test/test_molecular_diagnostic_intake.py test/test_static_adapt_builders.py \
  test/test_ap_mclachlan_avqds.py test/test_ap_mclachlan_trajectory.py test/test_qse_exact_reference.py \
  "test/test_adapt_vqe_integration.py::TestAdaptResolvedProblemContext"
# 3 failed, 146 passed in 3.04s
```

The 3 failures are `test_molecular_vibronic_h2_fixture_context_and_pool_smoke`,
`test_lih_molecular_fixture_loads_as_closed_shell_problem`,
`test_lih_real_builder_resolves_context_hamiltonian_and_pool_smoke`: missing
`test_support/molecular_problem_lih_sto3g.json` and the vibronic-H2 fixture at
`d1436ab1` (pre-existing; recorded in the 2026-10-14 plan). No existing module
is modified by this branch, so the broader suites are unaffected by
construction; they were not rerun in full (the facade suite alone is 79 s and
carries its own pre-existing red).

### 5.4 Graph receipts

GitNexus `impact` (main index): `_resolve_molecular_problem_context` UNKNOWN
(0 resolved callers; it is dispatched through the registry dict — confirmed by
text search and by the intake gate), `compute_qse_spectra` CRITICAL (84),
`apply_compiled_polynomial` CRITICAL (231), `run_append_mclachlan_trajectory`
HIGH (12). None of these is edited; the bridge adds callers. `detect_changes`
over the in-place worktree index is recorded in the commit message of the
implementation commit.

### 5.5 GPU parity receipts (isolated venv, `torch:mps:float32`, control `torch:cpu:float64`)

Receipts are committed under `agent_guidance/shared/handoffs/doron-bridge-receipts-20261019/`
(`<witness>_torch_mps_float32.json`, `<witness>_torch_cpu_float64.json`,
`probe_backends_gpu_venv.json`, `synthetic_ele_h5_*.json`). Every MPS receipt
carries `backend.exercised_device = "mps"`, `real_dtype = "float32"`,
`library_version = "2.14.1"`; every CPU-control receipt `exercised_device = "cpu"`,
`float64`. Status rule: `gpu_verified` only from a live `mps`/`cuda` tensor
(§3.2).

| witness (qubits) | method | MPS float32 vs reference | torch CPU float64 vs reference |
|---|---|---|---|
| contract4 (4) | mapped HF − tensor witness | 1.9e-7 | 2.2e-16 |
| contract4 (4) | exact E0 − repository target | 1.9e-7 | 0.0 |
| contract4 (4) | QSE pencil H / S − core | 1.9e-7 / 0.0 | 2.2e-16 / 0.0 |
| contract4 (4) | ADAPT gradients − repository (19 generators) | 1.5e-9 | 0.0 |
| random6 (6) | mapped HF − tensor witness; Hermiticity; sector leakage | 2.7e-7; 1.9e-8; 1.7e-8 | 2.2e-16; 7.9e-17; 6.9e-18 |
| random6 (6) | exact E0 − repository target | 4.6e-9 | 3.1e-15 |
| random6 (6) | QSE pencil H − core | 2.7e-7 | 4.4e-16 |
| random6 (6) | ADAPT gradients − repository (127 generators) | 8.3e-8 | 0.0 |
| random8 (8), E0 ≈ −14.76 | mapped HF − tensor witness; Hermiticity; leakage | 1.6e-6; 1.5e-7; 6.8e-8 | 0.0; 9.7e-16; 2.7e-16 |
| random8 (8) | exact E0 − repository target | 3.6e-7 | 5.3e-15 |
| random8 (8) | QSE pencil H − core | 1.6e-6 | 4.9e-15 |
| random8 (8) | ADAPT gradients − repository (388 generators) | 9.2e-7 | 0.0 |

Declared float32 tolerances: 5e-5 (scalar/vector, scaled by max(1, |value|)),
2e-4 (matrix); float64: 1e-11 / 1e-10. Wall time per MPS `run` (4 methods):
1.9 s (4 qubits), 2.1 s (6), 4.7 s (8); CPU float64 control 1.4 / 1.6 / 2.5 s —
at these sizes the device transfer dominates, as the 2026-08-25 runtime profile
predicted for nq ≪ 20. AP-McLachlan is not in these receipts: its route is CPU
only (§4).

HDF5 end to end: a synthetic `_ele.h5` was written with the producer's writer
logic (h5py `metadata` attrs, gzip `h1`/`h2`, scalar `nuclear_repulsion`,
`mo_occ`, `mo_energy`; file sha256
`e6fe97b0debd8e40033ea79a60be7fed14672d54a0102692cfce31593b7047b9`, 17040 bytes),
read by `load_qdynamics_matrix_elements_h5` through the CLI
(`export-qdynamics`), accepted by the strict intake (raw `134dad11…`, operator
`1a34a8f2…`, identical to the mapping-input operator digest for the same
tensors), and run on MPS: four methods `gpu_verified`. Its receipt records
`data_role = "ele_h5_file_input_origin_unverified"`: the schema is Doron's,
the bytes are not.

## 6. Reproduction for the verifier

```sh
cd /Users/lordskyler/Holstein_test.worktrees/claude-doron-bridge-20261019 && git log --oneline -3
PYTHONPATH=. /Users/lordskyler/Holstein_test/.venv/bin/python -m pipelines.chemistry_bridge probe-backends
# then §5.1 (tests) and §5.2 (receipts)
```

Things to assess rather than count: (i) the interleaved→blocked extraction and
the four spin-block checks against the producer source blobs named in §2;
(ii) the frozen-core contraction formula and its spectrum identity test;
(iii) that `verified_status` can never return `gpu_verified` from a CPU
exercise (`test_status_never_counts_a_cpu_exercise_as_gpu`,
`test_gpu_status_is_impossible_without_an_accelerator_device`); (iv) that the
QSE/APM runs call the repository routes (no re-implementation of the methods:
grep `compute_qse_spectra`, `run_append_mclachlan_trajectory`,
`resolve_pool_plan` in `methods.py`); (v) the notation-binding caveat in
`qdynamics_export.py` (symmetry is necessary, not sufficient).

## 7. Bounded request: GPU parity and HDF5 reading — **approved and executed 2026-10-19**

Approved by Jake in the thread ("Approved: create the isolated venv at
~/.venvs/doron-bridge-gpu per handoff section 7 …"). Executed as proposed; the
repository `.venv` and every campaign were untouched. Recorded versions
(`probe_backends_gpu_venv.json`): Python 3.12.14 (arm64), pip 26.2.1,
torch 2.14.1 (`mps_available=True`, `mps_built=True`, `cuda=False`), h5py 3.16.0,
numpy 2.3.5, scipy 1.16.3, qiskit 2.3.1, qiskit-aer 0.17.2; venv size 910 MB;
install exit 0 (not wall-timed). Results: §5.1 (56 passed / 9 skipped) and §5.5.
The commands below are the ones that ran.

```sh
/Users/lordskyler/Holstein_test/.venv/bin/python -m venv /Users/lordskyler/.venvs/doron-bridge-gpu
/Users/lordskyler/.venvs/doron-bridge-gpu/bin/pip install numpy==2.3.5 scipy==1.16.3 qiskit==2.3.1 qiskit-aer==0.17.2 PyYAML==6.0.3 pytest==9.1.1 h5py torch
cd /Users/lordskyler/Holstein_test.worktrees/claude-doron-bridge-20261019
PYTHONPATH=. /Users/lordskyler/.venvs/doron-bridge-gpu/bin/python -m pipelines.chemistry_bridge probe-backends
PYTHONPATH=. STATIC_ADAPT_HH_POOL_CACHE=off /Users/lordskyler/.venvs/doron-bridge-gpu/bin/python -m pytest -q -rs test/test_chemistry_bridge_gpu_parity.py test/test_chemistry_bridge_export.py
PYTHONPATH=. STATIC_ADAPT_HH_POOL_CACHE=off /Users/lordskyler/.venvs/doron-bridge-gpu/bin/python -m pipelines.chemistry_bridge run --manifest <S>/random6/problem.manifest.json --out-dir <S>/random6/out_mps --backend torch:mps:float32
```

| item | value |
|---|---|
| Device / backend | Apple M4 Pro Metal via `torch:mps:float32` (float64 on mps is expected to report unavailable; `torch:cpu:float64` also runs as a torch-path control) |
| Download | torch macOS-arm64 wheel plus dependencies, roughly 100–200 MB; h5py ≈ 5 MB; disk ≈ 0.5 GB in `~/.venvs/doron-bridge-gpu` |
| Memory | ≤ 12-qubit action tables: 4096 × 2 × 4 bytes × ≤ 2000 words ≈ 65 MB device memory; far below headroom |
| Wall limit | install ≤ 10 min; parity tests + one `run` ≤ 5 min |
| Cost / allocation | none (local); no CHTC, no cloud |
| Stop condition | pip resolution fails, `probe-backends` reports mps unavailable, or any parity check fails → report the receipt and stop; no retry loop, no fallback to CPU counted as GPU |
| Records | pin `torch.__version__` and `h5py.__version__` from `probe-backends --json` into §7 of this file; commit the receipt JSON under `agent_guidance/shared/handoffs/` |

## 8. Missing inputs (user / Doron side)

1. A pinned `<prefix>_ele.h5` produced by `jacopos86/qdynamics@e5ec5bc4`
   (`main_pyscf.py`) with its `.inp`, plus Doron's declarations: geometry and
   units as given to PySCF, basis, SCF thresholds, bohr constant, and that
   `method=RHF`, `spin=0`. The adapter reads that file as is.
2. The intended system: H2O at the inferred G3 (0.9578 Å / 104.49°, inferred
   STO-3G) is 14 qubits and needs the frozen-core contraction (CAS(8e,6o));
   H2 at 0.7414 Å is 4 qubits and needs nothing. The bridge does not choose.
3. Whether Doron's numerical outputs may be committed under `test_support/`
   (his code is GPL-3.0; the data license is unstated).
4. Lane ownership of a Doron-derived electronic problem (Paper IV evidence, a
   Papers I–III application, or a new lane) and, for Paper I, authorization to
   build the named lane-owned application (gated edits in `pools.py`,
   `engine.py`, `adapters.py`, `contracts.py`, `adapt_pipeline.py` allowlists;
   CRITICAL blast radius).
5. For Paper II: a dipole drive (d_pq and nuclear dipole) if driven dynamics
   is wanted; the family declares `supports_driven_realtime=False`.
6. For Paper III: the number of target roots R, sector(s), observables.
7. Approval of §7.

## 9. Limitations (honest state)

* Synthetic witnesses are synthetic; no molecule has entered the bridge.
* The HDF5 loader is exercised only when `h5py` is importable (the test writes
  a file with the producer's writer logic and round-trips it); without `h5py`
  the test asserts the explicit `BridgeDependencyUnavailable`.
* GPU evidence is Apple Metal float32 only (`torch:mps:float32`, torch 2.14.1,
  isolated venv); `mps:float64` is refused by the device and CUDA is absent, so
  there is no float64 GPU result and no CUDA parity. The repository venv still
  has no torch, so the GPU tests skip there by design.
* At 4–8 qubits the MPS runs are slower than CPU (device transfer dominates);
  the backend is a correctness/parity path at this scale, not a speed-up.
* The APM run is a diagnostic trajectory (controller defaults,
  `max_structural_pool_size=8`, rk4, no drive); fidelity against exact sector
  propagation is reported, not certified, and the large-coefficient random
  witnesses show coarse-grid error (table §5.2).
* The exact reference is dense per sector (≤ 12 qubits by the intake scope);
  no sparse path was added.
* The strict intake receipt remains snapshot-only (downstream mutation is not
  locked); the bridge receipts carry its digests but do not enforce them in
  the repository routes.
* `d81c5481` is not applied; the legacy H2-labelled fixture is unchanged and
  still rejected by the strict intake (test_molecular_diagnostic_intake.py:137).

## 10. Execution record (one block per increment)

### Increment 1 — evidence discovery and base
Status: done
Commands run: GitHub REST/raw reads of `jacopos86/qdynamics` (branches, commit
`e5ec5bc4`, tree, seven source blobs; blob SHA-1s verified against the tree),
`users/doronwu/repos`; local bounded `find`; 11-agent Opus reader workflow over
the five Codex handoffs, four Paper V architecture/review files, the
2026-10-14 plan, lane contracts and code seams, with a completeness critic
(journal `…/subagents/workflows/wf_3460d407-69f/journal.jsonl`).
Result: §0 items 1–3 and §2.
Verification: blob SHA-1 of fetched writer/integrals files equal the tree
entries (`3bf71a4d0d2c`, `8400fc64c5ac`).
Deviations: none.

### Increment 2 — worktree, cherry-pick, baseline
Status: done
Commands run: `git worktree add … -b claude/doron-chemistry-quantum-bridge-20261019 d1436ab1`;
`git cherry-pick -x 393e335a` → `2e876e7d`; intake gate.
Result: 80 passed in 2.18 s.
Deviations: none.

### Increment 3 — bridge package and tests
Status: done (CPU); GPU cases skipped_unsupported; Paper I blocked; real data blocked
Commands run: §5.1, §5.2, §5.3.
Result: 47 passed / 18 skipped; receipts table §5.2; facade verdict recorded.
Surprises: (a) the action's gather-form weight must use the phase at the
source index (fixed before any test was written against it); (b) the identity
Pauli coefficient is not the scalar slot (JW number operators contribute), so
that check was replaced by the HF witness; (c) per-Pauli-word AP coordinates
leak particle number on the 8-qubit witness, confirming DISPOSITION 026-4.

Files to edit: None further in this increment. Next increments (each needs the
named input): §7 GPU parity run; §8 item 1 real-data receipt; a named Paper-I
application (separate, gated, CRITICAL).

### Increment 4 — GPU parity (approved §7)
Status: done (MPS float32 verified; mps float64 and cuda unsupported here)
Commands run: §7 install; `probe-backends --json`; §5.1 GPU-venv gate; six
`run` invocations (3 witnesses × {torch:mps:float32, torch:cpu:float64}); the
HDF5 write/export/run chain of §5.5.
Result: 56 passed / 9 skipped; receipts under
`agent_guidance/shared/handoffs/doron-bridge-receipts-20261019/`.
Verification: every MPS receipt reads `exercised_device = mps`; parity numbers
in §5.5 are all inside the declared float32 tolerances; CPU-float64 torch
control at 1e-15.
Deviations: `data_role` for a file input was renamed to
`ele_h5_file_input_origin_unverified` before the receipts were committed, so
that a synthetic file in the producer's schema is never read as Doron's data.

### Commits

| commit | content |
|---|---|
| `2e876e7d` | cherry-pick of `393e335a` (strict intake A) onto `d1436ab1` |
| `5b5b26c96098ce84ec885472136423525c57e3d3` | bridge package, tests, this handoff |
