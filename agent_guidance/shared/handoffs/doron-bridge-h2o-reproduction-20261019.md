# Doron's H2O system through the bridge: his exporter, run here (2026-10-20 UTC)

Authorized by Jake in the "Doron chemistry–quantum bridge" thread ("yes approve": PySCF into the
isolated venv, shallow fetch of his exporter at the pinned commit, run it on H2O/STO-3G at his
inferred geometry, bridge with frozen core, commit receipts, 30-minute limit). Branch
`claude/doron-h2o-export-20261019` from `70d0b317` (the bridge branch itself is unchanged).

## 0. What this is and is not

* **Is:** Doron Wu's own exporter (`jacopos86/qdynamics` branch `doron-chemistry`, commit
  `e5ec5bc45d02169e77ce93b05126c9804a03cdb6`, GPL-3.0; blob ids `3bf71a4d0d2c`, `8400fc64c5ac`,
  `594293ae21d9`, `d9dc9eb32faa` verified against the fetched tree) executed unmodified on this
  machine, producing `h2o_ele.h5` in his schema for H2O/STO-3G/RHF at the geometry our July
  cross-code audit inferred from his slide row; then the complete bridge chain on that file.
* **Is not:** bytes from Doron. Every receipt records `origin = simulated`,
  `container_schema = qdynamics_matrix_elements_ele_h5`, the claimed producer (his code at the
  pinned commit, executed locally), and `producer_origin_established = False`. The geometry is
  inferred (R(OH) 0.9578 Å, ∠HOH 104.49°; his slide states no geometry or basis); the frozen-core
  contraction to CAS(8e,6o) is our choice to fit the intake's 12-qubit scope and is declared in
  the document (`nuclear_plus_frozen_core`, `already_contracted`).

## 1. Producer run (`producer_record.json`)

| item | value |
|---|---|
| Command | `PYTHONPATH=<fetch> ~/.venvs/doron-bridge-gpu/bin/python -m src.chemistry.pyscf.main_pyscf input/h2o.inp` (pyscf 2.14.0, h5py 3.16.0, numpy 2.3.5; 3.3 s) |
| Input | `input/h2o.inp`, sha256 `99775d78897928b905c451acde052a2eeccb02ec125dd65c0d9b59cd8774415e` (RHF, sto-3g, charge 0, spin 0, Angstrom, e_converg 1e-12, d_converg 1e-8) |
| Output | `input/h2o_ele.h5`, 51,856 bytes, sha256 `e413ecbf9e150d313dd2daa0ef1fb72c83f583d2f48ca849db1a74fe927c6648`; `h1 (14,14)`, `h2 (14,14,14,14)`, `nuclear_repulsion` scalar 9.189275612994711, `mo_coeff/mo_energy/mo_occ`, metadata attrs incl. `h1_convention=spin_orbital_interleaved`, `h2_convention=chemist_spin_orbital_coulomb`, `h2_antisymmetrized=False`, `converged=True`, `nelec=10`, `nmo=7`, `ms2=0` |
| E_SCF | **−74.96302344338558 Ha**: equal to our July 2026 PySCF cross-check (−74.9630234434) and 3.05e-7 Ha from the slide row "PySCF in qcdynamics" (−74.9630231385) |

## 2. Bridge documents

| document | content |
|---|---|
| `h2o_cas8e6o_problem.json` (+ manifest) | frozen core orbital 0 → 6 active orbitals, (4,4), **12 qubits**; scalar slot = E_nuc 9.189275612994711 + core −60.660646986580154; raw sha256 `92c5ff8dc43a0381cc82c2363ef100246a52426b748d9102870c087318756efd`; intake tensor `b31ce68fa8ad…`, operator `1065d877ac31…` (551 Pauli terms), HF-witness defect 1.4e-14, ERI symmetry defects ≤ 4.7e-16 |
| `h2o_full_space_problem.json` (+ manifest) | full 7-orbital space, 14 qubits, recorded only: it exceeds the strict intake's 12-qubit scope and was not run |

## 3. Results on the real system (CAS(8e,6o), 12 qubits; receipts under `receipts/`)

| method | numpy float64 (CPU) | torch:mps:float32 (Metal, `exercised_device=mps`) |
|---|---|---|
| mapping witness | `cpu_verified`; mapped HF −74.9630234434 = tensor witness (diff 0.0); Hermiticity 3.8e-15; sector leakage 2.3e-17; sector dim 225 | `gpu_verified`; HF diff 4.3e-5 (float32 at |E|≈75, tolerance 5e-5·75); Hermiticity 1.4e-7; leakage 6.6e-9 |
| exact sector reference | `cpu_verified`; **E0 = −75.0125019248 Ha** (repository dense target identical to 2.1e-13); low roots −75.012502, −74.614580, −74.554878, −74.510957, −74.508718; 53 s | `gpu_verified`; E0 −75.0125437610 (diff 4.2e-5 to float64); 45 s |
| QSE (identity + 92 UCC generators, Lowdin rank 93 of sector dim 225, not spanning) | `cpu_verified`; lowest Ritz −75.011797 (7.0e-4 above exact, all Ritz ≥ exact as required); pencil parity vs core 1.1e-13 | `gpu_verified`; same Ritz roots from the CPU core; backend pencil parity 6.8e-5 |
| ADAPT selection gradients (repository `full_meta` pool, 644 generators) | `cpu_verified`; top |g|: `uccsd_dbl(ab:1,7->5,11)` 0.30498, `uccsd_dbl(ab:0,6->4,10)` 0.20377, `uccsd_dbl(ab:2,7->4,11)` 0.15698; parity 0.0; `run_ra_adapt` gate recorded `admitted=False` (route **blocked**) | `gpu_verified`; gradient parity 6.0e-8 |
| AP-McLachlan (CPU only; executed-safe pool, logical_shared, rk4, t = 0, 0.02, 0.04) | `cpu_verified` as a diagnostic: 170/644 generators eligible (16 UCC singles, 76 UCC doubles, 78 Z-only words; 474 fail the executed-unitary audit: 472 X/Y words, 2 Hamiltonian blocks); no even-Y multi-word generator → `trajectory_moved=False`, energy constant at HF, leakage 0; fidelity to exact sector propagation 0.99984 at t=0.04 is the exact state's departure from HF, not a variational error; 55 s | not applicable (dense numpy route) |

Correlation energy recovered by exact CAS(8e,6o) diagonalization: E0 − E_HF = −0.0495 Ha.

## 4. Limits and open decisions

* Origin is `simulated` by construction; obtaining a file produced on Doron's side would let the
  same receipts carry a named attestation over its sha256 (still a claim, never "established").
* Geometry and basis are inferred; a declaration from Doron would replace the inference.
* The full 14-qubit space is out of the intake's scope; CAS(8e,6o) is ours and declared.
* QSE with the UCC basis does not span the sector (rank 93 of 225): the roots are upper bounds,
  not a converged spectrum; R, sectors and observables remain unspecified.
* ADAPT: selection screening only; RA-ADAPT/SNAKE stays facade-blocked.
* AP-McLachlan: the audited pool cannot move the state; an executed-safe even-Y molecular pool
  is an unselected scientific decision.
* GPU: Metal float32 kernel/pencil parity only; no CUDA, no float64 GPU, no GPU APM.
* Independent review: none yet for this reproduction; the 002 corrections await recheck 3.

## 5. Clock and files

Increment 003-A: start 2026-10-20T02:45:07Z (end recorded in the commit message);
driver 02:45:53–02:45:56Z; CPU bridge 57.7 s; Metal bridge 60.3 s; APM 55 s.
Files: this note; `doron-bridge-h2o-20261019/{producer_record.json, input/h2o.inp,
input/h2o_ele.h5, h2o_cas8e6o_problem.json, h2o_cas8e6o_problem.manifest.json,
h2o_full_space_problem.json, h2o_full_space_problem.manifest.json, receipts/*/bridge_receipts.json}`.
Nothing else changed; the fetched exporter tree lives only in the session scratchpad.
