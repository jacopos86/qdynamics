# Doron Wu's qubit Hamiltonian through our solvers: entry point and first runs (2026-11-04)

Branch `claude/molecular-vibronic-builder-20261102` (worktree
`Holstein_test.worktrees/claude-molecular-vibronic-builder-20261102`), on top of the vendored copy of
his module (`c77aa05f`, `third_party/qdynamics_chemistry`, jacopos86/qdynamics@38f40805, GPL-3.0) and
the interop module (`69c5d944`, `src/quantum/chemistry/qdynamics_interop.py`).

## What exists

`pipelines/chemistry_bridge/qubit_hamiltonian.py` (CLI `python -m pipelines.chemistry_bridge.qubit_hamiltonian
--hamiltonian <qdynamics JSON> --out-dir <dir> [--methods exact,qse,apm]`). The JSON is the whole input:
Pauli words over the fermion register (blocked spin order) and one boson register per mode, particle numbers,
cutoffs. Every method starts from the Hartree-Fock determinant (x) boson vacuum.

| method | route | what it reports |
|---|---|---|
| `exact` | `qdynamics_interop.ground_state_summary` (sparse sector matrix, `eigsh`) | low roots, `<HF x vac|H|HF x vac>`, ground-state boson occupations per mode, ground-state weight on the reference, one-apply sector-leakage witness |
| `qse` | `pipelines.qse_spectra.core.compute_qse_spectra` | Ritz roots on a mixed-register basis: identity + UCCSD generators lifted to the full register (`uccsd`), plus boson raising operators, their products with the singles and raising pairs (`uccsd_boson_ladders`), or plus every mode coordinate `Q_I` and `Q_I` times every UCCSD generator (`uccsd_times_displacements`); vanishing products on a truncated register are dropped and counted |
| `apm` | `pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory.run_append_mclachlan_trajectory` | energies, support sizes and patch kinds along the grid, sector leakage, fidelity against exact sector propagation (`expm_multiply` on the sparse sector matrix) and against the static reference; candidates = lifted Hermitian excitations, `Q_I`, `Q_I` times the singles, each audited by `audit_generator_sector_contract` with the fixed-count groups of the context; seed = largest initial McLachlan velocity |

The report `qubit_hamiltonian_report.json` (`qubit_hamiltonian_report_v1`) carries the file's sha256 and a
canonical operator digest, the repository commit and dirty flag, the environment, the declared origin
(`unknown | synthetic | simulated | attested`, never inferred from the bytes), the run class, and every
method's requested and executed settings; `exact_energies_used_online` is false by construction. Exit 0 only
when every method verified; 1 when one failed or raised (the traceback is in the report); 2 for a refused
input (no report). Tests: `test/test_chemistry_bridge_qubit_hamiltonian.py` (synthetic two-orbital, one-mode
model: all three methods, the padded-cutoff refusal, the CLI exit codes and report keys).

## Seam decisions

- Not through `BridgeContext`. The bridge's dense machinery (`BridgePolynomialAction`, `sector_matrix`) needs
  24 x 2^nq bytes per Pauli word and 16 x 2^nq x dim bytes for the block: 3.4 GB and 7.4 GB at 17 qubits,
  27 GB and hundreds of GB at 20. The entry point uses compiled Pauli actions (constant-size masks; the
  permutation tables are cached only up to `STATIC_ADAPT_PAULI_SIGN_CACHE_MAX_NQ`, settable with
  `--pauli-cache-nq`) and the repository sparse sector matrix. The bridge runners are untouched.
- One edit outside the new files: `pipelines/time_dynamics/ap_mclachlan/geometry_eval.py` takes a provider's
  `apply_at(time, psi)` when it exists and otherwise runs the dense `matrix_at` path unchanged. GitNexus rates
  `evaluate_mclachlan_geometry` CRITICAL (30 dependents, the whole Paper II lane). Evidence that nothing
  changed for existing providers: the full AP-McLachlan and bridge suite, 453 tests, passed under the complete
  environment (`.venv` of the main checkout) after the edit; under the isolated PySCF venv one route-parity
  test fails only because `qiskit_ibm_runtime` is absent there (it passes in the full environment).
- AP-McLachlan refuses binary registers with padded codes (cutoff + 1 < 2^qubits), where termwise boson
  generators leak; cutoffs 1 and 3 have none. The repository sector audit excludes truncation by design.
- RA-ADAPT is not in the entry point. The Paper IV lane runs it from a
  `molecular_vibronic_h2o_linear_fd_fixture_v1` file whose pool needs derivative tensors, mode vectors and
  labels the JSON does not carry, with size locks for CAS(8e,6o); the molecular closed-shell named application
  refuses boson registers. Next step (not done): a fixture producer from a `VibronicQubitModel` in his
  conventions on the fixture branch, plus the validator and `pools.py` size-lock changes the Paper IV reader
  mapped (about 250 lines), after which the lane CLI runs unchanged.

## Numbers, H2O/STO-3G at the RHF optimum (our Paper IV geometry), full MO space, 10 electrons in 7 orbitals

Files in `work/doron_interop_20261104/` (`his_code_run/`, `entry_point_runs/`). "His file" is his driver and
qubitizer run here with `isotope_masses` set, which equals our builder in his conventions to 7e-5 relative on
the coupling words (see `third_party/qdynamics_chemistry/README.md`). "Counter-term on" is our builder in his
conventions with `C -> C - <HF|C|HF>`. All methods from HF (x) vacuum, whose energy is -74.9415276 Ha for both.

| operator | cutoff / qubits | sector dimension | exact ground state (Ha) | ground-state boson occupations (bend, sym, asym) | reference weight |
|---|---|---|---|---|---|
| his file | 1 / 17 | 3,528 | -75.1190471 | 0.002 / 0.42 / 0.46 | 0.30 |
| his file | 3 / 20 | 28,224 | -75.2644600 | 0.06 / 1.63 / 1.78 | 0.006 |
| counter-term on | 1 / 17 | 3,528 | -75.0052072 | 0.03 / 0.05 / 0.003 | 0.89 |
| counter-term on | 3 / 20 | 28,224 | -75.0128338 | 0.87 / 0.63 / 0.10 | 0.25 |

His operator's ground state is not converged in the cutoff (1.6-1.8 quanta in the stretches at cutoff 3) and
has lost the reference almost entirely: the uncompensated Hartree-Fock force displaces the oscillators. With
the counter-term the cutoff-1 ground state is 89 % reference, but at cutoff 3 the bend still holds 0.87 quanta:
the correlated state feels a force the HF counter-term does not cancel, and the bend (0.0099 Ha) is soft.
The full-space electronic FCI is -75.0204389 Ha; FCI + zero point is -74.9960654 Ha.

| run (cutoff 1, 17 qubits) | QSE basis (elements) | lowest Ritz value (Ha) | above exact | AP-McLachlan t = 0 to 0.10 (6 points), support | energy drift | fidelity vs exact at t = 0.10 / static reference |
|---|---|---|---|---|---|---|
| his file | `uccsd_boson_ladders` (207) | -75.0846051 | 34.4 mHa | 2 -> 6, seed `Q(2)*herm_sing(alpha:0->5)` | 7.5 mHa | 0.99861 / 0.99831 |
| counter-term on | `uccsd_boson_ladders` (207) | -75.0026926 | 2.5 mHa | 2 -> 6, seed `Q(0)*herm_sing(alpha:0->5)` | 7.5 mHa | 0.99870 / 0.99840 |

QSE cost: about 2.3 s per basis element at 17 qubits (one compiled Hamiltonian apply each; 475 s for 207).
AP-McLachlan: 25 s for the six points (203 candidates, all pass the fixed-count audit); sector leakage 1e-21.
The two AP-McLachlan trajectories agree to 1e-7 in energy: on generators `Q_I x (excitation)` the force term
`F_I Q_I` has zero expectation to every order, so the counter-term is invisible to this pool over this horizon.

## Not done / untested

- QSE and AP-McLachlan at 20 qubits (cutoff 3) were not run; exact only (16 s).
- No RA-ADAPT run on either operator (see the seam note).
- His file here is "simulated" origin (his code, our run, our input deck); his own H2O deck is not in his repo.
- The QSE bases are bridge choices; from HF (x) vacuum they are far from the ground state on his operator
  because the ground state is displaced; a basis built on an RA-ADAPT state would be the right follow-up.
