# Exact-Metrics Benchmark

Independent exact-diagonalization (ED) references and accuracy validation tools
for benchmarking the hardcoded quantum algorithms.

## Key properties

- **No production Qiskit dependency** — most routes are pure numpy / scipy;
  Qiskit is optional and benchmark-local for explicit external/fake-backend rows.
- **Accuracy oracle** for VQE, Trotter, and future QPE results
- Produces reference eigenvalues, sector-filtered ground states, fidelity benchmarks

## Current contents

| File | Purpose |
|------|---------|
| `cross_check_suite.py` | Exact benchmark matrix across ansatz/VQE modes with JSON/PDF outputs |
| `hh_static_ground_state_benchmark.py` | Static/no-drive HH ground-state matrix runner with normalized row/proxy sidecars |
| `generic_static_hea_qiskit_vqe.py` | First-slice Hamiltonian-generic external HEA VQE row for one feasible non-HH case per family (`hubbard_L2`, `ionic_hubbard_L2`, `extended_hubbard_L2`, `ttprime_hubbard_L2`, `spinless_tv_L2`, `spin_boson_L1`, `bose_hubbard_L2`, `harmonic_kerr_chain_L2`, and H2 `molecular_restricted_closed_shell_L2`), using optional benchmark-local Qiskit state preparation plus repo-native VQE minimization; fixed-count and truncated-boson leakage diagnostics are reporting-only after optimization |
| `generic_static_adapt_variants.py` | Benchmark-local statevector ADAPT comparator variants, including the default full_meta append-only ADAPT row, Qubit/QEB, TETRIS, and Geo/Pos-Geo variants with local decision-noise surfaces |
| `generic_static_qiskit_adapt_vqe.py` | Exact-bench-only Qiskit Algorithms AdaptVQE append-only ADAPT reference/parity row for explicit nondefault runs; Hamiltonian Pauli terms become unit-coefficient operator-pool entries, exact references are reporting-only after optimization, Phase3/SNAKE/static_adapt controller code is not called, and opaque Qiskit decisions remain fail-closed for benchmark decision-noise |
| `generic_static_ed_reference.py` | Hamiltonian-generic static ED reference row over existing `ResolvedProblemContext.exact_target.resolve_energy(...)`, with molecular dense-resource guards |
| `qiskit_hea_adapter.py` | Lazy optional Qiskit HEA statevector adapter, isolated to exact-benchmark paths |
| `qiskit_adaptvqe_adapter.py` | Lazy optional Qiskit/qiskit-algorithms AdaptVQE imports plus repo Pauli-polynomial conversion helpers, isolated to exact-benchmark paths |
| `qiskit_community_dynamics_adapter.py` | Primary pinned Qiskit-community time-dynamics comparator adapter for Paper-II `dyn_qiskit_trotter_qrte`, `dyn_qiskit_pvqd`, and `dyn_qiskit_varqrte`; exact references are reporting-only after the Qiskit trajectory is produced |
| `qiskit_dynamics_adapter.py` | Dynamics parity sidecar adapter only; do not use this module as a primary Qiskit-community runner |
| `qnspsa_reference.py` | Lazy optional Qiskit QNSPSA benchmark/reference sanity harness; exact-bench only |
| `external_adapt/` | Benchmark-local provenance/fetch/adapter scaffold for external ADAPT competitors; keeps CEO/TETRIS/Overlap rows separate from Phase3 |
| `hh_noise_hardware_validation.py` | HH noisy/hardware-facing validation runner |
| `hh_noise_robustness_seq_report.py` | Sequential HH robustness report workflow |
| `benchmark_metrics_proxy.py` | Shared benchmark proxy metric utilities |
| `statevector_kernels.py` | Shared statevector kernel helpers for exact-bench runners |
| `noise_oracle_runtime.py` | Runtime/noise oracle support helpers, including shotless `aer_density_matrix` expectation replay for benchmark-local Qiskit noise studies |

Older propagator-benchmark helper scripts referenced in historical docs are not part
of this checkout’s active `exact_bench/` surface.

## Relationship to `test/`

- `test/` verifies **implementation correctness** (unit + integration)
- `exact_bench/` produces **reference data** and **physics-level accuracy reports**

Example:
- `test/test_ed_crosscheck.py` → "does the ED module compute correct eigenvalues?"
- `exact_bench/ed_reference_sweep.py` → "here are the reference eigenvalues for L=2..6, used to gate VQE accuracy"

## External competitor references

External ADAPT implementations are treated as benchmark inputs, not production
dependencies.  The `external_adapt/` package stores the reference catalog and
fetch helpers; third-party checkouts live outside the repo by default under
`~/.cache/holstein_external_competitors/`.  CEO and TETRIS public-code rows are
wired for the Hubbard first slice only; Overlap-ADAPT remains request-only /
explicitly skipped.  Do not emulate CEO, TETRIS, QEB, or Geo-style competitors
by toggling Phase3/static_adapt controller policies.
