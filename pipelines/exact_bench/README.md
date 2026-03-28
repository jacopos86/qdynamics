# Exact-Metrics Benchmark

Independent exact-diagonalization (ED) references and accuracy validation tools
for benchmarking the hardcoded quantum algorithms.

## Key properties

- **No Qiskit dependency** — pure numpy / scipy
- **Accuracy oracle** for VQE, Trotter, and future QPE results
- Produces reference eigenvalues, sector-filtered ground states, fidelity benchmarks

## Current contents

| File | Purpose |
|------|---------|
| `cross_check_suite.py` | Exact benchmark matrix across ansatz/VQE modes with JSON/PDF outputs |
| `hh_noise_hardware_validation.py` | HH noisy/hardware-facing validation runner |
| `hh_noise_robustness_seq_report.py` | Sequential HH robustness report workflow |
| `benchmark_metrics_proxy.py` | Shared benchmark proxy metric utilities |
| `statevector_kernels.py` | Shared statevector kernel helpers for exact-bench runners |
| `noise_oracle_runtime.py` | Runtime/noise oracle support helpers |

Older CFQM-benchmark helper scripts referenced in historical docs are not part
of this checkout’s active `exact_bench/` surface.

## Relationship to `test/`

- `test/` verifies **implementation correctness** (unit + integration)
- `exact_bench/` produces **reference data** and **physics-level accuracy reports**

Example:
- `test/test_ed_crosscheck.py` → "does the ED module compute correct eigenvalues?"
- `exact_bench/ed_reference_sweep.py` → "here are the reference eigenvalues for L=2..6, used to gate VQE accuracy"
