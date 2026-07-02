# Hubbard Pipeline Run Guide

This is the minimal runtime guide for the simplified repo layout.

Run from repo root:

```bash
cd "/Users/jakestrobel/Downloads/qdynamics-main/Fermi-Hamil-JW-VQE-TROTTER-PIPELINE"
```

## Runtime Scripts

- `pipelines/hardcoded_hubbard_pipeline.py`
  - Hardcoded Hamiltonian, hardcoded VQE, hardcoded dynamics.
  - QPE currently uses a temporary Qiskit adapter.
- `pipelines/qiskit_hubbard_baseline_pipeline.py`
  - Qiskit Hamiltonian, Qiskit VQE, Qiskit dynamics.
- `pipelines/compare_hardcoded_vs_qiskit_pipeline.py`
  - Runs both pipelines (or reads existing JSON), compares metrics, writes PDFs.

## State Source Behavior

`--initial-state-source` supports:

- `vqe`: dynamics starts from that pipeline's own VQE state.
- `exact`: dynamics starts from exact ground state.
- `hf`: dynamics starts from Hartree-Fock state.

If you want apples-to-apples hardcoded vs Qiskit from each ansatz, use `--initial-state-source vqe`.

## Parameter Options

Shared model/time parameters:

- `--L` (single script) or `--l-values` (compare script): lattice size(s)
- `--t` float: hopping
- `--u` float: onsite interaction
- `--dv` float: local potential term
- `--boundary`: `periodic` or `open`
- `--ordering`: `blocked` or `interleaved`
- `--t-final` float
- `--num-times` int
- `--suzuki-order` int (currently `2` expected)
- `--trotter-steps` int

VQE controls:

- `--vqe-reps` / `--hardcoded-vqe-reps` / `--qiskit-vqe-reps`
- `--vqe-restarts` / `--hardcoded-vqe-restarts` / `--qiskit-vqe-restarts`
- `--vqe-maxiter` / `--hardcoded-vqe-maxiter` / `--qiskit-vqe-maxiter`
- `--vqe-seed` / `--hardcoded-vqe-seed` / `--qiskit-vqe-seed`

QPE controls:

- `--qpe-eval-qubits`
- `--qpe-shots`
- `--qpe-seed`

Artifacts path:

- `--artifacts-dir` (compare script, default `artifacts/`)
- single-pipeline scripts default to `artifacts/` when output paths are not provided

## Full CLI (defaults)

### Hardcoded pipeline

```bash
/opt/anaconda3/bin/python pipelines/hardcoded_hubbard_pipeline.py --help
```

Defaults:

- `--t 1.0 --u 4.0 --dv 0.0`
- `--boundary periodic --ordering blocked`
- `--t-final 20.0 --num-times 201 --suzuki-order 2 --trotter-steps 64`
- `--term-order sorted` (`native|sorted`)
- `--vqe-reps 1 --vqe-restarts 1 --vqe-seed 7 --vqe-maxiter 120`
- `--qpe-eval-qubits 6 --qpe-shots 1024 --qpe-seed 11`
- `--initial-state-source exact` (`exact|vqe|hf`)

### Qiskit baseline pipeline

```bash
/opt/anaconda3/bin/python pipelines/qiskit_hubbard_baseline_pipeline.py --help
```

Defaults:

- `--t 1.0 --u 4.0 --dv 0.0`
- `--boundary periodic --ordering blocked`
- `--t-final 20.0 --num-times 201 --suzuki-order 2 --trotter-steps 64`
- `--term-order sorted` (`qiskit|sorted`)
- `--vqe-reps 2 --vqe-restarts 3 --vqe-seed 7 --vqe-maxiter 120`
- `--qpe-eval-qubits 6 --qpe-shots 1024 --qpe-seed 11`
- `--initial-state-source exact` (`exact|vqe|hf`)

### Compare runner

```bash
/opt/anaconda3/bin/python pipelines/compare_hardcoded_vs_qiskit_pipeline.py --help
```

Defaults:

- `--l-values 2,3,4,5`
- `--run-pipelines` (use `--no-run-pipelines` to reuse JSONs)
- `--t 1.0 --u 4.0 --dv 0.0`
- `--boundary periodic --ordering blocked`
- `--t-final 20.0 --num-times 121 --suzuki-order 2 --trotter-steps 32`
- `--hardcoded-vqe-reps 1 --hardcoded-vqe-restarts 1 --hardcoded-vqe-seed 7 --hardcoded-vqe-maxiter 40`
- `--qiskit-vqe-reps 2 --qiskit-vqe-restarts 3 --qiskit-vqe-seed 7 --qiskit-vqe-maxiter 12`
- `--qpe-eval-qubits 5 --qpe-shots 256 --qpe-seed 11`
- `--initial-state-source vqe` (`exact|vqe|hf`)
- `--artifacts-dir artifacts`

## Common Commands

### 1) Run full compare for L=2,3,4 with locked heavy settings

```bash
/opt/anaconda3/bin/python pipelines/compare_hardcoded_vs_qiskit_pipeline.py \
  --l-values 2,3,4 \
  --artifacts-dir artifacts \
  --initial-state-source vqe \
  --t 1.0 --u 4.0 --dv 0.0 --boundary periodic --ordering blocked \
  --t-final 20.0 --num-times 401 --suzuki-order 2 --trotter-steps 128 \
  --hardcoded-vqe-reps 2 --hardcoded-vqe-restarts 3 --hardcoded-vqe-maxiter 1800 --hardcoded-vqe-seed 7 \
  --qiskit-vqe-reps 2 --qiskit-vqe-restarts 3 --qiskit-vqe-maxiter 1800 --qiskit-vqe-seed 7 \
  --qpe-eval-qubits 8 --qpe-shots 4096 --qpe-seed 11 \
  --with-per-l-pdfs
```

### 2) Rebuild comparison PDFs/summary from existing JSON

```bash
/opt/anaconda3/bin/python pipelines/compare_hardcoded_vs_qiskit_pipeline.py \
  --l-values 2,3,4 \
  --artifacts-dir artifacts \
  --no-run-pipelines \
  --with-per-l-pdfs
```

### 3) Run hardcoded pipeline only

```bash
/opt/anaconda3/bin/python pipelines/hardcoded_hubbard_pipeline.py \
  --L 3 --initial-state-source vqe \
  --output-json artifacts/hardcoded_pipeline_L3.json \
  --output-pdf artifacts/hardcoded_pipeline_L3.pdf
```

### 4) Run Qiskit baseline only

```bash
/opt/anaconda3/bin/python pipelines/qiskit_hubbard_baseline_pipeline.py \
  --L 3 --initial-state-source vqe \
  --output-json artifacts/qiskit_pipeline_L3.json \
  --output-pdf artifacts/qiskit_pipeline_L3.pdf
```

### 5) Run the L=2/L=3 regression harness

```bash
bash pipelines/regression_L2_L3.sh
```

This writes `_reg` JSON/PDF outputs for L=2 and L=3, runs the compare runner, runs
`manual_compare_jsons.py`, and ends with `REGRESSION PASS` or `REGRESSION FAIL`.

### 6) Manual JSON-vs-JSON consistency check

```bash
/opt/anaconda3/bin/python pipelines/manual_compare_jsons.py \
  --hardcoded artifacts/hardcoded_pipeline_L3.json \
  --qiskit artifacts/qiskit_pipeline_L3.json \
  --metrics artifacts/hardcoded_vs_qiskit_pipeline_L3_metrics.json
```

## Generated Artifacts

Under `artifacts/`:

- `hardcoded_pipeline_L{L}.json`
- `qiskit_pipeline_L{L}.json`
- `hardcoded_vs_qiskit_pipeline_L{L}_metrics.json`
- `hardcoded_vs_qiskit_pipeline_L{L}_comparison.pdf`
- `hardcoded_vs_qiskit_pipeline_summary.json`
- `hardcoded_vs_qiskit_all_results_bundle.pdf`
- `pipeline_commands_run.txt`

VQE visibility:

- Per-L comparison PDFs include explicit VQE bar charts.
- Bundle PDF includes VQE comparison pages and per-L VQE pages (with `--with-per-l-pdfs`).
