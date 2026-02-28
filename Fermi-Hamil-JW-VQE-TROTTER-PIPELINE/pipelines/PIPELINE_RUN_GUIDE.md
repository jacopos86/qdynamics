# Hubbard Pipeline Run Guide

This is the comprehensive runtime guide for the simplified repo layout.

Run from the sub-repo root (`Fermi-Hamil-JW-VQE-TROTTER-PIPELINE/`).

---

## Runtime Scripts

| Script | Purpose |
|--------|---------|
| `pipelines/hardcoded_hubbard_pipeline.py` | Hardcoded Hamiltonian, hardcoded VQE, hardcoded Trotter dynamics, optional QPE |
| `pipelines/qiskit_hubbard_baseline_pipeline.py` | Qiskit Hamiltonian, Qiskit VQE, Qiskit Trotter dynamics, optional QPE |
| `pipelines/compare_hardcoded_vs_qiskit_pipeline.py` | Orchestrator — runs both, compares metrics, writes comparison PDFs |
| `pipelines/manual_compare_jsons.py` | Standalone JSON-vs-JSON consistency checker |
| `pipelines/regression_L2_L3.sh` | Automated L=2/L=3 regression harness |
| `pipelines/run_hva_uccsd_qiskit_L2_L3.sh` | Repro runner for hardcoded layer-wise UCCSD/HVA vs shared qiskit baseline on L=2,3 |
| `pipelines/run_L_drive_accurate.sh` | Shorthand runner for “run L”: drive-only, accuracy-gated (`delta_e < 1e-7`) with L-scaled heaviness |
| `pipelines/run_scaling_preset_L2_L6.sh` | Hardcoded+drive scaling preset for L=2..6 with VQE error gate and fallback ladder |

---

## State Source Behavior

`--initial-state-source` supports:

| Value | Behaviour |
|-------|-----------|
| `vqe` | Dynamics starts from that pipeline's own VQE-optimised state |
| `exact` | Dynamics starts from exact ground state (sector-filtered eigendecomposition) |
| `hf` | Dynamics starts from Hartree-Fock reference state |

If you want apples-to-apples hardcoded vs Qiskit from each ansatz, use `--initial-state-source vqe`.

---

## Complete Parameter Reference

### Model Parameters (all three pipelines)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--L` | int | *required* | Number of lattice sites (single pipelines) |
| `--l-values` | str | `"2,3,4,5"` | Comma-separated lattice sizes (compare pipeline only) |
| `--t` | float | `1.0` | Hopping coefficient t |
| `--u` | float | `4.0` | Onsite interaction U |
| `--dv` | float | `0.0` | Uniform local potential term v (H_v = −v n) |
| `--boundary` | choice | `periodic` | Boundary conditions: `periodic` or `open` |
| `--ordering` | choice | `blocked` | Qubit ordering: `blocked` or `interleaved` |

### Time-Evolution Parameters (all three pipelines)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--t-final` | float | `20.0` | Final evolution time |
| `--num-times` | int | `201` | Number of output time points |
| `--suzuki-order` | int | `2` | Suzuki–Trotter product-formula order |
| `--trotter-steps` | int | `64` | Number of Trotter steps |
| `--fidelity-subspace-energy-tol` | float | `1e-8` | Ground-manifold selection tolerance for trajectory subspace fidelity: include filtered-sector states with `E <= E0 + tol`. |
| `--term-order` | choice | `sorted` | Term ordering for Trotter product. Hardcoded: `native\|sorted`. Qiskit: `qiskit\|sorted` |

### Time-Dependent Drive Parameters (all three pipelines)

These flags control a Gaussian-envelope sinusoidal onsite density drive:

$$v(t) = A \cdot \sin(\omega t + \phi) \cdot \exp\!\Big(-\frac{(t - t_0)^2}{2\,\bar{t}^{\,2}}\Big)$$

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--enable-drive` | flag | `false` | Enable the time-dependent drive. When absent, no drive flags are forwarded and behaviour is identical to the static case. |
| `--drive-A` | float | `0.0`* | Drive amplitude A. *Compare pipeline default is `1.0`. |
| `--drive-omega` | float | `1.0` | Drive angular frequency ω |
| `--drive-tbar` | float | `1.0`** | Drive Gaussian half-width t̄ (must be > 0). **Compare pipeline default is `5.0`. |
| `--drive-phi` | float | `0.0` | Drive phase offset φ |
| `--drive-t0` | float | `0.0` | Drive start time t₀ |
| `--drive-pattern` | choice | `staggered` | Spatial weight pattern: `staggered`, `dimer_bias`, or `custom` |
| `--drive-custom-s` | str | `null` | JSON array of custom per-site weights, e.g. `'[1.0,-0.5]'`. Required when `--drive-pattern=custom`. |
| `--drive-include-identity` | flag | `false` | Include the identity (global-phase) term from n = (I−Z)/2 decomposition |
| `--drive-time-sampling` | choice | `midpoint` | Time-sampling rule within each Trotter slice: `midpoint`, `left`, or `right` |
| `--exact-steps-multiplier` | int | `1` | Reference-propagator refinement: N_ref = multiplier × trotter_steps. Has no effect when drive is disabled (static reference uses eigendecomposition). |

### VQE Parameters

**Single pipelines** (`hardcoded_hubbard_pipeline.py`, `qiskit_hubbard_baseline_pipeline.py`):

| Flag | Type | Default (HC) | Default (QK) | Description |
|------|------|-------------|-------------|-------------|
| `--vqe-ansatz` | choice | `uccsd` | n/a | Hardcoded-only ansatz family: `uccsd` or `hva` (both map to layer-wise implementations) |
| `--vqe-reps` | int | `2` | `2` | Number of ansatz repetitions (circuit depth) |
| `--vqe-restarts` | int | `1` | `3` | Number of independent VQE optimisation restarts |
| `--vqe-seed` | int | `7` | `7` | Random seed for VQE parameter initialisation |
| `--vqe-maxiter` | int | `120` | `120` | Maximum optimiser iterations per restart |

**Compare pipeline** (separate knobs for each sub-pipeline):

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--hardcoded-vqe-ansatzes` | str | `"uccsd"` | Comma-separated hardcoded ansatz set (shared qiskit baseline). |
| `--hardcoded-vqe-reps` | int | `2` | HC ansatz repetitions |
| `--hardcoded-vqe-restarts` | int | `3` | HC restarts |
| `--hardcoded-vqe-seed` | int | `7` | HC seed |
| `--hardcoded-vqe-maxiter` | int | `600` | HC max iterations |
| `--qiskit-vqe-reps` | int | `2` | QK ansatz repetitions |
| `--qiskit-vqe-restarts` | int | `3` | QK restarts |
| `--qiskit-vqe-seed` | int | `7` | QK seed |
| `--qiskit-vqe-maxiter` | int | `600` | QK max iterations |

### QPE Parameters (all three pipelines)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--qpe-eval-qubits` | int | `6` (single) / `5` (compare) | Number of evaluation qubits for QPE |
| `--qpe-shots` | int | `1024` (single) / `256` (compare) | Number of measurement shots |
| `--qpe-seed` | int | `11` | Random seed for QPE simulation |
| `--skip-qpe` | flag | `false` | Skip QPE execution entirely (marks payload as skipped) |

### Initial State

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--initial-state-source` | choice | `vqe` (compare) / `exact` (single) | State for dynamics: `exact`, `vqe`, or `hf` |

### Output / Artifact Controls

**Single pipelines:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--output-json` | path | auto | Path for output JSON |
| `--output-pdf` | path | auto | Path for output PDF |
| `--skip-pdf` | flag | `false` | Skip PDF generation |

**Compare pipeline:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--artifacts-dir` | path | `artifacts/` | Directory for all generated outputs |
| `--run-pipelines` | flag | `true` | Run both sub-pipelines (use `--no-run-pipelines` to reuse existing JSONs) |
| `--with-per-l-pdfs` | flag | `false` | Include per-L comparison pages in bundle and emit standalone per-L PDFs |

### Drive Amplitude Comparison (compare pipeline only)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--drive-amplitudes` | str | `"0.0,0.2"` | Comma-separated pair `A0,A1`. A0 is the trivial amplitude for the safe-test; A1 is the active amplitude. |
| `--with-drive-amplitude-comparison-pdf` | flag | `false` | Generate amplitude-comparison PDF per L. Runs both pipelines 3× per L (disabled, A0, A1 = 6 sub-runs). |
| `--report-verbose` | flag | `false` | Verbose report mode; forces full safe-test detail plots. |
| `--safe-test-near-threshold-factor` | float | `100.0` | Safe-test detail page gate: render when `max_safe_delta >= threshold/factor` (also on fail or `--report-verbose`). |

### Hardcoded layer-wise ansatz mapping

- `--vqe-ansatz uccsd` -> `HardcodedUCCSDLayerwiseAnsatz`
- `--vqe-ansatz hva` -> `HubbardLayerwiseAnsatz`
- Legacy term-wise classes remain available in `src/quantum/vqe_latex_python_pairs.py`, but runtime defaults route to the layer-wise classes above.
- Hardcoded VQE JSON now includes:
  - `vqe.ansatz`
  - `vqe.parameterization` (currently `"layerwise"`)
  - `vqe.exact_filtered_energy`

---

## Full CLI (defaults)

### Hardcoded pipeline

```bash
python pipelines/hardcoded_hubbard_pipeline.py --help
```

Defaults:

- `--t 1.0 --u 4.0 --dv 0.0`
- `--boundary periodic --ordering blocked`
- `--t-final 20.0 --num-times 201 --suzuki-order 2 --trotter-steps 64`
- `--fidelity-subspace-energy-tol 1e-8`
- `--term-order sorted` (`native|sorted`)
- `--vqe-ansatz uccsd` (`uccsd|hva`)
- `--vqe-reps 2 --vqe-restarts 1 --vqe-seed 7 --vqe-maxiter 120`
- `--qpe-eval-qubits 6 --qpe-shots 1024 --qpe-seed 11`
- `--initial-state-source vqe`
- Drive: disabled by default. Enable with `--enable-drive`.

### Qiskit baseline pipeline

```bash
python pipelines/qiskit_hubbard_baseline_pipeline.py --help
```

Defaults:

- `--t 1.0 --u 4.0 --dv 0.0`
- `--boundary periodic --ordering blocked`
- `--t-final 20.0 --num-times 201 --suzuki-order 2 --trotter-steps 64`
- `--fidelity-subspace-energy-tol 1e-8`
- `--term-order sorted` (`qiskit|sorted`)
- `--vqe-reps 2 --vqe-restarts 3 --vqe-seed 7 --vqe-maxiter 120`
- `--qpe-eval-qubits 6 --qpe-shots 1024 --qpe-seed 11`
- `--initial-state-source vqe`
- Drive: disabled by default. Enable with `--enable-drive`.

### Compare runner

```bash
python pipelines/compare_hardcoded_vs_qiskit_pipeline.py --help
```

Defaults:

- `--l-values 2,3,4,5`
- `--run-pipelines` (use `--no-run-pipelines` to reuse JSONs)
- `--t 1.0 --u 4.0 --dv 0.0`
- `--boundary periodic --ordering blocked`
- `--t-final 20.0 --num-times 201 --suzuki-order 2 --trotter-steps 64`
- `--fidelity-subspace-energy-tol 1e-8`
- `--hardcoded-vqe-ansatzes uccsd` (set `uccsd,hva` for 3-way hardcoded-vs-qiskit runs)
- `--hardcoded-vqe-reps 2 --hardcoded-vqe-restarts 3 --hardcoded-vqe-seed 7 --hardcoded-vqe-maxiter 600`
- `--qiskit-vqe-reps 2 --qiskit-vqe-restarts 3 --qiskit-vqe-seed 7 --qiskit-vqe-maxiter 600`
- `--qpe-eval-qubits 5 --qpe-shots 256 --qpe-seed 11`
- `--initial-state-source vqe` (`exact|vqe|hf`)
- `--artifacts-dir artifacts`
- Drive: disabled by default. Enable with `--enable-drive`.
- Amplitude comparison: disabled by default. Enable with `--with-drive-amplitude-comparison-pdf`.
- `--drive-amplitudes "0.0,0.2"` (only used when amplitude comparison is enabled)
- Compare acceptance now includes the VQE sanity condition for each hardcoded ansatz and qiskit:
  `vqe.energy >= exact_filtered_energy - 1e-8`

---

## Common Commands

### 0) Shorthand `run L` convention (drive-only + accurate)

```bash
bash pipelines/run_L_drive_accurate.sh --L 4
```

### 0b) HVA/UCCSD layer-wise L=2,3 runner

```bash
bash pipelines/run_hva_uccsd_qiskit_L2_L3.sh
```

Heavy preset:

```bash
bash pipelines/run_hva_uccsd_qiskit_L2_L3.sh --heavy
```

This enforces the default shorthand contract:

- drive is always enabled (never static),
- accuracy gate is enforced:
  `abs(vqe.energy - ground_state.exact_energy_filtered) < 1e-7`,
- settings scale with `L` (heavier defaults for larger systems),
- fallback attempts auto-escalate if the primary attempt misses the gate.

Primary per-L presets used by the shorthand runner:

| L | trotter_steps | exact_steps_multiplier | num_times | vqe_reps | vqe_restarts | vqe_method | vqe_maxiter |
|---|---:|---:|---:|---:|---:|---|---:|
| 2 | 128 | 2 | 201 | 2 | 2 | COBYLA | 1200 |
| 3 | 192 | 2 | 201 | 2 | 3 | COBYLA | 2400 |
| 4 | 256 | 3 | 241 | 4 | 4 | SLSQP | 6000 |
| 5 | 384 | 3 | 301 | 4 | 5 | SLSQP | 8000 |
| 6 | 512 | 4 | 361 | 5 | 6 | SLSQP | 10000 |

Fallback behavior:
- `fallback_A`: increase optimizer effort (`restarts + 2`, `maxiter * 2`, method `L-BFGS-B` for `L >= 4`).
- `fallback_B`: additionally increase ansatz/dynamics effort (`reps + 1`, `trotter_steps * 1.5`, `exact_steps_multiplier + 1`).

Optional flags:

```bash
bash pipelines/run_L_drive_accurate.sh --L 5 --with-pdf
```

```bash
bash pipelines/run_L_drive_accurate.sh --L 6 --budget-hours 12 --artifacts-dir artifacts
```

### 1) Run full compare for L=2,3,4 with locked heavy settings

```bash
python pipelines/compare_hardcoded_vs_qiskit_pipeline.py \
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
python pipelines/compare_hardcoded_vs_qiskit_pipeline.py \
  --l-values 2,3,4 \
  --artifacts-dir artifacts \
  --no-run-pipelines \
  --with-per-l-pdfs
```

### 3) Run hardcoded pipeline only

```bash
python pipelines/hardcoded_hubbard_pipeline.py \
  --L 3 --initial-state-source vqe \
  --output-json artifacts/json/H_L3_static_t1.0_U4.0_S64.json \
  --output-pdf artifacts/pdf/H_L3_static_t1.0_U4.0_S64.pdf
```

### 4) Run Qiskit baseline only

```bash
python pipelines/qiskit_hubbard_baseline_pipeline.py \
  --L 3 --initial-state-source vqe \
  --output-json artifacts/json/Q_L3_static_t1.0_U4.0_S64.json \
  --output-pdf artifacts/pdf/Q_L3_static_t1.0_U4.0_S64.pdf
```

### 5) Run with time-dependent drive enabled

```bash
python pipelines/hardcoded_hubbard_pipeline.py \
  --L 2 --initial-state-source vqe \
  --enable-drive --drive-A 0.5 --drive-omega 2.0 --drive-tbar 3.0 \
  --drive-pattern dimer_bias --drive-time-sampling midpoint \
  --t-final 10.0 --num-times 101 --trotter-steps 64 \
  --exact-steps-multiplier 4 \
  --output-json artifacts/json/H_L2_vt_t1.0_U4.0_S64.json \
  --output-pdf artifacts/pdf/H_L2_vt_t1.0_U4.0_S64.pdf
```

### 6) Compare pipeline with drive enabled

```bash
python pipelines/compare_hardcoded_vs_qiskit_pipeline.py \
  --l-values 2,3 --run-pipelines --enable-drive \
  --drive-A 0.5 --drive-omega 2.0 --drive-tbar 3.0 \
  --drive-pattern dimer_bias --drive-time-sampling midpoint \
  --t-final 10.0 --num-times 101 --trotter-steps 64 \
  --exact-steps-multiplier 4 --skip-qpe \
  --with-per-l-pdfs
```

### 7) Amplitude comparison PDF (scoreboard + physics response)

```bash
python pipelines/compare_hardcoded_vs_qiskit_pipeline.py \
  --l-values 2 --run-pipelines --enable-drive \
  --drive-pattern dimer_bias --drive-omega 2.0 --drive-tbar 2.0 \
  --t-final 2.0 --num-times 21 --trotter-steps 32 --skip-qpe \
  --drive-amplitudes '0.0,0.2' \
  --with-drive-amplitude-comparison-pdf
```

This runs 8 sub-pipeline invocations per L (2 main + 6 amplitude comparison) and generates:
- `pdf/amp_{tag}.pdf` — multi-page PDF with settings, scoreboard tables, drive waveform, response deltas, combined overlay, and residual-focused VQE page
- `json/amp_{tag}_metrics.json` — machine-readable safe-test + VQE delta metrics

### 8) Run the L=2/L=3 regression harness

```bash
bash pipelines/regression_L2_L3.sh
```

This writes `_reg` JSON/PDF outputs for L=2 and L=3, runs the compare runner, runs
`manual_compare_jsons.py`, and ends with `REGRESSION PASS` or `REGRESSION FAIL`.

### 9) Manual JSON-vs-JSON consistency check

```bash
python pipelines/manual_compare_jsons.py \
  --hardcoded artifacts/json/H_L3_static_t1.0_U4.0_S64.json \
  --qiskit artifacts/json/Q_L3_static_t1.0_U4.0_S64.json \
  --metrics artifacts/json/HvQ_L3_static_t1.0_U4.0_S64_metrics.json
```

### 10) Run the L=2..6 scaling preset with VQE error gate

```bash
bash pipelines/run_scaling_preset_L2_L6.sh
```

Defaults in this runner:

- Physics: `t=1.0, u=4.0, dv=0.0, periodic, blocked`.
- Drive: enabled with `A=0.5, omega=2.0, tbar=3.0, phi=0.0, pattern=staggered`.
- Error gate: `abs(vqe.energy - ground_state.exact_energy_filtered) < 1e-2`.
- L=2..5 budget guard: `10` hours (`L25_BUDGET_HOURS`).
- L6 run: enabled by default (`RUN_L6=1`).
- PDFs: skipped by default (`SKIP_PDF=1`) for production timing runs.

Useful overrides:

```bash
L25_BUDGET_HOURS=10 RUN_L6=0 bash pipelines/run_scaling_preset_L2_L6.sh
```

```bash
ERROR_THRESHOLD=5e-3 SKIP_PDF=0 bash pipelines/run_scaling_preset_L2_L6.sh
```

Artifacts are written to:

- `artifacts/scaling_preset_L2_L6_<timestamp>/json`
- `artifacts/scaling_preset_L2_L6_<timestamp>/logs/summary.tsv`
- `artifacts/scaling_preset_L2_L6_<timestamp>/logs/best.tsv`

---

## Trajectory Fidelity and Energy Observables

### Subspace Fidelity Semantics

`trajectory[].fidelity` is the **subspace fidelity**:

`F_sub(t) = <psi_ansatz_trot(t)|P_exact_gs_subspace(t)|psi_ansatz_trot(t)>`

where `P_exact_gs_subspace(t)` projects onto the time-evolved filtered-sector
ground manifold selected by:

`E <= E0 + tol`, with `tol = --fidelity-subspace-energy-tol` (default `1e-8`).

The JSON `settings` block records:

- `fidelity_definition_short`
- `fidelity_definition`
- `fidelity_subspace_energy_tol`
- `fidelity_reference_subspace`:
  - `sector = {n_up, n_dn}`
  - `ground_subspace_dimension`
  - `selection_rule = "E <= E0 + tol"`

Each trajectory row in the JSON output contains two families of energy fields:

| Key | Observable | Formula |
|-----|-----------|---------|
| `energy_static_exact` | Static Hamiltonian expectation (exact propagator) | ⟨ψ\_exact\|H\_static\|ψ\_exact⟩ |
| `energy_static_trotter` | Static Hamiltonian expectation (Trotter propagator) | ⟨ψ\_trotter\|H\_static\|ψ\_trotter⟩ |
| `energy_total_exact` | **Total** instantaneous energy (exact propagator) | ⟨ψ\_exact\|H\_static + H\_drive(t₀+t)\|ψ\_exact⟩ |
| `energy_total_trotter` | **Total** instantaneous energy (Trotter propagator) | ⟨ψ\_trotter\|H\_static + H\_drive(t₀+t)\|ψ\_trotter⟩ |

### Behaviour by drive state

| Drive state | `energy_total_*` |
|-------------|-------------------|
| Disabled (`--enable-drive` absent) | Identical to `energy_static_*` (no overhead) |
| Enabled with `A = 0` (safe-test) | Identical to `energy_static_*` within machine precision |
| Enabled with `A > 0` | Differs from `energy_static_*` by the drive contribution ⟨ψ\|H\_drive(t)\|ψ⟩ |

### Physical time convention

The drive Hamiltonian at observation time `t` is evaluated at physical time `drive_t0 + t`, consistent with the propagator convention.

### Settings metadata

The JSON `settings` block includes an `energy_observable_definition` string that documents the energy field semantics:

```
"energy_observable_definition": "energy_static_* measures <psi|H_static|psi>. energy_total_* measures <psi|H_static + H_drive(drive_t0 + t)|psi>. When drive is disabled, energy_total_* == energy_static_*. Drive sampling uses the same drive_t0 convention as propagation."
```

### Compare pipeline handling

The compare pipeline (`compare_hardcoded_vs_qiskit_pipeline.py`) handles both energy families:

- **Static energy**: `energy_static_trotter` HC−QK delta is a primary pass/fail gate (threshold `1e-3`).
- **Total energy**: `energy_total_trotter` HC−QK delta is also a pass/fail gate (threshold `1e-3`), included when both JSONs provide the field.
- **Plot overlay**: When drive is active and total energy differs from static, cyan and orange curves are overlaid on the energy plot.
- **Single-pipeline PDFs**: Both HC and QK pipelines overlay total-energy curves on the energy plot when the drive causes them to differ from static.

---

## Generated Artifacts

Under `artifacts/` (or the path given by `--artifacts-dir`):

```
artifacts/
├── json/        # All JSON outputs
├── pdf/         # All PDF outputs
└── commands.txt # Exact commands run
```

### Naming convention

Filenames use a **tag** encoding the run config:
`L{L}_{vt|static}_t{t}_U{u}_S{trotter_steps}` — e.g. `L2_static_t1.0_U4.0_S64`.

Prefixes: **H** = hardcoded, **Q** = Qiskit, **HvQ** = comparison, **amp** = amplitude comparison.

### Standard comparison outputs

| File | Description |
|------|-------------|
| `json/H_{tag}.json` | Hardcoded pipeline full output |
| `json/Q_{tag}.json` | Qiskit pipeline full output |
| `json/HvQ_{tag}_metrics.json` | Per-L comparison metrics (subspace fidelity, energy, VQE, QPE deltas) |
| `json/HvQ_summary.json` | Summary across all L values |
| `pdf/HvQ_bundle.pdf` | Multi-page comparison bundle PDF |
| `pdf/HvQ_{tag}.pdf` | Per-L standalone comparison PDF (with `--with-per-l-pdfs`) |
| `commands.txt` | Exact commands that were executed |

### Amplitude comparison outputs (with `--with-drive-amplitude-comparison-pdf`)

| File | Description |
|------|-------------|
| `pdf/amp_{tag}.pdf` | Multi-page PDF: command, settings, scoreboard + drive waveform, response deltas, combined HC/QK overlay, VQE residual table, optional safe-test detail/heatmap/spectrum pages |
| `json/amp_{tag}_metrics.json` | Machine-readable: `safe_test`, `delta_vqe_hc_minus_qk_at_A0`, `delta_vqe_hc_minus_qk_at_A1` |
| `json/amp_H_{tag}_{slug}.json` | HC intermediate outputs (slug = `disabled`, `A0`, `A1`) |
| `json/amp_Q_{tag}_{slug}.json` | QK intermediate outputs (slug = `disabled`, `A0`, `A1`) |

### VQE visibility

- Per-L comparison PDFs include explicit VQE comparison pages.
- Bundle PDF includes VQE comparison pages and per-L VQE pages (with `--with-per-l-pdfs`).
- Amplitude comparison PDF uses a residual-focused VQE table (`HC-QK`, `HC-exact`, `QK-exact`) for perceptible deltas.

### Metrics JSON schema (per-L comparison)

The `HvQ_{tag}_metrics.json` file includes `trajectory_deltas` with per-observable HC−QK statistics:
`trajectory_deltas.fidelity` keeps its key name for compatibility and stores
**subspace fidelity** deltas.

```json
{
  "trajectory_deltas": {
    "fidelity":                { "max_abs_delta": ..., "mean_abs_delta": ..., "final_abs_delta": ... },
    "energy_static_trotter":   { "max_abs_delta": ..., "mean_abs_delta": ..., "final_abs_delta": ... },
    "energy_total_exact":      { "max_abs_delta": ..., "mean_abs_delta": ..., "final_abs_delta": ... },
    "energy_total_trotter":    { "max_abs_delta": ..., "mean_abs_delta": ..., "final_abs_delta": ... },
    "n_up_site0_trotter":      { "max_abs_delta": ..., "mean_abs_delta": ..., "final_abs_delta": ... },
    "..."
  }
}
```

> `energy_total_trotter` is a pass/fail gate (threshold `1e-3`), same as `energy_static_trotter`.

### Metrics JSON schema (amplitude comparison)

```json
{
  "generated_utc": "2026-02-21T00:11:04.676856+00:00",
  "L": 2,
  "A0": 0.0,
  "A1": 0.2,
  "safe_test": {
    "passed": true,
    "threshold": 1e-10,
    "hc": { "max_fidelity_delta": 6.66e-16, "max_energy_delta": 0.0 },
    "qk": { "max_fidelity_delta": 8.88e-16, "max_energy_delta": 4.44e-16 }
  },
  "delta_vqe_hc_minus_qk_at_A0": -1.888e-09,
  "delta_vqe_hc_minus_qk_at_A1": -1.888e-09
}
```
