# Holstein_test

Canonical repository onboarding document.

This repo implements Hubbard-Holstein (HH) simulation workflows with
Jordan-Wigner operator construction, binary or unary bosonic encoding, blocked or periodic boundary conditions, with hardcoded HVA/ADAPT/VQE ground-state preparation, and exact vs Trotterized vs CFQM dynamics pipelines.

## Project focus

- Primary production model: `Hubbard-Holstein (HH)`.
- Pure Hubbard is retained as a limiting-case validation path.
- Standard regression checks: HH with `g_ep = 0` and `omega0 = 0` under
  matched settings should reduce to Hubbard behavior.
- Noiseless shots and Aer simulator should match the pipeline with noise simular turned off.


### Warm-start chain

Warm-start runs use a two-stage sequence:

1. Run HH-HVA VQE warm start with always default `hh_hva_ptw`.
2. Use that warm-start state as the ADAPT reference state.
3. Run ADAPT from that warm-start state with the agent max meta-pool: `UCCSD + HVA/PTW + (PAOP_full union PAOP_lf_full)`. This is a custom merged pool, not a single `--adapt-pool` value; `curdrag` is already included via PAOP-LF pools.




## Repository map (minimal)

- `src/quantum/`: operator algebra, Hamiltonian builders, ansatz/statevector math
- `pipelines/hardcoded/`: production hardcoded pipeline entrypoints
- `pipelines/exact_bench/`: exact-diagonalization benchmark tooling
- `reports/`: PDF and reporting utilities
- `docs/`: architecture, implementation, and status documents

## Visual overview

```mermaid
graph TB
  R["README.md canonical entrypoint"]

  subgraph DOCS
    D1["docs/repo_implementation_guide.md"]
    D2["docs/HH_IMPLEMENTATION_STATUS.md"]
    D3["docs/FERMI_HAMIL_README.md"]
  end

  subgraph RULES
    G0["AGENTS.md"]
    G1["pipelines/run_guide.md"]
  end

  subgraph SRC
    S0["src/quantum/"]
    S1["src/quantum/operator_pools/"]
  end

  subgraph PIPELINES
    P0["pipelines/hardcoded/"]
    P1["hubbard_pipeline.py"]
    P2["adapt_pipeline.py"]
    P3["pipelines/exact_bench/"]
    P4["cross_check_suite.py"]
    P5["pipelines/shell/"]
    P6["pipelines/qiskit_archive/"]
  end

  subgraph REPORTS
    RP0["reports/pdf_utils.py"]
    RP1["reports/README.md"]
  end

  subgraph OUTPUT
    O1["artifacts/json/"]
    O2["artifacts/pdf/"]
    O3["artifacts/overnight_l3_hh_4method/"]
  end

  subgraph TESTS
    T0["test/test_adapt_vqe_integration.py"]
  end

  R --> G0
  R --> G1
  R --> D1
  R --> D2
  R --> D3

  G1 --> P0
  G1 --> P3
  G1 --> P5
  G1 --> P6

  P0 --> P1
  P0 --> P2
  P3 --> P4

  P1 --> S0
  P2 --> S0
  P2 --> S1
  P1 --> RP0
  P2 --> RP0

  P1 --> O1
  P1 --> O2
  P2 --> O1
  P2 --> O2
  O1 --> O3
  O2 --> O3

  T0 --> P2
  T0 --> S0
```

## Physics algorithm flow (VQE / ADAPT / pools)

```mermaid
graph TB
  A["CLI run config: L,t,U,dv,problem,boundary,ordering,HH params,drive flags"] --> B["Build Hamiltonian H in JW PauliPolynomial with e/x/y/z"]
  B --> C{"Ground-state prep mode"}

  C --> V["VQE path"]
  C --> AD["ADAPT-VQE path"]

  subgraph VQE_MODE
    V --> V1{"Ansatz family"}
    V1 --> V2["uccsd or hva or hh_hva variants"]
    V2 --> V3["Optimize energy expval psi_theta with H"]
    V3 --> V4["Produce psi_vqe"]
  end

  subgraph ADAPT_MODE
    AD --> AD0["Reference state psi_ref: HF for Hubbard, HH reference for HH"]
    AD0 --> AD1{"Pool selection by problem"}

    AD1 --> HPOOL["Hubbard pools: uccsd, cse, full_hamiltonian"]
    AD1 --> HHPOOL["HH pools: hva, full_hamiltonian, paop_min, paop_std, paop_full"]

    HHPOOL --> PD1["PAOP disp terms: shifted_density times P_i"]
    HHPOOL --> PD2["PAOP hopdrag terms: K_ij times (P_i minus P_j)"]
    HHPOOL --> PD3["PAOP full extras: doublon and extended cloud"]
    HHPOOL --> HMERGE["If g_ep != 0: merge hva + hh_termwise_augmented + paop_* and deduplicate by polynomial signature"]

    HPOOL --> GCompute
    HHPOOL --> GCompute
    HMERGE --> GCompute
    PD1 --> GCompute
    PD2 --> GCompute
    PD3 --> GCompute

    GCompute["Compute commutator_grad for available operators"] --> GSelect["Select max magnitude operator and append"]
    GSelect --> Reopt["Re-optimize all parameters with COBYLA"]
    Reopt --> Stop{"Stop rule"}
    Stop -->|eps_grad or eps_energy or pool_exhausted or max_depth| ADOut["Produce psi_adapt"]
    Stop -->|continue| GCompute
  end

  V4 --> DYN
  ADOut --> DYN
  DYN["Time evolution branch: exact reference and Suzuki-2 Trotter, static or drive-enabled"] --> OUT["Artifacts: JSON and PDF manifests, plots, metrics"]
```

### ADAPT Pool Summary (plaintext fallback)

- `hubbard` pools: `uccsd`, `cse`, `full_hamiltonian`.
- `hh` pools: `hva`, `full_hamiltonian`, `paop_min`, `paop_std`, `paop_full`, `paop_lf` (`paop_lf_std` alias), `paop_lf2_std`, `paop_lf_full`.
- HH built-in combined preset: `uccsd_paop_lf_full` = `uccsd_lifted + paop_lf_full` (deduplicated) via one CLI value.
- Agent max custom meta-pool (not a single CLI value): `uccsd_lifted + hva + paop_full + paop_lf_full`.
- `paop_min`: displacement-focused PAOP operators.
- `paop_std`: displacement plus dressed-hopping (`hopdrag`) operators.
- `paop_full`: `paop_std` plus doublon dressing and extended cloud operators.
- `paop_lf_std`: `paop_std` plus LF-leading odd channel (`curdrag`).
- HH merge behavior (when `g_ep != 0`): merge `hva` + `hh_termwise_augmented` + selected `paop_*` pool, then deduplicate by polynomial signature.

### Compiled speedup stack note (2026-03-04)

The hardcoded VQE/ADAPT path now includes a shared compiled-action acceleration stack, with additive (backward-compatible) interfaces and parity tests.

- Shared compiled polynomial utility:
  - `src/quantum/compiled_polynomial.py`
  - Provides `compile_polynomial_action`, `apply_compiled_polynomial`, `energy_via_one_apply`, and `adapt_commutator_grad_from_hpsi`.
- Compiled ansatz executor:
  - `src/quantum/compiled_ansatz.py`
  - Applies Pauli rotations through compiled permutation+phase actions (no per-amplitude string loops).
- VQE one-apply energy backend:
  - `src/quantum/vqe_latex_python_pairs.py` adds `expval_pauli_polynomial_one_apply(...)`.
  - `vqe_minimize(...)` supports `energy_backend="legacy"|"one_apply_compiled"` (default remains legacy).
  - `pipelines/hardcoded/hubbard_pipeline.py` exposes `--vqe-energy-backend {legacy,one_apply_compiled}` and defaults to `one_apply_compiled`.
  - Hardcoded VQE can emit live progress heartbeats via `--vqe-progress-every-s` (default `60` seconds), including restart lifecycle and periodic energy/nfev telemetry.
- ADAPT runtime acceleration:
  - `pipelines/hardcoded/adapt_pipeline.py` compiles Hamiltonian/pool once, computes `H|psi>` once per depth, evaluates pool gradients via `2*Im(<Hpsi|Apsi>)`, and uses compiled ansatz execution in COBYLA objective/state updates.
- Regression coverage added:
  - `test/test_compiled_polynomial.py`
  - `test/test_compiled_ansatz.py`
  - `test/test_vqe_energy_backend.py`
  - existing ADAPT integration suite remains passing.
- Additive ADAPT telemetry fields:
  - `adapt_vqe.compiled_pauli_cache`
  - `adapt_vqe.history[*].gradient_eval_elapsed_s`
  - `adapt_vqe.history[*].optimizer_elapsed_s`
- New benchmark utility:
  - `pipelines/hardcoded/bench_compiled_energy_and_grad.py`
  - Times legacy vs compiled energy and gradient scoring paths, reports speedups and parity deltas.

Fast VQE-from-ADAPT replay (HH, hardcoded):

```bash
python pipelines/hardcoded/hubbard_pipeline.py \
  --problem hh \
  --L 4 --boundary open --ordering blocked \
  --boson-encoding binary --n-ph-max 1 \
  --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 \
  --vqe-ansatz hh_hva_tw --vqe-reps 1 \
  --vqe-restarts 16 --vqe-maxiter 12000 --vqe-method COBYLA --vqe-seed 7 \
  --vqe-energy-backend one_apply_compiled --vqe-progress-every-s 60 \
  --initial-state-source adapt_json \
  --adapt-input-json .vscode-userdata/artifacts/useful/L4/l4_hh_seq_20260302_215706_resume_adaptB_20260303_111311_adapt_B_B_probe_checkpoint_state.json \
  --skip-qpe --num-times 1 --t-final 0.0 --skip-pdf \
  --output-json artifacts/json/hc_hh_L4_from_adaptB_fastcomp.json
```

Keep strict metadata matching enabled by default. Use `--no-adapt-strict-match` only for legacy/import files that do not carry complete HH metadata keys.

## Start here (doc priority)

Use this order when onboarding:

1. `AGENTS.md` - repo conventions and non-negotiable implementation rules
2. `pipelines/run_guide.md` - CLI and runbook for active pipelines
3. `docs/repo_implementation_guide.md` - implementation-deep walkthrough
4. `docs/HH_IMPLEMENTATION_STATUS.md` - current HH status and remaining work

## Important note on README files

Subdirectory README files are component-scoped documentation, not repo-canonical
onboarding docs. Use this root `README.md` first, then drill into local READMEs
for module-specific details.

## Quick run examples

ADAPT-VQE (HH, PAOP pool):

```bash
python pipelines/hardcoded/adapt_pipeline.py \
  --L 2 --problem hh --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-pool paop_std --paop-r 1 --paop-normalization none \
  --adapt-max-depth 30 --adapt-eps-grad 1e-5 --adapt-maxiter 600 \
  --initial-state-source adapt_vqe --skip-pdf \
  --output-json artifacts/json/adapt_L2_hh_paop_std.json
```

Compiled energy/gradient micro-benchmark (HH):

```bash
python pipelines/hardcoded/bench_compiled_energy_and_grad.py \
  --L 3 --n-ph-max 1 --repeats 5
```

Cross-check suite (exact benchmark; auto-scaled by L/problem defaults):

```bash
python pipelines/exact_bench/cross_check_suite.py --L 2 --problem hubbard
```

CFQM propagation (hardcoded pipeline):

```bash
python pipelines/hardcoded/hubbard_pipeline.py \
  --L 2 --problem hubbard \
  --propagator cfqm4 \
  --cfqm-stage-exp expm_multiply_sparse \
  --cfqm-coeff-drop-abs-tol 0.0 \
  --trotter-steps 64 --t-final 10.0 --num-times 201 \
  --skip-qpe
```

CFQM propagation status (hardcoded pipeline):
- `--propagator` defaults to `suzuki2`; existing run behavior is unchanged unless `cfqm4`/`cfqm6` is selected.
- CFQM uses fixed scheme nodes (`c_j`) and ignores legacy midpoint/left/right `--drive-time-sampling`.
- `--exact-steps-multiplier` remains a reference-only control and does not change CFQM macro-step count.
- `--cfqm-stage-exp` default is `expm_multiply_sparse`; `--cfqm-coeff-drop-abs-tol` default is `0.0`; `--cfqm-normalize` default is off.
- Sparse CFQM stage backend uses native sparse stage assembly + `scipy.sparse.linalg.expm_multiply` (no dense->csc stage materialization).
- Shared Pauli-term exponentiation helpers are centralized in `src/quantum/pauli_actions.py` (used by both the hardcoded pipeline and CFQM backend).
- Unknown drive labels are handled with a guardrail policy: nontrivial coefficients warn once per label then are ignored; tiny coefficients (`abs(coeff) <= 1e-14`) are silently ignored.
- A=0 invariance is preserved by the zero-increment insertion guard; safe-test target is `<= 1e-10`.
- If non-midpoint sampling is supplied under CFQM, runtime warns:
  `CFQM ignores midpoint/left/right sampling; uses fixed scheme nodes c_j.`
- If `--cfqm-stage-exp pauli_suzuki2` is selected, runtime warns:
  `Inner Suzuki-2 makes overall method 2nd order; use expm_multiply_sparse/dense_expm for true CFQM order.`

CFQM6 command:

```bash
python pipelines/hardcoded/hubbard_pipeline.py \
  --L 2 --problem hubbard \
  --propagator cfqm6 \
  --cfqm-stage-exp expm_multiply_sparse \
  --cfqm-coeff-drop-abs-tol 0.0 \
  --trotter-steps 64 --t-final 10.0 --num-times 201 \
  --skip-qpe
```

CFQM tests:

```bash
pytest -q test/test_cfqm_schemes.py test/test_cfqm_propagator.py test/test_cfqm_acceptance.py
```

Acceptance highlights:
- static regression vs exact expm
- A=0 invariance (drive provider present vs absent)
- manufactured 1-qubit order slopes (`~4` for cfqm4, `~6` for cfqm6, `~2` with inner suzuki2)
- small HH sanity trend vs fine piecewise reference

Quantum-processor proxy benchmark (CFQM vs Suzuki):

```bash
python pipelines/exact_bench/cfqm_vs_suzuki_qproc_proxy_benchmark.py \
  --problem hubbard --L 2 \
  --methods suzuki2,cfqm4,cfqm6 \
  --steps-grid 64,128,256,512 \
  --reference-steps 2048 \
  --drive-enabled

# Equal-cost policy (optional, apples-to-apples):
python pipelines/exact_bench/cfqm_vs_suzuki_qproc_proxy_benchmark.py \
  --problem hubbard --L 2 \
  --methods suzuki2,cfqm4,cfqm6 \
  --steps-grid 64,128,256,512 \
  --reference-steps 2048 \
  --compare-policy cost_match \
  --cost-match-metric cx_proxy_total \
  --cost-match-tolerance 0.0 \
  --drive-enabled
```

Why this benchmark:
- Local runtime is machine-dependent and not the main comparison axis.
- The benchmark ranks methods by final energy error versus processor-oriented cost proxies:
  - term exponential count
  - 2-qubit gate proxy (`cx_proxy_total`)
  - 1-qubit gate proxy (`sq_proxy_total`)
- `S` in the result tables is macro-step count (`trotter_steps`), not a cost metric.
- Use `--compare-policy cost_match` for fair, equal-cost comparisons; default remains
  sweep-only row listing.
- Default cost axis for fair matching is `cx_proxy_total`; fallback metric is `term_exp_count_total` when requested.
- CFQM runs use `pauli_suzuki2` stage exponentials in this benchmark to produce hardware-comparable termwise gate proxies (this is a benchmarking profile, not the high-order dense/sparse CFQM profile).

Artifacts:
- `artifacts/cfqm_benchmark/cfqm_vs_suzuki_proxy_runs.json`
- `artifacts/cfqm_benchmark/cfqm_vs_suzuki_proxy_runs.csv`
- `artifacts/cfqm_benchmark/cfqm_vs_suzuki_proxy_summary.json`

CFQM efficiency suite (error-vs-cost, apples-to-apples):

```bash
python pipelines/exact_bench/cfqm_vs_suzuki_efficiency_suite.py \
  --problem-grid hubbard_L4,hh_L2_nb1,hh_L2_nb2,hh_L2_nb3,hh_L3_nb1 \
  --drive-grid sinusoid,gaussian_sharp \
  --methods suzuki2,cfqm4,cfqm6 \
  --stage-mode-grid exact_sparse,exact_dense,pauli_suzuki2 \
  --reference-steps-multiplier 8 \
  --equal-cost-axis cx_proxy,pauli_rot_count,expm_calls,wall_time \
  --equal-cost-policy exact_tie_only \
  --calibrate-transpile \
  --output-dir artifacts/cfqm_efficiency_benchmark
```

L2 HH statevector warm-start variant:

```bash
python pipelines/exact_bench/cfqm_vs_suzuki_efficiency_suite.py \
  --problem-grid hh_L2_nb1 \
  --drive-grid sinusoid,gaussian_sharp \
  --methods suzuki2,cfqm4,cfqm6 \
  --stage-mode-grid exact_sparse,exact_dense,pauli_suzuki2 \
  --initial-state-source adapt_json \
  --adapt-input-json artifacts/useful/L2/H_L2_hh_termwise_regular_lbfgs_t1.0_U2.0_g1_nph1.json \
  --no-adapt-strict-match \
  --reference-steps-multiplier 8 \
  --equal-cost-axis cx_proxy,pauli_rot_count,expm_calls,wall_time \
  --equal-cost-policy exact_tie_only \
  --calibrate-transpile \
  --output-dir artifacts/cfqm_efficiency_benchmark/overnight_hh_L2_nb1_lbfgs_state
```

Note: strict ADAPT metadata matching is on by default; this command disables it because the referenced warm-start JSON does not carry full HH metadata keys.

Efficiency-suite ADAPT pool default is `auto`:
- `hubbard*` scenarios resolve to `--adapt-pool uccsd`
- `hh*` scenarios resolve to `--adapt-pool paop_std`

Efficiency-suite outputs:
- `artifacts/cfqm_efficiency_benchmark/runs_full.json`
- `artifacts/cfqm_efficiency_benchmark/runs_full.csv`
- `artifacts/cfqm_efficiency_benchmark/summary_by_scenario.json`
- `artifacts/cfqm_efficiency_benchmark/pareto_by_metric.json`
- `artifacts/cfqm_efficiency_benchmark/slope_fits.json`
- `artifacts/cfqm_efficiency_benchmark/equal_cost_exact_ties_<metric>.csv`
- `artifacts/cfqm_efficiency_benchmark/cfqm_efficiency_suite.pdf`

Efficiency-suite interpretation rules:
- Main fair tables are exact-cost ties only (`delta=0`) for `cx_proxy`, `pauli_rot_count`, and `expm_calls`.
- Wall-time comparisons are near-tie bins and explicitly marked approximate.
- Fallback nearest-neighbor matches are appendix-only (non-fair direct comparisons).
- `S` always means macro-step count (`trotter_steps`), never a fairness axis.

For compare/orchestration workflows, use `pipelines/run_guide.md`.

## Major Markdown docs index

- `AGENTS.md`
- `pipelines/run_guide.md`
- `docs/LLM_RESEARCH_CONTEXT.md`
- `docs/repo_implementation_guide.md`
- `docs/HH_IMPLEMENTATION_STATUS.md`
- `docs/FERMI_HAMIL_README.md`
- `reports/README.md`
- `pipelines/exact_bench/README.md`
- `pipelines/qiskit_archive/README.md`
- `pipelines/qiskit_archive/DESIGN_NOTE_TIMEDEP.md`

## HH noisy estimator validation

The repo now includes an HH-first noisy/hardware validation pipeline:
- `pipelines/exact_bench/hh_noise_hardware_validation.py`

It provides one shared expectation oracle across `ideal`, `shots`, `aer_noise`, and `runtime` modes.  
`shots`/`aer_noise` emulate finite-shot measurement noise using Qiskit `AerSimulator`, with optional noisy ADAPT and PDF/JSON reporting.  
Use `pipelines/run_guide.md` section 11+ for operational commands and mode-by-mode guidance.

For staged heavy HH robustness with warm-start -> ADAPT Pool B -> final VQE and noisy dynamics benchmarking:
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py`

Key additions:
- strict ADAPT Pool B composition enforcement (`UCCSD_lifted + HVA + PAOP_full`)
- noisy dynamics methods via `--noisy-methods` (default `suzuki2,cfqm4`)
- embedded benchmark metrics in JSON/PDF (`term_exp_count_total`, `cx_proxy_total`, `sq_proxy_total`, `depth_proxy_total`, `wall_total_s`, `oracle_eval_s_total`)
- backward-compatible `dynamics_noisy.profiles.<profile>.modes` alias mirroring `suzuki2`

For a long, code/docs-only repository guide focused on noise-model behavior:
- `pipelines/exact_bench/hh_noise_model_repo_guide.py`
- convenience wrapper: `pipelines/shell/build_hh_noise_model_repo_guide.sh`

Default guide artifacts:
- `docs/HH_noise_model_repo_guide.pdf`
- `artifacts/json/hh_noise_model_repo_guide_index.json`
