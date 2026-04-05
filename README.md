# Holstein_test

This path is the canonical repository onboarding document.

## Active checkout snapshot (2026-03-26)

This README reflects the active non-archived toolchain in this repository:

- canonical direct HH ADAPT: `pipelines/hardcoded/adapt_pipeline.py`
- staged compatibility wrappers: `pipelines/hardcoded/hh_staged_noiseless.py`, `pipelines/hardcoded/hh_staged_noise.py`
- replay / follow-on exact-bench surfaces: `pipelines/hardcoded/hh_vqe_from_adapt_family.py`, `pipelines/exact_bench/cross_check_suite.py`, `pipelines/exact_bench/hh_noise_hardware_validation.py`
- fixed-manifold / realtime sweep surfaces: `pipelines/hardcoded/hh_fixed_manifold_mclachlan.py`, `pipelines/hardcoded/hh_fixed_manifold_measured.py`, `pipelines/hardcoded/hh_l2_static_realtime_pareto_sweep.py`, `pipelines/hardcoded/hh_l2_driven_realtime_pareto_sweep.py`
- legacy shorthand helper: `pipelines/shell/run_drive_accurate.sh`

Some older README examples below had drifted toward removed scripts. This pass keeps only surfaces that exist in this checkout and marks legacy-only seams explicitly.

This repo implements Hubbard-Holstein (HH) simulation workflows with
Jordan-Wigner operator construction, binary or unary bosonic encoding, blocked or periodic boundary conditions, with hardcoded HVA/ADAPT/VQE ground-state preparation, and exact vs Trotterized vs CFQM dynamics pipelines.

## Project focus

- Primary production model: `Hubbard-Holstein (HH)`.
- Pure Hubbard is a legacy / dead model for default planning and should be ignored unless explicitly requested.
- Standard regression checks may still use the HH -> Hubbard limiting case when the task explicitly calls for that consistency check.
- Noiseless shots and Aer simulator should match the pipeline with noise simulation turned off.


### Canonical HH ADAPT path

The only non-archaic HH ADAPT default path is direct `pipelines/hardcoded/adapt_pipeline.py`.

- On the direct CLI, omitting `--adapt-continuation-mode` now defaults to `phase3_v1`.
- This is the full Phase 3 HH path with the current cheap pre-cap + ratio rerank selector.
- For new HH ADAPT work, depth-0 starts from the narrow physics-aligned core; current runtime resolves this to `paop_lf_std`.
- `full_meta` remains a supported broad-pool preset, but only as an explicit compatibility/broad-search choice, not the default depth-0 path.
- `legacy`, `phase1_v1`, and `phase2_v1` remain explicit historical/compatibility modes only.

### Historical staged compatibility chain

The staged `VQE -> ADAPT -> VQE` wrappers remain available for reproduction and compatibility, but they are not the canonical default path for new HH ADAPT work.

1. Run HH-HVA VQE warm start with `hh_hva_ptw`.
   - `hh_hva_ptw` remains the staged warm-start default.
   - `hh_hva` remains an explicit override only.
2. Use that warm-start state as the ADAPT reference state.
3. Run ADAPT from that prepared state in staged wrapper continuation mode.
   - Older repo-doc references to staged `phase1_v1` default behavior are archival only.
   - The manuscript is the source of truth for the canonical HH ADAPT surface; when older repo docs disagree, follow the manuscript and current CLI/runtime behavior (`phase3_v1`).
   - The manuscript-facing canonical path keeps Phase-3 runtime splitting disabled; widened `--phase3-symmetry-mitigation-mode` choices remain phase-3 metadata/telemetry hooks on raw staged/hardcoded/replay paths.
4. Replay conventional VQE from ADAPT with ADAPT-family matching (`--generator-family match_adapt`, fallback `full_meta`) via `pipelines/hardcoded/hh_vqe_from_adapt_family.py`.

Optional staged seed-refine insertion:

- `pipelines/hardcoded/hh_staged_noiseless.py` can insert one explicit-family conventional VQE refine stage between warm start and ADAPT via `--seed-refine-family`.
- Supported explicit seed-refine families are:
  - `uccsd_otimes_paop_lf_std`
  - `uccsd_otimes_paop_lf2_std`
  - `uccsd_otimes_paop_bond_disp_std`
- The refine stage materializes the requested family directly; it does **not** use `match_adapt` and does **not** auto-fallback to `full_meta`.
- If the refine stage succeeds, the handoff bundle carries additive `seed_provenance`.
- If the refine stage fails, the staged workflow aborts before ADAPT rather than silently skipping forward.

One-shot noiseless wrapper for the historical/default staged contract:

```bash
python pipelines/hardcoded/hh_staged_noiseless.py --L 2

# Optional refine insertion:
python pipelines/hardcoded/hh_staged_noiseless.py --L 2 \
  --seed-refine-family uccsd_otimes_paop_lf_std
```

This wrapper keeps drive opt-in, runs final matched-family replay (not fixed `hh_hva_*` replay), and reports Suzuki/CFQM dynamics from the replay seed with GS-baseline energy error plus seeded exact-reference fidelity. It remains a historical/compatibility surface rather than the default path for new HH ADAPT runs.

Historical note:
- Older docs may mention `pipelines/hardcoded/hh_staged_circuit_report.py`.
- That script is **not present in this checkout**; treat those references as stale.

## Repository map (minimal)

- `src/quantum/`: operator algebra, Hamiltonian builders, ansatz/statevector math
- `pipelines/hardcoded/`: production hardcoded pipeline entrypoints
- `pipelines/exact_bench/`: exact-diagonalization benchmark tooling
- `docs/reports/`: PDF and reporting utilities
- root markdown docs: active repo-facing contracts and workflow notes
- `MATH/`: near-term HH implementation notes and math targets

## Visual overview

```mermaid
graph TB
  R["README.md canonical entrypoint"]

  subgraph DOCS
    D1["AGENTS.md"]
    D2["pipelines/run_guide.md"]
    D3["MATH/IMPLEMENT_NEXT.md"]
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
  end

  subgraph REPORTS
    RP0["docs/reports/pdf_utils.py"]
  end

  subgraph TESTS
    T0["test/test_adapt_vqe_integration.py"]
  end

  R --> D1
  R --> D2
  R --> D3

  D2 --> P0
  D2 --> P3
  D2 --> P5

  P0 --> P1
  P0 --> P2
  P3 --> P4

  P1 --> S0
  P2 --> S0
  P2 --> S1
  P1 --> RP0
  P2 --> RP0

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
    GSelect --> Reopt["Re-optimize all parameters (HH workflow: SPSA)"]
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
- Experimental offline/local exact-noiseless probe families: `paop_lf3_std`, `paop_lf4_std`, `paop_sq_std`, `paop_sq_full`.
- HH direct ADAPT default for new agent work: `phase3_v1` starts from the narrow HH core and runtime-resolves depth-0 HH ADAPT to `paop_lf_std`.
- Older repo-doc statements that staged wrappers default to `phase1_v1` are archival only; the manuscript/current canonical HH ADAPT surface uses `phase3_v1`, and manuscript guidance wins when those older docs disagree.
- HH built-in combined preset: `uccsd_paop_lf_full` = `uccsd_lifted + paop_lf_full` (deduplicated) via one CLI value.
- HH explicit product families: `uccsd_otimes_paop_lf_std`, `uccsd_otimes_paop_lf2_std`, `uccsd_otimes_paop_bond_disp_std`.
  - These are the canonical lifted-UCCSD ⊗ boson-only-phonon constructions in this repo: one lifted fermionic UCCSD factor times one boson-only phonon motif, locality-filtered, canonicalized, and deduplicated.
  - They are available as explicit families for seed-refine, replay, and direct ADAPT pool materialization without mutating the older additive unions.
- HH logical two-parameter product variants: `uccsd_otimes_paop_lf_std_seq2p`, `uccsd_otimes_paop_lf2_std_seq2p`, `uccsd_otimes_paop_bond_disp_std_seq2p`.
  - These treat one logical `(F_a, M_μ)` pair as separate fermion/motif parameters during execution and replay.
  - They are additive opt-in surfaces and do not change the direct `phase3_v1` default path.
- HH full-meta preset: `full_meta` = `uccsd_lifted + hva + paop_full + paop_lf_full` (deduplicated) via one CLI value; keep it as a compatibility/broad-pool preset and replay fallback, not the default depth-0 staged HH pool.
- HH lean replay/export presets: `pareto_lean` and `pareto_lean_l2`.
  - `pareto_lean_l2` is intentionally narrow: valid only for `L=2` and `n_ph_max=1`.
- Phase-3 runtime split is disabled on the canonical public HH run surfaces. The old shortlist Pauli-child split implementation is retained only as an internal archival/testing path and is not part of the manuscript-facing algorithm contract.
- ADAPT/replay parameter contract:
  - `operators` / `ansatz_depth` remain the logical generator scaffold.
  - `optimal_point` / `num_parameters` are the runtime per-Pauli rotation vector/count.
  - `logical_optimal_point` / `logical_num_parameters` preserve one-value-per-generator reporting.
  - `parameterization` stores the logical-to-runtime block map used by replay and cost reconstruction.
- `paop_min`: displacement-focused PAOP operators.
- `paop_std`: displacement plus dressed-hopping (`hopdrag`) operators.
- `paop_full`: `paop_std` plus doublon dressing and extended cloud operators.
- `paop_lf_std`: `paop_std` plus LF-leading odd channel (`curdrag`).
- These experimental families are opt-in only; they are not part of the canonical direct `phase3_v1` default path and are not folded into default `full_meta`.
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
- `vqe_minimize(...)` supports `energy_backend="legacy"|"one_apply_compiled"` (default is `one_apply_compiled`).
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
Fast VQE-from-ADAPT replay (HH, ADAPT-family matched):

```bash
python pipelines/hardcoded/hh_vqe_from_adapt_family.py \
  --adapt-input-json <adapt_hh_json_path> \
  --generator-family match_adapt --fallback-family full_meta \
  --L 4 --boundary open --ordering blocked \
  --boson-encoding binary --n-ph-max 1 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 \
  --reps 4 --restarts 16 --maxiter 12000 --method SPSA --seed 7 \
  --energy-backend one_apply_compiled --progress-every-s 60 \
  --output-json artifacts/json/hc_hh_L4_from_adaptB_family_matched_fastcomp.json
```

This path matches the ADAPT generator-family contract. Replay remains the matched-family follow-on path via `--generator-family match_adapt` with `--fallback-family full_meta`, and replay continuation modes stay `legacy | phase1_v1 | phase2_v1 | phase3_v1`.
If an opt-in runtime split admitted child labels outside the resolved family pool, replay can still rebuild them from serialized continuation metadata when that metadata is present.
`hubbard_pipeline.py --vqe-ansatz hh_hva_*` remains a fixed-ansatz baseline.

## Start here (doc priority)

Use this order when onboarding:

1. `AGENTS.md` - repo conventions and non-negotiable implementation rules
2. `pipelines/run_guide.md` - CLI and runbook for active pipelines
3. `README.md` - current repo map and kept workflow surface
4. `MATH/IMPLEMENT_NEXT.md` - next HH implementation target when math work is in scope

Canonical authority chain: `AGENTS.md` -> `pipelines/run_guide.md` -> `README.md` -> task-specific `MATH/` notes.
Agent-facing automation should ignore `docs/` unless PDF/report output is in scope, in which case use `docs/reports/`.

Task-type doc split:
- `AGENTS.md`: hard policy and escalation rules.
- `pipelines/run_guide.md`: executable commands and run contracts.
- `README.md`: repo map and active workflow overview.
- `MATH/IMPLEMENT_NEXT.md`: near-term HH implementation target.

## Important note on README files

Subdirectory README files are component-scoped documentation, not repo-canonical
onboarding docs. Use this root `README.md` first, then drill into local READMEs
for module-specific details.

## AI run/report contract

For agent-run work in this repo:

- frame the run around an **Objective** first: a short scientific / mathematical / physical sub-problem that could improve the real-QPU `ΔE / K` Pareto front
- keep **Objective** separate from **Execution mode**:
  - `fresh_run`
  - `reuse_artifact`
  - `compare_artifacts`
  - `promote_candidate`
- default emphasis is **HH**, **`L=2`**, and **driven** dynamics unless the user says otherwise; this is an **agent planning priority**, not a CLI default
- in RepoPrompt agent mode, default to three compact lines with **no blank lines**: `Objective<...>`, `Why/Intent<...>`, `Suggested Next step/how this fits into broader picture<...>`; each line should be **1-3 sentences max**
- frame verification as soft expectations by default, unless the user or the chosen repo surface explicitly defines a hard gate
- the **agent wrapper** should write machine-oriented logs under:
  - `artifacts/agent_runs/<tag>/`
  - `artifacts/agent_runs/<tag>/logs/`
  - with `command.sh`, `stdout.log`, `stderr.log`, and `progress.json` when supported
- if the user says **execute**, the agent should execute without an extra permission prompt unless a real runtime/policy choice is still unresolved
- as an **agent post-processing convention**, after each run first give a short in-chat report that retells the objective and result; only write/update markdown or PDF report files when the user explicitly asks or report output is already in scope

## Quick run examples

Default hard gate policy for agent execution:
- Final conventional VQE hard gate: `ΔE_abs < 1e-4`.
- In this checkout, `run_drive_accurate.sh` enforces `ΔE_abs < 1e-7` with no built-in strict-mode toggle. This is stricter than the AGENTS default.

ADAPT-VQE (HH, canonical direct phase3 path):

```bash
python pipelines/hardcoded/adapt_pipeline.py \
  --L 2 --problem hh --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-max-depth 30 --adapt-eps-grad 1e-5 --adapt-maxiter 800 \
  --initial-state-source adapt_vqe --skip-pdf \
  --output-json artifacts/json/adapt_L2_hh_phase3_default.json
```

Omitting `--adapt-continuation-mode` on the direct CLI now means `phase3_v1`. Pass `legacy`, `phase1_v1`, or `phase2_v1` explicitly only when reproducing older behavior.

Runtime note: ADAPT execution now applies one variational parameter per active Pauli term inside each selected generator. Exported JSON therefore distinguishes logical scaffold size (`ansatz_depth`, `logical_*`) from runtime rotation count (`num_parameters`, `optimal_point`).

Cross-check suite (exact benchmark; auto-scaled by L/problem defaults):

```bash
python pipelines/exact_bench/cross_check_suite.py --L 2 --problem hubbard

# HH exact-benchmark point:
python pipelines/exact_bench/cross_check_suite.py \
  --problem hh --L 2 --omega0 1.0 --g-ep 0.5 --n-ph-max 1
```

Cross-check note:
- In this checkout, `cross_check_suite.py --help` exposes only the benchmark-matrix CLI shown above.
- Older README references to `--hh-seed-refine-surface` and `--hh-seed-benchmark-preset` were stale and are removed here.

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
- `--propagator` defaults to `cfqm4`; use `suzuki2` explicitly for baseline comparison.
- CFQM uses fixed scheme nodes (`c_j`) and ignores legacy midpoint/left/right `--drive-time-sampling`.
- `--exact-steps-multiplier` remains a reference-only control and does not change CFQM macro-step count.
- `--cfqm-stage-exp` default is `expm_multiply_sparse`; `--cfqm-coeff-drop-abs-tol` default is `0.0`; `--cfqm-normalize` default is off.
- `--cfqm-stage-exp expm_multiply_sparse` and `--cfqm-stage-exp dense_expm` are the true numerical CFQM stage backends.
- Sparse CFQM stage backend uses native sparse stage assembly + `scipy.sparse.linalg.expm_multiply` (no dense->csc stage materialization).
- Shared Pauli-term exponentiation helpers are centralized in `src/quantum/pauli_actions.py` (used by both the hardcoded pipeline and CFQM backend).
- Unknown drive labels are handled with a guardrail policy: nontrivial coefficients warn once per label then are ignored; tiny coefficients (`abs(coeff) <= 1e-14`) are silently ignored.
- A=0 invariance is preserved by the zero-increment insertion guard; safe-test target is `<= 1e-10`.
- If non-midpoint sampling is supplied under CFQM, runtime warns:
  `CFQM ignores midpoint/left/right sampling; uses fixed scheme nodes c_j.`
- If `--cfqm-stage-exp pauli_suzuki2` is selected, runtime warns:
  `Inner Suzuki-2 makes overall method 2nd order; use expm_multiply_sparse/dense_expm for true CFQM order.`
- `--cfqm-stage-exp pauli_suzuki2` is the only circuitizable CFQM profile in this repo. It is a hardware-facing surrogate, not true CFQM4/CFQM6 order.
- Honest compiled/transpiled/CX/QPU artifacts exist only for `pauli_suzuki2`; dense/sparse CFQM remain numerical-only and are rejected or skipped by report/transpile/hardware-facing paths.
- Pitfall: a `compiled/transpiled CFQM4 circuit` in this repo can describe the `pauli_suzuki2` surrogate profile rather than the true numerical CFQM4 implementation.

Current fixed-manifold / realtime surfaces:
- exact locked-manifold compare:

```bash
python -m pipelines.hardcoded.hh_fixed_manifold_mclachlan
```

- measured/oracle fixed-manifold run:

```bash
python -m pipelines.hardcoded.hh_fixed_manifold_measured \
  --manifold locked_7term --enable-drive \
  --drive-A 0.6 --exact-steps-multiplier 2
```

- L=2 static realtime sweep from saved artifacts:

```bash
python -m pipelines.hardcoded.hh_l2_static_realtime_pareto_sweep
```

- L=2 driven realtime sweep from saved artifacts:

```bash
python -m pipelines.hardcoded.hh_l2_driven_realtime_pareto_sweep
```

Notes:
- Use `python -m ...` for these newer hardcoded modules; some do not support direct file-path invocation cleanly.
- `hh_fixed_manifold_measured.py` currently supports `noise_mode=ideal`, `oracle_repeats=1`, and mean aggregation only.
- The `hh_l2_*_realtime_pareto_sweep.py` surfaces are specifically `L=2` saved-artifact sweeps, not generic `run L` wrappers.

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

Missing benchmark-helper note:
- Older README text referenced `cfqm_vs_suzuki_qproc_proxy_benchmark.py` and `cfqm_vs_suzuki_efficiency_suite.py`.
- Those scripts are **not present in this checkout**.
- The active CFQM surface here is `pipelines/hardcoded/hubbard_pipeline.py` plus the CFQM test trio:

```bash
pytest -q test/test_cfqm_schemes.py test/test_cfqm_propagator.py test/test_cfqm_acceptance.py
```

For compare/orchestration workflows that still exist in this checkout, use `pipelines/run_guide.md`.

## Major Markdown docs index

- `AGENTS.md`
- `README.md`
- `pipelines/run_guide.md`
- `MATH/IMPLEMENT_NEXT.md`
- `MATH/IMPLEMENT_SOON.md`
- `pipelines/exact_bench/README.md`

Legacy archived docs live under `docs/archive/` and are non-canonical.

## HH noisy estimator validation

The repo now includes an HH-first noisy/hardware validation pipeline:
- `pipelines/exact_bench/hh_noise_hardware_validation.py`

It provides one shared expectation oracle across `ideal`, `shots`, `aer_noise`, and `runtime` modes.  
`shots`/`aer_noise` emulate finite-shot measurement noise using Qiskit `AerSimulator`, with optional noisy ADAPT and PDF/JSON reporting.  
Use the current-surface summary at the top of `pipelines/run_guide.md` for operational commands.

High-level symmetry note:
- `--symmetry-mitigation-mode` is the active oracle-backed symmetry surface in the noise validation / robustness flows; default is `off`.
- Active modes (`postselect_diag_v1`, `projector_renorm_v1`) are intentionally narrow first versions: they run only on eligible diagonal/counts-compatible paths and fall back explicitly to `verify_only` when unsupported.
- This differs from `--phase3-symmetry-mitigation-mode` on raw staged ADAPT / hardcoded / replay paths, where the flag is a continuation metadata/telemetry hook unless the workflow is routed through the oracle runtime.

For staged heavy HH robustness with warm-start -> ADAPT Pool B -> final VQE and noisy dynamics benchmarking:
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py`

Key additions:
- strict ADAPT Pool B composition enforcement (`UCCSD_lifted + HVA + PAOP_full`)
- noisy dynamics methods via `--noisy-methods` (default `suzuki2,cfqm4`)
- when a CFQM noisy run is turned into circuit/proxy/transpile metrics, the hardware-facing surface is the `pauli_suzuki2` surrogate rather than dense/sparse numerical CFQM
- shared oracle-backed `--symmetry-mitigation-mode` surface (default `off`; active modes remain opt-in and diagnostics-backed)
- embedded benchmark metrics in JSON/PDF (`term_exp_count_total`, `cx_proxy_total`, `sq_proxy_total`, `depth_proxy_total`, `wall_total_s`, `oracle_eval_s_total`)
- backward-compatible `dynamics_noisy.profiles.<profile>.modes` alias mirroring `suzuki2`
