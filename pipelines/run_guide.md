<!--
This guide is consumed by AI coding agents.

Editing contract:
- Keep machine-parseable: use tables, code blocks, explicit paths.
- Every artifact path must be relative to repo root.
- Every command must be copy-pasteable from repo root.
- Mark broken/historical items explicitly so agents skip them.
- When adding new workflows, put them in the right tier (QPU > noisy sim > noiseless).
-->

# HH Run Guide

**Target system:** Hubbard–Holstein (HH), L=2, `t=1.0 U=4.0 g_ep=0.5 omega0=1.0 n_ph_max=1`, open boundary, blocked ordering, binary boson encoding. 6 qubits total (4 fermion + 2 phonon).

**Target QPU:** `ibm_marrakesh` (Heron r2, CZ-native, heavy-hex 156q).

**Current state:** This guide still contains the detailed 20260323 Marrakesh 7-term lane, but the checkout now also has newer fixed-manifold / realtime sweep surfaces (20260325-20260326) plus the canonical direct HH `phase3_v1` ADAPT path.

---

## 0. Current surface map (2026-03-26)

Use this section first. Sections 1-6 remain the narrow Marrakesh 7-term line; they are not the whole active run surface.

| Intent | Preferred entrypoint | Invocation note | Scope |
|--------|-----------------------|-----------------|-------|
| canonical direct HH ADAPT | `pipelines/hardcoded/adapt_pipeline.py` | direct CLI; HH omission defaults to `phase3_v1` | current default |
| staged HH compatibility workflow | `pipelines/hardcoded/hh_staged_noiseless.py` | historical wrapper | compatibility |
| staged HH noisy/import-side follow-ons | `pipelines/hardcoded/hh_staged_noise.py` | local noisy extension and imported fixed-scaffold routes | current |
| HH noise validation / parity | `pipelines/exact_bench/hh_noise_hardware_validation.py` | structured JSON/PDF validator | current |
| HH exact cross-check matrix | `pipelines/exact_bench/cross_check_suite.py` | current CLI only; older seed-surface flags are absent here | current |
| fixed-manifold exact compare | `python -m pipelines.hardcoded.hh_fixed_manifold_mclachlan` | use `-m` | current saved-artifact exact compare |
| fixed-manifold measured/oracle run | `python -m pipelines.hardcoded.hh_fixed_manifold_measured --manifold locked_7term --enable-drive --drive-A 0.6 --exact-steps-multiplier 2` | use `-m`; currently `noise_mode=ideal` only | current |
| L=2 static realtime sweep | `python -m pipelines.hardcoded.hh_l2_static_realtime_pareto_sweep` | use `-m`; writes `summary.json` + `progress.json` | current L=2-only |
| L=2 driven realtime sweep | `python -m pipelines.hardcoded.hh_l2_driven_realtime_pareto_sweep` | use `-m`; writes `summary.json` + `progress.json` + PNG | current L=2-only |
| literal legacy shorthand | `bash pipelines/shell/run_drive_accurate.sh --L <L>` | legacy helper; script gate is `1e-7`, not AGENTS `1e-4` | legacy |

Quick commands:

```bash
# Canonical direct HH ADAPT
python pipelines/hardcoded/adapt_pipeline.py \
  --L 2 --problem hh --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-max-depth 30 --adapt-eps-grad 1e-5 --adapt-maxiter 800 \
  --initial-state-source adapt_vqe --skip-pdf \
  --output-json artifacts/json/adapt_hh_L2_phase3_v1.json

# Current fixed-manifold exact compare
python -m pipelines.hardcoded.hh_fixed_manifold_mclachlan

# Current fixed-manifold measured/driven run
python -m pipelines.hardcoded.hh_fixed_manifold_measured \
  --manifold locked_7term --enable-drive \
  --drive-A 0.6 --exact-steps-multiplier 2

# Current driven L=2 realtime sweep
python -m pipelines.hardcoded.hh_l2_driven_realtime_pareto_sweep
```

Notes:
- Use `python -m` for the newer fixed-manifold / realtime sweep modules; direct file-path invocation can fail on imports.
- `hh_fixed_manifold_measured.py` currently supports only `noise_mode=ideal`, `oracle_repeats=1`, and mean aggregation.
- `hh_l2_static_realtime_pareto_sweep.py` and `hh_l2_driven_realtime_pareto_sweep.py` are saved-artifact `L=2` workflows, not generic `run L` wrappers.
- `run_drive_accurate.sh` remains a legacy shorthand helper with a stricter `1e-7` gate than the AGENTS/README shorthand target.

---

## 0a. Agent run/report contract

| Field | Contract |
| --- | --- |
| Objective | short scientific / mathematical / physical sub-problem that could improve the real-QPU `ΔE / K` Pareto front |
| Execution mode | keep separate from objective; use `fresh_run`, `reuse_artifact`, `compare_artifacts`, or `promote_candidate` |
| Default emphasis | HH, `L=2`, driven dynamics, QPU-preparatory; this is an agent planning priority, not a universal CLI default |
| Verification style | soft expectations / sanity targets by default; preserve explicit hard gates when the user or the chosen repo surface defines them |
| Logging | the agent wrapper should write `artifacts/agent_runs/<tag>/logs/command.sh`, `stdout.log`, `stderr.log`, and `progress.json` when supported; native CLI outputs still vary by surface |
| Execute behavior | if the user says `execute`, run without an extra **repo-level** confirmation unless a real runtime/policy choice remains unresolved; still obey host sandbox/approval policy |
| Auto-report | agent post-processing convention: after a run, first give a short in-chat report that retells the objective and result; only write/update markdown or PDF report files when report output is in scope or the user explicitly asks |
| Style | in RepoPrompt agent mode, default to three compact lines with no blank lines: `Objective<...>`, `Why/Intent<...>`, `Suggested Next step/how this fits into broader picture<...>`; each line should be 1-3 sentences max and logic/math/physics-first |

## 0b. Skill-aware agent routing

When the relevant skills are available in the agent environment:

- Use `hh-experiment` for **run choice, run planning, or execution**.
- Use `hh-reporting` for **artifact interpretation, comparison, or report generation**.
- When report output is in scope and the user has not narrowed the task to execution-only, the default agent convention after a completed run is:
  1. execute through the run path
  2. collect logs + structured artifacts
  3. hand off immediately to the reporting path for a short objective-aware retell

Clarifications:

- These skills do **not** replace the repo run surfaces listed in Section 0.
- `hh-experiment` is the wrapper for choosing/planning/executing those surfaces.
- `hh-reporting` is the wrapper for artifact interpretation, report generation, and post-run summary updates.
- These skills do **not** outrank repo policy. Authority remains `AGENTS.md` -> `pipelines/run_guide.md` -> `README.md`.
- If a skill expectation and current code/CLI behavior diverge in a way that affects the run/report choice, stop and ask the user before proceeding.
- The reporting path should default to the compact three-line agent-mode format: `Objective<...>`, `Why/Intent<...>`, `Suggested Next step/how this fits into broader picture<...>`, with no blank lines and 1-3 sentences max per line. Persistent markdown/PDF report files are optional and should only be written when requested or already in scope.

---

## 1. 20260323 Marrakesh 7-term QPU lane

This is a specific historical/narrow workflow. Each step has a concrete artifact and script.

### 1.1 Pipeline overview

```
ADAPT-VQE (noiseless, done)
  → Gate pruning + Pareto analysis (done)
    → Marrakesh-conditioned compilation (done)
      → Noisy VQE re-optimization (FakeNighthawk done; Marrakesh TODO)
        → Fixed-theta QPU submission (ibm_marrakesh)
          → Energy-only |ΔE| evaluation
```

### 1.2 Trustworthy artifacts

| Artifact | Path | Status | Notes |
|----------|------|--------|-------|
| Gate-pruned 7-term (ADAPT order) | `artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json` | **Validated** | |ΔE|=4.2e-4, fidelity=0.9998, 25 CZ on Marrakesh |
| Circuit-optimized 7-term (disp_first) | `artifacts/json/hh_prune_nighthawk_circuit_optimized_7term.json` | **BROKEN** | Variationally broken (|ΔE|=0.34) — disp_first ordering breaks non-commuting per_pauli_term rotations |
| Marrakesh Pareto prune menu | `artifacts/json/marrakesh_gatepruned7_prune_menu_20260323.json` | **Validated** | Frozen-theta Pareto sweep: 7/6/5-term rows with Marrakesh compile metrics |
| Marrakesh reoptimized prune menu | `artifacts/json/marrakesh_gatepruned7_reoptimized_prune_menu_20260323.json` | Current | Re-optimized parameters for pruned variants |
| Noisy VQE (FakeNighthawk, M3 ON) | `artifacts/json/hh_noisy_vqe_7term_20260323T050918Z.json` | **Validated** | 8192 shots, 200 SPSA iter, SV energy 0.1602 (|ΔE|=1.5e-3) |
| Fixed-theta Marrakesh QPU result | `artifacts/json/hh_gatepruned7_fixedtheta_runtime_ibm_marrakesh_20260323T145219Z_reconstructed.json` | Poor | Fixed-theta run; needs noisy re-opt |
| Marrakesh compile comparison | `artifacts/json/investigation_marrakesh_compile_compare_20260323.json` | Reference | Compile costs across all artifact variants |

### 1.3 Marrakesh Pareto frontier (frozen-theta)

From `marrakesh_gatepruned7_prune_menu_20260323.json`:

| Terms | Kept indices | Dropped | 2Q gates | Depth | |ΔE| | Fidelity | Notes |
|-------|-------------|---------|----------|-------|------|----------|-------|
| 7 | [0,1,2,3,4,5,6] | — | 25 | 63 | 4.2e-4 | 0.9998 | Full gate-pruned circuit |
| **6** | [0,1,2,3,4,6] | [5]=eyeeez | **14** | 56 | 5.0e-3 | 0.9979 | **Best Pareto: 44% gate reduction** |
| 5 | [0,1,2,3,4] | [5,6] | 12 | 48 | 7.0e-3 | 0.9986 | Marginal gate savings over 6-term |
| 4 | — | — | — | — | >0.5 | <0.5 | **Accuracy cliff — not viable** |

The 6-term circuit at 14 CZ is at the theoretical minimum for independent per_pauli_term rotations. To go below 14, CNOT cascade sharing between adjacent terms is needed (see Section 5).

### 1.4 The 7 runtime terms

These are the per-Pauli-term rotations in ADAPT ordering. Each applies `exp(-i·θ/2·P)`.

| Index | Pauli (exyz) | Coeff | Label | Active qubits | CZ cost |
|-------|-------------|-------|-------|---------------|---------|
| 0 | `eeeexy` | -0.5 | uccsd_sing(alpha:0→1) | {4,5} | 2 |
| 1 | `eeyxee` | 0.5 | uccsd_sing(beta:2→3) | {2,3} | 2 |
| 2 | `yeeeee` | 0.25 | paop_dbl_p(site=1→phonon=1) | {0} | 0 |
| 3 | `yezeze` | 0.25 | paop_dbl_p(site=1→phonon=1) | {0,2,4} | 4 |
| 4 | `yeyyee` | -0.5 | paop_hopdrag(0,1)::child_set[0,2] | {0,2,3} | 4 |
| 5 | `eyeeez` | -0.5 | paop_disp(site=0) | {1,5} | 2 |
| 6 | `eyezee` | -0.5 | paop_disp(site=0) | {1,3} | 2 |

**Qubit mapping convention:** exyz position `i` → Qiskit qubit `nq - 1 - i` (position 0 = MSB in kron product = Qiskit qubit 5).

### 1.5 Key scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `pipelines/hardcoded/hh_prune_nighthawk.py` | Pruning, re-optimization, scaffold export, Pareto analysis | Active |
| `pipelines/hardcoded/hh_prune_marginal_analysis.py` | Marginal term-drop analysis | Active |
| `pipelines/hardcoded/hh_noisy_vqe_7term.py` | Noisy VQE on FakeNighthawk with M3 + SPSA | Active |
| `pipelines/hardcoded/hh_backend_compile_oracle.py` | Multi-backend compilation and ranking | Active |
| `pipelines/hardcoded/adapt_circuit_cost.py` | Compile scout (transpile + rank by 2Q/depth) | Active |
| `pipelines/hardcoded/adapt_circuit_execution.py` | Circuit execution helpers | Active |
| `pipelines/hardcoded/imported_artifact_resolution.py` | Default artifact routing (prefers gate_pruned over circuit_optimized) | Active |
| `pipelines/hardcoded/hh_staged_noise_workflow.py` | Staged noise workflow (local imported fixed-scaffold scout/replay/attribution) | Active |
| `pipelines/exact_bench/hh_noise_robustness_seq_report.py` | Full-circuit audit + noise attribution | Active |
| `pipelines/exact_bench/noise_oracle_runtime.py` | Oracle runtime (shots/aer_noise/runtime modes) | Active |

---

## 2. Commands (copy-paste from repo root)

### 2.1 Gate pruning and Pareto analysis

```bash
# Full 7-term pruning with scaffold VQE re-optimization
python -m pipelines.hardcoded.hh_prune_nighthawk \
  --input-json artifacts/json/hh_backend_adapt_fullhorse_powell_20260322T194155Z_fakenighthawk.json \
  --mode both \
  --prune-threshold 1e-4 \
  --output-json artifacts/json/hh_prune_nighthawk_$(date -u +%Y%m%dT%H%M%SZ).json
```

```bash
# Export fixed-scaffold 7-term artifacts (gate_pruned + circuit_optimized)
python -m pipelines.hardcoded.hh_prune_nighthawk \
  --mode export_fixed_scaffolds \
  --source-artifact-json artifacts/json/hh_prune_nighthawk_aggressive_5op.json
```

### 2.2 Compile scout (rank backends by 2Q gate count)

```bash
python -m pipelines.hardcoded.adapt_circuit_cost \
  --artifact-json artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json \
  --backend-name ibm_marrakesh \
  --seed-transpiler 7 --optimization-level 1 \
  --sweep-backends
```

### 2.3 Noisy VQE on FakeNighthawk (7-term, M3 + SPSA)

```bash
python -m pipelines.hardcoded.hh_noisy_vqe_7term \
  --shots 8192 --maxiter 200 --reps 3
```

Output: `artifacts/json/hh_noisy_vqe_7term_<timestamp>.json`

Key implementation details for agents reproducing this:
- Circuit uses ADAPT ordering [0,1,2,3,4,5,6] — **not** disp_first
- `AerSimulator.from_backend(FakeNighthawk())` — preserves full 127-qubit layout
- M3 mitigator calibrated on the 6 logical qubits after transpilation
- Parameter binding is direct: `theta[i]` → `params[i]`, no coefficient scaling
- SparsePauliOp labels are **reversed** vs exyz: exyz `eeeexy` → SparsePauliOp `YXEEEE` (Qiskit convention)
- HF reference state: qubit 5 (exyz position 0) = |1⟩, qubit 4 (exyz position 1) = |1⟩ for spin-up; qubit 3 (exyz position 2) = |1⟩, qubit 2 (exyz position 3) = |1⟩ for spin-down

### 2.4 Local fixed-scaffold compile-control scout (gate-pruned 7-term)

This is the committed local scout route for the active 7-term line. It is **not** the later Runtime energy-only submission path.

```bash
python -m pipelines.hardcoded.hh_staged_noise \
  --include-fixed-scaffold-compile-control-scout \
  --fixed-final-state-json artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json \
  --use-fake-backend --backend-name FakeNighthawk \
  --shots 4096 --oracle-repeats 8 --oracle-aggregate mean
```

Contract:
- locked imported gate-pruned 7-term scaffold only
- local fake-backend only
- backend_scheduled + readout/mthree
- small transpile seed / optimization-level grid
- energy-only ranking, not continuous parameter optimization

Historical note:
- `--include-energy-only-import` is **not** a committed CLI route here; treat older references to that flag as stale.
- The later real Runtime energy-only path remains a separate follow-on slice.

### 2.5 Full-circuit audit (multi-observable, for diagnostics only)

```bash
python -m pipelines.hardcoded.hh_staged_noise \
  --include-full-circuit-audit \
  --fixed-final-state-json artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json \
  --use-fake-backend --backend-name FakeNighthawk \
  --mitigation readout --local-readout-strategy mthree \
  --local-gate-twirling \
  --shots 4096 --oracle-repeats 8 --oracle-aggregate mean
```

Notes:
- `--local-gate-twirling` is an opt-in local fake-backend heuristic using Qiskit circuit-level 2Q Pauli twirling on the compiled `backend_scheduled` base circuit.
- It is meant to approximate the Runtime gate-twirling direction locally; it is **not** a full local TREX/measurement-twirling implementation.

### 2.5a Fixed-scaffold noisy replay (active local 6-term replay route)

```bash
python -m pipelines.hardcoded.hh_staged_noise \
  --include-fixed-scaffold-noisy-replay \
  --fixed-final-state-json artifacts/json/hh_marrakesh_fixed_scaffold_6term_drop_eyezee_20260323T171528Z.json \
  --use-fake-backend --backend-name FakeMarrakesh \
  --final-method SPSA \
  --mitigation readout --local-readout-strategy mthree \
  --shots 2048 --oracle-repeats 4 --oracle-aggregate mean
```

Contract:
- pinned imported Marrakesh/Heron **gate-pruned 6-term** scaffold
- local fake-backend only
- initializes from saved imported `theta_runtime`
- replay optimizer runs under `backend_scheduled` noise with readout/mthree
- `--local-gate-twirling` is the opt-in in-loop local twirling variant
- `--dd-sequence XpXm` adds a **saved-theta local DD probe** block only; it is not optimizer-loop DD

### 2.6 Runtime energy-only baseline (active 6-term Marrakesh/Heron candidate)

This is the committed **real Runtime** fixed-scaffold baseline for the active
Heron/Marrakesh candidate. It is separate from the local fake-backend scout.

```bash
python -m pipelines.hardcoded.hh_staged_noise \
  --include-fixed-scaffold-runtime-energy-only-baseline \
  --backend-name ibm_marrakesh \
  --shots 4096 --oracle-repeats 8 --oracle-aggregate mean
```

Contract:
- defaults to the imported Marrakesh/Heron **gate-pruned 6-term** candidate when no import JSON is given
- requires real Runtime backend access (`--use-fake-backend` must stay off)
- default Runtime profile is `main_twirled_readout_v1`
- default Runtime batching policy is `require_session`
- default mitigation surface is explicit Runtime readout + measure twirling + 2Q gate twirling
- symmetry mitigation is forced off on this route today
- DD is **probe-only** via `--include-fixed-scaffold-runtime-dd-probe`
- ZNE is **final-audit-only** via `--include-fixed-scaffold-runtime-final-zne-audit`

---

## 3. Critical Implementation Details

Agents must get these right or the circuit produces garbage.

### 3.1 Qubit mapping (exyz → Qiskit)

```
exyz position i  →  Qiskit qubit (nq - 1 - i)

exyz:   [0]  [1]  [2]  [3]  [4]  [5]
Qiskit:  q5   q4   q3   q2   q1   q0
```

This is because exyz position 0 is MSB in the kron product, while Qiskit qubit 0 is LSB.

### 3.2 SparsePauliOp label convention

Qiskit SparsePauliOp labels are **right-to-left**: label index 0 = qubit 0 (LSB).

```python
# exyz "eeeexy" → active at exyz positions 4(x), 5(y)
#   → Qiskit qubits 1(x), 0(y)
#   → SparsePauliOp label "EEEEXY" reversed = "YXEEEE"
# But SparsePauliOp uses I not E:
"YXIIII"  # qubit 0=Y, qubit 1=X, rest identity
```

### 3.3 Per-Pauli-term rotation circuit

Each term applies `exp(-i·θ/2·P)`:

```python
def _pauli_rotation_circuit(nq, pauli_str, param):
    qc = QuantumCircuit(nq)
    active = []
    for i, p in enumerate(pauli_str):
        q = nq - 1 - i  # exyz position → Qiskit qubit
        if p == "x":
            qc.h(q); active.append(q)
        elif p == "y":
            qc.sdg(q); qc.h(q); active.append(q)
        elif p == "z":
            active.append(q)
    # CNOT ladder
    for j in range(len(active) - 1):
        qc.cx(active[j], active[j + 1])
    # RZ on last active qubit
    qc.rz(param, active[-1])
    # Reverse CNOT ladder
    for j in range(len(active) - 2, -1, -1):
        qc.cx(active[j], active[j + 1])
    # Undo basis changes (reverse order)
    for i, p in enumerate(pauli_str):
        q = nq - 1 - i
        if p == "x":
            qc.h(q)
        elif p == "y":
            qc.h(q); qc.s(q)
    return qc
```

### 3.4 Operator ordering matters

Per_pauli_term rotations **do not commute**. The ADAPT ordering [0,1,2,3,4,5,6] is the only variationally validated ordering.

The `disp_first` ordering [5,6,2,3,4,0,1] compiles to 16 CZ (vs 18 for ADAPT order on Nighthawk) but is **variationally broken**: a global optimizer with 420k evaluations can only reach |ΔE|=0.34. This is because reordering changes the energy landscape fundamentally.

### 3.5 Parameter binding

The circuit parameters are the raw `θ` values passed directly to `RZ(θ)` gates. There is no coefficient scaling at the circuit level. The executor uses `exp(-i·dt·coeff·P)` internally, but the circuit uses `exp(-i·θ/2·P)`, so:

```
θ_circuit = 2 · dt · coeff
```

When initializing from executor parameters, convert: `theta_circuit[i] = 2 * dt[i] * coeff[i]`.

### 3.6 Hartree-Fock reference state

Half-filled: 1↑ electron at site 0, 1↓ electron at site 0, for L=2 HH with n_ph_max=1.

```python
# exyz positions: [phonon1, phonon0, dn1, dn0, up1, up0]
# HF occupies: up0 (exyz pos 5 → qubit 0), dn0 (exyz pos 3 → qubit 2)
# Wait — use the canonical helper:
from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
hf = hubbard_holstein_reference_state(L=2, n_ph_max=1, encoding="binary")
```

### 3.7 Reference energies

| Energy | Value | Source |
|--------|-------|--------|
| Sector-filtered exact GS (1↑,1↓) | **0.15867** | `build_hh_sector_hamiltonian_ed()` |
| Full Hamiltonian GS | -0.15068 | Full 2⁶ diagonalization (wrong sector) |
| 7-term noiseless (ADAPT order) | 0.15909 | |ΔE|=4.2e-4 |
| 7-term noisy (FakeNighthawk, M3) | 0.228 | |ΔE|=0.07 (gate noise dominated) |
| 7-term SV at SPSA-best θ | 0.1602 | |ΔE|=1.5e-3 |

The VQE targets the sector-filtered energy (0.15867), not the full-Hamiltonian GS (-0.15068).

---

## 4. Noise and Mitigation

### 4.1 Noise budget (FakeNighthawk, 7-term circuit)

| Source | Contribution | Mitigable? |
|--------|-------------|------------|
| Readout error | ~0.03 | Yes (M3 mthree) |
| Gate error | ~0.07 | No (would need ZNE/PEC) |
| SPSA convergence | ~1.5e-3 | Better with more iterations |
| Variational approximation | 4.2e-4 | Fundamental limit of 7-term ansatz |

Gate noise dominates. M3 fixes only readout.

### 4.2 AerSimulator creation (agents: get this right)

```python
# CORRECT — preserves full backend layout (127 qubits for Nighthawk)
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeNighthawk
sim = AerSimulator.from_backend(FakeNighthawk())

# WRONG — creates only nq qubits, transpilation fails on 127q layout
from qiskit_aer.noise import NoiseModel
noise = NoiseModel.from_backend(FakeNighthawk())
sim = AerSimulator(noise_model=noise)  # only 30 qubits!
```

### 4.3 M3 readout error mitigation

```python
import mthree
mit = mthree.M3Mitigation(sim)
mit.cals_from_system(qubits=physical_qubits, shots=8192)
# After execution:
quasi_dist = mit.apply_correction(counts, qubits=physical_qubits)
```

---

## 5. Circuit Reduction Strategies

### 5.1 Current theoretical minimum

The 6-term circuit (indices [0,1,2,3,4,6]) is at **14 CZ** — the theoretical minimum for independent per_pauli_term rotations:

| Term | Active qubits | k | 2(k-1) CZ |
|------|--------------|---|-----------|
| eeeexy | {4,5} | 2 | 2 |
| eeyxee | {2,3} | 2 | 2 |
| yeeeee | {0} | 1 | 0 |
| yezeze | {0,2,4} | 3 | 4 |
| yeyyee | {0,2,3} | 3 | 4 |
| eyezee | {1,3} | 2 | 2 |
| **Total** | | | **14** |

### 5.2 CNOT cascade sharing (to go below 14 CZ)

Terms T3 (`yezeze`, qubits {0,2,4}) and T4 (`yeyyee`, qubits {0,2,3}) share the sub-chain qubit 0 → qubit 2. If placed adjacent with a shared CNOT prefix, the undo of T3's 0→2 leg cancels with T4's compute of 0→2, saving 2 CZ → **12 CZ** total.

This requires building a custom circuit rather than relying on the transpiler.

### 5.3 Parameter re-optimization

The prune menu uses frozen 7-term parameters. Re-optimizing the surviving parameters (POWELL noiseless or SPSA noisy) can recover accuracy:
- 6-term frozen: |ΔE|=5.0e-3
- 6-term re-optimized: potentially ~1e-3 (see `marrakesh_gatepruned7_reoptimized_prune_menu_20260323.json`)

---

## 6. Physics Constants

### 6.1 Model parameters

```
L = 2          # lattice sites
t = 1.0        # hopping
U = 4.0        # onsite interaction
g_ep = 0.5     # electron-phonon coupling
omega0 = 1.0   # phonon frequency
n_ph_max = 1   # max phonon occupation per site
boundary = open
ordering = blocked
boson_encoding = binary
```

### 6.2 Qubit layout

```
Total qubits: 6
  Fermion: 4 (2L)  — [up_site0, up_site1, dn_site0, dn_site1]
  Phonon:  2 (L×1) — [phonon_site0, phonon_site1]

exyz register order: [phonon1, phonon0, dn1, dn0, up1, up0]
  → exyz position 0 = phonon1 (MSB) = Qiskit qubit 5
  → exyz position 5 = up0 (LSB) = Qiskit qubit 0
```

### 6.3 Hamiltonian

17 Pauli terms total. The 7-term ansatz was selected by ADAPT-VQE from a `paop_lf_full + uccsd_lifted` pool with per_pauli_term parameterization.

---

## 7. Canonical direct HH ADAPT path and historical staged wrapper

For new HH ADAPT work, use the direct ADAPT pipeline. On the direct CLI, omitting `--adapt-continuation-mode` now defaults to `phase3_v1`, which is the canonical current HH path.

```bash
python -m pipelines.hardcoded.adapt_pipeline \
  --L 2 --problem hh --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-max-depth 30 --adapt-eps-grad 1e-5 --adapt-maxiter 800 \
  --initial-state-source adapt_vqe --skip-pdf \
  --output-json artifacts/json/adapt_hh_L2.json
```

True local noisy `phase3_v1` continuation (selection/scoring under oracle-backed local noise; reoptimization stays exact in v1):

```bash
python -m pipelines.hardcoded.adapt_pipeline \
  --L 2 --problem hh --boundary open --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-max-depth 30 --adapt-eps-grad 1e-5 --adapt-maxiter 800 \
  --adapt-continuation-mode phase3_v1 \
  --phase1-score-z-alpha 1.0 \
  --adapt-no-finite-angle-fallback --phase1-no-prune \
  --phase3-oracle-gradient-mode backend_scheduled \
  --phase3-oracle-use-fake-backend \
  --phase3-oracle-backend-name FakeNighthawk \
  --phase3-oracle-shots 2048 \
  --phase3-oracle-repeats 8 \
  --phase3-oracle-mitigation readout \
  --phase3-oracle-local-readout-strategy mthree \
  --initial-state-source adapt_vqe --skip-pdf \
  --output-json artifacts/json/hh_phase3_v1_local_noisy_backend_scheduled.json
```

Historical/compatibility staged wrapper:

```bash
# Historical staged: HF -> warm-start -> ADAPT -> matched-family replay
python -m pipelines.hardcoded.hh_staged_noiseless --L 2
```

The staged wrapper already performs replay and keeps `phase1_v1` as its compatibility default through `--replay-continuation-mode auto`; use explicit staged flags only when reproducing older VQE->ADAPT->VQE behavior. Fresh-stage `hh_staged_noise` remains a noiseless ADAPT stage plus noisy follow-on profiles; it is not the true noisy direct `phase3_v1` continuation path above.

---

## 8. Backend Compilation Reference

### 8.1 Marrakesh (Heron r2)

- Native 2Q gate: CZ
- Topology: heavy-hex, 156 qubits
- Transpiler: `optimization_level=1`, seeds 0-9

### 8.2 Compile costs by artifact (on ibm_marrakesh)

From `investigation_marrakesh_compile_compare_20260323.json`:

| Artifact | 2Q gates | Depth | Size | |ΔE| (noiseless) | Viable? |
|----------|----------|-------|------|-----------------|---------|
| circuit_optimized_7term | 22 | 52 | 107 | 0.338 | **NO** (variationally broken) |
| gate_pruned_7term | 25 | 63-75 | 110-120 | 4.2e-4 | **YES** |
| aggressive_5op | 37-40 | 110-135 | 161-189 | — | Heavy, replay fidelity=0.57 |
| readapt_5op | 37-40 | 110-135 | 161-189 | — | Heavy, replay fidelity=0.57 |

The 5-op family is **not** faithfully executable through the current import/replay seam (replay fidelity ~0.57). Only the 7-term variants are trustworthy executable artifacts.

### 8.3 FakeNighthawk (Eagle)

- Native 2Q gate: CZ
- Topology: heavy-hex, 127 qubits
- Used for: noisy VQE validation (done)
- 7-term gate_pruned: 18 CZ, depth 51 (ADAPT order)

---

## 9. Known Pitfalls (for agents)

1. **Do not use `circuit_optimized_7term`** — it uses disp_first ordering which is variationally broken (|ΔE|=0.34). Always use `gate_pruned_7term`.

2. **Do not reorder per_pauli_term rotations** — operators don't commute. ADAPT ordering [0,1,2,3,4,5,6] is the only validated ordering.

3. **Do not use `AerSimulator(noise_model=...)` for backend simulation** — use `AerSimulator.from_backend()` to preserve the full qubit layout.

4. **Do not scale parameters by coefficients at the circuit level** — the circuit takes raw θ values. Coefficient scaling is internal to the executor.

5. **Do not use the full-circuit audit surface for energy-only QPU checks** — it creates 6 Runtime jobs. Use the energy-only path.

6. **Sector-filtered energy (0.159) is the target**, not full-Hamiltonian GS (-0.151).

7. **5-op artifacts are NOT executable on Marrakesh** — replay fidelity is only 0.57. Don't submit them to a QPU.

8. **SparsePauliOp labels are reversed** relative to exyz notation.

---

## Appendix A: Full CLI Parameter Reference

### A.1 Model parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--L` | int | required | Lattice sites |
| `--problem` | choice | `hubbard` | `hubbard` or `hh` |
| `--t` | float | 1.0 | Hopping |
| `--u` | float | 4.0 | Onsite interaction |
| `--omega0` | float | 1.0 | Phonon frequency |
| `--g-ep` | float | 0.5 | Electron-phonon coupling |
| `--n-ph-max` | int | 1 | Max phonon occupation |
| `--boundary` | choice | `open` | `open` or `periodic` |
| `--ordering` | choice | `blocked` | `blocked` or `interleaved` |
| `--boson-encoding` | choice | `binary` | `binary` or `unary` |

### A.2 ADAPT-VQE parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--adapt-pool` | choice | runtime-resolved | Pool type. Direct CLI resolves `hubbard->uccsd`; HH `phase3_v1` resolves to the narrow core + residual `full_meta`; HH `legacy` keeps the broad `full_meta` compatibility path. |
| `--adapt-max-depth` | int | 20 | Max ADAPT iterations |
| `--adapt-eps-grad` | float | 1e-4 | Gradient convergence |
| `--adapt-eps-energy` | float | 1e-8 | Energy convergence guard for Hubbard / HH legacy; telemetry-only in HH `phase1_v1`, `phase2_v1`, and `phase3_v1` |
| `--adapt-inner-optimizer` | choice | `SPSA` | `COBYLA` or `SPSA` |
| `--adapt-reopt-policy` | choice | `append_only` | `append_only`, `full`, `windowed` |
| `--adapt-continuation-mode` | choice | `phase3_v1` (direct CLI) | Direct ADAPT default is `phase3_v1`; `legacy`, `phase1_v1`, and `phase2_v1` are historical/compatibility modes. Staged wrappers keep `phase1_v1` as their compatibility default. |
| `--adapt-maxiter` | int | 300 | Inner optimizer maxiter |
| `--adapt-seed` | int | 7 | RNG seed |

### A.3 Noise/hardware parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--noise-mode` | choice | — | `ideal`, `shots`, `aer_noise`, `runtime` |
| `--use-fake-backend` | flag | false | Use offline fake backend |
| `--backend-name` | str | — | Backend name (e.g., `ibm_marrakesh`, `FakeNighthawk`) |
| `--shots` | int | — | Measurement shots per circuit |
| `--oracle-repeats` | int | 1 | Repeat evaluations for averaging |
| `--oracle-aggregate` | choice | `mean` | `mean` or `median` |
| `--mitigation` | choice | — | `readout`, `zne`, `dd` |
| `--local-readout-strategy` | choice | — | `mthree` |

### A.4 VQE/SPSA parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--vqe-method` | choice | `SPSA` | Optimizer method |
| `--vqe-maxiter` | int | 120 | Max iterations |
| `--vqe-reps` | int | 2 | Ansatz repetitions |
| `--vqe-restarts` | int | 1 | Independent restarts |
| `--vqe-spsa-a` | float | 0.2 | SPSA a parameter |
| `--vqe-spsa-c` | float | 0.1 | SPSA c parameter |
| `--vqe-spsa-alpha` | float | 0.602 | SPSA alpha |
| `--vqe-spsa-gamma` | float | 0.101 | SPSA gamma |
| `--vqe-spsa-A` | float | 10.0 | SPSA A parameter |

---

## Appendix B: Historical/Scaling Workflows

These sections document legacy workflows that are **not part of the active QPU pipeline**. They are retained for provenance.

### B.1 L-scaling autoscaling (L=2..10)

For agent-run overnight scaling studies (not current work):

- `t_final(L) = 5L`
- `trotter_steps(L) = 64L`
- `adapt_max_depth(L) = 40L`
- `adapt_maxiter(L) = round(5000·L²/9)`

Reference baseline: L=3 Run A (`drive_from_fix1_warm_start_B_full.json`, 2026-03-02).

### B.2 CFQM propagator benchmarks

For propagator comparison studies (not current work). The benchmark scripts (`cfqm_vs_suzuki_*`) are not present in this checkout. Use `hubbard_pipeline.py` with `--propagator cfqm4` directly if needed.

### B.3 Drive-enabled dynamics

For time-dependent drive studies (not current work):

```bash
python -m pipelines.hardcoded.hubbard_pipeline \
  --L 2 --problem hh --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --enable-drive --drive-A 0.5 --drive-omega 2.0 --drive-tbar 3.0 \
  --drive-pattern staggered \
  --t-final 10.0 --num-times 101 --trotter-steps 128 \
  --initial-state-source vqe --skip-qpe
```

### B.4 Archived runners (not present in this checkout)

- `archive/qiskit_compare/*` — comparison tooling (absent)
- `pipelines/shell/run_scaling_L2_L6.sh` — scaling preset (absent)

### B.5 State source options

| Value | Description |
|-------|-------------|
| `vqe` | From pipeline's own VQE |
| `exact` | From exact GS (sector-filtered) |
| `hf` | From Hartree-Fock reference |
| `adapt_json` | From imported ADAPT JSON |

### B.6 Terminology

- **per_pauli_term**: Each Pauli string in a multi-term generator gets its own rotation parameter. This is the parameterization used for all 7-term circuit work.
- **ADAPT ordering**: The order in which ADAPT-VQE selected the operators [0,1,2,3,4,5,6].
- **disp_first ordering**: [5,6,2,3,4,0,1] — puts displacement terms first for better transpilation, but breaks the variational landscape. **Do not use.**
- **Sector-filtered energy**: Ground state energy restricted to the correct particle number sector (1↑, 1↓). This is the physically meaningful target.
- **gate_pruned vs circuit_optimized**: gate_pruned keeps ADAPT ordering (correct); circuit_optimized uses disp_first (broken).
