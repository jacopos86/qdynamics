<!--
This guide is consumed by AI agents.

Editing contract (keep stable):
- Prefer additive edits and new sections over rewrites.
- When introducing “autoscaling”, express formulas explicitly (closed form), plus a reference-run anchored form.
- Record empirically validated baselines with timestamps + artifact filenames.
-->

# Hubbard Pipeline Run Guide

This file is the executable runbook layer (commands + operational contracts).
Active contract surface: `AGENTS.md` and this run guide.

## Active checkout status (2026-03-09)

- **Active production surface in this checkout**
  - `pipelines/hardcoded/hubbard_pipeline.py` (single-run hardcoded VQE/propagation)
  - `pipelines/hardcoded/adapt_pipeline.py` (ADAPT-VQE)
  - `pipelines/shell/run_drive_accurate.sh` (canonical shorthand `run L` contract)
  - `pipelines/exact_bench/cross_check_suite.py` and the `pipelines/exact_bench/` benchmarking/validation tools
- `archive/` compare/Qiskit runners are not present in this checkout.
- Sections labeled **Historical** / **Historical (not runnable)** are kept for provenance only.
- Parser defaults used by the active scripts are `--boundary open` and `--ordering blocked` unless explicitly overridden.

## HH + drive production prologue (2026-03-02)

### Scope (current reality)

- **Production target:** **Hubbard–Holstein (HH)** with **time-dependent drive enabled**.
- **Pure Hubbard** is a **legacy / dead model** for default planning in this repo; ignore it unless the user explicitly asks for a limiting-case / consistency run.
- The **drive waveform parameters** (A, ω, t̄, ϕ, pattern) are orthogonal to the **state-prep** knobs; this guide’s autoscaling focuses on the knobs that dominate convergence:
  - warm-start (conventional VQE seed stage; intermediate HH ansatz `hh_hva_ptw`),
  - ADAPT-VQE (target curriculum from `MATH/IMPLEMENT_SOON.md`: narrow HH
    pool first; `full_meta` only as controlled residual enrichment),
  - and the time-evolution grid (trotter_steps / t_final / num_times).

### Last known-good HH, drive-enabled L=3 artifacts (UTC)

Scoped to **HH + drive-enabled** with **L=3**, the last successful runs are:

- `2026-03-02T16:32:50Z` — `drive_from_fix1_warm_start_B_full.json`
- `2026-03-02T15:10:11Z` — `drive_from_fix1_warm_start_B_depth15.json`

### Common execution context (both runs)

- `problem=hh`, `L=3`, `t=1.0`, `U=4.0`, `dv=0.0`, `omega0=1.0`, `g_ep=0.5`, `n_ph_max=1`
- `ordering=blocked`, `boundary=open`
- Evolution grid: `trotter_steps=192`, `num_times=201`, `t_final=15.0`
- Conventional branch ansatz: `hh_hva_ptw`
  - Internal “regular VQE branch” stays poor in both: `|ΔE_abs| ≈ 1.7768e-01`

### Run A (full warm-start + deeper ADAPT) — **baseline to scale from**

- Imported ADAPT state from `fix1_warm_start_B_full_state.json`
- ADAPT branch: `pool=paop_lf_std`, `ansatz_depth=42`, `num_parameters=42`
- State quality:
  - `E=0.25144823353`
  - `E_exact_filtered=0.24494070013`
  - `ΔE_abs = |E - E_exact_filtered| = 6.5075e-03`
- Fidelity during drive: ~`0.9957` to `0.9960`
- State-build knobs behind the input state:
  - warm-start: `reps=3`, `restarts=5`, `maxiter=4000`
  - ADAPT rung: `max_depth=120`, `maxiter=5000`, `eps_grad=5e-7`, `eps_energy=1e-9` (wallclock-capped)

### Run B (depth-15 proxy) — **do not treat as production-quality**

- Imported ADAPT state from `fix1_warm_start_B_depth15_state.json`
- ADAPT branch: `pool=paop_lf_std`, `ansatz_depth=15`, `num_parameters=15`
- State quality:
  - `E=0.43085525147`
  - `E_exact_filtered=0.24494070013`
  - `ΔE_abs = 1.8591e-01`
- Fidelity during drive: ~`0.8323` to `0.8333`
- State-build knobs: warm-start `reps=3,restarts=1,maxiter=600`; ADAPT `max_depth=15,maxiter=300,eps_grad=5e-7,eps_energy=1e-9`

### Bottom line for HH+drive state-prep in this repo

- The parameter set that produced the expected convergence behavior is **Run A**, not the depth-15 proxy.
- **Current best observed HH L=3 drive-start convergence is** `ΔE_abs ≈ 6.5e-03`.
- **Default Hard Gate (final conventional VQE):** `ΔE_abs < 1e-4`.
- In this checkout, `run_drive_accurate.sh` enforces `<1e-7` unconditionally.

---

## HH autoscaling preset for L ≤ 10 (warm-start + ADAPT → drive)

This section is designed so an agent can compute defaults deterministically from:
- an **empirical reference run** (currently the L=3 Run A baseline), and
- a target lattice size `L ≤ 10`.

### Staged HH convergence gates

Define workflow-level energy gates for HH warm-start → ADAPT → final VQE:

- `ecut_1` (handoff gate): `ΔE_ws = |E_ws - E_exact_filtered| <= 1e-1` by default.
  - Use after warm-start VQE to decide whether to switch to ADAPT.
- `ecut_2` (final acceptance gate): `ΔE_final = |E_final - E_exact_filtered| <= 1e-4` by default.
  - Apply after final VQE on top of ADAPT.

Notes:
- ADAPT internal stopping remains controlled by existing knobs (`adapt-eps-grad`, `adapt-eps-energy`, `adapt-max-depth`, `adapt-maxiter`) and is separate from `ecut_*`.
- This is a runbook convention. If you script this, enforce `ecut_1/ecut_2` as post-stage checks in the orchestration layer; defaults above apply unless overridden by experiment policy.
- `pipelines/shell/run_drive_accurate.sh` is the active shorthand runner contract in this checkout. It enforces a fixed `1e-7` gate in code with no strict/loose runtime switch.

### Gate semantics by context

| Context | Gate | Basis |
|---|---|---|
| Final conventional VQE (agent policy) | `ΔE_abs < 1e-4` | AGENTS/run-guide default policy for this stage |
| Active shorthand runner (`pipelines/shell/run_drive_accurate.sh`) | `ΔE_abs < 1e-7` | Current CLI behavior (hardcoded in this checkout) |
| HH staged handoff (`ecut_1`) | `ΔE_ws <= 1e-2` | Diagnostic handoff guidance (pre-VQE) |
| HH staged final (`ecut_2`) | `ΔE_final <= 1e-4` | Diagnostic replay-stage target when replay is enabled |
| HH production pass gate (runbook) | `ΔE_abs <= 1e-2` | Practical quality indicator, not default hard stop |

### Policy-vs-code note

- `AGENTS target`: shorthand default hard gate `1e-4`; optional strict mode `1e-7`.
- `Current code behavior` (as of 2026-03-09): `pipelines/shell/run_drive_accurate.sh` hardcodes `ERROR_THRESHOLD=1e-7` and exposes no strict-mode switch.
- Guide contract rule: active command sections follow executable behavior, and this mismatch is intentionally visible here for operators.

### Terminology contract (agent-run commands)
- When the user says **"conventional VQE"**, interpret it as the **non-ADAPT VQE** path.
- In this repo, **"conventional VQE"** maps to hardcoded non-ADAPT VQE flows (for example, the VQE stage in `pipelines/hardcoded/hubbard_pipeline.py` and non-ADAPT replay paths).
- **"ADAPT"** / **"ADAPT-VQE"** refers specifically to `pipelines/hardcoded/adapt_pipeline.py` and ADAPT stages.
- The phrase **"hardcoded pipeline"** in repo history/agent direction should be interpreted as the conventional (**non-ADAPT**) path unless ADAPT is explicitly named.

### Agent stage contract (intermediate -> ADAPT -> switch -> replay)

For agent-run HH workflows, use this stage contract:

1. Warm-start stage: conventional VQE with intermediate HH ansatz `hh_hva_ptw`.
   - `hh_hva_ptw` is the canonical staged default.
   - `hh_hva` remains an explicit override only.
2. Optional seed-refine stage: when `--seed-refine-family` is set, run one
   explicit-family conventional VQE refine stage between warm-start and ADAPT.
   - Supported v1 families: `uccsd_otimes_paop_lf_std`,
     `uccsd_otimes_paop_lf2_std`, `uccsd_otimes_paop_bond_disp_std`.
   - This stage materializes the requested explicit family directly; it does
     **not** use `match_adapt` and does **not** auto-fallback to `full_meta`.
   - If refine fails, abort before ADAPT.
3. ADAPT stage: follow the target pool curriculum from `MATH/IMPLEMENT_SOON.md`:
   start from a narrow HH physics-aligned pool and do **not** open `full_meta`
   at depth 0; treat `full_meta` only as controlled residual enrichment after
   plateau diagnosis.
   - Canonical stage selection mode for new HH agent-directed runs is `phase3_v1` (phase1_v1 and phase2_v1 remain opt-in).
4. ADAPT -> final VQE switch: apply an energy-drop switching criterion (see "ADAPT continuation stop policy (energy-first, mandatory for agent runs)").
5. Optional final VQE replay: when `--run-replay` is enabled, initialize from ADAPT state and replay with the same variational generator family ADAPT used (`--generator-family match_adapt`, fallback `full_meta`), using `vqe_reps=L` by default.

Pool curriculum transition note:
- `AGENTS target`: for this HH pool-curriculum transition, treat
  `MATH/IMPLEMENT_SOON.md` as the target spec for new agent-directed pool
  decisions. Depth-0 `full_meta` is not the intended default.
- `Current code behavior`: current CLI and older workflows still support
  `--adapt-pool full_meta`, and historical reference runs below may use it.
- `Required action: ask user before proceeding` if you plan to start a new
  agent-directed HH ADAPT run at depth 0 with `--adapt-pool full_meta`.

CLI note:
- `--adapt-pool full_meta` remains a supported HH pool token in current code; do
  not treat it as the canonical depth-0 target for new agent work.
- Canonical continuation default for new HH staged runs is `--adapt-continuation-mode phase3_v1`.
  (`phase1_v1` and `phase2_v1` are opt-in legacy/experimental follow-ons.)
- Legacy nearest subset remains `--adapt-pool uccsd_paop_lf_full` (`uccsd_lifted + paop_lf_full`).
- Explicit product families for refine/replay or direct ADAPT materialization are
  `uccsd_otimes_paop_lf_std`, `uccsd_otimes_paop_lf2_std`,
  `uccsd_otimes_paop_bond_disp_std`.
- Logical two-parameter variants
  (`uccsd_otimes_paop_lf_std_seq2p`, `...lf2_std_seq2p`,
  `...bond_disp_std_seq2p`) are additive opt-in surfaces and do not change the
  staged `phase3_v1` default contract.

Opt-in phase-3 follow-ons (keep defaults off unless explicitly requested):
- `--phase3-runtime-split-mode shortlist_pauli_children_v1` is an optional continuation aid for HH staged ADAPT/hardcoded paths: shortlisted macro generators may be probed as single-term children, with parent/child provenance exported in continuation metadata.
- `--phase3-symmetry-mitigation-mode {off,verify_only,postselect_diag_v1,projector_renorm_v1}` is an optional phase-3 continuation hook. On raw ADAPT / hardcoded / replay paths it is a metadata-and-telemetry surface; active counts-based symmetry mitigation is enforced only in the oracle-backed noise runners.
- These follow-ons do **not** change the canonical HH contract above: narrow-core first, no depth-0 `full_meta` for new agent-directed runs, and opt-in matched-family replay via `--generator-family match_adapt` with `full_meta` fallback.

### Symbols

- `L`: target lattice size.
- `L_ref`: reference lattice size (default: `3`).
- `s := L / L_ref`: scale factor.
- `E_exact_filtered(L)`: exact filtered-sector ground energy (assumed available every run).
- `E_best(L)`: best energy reached by state-prep (warm-start+ADAPT).
- `ΔE_abs(L) := |E_best(L) - E_exact_filtered(L)|`.

### Reference run (locked to the known-good L=3 baseline)

Use these as the reference knobs unless you intentionally re-calibrate:

**Physics (HH):**
- `t=1.0`, `U=4.0`, `dv=0.0`, `omega0=1.0`, `g_ep=0.5`, `n_ph_max=1`
- `boundary=open`, `ordering=blocked`

**Time grid (drive run):**
- `t_final_ref = 15.0`
- `num_times_ref = 201`
- `trotter_steps_ref = 192`

**Warm-start (seed stage):**
- `ws_reps_ref = 3`
- `ws_restarts_ref = 5`
- `ws_maxiter_ref = 4000`

**ADAPT (historical Run A stage):**
- `adapt_pool = full_meta`
- This is a historical/current executable baseline, not the target depth-0 HH
  pool curriculum for new agent work.
- If reproducing this baseline is an operator decision rather than a historical
  replay/comparison, ask the user before proceeding.
- `adapt_max_depth_ref = 120`
- `adapt_maxiter_ref = 5000`
- `adapt_eps_grad = 5e-7`
- `adapt_eps_energy = 1e-9`

### Scaling philosophy (what scales vs what stays fixed)

- Scale “how hard you search” with `L`:
  - warm-start reps/restarts/maxiter,
  - ADAPT max_depth/maxiter,
  - final VQE replay depth from imported ADAPT state (`vqe_reps(L) = L`),
  - trotter_steps and total evolution time (if you want longer physical windows for larger L).
- Keep convergence thresholds fixed unless you have a specific reason to change them:
  - `adapt_eps_grad`, `adapt_eps_energy` remain constant.
- Drive waveform parameters are **not** scaled here (treat as experiment design), but the **time grid** is scaled.

### Default scaling formulas (anchored to L_ref)

#### Time grid (drive run)

Define the reference step sizes:
- `dt_ref := t_final_ref / (num_times_ref - 1)`
- `dt_trot_ref := t_final_ref / trotter_steps_ref`

Defaults:
- `t_final(L) := t_final_ref * s`
- `trotter_steps(L) := round_to_multiple(trotter_steps_ref * s, 64)`
- `num_times(L) := 1 + ceil( t_final(L) / dt_ref )`

Helper:
- `round_to_multiple(x, m) := m * round(x / m)`

**Closed form with L_ref=3 baseline:**
- `t_final(L) = 5 L`
- `trotter_steps(L) = 64 L`
- `num_times(L) = 1 + ceil(200 L / 3)`  (reproduces 201 at L=3)

Optional (reference-propagator refinement):
- `exact_steps_multiplier(L) := ceil((L + 1)/2)`

#### Warm-start (seed stage)

- `ws_reps(L) := max(1, round(ws_reps_ref * s))`
- `ws_restarts(L) := max(1, ceil(ws_restarts_ref * s))`
- `ws_maxiter(L) := max(200, round(ws_maxiter_ref * s^2))`

**Closed form with L_ref=3 baseline:**
- `ws_reps(L) = L`
- `ws_restarts(L) = ceil(5 L / 3)`
- `ws_maxiter(L) = round(4000 L^2 / 9)`

#### ADAPT (PAOP+LF stage)

- `adapt_max_depth(L) := max(15, round(adapt_max_depth_ref * s))`
- `adapt_maxiter(L) := max(300, round(adapt_maxiter_ref * s^2))`
- `adapt_eps_grad = 5e-7`, `adapt_eps_energy = 1e-9`

**Closed form with L_ref=3 baseline:**
- `adapt_max_depth(L) = 40 L`
- `adapt_maxiter(L) = round(5000 L^2 / 9)`

#### Final VQE replay (from ADAPT state)

- `vqe_reps(L) := L`
- `initial_state_source := adapt_json`
- `adapt_input_json := <exported ADAPT state json>`

### Convergence gates (absolute error)

Because `E_exact_filtered` is always computed:

- `ΔE_abs(L) := |E_best(L) - E_exact_filtered(L)|`

Gates (tuned to separate Run A “good” from the depth-15 proxy “bad”):
- **Probe gate (initial convergence):** `ΔE_abs ≤ 5e-2`
- **Production gate (good convergence):** `ΔE_abs ≤ 1e-2`
- Optional aspirational target: `ΔE_abs ≤ 5e-3` (do not hard-fail on this unless you have evidence at that L)

### Probe → production ladder (agent-friendly)

Goal: avoid spending full budget if state-prep is obviously failing.

**Probe settings from the full scaled defaults:**
- `ws_restarts_probe := max(1, ceil(ws_restarts(L) / 2))`
- `ws_maxiter_probe := max(200, ceil(ws_maxiter(L) / 4))`
- `adapt_max_depth_probe := max(15, ceil(adapt_max_depth(L) / 3))`
- `adapt_maxiter_probe := max(300, ceil(adapt_maxiter(L) / 5))`

**Workflow:**
1) Run warm-start + ADAPT with probe settings.
2) Compute `ΔE_abs`.
3) If `ΔE_abs > 5e-2`: escalate and rerun probe (see below).
4) If probe passes: run the production settings.
5) Production passes if `ΔE_abs ≤ 1e-2` (or record best achieved if wallclock-capped).

### Escalation ladder (deterministic)

If probe fails (`ΔE_abs > 5e-2`), apply in order:

- **Escalation A (optimizer effort):**
  - `ws_restarts += ceil(L/3)`
  - `ws_maxiter *= 2`
  - `adapt_maxiter *= 2`

- **Escalation B (ansatz/search space):**
  - `adapt_max_depth := ceil(1.25 * adapt_max_depth)`  (cap optional: `≤ 60L` for L≤10)

Stop escalating once probe passes or you hit a wallclock/budget cap.

### Agent-run warm cutoff + reusable state export (no manual Ctrl+C)

For Codex/agent-driven runs (no human keypress control), the L4 HH sequential
workflow is now an archived example. The active handoff contract is persistent
state export plus `hubbard_pipeline.py --initial-state-source adapt_json`.

Active helper: `pipelines/hardcoded/handoff_state_bundle.py`

Historical example script: `archive/handoff/l4_hh_warmstart_uccsd_paop_hva_seq_probe.py` (not runnable in this checkout since `archive/` is absent).

Key flags:
- `--warm-auto-cutoff`
- `--warm-cutoff-min-heartbeats` (default `4`)
- `--warm-cutoff-window-heartbeats` (default `3`)
- `--warm-cutoff-slope-threshold` (default `5e-4` per second)
- `--warm-cutoff-min-elapsed-s` (default `180`)
- `--state-export-dir`, `--state-export-prefix`
- staged noiseless wrapper parity flags: `--warm-stop-energy`,
  `--warm-stop-delta-abs`, `--state-export-dir`, `--state-export-prefix`,
  `--resume-from-warm-checkpoint`, `--handoff-from-warm-checkpoint`

Archived-script behavior:
- Warm stage continuously writes `*_warm_checkpoint_state.json` whenever a new
  best warm energy is observed.
- If the warm best-|DeltaE| slope plateaus above the threshold window, warm
  stops automatically and the best-so-far warm state is used to continue ADAPT.
- On normal stage completion, each stage writes:
  - `*_warm_state.json`
  - `*_A_probe_state.json`, `*_A_medium_state.json`, `*_B_probe_state.json`, `*_B_medium_state.json`
  - checkpoint variants `*_<run_id>_checkpoint_state.json`

Conventional pipeline handoff:
- Example:
  - `--initial-state-source adapt_json --adapt-input-json artifacts/useful/L4/<prefix>_A_probe_state.json`
- For HH staged reruns, keep the warm JSON if it already carries
  `ground_state.exact_energy_filtered`; `adapt_pipeline.py --adapt-ref-json`
  reuses that scalar when the imported metadata matches the rerun settings.
  `hh_vqe_from_adapt_family.py` already reuses exact energy from its input JSON.

### ADAPT continuation stop policy (energy-first, mandatory for agent runs)

For HH ADAPT continuation/handoff decisions, use **energy-error drop** as the
primary stop signal. Do **not** treat raw gradient magnitude as a direct proxy
for energy-error improvement.

Definitions:
- `ΔE_abs(d) := |E_best(d) - E_exact_filtered|` at ADAPT depth `d`
- `drop(d) := ΔE_abs(d-1) - ΔE_abs(d)` (positive means improvement)

Policy:
1) Primary stop criterion (required):
- Stop ADAPT when `drop(d) < drop_floor` for `M` consecutive completed depths,
  after a minimum depth guard `d >= d_min`.
2) Secondary criterion (optional safety):
- A gradient floor (`pre-opt max|g| < g_floor` for `M` depths) may be used as
  an additional guard, but it must not be the only stop signal.
3) Hardcoded ADAPT eps-energy telemetry / guard (`pipelines/hardcoded/adapt_pipeline.py`):
- Default resolved staged-HH stop knobs when the user does **not** pass explicit overrides:
  - `adapt_drop_floor = 5e-4`
  - `adapt_drop_patience = 3`
  - `adapt_drop_min_depth = 12`
  - `adapt_grad_floor = 2e-2`
- Explicit CLI values override these resolved defaults; passing negative/off values disables the corresponding staged guard explicitly.
- In HH `phase1_v1` / `phase2_v1` / `phase3_v1`, `eps_energy` telemetry remains active but does **not** terminate ADAPT.
- In HH `phase1_v1` / `phase2_v1` / `phase3_v1`, legacy `eps_grad` is no longer a terminating stop path; low-gradient diagnostics feed the drop-first plateau policy instead.
- In Hubbard and HH `legacy`, the `eps_energy` guard remains depth-gated and patience-gated by defaults:
  - `--adapt-eps-energy-min-extra-depth=-1` resolves to `L`
  - `--adapt-eps-energy-patience=-1` resolves to `L`
- Where the guard is active, `eps_energy` stop triggers only when both are true:
  - local ADAPT depth `>= min_extra_depth`
  - `|E(d)-E(d-1)| < eps_energy` for `patience` consecutive depths (counted after the gate opens)
- Payload telemetry still reports cumulative gate depth as:
  `adapt_ref_base_depth + min_extra_depth_effective`.

Recommended L=4 overnight defaults:
- `drop_floor = 5e-4`
- `M = 3`
- `d_min = 12`
- optional `g_floor = 2e-2` (secondary only)

Interpretation note:
- `max|g|` is a selection/landscape signal and has different units/scale than
  `ΔE_abs`; it is not “how much closer to zero energy error” a depth step gets.

### Example computed defaults (from the L_ref=3 baseline)

These are direct evaluations of the closed-form rules above.

| L | t_final | num_times | trotter_steps | ws_reps | ws_restarts | ws_maxiter | adapt_max_depth | adapt_maxiter | final_vqe_reps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 10.0 | 135 | 128 | 2 | 4 | 1778 | 80 | 2222 | 2 |
| 3 | 15.0 | 201 | 192 | 3 | 5 | 4000 | 120 | 5000 | 3 |
| 4 | 20.0 | 268 | 256 | 4 | 7 | 7111 | 160 | 8889 | 4 |
| 5 | 25.0 | 335 | 320 | 5 | 9 | 11111 | 200 | 13889 | 5 |
| 6 | 30.0 | 401 | 384 | 6 | 10 | 16000 | 240 | 20000 | 6 |
| 7 | 35.0 | 468 | 448 | 7 | 12 | 21778 | 280 | 27222 | 7 |
| 8 | 40.0 | 535 | 512 | 8 | 14 | 28444 | 320 | 35556 | 8 |
| 9 | 45.0 | 601 | 576 | 9 | 15 | 36000 | 360 | 45000 | 9 |
| 10 | 50.0 | 668 | 640 | 10 | 17 | 44444 | 400 | 55556 | 10 |

---

## HH-First Workflow (Hubbard as limiting-case validation)

Project convention moving forward:

- Primary production model: **Hubbard-Holstein (HH)**.
- Pure Hubbard is a legacy / explicit-only validation limit and should otherwise be ignored.
- Standard limit check: compare Hubbard vs HH at vanishing coupling/frequency:
  - `g_ep = 0`
  - `omega0 = 0`
- Keep all other run-defining parameters identical (`L, t, U, dv, boundary, ordering, time-grid, trotter settings, VQE settings`) when making this comparison.

### Why this check exists

At `g_ep = 0` and `omega0 = 0`, HH should reduce to the corresponding Hubbard behavior under matched settings.
This is used as a regression/consistency gate, not as a primary physics target.

### Canonical validation recipe (copy/paste)

#### 1) Hubbard reference run

```bash
python pipelines/hardcoded/hubbard_pipeline.py \
  --L 2 \
  --problem hubbard \
  --t 1.0 --u 4.0 --dv 0.0 \
  --boundary open --ordering blocked \
  --vqe-ansatz uccsd --vqe-reps 2 --vqe-restarts 3 --vqe-maxiter 600 --vqe-seed 7 \
  --t-final 10.0 --num-times 101 --suzuki-order 2 --trotter-steps 128 \
  --initial-state-source vqe --skip-qpe \
  --output-json artifacts/json/hc_hubbard_L2_ref.json \
  --output-pdf artifacts/pdf/hc_hubbard_L2_ref.pdf
```

---

## Comprehensive Runtime Reference

Run from the repository root (`Holstein_test/`).

---

## Runtime Scripts

| Script | Purpose |
|--------|---------|
| `pipelines/hardcoded/hubbard_pipeline.py` | Hardcoded Hamiltonian, hardcoded VQE, hardcoded Trotter dynamics, optional QPE |
| `pipelines/hardcoded/adapt_pipeline.py` | Hardcoded ADAPT-VQE (greedy operator selection, COBYLA re-opt) + Trotter dynamics |
| `pipelines/shell/run_drive_accurate.sh` | Shorthand runner for "run L": drive-only, active hard gate `<1e-7` in this checkout, with L-scaled heaviness |

Active historical notes:

- Archive comparison workflows (`archive/qiskit_compare/*`) are not present in this checkout (verified 2026-03-09).
- `pipelines/shell/run_scaling_L2_L6.sh` is not present in this checkout.
- `hh_hva_tw` (HH Pauli-termwise fixed ansatz) is an archaic retained artifact. Away from smallest accidental cases it does not preserve the target fermion sector, so it is omitted from the canonical symbolic HH fixed-family presentation; prefer `hh_hva` or `hh_hva_ptw` when a sector-preserving fixed HH family is intended.

> **Archive note:** Compare/Qiskit workflows are retained under `archive/qiskit_compare/` only for provenance and are not part of this checkout's active runtime lane.
> **Policy-vs-code:** AGENTS default final gate is `ΔE_abs < 1e-4`, while `run_drive_accurate.sh` currently enforces `<1e-7` with no strict-mode toggle (as observed on 2026-03-09).

---

## State Source Behavior

`--initial-state-source` supports:

| Value | Behaviour |
|-------|-----------|
| `vqe` | Dynamics starts from that pipeline's own VQE-optimised state |
| `exact` | Dynamics starts from exact ground state (sector-filtered eigendecomposition) |
| `hf` | Dynamics starts from Hartree-Fock reference state |
| `adapt_json` | `hardcoded/hubbard_pipeline.py` only: dynamics starts from an imported ADAPT statevector JSON (`--adapt-input-json`) |

If you want apples-to-apples hardcoded vs Qiskit from each ansatz, use `--initial-state-source vqe`.
If you want ADAPT GS preparation with hardcoded driven dynamics, use `--initial-state-source adapt_json`.

Hardcoded comprehensive PDFs now use explicit dual-ansatz branch semantics for scalar trajectories:
- `exact_gs_filtered`
- `exact_paop`, `trotter_paop`
- `exact_hva`, `trotter_hva`

When `--initial-state-source adapt_json` is not used, hardcoded runs internal ADAPT (default `--adapt-pool paop_std`) to construct the PAOP branch.

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
| `--boundary` | choice | `open` | Boundary conditions: `periodic` or `open` |
| `--ordering` | choice | `blocked` | Qubit ordering: `blocked` or `interleaved` |

### Hubbard-Holstein (HH) Model Parameters (hardcoded pipeline only)

These flags activate the Hubbard-Holstein model with electron-phonon coupling.
The default problem is **Hubbard** (`--problem hubbard`); HH phonon parameters
are used only when `--problem hh` is set.

> **Scope note:** The compare pipeline and the Qiskit baseline pipeline do not
> support `--problem hh`. Hubbard-Holstein must be run directly via the
> hardcoded pipeline (`hardcoded/hubbard_pipeline.py --problem hh`).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--problem` | choice | `hubbard` | Model selection: `hubbard` (pure Fermi-Hubbard) or `hh` (Hubbard-Holstein) |
| `--omega0` | float | `1.0` | Phonon frequency ω₀ |
| `--g-ep` | float | `0.5` | Electron-phonon coupling strength g |
| `--n-ph-max` | int | `1` | Maximum phonon occupation per site |
| `--boson-encoding` | choice | `binary` | Boson qubit encoding: `binary` or `unary` |

**Qubit layout (HH):**
`[2L fermion qubits | L × qpb phonon qubits]` where `qpb = ceil(log2(n_ph_max + 1))`.

**Sector filtering (HH):**
The VQE sector filter acts only on the 2L fermion qubits; phonon qubits are
left unconstrained. The exact filtered energy uses the same fermion-only
projection.

### Time-Evolution Parameters (all three pipelines)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--t-final` | float | `20.0` | Final evolution time |
| `--num-times` | int | `201` | Number of output time points |
| `--suzuki-order` | int | `2` | Suzuki–Trotter product-formula order |
| `--trotter-steps` | int | `64` | Number of Trotter steps |
| `--fidelity-subspace-energy-tol` | float | `1e-8` | Ground-manifold selection tolerance for trajectory subspace fidelity: include filtered-sector states with `E <= E0 + tol`. |
| `--term-order` | choice | `sorted` | Term ordering for Trotter product. Hardcoded: `native|sorted`. Qiskit: `qiskit|sorted`. |

### CFQM propagators (hardcoded pipeline only)

`pipelines/hardcoded/hubbard_pipeline.py` supports CFQM alongside legacy propagators.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--propagator` | choice | `cfqm4` | `suzuki2`, `piecewise_exact`, `cfqm4`, `cfqm6`. Default is `cfqm4`; use `suzuki2` explicitly for baseline comparison. |
| `--cfqm-stage-exp` | choice | `expm_multiply_sparse` | Stage exponential backend for CFQM: `expm_multiply_sparse`, `dense_expm`, `pauli_suzuki2`. |
| `--cfqm-coeff-drop-abs-tol` | float | `0.0` | Drops stage coefficients with `abs(coeff) < tol` after stage-map accumulation. |
| `--cfqm-normalize` | flag | `false` | If set, renormalizes state after each CFQM macro-step. |

CFQM runtime semantics:
- CFQM uses fixed scheme nodes `c_j`; legacy `--drive-time-sampling midpoint|left|right` does not change CFQM node sampling.
- `--exact-steps-multiplier` remains reference-only (`piecewise_exact` reference refinement) and does not alter CFQM macro-step count.
- True numerical CFQM4/CFQM6 uses `--cfqm-stage-exp expm_multiply_sparse` or `--cfqm-stage-exp dense_expm`.
- If `--cfqm-stage-exp pauli_suzuki2` is chosen, runtime warns that inner Suzuki-2 reduces overall order to second order.
- Hardware/report/transpile contract: circuitized CFQM artifacts are supported only when `--cfqm-stage-exp pauli_suzuki2`. That path is a circuitizable surrogate profile, not true CFQM4/CFQM6 order.
- Numerical stage-exp modes (`expm_multiply_sparse`, `dense_expm`) stay numerical-only and must not be presented as compiled/transpiled/QPU CFQM circuits.
- Pitfall: in this repo, a `compiled/transpiled CFQM4 circuit` can mean the `pauli_suzuki2` surrogate rather than the true numerical CFQM4 implementation.
- Defensive checks are enabled: `dt > 0`, `n_steps >= 1`, validated CFQM scheme invariants, and finite drive coefficients (NaN/inf raises with label/time context).

CFQM guardrails + policies:
- Warning string (sampling semantics): `CFQM ignores midpoint/left/right sampling; uses fixed scheme nodes c_j.`
- Warning string (inner order collapse): `Inner Suzuki-2 makes overall method 2nd order; use expm_multiply_sparse/dense_expm for true CFQM order.`
- Unknown drive labels use a default safety policy: nontrivial coefficients warn once per label and are ignored; tiny coefficients (`abs(coeff) <= 1e-14`) are silently ignored.
- Unknown labels are never inserted into stage maps (deterministic stage assembly over `ordered_labels` only).
- A=0 invariance is preserved via zero-increment insertion guard, with safe-test target `<= 1e-10`.

Backend implementation note:
- `dense_expm`: dense stage matrix + `scipy.linalg.expm`.
- `expm_multiply_sparse`: native sparse stage matrix assembly + `scipy.sparse.linalg.expm_multiply`.
- `pauli_suzuki2`: symmetric inner Suzuki-2 over Pauli terms; circuitizable surrogate profile for report/transpile/hardware surfaces (runtime order-collapse warning above).
- Shared Pauli-string action helpers for termwise exponentials live in `src/quantum/pauli_actions.py` (prevents library code from depending on pipeline modules).

Diagnostics summary:
- With `normalize=false`, observed CFQM norm drift is typically near machine precision (~1e-15 in the added regression diagnostics).
- Nonzero `--cfqm-coeff-drop-abs-tol` can change trajectories (expected).
- A=0 invariance remains intact when the drive provider returns exact zeros.

Budgeting rule:
- Honest CX/transpile/QPU budgets exist only for `--cfqm-stage-exp pauli_suzuki2`.
- If you need true numerical CFQM4/CFQM6 order, stay on `expm_multiply_sparse` or `dense_expm` and do not quote compiled-circuit budgets as if they belong to the numerical backend.
- There are two honest propagator budget surfaces in this repo:
  - **full-trajectory propagator budget**: one compiled/transpiled dynamics circuit to `t_final`
  - **snapshot propagator budget**: max compiled/transpiled dynamics circuit over sampled `t_i` when each timepoint is a separate job
- `pipelines/exact_bench/hh_fixed_seed_qpu_prep_sweep.py --budget-mode full_trajectory|snapshot` is the fixed-seed driver for these two propagator surfaces.
- `pipelines/exact_bench/hh_fixed_seed_local_checkpoint_fit.py` is different: it fits each sampled checkpoint independently to the exact driven state and reports the max fitted-snapshot circuit cost. It is a practical low-depth surrogate surface, not an honest Suzuki/CFQM propagator surface.
- Operationally, the snapshot propagator budget is often closer to the actual QPU job model because each sampled time is a separate circuit submission. But the checkpoint-fit/local-fit surface is still a different algorithmic object and must not be merged into Suzuki/CFQM propagator tables without explicit labeling.
- Important nuance: for propagated Suzuki/CFQM circuits, `--budget-mode snapshot` does **not** guarantee lower max CX/depth than `full_trajectory`. If each sampled `t_i` uses the same macro-step count `S`, the worst propagated snapshot circuit can remain equal to the `t_final` circuit. Treat snapshot mode as a different job-cost interpretation, not as a free gate reduction.

CFQM baseline commands (production-safe, JSON-only):

```bash
# NOTE: set --adapt-pool uccsd to avoid unsupported paop_std in hubbard ADAPT path.

# 1) Baseline suzuki2
python pipelines/hardcoded/hubbard_pipeline.py \
  --L 2 --problem hubbard --trotter-steps 64 --num-times 201 --t-final 10.0 \
  --vqe-reps 2 --vqe-restarts 2 --vqe-maxiter 600 --vqe-method COBYLA --adapt-pool uccsd \
  --enable-drive --drive-A 0.2 --drive-omega 1.0 --drive-tbar 1.0 --drive-phi 0.0 \
  --drive-pattern staggered --drive-time-sampling left --exact-steps-multiplier 2 \
  --propagator suzuki2 --skip-qpe --skip-pdf

# 2) CFQM4
python pipelines/hardcoded/hubbard_pipeline.py \
  --L 2 --problem hubbard --trotter-steps 64 --num-times 201 --t-final 10.0 \
  --vqe-reps 2 --vqe-restarts 2 --vqe-maxiter 600 --vqe-method COBYLA --adapt-pool uccsd \
  --enable-drive --drive-A 0.2 --drive-omega 1.0 --drive-tbar 1.0 --drive-phi 0.0 \
  --drive-pattern staggered --drive-time-sampling left --exact-steps-multiplier 2 \
  --propagator cfqm4 --cfqm-stage-exp expm_multiply_sparse --skip-qpe --skip-pdf

# 3) piecewise_exact
python pipelines/hardcoded/hubbard_pipeline.py \
  --L 2 --problem hubbard --trotter-steps 64 --num-times 201 --t-final 10.0 \
  --vqe-reps 2 --vqe-restarts 2 --vqe-maxiter 600 --vqe-method COBYLA --adapt-pool uccsd \
  --enable-drive --drive-A 0.2 --drive-omega 1.0 --drive-tbar 1.0 --drive-phi 0.0 \
  --drive-pattern staggered --drive-time-sampling left --exact-steps-multiplier 2 \
  --propagator piecewise_exact --skip-qpe --skip-pdf
```

### Quantum-Processor Proxy Benchmark (CFQM vs Suzuki)

Wrapper benchmark (no core propagator changes):

```bash
python pipelines/exact_bench/cfqm_vs_suzuki_qproc_proxy_benchmark.py \
  --problem hubbard --L 2 \
  --methods suzuki2,cfqm4,cfqm6 \
  --steps-grid 64,128,256,512 \
  --reference-steps 2048 \
  --drive-enabled
```

```bash
python pipelines/exact_bench/cfqm_vs_suzuki_qproc_proxy_benchmark.py \
  --problem hubbard --L 2 \
  --methods suzuki2,cfqm4,cfqm6 \
  --steps-grid 64,128,256,512 \
  --reference-steps 2048 \
  --compare-policy cost_match \
  --cost-match-metric cx_proxy_total \
  --drive-enabled
```

### HH staged noisy dynamics benchmark (Phase 2D)

Heavy staged workflow entrypoint:
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py`

Pipeline sequence:
1. HVA warm-start with `hh_hva_ptw` (intermediate HH variant)
2. ADAPT stage:
   - target policy for new agent work: narrow HH physics-aligned pool first;
     `full_meta` only as controlled residual enrichment after plateau diagnosis
     (see `MATH/IMPLEMENT_SOON.md`)
   - strict legacy mode: Pool B union (`UCCSD_lifted + HVA + PAOP_full`)
   - current broad-pool executable mode: `--adapt-pool full_meta`
     (`UCCSD_lifted + HVA + PAOP_full + PAOP_lf_full`)
   - `Required action: ask user before proceeding` before using depth-0
     `full_meta` for a new agent-directed HH staged run
3. switch from ADAPT to final VQE using the energy-drop criterion (see ADAPT continuation stop policy)
4. conventional VQE replay seeded from ADAPT operator/theta sequence with `vqe_reps = L`
5. noisy dynamics benchmark for selected methods (default `cfqm4,suzuki2`)

New interface flags:
- `--noisy-methods` (CSV, default `cfqm4,suzuki2`; allowed: `suzuki2,cfqm4,cfqm6`)
- `--benchmark-active-coeff-tol` (default `1e-12`)
- `--symmetry-mitigation-mode {off,verify_only,postselect_diag_v1,projector_renorm_v1}` (default `off`; oracle-backed, opt-in)

Phase-3 follow-on note:
- This report keeps the same narrow-core HH stage contract described above.
- Active symmetry mitigation here is oracle-backed and opt-in via `--symmetry-mitigation-mode`.
- Runtime macro splitting remains a lower-level staged ADAPT / hardcoded continuation option (`--phase3-runtime-split-mode`) and is **not** a separate default benchmark mode in this report.

New output schema:
- `dynamics_noisy.profiles.<profile>.methods.<method>.modes.<mode>`
- benchmark fields under each successful mode:
  - `benchmark_cost.term_exp_count_total`
  - `benchmark_cost.pauli_rot_count_total`
  - `benchmark_cost.cx_proxy_total`
  - `benchmark_cost.sq_proxy_total`
  - `benchmark_cost.depth_proxy_total`
  - `benchmark_runtime.wall_total_s`
  - `benchmark_runtime.oracle_eval_s_total`
- backward-compatible alias:
  - `dynamics_noisy.profiles.<profile>.modes` mirrors `methods.suzuki2.modes`

Strict Pool B requirement (strict legacy mode only):
- Runtime fails fast if ADAPT pool composition metadata is not exactly the Pool B family set.
- Audit is persisted at `diagnostics.pool_b_audit`.

Propagation policy (Phase 2D report path):
- `warm`, `adapt`, and `final` are optimization-stage checkpoints only.
- Time dynamics (`dynamics_noiseless`, `dynamics_noisy`) are propagated from `psi_final` only.
- Staged HH energy-error plots/tables use the replay exact-sector ground-state energy as the baseline, so `t=0` reflects preparation error.
- Seeded exact-reference trajectory arrays remain diagnostic only; fidelity is measured against exact propagation from that same `psi_final` seed.
- `hardcoded_superset` is retained in JSON for schema compatibility as deactivated metadata:
  - `{"profiles": {}, "disabled": true, "reason": "branch propagation deactivated; final-only dynamics"}`

What it measures:
- Headline metric: final absolute energy error versus a fine `piecewise_exact` reference.
- Cost axis: hardware-oriented proxy budgets (`term_exp_count_total`, `cx_proxy_total`, `sq_proxy_total`).
- `S` in output rows means macro-steps (`trotter_steps`) and is not itself a cost measure.
- Stage rows/labels (`warm`, `adapt`, `final`) in the PDF summarize optimizer checkpoints only and are not separate propagated trajectories.
- Ranking outputs include Pareto front and best-by-budget summary.
- `--compare-policy` controls apples-to-apples matching:
  - `sweep_only` (default): raw sweep rows only.
  - `cost_match`: build equal-cost rows for each metric target.
- Use `--cost-match-metric cx_proxy_total` (default) for primary fairness check; `term_exp_count_total` is available as an alternate metric.
- Optional `--cost-match-tolerance` relaxes near-budget matching when exact ties are absent.

Important benchmark profile caveat:
- CFQM entries use `--cfqm-stage-exp pauli_suzuki2` in this benchmark so stage exponentials are represented as termwise product formulas for processor comparability.
- This profile is for hardware-cost comparability; it is not the high-order dense/sparse CFQM stage-exp profile used for asymptotic order studies.
- Any compiled/transpiled CFQM4 artifact from this benchmark is therefore the surrogate circuitized path, not the true numerical CFQM4 implementation.

Artifacts:
- `artifacts/cfqm_benchmark/cfqm_vs_suzuki_proxy_runs.json`
- `artifacts/cfqm_benchmark/cfqm_vs_suzuki_proxy_runs.csv`
- `artifacts/cfqm_benchmark/cfqm_vs_suzuki_proxy_summary.json`

### CFQM Efficiency Suite (Error-vs-Cost, Apple-to-Apple)

This suite runs both integrator-order and hardware-proxy comparisons and writes a long PDF with slope/pareto/tie tables.

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

HH L2 warm-start benchmark (imported ADAPT statevector):

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

Note: keep strict ADAPT metadata matching enabled by default; use `--no-adapt-strict-match` only when importing legacy warm-start JSONs missing HH metadata keys.

Efficiency-suite ADAPT pool default is `auto`:
- Hubbard scenarios map to `--adapt-pool uccsd`
- HH scenarios map to `--adapt-pool paop_std`

What it reports:
- Error metrics: `final_infidelity`, `max_doublon_abs_err`, `max_energy_abs_err`.
- Cost metrics: `expm_multiply_calls_*`, `pauli_rot_count_total`, `cx_proxy_total`, `depth_proxy_total`, `wall_time_s`.
- Convergence slopes from `log(error)` vs `log(dt)`.
- Pareto fronts by cost axis.

Fairness policy:
- Main apple-to-apple tables keep only exact cost ties (`delta=0`) for:
  - `cx_proxy`
  - `pauli_rot_count`
  - `expm_calls`
- Interpret `exact_sparse` / `exact_dense` rows as numerical CFQM studies. Only `pauli_suzuki2` rows are honest circuitized/transpile/proxy surfaces.
- Wall-time is grouped into near-tie bins and explicitly labeled approximate.
- Nearest-neighbor fallback matches are appendix-only and must not be used as headline comparisons.

Artifacts:
- `artifacts/cfqm_efficiency_benchmark/runs_full.json`
- `artifacts/cfqm_efficiency_benchmark/runs_full.csv`
- `artifacts/cfqm_efficiency_benchmark/summary_by_scenario.json`
- `artifacts/cfqm_efficiency_benchmark/pareto_by_metric.json`
- `artifacts/cfqm_efficiency_benchmark/slope_fits.json`
- `artifacts/cfqm_efficiency_benchmark/equal_cost_exact_ties_<metric>.csv`
- `artifacts/cfqm_efficiency_benchmark/cfqm_efficiency_suite.pdf`

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
| `--drive-time-sampling` | choice | `midpoint` | Time-sampling rule within each Trotter slice: `midpoint`, `left`, or `right`. CFQM ignores this and uses fixed scheme nodes `c_j`. |
| `--exact-steps-multiplier` | int | `1` | Reference-propagator refinement: N_ref = multiplier × trotter_steps. Has no effect when drive is disabled (static reference uses eigendecomposition), and does not apply to CFQM macro-step counts. |

### VQE Parameters

**Single runtimes** (`hardcoded/hubbard_pipeline.py`; `qiskit_compare` baseline entries are archived only):

| Flag | Type | Default (HC) | Default (QK) | Description |
|------|------|-------------|-------------|-------------|
| `--vqe-ansatz` | choice | `uccsd` | N/A | Hardcoded-only ansatz family: `uccsd`, `hva`, `hh_hva`, `hh_hva_tw` (legacy / non-sector-preserving in general), `hh_hva_ptw` |
| `--vqe-reps` | int | `2` | `2` | Number of ansatz repetitions (circuit depth) |
| `--vqe-restarts` | int | `1` | `3` | Number of independent VQE optimisation restarts |
| `--vqe-seed` | int | `7` | `7` | Random seed for VQE parameter initialisation |
| `--vqe-maxiter` | int | `120` | `120` | Maximum optimiser iterations per restart |
| `--vqe-energy-backend` | choice | `one_apply_compiled` | N/A | Hardcoded-only VQE objective backend: `legacy` or `one_apply_compiled` |
| `--vqe-progress-every-s` | float | `60.0` | N/A | Hardcoded-only VQE heartbeat interval in seconds for progress `AI_LOG` events |

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

### SPSA optimizer (noise-tolerant)

Use SPSA when objective evaluations are noisy (shot noise, hardware noise, stochastic estimators).

- Hardcoded VQE (`pipelines/hardcoded/hubbard_pipeline.py`):
  - Enable with `--vqe-method SPSA`
  - SPSA knobs: `--vqe-spsa-a`, `--vqe-spsa-c`, `--vqe-spsa-alpha`, `--vqe-spsa-gamma`, `--vqe-spsa-A`, `--vqe-spsa-avg-last`, `--vqe-spsa-eval-repeats`, `--vqe-spsa-eval-agg`
- ADAPT inner optimizer (`pipelines/hardcoded/adapt_pipeline.py`):
  - Enable with `--adapt-inner-optimizer SPSA`
  - SPSA knobs: `--adapt-spsa-a`, `--adapt-spsa-c`, `--adapt-spsa-alpha`, `--adapt-spsa-gamma`, `--adapt-spsa-A`, `--adapt-spsa-avg-last`, `--adapt-spsa-eval-repeats`, `--adapt-spsa-eval-agg`

Noise-control note:
- `eval_repeats > 1` reduces estimator variance but increases wallclock roughly proportionally.

Typical starting values:
- `a=0.2`, `c=0.1`, `A=10.0`, `alpha=0.602`, `gamma=0.101`, `avg_last=0` (or `20` for end-of-run averaging).

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
| `--initial-state-source` | choice | `vqe` (compare) / `exact` (single) | State for dynamics: `exact`, `vqe`, `hf`; plus `adapt_json` for hardcoded pipeline import mode |
| `--adapt-input-json` | path | `null` | Hardcoded pipeline only. Required when `--initial-state-source adapt_json`. Path to ADAPT JSON containing `initial_state.amplitudes_qn_to_q0`. |
| `--adapt-strict-match` / `--no-adapt-strict-match` | flag pair | strict on | Hardcoded pipeline only. Enforce (or relax) metadata matching between current run physics and ADAPT JSON physics settings. |
| `--adapt-summary-in-pdf` / `--no-adapt-summary-in-pdf` | flag pair | summary on | Hardcoded pipeline only. Include (or skip) ADAPT provenance page in the comprehensive PDF. |
| `--adapt-pool` | choice | `paop_std` | Hardcoded pipeline only. ADAPT pool used for internal PAOP branch construction when no ADAPT JSON is imported. |
| `--adapt-max-depth` | int | `30` | Max ADAPT depth for internal PAOP branch construction. |
| `--adapt-eps-grad` | float | `1e-5` | ADAPT gradient stopping threshold for internal PAOP branch run. |
| `--adapt-eps-energy` | float | `1e-8` | ADAPT energy-improvement stopping threshold for internal PAOP branch run. |
| `--adapt-inner-optimizer` | choice | `SPSA` | Inner optimizer per ADAPT re-optimization step: `COBYLA` or `SPSA`. |
| `--adapt-maxiter` | int | `800` | Inner optimizer maxiter per ADAPT re-optimization step. |
| `--adapt-seed` | int | `7` | RNG seed for internal ADAPT branch run. |
| `--adapt-allow-repeats` / `--adapt-no-repeats` | flag pair | repeats on | Allow/disallow operator repeats in internal ADAPT. |
| `--adapt-finite-angle-fallback` / `--adapt-no-finite-angle-fallback` | flag pair | fallback on | Enable finite-angle continuation when gradients are near threshold. |
| `--adapt-finite-angle` | float | `0.1` | Probe angle for finite-angle fallback. |
| `--adapt-finite-angle-min-improvement` | float | `1e-12` | Minimum energy drop to accept finite-angle fallback selection. |
| `--adapt-disable-hh-seed` | flag | `false` | Disable HH seed preconditioning block for internal ADAPT. |
| `--paop-r` | int | `1` | Cloud radius for PAOP-style pools in internal ADAPT. |
| `--paop-split-paulis` | flag | `false` | Split PAOP generators into single-Pauli operators. |
| `--paop-prune-eps` | float | `0.0` | Prune PAOP terms below absolute threshold. |
| `--paop-normalization` | choice | `none` | PAOP normalization mode: `none`, `fro`, `maxcoeff`. |

Replay provenance notes:
- Exported staged HH payloads stamp `initial_state.handoff_state_kind` when available.
- `prepared_state` means replay `--replay-seed-policy auto` resolves to `residual_only`; `reference_state` means `auto` resolves to `scaffold_plus_zero`.
- If an opt-in runtime split selected child labels that are not present in the resolved replay family pool, keep `continuation.selected_generator_metadata[*].compile_metadata.serialized_terms_exyz`; replay uses that serialized metadata to rebuild those operators.

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
- `--vqe-ansatz hh_hva` -> `HubbardHolsteinLayerwiseAnsatz` (requires `--problem hh`)
- `--vqe-ansatz hh_hva_ptw` -> `HubbardHolsteinPhysicalTermwiseAnsatz` (requires `--problem hh`; sector-preserving fixed-family HH surface)
- `--vqe-ansatz hh_hva_tw` -> `HubbardHolsteinTermwiseAnsatz` (requires `--problem hh`; legacy compatibility surface, non-sector-preserving in general)
- Legacy term-wise classes remain available in `src/quantum/vqe_latex_python_pairs.py`, but runtime defaults route to the layer-wise / sector-preserving families above.
- Hardcoded VQE JSON now includes:
  - `vqe.ansatz`
  - `vqe.parameterization` (currently `"layerwise"`)
  - `vqe.exact_filtered_energy`

---

## ADAPT-VQE Pipeline (`hardcoded/adapt_pipeline.py`)

The ADAPT-VQE pipeline greedily selects operators from a pool, one per iteration.
Default per-depth re-optimization policy is `append_only`: freeze the existing
theta prefix and optimize only the newest appended parameter. Use
`--adapt-reopt-policy full` for legacy full-prefix re-optimization, or
`--adapt-reopt-policy windowed` for a sliding-window compromise that re-optimises
the newest `W` parameters plus the `K` most-significant older ones (by `|θ|`),
with optional periodic full-prefix refits and a final full refit before export.

### Performance implementation note

- ADAPT commutator-gradient evaluation uses an always-on compiled Pauli-action cache in `hardcoded/adapt_pipeline.py`.
- Cache build happens once per ADAPT run for the Hamiltonian and pool operators, then is reused across gradient sweeps.
- Shared compiled helpers are centralized in:
  - `src/quantum/compiled_polynomial.py`
  - `src/quantum/compiled_ansatz.py`
- Per-depth ADAPT gradient scoring reuses one Hamiltonian action:
  - compute `Hpsi = H|psi>` once per depth
  - score pool operators with `g_i = 2 * Im(vdot(Hpsi, A_i psi))`
- ADAPT inner COBYLA objective now uses compiled ansatz state preparation + one-apply compiled energy.
- There is no separate CLI flag for enabling/disabling this compiled cache path.
- Hardcoded conventional VQE (in `hubbard_pipeline.py`) has explicit backend control via `--vqe-energy-backend`; default is `one_apply_compiled`.
- Hardcoded conventional VQE emits lifecycle + heartbeat events when running via `hubbard_pipeline.py`:
  - `hardcoded_vqe_run_start`
  - `hardcoded_vqe_restart_start`
  - `hardcoded_vqe_heartbeat`
  - `hardcoded_vqe_restart_end`
  - `hardcoded_vqe_run_end`
- Output telemetry fields:
  - `adapt_vqe.compiled_pauli_cache.enabled`
  - `adapt_vqe.compiled_pauli_cache.compile_elapsed_s`
  - `adapt_vqe.compiled_pauli_cache.h_terms`
  - `adapt_vqe.compiled_pauli_cache.pool_terms_total`
  - `adapt_vqe.history[*].gradient_eval_elapsed_s`
  - `adapt_vqe.history[*].optimizer_elapsed_s`
- ADAPT timing `AI_LOG` events:
  - `hardcoded_adapt_compile_timing`
  - `hardcoded_adapt_gradient_timing`
  - `hardcoded_adapt_optimizer_timing`
- These fields are additive and backward-compatible for existing JSON consumers.

### ADAPT/VQE compiled micro-benchmark

Non-pytest benchmark script for local wallclock deltas:

```bash
python pipelines/hardcoded/bench_compiled_energy_and_grad.py \
  --L 3 --n-ph-max 1 --repeats 5
```

What it reports:
- legacy vs compiled energy evaluation time
- legacy vs `Hpsi`-reuse gradient scoring time over a PAOP pool
- speedup factors
- parity diagnostics (`abs_delta_energy`, `max_abs_delta_grad`)

### ADAPT-VQE Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--adapt-pool` | choice | `uccsd` | Pool type: `uccsd`, `cse`, `full_hamiltonian`, `hva` (HH only), `full_meta` (HH only), `paop`, `paop_min`, `paop_std`, `paop_full`, `paop_lf`, `paop_lf_std`, `paop_lf2_std`, `paop_lf_full` (HH only) |
| `--adapt-max-depth` | int | `20` | Maximum ADAPT iterations (operators appended) |
| `--adapt-eps-grad` | float | `1e-4` | Gradient convergence threshold |
| `--adapt-eps-energy` | float | `1e-8` | Energy convergence threshold. Hard-stop guard for Hubbard / HH `legacy`; telemetry-only in HH `phase1_v1|phase2_v1|phase3_v1` |
| `--adapt-eps-energy-min-extra-depth` | int | `-1` | Minimum extra depth before eps-energy guard can trigger; `-1 => L`. Telemetry-only in HH `phase1_v1|phase2_v1|phase3_v1` |
| `--adapt-eps-energy-patience` | int | `-1` | Consecutive low-improvement depths required for eps-energy guard; `-1 => L`. Telemetry-only in HH `phase1_v1|phase2_v1|phase3_v1` |
| `--adapt-inner-optimizer` | choice | `SPSA` | Inner optimizer per ADAPT re-optimization step: `COBYLA` or `SPSA`. |
| `--adapt-state-backend` | choice | `compiled` | ADAPT state action backend: `compiled` (production cached path) or `legacy` (lower-memory fallback) |
| `--adapt-reopt-policy` | choice | `append_only` | Per-depth ADAPT re-optimization policy: `append_only` (default; newest theta only), `full` (legacy all-parameter re-opt), or `windowed` (sliding window + top-k carry). |
| `--adapt-window-size` | int | `3` | Window width W for `windowed` policy (newest W parameters always active). |
| `--adapt-window-topk` | int | `0` | Top-K older parameters (by `|θ|`) carried into the active set; `0` = window only. |
| `--adapt-full-refit-every` | int | `0` | Periodic full-prefix refit cadence (every N cumulative depths); `0` = disabled. |
| `--adapt-final-full-refit` | str | `true` | Run a post-loop full-prefix refit before export (windowed only); `true`/`false`. |
| `--adapt-maxiter` | int | `300` | Inner optimizer maxiter per re-optimization |
| `--adapt-seed` | int | `7` | Random seed |
| `--phase3-symmetry-mitigation-mode` | choice | `off` | Phase-3 continuation symmetry hook: `off`, `verify_only`, `postselect_diag_v1`, `projector_renorm_v1`. On ADAPT/hardcoded paths active estimator behavior is enforced only in oracle-backed noise runners. |
| `--phase3-runtime-split-mode` | choice | `off` | HH continuation add-on: `off` or `shortlist_pauli_children_v1`. Shortlist-only macro splitting for staged continuation/replay metadata; not a default pool-expansion policy. |
| `--adapt-allow-repeats` / `--adapt-no-repeats` | flag | `allow` | Allow selecting the same pool operator more than once |
| `--adapt-finite-angle-fallback` / `--adapt-no-finite-angle-fallback` | flag | `enabled` | Scan ±theta probes when gradients are below threshold |
| `--adapt-finite-angle` | float | `0.1` | Probe angle for finite-angle fallback |
| `--adapt-finite-angle-min-improvement` | float | `1e-12` | Minimum energy drop from probe to accept fallback |
| `--adapt-disable-hh-seed` | flag | `false` | Disable HH quadrature seed pre-optimization |
| `--adapt-drop-floor` | float | `auto` | Energy-drop plateau floor (`drop = ΔE_abs(d-1)-ΔE_abs(d)`). Omitted => HH `phase1_v1|phase2_v1|phase3_v1` resolves to `5e-4`; Hubbard / HH `legacy` stay off; pass negative to disable explicitly |
| `--adapt-drop-patience` | int | `auto` | Consecutive low-drop depths needed for plateau stop. Omitted => HH `phase1_v1|phase2_v1|phase3_v1` resolves to `3`; Hubbard / HH `legacy` stay off |
| `--adapt-drop-min-depth` | int | `auto` | Minimum depth before applying drop plateau stop. Omitted => HH `phase1_v1|phase2_v1|phase3_v1` resolves to `12`; Hubbard / HH `legacy` stay off |
| `--adapt-grad-floor` | float | `auto` | Optional secondary gradient floor guard for plateau stop. Omitted => HH `phase1_v1|phase2_v1|phase3_v1` resolves to `2e-2`; Hubbard / HH `legacy` disable it; pass negative to disable explicitly |
| `--adapt-ref-json` | path | `None` | Import ADAPT reference state from JSON `initial_state.amplitudes_qn_to_q0`; in HH `phase1_v1`/`phase2_v1`/`phase3_v1`, metadata-compatible warm/ADAPT JSON also reuses `ground_state.exact_energy_filtered` when present |
| `--dense-eigh-max-dim` | int | `8192` | Skip full dense diagonalization when Hilbert dim exceeds threshold (sector exact remains; trajectory skipped) |

### PAOP Pool Parameters (HH only)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--paop-r` | int | `1` | Cloud radius R for `paop_full` / `paop_lf_full` |
| `--paop-split-paulis` | flag | `false` | Split composite generators into single Pauli terms |
| `--paop-prune-eps` | float | `0.0` | Prune Pauli terms below this coefficient |
| `--paop-normalization` | choice | `none` | Generator normalization: `none`, `fro`, `maxcoeff` |

### Pool types by problem

| Problem | Available pools |
|---------|----------------|
| `hubbard` | `uccsd`, `cse`, `full_hamiltonian` |
| `hh` | `hva`, `full_meta`, `full_hamiltonian`, `paop`, `paop_min`, `paop_std`, `paop_full`, `paop_lf`, `paop_lf_std`, `paop_lf2_std`, `paop_lf_full` |

**Pool details:**
- `uccsd` — UCCSD single + double excitation generators (same as VQE pipeline)
- `cse` — Term-wise Hubbard ansatz terms (Hamiltonian-variational style)
- `full_hamiltonian` — One generator per non-identity Hamiltonian Pauli term
- `hva` (HH) — HH layerwise generators + UCCSD lifted to HH register + termwise-augmented (merged, deduplicated)
- `full_meta` (HH) — deduplicated union `uccsd_lifted + hva + paop_full + paop_lf_full`
- `paop_min` — Displacement-only polaron operators (local conditional displacement)
- `paop_std` — Displacement + dressed hopping
- `paop_full` — All polaron operators (displacement + doublon dressing + dressed hopping + extended cloud)
- `paop` — alias for `paop_std`
- `paop_lf_std` — `paop_std` plus LF-leading odd channel `curdrag = J_{ij}(P_i-P_j)`
- `paop_lf` — alias for `paop_lf_std`
- `paop_lf2_std` — `paop_lf_std` plus LF second-order even channel `hop2 = K_{ij}(P_i-P_j)^2` (phonon-identity terms dropped)
- `paop_lf_full` — LF full pool (`paop_lf2_std` + extended cloud + doublon-conditioned phonon translation `D_i p_j` / `D_i x_j`), while legacy `paop_full` remains unchanged

### Sector filtering (ADAPT)

For `--problem hh`, the ADAPT pipeline uses **fermion-only sector filtering**
(via `exact_ground_energy_sector_hh`): phonon qubits are unconstrained. This
matches the VQE pipeline convention.

For `--problem hubbard`, standard full-register sector filtering is used.

---

## Full CLI (defaults)

### Hardcoded pipeline

```bash
python pipelines/hardcoded/hubbard_pipeline.py --help
```

Defaults:

- `--t 1.0 --u 4.0 --dv 0.0`
- `--boundary open --ordering blocked`
- `--t-final 20.0 --num-times 201 --suzuki-order 2 --trotter-steps 64`
- `--fidelity-subspace-energy-tol 1e-8`
- `--term-order sorted` (`native|sorted`)
- `--vqe-ansatz uccsd` (`uccsd|hva|hh_hva|hh_hva_tw|hh_hva_ptw`; `hh_hva_tw` is legacy / non-sector-preserving in general)
- `--vqe-method SPSA` (current hardcoded default; HH runs are enforced to SPSA-only in `hubbard_pipeline.py`)
- `--vqe-reps 2 --vqe-restarts 1 --vqe-seed 7 --vqe-maxiter 120`
- `--qpe-eval-qubits 6 --qpe-shots 1024 --qpe-seed 11`
- `--initial-state-source vqe`
- `--phase3-symmetry-mitigation-mode off`
- `--phase3-runtime-split-mode off`
- Drive: disabled by default. Enable with `--enable-drive`.
- Problem: `hubbard` by default. Use `--problem hh` for Hubbard-Holstein.

### ADAPT-VQE pipeline

```bash
python pipelines/hardcoded/adapt_pipeline.py --help
```

Defaults:

- `--t 1.0 --u 4.0 --dv 0.0`
- `--boundary open --ordering blocked`
- `--problem hubbard` (use `--problem hh` for Hubbard-Holstein)
- `--adapt-pool uccsd` (`uccsd|cse|full_hamiltonian|hva|full_meta|paop|paop_min|paop_std|paop_full|paop_lf|paop_lf_std|paop_lf2_std|paop_lf_full`)
- `--adapt-max-depth 20 --adapt-eps-grad 1e-4 --adapt-eps-energy 1e-8`
- `--adapt-state-backend compiled` (`compiled|legacy`)
- `--adapt-maxiter 300 --adapt-seed 7`
- `--phase3-symmetry-mitigation-mode off`
- `--phase3-runtime-split-mode off`
- `--adapt-allow-repeats --adapt-finite-angle-fallback`
- `--adapt-finite-angle 0.1 --adapt-finite-angle-min-improvement 1e-12`
- `--dense-eigh-max-dim 8192`
- `--t-final 20.0 --num-times 201 --suzuki-order 2 --trotter-steps 64`
- `--initial-state-source adapt_vqe` (`adapt_vqe|exact|hf`)

### Historical (not runnable in this checkout): Qiskit baseline

This checkpoint does not include `archive/` (checked 2026-03-09), so these command examples are historical references only and are not executable here.

- `python archive/qiskit_compare/qiskit_baseline.py --help`

Historical behavior notes:
- The historical baseline used `FermiHubbardModel`.
- HH was not supported in that baseline (`--problem hh` would error).

Defaults:

- `--t 1.0 --u 4.0 --dv 0.0`
- `--boundary open --ordering blocked`
- `--t-final 20.0 --num-times 201 --suzuki-order 2 --trotter-steps 64`
- `--fidelity-subspace-energy-tol 1e-8`
- `--term-order sorted` (`qiskit|sorted`)
- `--vqe-reps 2 --vqe-restarts 3 --vqe-seed 7 --vqe-maxiter 120`
- `--qpe-eval-qubits 6 --qpe-shots 1024 --qpe-seed 11`
- `--initial-state-source vqe`
- Drive: disabled by default. Enable with `--enable-drive`.

### Archived compare runner

This historical compare runner is archived and not runnable in this checkout (no
`archive/` directory present, 2026-03-09).

- `python archive/qiskit_compare/compare_hc_vs_qk.py --help`

Historical HH scope note:
- The historical compare path coordinated the hardcoded and Qiskit baselines.
- It used the historical HH guard that HH is not supported in the Qiskit baseline.

Defaults:

- `--l-values 2,3,4,5`
- `--run-pipelines` (use `--no-run-pipelines` to reuse existing JSONs)
- `--t 1.0 --u 4.0 --dv 0.0`
- `--boundary open --ordering blocked`
- `--t-final 20.0 --num-times 201 --suzuki-order 2 --trotter-steps 64`
- `--fidelity-subspace-energy-tol 1e-8`
- `--hardcoded-vqe-ansatzes uccsd` (set `uccsd,hva` for 3-way hardcoded-vs-qiskit runs)
- `--hardcoded-vqe-reps 2 --hardcoded-vqe-restarts 3 --hardcoded-vqe-seed 7 --hardcoded-vqe-maxiter 600`
- `--qiskit-vqe-reps 2 --qiskit-vqe-restarts 3 --qiskit-vqe-maxiter 600 --qiskit-vqe-seed 7`
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
bash pipelines/shell/run_drive_accurate.sh --L 4
```

### 0b) Historical L=2,3 shorthand profile (archived, not runnable)

`archive/qiskit_compare/run_qiskit_L2_L3.sh` is not present in this checkout.

Use `run_drive_accurate.sh` directly for active shorthand runs:

```bash
bash pipelines/shell/run_drive_accurate.sh --L 2
bash pipelines/shell/run_drive_accurate.sh --L 3
```

Primary per-L presets used by the shorthand runner:

| L | trotter_steps | exact_steps_multiplier | num_times | vqe_reps | vqe_restarts | vqe_method | vqe_maxiter |
|---|---:|---:|---:|---:|---:|---|---:|
| 2 | 128 | 2 | 201 | 2 | 2 | SPSA | 1200 |
| 3 | 192 | 2 | 201 | 2 | 3 | SPSA | 2400 |
| 4 | 256 | 3 | 241 | 4 | 4 | SPSA | 6000 |
| 5 | 384 | 3 | 301 | 4 | 5 | SPSA | 8000 |
| 6 | 512 | 4 | 361 | 5 | 6 | SPSA | 10000 |

Fallback behavior:
- `fallback_A`: increase optimizer effort (`restarts + 2`, `maxiter * 2`, method `L-BFGS-B` for `L >= 4`).
- `fallback_B`: additionally increase ansatz/dynamics effort (`reps + 1`, `trotter_steps * 1.5`, `exact_steps_multiplier + 1`).

Optional flags:

```bash
bash pipelines/shell/run_drive_accurate.sh --L 5 --with-pdf
```

```bash
bash pipelines/shell/run_drive_accurate.sh --L 6 --budget-hours 12 --artifacts-dir artifacts
```

HH mode (auto-defaults to `--vqe-ansatz hh_hva_ptw` when `--problem hh`):

```bash
bash pipelines/shell/run_drive_accurate.sh --L 2 \
  --problem hh --omega0 1.0 --g-ep 0.5 --n-ph-max 1 --boson-encoding unary
```

### 1) Run full compare for L=2,3,4 with locked heavy settings

This workflow is historical and not runnable in this checkout because `archive/qiskit_compare` comparison tooling is not present.

### 2) Rebuild comparison PDFs/summary from existing JSON

This workflow is historical and not runnable in this checkout because `archive/qiskit_compare` comparison tooling is not present.

### 3) Run hardcoded pipeline only

```bash
python pipelines/hardcoded/hubbard_pipeline.py \
  --L 3 --initial-state-source vqe \
  --output-json artifacts/json/hc_hubbard_L3_static_t1.0_U4.0_S64.json \
  --output-pdf artifacts/pdf/hc_hubbard_L3_static_t1.0_U4.0_S64.pdf
```

### 4) Run archived Qiskit baseline only

This workflow is historical and not runnable in this checkout (`archive/` is absent).

### 5) Run with time-dependent drive enabled

```bash
python pipelines/hardcoded/hubbard_pipeline.py \
  --L 2 --initial-state-source vqe \
  --enable-drive --drive-A 0.5 --drive-omega 2.0 --drive-tbar 3.0 \
  --drive-pattern dimer_bias --drive-time-sampling midpoint \
  --t-final 10.0 --num-times 101 --trotter-steps 64 \
  --exact-steps-multiplier 4 \
  --output-json artifacts/json/hc_hubbard_L2_drive_t1.0_U4.0_S64.json \
  --output-pdf artifacts/pdf/hc_hubbard_L2_drive_t1.0_U4.0_S64.pdf
```

### 5b) Run hardcoded pipeline with Hubbard-Holstein model

```bash
python pipelines/hardcoded/hubbard_pipeline.py \
  --L 2 --problem hh --omega0 1.0 --g-ep 0.5 --n-ph-max 1 --boson-encoding binary \
  --vqe-ansatz hh_hva --vqe-reps 2 --vqe-restarts 3 --vqe-maxiter 600 \
  --initial-state-source vqe \
  --output-json artifacts/json/hc_hh_L2_static_t1.0_U4.0_S64.json \
  --output-pdf artifacts/pdf/hc_hh_L2_static_t1.0_U4.0_S64.pdf
```

### 5c) Run ADAPT-VQE pipeline (Hubbard, UCCSD pool)

```bash
python pipelines/hardcoded/adapt_pipeline.py \
  --L 2 --problem hubbard --adapt-pool uccsd \
  --adapt-max-depth 20 --adapt-eps-grad 1e-4 --adapt-maxiter 300 \
  --initial-state-source adapt_vqe --skip-pdf \
  --output-json artifacts/json/adapt_L2_uccsd.json
```

### 5d) Run ADAPT-VQE pipeline (HH, HVA pool)

```bash
python pipelines/hardcoded/adapt_pipeline.py \
  --L 2 --problem hh --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-pool hva --adapt-max-depth 30 --adapt-eps-grad 1e-5 --adapt-maxiter 600 \
  --initial-state-source adapt_vqe \
  --output-json artifacts/json/adapt_L2_hh_hva.json \
  --output-pdf artifacts/pdf/adapt_L2_hh_hva.pdf
```

### 5e) Run ADAPT-VQE pipeline (HH, PAOP pool)

```bash
python pipelines/hardcoded/adapt_pipeline.py \
  --L 2 --problem hh --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-pool paop_std --paop-r 1 --paop-normalization none \
  --adapt-max-depth 30 --adapt-eps-grad 1e-5 --adapt-maxiter 600 \
  --initial-state-source adapt_vqe --skip-pdf \
  --output-json artifacts/json/adapt_L2_hh_paop_std.json
```

### 5f) Run hardcoded pipeline with Hubbard-Holstein + drive + Trotter dynamics

```bash
python pipelines/hardcoded/hubbard_pipeline.py \
  --L 2 --problem hh --omega0 1.0 --g-ep 0.5 --n-ph-max 1 --boson-encoding unary \
  --vqe-ansatz hh_hva --vqe-reps 2 --vqe-restarts 3 --vqe-maxiter 600 \
  --enable-drive --drive-A 0.5 --drive-omega 2.0 --drive-tbar 3.0 \
  --drive-pattern staggered --drive-time-sampling midpoint \
  --t-final 10.0 --num-times 101 --trotter-steps 128 \
  --exact-steps-multiplier 2 --suzuki-order 2 \
  --initial-state-source vqe --skip-qpe \
  --output-json artifacts/json/hc_hh_L2_drive_t1.0_U4.0_S128.json \
  --output-pdf artifacts/pdf/hc_hh_L2_drive_t1.0_U4.0_S128.pdf
```

### 5g) ADAPT ground-state prep -> hardcoded driven dynamics

```bash
# 1) Static ADAPT-VQE ground-state preparation
python pipelines/hardcoded/adapt_pipeline.py \
  --L 2 --problem hh --omega0 1.0 --g-ep 0.5 --n-ph-max 1 --boson-encoding binary \
  --t 1.0 --u 2.0 --dv 0.0 --boundary open --ordering blocked \
  --adapt-pool paop_std --adapt-max-depth 30 --adapt-eps-grad 1e-5 --adapt-maxiter 800 \
  --initial-state-source adapt_vqe --skip-pdf \
  --output-json artifacts/json/adapt_hh_L2_seed.json

# 2) Drive-enabled hardcoded trajectory initialized from imported ADAPT state
python pipelines/hardcoded/hubbard_pipeline.py \
  --L 2 --problem hh --omega0 1.0 --g-ep 0.5 --n-ph-max 1 --boson-encoding binary \
  --t 1.0 --u 2.0 --dv 0.0 --boundary open --ordering blocked \
  --initial-state-source adapt_json \
  --adapt-input-json artifacts/json/adapt_hh_L2_seed.json \
  --enable-drive --drive-A 0.5 --drive-omega 1.0 --drive-tbar 3.0 --drive-pattern staggered \
  --drive-time-sampling midpoint --exact-steps-multiplier 2 \
  --t-final 10.0 --num-times 201 --trotter-steps 64 \
  --vqe-ansatz hh_hva_ptw --vqe-reps 2 --vqe-restarts 3 --vqe-maxiter 800 \
  --skip-qpe \
  --output-json artifacts/json/hc_hh_L2_drive_from_adapt.json \
  --output-pdf artifacts/pdf/hc_hh_L2_drive_from_adapt.pdf
```

> **Note:** The drive operates on the full `nq_total = 2L + L×qpb` Hilbert space,
> including phonon qubits. The reference propagator uses `expm_multiply` with
> piecewise-constant H(t) when drive is enabled, matching the static
> eigendecomposition reference when `A=0`.

### 5h) Fast conventional VQE replay from imported ADAPT state (HH, ADAPT-family matched)

```bash
python pipelines/hardcoded/hh_vqe_from_adapt_family.py \
  --adapt-input-json artifacts/useful/L4/l4_hh_warmstart_A_probe_final.json \
  --generator-family match_adapt --fallback-family full_meta \
  --replay-seed-policy auto \
  --L 4 --boundary open --ordering blocked \
  --boson-encoding binary --n-ph-max 1 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 \
  --reps 4 --restarts 16 --maxiter 12000 --method SPSA --seed 7 \
  --energy-backend one_apply_compiled --progress-every-s 60 \
  --output-json artifacts/json/hc_hh_L4_from_adaptB_family_matched_fastcomp.json
```

Notes:
- The replay runner resolves family from ADAPT metadata (`adapt_vqe.pool_type`, then `settings.adapt_pool`, then legacy checkpoint hints). If unresolved, it uses fallback `full_meta`.
- Replay uses ADAPT-selected generators from `adapt_vqe.operators` as the base block, repeated by `--reps`.
- ADAPT replay input requires both `adapt_vqe.operators` and `adapt_vqe.optimal_point` with equal non-zero length.
- This is the canonical ADAPT-family replay path: keep `--generator-family match_adapt` as the default and treat `full_meta` as fallback compatibility, not as a reason to broaden the default HH run contract.
- Replay continuation modes remain explicit (`legacy`, `phase1_v1`, `phase2_v1`, `phase3_v1`). For canonical new HH runs, replay defaults to `phase3_v1` unless explicitly overridden. The follow-on runtime split behavior does **not** introduce a new replay mode.
- If an opt-in runtime split admitted child labels that are not present in the resolved family pool, replay reconstructs them from `continuation.selected_generator_metadata[*].compile_metadata.serialized_terms_exyz` when that serialized metadata is present.
- `hubbard_pipeline.py --vqe-ansatz hh_hva_*` remains a fixed-ansatz baseline path.

### 5i) One-shot staged HH noiseless wrapper

For the staged HH state-prep chain in one command
(HF -> `hh_hva_ptw` warm-start -> optional explicit-family seed refine ->
staged ADAPT, with replay and dynamics opt-in),
use:

```bash
python pipelines/hardcoded/hh_staged_noiseless.py --L 2

# Optional refine insertion:
python pipelines/hardcoded/hh_staged_noiseless.py --L 2 \
  --seed-refine-family uccsd_otimes_paop_lf_std

# Opt in to replay and noiseless dynamics:
python pipelines/hardcoded/hh_staged_noiseless.py --L 2 \
  --run-replay --run-dynamics
```

Wrapper contract:
- replay is **opt-in** (`--run-replay`); default staged endpoint is the ADAPT handoff,
- noiseless dynamics are **opt-in** (`--run-dynamics`); no static/drive trajectory work is done unless requested,
- drive stays **opt-in** (`--enable-drive`) inside the dynamics branch,
- default stage effort is resolved from the HH scaling formulas in this guide,
- when this guide does not specify separate replay optimizer effort, the wrapper reuses the warm-stage restart/maxiter scaling for replay,
- default replay continuation mode follows ADAPT continuation mode unless explicitly overridden,
- `--seed-refine-family` is disabled by default; when omitted, staged behavior is unchanged,
- supported refine families are `uccsd_otimes_paop_lf_std`, `uccsd_otimes_paop_lf2_std`, and `uccsd_otimes_paop_bond_disp_std`,
- `--seed-refine-reps` defaults to `final_reps(L)`, `--seed-refine-maxiter` defaults to `final_maxiter(L)`, and `--seed-refine-optimizer` defaults to `SPSA`,
- when refine is enabled, the workflow runs warm -> refine -> ADAPT -> replay and exports `<state-export-prefix>_seed_refine_state.json`,
- if the refine stage fails, the workflow aborts before ADAPT rather than silently continuing,
- warm-stage handoff is persistent and resumable: each new best warm point rewrites `<state-export-prefix>_warm_checkpoint_state.json`,
- when `--warm-stop-energy` and/or `--warm-stop-delta-abs` is provided, the wrapper auto-stops warm HH-VQE at the requested cutoff, writes `<state-export-prefix>_warm_cutover_state.json`, and seeds ADAPT from that JSON via the `adapt_ref_json` path,
- when warm HH-VQE finishes below the requested cutoff, the wrapper still promotes the best persisted warm bundle to `<state-export-prefix>_warm_cutover_state.json`, logs that the cutoff was not reached, and seeds ADAPT from that JSON,
- `--resume-from-warm-checkpoint <json>` resumes warm HH-VQE from the saved warm bundle after validating HH settings/exact-energy metadata; if that checkpoint already satisfies the cutoff, the wrapper cuts over directly,
- `--handoff-from-warm-checkpoint <json>` skips warm HH-VQE and seeds staged ADAPT directly from the saved warm bundle after the same metadata validation,
- staged handoff bundles may add a top-level `seed_provenance` block with
  warm ansatz, refine family, family kind, motif families, and refine reps;
  readers must treat a missing block as “no refine stage”,
- replay `match_adapt` / `full_meta` resolution ignores `seed_provenance`; it is provenance only,
- when dynamics are enabled, staged energy-error plots/tables use the terminal prepared-state exact-sector baseline (replay if run, otherwise ADAPT), while fidelity remains against the seeded exact-reference trajectory,
- diagnostics record `ecut_1` / `ecut_2` in the workflow payload instead of stopping mid-run.

Primary artifacts:
- workflow JSON/PDF: `artifacts/json/<tag>.json`, `artifacts/pdf/<tag>.pdf`
- warm checkpoints/log: `<state-export-dir>/<state-export-prefix>_warm_checkpoint_state.json`, `<state-export-dir>/<state-export-prefix>_warm_cutover_state.json`, `artifacts/logs/<tag>.log`
- optional refine state JSON: `<state-export-dir>/<state-export-prefix>_seed_refine_state.json`
- ADAPT handoff JSON: `artifacts/json/<tag>_adapt_handoff.json`
- replay sidecars (only when `--run-replay`): `artifacts/json/<tag>_replay.{json,csv}`, `artifacts/useful/L{L}/<tag>_replay.md`, `artifacts/logs/<tag>_replay.log`

#### Reuse an existing staged HH run for local fake-backend noise

To take an already-good staged HH prepared state and run the local noisy
final-only dynamics extension, use:

```bash
python pipelines/hardcoded/hh_staged_noise.py \
  --fixed-final-state-json artifacts/json/<tag>.json \
  --use-fake-backend --backend-name FakeManilaV2 \
  --noise-modes backend_scheduled \
  --noisy-methods cfqm4 \
  --shots 4096 --oracle-repeats 8 --oracle-aggregate mean \
  --symmetry-mitigation-mode postselect_diag_v1
```

Reuse contract:
- `--fixed-final-state-json` may point either to:
  - a direct handoff-style bundle with top-level `initial_state`, or
  - a top-level `hh_staged_*.json` workflow payload; the runner will auto-follow
    its saved `artifacts.intermediate.adapt_handoff_json` sidecar.
- local fake-backend compiled-circuit modes are `backend_basic`,
  `backend_scheduled`, and `patch_snapshot`;
  `backend_scheduled` is the standard “compile to a fake backend noisy circuit”
  path.
- local fake-backend mitigation now has two distinct families:
  - symmetry filtering:
    `--symmetry-mitigation-mode {off,verify_only,postselect_diag_v1,projector_renorm_v1}`
  - local readout mitigation:
    `--mitigation readout --local-readout-strategy {tensor,correlated}`
- local readout mitigation is counts-side Aer execution (no Runtime / QPU
  tokens). `tensor` uses the frozen/fake snapshot per-qubit readout
  probabilities; `correlated` fits a full assignment matrix from simulated local
  calibration circuits on the same locked fake-backend layout.
- runtime-only items remain:
  `--mitigation zne`, `--mitigation dd`, and runtime twirling flags.
- v1 readout scope is intended for small final prepared-state / audit surfaces;
  correlated local readout is heavier than tensor mode.
- output artifacts follow the staged-noise tag:
  `artifacts/json/<noise_tag>.json`, `artifacts/pdf/<noise_tag>.pdf`, plus fresh
  `*_adapt_handoff.json` / `*_replay.json` sidecars for the imported run.

### 5j) Combined staged HH circuit PDF (`L=2,3`)

For a combined circuit-facing PDF that runs the active staged HH chain for `L=2` and `L=3` and renders the workflow circuits directly, use:

```bash
python pipelines/hardcoded/hh_staged_circuit_report.py
```

Optional overrides:

```bash
python pipelines/hardcoded/hh_staged_circuit_report.py \
  --l-values 2,3 \
  --output-pdf artifacts/pdf/hh_staged_circuit_report_L2_L3.pdf
```

Default artifact:
- `artifacts/pdf/hh_staged_circuit_report_L2_L3.pdf`

Per-`L` section contents:
- parameter manifest,
- stage summary,
- warm HH-HVA circuit,
- ADAPT circuit,
- matched-family replay circuit (only when replay is enabled),
- Suzuki2 dynamics circuit (only when dynamics are enabled),
- CFQM4 dynamics section (only when dynamics are enabled).

Circuit rendering semantics:
- each stage/method includes a representative view that keeps high-level `PauliEvolutionGate` blocks intact,
- each stage/method also includes an expanded view that performs one decomposition pass to expose readable term-level layers when circuitization is supported,
- dynamics pages show a single representative macro-step only, not the full unrolled `trotter_steps` chain,
- the PDF reports the full repeat count and proxy totals (`term_exp_count_total`, `cx_proxy_total`, `sq_proxy_total`) for the complete trajectory when a circuitized path exists,
- numerical-only CFQM (`dense_expm`, `expm_multiply_sparse`) is marked unsupported for circuit artifacts and skips representative/expanded circuit pages plus transpile/proxy summaries to avoid misleading circuit claims.

Current default scope:
- replay disabled by default,
- noiseless dynamics disabled by default,
- drive remains opt-in through the shared staged HH flags inside the dynamics branch.

#### Replay seed policy (`--replay-seed-policy`)

The replay seed policy controls how the initial parameter vector is built for VQE replay. Default is `auto`.

| Policy | Seed layout | When to use |
|--------|-------------|-------------|
| `auto` (default) | Branch on `initial_state.handoff_state_kind`: `prepared_state` → `residual_only`, `reference_state` → `scaffold_plus_zero` | Always use this for new runs. |
| `scaffold_plus_zero` | `[θ*, 0, 0, ...]` — first block = ADAPT theta, rest zero | Input is a reference/HF state; scaffold must be replayed explicitly. |
| `residual_only` | `[0, 0, 0, ...]` — all blocks start at zero | Input is an already-prepared state containing the ADAPT scaffold; replay blocks are residual capacity. |
| `tile_adapt` | `[θ*, θ*, θ*, ...]` — ADAPT theta tiled per rep (legacy) | Explicit legacy mode only. Not recommended for new runs. |

#### Handoff-state provenance (`initial_state.handoff_state_kind`)

Producer outputs now stamp `initial_state.handoff_state_kind`:
- `"prepared_state"` — the statevector already includes the ADAPT/warm-start scaffold.
- `"reference_state"` — the statevector is a bare reference (HF, exact ground, etc.).

For old payloads without this field, the replay runner infers provenance from `initial_state.source` (e.g., `"hf"` → reference, `"A_probe_final"` → prepared). If inference is ambiguous, the runner raises an error; use an explicit `--replay-seed-policy` to proceed.

### 5k) HH seed benchmark surface (opt-in cross-check extension)

Use `pipelines/exact_bench/cross_check_suite.py` for the pre-ADAPT seed
comparison surface. This is additive and opt-in; it does **not** replace the
staged HH workflow and it does **not** run the staged ADAPT follow-up itself.

Single-point HH seed surface:

```bash
python pipelines/exact_bench/cross_check_suite.py \
  --problem hh --L 2 --omega0 1.0 --g-ep 0.8 --n-ph-max 1 \
  --hh-seed-refine-surface
```

Mini preset (`L=2,3`, `g=0.8,1.2`, with `t=1`, `U=4`, `omega0=1`, `n_ph_max=1`):

```bash
python pipelines/exact_bench/cross_check_suite.py \
  --problem hh --hh-seed-benchmark-preset mini4
```

Surface contract:
- default HH cross-check matrix remains `HH-Termwise` (historical retained baseline), `HH-Layerwise`, `ADAPT(full_H)`,
- `--hh-seed-refine-surface` additively appends:
  - `HH-PhysicalTermwise`,
  - `HH-PhysicalTermwise + uccsd_paop_lf_full`,
  - `HH-PhysicalTermwise + uccsd_otimes_paop_lf_std`,
  - `HH-PhysicalTermwise + uccsd_otimes_paop_lf2_std`,
  - `HH-PhysicalTermwise + uccsd_otimes_paop_bond_disp_std`,
  - `HH-Layerwise + uccsd_otimes_paop_lf_std` on the smallest-system comparison only,
- the flag is HH-only and HH cross-check remains SPSA-only,
- sidecars keep proxy formulas sourced from `docs/reports/qiskit_circuit_report.py`,
- primary ranking metric is energy improvement per added `cx_proxy`,
- secondary views are energy improvement per added `depth_proxy` and per added `sq_proxy`,
- staged ADAPT follow-up comparisons still belong in `hh_staged_noiseless.py`
  with fixed `phase3_v1` continuation settings, not in `cross_check_suite.py`.

### 6) Compare pipeline with drive enabled

Historical (not runnable in this checkout): archived compare runner is unavailable (`archive/qiskit_compare/` missing).

### 7) Amplitude comparison PDF (scoreboard + physics response)

Historical (not runnable in this checkout): archived compare runner is unavailable (`archive/qiskit_compare/` missing).

This runs 8 sub-pipeline invocations per L (2 main + 6 amplitude comparison) and generates:
- `pdf/amp_cmp_hubbard_{tag}.pdf` — multi-page PDF with settings, scoreboard tables, drive waveform, response deltas, combined overlay, and residual-focused VQE page
- `json/amp_cmp_hubbard_{tag}_metrics.json` — machine-readable safe-test + VQE delta metrics

### 8) Run the L=2/L=3 regression harness

Historical (not runnable in this checkout): `archive/qiskit_compare/regression_L2_L3.sh` is missing here.

### 9) Manual JSON-vs-JSON consistency check

Historical (not runnable in this checkout): `archive/qiskit_compare/compare_jsons.py` is missing here.

### 10) Run the L=2..6 scaling preset with VQE error gate

```bash
for L in 2 3 4 5 6; do
  bash pipelines/shell/run_drive_accurate.sh --L "${L}" --artifacts-dir artifacts
done
```

This command replaces the missing `run_scaling_L2_L6.sh` preset and uses the
active `run_drive_accurate.sh` ladder for the same L range.

Useful historical overrides from the archived preset are not supported by the
active script in this checkout.

Artifacts follow `run_drive_accurate.sh` defaults.

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
|------|-----------|---------|
| `energy_static_exact` | Static Hamiltonian expectation (exact propagator) | ⟨ψ_exact\|H_static\|ψ_exact⟩ |
| `energy_static_trotter` | Static Hamiltonian expectation (Trotter propagator) | ⟨ψ_trotter\|H_static\|ψ_trotter⟩ |
| `energy_total_exact` | **Total** instantaneous energy (exact propagator) | ⟨ψ_exact\|H_static + H_drive(t₀+t)\|ψ_exact⟩ |
| `energy_total_trotter` | **Total** instantaneous energy (Trotter propagator) | ⟨ψ_trotter\|H_static + H_drive(t₀+t)\|ψ_trotter⟩ |

### Behaviour by drive state

| Drive state | `energy_total_*` |
|-------------|-------------------|
| Disabled (`--enable-drive` absent) | Identical to `energy_static_*` (no overhead) |
| Enabled with `A = 0` (safe-test) | Identical to `energy_static_*` within machine precision |
| Enabled with `A > 0` | Differs from `energy_static_*` by the drive contribution ⟨ψ\|H_drive(t)\|ψ⟩ |

### Physical time convention

The drive Hamiltonian at observation time `t` is evaluated at physical time `drive_t0 + t`, consistent with the propagator convention.

### Settings metadata

The JSON `settings` block includes an `energy_observable_definition` string that documents the energy field semantics:

```
"energy_observable_definition": "energy_static_* measures <psi|H_static|psi>. energy_total_* measures <psi|H_static + H_drive(drive_t0 + t)|psi>. When drive is disabled, energy_total_* == energy_static_*. Drive sampling uses the same drive_t0 convention as propagation."
```

### Compare pipeline handling (archived only)

The historical compare pipeline (`archive/qiskit_compare/compare_hc_vs_qk.py`) handled both energy families while present in prior snapshots; it is not runnable in this checkout because `archive/` is absent.

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
`L{L}_{drive|static}_t{t}_U{u}_S{trotter_steps}` — e.g. `L2_static_t1.0_U4.0_S64`.

Prefixes: **hc** = hardcoded, **qk** = Qiskit, **cmp** = comparison, **adapt** = ADAPT-VQE, **xchk** = cross-check suite, **amp** = amplitude comparison.

### Standard comparison outputs

| File | Description |
|------|-------------|
| `json/hc_hubbard_{tag}.json` | Hardcoded pipeline full output |
| `json/qk_hubbard_{tag}.json` | Qiskit pipeline full output |
| `json/cmp_hubbard_{tag}_metrics.json` | Per-L comparison metrics (subspace fidelity, energy, VQE, QPE deltas) |
| `json/cmp_hubbard_bundle_summary.json` | Summary across all L values |
| `pdf/cmp_hubbard_bundle.pdf` | Multi-page comparison bundle PDF |
| `pdf/cmp_hubbard_{tag}.pdf` | Per-L standalone comparison PDF (with `--with-per-l-pdfs`) |

### Amplitude comparison outputs (with `--with-drive-amplitude-comparison-pdf`)

| File | Description |
|------|-------------|
| `pdf/amp_cmp_hubbard_{tag}.pdf` | Multi-page PDF: command, settings, scoreboard + drive waveform, response deltas, combined HC/QK overlay, VQE residual table, optional safe-test detail/heatmap/spectrum pages |
| `json/amp_cmp_hubbard_{tag}_metrics.json` | Machine-readable: `safe_test`, `delta_vqe_hc_minus_qk_at_A0`, `delta_vqe_hc_minus_qk_at_A1` |
| `json/amp_hc_hubbard_{tag}_{slug}.json` | HC intermediate outputs (slug = `disabled`, `A0`, `A1`) |
| `json/amp_qk_hubbard_{tag}_{slug}.json` | QK intermediate outputs (slug = `disabled`, `A0`, `A1`) |

### VQE visibility

- Per-L comparison PDFs include explicit VQE comparison pages.
- Bundle PDF includes VQE comparison pages and per-L VQE pages (with `--with-per-l-pdfs`).
- Amplitude comparison PDF uses a residual-focused VQE table (`HC-QK`, `HC-exact`, `QK-exact`) for perceptible deltas.

### Metrics JSON schema (per-L comparison)

The `cmp_hubbard_{tag}_metrics.json` file includes `trajectory_deltas` with per-observable HC−QK statistics:
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

## 11) HH noise/hardware validation (new)

Use `pipelines/exact_bench/hh_noise_hardware_validation.py` to validate noisy VQE and noisy Trotter observables with one shared expectation oracle.

Hardware-facing CFQM note:
- Use `--propagator cfqm4` or `--propagator cfqm6` together with `--cfqm-stage-exp pauli_suzuki2` when you need a real circuit/transpile/hardware-faithful path.
- `--cfqm-stage-exp expm_multiply_sparse` and `--cfqm-stage-exp dense_expm` remain numerical-only. The validation/report/transpile paths reject them for circuitized CFQM instead of silently lowering them to Suzuki.
- Do not quote an honest CX/QPU budget for dense/sparse CFQM from this runner; those modes are rejected before circuit execution.

### 11a) Ideal baseline (HH)

```bash
python pipelines/exact_bench/hh_noise_hardware_validation.py \
  --problem hh --ansatz hh_hva_ptw --L 2 \
  --t 1.0 --u 4.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --noise-mode ideal --oracle-repeats 1 --oracle-aggregate mean
```

### 11b) Shots-only noise (HH)

```bash
python pipelines/exact_bench/hh_noise_hardware_validation.py \
  --problem hh --ansatz hh_hva_ptw --L 2 \
  --t 1.0 --u 4.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --noise-mode shots --shots 2048 --oracle-repeats 8 --oracle-aggregate mean --seed 7
```

### 11c) Aer device-noise emulation (HH)

```bash
python pipelines/exact_bench/hh_noise_hardware_validation.py \
  --problem hh --ansatz hh_hva_ptw --L 2 \
  --t 1.0 --u 4.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --noise-mode aer_noise --use-fake-backend --backend-name FakeManilaV2 \
  --shots 4096 --oracle-repeats 8 --oracle-aggregate mean --seed 7
```

### 11d) IBM Runtime hardware mode

```bash
python pipelines/exact_bench/hh_noise_hardware_validation.py \
  --problem hh --ansatz hh_hva_ptw --L 2 \
  --t 1.0 --u 4.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --noise-mode runtime --backend-name ibm_brisbane \
  --shots 4096 --oracle-repeats 8 --oracle-aggregate mean --seed 7
```

### 11e) Output artifacts

- JSON: `artifacts/json/hh_noise_validation_L{L}_{problem}_{ansatz}_{noise_mode}.json`
- PDF: `artifacts/pdf/hh_noise_validation_L{L}_{problem}_{ansatz}_{noise_mode}.pdf`

PDF page 1 contains the mandatory parameter manifest (model, ansatz, drive-enabled=false, core physics parameters, and noise/backend settings).

### 11f) Noise interpretation

- `optimizer stochasticity` is induced by noisy objective calls in VQE (`energy_noisy` with per-call spread).
- `measurement/device noise` appears in trajectory fields as noisy-vs-ideal deltas:
  - `energy_static_trotter_delta_noisy_minus_ideal`
  - `n_up_site0_trotter_delta_noisy_minus_ideal`
  - `n_dn_site0_trotter_delta_noisy_minus_ideal`
  - `doublon_trotter_delta_noisy_minus_ideal`

Active symmetry mitigation (oracle-backed, opt-in):
- `--symmetry-mitigation-mode {off,verify_only,postselect_diag_v1,projector_renorm_v1}` defaults to `off`.
- `verify_only` preserves the compatibility baseline: record/verify sector semantics without changing the estimator path.
- `postselect_diag_v1` and `projector_renorm_v1` are currently limited to diagonal/counts-compatible observable paths inside `noise_oracle_runtime.py`.
- Unsupported observables, unavailable counts-compatible paths, or zero retained target-sector probability fall back explicitly to `verify_only`; this is recorded in diagnostics instead of being treated as silent success.
- In `--noise-mode runtime`, the ideal-reference leg is downgraded to `verify_only` when counts-based active symmetry mitigation is unavailable.
- JSON/backend diagnostics include applied mode, fallback reason, retained-fraction / sector-probability summaries, and estimator form.

### 11g) Hardware-facing reproducibility checklist

- Use fixed seeds (`--seed`, `--vqe-seed`) for reproducible stochastic traces.
- Keep `L,t,U,dv,omega0,g-ep,n-ph-max,ordering,boundary` unchanged across mode comparisons.
- Compare `ideal -> shots -> aer_noise -> runtime` in that order.
- For intentionally weak smoke runs, pass `--smoke-test-intentionally-weak`.
- Default parameter guards enforce AGENTS minimum VQE/Trotter settings unless smoke mode is explicitly requested.
- For HH validation runs, the default ansatz/optimizer pair is `hh_hva_ptw` + `SPSA`.

### 11h) SPSA + compiled backend controls (VQE)

`hh_noise_hardware_validation.py` now accepts SPSA controls and a VQE energy backend selector:

- `--vqe-method ...` includes `SPSA`
- `--vqe-energy-backend {legacy,one_apply_compiled}`
- `--vqe-spsa-a`, `--vqe-spsa-c`, `--vqe-spsa-alpha`, `--vqe-spsa-gamma`, `--vqe-spsa-A`
- `--vqe-spsa-avg-last`, `--vqe-spsa-eval-repeats`, `--vqe-spsa-eval-agg`

Noise-authoritative contract:

- In non-ideal noise modes (`shots`, `aer_noise`, `runtime`), optimization remains noisy-oracle driven.
- The compiled one-apply backend is used only in compatible deterministic (`noise-mode ideal`) objective paths.

Example (shots + SPSA):

```bash
python pipelines/exact_bench/hh_noise_hardware_validation.py \
  --problem hh --ansatz hh_hva_ptw --L 2 \
  --run-adapt --run-vqe --run-trotter --initial-state-source adapt \
  --noise-mode shots --shots 2048 --oracle-repeats 8 --oracle-aggregate mean --seed 7 \
  --vqe-method SPSA --vqe-energy-backend one_apply_compiled \
  --vqe-spsa-a 0.2 --vqe-spsa-c 0.1 --vqe-spsa-alpha 0.602 --vqe-spsa-gamma 0.101 --vqe-spsa-A 10.0 \
  --vqe-spsa-eval-repeats 1 --vqe-spsa-eval-agg mean --vqe-spsa-avg-last 0
```

### 11i) Troubleshooting: OMP/SHM2 startup abort in Aer modes

If `--noise-mode shots` or `--noise-mode aer_noise` fails immediately with output like:

- `OMP: Error #178: Function Can't open SHM2 failed`
- `OMP: System error ...`

this indicates an **environment-level OpenMP shared-memory restriction**, not a physics/script logic bug.

Important mode distinctions:

- `shots` and `aer_noise` are **local/offline Aer paths** and do **not** require IBM Runtime credentials.
- `runtime` is the only mode that requires IBM credentials/network (`QISKIT_IBM_TOKEN`, backend access).

Recommended fix path:

- run the command in a non-restricted shell/runtime with working shared-memory support (`/dev/shm` or equivalent),
- avoid sandbox/container setups that block OpenMP SHM primitives.

## 12) Phase-2 ADAPT noise integration

`pipelines/exact_bench/hh_noise_hardware_validation.py` now supports an optional noisy ADAPT stage.

Symmetry-surface clarification:
- In this noise-validation runner, `--symmetry-mitigation-mode` affects oracle-backed ADAPT / VQE / Trotter evaluations when the observable path is eligible.
- This is different from `--phase3-symmetry-mitigation-mode` on raw staged ADAPT / hardcoded / replay surfaces, which remains a continuation metadata-and-telemetry hook unless the workflow is routed through the oracle runtime.

### 12a) Run ADAPT-only noisy search (HH)

```bash
python pipelines/exact_bench/hh_noise_hardware_validation.py \
  --problem hh --ansatz hh_hva_ptw --L 2 \
  --run-adapt --no-run-vqe --no-run-trotter \
  --adapt-pool hva --adapt-max-depth 12 --adapt-maxiter 200 \
  --adapt-eps-grad 1e-5 --adapt-eps-energy 1e-8 \
  --adapt-gradient-step 0.1 --adapt-min-confidence 0.0 \
  --noise-mode ideal
```

### 12b) Use ADAPT state as Trotter initial state

```bash
python pipelines/exact_bench/hh_noise_hardware_validation.py \
  --problem hh --ansatz hh_hva_ptw --L 2 \
  --run-adapt --no-run-vqe --run-trotter \
  --initial-state-source adapt \
  --adapt-pool hva --adapt-max-depth 12 --adapt-maxiter 200 \
  --noise-mode shots --shots 2048 --oracle-repeats 8 --seed 7
```

### 12c) ADAPT payload fields

JSON now includes an `adapt` block with:

- `pool_type`, `pool_size`, `allow_repeats`
- `ansatz_depth`, `num_parameters`, selected `operators`
- `energy_noisy`, `energy_ideal_reference`, `delta_noisy_minus_ideal`
- stopping diagnostics: `stop_reason`, gradient/confidence data per iteration in `history`

### 12d) Interpretation

- ADAPT operator ranking noise is controlled by `--adapt-gradient-step` and `--adapt-min-confidence`.
- `gradient_confidence = |grad| / grad_std`; low confidence can trigger early stop (`low_gradient_confidence`).
- Keep phase-1 ladder (`ideal -> shots -> aer_noise -> runtime`) for the same physics setup before trusting ADAPT selections on hardware.

### 12e) ADAPT inner optimizer SPSA

Noisy ADAPT now supports an inner optimizer selector:

- `--adapt-inner-optimizer {COBYLA,SPSA}` (default `SPSA`)
- SPSA knobs:
  - `--adapt-spsa-a`, `--adapt-spsa-c`, `--adapt-spsa-alpha`, `--adapt-spsa-gamma`, `--adapt-spsa-A`
  - `--adapt-spsa-avg-last`, `--adapt-spsa-eval-repeats`, `--adapt-spsa-eval-agg`

Example (ADAPT inner SPSA + noisy objective):

```bash
python pipelines/exact_bench/hh_noise_hardware_validation.py \
  --problem hh --ansatz hh_hva_ptw --L 2 \
  --run-adapt --run-vqe --run-trotter --initial-state-source adapt \
  --adapt-pool hva --adapt-max-depth 12 --adapt-maxiter 200 \
  --adapt-inner-optimizer SPSA \
  --adapt-spsa-a 0.2 --adapt-spsa-c 0.1 --adapt-spsa-alpha 0.602 --adapt-spsa-gamma 0.101 --adapt-spsa-A 10.0 \
  --adapt-spsa-eval-repeats 1 --adapt-spsa-eval-agg mean --adapt-spsa-avg-last 0 \
  --noise-mode shots --shots 2048 --oracle-repeats 8 --seed 7
```

### 12f) Aer fallback controls (new)

`hh_noise_hardware_validation.py` now supports reliability flags for constrained environments:

- `--allow-aer-fallback` / `--no-allow-aer-fallback`
- `--omp-shm-workaround` / `--no-omp-shm-workaround`

Default behavior:

- fallback disabled / fail-loud by default; opt in with `--allow-noisy-fallback` or `--allow-aer-fallback`
- OMP/SHM workaround enabled (`--omp-shm-workaround`)

If Aer fails with OMP/SHM errors in `shots` or `aer_noise`, the oracle auto-switches to a sampler-based shot fallback and records diagnostics in JSON:

- `execution_fallback.used`
- `execution_fallback.mode`
- `execution_fallback.reason`
- `backend.details.fallback_used`
- `backend.details.fallback_reason`
- `backend.details.env_workaround_applied`

### 11j) Legacy parity gate (pre-noise HH anchor)

Use this when validating that the new noiseless-estimator path reproduces a legacy pre-noise HH run exactly.

Locked baseline for this repo:
- `artifacts/json/hc_hh_L2_static_t1.0_U2.0_g1.0_nph1.json`

Strict gate:
- `max_abs_delta <= 1e-10` for selected observables.
- time grid must match exactly.

New CLI flags in `hh_noise_hardware_validation.py`:
- `--legacy-reference-json PATH`
- `--legacy-parity-tol FLOAT` (default `1e-10`)
- `--compare-observables CSV` (default `energy_static_trotter,doublon_trotter`)
- `--output-compare-plot PATH`

JSON reporting:
- `legacy_parity.reference_json`
- `legacy_parity.observables`
- `legacy_parity.time_grid_match`
- `legacy_parity.per_observable.<obs>.max_abs_delta`
- `legacy_parity.per_observable.<obs>.mean_abs_delta`
- `legacy_parity.per_observable.<obs>.final_abs_delta`
- `legacy_parity.per_observable.<obs>.passed`
- `legacy_parity.passed_all`

Recommended parity command (full-match anchor):

```bash
python pipelines/exact_bench/hh_noise_hardware_validation.py \
  --problem hh --ansatz hh_hva_tw --L 2 \
  --t 1.0 --u 2.0 --dv 0.0 --omega0 1.0 --g-ep 1.0 --n-ph-max 1 \
  --boundary open --ordering blocked \
  --noise-mode ideal \
  --run-vqe --run-trotter --no-run-adapt --initial-state-source vqe \
  --vqe-reps 3 --vqe-restarts 1 --vqe-maxiter 3000 --vqe-method SPSA \
  --t-final 20.0 --num-times 201 --suzuki-order 2 --trotter-steps 64 \
  --legacy-reference-json artifacts/json/hc_hh_L2_static_t1.0_U2.0_g1.0_nph1.json \
  --legacy-parity-tol 1e-10 \
  --output-json artifacts/json/hh_noise_L2_legacy_parity_ideal.json \
  --output-pdf artifacts/pdf/hh_noise_L2_legacy_parity_ideal.pdf \
  --output-compare-plot artifacts/pdf/hh_noise_L2_legacy_parity_compare.png
```

### 11k) Full repository noise-model guide PDF (code/docs only)

Use this documentation runner when you need an exhaustive map of noise-model
implementation surfaces and contracts (without running trajectory experiments):

```bash
pipelines/shell/build_hh_noise_model_repo_guide.sh
```

Direct runner form:

```bash
python pipelines/exact_bench/hh_noise_model_repo_guide.py \
  --output-pdf docs/HH_noise_model_repo_guide.pdf \
  --output-json artifacts/json/hh_noise_model_repo_guide_index.json
```

Default outputs:
- `docs/HH_noise_model_repo_guide.pdf`
- `artifacts/json/hh_noise_model_repo_guide_index.json`
