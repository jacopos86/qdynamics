This repo should be optimized for AI Agents; Humans do not read this repo code nor interact with it except for PDFS of run results.

## Default working scope

Default focus for coding tasks:
- `src/`
- `pipelines/static_adapt/`
- `pipelines/time_dynamics/`
- `pipelines/scaffold/`
- `pipelines/hardcoded/`
- `pipelines/reporting/`
- `test/`
- `run_guide.md`
- `AGENTS.md`
- `README.md`

Ignore by default unless the user explicitly asks for them:
- `.obsidian/`
- `.vscode-extensions/`
- `.vscode-home/`
- `.vscode-userdata/`
- `archive/`
- `docs/` except `docs/reports/` when PDF/report output is in scope
- `claude_code_adapt_wave/`

Use only when the task explicitly requires them:
- `MATH/`
- `docs/reports/`
- `pipelines/exact_bench/`
- `pipelines/shell/`

If the user narrows scope further (for example, “focus only on X and Y”), obey the user’s narrower scope over this default.

```markdown
<!-- agents.md -->

# Agent Instructions (Repo Rules of Engagement)

This file defines how automated coding agents (and humans) should modify this repository.

The priority is **correctness and consistency of operator conventions**, not “cleverness”.

---

## 1) Non-negotiable conventions

### Runbook authority for operational workflows
- For HH staging, run presets, and the new `ecut_1`/`ecut_2` interpretation, agents must consult `run_guide.md` before editing pipeline invocation defaults, scaling tables, or manual run plans.
- Treat this as the canonical source for execution contracts that are not operator-level invariants (e.g., thresholds, gating policy, and recommended run ladders).
- Canonical doc order for agent decisions: `AGENTS.md` -> `run_guide.md` -> `README.md` -> task-specific `MATH/` notes.
- Root-level supporting docs for agents: `README.md` and task-specific `MATH/` notes.
- Math naming contract for agents:
  - `MATH/Math.md` is the markdown authoring source for manuscript sync.
  - `MATH/Math.tex` is the generated typesetting twin corresponding to `MATH/Math.pdf`.
  - Regenerate/sync the TeX/PDF pair with `python MATH/build_math_from_md.py` or `MATH/build_math.sh`.
  - `MATH/archaic_repo_math.md` is archival only; do not treat it as the default math file unless the user explicitly asks for the archaic repo-oriented notes.
- Agents should ignore `docs/` unless the user explicitly asks for material from that folder or PDF/report output is in scope.
- Agents may use `docs/reports/` when repairing or extending PDF/report output.

### Skill-aware run/report routing
- When the task is to **choose, plan, or execute** an HH run, agents should route through the `hh-experiment` skill **if that skill is available in the current agent environment**.
- When the task is to **interpret, compare, or summarize** HH artifacts — or when a run has just completed — agents should route through the `hh-reporting` skill **if that skill is available in the current agent environment**.
- These skills are **wrappers over this repo's authority docs**, not replacements for them. Authority order remains: `AGENTS.md` -> `run_guide.md` -> `README.md`.
- If a skill contract and repo policy ever diverge, agents must surface:
  - `AGENTS target`
  - `Skill/runtime behavior`
  - `Required action: ask user before proceeding`
- Repo-facing run plans should follow the short objective-first run contract already captured in `run_guide.md §0a`.
- Repo-facing reports should mirror that contract, restate the original objective, and keep interpretation logic/math/physics-first rather than only repo-prose-first. The default report deliverable is a short in-chat retell in the compact three-line format `Objective<...>`, `Why/Intent<...>`, `Suggested Next step/how this fits into broader picture<...>`, with no blank lines and 1-3 sentences max per line.
- After a completed run, the default repo-agent convention is to hand the artifact bundle to the reporting path for a short objective-aware retell. Only generate/update persistent report files when report output is in scope and the user has not narrowed the task to execution-only.

### GPT export reveal rule
- When the task creates a persistent GPT/Atlas/ChatGPT handoff markdown in `prompt-exports/`, agents must automatically reveal the final user-facing `.md` file in Finder after writing it.
- Use `open -R <absolute-path>` on the final persistent export file.
- Do this without asking the user for an extra confirmation.
- Reveal the final GPT-facing export, not a temporary `/tmp` sender file.
- If both a raw intermediate export and a final GPT-facing export exist, reveal the final GPT-facing export unless the user explicitly asks to inspect the raw intermediate too.

### GPT export length rule
- For GPT prompt exports in this repo, the default user-facing handoff should be much longer than the old compact style.
- Treat the first concise draft as incomplete by default.
- Expand the final GPT-facing export to about **2.5x** the length of that first short draft unless the user explicitly asks for brevity.
- For HH exports, spend that added length on `MATH/Math.md`, notation, equations, quantitative comparisons, artifact evidence, commands, caveats, and unresolved tensions rather than repo-agent prose.
- If a broad HH GPT export lands only around a few thousand words, assume it is still too short and expand it further before finalizing.

### Policy-vs-code conflict rule (mandatory)
- If AGENTS policy and current code/CLI behavior diverge, agents must **stop and ask the user before proceeding**.
- In docs, when such a mismatch exists, present:
  - `AGENTS target`
  - `Current code behavior`
  - `Required action: ask user before proceeding`

### Terminology contract (agent-run commands)
- When the user says **"conventional VQE"**, interpret it as the **non-ADAPT VQE** path.
- In this repo, **"conventional VQE"** maps to hardcoded non-ADAPT VQE flows (for example, the VQE stage in `pipelines/hardcoded/hubbard_pipeline.py` and non-ADAPT replay paths).
- **"ADAPT"** / **"ADAPT-VQE"** refers specifically to `pipelines/hardcoded/adapt_pipeline.py` and ADAPT stages.
- The phrase **"hardcoded pipeline"** in repo history/agent direction should be interpreted as the conventional (**non-ADAPT**) path unless ADAPT is explicitly named.

### Pauli symbols
- Use `e/x/y/z` internally (`e` = identity)
- If you need I/X/Y/Z output for reports, convert at the boundaries only.

### Pauli-string qubit ordering
- Pauli word string is ordered:
  - **left-to-right = q_(n-1) ... q_0**
  - **qubit 0 is rightmost character**
- All statevector bit indexing must match this.

### JW mapping source of truth
Do not re-derive JW mapping ad-hoc in new files. Use:
- `fermion_plus_operator(repr_mode="JW", nq, j)`
- `fermion_minus_operator(repr_mode="JW", nq, j)`
from `pauli_polynomial_class.py`

If you need number operators, implement:
- `n_p = (I - Z_p)/2`
in a way consistent with the Pauli-string convention above.

### PDF artifact parameter manifest (mandatory)
Every generated PDF artifact must include a **clear, list-style parameter manifest at the start of the document** (first page or first text-summary page), not just a raw command dump.

Required fields:
- Model family/name (for this repo: `Hubbard` unless/ until additional models are added)
- Ansatz type(s) used
- Whether drive is enabled (`--enable-drive` true/false)
- Core physical parameters: `t`, `U`, `dv`
- Any other run-defining parameters needed to reproduce the physics for that PDF

This rule applies to all PDF outputs (single-pipeline PDFs, compare PDFs, bundle PDFs, amplitude-comparison PDFs, and future report PDFs).

---

## 2) Keep the operator layer clean

### Operator layer responsibilities
The following modules are “operator algebra core”:
- `pauli_letters_module.py` (Symbol Product Map + PauliLetter)
- `qubitization_module.py` (PauliTerm)
- `pauli_polynomial_class.py` (PauliPolynomial + JW ladder operators)

### PauliTerm canonical source (mandatory)
Canonical `PauliTerm` source:
- `src.quantum.qubitization_module.PauliTerm`

Compatibility aliases (same class, not separate definitions):
- `src.quantum.pauli_words.PauliTerm`
- `pydephasing.quantum.pauli_words.PauliTerm`

Rules:
- Core package code **must** import `PauliTerm` from `qubitization_module.py`.
- Compatibility scripts may import `pauli_words.PauliTerm` only when required by existing interfaces; they must not introduce a new `PauliTerm` implementation.
- Base operator files **must remain unchanged**: `pauli_letters_module.py`, `pauli_words.py`, `qubitization_module.py`.
- Repo integration changes **must** be implemented with wrappers/shims around base files.

---

## 3) VQE implementation rules

### No Qiskit in the core VQE path
Qiskit is allowed only for:
- validation scripts/notebooks
- reference data generation/comparison

The production VQE path must be:
- numpy statevector backend
- minimal optimizer dependencies (SciPy optional; provide fallback if absent)

### VQE structure (required)
Implement VQE using the notation:

- Hamiltonian:
  `H = Σ_j h_j P_j`
- Energy:
  `E(θ) = Σ_j h_j ⟨ψ(θ)|P_j|ψ(θ)⟩`
- Ansatz:
  `|ψ(θ)⟩ = U_p(θ_p)…U_1(θ_1)|ψ_ref⟩`

### Ansatz selection
Default ansatz should be compatible with future time evolution:
- prefer “term-wise” or “Hamiltonian-variational” style layers
- each unitary should be representable as exp(-i θ * (PauliPolynomial))

Do not hardcode an ansatz that cannot be decomposed into Pauli exponentials.

### ADAPT gradient cache invariant
- The production ADAPT gradient path in `pipelines/hardcoded/adapt_pipeline.py` must keep compiled Pauli-action caching enabled for repeated operator evaluations.
- Do not replace the cached production gradient path with uncached per-term `apply_pauli_string` loops.
- If refactoring this area, preserve cached-vs-uncached numerical parity and keep regression tests for that parity.

### Phase 3 ADAPT and Pool Requirements
- **Rule of Record:** New ADAPT and time-dynamics implementations must explicitly default to the `phase3_v1` logic defined in `MATH/adaptive_selection_staged_continuation.tex`.
- **Banned Full Meta Pools:** The "full meta" pool is explicitly banned for standard runs to prevent excessive QPU depths and prohibitive memory overheads.
- **Required Reduced Pools:** When running $L=2$ and $L=3$, agents must default to the specific reduced winning pools documented in `MATH/Math.md`.

---

## 4) Time-dynamics readiness (Suzuki–Trotter / QPE)

When implementing primitives, favor ones reusable for time evolution:
- Implement "apply exp(-i θ * PauliTerm)" and "apply exp(-i θ * PauliPolynomial)" as first-class utilities.
- Keep functions that return **term lists** (coeff, pauli_string) available for later grouping/ordering.
- Avoid architectures that require opaque circuit objects.

If adding higher-order Suzuki–Trotter later:
- do it by composition on top of the same primitive exp(PauliTerm) backend.

### Secant Time Dynamics standard
- **Primary Controller:** The `secant` time dynamics path is established as the "winning" standard and default time-dynamics solver/controller.
- **Agent Pathing:** Any real-time propagation tasks should evaluate or implement configurations like `secant_lead100` before attempting exploratory or novel ODE approaches. 
- Detailed mathematical backing is available in `MATH/Math.md`.

## 4a) Time-dependent drive implementation rules

The repo supports a **time-dependent onsite density drive** with a Gaussian-envelope sinusoidal waveform:

```
v(t) = A · sin(ω t + φ) · exp(-(t - t₀)² / (2 t̄²))
```

### Drive architecture
- The drive waveform and spatial patterns are defined in `src/quantum/drives_time_potential.py` (if present) or inline in the pipeline files.
- The compare pipeline forwards drive flags verbatim to both sub-pipelines via `_build_drive_args()` and `_build_drive_args_with_amplitude()`.
- All drive parameters are pass-through CLI flags; the compare pipeline does **not** interpret drive physics, only routes them.

### Drive reference propagator
- When drive is enabled, the **reference (exact) propagator** uses `scipy.sparse.linalg.expm_multiply` with piecewise-constant H(t) at each time step.
- The `--exact-steps-multiplier` flag controls refinement: `N_ref = multiplier × trotter_steps`.
- The `reference_method_name` in JSON output records which method was used (`expm_multiply_sparse_timedep` vs `eigendecomposition`).
- When drive is disabled, the static reference uses exact eigendecomposition — no changes.

### Spatial patterns
Three built-in patterns (`--drive-pattern`):
| Pattern | Weights s_j per site j |
|---------|------------------------|
| `staggered` | `(-1)^j` alternating sign |
| `dimer_bias` | `[+1, -1, +1, -1, ...]` (same as staggered for even L) |
| `custom` | User-supplied JSON array via `--drive-custom-s` |

### Rules for agents modifying drive code
- Do **not** add new drive parameters without also updating: (1) both pipeline `parse_args()`, (2) the compare pipeline's `_build_drive_args()` and `_build_drive_args_with_amplitude()`, (3) `run_guide.md`.
- Drive must be **opt-in** (`--enable-drive`). Default behaviour (no flag) must be bit-for-bit identical to the static case.
- Ground-state ADAPT is **static / undriven by default**; drive belongs to later dynamics or compare workflows unless a run explicitly opts in with drive flags.
- All drive-related CLI args must have the `--drive-` prefix (except `--enable-drive` and `--exact-steps-multiplier`).
- The safe-test (`_safe_test_check`) must remain: A=0 drive must produce trajectories identical to the no-drive case within `_SAFE_TEST_THRESHOLD = 1e-10`.

## 4b) Drive amplitude comparison PDF

The compare pipeline supports `--with-drive-amplitude-comparison-pdf` which:
1. Runs both pipelines 3× per L: drive-disabled, A0-enabled, A1-enabled (6 sub-runs total per L).
2. Generates a multi-page physics-facing PDF per L with scoreboard tables, drive waveform, response deltas, and a combined HC/QK overlay.
3. Writes `json/amp_cmp_hubbard_{tag}_metrics.json` with `safe_test`, `delta_vqe_hc_minus_qk_at_A0`, `delta_vqe_hc_minus_qk_at_A1`.

### Rules for agents modifying amplitude comparison
- All artifacts go to `json/` or `pdf/` subdirectories. Filenames use the tag convention `L{L}_{drive|static}_t{t}_U{u}_S{steps}`.
- Intermediate JSON files use the `amp_hc_hubbard_` / `amp_qk_hubbard_` prefix: `json/amp_hc_hubbard_L2_static_t1.0_U4.0_S32_disabled.json`, `json/amp_qk_hubbard_L2_static_t1.0_U4.0_S32_A0.json`, etc.
- Safe-test scalar metrics must always be reported on the scoreboard table. The full safe-test timeseries page is conditional (fail, near-threshold, or `--report-verbose`).
- VQE delta is defined as `ΔE = VQE_hardcoded − VQE_qiskit` (the sector-filtered energy, not full-Hilbert).
- New amplitude comparison CLI args: `--drive-amplitudes A0,A1`, `--with-drive-amplitude-comparison-pdf`, `--report-verbose`, and `--safe-test-near-threshold-factor`.

## 4c) User shorthand run convention (`run L`)

When the user requests a shorthand run like:
- "run L=4"
- "run a number L"
- "run L 5"

interpret it with the following **default contract**:

1. The run is **drive-enabled, never static**.
2. **Default Hard Gate (final conventional VQE):**
   `abs(vqe.energy - ground_state.exact_energy_filtered) < 1e-4`.
3. Use **L-scaled heaviness** (stronger settings for larger L), not one-size-fits-all settings.
4. Pre-VQE stages (warm-start HVA / ADAPT) are **diagnostic** by default and are not hard-fail gates unless explicitly requested by the user.

Implementation rule:
- Prefer `pipelines/shell/run_drive_accurate.sh --L <L>` when available.
- If this script is unavailable, emulate its semantics manually:
  - drive enabled with scaling profile defaults,
  - per-L parameter table,
  - enforce at least the shorthand contract gate (`< 1e-4`),
  - treat `< 1e-7` as **optional strict mode**, not the default hard stop.

## 4d) Mandatory minimum VQE / Trotter parameters per L

**Agents must never run a pipeline with settings weaker than the table below.**
Under-parameterised runs waste wall-clock time and produce unconverged results
that are useless for diagnostics — you cannot tell whether a failure is a code
bug or just insufficient optimiser effort.

If in doubt, **round up** to the next row.

### Hubbard (pure) — minimum settings

| L | `--trotter-steps` | `--exact-steps-multiplier` | `--num-times` | `--vqe-reps` | `--vqe-restarts` | `--vqe-maxiter` | optimizer | `--t-final` |
|---|---|---|---|---|---|---|---|---|
| 2 | 64 | 2 | 201 | 2 | 2 | 600 | COBYLA | 10.0 |
| 3 | 128 | 2 | 201 | 2 | 3 | 1200 | COBYLA | 15.0 |
| 4 | 256 | 3 | 241 | 3 | 4 | 6000 | SLSQP | 20.0 |
| 5 | 384 | 3 | 301 | 3 | 5 | 8000 | SLSQP | 20.0 |
| 6 | 512 | 4 | 361 | 4 | 6 | 10000 | SLSQP | 20.0 |

### Hubbard-Holstein (HH) — minimum settings

HH requires heavier settings than pure Hubbard at the same L due to
the enlarged Hilbert space (phonon modes).

| L | `--n-ph-max` | `--trotter-steps` | `--vqe-reps` | `--vqe-restarts` | `--vqe-maxiter` | optimizer |
|---|---|---|---|---|---|---|
| 2 | 1 | 64 | 2 | 3 | 800 | SPSA |
| 2 | 2 | 128 | 3 | 4 | 1500 | SPSA |
| 3 | 1 | 192 | 2 | 4 | 2400 | SPSA |

### Rules

1. **Never use L=2 defaults for L≥3.** The Hilbert space grows as $2^{2L}$
   (Hubbard) or $2^{2L} \cdot (n_{ph}+1)^L$ (HH). Parameters that converge
   at L=2 are catastrophically insufficient at L=3+.
2. If the user says "run L=3" without specifying parameters, use this table
   (or `pipelines/shell/run_drive_accurate.sh`) — do not invent lighter settings.
3. For validation / smoke-test runs that intentionally use weak settings,
   add an explicit comment: `# SMOKE TEST — intentionally weak settings`.
4. When writing tests, light settings (e.g., `maxiter=40`) are acceptable
   because tests verify implementation correctness, not convergence quality.
   But pipeline runs and demo artifacts must meet the table above.

## 4e) Cross-check suite (`pipelines/exact_bench/cross_check_suite.py`)

The cross-check suite compares **all available ansätze × VQE modes** against
exact ED for a given L, with Trotter dynamics and multi-page PDF output.

### Usage

```bash
# Pure Hubbard, auto-scaled parameters from §4d:
python pipelines/exact_bench/cross_check_suite.py --L 2 --problem hubbard

# Hubbard-Holstein:
python pipelines/exact_bench/cross_check_suite.py --L 2 --problem hh --omega0 1.0 --g-ep 0.5

# Override auto-scaled params (e.g. for smoke tests):
python pipelines/exact_bench/cross_check_suite.py --L 2 --problem hubbard \
  --vqe-reps 1 --vqe-restarts 1 --vqe-maxiter 40 --trotter-steps 8
```

### Shorthand convention

When the user says "run cross-check L=3" or "cross-check L 4":
1. Run `cross_check_suite.py --L <L> --problem hubbard` (auto-scaled from §4d).
2. If the user says "cross-check HH L=2", add `--problem hh`.
3. Do **not** override `--vqe-maxiter` or `--trotter-steps` below §4d minimums.

### Trial matrix

| Problem | Ansätze |
|---------|---------|
| `hubbard` | HVA-Layerwise, UCCSD-Layerwise, ADAPT(UCCSD), ADAPT(full_H) |
| `hh` | HH-Termwise, HH-Layerwise, ADAPT(full_H) |

### Output

- JSON: `<output-dir>/xchk_L{L}_{problem}_t{t}_U{U}.json`
- PDF: same path with `.pdf` — parameter manifest, scoreboard table, per-ansatz 3-panel trajectory plots (fidelity, energy, occupation), fidelity/energy/doublon overlay pages, command page.

## 4f) Trajectory propagation rules (`hubbard_pipeline.py`)

The live hardcoded trajectory surface keeps:
- `--propagator suzuki2`
- `--propagator piecewise_exact`

### Propagation semantics (must preserve)
- `suzuki2` is the only maintained approximate hardcoded propagator.
- `piecewise_exact` remains the reference-style propagation option on the reported trajectory grid.
- `--exact-steps-multiplier` is reference-only refinement; it must not change the reported Suzuki macro-step count.

### Invariants and guardrails
- Keep A=0 safe-test invariant: drive-enabled run with `A=0` must match no-drive within `<= 1e-10`.
- Keep deterministic ordered-label handling in drive assembly.
- Drive labels not present in `ordered_labels` must not be inserted.
- Shared Pauli action primitives live in `src/quantum/pauli_actions.py`; do not reintroduce `src/quantum` -> `pipelines/*` import dependency.

### Minimal post-edit verification commands

```bash
python pipelines/hardcoded/hubbard_pipeline.py --help | rg -n "propagator|piecewise_exact|suzuki2"
```

## 4g) Codex-run HH warm cutoff + state handoff (no manual keypresses)

For Codex-agent runs, do **not** rely on interactive `Ctrl+C` behavior as part
of the normal workflow. Use exported state bundles as the active handoff
contract, and treat warm-cutoff orchestration scripts as archived examples.

Canonical active contract:
- `pipelines/hardcoded/handoff_state_bundle.py`
- `pipelines/hardcoded/hubbard_pipeline.py --initial-state-source adapt_json`

Archived workflow examples:
- `archive/handoff/l4_hh_warmstart_uccsd_paop_hva_seq_probe.py`

Required conventions for agent runs:
1. If warm-stage runtime must be bounded by convergence trend in an archived
   sequential workflow, enable:
   - `--warm-auto-cutoff`
   - slope/window knobs (`--warm-cutoff-*`)
2. Always set state export paths (`--state-export-dir`, `--state-export-prefix`)
   so warm and ADAPT checkpoints are persisted, then write reusable handoff
   bundles with `pipelines/hardcoded/handoff_state_bundle.py`.
3. Use exported `*_A_probe_state.json` / `*_A_medium_state.json` as
   `adapt_json` handoff into `pipelines/hardcoded/hubbard_pipeline.py` when
   running conventional VQE+trotter trajectories from a saved state.
4. For “UCCSD + PAOP only” handoff, use **A-arm** exports; do not use B-arm
   files (`B_*` includes HVA in pool construction).
5. ADAPT stop/handoff decisions must be **energy-error-drop first**:
   - Primary signal: per-depth `ΔE_abs` improvement (`drop = ΔE_abs(d-1)-ΔE_abs(d)`).
   - Use patience over completed depths (`M` consecutive low-drop depths) with
     a minimum depth guard (`d_min`) before stopping.
   - Gradient floors (`max|g|`) are secondary diagnostics/safety checks only;
     they must not be the sole stop reason in agent-run HH workflows.
   - Do not interpret `max|g|` as “the energy-error drop per depth.”

---

## 5) Style and maintainability

### Clean/simple code
- Prefer pure functions where possible (no hidden global state).
- Keep modules small, with a single responsibility.
- Use explicit types for public function signatures.
- Prefer explicit errors (`log.error(...)` or raising) over silent coercions.

### Built Math-Symbols/Description above Python pairing
When adding new modules :
- include the Built-in math symbolic expression in a string right above the function that implements it
- keep the math and code aligned 1:1

### Regression/validation
Whenever you modify:
- Hubbard Hamiltonian construction
- indexing conventions
- JW mapping / number operator

You must update or re-run reference checks against:
- `hubbard_jw_*.json`

Qiskit baseline scripts may be used to sanity check, but they are not the core test oracle.

---

## 6) What an agent should NOT do
- Do not change Pauli-string ordering conventions.
- Do not introduce Qiskit into core/'hardcoded' algorithm modules.
- Do not add heavy dependencies without a strong reason.
- Do not "optimize" by rewriting algebra rules unless correctness is proven with regression tests.
- Do not add new drive parameters without updating all three pipelines' `parse_args()`, `_build_drive_args()`, `_build_drive_args_with_amplitude()`, and `run_guide.md`.
- Do not break the safe-test invariant (A=0 drive must equal no-drive to machine precision).
- Do not stop a run because you think it is taking up too much run-time. The only acceptable reason to stop/interrupt an already active run/script is for debugging.
- **Do not run a pipeline with parameters below the §4d minimum table.** If the user does not specify parameters, look up the table — never guess or use L=2 defaults for larger L.



--Note -- Take your time coding! Be safe, and do not rush. The user has a lot of time and does not need things quickly.

## Plans

- Make the plan extremely consise. Sacrifice grammar for the sake of concision.
- Near the end of each plan, give me a list of unresolved questions to answer/problems, if any, and the files you will edit.
- At the end of each plan, state all files intended to alter, and functions and classes to be altered. If none, write 'Files to edit: None'.

## 4h) Legacy noiseless-estimator parity rule (HH anchor)

When a user asks to verify that the new noiseless-estimator path is equivalent to the pre-noise HH pipeline:

1. Use the locked L=2 baseline artifact:
- `artifacts/json/hc_hh_L2_static_t1.0_U2.0_g1.0_nph1.json`
2. Run a full-match parity case (do not downscale run knobs for this check).
3. Enforce strict gate: `max_abs_delta <= 1e-10` on selected observables, with exact time-grid match required.
4. Record parity fields in JSON/PDF (`legacy_parity.*`) and emit the comparison plot if requested.

This is a validation exception to “very light” preference: parity verdicts must use full baseline-matched settings.

## 4i) HH noisy `ΔE` contract (mandatory)

For HH noisy / mitigation runs and reports, the default meaning of
`|ΔE|` is:

`|ΔE| = |E_exact - E_noisy(with mitigation)|`

Rules:

1. `E_exact` is the exact filtered ground-state energy for the same physics point.
2. `E_noisy(with mitigation)` is the final noisy energy after all enabled mitigation and suppression steps have been applied.
3. Do **not** use imported-circuit ideal energy as the default `ΔE` reference.
4. If the repo also reports noisy-vs-ideal imported-circuit bias, it must be labeled explicitly as something like:
   - `E_noisy(with mitigation) - E_ideal(imported circuit)`, or
   - `ΔE_to_ideal`
5. When both exact-target and ideal-target gaps are present, exact-target `|ΔE|` is the primary metric for rankings, summaries, tables, reports, and user-facing statements.
