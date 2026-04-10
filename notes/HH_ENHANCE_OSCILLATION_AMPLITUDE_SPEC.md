# Strategy to Increase Oscillation Amplitude in HH Time-Dynamics

## 1. Objective and Physical Context
The current time-dynamics runs (e.g., `secant_lead100_spectra_report.json`) show that under the time-dependent drive:
- The exact occupation pair difference oscillation span is `~0.37`.
- The controller occupation span degrades slightly to `~0.24`.
- Exact Doublon and Energy amplitudes are even smaller (`~0.002` and `~0.004`).

For testing noise robustness and assessing algorithm degradation over time, having a highly squeezed dynamic range makes absolute errors mathematically harsh, leading to small signal-to-noise ratios. By tuning the model into a regime of **larger natural oscillations** (e.g., a total charge transfer $1.0$ instead of $0.3$), small algorithmic deviations from exact reference become visibly clear and practically manageable.

## 2. Tuning Knobs for Hubbard-Holstein Dynamics
To induce violent, wide-ranging charge shifts across the lattice, we must increase the probability of electrons moving between sites. Because we start with a Ground State (half-filled Mott/Peierls insulator), movement is restricted by the effective repulsion $U_{eff}$.

### Knob A: Drive Amplitude ($A$)
**Current:** `A = 0.55`
**Strategy:** Increase `A`. The time-dependent chemical potential `dv(t)` pushes charges across the bond. A stronger electric field will forcibly relocate more charge.
- **Action:** Push to `A = 1.0` or `1.5`.

### Knob B: Drive Frequency ($\omega$)
**Current:** `omega = 2.0`
**Strategy:** Tune closer to Resonance. If the drive frequency matches the exact charge gap $\Delta_{charge} = E_1 - E_0$, transitions will explode into a Rabi-like macroscopic oscillation.
- **Action:** Run a fast Exact Diagonalization (ED) sweep to find the single-particle charge gap of the current $(t=1, U=4, g_{ep}=0.5)$ ground state and set `omega` exactly there (or near it to avoid total chaotic thermalization).

### Knob C: Lowering Hubble Repulsion ($U$)
**Current:** `U = 4.0` (Strongly correlated insulator for $t=1$)
**Strategy:** The ground state of $U=4$ is tightly localized. Electrons do not want to become doublons. If we drop $U$ to `2.0` or `1.0`, the system becomes softer (metallic). The $A=0.55$ drive will suddenly cause massive charge movement because the penalty for moving is reduced.
- **Action:** Test `U = 2.0` (Weakly interacting regime). This will drastically increase doublon presence and site density fluctuations.

### Knob D: Increasing Phonon Coupling ($g_{ep}$)
**Current:** `g_ep = 0.5`
**Strategy:** Strong electron-phonon coupling triggers charge-density wave (CDW) pairing. If we push to $g_{ep} = 1.0$ to $1.5$ (Polaron regime), electrons drag massive phonon clouds, and a strong drive causes massive CDW disruption.
- **Action:** Test a Polaron quench. But beware: Hilbert space required for phonons will shoot up (needs `n_ph_max >= 2`).

---

## 3. Immediate Implementation Steps for the Agent
Before writing the next agent execution script, the agent should configure a `drive_amplitude_scan` using the exact reference solvers ONLY.

**Step 1: Exact Parameter Sweep**
Use Python to iterate Exact reference evolutions across these configurations to find which produces an `oscillation_span_exact > 0.8`:
1. `(U=4.0, A=1.0, omega=2.0)` -> Brute force amplitude.
2. `(U=2.0, A=0.55, omega=2.0)` -> Softened Mott gap.
3. `(U=4.0, A=0.55, omega=3.5)` -> Attempting to hit resonance (depends on the exact charge gap).

**Step 2: Isolate the Best Config**
Select the setting that maximizes `oscillation_span_exact` for the `pair_difference(0,1)` observable without causing numerical explosions in the RK4/Magnus exact solvers.

**Step 3: Update the Controller Baseline**
Once the new physical parameters (`t, u, g_ep, omega0`) are chosen, re-run the `phase3_v1` ADAPT checkpoint to form the starting state, and generate the new Secant run. The absolute errors of the Secant controller will now fall against a massive $> 0.8$ amplitude swing, giving a beautiful visual proof of tracking ability.
# Analytic Noise Injection Spec — Review Addendum / Tightening Pass

This document preserves the implementation intent of the attached spec, but tightens several requirements so an implementation agent has fewer ambiguous choices and the resulting behavior is easier to analyze scientifically.

The original spec already makes several good structural choices: it centralizes ADAPT energy noise at a single chokepoint, avoids touching the already-noisy oracle path, and logs the new controls in run metadata. The main issues to fix are reproducibility, one possible double-counting path in the controller, and a few places where the current wording overstates what the Gaussian proxy can claim to model.

---

## 1. High-priority changes before implementation

### 1.1 Reframe the scientific claim
Replace wording such as **"accurately predict"** hardware behavior with language such as **"fast stochastic proxy"**, **"first-order stress-test proxy"**, or **"coarse analytic approximation"**.

Reason: the proposed additive Gaussian model is useful, but it is not a faithful physical noise model for all of the quantities being perturbed (energies, commutator gradients, geometric tensors, pseudo-inverse-driven controller metrics). Without a calibration protocol, the current wording claims more fidelity than the implementation can justify.

If you want to keep the word **"calibrated"**, add a short calibration protocol (see §4 below). Otherwise, call the knobs **user-specified Gaussian analytic noise**.

### 1.2 Add reproducibility controls
Add optional per-pipeline seeds and use isolated RNG objects.

Recommended new controls:

- `--adapt-analytic-noise-seed` (optional `int`, default `None`)
- `--checkpoint-controller-analytic-noise-seed` (optional `int`, default `None`)

Use `np.random.default_rng(seed)` and keep the RNG local to the relevant run/controller. Do **not** rely on the module-global `np.random.normal(...)` state.

Required semantics:

- Each noisy oracle call receives a fresh independent draw from the run-scoped RNG.
- Re-evaluating the same parameter vector twice within a single run should produce different draws.
- Two full runs with the same seed and the same code path should be reproducible.
- The analytic-noise RNG must not interfere with any unrelated stochastic code (for example, SPSA randomness or any other repository-wide RNG use).

### 1.3 Validate all noise std values
All analytic-noise standard deviations must satisfy `std >= 0.0`.

Fail fast during CLI/config parsing if a negative value is provided. Do not silently coerce negative values.

### 1.4 Decide whether final reported energies should be noisy or exact
The ADAPT spec injects noise at `_evaluate_selected_energy_objective`, which the original document correctly identifies as the chokepoint for SPSA objectives, final refits, and prune refits. That means the final reported energies may also become noisy, not just the search-time objective values.

This needs an explicit decision.

Recommended behavior:

- Use noisy energies for **search / optimization / pruning decisions**.
- Also compute and log an **exact final energy** for post-analysis whenever the run reaches a final selected ansatz.

If you do **not** want any call-site edits right now, then state explicitly that all energies returned through `_evaluate_selected_energy_objective` are intentionally noisy, including final refit/prune energies.

Do not leave this implicit.

### 1.5 Avoid likely double-counting in controller `rho_miss`
The current controller spec perturbs `G` and `f`, then later adds a second Gaussian perturbation directly to `rho_miss` at the decision boundary.

That is a second independent noise source. In many implementations, noisy `G` and `f` already perturb the downstream `theta_dot_*` quantities and therefore the derived `rho_miss`.

Recommended v1 behavior:

- Inject noise into `G` and `f`.
- **Do not** inject a second independent noise term directly into `rho_miss`.

If you later want threshold-jitter experiments, add a **separate** control such as `--checkpoint-controller-rho-miss-noise-std`, document it as a second-stage heuristic, and keep it disabled by default.

### 1.6 Clarify the meaning of the controller std
A single absolute standard deviation applied to both `G` and `f` is a **coarse stress-test knob**, not a dimensionally faithful backend model.

Document that explicitly in the CLI help and in the spec.

If you later want a closer calibration, split it into separate controls, for example:

- geometry / tensor noise std
- gradient / force-vector noise std
- optional threshold / decision-metric noise std

For this revision, a single controller std is acceptable **if** you state that it is intentionally coarse and applied in the native units of each quantity.

### 1.7 Treat line numbers as soft anchors only
The current spec references exact line numbers throughout. Keep those as convenience anchors, but state clearly that:

- function names,
- signatures,
- and code semantics

are authoritative.

The implementation agent should search by function name if the repository has shifted.

### 1.8 Add a minimal failure policy for non-finite noisy controller metrics
Noisy `G`, `K`, pseudo-inverses, or derived scalars may become ill-conditioned or non-finite under aggressive stress settings.

Specify a conservative failure policy instead of leaving behavior undefined.

Recommended rule:

- If analytic noise produces non-finite controller metrics (`NaN`/`inf`) at a decision point, do **not** crash the run.
- Prefer the conservative action:
  - disallow pruning,
  - and either append or mark the step degraded,
- while logging a machine-readable `degraded_reason`, e.g. `analytic_noise_nonfinite_metric`.

This turns large-noise runs into usable stress tests instead of brittle failures.

---

## 2. Required implementation clarifications and additions

### 2.1 New seed fields and logging
If you adopt §1.2, add and log the following.

#### ADAPT pipeline

Add CLI argument:

```python
p.add_argument("--adapt-analytic-noise-seed", type=int, default=None,
               help="Optional RNG seed for analytic Gaussian noise draws in exact ADAPT energy/gradient evaluations.")
```

In `run_adapt_vqe`, instantiate a local RNG once and capture it in the enclosing scope used by the energy and gradient paths.

```python
adapt_analytic_noise_seed = getattr(args, "adapt_analytic_noise_seed", None)
adapt_noise_rng = np.random.default_rng(adapt_analytic_noise_seed)
```

Log both:

- `adapt_analytic_noise_std`
- `adapt_analytic_noise_seed`

#### Realtime controller

Add config field (match repository typing style):

```python
analytic_noise_seed: int | None = None
```

Add CLI argument:

```python
p.add_argument("--checkpoint-controller-analytic-noise-seed", type=int, default=None,
               help="Optional RNG seed for analytic Gaussian noise draws in the checkpoint controller.")
```

Wire the seed into `RealtimeCheckpointConfig(...)`, instantiate a controller-local RNG once, and log both:

- `analytic_noise_std`
- `analytic_noise_seed`

If the repo targets Python versions where `int | None` is undesirable, use `Optional[int]` instead.

### 2.2 Use helper functions instead of duplicating ad hoc noise logic
This is not strictly required, but it is strongly recommended.

Suggested helpers:

```python
def _add_scalar_gaussian_noise(x: float, std: float, rng) -> float:
    return float(x) if std <= 0.0 else float(x) + float(rng.normal(0.0, std))


def _add_vector_gaussian_noise(x: np.ndarray, std: float, rng) -> np.ndarray:
    return np.asarray(x, dtype=float) if std <= 0.0 else np.asarray(x, dtype=float) + rng.normal(0.0, std, size=x.shape)


def _add_symmetric_gaussian_noise(x: np.ndarray, std: float, rng) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if std <= 0.0:
        return x
    n = x.shape[0]
    noise = rng.normal(0.0, std, size=(n, n))
    noise = np.triu(noise)
    noise = noise + np.triu(noise, 1).T
    return x + noise
```

Benefits:

- isolates RNG behavior,
- centralizes validation assumptions,
- reduces copy/paste drift,
- makes later calibration changes easier.

### 2.3 Sample symmetric matrix noise directly
The original controller snippet adds a full iid matrix and then symmetrizes via `0.5 * (G + G.T)`.

That preserves symmetry, but it does **not** preserve the requested standard deviation on off-diagonal entries: the off-diagonal effective std becomes `std / sqrt(2)`.

If you want the configured std to mean what it says for each unique symmetric entry, sample the symmetric noise directly.

Recommended pattern:

```python
if float(self.cfg.analytic_noise_std) > 0.0 and G.size:
    _noise_std = float(self.cfg.analytic_noise_std)
    sym_noise = self._analytic_noise_rng.normal(0.0, _noise_std, size=G.shape)
    sym_noise = np.triu(sym_noise)
    sym_noise = sym_noise + np.triu(sym_noise, 1).T
    G = np.asarray(G + sym_noise, dtype=float)
    f = np.asarray(
        f + self._analytic_noise_rng.normal(0.0, _noise_std, size=f.shape),
        dtype=float,
    )
```

If you intentionally prefer the current "sample then symmetrize" behavior, document the effective off-diagonal variance change explicitly.

### 2.4 Conservative handling of non-finite outputs
After noise injection and pseudo-inverse computation, add a minimal guard around controller outputs.

Recommended behavior:

```python
if not np.all(np.isfinite(G)) or not np.all(np.isfinite(f)):
    ... conservative fallback / degraded logging ...
```

and similarly for `theta_dot_proj`, `theta_dot_step`, or `rho_miss` if they are the first convenient post-compute checkpoints.

The key requirement is: **do not let large-noise experiments crash the workflow unnecessarily**.

---

## 3. Specific section edits to the original spec

### 3.1 Section 1 — Executive Summary & Objectives
Change the opening objective language from strong prediction claims to a stress-test framing.

Recommended replacement idea:

> Our goal is to build a fast stochastic proxy for how the active decision loops may degrade under measurement-like uncertainty on superconducting hardware. This proxy is intended for rapid comparative stress tests, not as a fully faithful hardware-noise simulator.

### 3.2 Section 2 — Theory
Keep the observable-variance formula, but add one sentence making the approximation boundary explicit.

Recommended addition:

> For gradients, geometric-tensor entries, and controller diagnostics, the same Gaussian additive rule is used as a heuristic proxy. It should be interpreted as an intentionally coarse perturbation model rather than the exact sampling distribution of each estimator.

### 3.3 Section 3.2 — ADAPT energy objective
Keep the chokepoint approach. It is a good choice.

Add the following clarifications:

- noise draws are independent per evaluation;
- the RNG is local to the run;
- `std = 0.0` is a strict no-op;
- negative `std` is invalid;
- state explicitly whether final reported energies are noisy, exact, or both.

### 3.4 Section 3.3 — ADAPT pool gradient scorer
Keep the current exact-gradient injection point.

Add one implementation detail:

- cast/store the noisy gradient as a real float in the same dtype convention already expected by downstream code.

For example:

```python
grad_exact = float(adapt_commutator_grad_from_hpsi(hpsi_current, apsi))
if adapt_analytic_noise_std > 0.0:
    grad_exact += float(adapt_noise_rng.normal(0.0, adapt_analytic_noise_std))
gradients[i] = grad_exact
grad_magnitudes[i] = abs(grad_exact)
```

This avoids hidden dtype drift if the original function returns a NumPy scalar.

### 3.5 Section 4.4 — Controller geometry noise
Keep the injection location (before `pinv`) but change the implementation details as follows:

- use a controller-local RNG;
- sample symmetric `G` noise directly;
- document that this is a coarse native-units perturbation;
- add non-finite guards downstream.

### 3.6 Section 4.5 — Controller `rho_miss`
Recommended change: **remove this section from v1**.

Reason: noisy `G` and `f` already propagate uncertainty into the controller dynamics and the derived diagnostics. A second direct scalar perturbation to `rho_miss` is a different experiment and should not be bundled into the base feature without its own control flag.

If you want to keep it anyway, rewrite the section to say that it is an **optional second-stage heuristic** with its own std parameter, defaulting to `0.0`.

### 3.7 Section 5 — Guardrails
Add the following guardrails:

- analytic-noise RNG must be isolated from unrelated stochastic code;
- all std parameters must be non-negative;
- line numbers are approximate anchors only;
- large-noise runs must degrade conservatively rather than crash;
- if seeds are added, all metadata outputs must record both std and seed.

---

## 4. Calibration note (optional but recommended)
If the word **"calibrated"** remains anywhere in the document, add a minimal calibration procedure.

One workable lightweight protocol:

1. Choose a small representative set of states / controller snapshots.
2. For each target quantity class (energy, commutator gradient, controller geometry terms), compare exact values against a small batch of repeated noisy evaluations on the backend or on a noisy simulator.
3. Estimate empirical residual scales.
4. Set the analytic-noise std to roughly match the observed dispersion for the intended stress-test regime.
5. Record the calibration source in the run notes or metadata.

Without a step like this, the implementation is still useful, but the word **"calibrated"** should be avoided.

---

## 5. Minimal acceptance tests the implementation agent should run

### 5.1 Zero-noise regression
With all new std parameters set to `0.0`, behavior must match the prior exact baseline within existing tolerances.

### 5.2 Seeded reproducibility
With a fixed seed and fixed inputs:

- repeated full runs should match,
- changing the seed should change the noisy trajectory.

### 5.3 Metadata coverage
Verify that JSON / ledger outputs include the new noise std values and any added seeds.

### 5.4 Controller symmetry invariant
Verify that the noisy `G` passed into pseudo-inverse computation is symmetric.

### 5.5 Conservative failure handling
Under intentionally large std values, the workflow should not fail with an unhandled numerical exception. It should either continue conservatively or emit a degraded reason.

### 5.6 Exact-vs-noisy reporting behavior
Whichever choice you make in §1.4 must be tested explicitly:

- either confirm that final reported energies are intentionally noisy,
- or confirm that exact final energies are logged separately.

---

## 6. Bottom-line recommendation
If you want the smallest change set that materially improves the original spec, do these four things before handing it to an implementation agent:

1. add per-pipeline analytic-noise seeds and isolated RNGs,
2. state clearly whether final reported energies are noisy or exact,
3. remove direct `rho_miss` noise from v1,
4. add a small acceptance-test section.

Those changes will make the implementation much less ambiguous while preserving the original architecture and chokepoints.

# High-Amplitude Spectra Verification Plan (3-Step Pipeline)

## 1. Context and Aim

According to the lessons detailed in `docs/reports/2026-04-05-qpu-spectra-accuracy-checklist.md`, checking energy convergence (`Delta E`) and global state fidelity is not sufficient to guarantee that actual density response spectra (e.g., $d(t) = n_0(t) - n_1(t)$) obtained on QPU will be accurate. If the oscillation span of the target observable is too small (e.g. `~0.3`), numerical and algorithmic tracking errors swamp the signal, leading to spurious spectral peaks.

**Our Aim:**
We will forcefully increase the observable signal by softening the Hamiltonian and driving it harder:
- **Hamiltonian Tuning:** Drop the effective repulsion $U$ from $4.0$ to $1.0$ (making the lattice more metallic). 
- **Drive Tuning:** Increase the active driving amplitude $A$ to $1.5$ and use an off-resonance frequency (e.g., $\omega = 1.2$).

These alterations ensure $d(t)$ will execute huge, easily perceptible swings. This makes deviations by the controller very plain to see and mathematically robust when analyzing spectral accuracy.

We will use the top performing ADAPT Ground-State framework documented in `MATH/Math.md` (`pareto_lean`, `phase3_v1`, `SPSA maxiter=3200`, `frontier 0.85`), passing its output to the best-performing heavy dynamics controller (`secant_lead100`). Once the dynamic evolution is complete, we run a native FFT script to analyze the spectral fidelity.

## 2. The 3-Step Execution Plan

### Step 1: Initial State Preparation (ADAPT-VQE)
Create the starting scaffold using the exact "winner" configuration from `Math.md` with altered `U=1.0`. Wait for this command to finish and verify `step1_adapt_state.json` is created.

```bash
mkdir -p artifacts/agent_runs/high_amp_spectra_run

python -u -m pipelines.hardcoded.adapt_pipeline \
  --L 2 --problem hh --t 1.0 --u 1.0 --dv 0.0 \
  --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --boson-encoding binary --ordering blocked --boundary open --term-order sorted \
  --adapt-pool pareto_lean --adapt-continuation-mode phase3_v1 \
  --adapt-max-depth 160 --adapt-eps-grad 5e-7 --adapt-eps-energy 1e-9 --adapt-seed 5 \
  --adapt-inner-optimizer SPSA \
  --adapt-spsa-a 0.1 --adapt-spsa-c 0.02 --adapt-spsa-A 5.0 \
  --adapt-spsa-callback-every 5 --adapt-spsa-progress-every-s 30 \
  --adapt-maxiter 3200 \
  --adapt-state-backend compiled \
  --adapt-reopt-policy windowed --adapt-window-size 999999 --adapt-window-topk 999999 \
  --adapt-full-refit-every 8 --adapt-final-full-refit true \
  --adapt-beam-live-branches 2 --adapt-beam-children-per-parent 2 --adapt-beam-terminated-keep 2 \
  --phase1-prune-enabled --phase1-prune-fraction 0.25 --phase1-prune-max-candidates 6 --phase1-prune-max-regression 1e-8 \
  --phase1-probe-max-positions 999999 --phase1-trough-margin-ratio 1.0 --phase1-shortlist-size 64 \
  --phase2-shortlist-fraction 1.0 --phase2-shortlist-size 64 \
  --phase2-frontier-ratio 0.85 \
  --phase3-frontier-ratio 0.85 \
  --phase3-symmetry-mitigation-mode verify_only --phase3-enable-rescue \
  --phase3-backend-cost-mode proxy --phase3-runtime-split-mode off --phase3-lifetime-cost-mode off \
  --adapt-drop-floor 0.0005 --adapt-drop-patience 3 --adapt-drop-min-depth 12 \
  --initial-state-source adapt_vqe --skip-pdf --phase2-no-batching \
  --output-json artifacts/agent_runs/high_amp_spectra_run/step1_adapt_state.json
```

### Step 2: Time-Dynamics with the `secant_lead100` Controller
We execute a script that mimics the pipeline logic recorded in `artifacts/agent_runs/20260405_controller_drive_heavier_secant_lead_scan_v1/logs/command.sh`, loading the scaffold generated in Step 1, using the $A=1.5, \omega=1.2$ drive, and saving the trajectory to a new JSON.

Save the following into `run_step2_dynamics.py` and run it with `python run_step2_dynamics.py`:

```python
import json
from pathlib import Path
import numpy as np

from pipelines.hardcoded.hh_fixed_manifold_mclachlan import FixedManifoldRunSpec, load_run_context
from pipelines.hardcoded.hh_realtime_checkpoint_controller import ControllerDriveConfig, RealtimeCheckpointController
from pipelines.hardcoded.hh_realtime_checkpoint_types import RealtimeCheckpointConfig
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix

ARTIFACT = Path('artifacts/agent_runs/high_amp_spectra_run/step1_adapt_state.json')
OUT_JSON = Path('artifacts/agent_runs/high_amp_spectra_run/step2_dynamics.json')

SPEC = FixedManifoldRunSpec(
    name='adaptive_heavy_drive_secant',
    artifact_json=ARTIFACT,
    loader_mode='replay_family',
    generator_family='match_adapt',
    fallback_family='full_meta',
)

loaded = load_run_context(SPEC, tag='heavy_drive', lock_fixed_manifold=False)
hmat = np.asarray(hamiltonian_matrix(loaded.replay_context.h_poly), dtype=complex)

drive_cfg = ControllerDriveConfig(
    enabled=True,
    n_sites=int(loaded.cfg.L),
    ordering=str(loaded.cfg.ordering),
    drive_A=1.5,
    drive_omega=1.2,
    drive_tbar=4.0,
    drive_phi=0.0,
    drive_pattern='staggered',
    drive_custom_weights=None,
    drive_include_identity=False,
    drive_time_sampling='midpoint',
    drive_t0=0.0,
    exact_steps_multiplier=4,
)

cfg = RealtimeCheckpointConfig(
    mode='exact_v1',
    miss_threshold=0.05,
    exact_forecast_baseline_blend_weights=(-0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0),
    exact_forecast_include_tangent_secant_proposal=True,
    exact_forecast_tangent_secant_trust_radius=0.75,
    exact_forecast_tangent_secant_signed_energy_lead_limit=1.0,
    exact_forecast_tracking_horizon_steps=2,
    exact_forecast_tracking_horizon_weights=(2.0, 1.0)
)

controller = RealtimeCheckpointController(
    cfg=cfg,
    replay_context=loaded.replay_context,
    h_poly=loaded.replay_context.h_poly,
    hmat=hmat,
    psi_initial=loaded.psi_initial,
    best_theta=loaded.replay_context.adapt_theta_runtime,
    allow_repeats=False,
    t_final=8.0,
    num_times=161,
    drive_config=drive_cfg
)

result = controller.run()
output_data = {
    'trajectory': [dict(row) for row in result.trajectory]
}
OUT_JSON.write_text(json.dumps(output_data, indent=2))
print("Step 2 completed successfully.")
```

### Step 3: Density Fourier Transform & Plotting
Taking the time evolution created in Step 2, run the FFT parsing tool, isolating the density pair $0,1$ (which interprets to $d(t) = n_0(t) - n_1(t)$), detrending it, and evaluating the spectra.

```bash
python -m pipelines.hardcoded.hh_time_dynamics_spectra \
  --input-json artifacts/agent_runs/high_amp_spectra_run/step2_dynamics.json \
  --output-png artifacts/agent_runs/high_amp_spectra_run/step3_density_spectrum.png \
  --pair 0,1 \
  --time-key time \
  --window hann \
  --detrend linear
```

This completes the 3-step validation pipeline!