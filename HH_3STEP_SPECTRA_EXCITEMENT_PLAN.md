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