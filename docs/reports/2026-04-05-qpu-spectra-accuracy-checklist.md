# QPU spectra accuracy checklist for HH time dynamics

## Objective

We want spectra of site occupation, site occupation difference, or staggered density that remain physically meaningful on a real QPU run. The key lesson from the current controller benchmarks is that good total energy tracking and good global fidelity do not by themselves guarantee accurate density spectra. If the target density observable is wrong in time, its Fourier spectrum will also be wrong.

## Core principle

Optimize and validate the same observable that will be Fourier transformed.

For the present L = 2 use case, the most useful primary signal is:

- `d(t) = n_0(t) - n_1(t)`

This carries the same physics content as normalized staggered density but has larger amplitude. For larger L, use one or more of:

- `m(t) = (1/L) sum_j (-1)^j n_j(t)`
- `Delta_eo(t) = sum_{j in even} n_j(t) - sum_{j in odd} n_j(t)`
- `rho_q(t) = (1/L) sum_j exp(-i q j) (n_j(t) - nbar(t))`

If the eventual spectrum is for `d(t)`, then controller tuning, benchmark scoring, and QPU measurement design should all center on `d(t)`, not only on `Delta E` or fidelity.

## What to measure on a real QPU run

### 1. Measure the exact observables needed for the spectrum

At each sampled time, record:

- time `t_n`
- per-site occupations `n_j(t_n)`
- if possible, spin-resolved occupations `n_j_up(t_n)` and `n_j_dn(t_n)`
- derived imbalance observables such as `d(t)` or `m(t)`
- doublon if physically relevant
- total energy only as a secondary diagnostic

For L = 2, if the target spectrum is density response under a staggered drive, `d(t) = n_0 - n_1` should be treated as the primary observable.

### 2. Keep the time grid uniform and physically matched to the spectrum goal

Record and preserve:

- uniform sampling interval `Delta t`
- total observation time `T_obs`
- number of time points `N`
- exact drive parameters used on hardware: amplitude, frequency, phase, envelope center, envelope width, and drive pattern

Practical targets:

- choose `Delta t` so there are about 8 to 10 samples per shortest period of interest
- choose `T_obs` long enough to resolve the smallest spectral spacing of interest
- prefer at least `N >= 128` if the goal is a resolved spectrum rather than a coarse sanity check
- include enough late-time data if the goal is a steady or quasi-periodic response spectrum

### 3. Measure uncertainty, not just the mean

At each time point, keep enough information to estimate error bars on the target observable:

- shot count
- estimated mean
- estimated standard error
- readout-mitigation status and calibration metadata
- repeated acquisitions if feasible

For difference observables such as `d = n_0 - n_1`, preserve simultaneous measurement information when possible. The uncertainty in a difference depends on covariance, not only on separate marginal errors.

### 4. Monitor hardware drift and baseline artifacts

For a real QPU campaign, also record:

- calibration epoch / job timestamp
- readout assignment data or mitigation matrices
- a no-drive or A = 0 baseline when possible
- repeated reference points during the scan if the run is long

This matters because low-frequency hardware drift can masquerade as genuine spectral weight.

## What to compute from the measured time series

### 1. Work with the fluctuation signal, not only the raw observable

For a chosen observable `x(t)`, define a fluctuation signal on the analysis window:

`delta x(t) = x(t) - <x>_window`

Use the windowed fluctuation signal for spectral analysis. For finite windows, a Hann taper is a good default.

### 2. Primary spectral products

For each target observable, keep:

- raw time trace
- mean-subtracted fluctuation trace
- one-sided amplitude spectrum
- peak list with amplitude, angular frequency, and uncertainty proxy
- drive-locked amplitudes and phases at `omega_d`, `2 omega_d`, `3 omega_d`, ... when the drive frequency is known

### 3. Uncertainty propagation to the spectrum

For QPU data, do not stop at a single FFT curve. Also estimate uncertainty by one or more of:

- bootstrap over repeated acquisitions
- shot-noise resampling based on measured standard errors
- repeated hardware runs over the same time grid
- sensitivity under mean subtraction vs linear detrending and under different windows

## What to measure when exact reference is available

When exact propagation or a trusted classical benchmark is available, the most important metrics are on the same observable that will be Fourier transformed.

For a target observable `x(t)`, define the oscillatory relative error:

`epsilon_osc = ||delta x_ctrl - delta x_exact||_2 / ||delta x_exact||_2`

Interpretation:

- `epsilon_osc < 0.2`: very encouraging for spectral trust
- `0.2 to 0.3`: maybe usable, but check line-by-line robustness
- `> 0.3`: spectral claims are risky

Also measure:

- mean absolute observable error over time
- RMS observable error over time
- phase error at the drive line and low harmonics
- amplitude error at the drive line and low harmonics
- spectral overlap between controller and exact amplitude spectra

A practical amplitude rule:

- if the exact oscillation scale is of order `0.06`, target observable error should ideally be well below `0.01`
- if observable error is comparable to the oscillation size, the spectrum will usually be distorted even if `Delta E` looks good

## What to use when exact reference is not available on the QPU

On the real QPU we may not have a full exact reference at the target size. In that case, use the following proxies.

### A. Stability under measurement and post-processing choices

A trustworthy spectral feature should be stable under:

- repeated hardware runs
- small changes in shot budget
- Hann vs rectangular window
- full-window vs late-time-window analysis
- mild detrending changes

### B. Stability under controller/model refinement

A trustworthy spectral feature should also survive:

- larger or richer manifold / ansatz
- smaller propagation step size if applicable
- stronger local solver tolerance
- small controller hyperparameter changes

If a claimed peak moves a lot under these changes, it is not yet trustworthy.

### C. Drive-locked consistency

If the drive frequency is known, fit or lock in at `omega_d` and a few harmonics. This is often more robust on hardware than trusting every FFT bin.

Useful QPU-facing quantities:

- amplitude at `omega_d`
- phase lag at `omega_d`
- amplitude at `2 omega_d` and `3 omega_d`
- ratio of harmonic amplitudes

These are often the first physically meaningful spectral observables to trust.

## What to optimize if the goal is accurate spectra

This is the most important section for controller or objective design.

### Optimize directly for target density observables

If the desired spectrum is from site occupations or imbalance modes, the controller score should explicitly reward accuracy in those observables.

High-priority objective terms:

- `abs(d_ctrl - d_exact)` or `abs(m_ctrl - m_exact)`
- max or RMS site-occupation error
- doublon error if doublon sidebands are physically relevant
- oscillatory error after mean removal, not only raw error
- late-time-window observable error if the spectrum is intended to represent a late-time regime

### Optimize drive-locked amplitude and phase, not only total state closeness

For driven problems, add explicit targets such as:

- amplitude error at `omega_d`
- phase error at `omega_d`
- amplitude error at `2 omega_d` and `3 omega_d`
- spectral weight in the expected response band

This is much closer to the eventual scientific objective than only minimizing `Delta E`.

### Optimize frequency content, not only pointwise time traces

Useful secondary objectives include:

- overlap between controller and benchmark spectra for the target observable
- error in the first few dominant spectral peaks
- penalty for spurious low-frequency drift if that drift is known to be nonphysical

### Keep energy and fidelity, but demote them to supporting metrics

`Delta E` and fidelity remain useful, but they should be treated as secondary guards, not the only headline objectives.

Why:

- energy can track well while density response is wrong
- fidelity can be high while phase-sensitive observables are still distorted
- a controller can match a smooth energy envelope yet miss the mode content that produces the target spectrum

## Metrics to include in future benchmark summaries

For each run, include a compact spectral-trust block with at least:

- target observable name
- exact or benchmark oscillation span
- controller oscillation span
- mean absolute observable error
- RMS observable error
- `epsilon_osc`
- drive-line amplitude and phase
- first few harmonic amplitudes
- spectrum robustness flags across windows / detrending / repeated runs
- energy error and fidelity as secondary context

## Minimal acceptance gate before claiming a spectrum

Before making a strong spectral claim, require:

1. observable error much smaller than the oscillation scale of the target mode
2. dominant drive-locked amplitude and phase reasonably stable
3. main peaks stable under mild preprocessing changes
4. no-drive or baseline runs free of comparable spurious peaks
5. convergence under modest controller or ansatz refinement

If these do not hold, the safest statement is that the run shows qualitative oscillatory response, not yet a quantitatively reliable spectrum.

## Practical QPU recommendation

If shot budget is limited, prioritize the target observable and spectral trust over a large menu of secondary observables.

A good priority order is:

1. `n_j(t)` sufficient to reconstruct `d(t)` or `m(t)`
2. repeated time traces for uncertainty and drift estimation
3. drive-locked harmonic extraction
4. total energy only if still affordable

For the present problem, a QPU-oriented acquisition plan should be built around obtaining the cleanest possible estimate of `d(t) = n_0 - n_1` on a uniform time grid with reliable uncertainty bars.

## What the current repo benchmarks are telling us

Current controller evidence already shows:

- energy agreement can look reasonably good while density observables remain inaccurate
- making the exact density oscillation larger helps the signal-to-noise side of the problem
- but larger signal alone is not enough if the controller still undertracks the target density mode or puts too much weight in the wrong frequency band

So the right direction is not only stronger response, but also objective functions and validation metrics that directly target the density mode whose spectrum we ultimately care about.

## Suggested next repo changes

1. Add a `spectral_trust` summary block to time-dynamics JSON outputs.
2. Report `d(t) = n_0 - n_1` as a first-class observable for L = 2 runs.
3. Add drive-locked amplitude and phase diagnostics to the post-processing script.
4. In controller scans, rank runs partly by target-observable oscillatory error rather than only by energy-style scores.
5. When possible, compare full-window and late-window spectra in the same artifact.

## Machine appendix

Reference artifacts used to motivate this checklist:

- `artifacts/agent_runs/20260405_controller_drive_excursion_under_scan_v1/json/drive_A1p5_h2_slope500_curv200_exc800.json`
- `artifacts/agent_runs/20260405_controller_drive_excursion_band_scan_v1/json/drive_A1p5_h2_slope500_curv200_band_u900_o120_t03.json`
- `artifacts/agent_runs/20260405_controller_drive_heavier_target_v1/json/drive_A0p55_tbar4p0_t00p0_current_best.json`
- `artifacts/agent_runs/20260405_controller_drive_heavier_signed_blend_scan_v1/json/signed_mild_front.json`
- `artifacts/agent_runs/20260405_controller_drive_heavier_slope_anchor_refresh_v1/json/signed_anchor_new.json`

Empirical lesson from those artifacts:

- density spectra are controlled by density-observable accuracy, not by `Delta E` alone
- for the heavier-drive case, the exact pair-difference oscillation is larger, but the controller pair-difference error remains large enough that spectral values are still not yet trustworthy
