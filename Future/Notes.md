### Optional PAOP-LF tweaks (recommended defaults)

**1) Force `paop_r >= 1` for `paop_lf_full` (match prior `paop_full` behavior).**  
Rationale: `paop_lf_full` is intended to include *extended cloud* terms; allowing `paop_r=0` silently removes nonlocal dressing and can weaken the “LF-full” meaning.

Suggested behavior:
- If `--adapt-pool paop_lf_full` and `--paop-r 0`, internally promote to `paop_r = 1`.

**2) Make `hop2` phonon-identity dropping user-configurable (keep current default ON).**  
Current: `paop_hop2(i,j) = K_ij (p_i - p_j)^2` then drop terms that are identity on all phonon qubits.  
Rationale: dropping prevents `hop2` from degenerating into a pure `K_ij` copy (can distort ADAPT selection), but it may be useful to toggle for ablation.

Suggested CLI flag:
- `--paop-hop2-drop-phonon-identity {0,1}` (default `1`)




ansatz factorization, measurement
grouping, symmetry checks

# Notes — “From a VQE–Trotter Hubbard Pipeline to Multiple Publishable Quantum Computing Manuscripts”



## 2) The “six paper” portfolio (high-signal splits)

### A) Eigenphase / leakage diagnostics + correction (VQE → dynamics interface)

* Core question: what *spectral leakage / phase drift* occurs when you evolve a **not-quite-eigenstate** VQE output under product-formula dynamics?
* Deliverable: a **diagnostics-and-correction framework** separating (i) initial-state contamination vs (ii) propagation-induced error, using variance/short-time tests + small subspace (Krylov/QSE-like) phase alignment.

### B) Driven Hubbard nonequilibrium benchmark + error budget

* Deliverable: reproducible driven-observable dataset under Gaussian-envelope drives (densities, doublons, correlators, etc.).
* Key emphasis: a grounded **error budget** decomposing (i) VQE infidelity, (ii) drive discretization + Strang splitting error, (iii) compilation/ordering effects.

### C) Integrator comparison: Strang vs CFQM (time-dependent simulation)

* Deliverable: “control-like workload” benchmark using your drive protocol to compare **Strang** against **commutator-free quasi-Magnus (CFQM/CFET)** schemes.
* Key result aim: identify crossover regimes where CFQM reduces depth for fixed error, with an explicit cost model.

### D) Hubbard–Holstein ADAPT-VQE + unary phonons

* Deliverable: an explicitly digital, symmetry-aware **ADAPT-VQE benchmark for HH** with **unary-encoded truncated phonons**, including operator-pool design and Pareto fronts (qubits vs depth vs measurement cost).
* Includes a “FACT CHECK” style mandate: verify/clean up resource-count and bosonic-operator formulas under truncation/encoding.

###

## 3) Implementation Contract: HH L=4 staged HVA -> ADAPT -> final VQE

### Goal and rationale
- Hartree-Fock-only starts can produce near-zero gradients and stall ADAPT.
- HVA is used as a short seed stage to produce a non-dead reference.
- ADAPT is then used to build expressive ansatz structure efficiently.
- Once marginal ADAPT gains flatten, hand off to fixed-ansatz VQE for final coefficient refinement.

### Default run constants (L=4 HH)
- `adapt_max_depth_probe = 54`
- `adapt_maxiter_probe = 1778`
- Probe gate: `delta_E_abs_best <= 5e-2`
- Production gate: `delta_E_abs_best <= 1e-2`
- Aspirational target (report-only): `<= 5e-3`

### Escalation ladder when probe fails (`delta_E_abs_best > 5e-2`)
1. Escalation A
- `ws_restarts += ceil(L/3)` (for `L=4`, add `2`)
- `ws_maxiter *= 2`
- `adapt_maxiter *= 2`
2. Escalation B
- `adapt_max_depth = ceil(1.25 * adapt_max_depth)`
- Depth cap: `60 * L` (for `L=4`, max `240`)

### Wallclock and stall policy
- Do not run blind forever.
- If a stage emits no new log lines for `>20 min`, mark stalled and proceed to next attempt.
- Hard wallclock caps:
- probe attempts: `1800s`
- production attempts: `7200s`
- Continue on attempt-level errors; do not abort entire workflow because one attempt fails.

### ADAPT handoff cutoff policy (combined criteria)
Use an energy-first policy; hand off ADAPT -> final VQE when primary criterion is satisfied:

1. Primary: energy-error-drop + patience cutoff
- Track per-depth `delta_E_abs_best` and `drop(d) = delta_E_abs(d-1) - delta_E_abs(d)`.
- If `drop(d)` stays below a configured floor for `N` consecutive completed depths
  (after minimum depth guard), stop ADAPT.

2. Secondary: gradient-floor diagnostic (optional)
- Track `max|g|` during ADAPT for diagnostics/safety.
- Use gradient floor only as a secondary guard; never as the sole stop trigger.

Initial tuning placeholders (must be set before production campaign):
- drop floor
- patience windows `N`
- minimum depth guard
- optional gradient floor `g_min` (secondary only)

### Execution sequence
1. Run PROBE attempt with probe settings.
2. Parse probe JSON and compute `delta_E_abs_best`.
3. If probe fails gate, apply Escalation A and rerun probe once.
4. If still failing, apply Escalation B and rerun probe once.
5. On first probe pass, run PRODUCTION with full L=4 defaults.
6. If production misses `1e-2`, keep best run and record as `wallclock/budget-limited`.

### Artifact and reporting contract
- Required summary artifact:
- `artifacts/useful/L4/${TAG}_summary.md`
- Required compact attempts table:
- `artifacts/useful/L4/${TAG}_attempts.csv`
- Required CSV columns:
- `attempt_id,stage,delta_E_abs_best,relative_error_best,gate_pass,runtime_s,stop_reason,artifact_json,artifact_log`
- Keep per-attempt JSON/log/CSV/MD artifacts for probe, probeA, probeB, and production attempts.

### Operator-first benchmark extras to include in summaries
- Pool composition counts (raw and deduplicated A/B pools)
- ADAPT depth reached per stage
- `nfev_total` per stage
- ADAPT stop reasons
- Warm-stage runtime and optimizer effort stats



 If you want true “slope-triggered warm cutoff then continue ADAPT in the
     same run,” I can implement a minimal code change to add that warm handoff
     behavior directly in the script.
Basically, we want the HVA algorithm to stop after a fixed number of depths, when the gradient is less than or equal to some number, say e-3. The depth or the last layer should depend on L. Then we want to have the analogous regimen for the ADAPT VQE, where once the gradient for a certain number of iterations becomes less than some value, we'll stop that. This will be our ansatz that we will assume is sufficiently expressive, and we'll plug it into the Conventional VQE. 
## Phase 2C follow-on ideas (legacy parity + noise validation)

1. Runtime skip artifacts:
- When `--noise-mode runtime` is unavailable (credentials/access), write a structured JSON `skipped` result instead of hard-failing.

2. Baseline registry:
- Add a canonical registry file (for example `artifacts/json/legacy_baselines.json`) that maps named parity anchors to artifact paths and required observables.
- Avoid hardcoded baseline paths in CLI examples.

3. CI parity checks:
- Add a lightweight CI job that runs legacy-parity checks on locked anchors and fails on gate regressions.
- Surface `legacy_parity.passed_all` and max deltas as CI summary outputs.

4. Optional non-matching-grid interpolation mode:
- Keep strict default (`time_grid_match` required).
- Add an explicit opt-in interpolation compare mode for diagnostics only, never as the parity pass/fail gate.


## L4 HH Overnight Diagnostic Plan (2026-03-04)

### Context
- Observation: convergence is slower than expected despite using an expressive pool path.
- Current decision: continue seeded conventional VQE as primary branch; ADAPT branch is stopped.
- Requirement: preserve current overnight run continuity and extend runtime safety cap by `+12h`.

### Fixed physics contract (must remain unchanged)
- `problem=HH`
- `L=4`
- `boundary=open`
- `ordering=blocked`
- `boson_encoding=binary`
- `n_ph_max=1`
- `sector=(2,2)`
- `t=1.0`, `U=4.0`, `dv=0.0`, `omega0=1.0`, `g_ep=0.5`

### Diagnostic objective
Disambiguate which factor dominates slow progress:
- expressivity ceiling
- optimizer/conditioning bottleneck
- initialization/restart strategy inefficiency

### Diagnostic matrix (future runs; same wallclock budget per branch)
1. Branch A: seeded conventional VQE continuation (baseline).
2. Branch B: seeded ADAPT continuation with the same seed state and equal budget.
3. Branch C: seeded VQE variant with optimizer perturbation (method/restart behavior), same physics.

Each branch should run with:
- same initial checkpoint state source
- same heartbeat cadence
- equal fixed budget per diagnostic window (for example 45-60 min)

### Metrics to record per branch
- best `|DeltaE_abs|` trajectory
- slope windows on best `|DeltaE_abs|` vs time:
  - last 5 heartbeats
  - last 10 heartbeats
  - last 20 heartbeats
- efficiency:
  - drop in best `|DeltaE_abs|` per minute
  - drop in best `|DeltaE_abs|` per 100 objective calls
- event structure:
  - restart boundaries
  - abrupt curvature/slope regime changes
  - flat-tail onset indicators

### Decision gates
- Primary winner: highest stable decrease rate over the last 10-heartbeat window.
- Tie-break (<10% difference): choose lower-variance slope branch.
- Plateau flag: trigger if last 20-heartbeat drop `< 5e-4` and slope magnitude decreases over 3 consecutive windows.

### Operational handoff rule (overnight safety)
- If run has finite cap, extend by restarting from latest checkpoint and increasing `budget_s`.
- Extension target for this campaign: `+12h` from prior cap.
- Preserve checkpoint continuity in logs and artifacts; no physics changes during handoff.

### Expected outcome statement
- Near-zero `|DeltaE_abs|` is not assumed on current trend.
- Use measured rate windows to decide continue vs strategy shift (optimizer, ansatz parameterization, or ADAPT re-entry).

## Execution Status Snapshot (L4 HH, active campaign)

- Current active branch: seeded conventional VQE from the ADAPT-B checkpoint.
- Last checked heartbeat (UTC): `2026-03-04T05:37:27Z`.
- `|ΔE|` (current): `4.965389e-01`.
- `|ΔE|` best-so-far: `4.963192e-01`.
- Warm-seed restart summary:
  - restart 1/16 end (04:47:44Z): best `4.989841e-01`
  - restart 2/16 end (05:08:59Z): best `4.963192e-01`
  - restart 3/16 end (05:35:27Z): best `4.963192e-01`
  - restart 4/16 currently running (05:37:27Z).
- Latest crossover metric is near-flat best-trajectory slope; current run is in the low-improvement regime.

#### Live slope diagnostics
- Best-trajectory slope over last 5 heartbeats: effectively `0 /s` (best has not improved since 05:08:59).
- Current trajectory `|ΔE|` slope over last 5 heartbeats: `-4.9e-06 /s`.
- Approx. best `|ΔE|` gain from first heartbeat to now: `~2.67e-03` over `~74 min` (`~6.1e-7 /s`).

#### Relevant artifacts
- Running VQE log: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_parallel_vqe_from_adaptB_ext12h_20260303_222230_vqe_heavy.log`
- Last checkpoint state: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_parallel_vqe_from_adaptB_ext12h_20260303_222230_vqe_checkpoint_state.json`
- Seed checkpoint used: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_resume_adaptB_20260303_111311_adapt_B_B_probe_checkpoint_state.json`
- Summary index: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/hh_runs/L4_HH_20260303_poolA_stop_poolB_continue/L4_HH_overall_results.md`


- Confusion matrix
- Inner optimizer improvements, such as maybe Bayesian inference .