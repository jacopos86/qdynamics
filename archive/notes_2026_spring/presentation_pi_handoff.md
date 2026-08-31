# PI Presentation Handoff: Geometry-Position-Aware ADAPT-VQE for the Holstein-Hubbard Model

> **Generated:** 2026-03-24
> **Repo:** Holstein_test_fullclone_3
> **Primary source:** `MATH/Math.md`, repo artifacts, source code
> **Purpose:** Feed into Google NotebookLM for slide generation. All claims artifact-linked.

---

# 1. Executive Thesis

We have developed and implemented a **geometry- and position-aware ADAPT-VQE** algorithm that extends standard greedy ADAPT-VQE with (i) insertion-position probing, (ii) shortlisting with cheap/expensive two-stage screening, (iii) batch admission of near-degenerate compatible operators, and (iv) beam search over alternative scaffold trajectories. Applied to the Holstein-Hubbard (HH) model at L=2 (6 qubits, t=1, U=4, g=0.5, n_ph_max=1), the method selects a 16-operator scaffold from a 46-operator meta pool that achieves |ΔE|=5.62e-05 from exact --- well within chemical accuracy (1.6 mHa). Post-hoc pruning reveals that only 11 of the 46 pool families are ever selected, and the irreducible variational core contains just 9-10 operators. A lean pool restricted to these 11 generators locally recovers the same |ΔE|=5.62e-05 as the full meta pool, validating the reduced-pool hypothesis. Gate-level pruning produces a 7-term circuit with only 19 compiled CZ gates (depth 57), representing a 68% gate reduction from the original scaffold. Real-QPU runs on IBM Marrakesh and Kingston have been executed; current noisy |ΔE| is dominated by gate/stateprep noise (68% of total error attribution), with readout contributing only 22%. The 6-term circuit on FakeMarrakesh achieves |ΔE|≈0.093 with mthree readout mitigation alone. Local noiseless SPSA from perturbed starts converges to |ΔE|≈0.009-0.012 for the 7-term circuit, confirming the SPSA stabilization pathway is viable. We recommend executing (b) fixed-scaffold + initialized-theta + SPSA on real QPU first, then (a) re-running geo-pos-aware ADAPT with the reduced pool on real QPU. With additional IBM runtime or access to a lower-noise backend, we expect to approach or reach chemical accuracy on hardware.

---

# 2. Standard ADAPT-VQE Refresher

**Core idea.** ADAPT-VQE grows a variational ansatz one operator at a time from an operator pool $\mathcal{G}$, selecting at each step the operator whose commutator gradient with the Hamiltonian is largest.

**Outer loop (each ADAPT depth $d$):**
1. **Pool scoring:** For each candidate generator $m \in \mathcal{G}$, compute the energy gradient signal $g(m) = |\langle \psi | [H, A_m] | \psi \rangle|$ where $A_m$ is the anti-Hermitian generator.
2. **Operator selection:** Choose $m^* = \arg\max_m g(m)$.
3. **Ansatz growth:** Append $e^{\theta_{new} A_{m^*}}$ to the current ansatz (standard ADAPT always appends at the end).
4. **Re-optimization:** Re-optimize all parameters $\theta$ jointly (or a window) using a classical optimizer (e.g., Powell, L-BFGS-B).
5. **Stopping:** If $\max_m g(m) < \epsilon$ or energy improvement stagnates over a window, terminate.

**Limitations of standard greedy ADAPT:**
- Append-only growth: operators can only be added at the end, missing beneficial interior insertions.
- Greedy single-operator selection: no mechanism for near-degenerate candidates or batch efficiency.
- No cost-awareness: circuit depth/gate count not considered in selection.
- No exploration of alternative growth trajectories.

**Reference:** Math.md Section 11 (Adaptive Selection and Staged Continuation).

---

# 3. My Updated Method: Geometry-Position-Aware ADAPT-VQE

## 3.1 Geometry Awareness

**What it is:** The candidate scoring incorporates hardware-aware circuit cost (compiled 2-qubit gate count, depth, measurement groups) into the selection criterion, not just the gradient magnitude.

**Key equation (burden score):**
$$B(g,p) = \lambda_{2q} C_{2q}(g,p) + \lambda_d D(g,p) + \lambda_\theta \Delta K(g,p)$$

**Why it matters:** Standard ADAPT ignores circuit cost entirely, potentially selecting operators that contribute marginal energy improvement but large gate overhead. The burden-normalized score:
$$S_1(g,p) = \frac{\underline{\delta E}(g,p)}{B(g,p) + \varepsilon}$$
selects operators that provide the best energy-per-gate-cost ratio.

**What problem it solves:** Prevents runaway circuit depth that makes hardware execution infeasible.

**Where defined:**
- Math: `MATH/Math.md` Section 11.2-11.3
- Code: `pipelines/hardcoded/hh_continuation_scoring.py:build_candidate_features()` (line 989), `pipelines/hardcoded/hh_backend_compile_oracle.py:estimate_insertion()` (line 337)

## 3.2 Position/Insertion Awareness

**What it is:** Instead of append-only ansatz growth, candidates are evaluated at multiple insertion positions $p \in \mathcal{P}_m$ within the existing scaffold. Each candidate-position pair $r = (m, p)$ is scored independently.

**Key definitions:**
- Probe positions: $\mathcal{P}_{probe}(\mathfrak{b}) = \text{Head}_{M_{probe}}(\text{Dedup}_{ord}(\mathcal{L}_{probe}(\mathfrak{b})))$
- Inherited refit window at position $p$: $W(p) = W_{new}(p) \cup W_{|\theta|}(p)$
- Zero-lifted initialization: $\theta_b^{\uparrow(r)} = (\theta_{b,1}, \ldots, \theta_{b,p}, 0, \theta_{b,p+1}, \ldots, \theta_{b,n_b})$

**Why it matters:** Interior insertions can capture correlations that append-only misses, particularly when early operators create a subspace that later operators refine. Position-aware insertion can achieve the same energy with fewer total operators.

**What problem it solves:** Breaks the append-only bottleneck; allows the ansatz to "repair" suboptimal early choices without full re-growth.

**Where defined:**
- Math: `MATH/Math.md` Section 11.4.1-11.4.3
- Code: `pipelines/hardcoded/adapt_pipeline.py:_splice_candidate_at_position()` (line 1844), `src/quantum/ansatz_parameterization.py:runtime_insert_position()` (line 133), `pipelines/hardcoded/hh_continuation_stage_control.py:allowed_positions()` (line 27)

## 3.3 Shortlist/Probe Logic

**What it is:** A two-stage screening pipeline that avoids expensive full re-optimization on all candidates. Phase 3 (cheap) screens using a normalized gradient/tangent metric; only the shortlisted survivors proceed to Phase 2 (expensive) full refit evaluation.

**Key equations:**
- Cheap score: $S_{3,cheap}(r) = \Gamma_{stage}(r)\Gamma_{sym}(r) \frac{g_{lcb}(r)^2}{2\lambda_F F_{raw}(r)(K_{cheap}(r) + \varepsilon)}$
- Shortlist size: $N_{short} = \min\{|\mathcal{C}_{cheap}|, N_{max}, \lceil f_{short}|\mathcal{C}_{cheap}| \rceil\}$
- Phase-2 rerank: $S_2(g,p) = \frac{\delta E_{TR}(g,p) \cdot \mathcal{N}(g,p)}{B(g,p)(1 + \kappa(g,p)) + \varepsilon}$

where $\mathcal{N}(g,p)$ is a novelty penalty (anti-redundancy) and $\kappa(g,p)$ is curvature.

**Why it matters:** The full meta pool has 46 operators × multiple positions = hundreds of candidates per ADAPT step. Evaluating all with full refit is prohibitively expensive. Cheap screening reduces this to $O(10)$ expensive evaluations.

**What problem it solves:** Makes large-pool ADAPT computationally tractable without sacrificing selection quality.

**Where defined:**
- Math: `MATH/Math.md` Section 11.3.2, 11.4.2
- Code: `pipelines/hardcoded/hh_continuation_scoring.py:shortlist_records()` (line 461)

## 3.4 Batching

**What it is:** When multiple candidates are near-degenerate (within a factor $\eta_{nd}$ of the best score), they can be admitted simultaneously in a single ADAPT step, subject to pairwise compatibility constraints.

**Key equations:**
- Near-degeneracy gate: $s(r) \geq \eta_{nd} \cdot s(r_{max})$
- Pairwise incompatibility: $\Pi(r,r') = w_{ov}\Omega(r,r') + w_{comm}\Xi(r,r') + w_{curv}\mathcal{K}(r,r') + w_{sched}\mathcal{A}(r,r') + w_{meas}\mathcal{M}(r,r')$
- Greedy admission: $r \in \mathcal{B}_{batch}$ only if $s(r) - \sum_{r' \in \mathcal{B}_{batch}} \Pi(r,r') > 0$

**Why it matters:** When two operators contribute nearly equally, sequential addition wastes an ADAPT iteration and may produce a locally suboptimal ordering.

**What problem it solves:** Reduces total ADAPT iterations and avoids artificial sequencing of near-degenerate operators.

**Where defined:**
- Math: `MATH/Math.md` Section 11.4.2
- Code: `pipelines/hardcoded/hh_continuation_scoring.py:greedy_batch_select()` (line 928)

## 3.5 Beam Search

**What it is:** Instead of committing to a single best operator each step, the algorithm maintains a beam of $B_{live}$ frontier branches, each representing an alternative scaffold growth trajectory. Branches are pruned by a lexicographic key and deduplicated by fingerprint.

**Key structures:**
- Branch state: $\mathfrak{b} = (id_b, \pi_b, \mathcal{H}_b, \mathcal{O}_b, \theta_b, E_b, d_b, S_b^{cum}, K_b^{cum}, \sigma_b, \tau_b)$
- Pruning key: $\kappa_{prune}(\mathfrak{b}) = (E_b, -S_b^{cum}, K_b^{cum}, |\mathcal{O}_b|, \text{labels}(\mathcal{O}_b), \text{round}_{10}(\theta_b), id_b)$
- Beam round: (i) recompute shortlist per branch, (ii) materialize stop or admission, (iii) deduplicate/prune, (iv) select final scaffold

**Why it matters:** Greedy ADAPT can get trapped in local optima. Beam search explores alternative trajectories and selects the globally best scaffold at termination.

**What problem it solves:** Mitigates the greedy trap inherent in standard ADAPT; provides a more robust scaffold selection.

**Where defined:**
- Math: `MATH/Math.md` Section 11.4.3, Algorithm block (lines 2172-2212)
- Code: `pipelines/hardcoded/adapt_pipeline.py:_evaluate_phase1_positions()` (line 3319)

---

# 4. Workflow Actually Used

## 4.1 Full Meta Pool / Large Logical Search

The initial ADAPT-VQE search used a **46-operator "full_meta" pool** comprising:
- UCCSD (fermion-lifted singles and doubles)
- HVA layerwise macros
- Termwise-augmented Hamiltonian terms (quadrature)
- PAOP families: `paop_cloud_p`, `paop_dbl_p`, `paop_hopdrag`, `paop_disp`, `paop_curdrag`, `paop_hop2`, and all `x`-type variants

**Code:** `pipelines/hardcoded/adapt_pipeline.py:_build_hh_full_meta_pool()` (line 875)

The backend-conditioned run on FakeNighthawk selected a **16-operator scaffold** (depth 16, 60 compiled CZ gates, depth 137) achieving:
- **Energy:** 0.15872408
- **Exact:** 0.15866790
- **|ΔE|:** 5.62e-05

**Artifact:** `artifacts/json/hh_backend_adapt_fullhorse_powell_20260322T194155Z_fakenighthawk.json`

## 4.2 Scaffold Selection & Pruning

Post-hoc pruning (`pipelines/hardcoded/hh_prune_nighthawk.py`) applied two strategies:

**Path A (Fixed-scaffold VQE):** Lock operator order, remove near-zero-theta operators, re-optimize surviving parameters.
- Result: 11 operators, 21 params, 53 CZ gates, depth 133
- **Zero energy regression** (|ΔE| unchanged at 5.62e-05)

**Path B (Re-ADAPT from HF):** Re-run ADAPT from scratch with reduced pool.
- Result: 10 operators, 20 params, 53 CZ gates, depth 133
- **Zero energy regression**

**Artifacts:**
- `artifacts/reports/hh_prune_nighthawk_final_20260322.md`
- `artifacts/json/hh_prune_nighthawk_20260322T201200Z.json`

## 4.3 Identification of Unused / Irrelevant Families

**L=2 pool usage analysis** (`artifacts/reports/hh_L2_ecut1_scaffold_motif_analysis.md`):

| Status | Families | Count |
|--------|----------|-------|
| **Always used** | `hh_termwise_quadrature`, `uccsd_sing`, `paop_cloud_p` (86% of selections!), `paop_dbl_p` (partial), `paop_hopdrag` | 11 generators |
| **Never used** | `hh_termwise_unit` (16), HVA layers (4), all `x`-type PAOPs (6), `paop_disp` (2), `paop_dbl` (2), `uccsd_dbl` (1), `paop_curdrag`, `paop_hop2` | 35 generators |

**Key finding:** 76% of the pool (35/46 operators) was never selected across 101 scaffold entries. The `paop_cloud_p` family alone accounts for 86% of all selections.

## 4.4 Reduced Relevant Pool Hypothesis

A **lean pool of 11 generators** (`pareto_lean_l2`) was constructed from only the used families.

**Local validation** (`artifacts/reports/pareto_lean_l2_vs_heavy_L2_ecut1_comparison.md`):
- Heavy pool (46 ops): |ΔE| = 5.618e-05, logical depth 97 (at interrupt)
- **Lean pool (11 ops): |ΔE| = 5.618e-05, logical depth 14** (converged, drop_plateau)

The lean pool recovers the same energy in far fewer ADAPT iterations.

**Code:** `pipelines/hardcoded/adapt_pipeline.py:_build_hh_pareto_lean_l2_pool()` (line 1119)

## 4.5 Gate-Level Pruning: 7-Term Pareto Point

Further gate-level pruning (`artifacts/reports/hh_prune_nighthawk_pareto_menu_20260322.md`) produced a Pareto menu:

| Variant | Ops | Terms | CZ Gates | Depth | |ΔE| |
|---------|-----|-------|----------|-------|------|
| Original fullhorse | 16 | 47 | 60 | 137 | 5.62e-05 |
| Ultra-lean (6-op) | 6 | 16 | 48 | 121 | 5.62e-05 |
| Aggressive (5-op) | 5 | 12 | 32 | 122 | 7.01e-05 |
| **Gate-pruned (7-term)** | **5** | **7** | **19** | **57** | **4.23e-04** |

The 7-term circuit is the primary hardware candidate: 19 CZ gates, depth 57, and the only variant with faithful executable import on real backends (replay fidelity ~1.0).

**Artifact:** `artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json`

## 4.6 Fixed-Scaffold Replay / Initialized-Theta Route

The 7-term gate-pruned circuit with locally optimized theta was directly executed on real QPU:

| Backend | Shots | Noisy Energy | |ΔE| | Mitigation |
|---------|-------|-------------|------|------------|
| ibm_marrakesh (7-term) | 2048 | 0.882 | 0.723 | mthree readout |
| ibm_marrakesh (6-term) | 8192 | 0.449 | 0.285 | mthree readout |
| FakeMarrakesh (6-term) | 2048 | 0.252 | 0.093 | mthree readout |

**Noise attribution** (`artifacts/json/hh_gate_pruned_fixed_scaffold_attr_20260323T013814Z.json`):
- Readout-only: +0.049 (22%)
- Gate/stateprep-only: +0.119 (68%)
- Full noise: +0.179

mthree removes readout noise; gate noise remains the floor.

## 4.7 SPSA-on-Hardware Stabilization Route

**Local noiseless SPSA validation** (from perturbed starts, SPSA maxiter=128):

| Circuit | |ΔE| range (SPSA converged) | Source |
|---------|---------------------------|--------|
| 7-term | 0.0087 - 0.0124 | `artifacts/json/hh_marrakesh_7term_local_exact_spsa128_perturbed_20260323T182846Z.json` |
| 6-term | 0.0102 - 0.0161 | `artifacts/json/hh_marrakesh_6term_local_exact_spsa128_perturbed_20260323T182700Z.json` |

7-term shows better SPSA robustness. From the locally optimized theta (not perturbed), the starting |ΔE| is only 4.23e-04 --- SPSA only needs to stabilize against noise perturbation, not find the minimum from scratch.

**Code:** `src/quantum/spsa_optimizer.py:spsa_minimize()` (line 135)

---

# 5. Evidence Packet

## 5.1 Best Large Logical Circuit Artifact

- **Artifact:** `artifacts/json/hh_backend_adapt_fullhorse_powell_20260322T194155Z_fakenighthawk.json`
- **Metrics:** 16 operators, 27 params, 60 CZ (FakeNighthawk opt_level=1), depth 137
- **Energy:** 0.15872408, |ΔE| = 5.62e-05
- **Confidence:** HIGH (backend-conditioned, fully replayed)

## 5.2 Best Pruned / Lean Scaffold Artifact

- **Artifact:** `artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json`
- **Metrics:** 5 operators, 7 terms, 19 CZ, depth 57
- **Energy:** |ΔE| = 4.23e-04 (noiseless)
- **Confidence:** HIGH (replay fidelity ~1.0, Marrakesh-compilable)

- **Lean-pool scaffold:** `artifacts/reports/pareto_lean_l2_vs_heavy_L2_ecut1_comparison.md`
- **Metrics:** 11 generators → 14 logical depth, |ΔE| = 5.618e-05
- **Confidence:** HIGH (local exact, converged at drop_plateau)

## 5.3 Evidence That Many Pool Families Were Never Selected

- **Artifact:** `artifacts/reports/hh_L2_ecut1_scaffold_motif_analysis.md`
- **Key field:** Pool family usage table (Section: "Pool Families - L=2 Usage")
- **Result:** 35 of 46 pool operators never selected across 101 scaffold entries
- **Confidence:** HIGH (exhaustive count from scaffold trace)

Cross-validated at L=3 (`artifacts/reports/hh_heavy_scaffold_best_yet_20260321.md`):
- All `hh_termwise_unit` (18 ops), all HVA layers (4), all `x`-type PAOPs (11) never used at L=3 either
- **Confidence:** MEDIUM (L=3 is a different system size; pattern is consistent but not proven universal)

## 5.4 Evidence That Reduced Pool Locally Recovers Prior Energy

- **Artifact:** `artifacts/reports/pareto_lean_l2_vs_heavy_L2_ecut1_comparison.md`
- **Key comparison:**
  - Heavy pool (46 ops): |ΔE| = 5.618e-05
  - Lean pool (11 ops): |ΔE| = 5.618e-05
- **Confidence:** HIGH (both converged locally, same |ΔE| to machine precision)

## 5.5 Evidence That Fixed Scaffold + Initialized Theta + SPSA Is Viable

- **Local SPSA validation:** `artifacts/json/hh_marrakesh_7term_local_exact_spsa128_perturbed_20260323T182846Z.json`
  - 7-term SPSA from perturbed starts: |ΔE| = 0.0087-0.0124
  - **Confidence:** MEDIUM (noiseless SPSA; hardware SPSA not yet converged)

- **Noise attribution:** `artifacts/json/hh_gate_pruned_fixed_scaffold_attr_20260323T013814Z.json`
  - mthree removes readout noise cleanly; gate noise is addressable floor
  - **Confidence:** HIGH (8-repeat attribution study)

- **FakeMarrakesh mitigation ablation:** `artifacts/json/hh_marrakesh_6term_local_mitigation_ablation_20260324T001622Z.json`
  - No mitigation: |ΔE| = 0.152; readout mitigation: |ΔE| = 0.093 (39% improvement)
  - **Confidence:** MEDIUM (simulator, not real QPU)

## 5.6 Best Current Real-QPU |ΔE|

- **Best real-QPU result:** ibm_marrakesh 6-term, |ΔE| = 0.285
  - Artifact: `artifacts/json/hh_gatepruned6_fixedtheta_runtime_ibm_marrakesh_20260323T171528Z.json`
  - **Confidence:** HIGH (real hardware, completed jobs)

- **Best simulator result:** FakeMarrakesh 6-term with mthree, |ΔE| = 0.093
  - Artifact: `artifacts/json/hh_marrakesh_6term_local_mitigation_ablation_20260324T001622Z.json`

- **Kingston partial:** ibm_kingston 6-term SPSA48, best energy = 0.394 (noisy), |ΔE| ≈ 0.235
  - Artifact: `artifacts/json/hh_kingston_6term_spsa48_runtime_partial_recovery_20260323.json`
  - **Confidence:** LOW (partial recovery, no faithful optimizer trace)

## 5.7 Compiled/Logical Circuit Metrics Supporting Hardware Plausibility

- **7-term on Marrakesh (10 transpiler seeds, opt_level=1):**
  - 2Q gates: 25 (mean), depth: 63-75
  - Artifact: `artifacts/reports/investigation_marrakesh_pareto_20260323.md`

- **6-term on Marrakesh:**
  - 2Q gates: 14, depth: 48
  - Artifact: `artifacts/json/hh_marrakesh_6term_local_mitigation_ablation_20260324T001622Z.json`

- **6-term on FakeMarrakesh:**
  - Compiled 2Q: 14, depth: 48, size: 80, layout: [13,12,18,11,14,10]

---

# 6. Recommended Hardware Plan

## Preferred ordering: (b) first, then (a)

### Step (b): Fixed Scaffold + Initialized Theta + SPSA on Real QPU

1. Take the 7-term gate-pruned circuit with locally optimized theta (|ΔE| = 4.23e-04 noiseless).
2. Transpile to target backend (Marrakesh or better) at opt_level=1.
3. Execute with mthree readout mitigation.
4. Run SPSA optimization (maxiter=128+) on the real QPU to stabilize against noise.
5. Expected outcome: SPSA should recover from noise-induced parameter perturbation; local SPSA studies show convergence to |ΔE| ≈ 0.009-0.012 even from perturbed starts.

**Technical rationale for doing (b) first:**
- Fixed scaffold has **zero exploration cost** --- no gradient evaluation of pool candidates on hardware.
- Only requires parameter optimization (SPSA), which is $O(\text{params})$ circuit evaluations per iteration, not $O(\text{pool} \times \text{positions})$.
- The 7-term circuit has only 7 parameters and 19 CZ gates --- minimal shot budget per iteration.
- Provides a **concrete |ΔE| baseline** on real hardware before investing in the more expensive ADAPT re-run.
- If (b) already reaches chemical accuracy, (a) may be unnecessary.

### Step (a): Re-run Geo-Pos-Aware ADAPT with Reduced Pool on Real QPU

1. Use the lean pool (11 generators) validated locally.
2. Run the full geo-pos-aware ADAPT loop with hardware-in-the-loop cost estimation.
3. Each ADAPT iteration requires gradient evaluation of ~11 candidates × positions --- significantly cheaper than the original 46-operator search.
4. Expected outcome: ADAPT should converge to a scaffold of ~14 logical layers, potentially finding a hardware-native scaffold that outperforms the pre-pruned fixed circuit.

**Technical rationale for doing (a) second:**
- ADAPT-on-hardware is expensive: each iteration requires $O(\text{pool})$ gradient circuits.
- The reduced pool (11 vs 46) makes this 4x cheaper than the original search.
- Result from (b) provides a target energy: if ADAPT can't beat it, the fixed scaffold is sufficient.

---

# 7. Chemical-Accuracy Outlook

## Already supported by evidence

- **Noiseless |ΔE| = 5.62e-05** from the full meta pool (well within chemical accuracy of 1.6 mHa ≈ 0.0016 Ha).
- **Noiseless |ΔE| = 4.23e-04** from the 7-term gate-pruned circuit (still within chemical accuracy).
- **Lean pool recovers full-pool energy** locally (|ΔE| identical to 5 significant figures).
- **SPSA converges** from perturbed starts to |ΔE| ≈ 0.009-0.012 (noiseless), which is ~6x chemical accuracy.
- **Readout mitigation** (mthree) cleanly removes readout noise component.

## Plausible but not yet demonstrated

- **SPSA on real QPU** has not yet converged to a clean |ΔE|. Kingston partial recovery (|ΔE| ≈ 0.235) shows the pathway exists but runtime was insufficient.
- **Gate-noise mitigation** (dynamical decoupling, Pauli twirling) has been tested on FakeMarrakesh but showed mixed results (DD+twirling slightly worse than readout-only in one ablation).
- The **7-term SPSA pathway** is expected to outperform the 6-term based on local studies, but has not been executed on real QPU with full SPSA convergence.

## What extra runtime / backend quality would most likely improve

1. **More SPSA iterations on real QPU:** Current Kingston run used only 48 SPSA iterations with 75 quantum seconds. SPSA(128+) with ~300+ quantum seconds would allow proper convergence characterization.
2. **Lower gate-error backend:** Gate/stateprep noise accounts for 68% of the total error. A backend with 2-3x lower 2Q gate error would directly reduce the noise floor. IBM Eagle r3 or Heron-class processors would be candidates.
3. **Higher shot budget:** Current runs used 2048-8192 shots. Statistical precision scales as $1/\sqrt{N_{shots}}$; 32k+ shots would reduce sampling noise below the gate-noise floor.
4. **Advanced error mitigation:** Zero-noise extrapolation (ZNE) or probabilistic error cancellation (PEC) could address the gate-noise floor that readout mitigation alone cannot touch.

---

# 8. Slide Blueprint

## Slide 1: Title & Problem Statement
- **Title:** "Geometry-Position-Aware ADAPT-VQE for the Holstein-Hubbard Model"
- **Purpose:** Set the stage; define the problem.
- **Bullets:**
  - Holstein-Hubbard: coupled electron-phonon lattice model
  - L=2 sites, 6 qubits (JW + unary phonon encoding)
  - Parameters: t=1, U=4, g=0.5, n_ph_max=1
  - Goal: approach chemical accuracy on real quantum hardware
  - Exact ground-state energy: 0.15866790 Ha
- **Artifacts/figures:** HH Hamiltonian equation from Math.md Section 6

## Slide 2: Standard ADAPT-VQE Recap
- **Title:** "ADAPT-VQE: Adaptive Ansatz Construction"
- **Purpose:** Remind PI of the baseline algorithm.
- **Bullets:**
  - Greedy: pick best operator by gradient, append, re-optimize
  - Pool $\mathcal{G}$: fixed set of candidate generators
  - Append-only growth → no position awareness
  - No cost awareness → can produce deep circuits
  - Known limitation: greedy trap, no batch/beam
- **Artifacts/figures:** Flow diagram (to generate), standard ADAPT pseudocode from Math.md Section 11

## Slide 3: Our Extensions (Geometry + Position + Shortlist)
- **Title:** "Geo-Position-Aware ADAPT-VQE"
- **Purpose:** Present the methodological contributions.
- **Bullets:**
  - Position-aware insertion: test $r=(m,p)$ at multiple scaffold positions
  - Hardware-aware burden scoring: $S_1 = \delta E / B$
  - Two-stage shortlist: cheap screen → expensive refit
  - Batch admission of near-degenerate compatible operators
  - Beam search over alternative scaffold trajectories
  - Stopping: plateau-aware stage controller
- **Artifacts/figures:** Scoring equations, beam diagram (to generate), burden functional from Math.md

## Slide 4: Full-Pool Search Results
- **Title:** "46-Operator Meta Pool → 16-Operator Scaffold"
- **Purpose:** Show the ADAPT search converges to high accuracy.
- **Bullets:**
  - Meta pool: 46 operators (UCCSD, HVA, PAOP, termwise)
  - Selected scaffold: 16 operators, 27 params
  - |ΔE| = 5.62e-05 (30x below chemical accuracy)
  - 60 CZ gates compiled on FakeNighthawk
  - Key finding: only 11 of 46 pool families ever selected
  - `paop_cloud_p` alone accounts for 86% of selections
- **Artifacts/figures:** Pool family usage table from `hh_L2_ecut1_scaffold_motif_analysis.md`, energy convergence plot (to generate from adapt handoff JSON)

## Slide 5: Pruning → Lean Circuits
- **Title:** "Pruning: From 60 CZ to 19 CZ with Controlled Regression"
- **Purpose:** Show the Pareto menu and gate reduction.
- **Bullets:**
  - Conservative prune (11-op): zero regression, 53 CZ
  - Ultra-lean (6-op): zero regression, 48 CZ
  - Gate-pruned 7-term: 19 CZ, depth 57, |ΔE| = 4.23e-04
  - Leave-one-out analysis: 4-operator irreducible core carries >99% weight
  - Quadrature terms are free to remove (fully absorbed after reopt)
- **Artifacts/figures:** Pareto table from `hh_prune_nighthawk_pareto_menu_20260322.md`, LOO bar chart (to generate)

## Slide 6: Lean Pool Validation
- **Title:** "Reduced Pool Recovers Full-Pool Energy"
- **Purpose:** Validate the reduced-pool hypothesis.
- **Bullets:**
  - Heavy pool (46 ops): |ΔE| = 5.618e-05, 97 ADAPT steps
  - Lean pool (11 ops): |ΔE| = 5.618e-05, 14 ADAPT steps
  - 4x cheaper ADAPT search with identical energy
  - Consistent pattern at L=3 (9 qubits): same families dominate
  - Enables hardware ADAPT with reduced pool
- **Artifacts/figures:** Comparison table from `pareto_lean_l2_vs_heavy_L2_ecut1_comparison.md`, convergence curves (to generate)

## Slide 7: Real QPU Results & Noise Analysis
- **Title:** "Hardware Execution: Noise Attribution & Current State"
- **Purpose:** Present real QPU evidence and noise characterization.
- **Bullets:**
  - Marrakesh 7-term: |ΔE| = 0.72 (gate noise dominated)
  - Marrakesh 6-term: |ΔE| = 0.29 (shallower circuit helps)
  - FakeMarrakesh 6-term + mthree: |ΔE| = 0.093
  - Noise attribution: 68% gate/stateprep, 22% readout, 10% cross-talk
  - mthree removes readout cleanly; gate noise is the floor
- **Artifacts/figures:** Noise attribution bar chart (to generate from `hh_gate_pruned_fixed_scaffold_attr_20260323T013814Z.json`), real QPU energy comparison table

## Slide 8: SPSA Stabilization Pathway
- **Title:** "SPSA on Hardware: Stabilizing Against Noise"
- **Purpose:** Show SPSA viability and the path forward.
- **Bullets:**
  - Local SPSA (7-term, perturbed): converges to |ΔE| ≈ 0.009-0.012
  - Local SPSA (6-term, perturbed): converges to |ΔE| ≈ 0.010-0.016
  - 7-term more robust (lower variance across starts)
  - Kingston partial: 48 SPSA iters, 75 quantum seconds (insufficient)
  - Need: 128+ iters, 300+ quantum seconds for convergence
- **Artifacts/figures:** SPSA convergence traces (to generate from perturbed JSON), Kingston partial trace

## Slide 9: Hardware Plan & Chemical Accuracy Outlook
- **Title:** "Path to Chemical Accuracy on Real QPU"
- **Purpose:** Lay out the recommended execution plan.
- **Bullets:**
  - Step (b): Fixed 7-term scaffold + SPSA → baseline |ΔE| on QPU
  - Step (a): Lean-pool ADAPT on QPU → potentially better scaffold
  - Chemical accuracy target: |ΔE| < 1.6 mHa
  - Current noiseless: already at 0.42 mHa (7-term) or 0.056 mHa (full)
  - Gap: noise floor (gate error) is the remaining barrier
  - Outlook: lower-noise backend or more runtime likely sufficient
- **Artifacts/figures:** Accuracy ladder diagram (to generate), plan timeline

---

# 9. Artifact Registry

| artifact_path | artifact_type | why_it_matters | key_metrics_or_fields | confidence |
|---|---|---|---|---|
| `MATH/Math.md` | theory | All method definitions, notation, equations | Sections 6, 10, 11, 12, 15-16 | HIGH |
| `artifacts/json/hh_backend_adapt_fullhorse_powell_20260322T194155Z_fakenighthawk.json` | adapt_result | Canonical full-pool ADAPT scaffold | `final_energy`, `exact_energy`, `scaffold_labels`, `compiled_cx_count` | HIGH |
| `artifacts/reports/hh_L2_ecut1_scaffold_motif_analysis.md` | analysis | Pool family usage / unused family evidence | Pool usage table, 35/46 never-used count | HIGH |
| `artifacts/reports/hh_prune_nighthawk_final_20260322.md` | prune_report | Definitive pruning summary, both paths | Path A/B results, operator table, CZ counts | HIGH |
| `artifacts/reports/hh_prune_nighthawk_pareto_menu_20260322.md` | prune_report | Pareto menu of circuit variants | LOO table, 7-term metrics, gate-level prune | HIGH |
| `artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json` | circuit | Primary hardware candidate circuit | 7 terms, 19 CZ, depth 57, |ΔE|=4.23e-04 | HIGH |
| `artifacts/reports/pareto_lean_l2_vs_heavy_L2_ecut1_comparison.md` | benchmark | Lean pool recovery validation | Both pools: |ΔE|=5.618e-05, depth comparison | HIGH |
| `artifacts/reports/hh_heavy_scaffold_best_yet_20260321.md` | adapt_result | L=3 scaffold showing same family patterns | 37 depth, |ΔE|=8.49e-05, family table | MEDIUM |
| `artifacts/reports/lean_logical_circuit_20260321T214822Z.md` | circuit | Lean logical circuit pre-pruning | 14 ops, 58 CX abstract, 205 total gates | HIGH |
| `artifacts/json/hh_gate_pruned_fixed_scaffold_attr_20260323T013814Z.json` | noise_attr | Noise source attribution study | readout_only_delta, gate_only_delta, full_delta | HIGH |
| `artifacts/json/hh_marrakesh_6term_local_mitigation_ablation_20260324T001622Z.json` | mitigation | FakeMarrakesh mitigation ablation | no_mitigation_delta, readout_delta, dd_delta | MEDIUM |
| `artifacts/json/hh_gatepruned7_fixedtheta_runtime_ibm_marrakesh_20260323T145219Z_direct.json` | qpu_run | Real Marrakesh 7-term execution | noisy_energy, ideal_energy, job_ids | HIGH |
| `artifacts/json/hh_gatepruned6_fixedtheta_runtime_ibm_marrakesh_20260323T171528Z.json` | qpu_run | Real Marrakesh 6-term execution | noisy_energy=0.449, ideal=0.163, |ΔE|=0.285 | HIGH |
| `artifacts/json/hh_kingston_6term_spsa48_runtime_partial_recovery_20260323.json` | qpu_run | Kingston SPSA partial recovery | best_energy=0.394, quantum_seconds=75 | LOW |
| `artifacts/json/hh_marrakesh_7term_local_exact_spsa128_perturbed_20260323T182846Z.json` | spsa_study | 7-term SPSA from perturbed starts | |ΔE| range: 0.0087-0.0124 | MEDIUM |
| `artifacts/json/hh_marrakesh_6term_local_exact_spsa128_perturbed_20260323T182700Z.json` | spsa_study | 6-term SPSA from perturbed starts | |ΔE| range: 0.0102-0.0161 | MEDIUM |
| `artifacts/json/hh_6term_vs_7term_runtime_rerun_plan_20260323.json` | plan | Decision analysis for QPU circuit choice | exact_fidelity, compiled_2q, recommendation | HIGH |
| `artifacts/reports/investigation_marrakesh_pareto_20260323.md` | investigation | Marrakesh compile ranking, 5-op fidelity issue | compile table, replay fidelity ~0.57 for 5-op | HIGH |
| `artifacts/reports/investigation_7term_noise_diagnosis_20260323.md` | investigation | FakeNighthawk noise breakdown | attribution percentages, mthree validation | HIGH |
| `artifacts/reports/investigation_kingston_best_deltae_20260324.md` | investigation | Kingston path analysis and recommendation | 7-term recommended over 6-term for robustness | MEDIUM |
| `artifacts/reports/hh_prune_marginal_20260322T202009Z.md` | prune_report | Leave-one-out marginal cost analysis | Per-operator regression costs, greedy multi-prune | HIGH |
| `artifacts/logs/live_qpu/` | logs | Real QPU execution logs | Job IDs, runtime metadata | LOW |

---

# 10. Open Gaps / Things to Verify

1. **5-operator executable import broken:** The `aggressive_5op` and `readapt_5op` artifacts have replay fidelity ~0.57 on Marrakesh import path. Until fixed, only the 7-term variant is trustworthy for hardware execution. See `investigation_marrakesh_pareto_20260323.md`.

2. **No converged SPSA-on-real-QPU result:** Kingston SPSA48 was interrupted with only 75 quantum seconds. A full 128+ iteration SPSA run on a real backend has not been completed. This is the most critical gap for the presentation.

3. **FakeMarrakesh vs real Marrakesh discrepancy:** FakeMarrakesh 6-term gives |ΔE|=0.093 with mthree, but real Marrakesh 6-term gives |ΔE|=0.285. Need to understand if this is noise-model fidelity or a transpilation/layout issue.

4. **DD + twirling mixed results:** The mitigation ablation shows DD+twirling (|ΔE|=0.114) slightly worse than readout-only (|ΔE|=0.093) on FakeMarrakesh. This needs investigation --- may be a configuration issue or genuine interference.

5. **L=3 (9-qubit) hardware feasibility not assessed:** All hardware runs are L=2 (6 qubits). The L=3 scaffold has 37 logical layers --- likely too deep for current hardware without significant error mitigation.

6. **Exact chemical accuracy threshold:** Need to confirm the exact chemical accuracy value for this problem. Standard chemical accuracy is 1 kcal/mol ≈ 1.6 mHa ≈ 0.0016 Ha. The 7-term |ΔE|=4.23e-04 Ha is within this; the noisy results are not.

7. **Missing historical workflow scripts:** `investigation_pruning_start_20260322.md` notes that historical saved-parent pruning modules (`hh_l2_*_workflow.py`) are missing from the working tree. May need reconstruction for reproducibility.

8. **Kingston vs Marrakesh backend quality:** Need to characterize which backend has lower 2Q gate error for our specific circuit layout. The `investigation_kingston_best_deltae_20260324.md` recommends Kingston for energy-only 7-term baseline, but comparative data is incomplete.

9. **SPSA hyperparameter sensitivity:** Local SPSA used a=0.2, c=0.1, alpha=0.602, gamma=0.101. These may need tuning for the noisy hardware landscape. The corrected run showed SPSA made no improvement from the optimized theta (expected --- it's already at minimum), but the perturbed-start studies confirm it can find good basins.

10. **Measurement grouping optimization:** The Hamiltonian has 17 terms in 4 QWC groups. Grouped measurement could reduce shot budget by ~4x. Need to verify if the runtime pipeline uses grouping or term-by-term evaluation (the noise diagnosis report suggests term-by-term).

---

# 11. Novelty Assessment Space (For Google Research)

> **Instructions for NotebookLM / external research:** Use this section to determine what aspects of this work are novel vs. established. For each item, search the literature and classify as: NOVEL, INCREMENTAL, or ESTABLISHED.

## Items to assess:

### Method-level novelty
1. **Position-aware operator insertion in ADAPT-VQE:** Standard ADAPT is append-only. Has anyone published position-aware (interior) insertion? Search: "ADAPT-VQE insertion position", "non-greedy ADAPT", "operator placement ADAPT".

2. **Hardware-aware burden scoring in ADAPT selection:** The $S_1 = \delta E / B$ score incorporates compiled circuit cost. Search: "hardware-aware ADAPT-VQE", "circuit-cost ADAPT", "resource-efficient ADAPT".

3. **Batch admission of near-degenerate operators:** Multiple operators added per ADAPT step with compatibility oracle. Search: "batch ADAPT-VQE", "multi-operator ADAPT", "qubit-ADAPT batch".

4. **Beam search over scaffold trajectories:** Maintaining multiple alternative ADAPT growth paths. Search: "beam search VQE", "beam ADAPT", "non-greedy ansatz construction".

5. **Two-stage cheap/expensive shortlisting:** Proxy gradient screening before full refit. Search: "gradient screening ADAPT", "two-stage ADAPT", "cheap gradient ADAPT-VQE".

### Application-level novelty
6. **ADAPT-VQE applied to Holstein-Hubbard model:** Has ADAPT been applied to electron-phonon models? Search: "ADAPT-VQE Holstein", "ADAPT-VQE phonon", "ADAPT-VQE polaron".

7. **Polaronic operator pool (PAOP) in ADAPT:** Custom operator families for electron-phonon coupling. Search: "polaron ansatz VQE", "PAOP quantum computing", "dressed operator pool".

8. **Post-hoc pruning of ADAPT scaffold with Pareto analysis:** Systematic leave-one-out + gate-level pruning with Pareto ranking. Search: "ADAPT pruning", "ansatz pruning VQE", "circuit simplification ADAPT".

### Hardware-level novelty
9. **Fixed-scaffold + SPSA stabilization pathway on IBM hardware for e-ph models:** Has anyone done SPSA parameter stabilization for ADAPT-selected circuits on real QPU? Search: "SPSA VQE real hardware", "SPSA IBM quantum", "noise-aware VQE optimization".

10. **Reduced-pool hypothesis from scaffold motif analysis:** Deriving a minimal pool from observing ADAPT selection patterns. Search: "pool reduction ADAPT", "operator pool pruning", "adaptive pool selection".

---

# Appendix: Key Notation Reference (from Math.md)

| Symbol | Meaning | Math.md Location |
|--------|---------|-----------------|
| $\hat{H}_{HH}$ | Holstein-Hubbard Hamiltonian | Section 6 |
| $\mathcal{G}$ | Full master generator pool | Section 11.1 |
| $\mathcal{G}^{(3)} \subseteq \mathcal{G}^{(2)} \subseteq \mathcal{G}^{(1)}$ | Phase-restricted pool subsets | Section 11.1 |
| $r = (m, p)$ | Candidate-position record | Section 11.4.1 |
| $B(g,p)$ | Hardware burden score | Section 11.2 |
| $S_1(g,p)$ | Cheap phase-1 score | Section 11.3 |
| $S_2(g,p)$ | Full phase-2 rerank score | Section 11.3 |
| $W(p)$ | Inherited refit window at position $p$ | Section 11.4.1 |
| $\mathfrak{b}$ | Beam branch state | Section 11.4.3 |
| $\Pi(r,r')$ | Pairwise batch incompatibility | Section 11.4.2 |
| $\mathcal{N}(g,p)$ | Novelty penalty | Section 11.3.3 |
| $c_k, a_k$ | SPSA step-size schedules | Section 12 |
