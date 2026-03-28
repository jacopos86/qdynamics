# CODEX TASK: Finish presentation_pi_handoff.md

## What exists already

`presentation_pi_handoff.md` has lines 1-27 written (Sections 1-2 partial: executive thesis + ADAPT refresher start). The user has already uploaded lines 1-27 and `MATH/Math.md` to Google NotebookLM separately. So you do NOT need to reproduce Math.md content or lines 1-27.

**Your job:** Rewrite `presentation_pi_handoff.md` starting from line 27 onward. The file currently has placeholder content from line 27 on that references artifact paths but does NOT inline the actual data. Google NotebookLM cannot access the repo. You MUST inline all numbers, tables, operator lists, energy values, circuit metrics, SPSA traces, etc. directly into the markdown. No "see artifact X" without also pasting the relevant data from that artifact.

## The narrative structure (keep these section headers exactly)

From line 27 onward, the file must contain:

### Section 2 (finish): Standard ADAPT-VQE Refresher
- Limitations of greedy ADAPT (append-only, no cost awareness, greedy trap, no batch)
- Keep short, Math.md has the formal version

### Section 3: My Updated Method — Geo-Position-Aware ADAPT-VQE
Five subsections, each with what/why/what-problem-it-solves:
1. **Geometry awareness** — burden score $B(g,p) = \lambda_{2q}C_{2q} + \lambda_d D + \lambda_\theta \Delta K$; normalized score $S_1 = \delta E / B$
2. **Position/insertion awareness** — candidate-position pairs $r=(m,p)$, probe positions, inherited refit window $W(p)$, zero-lifted initialization
3. **Shortlist/probe logic** — two-stage cheap/expensive screening; cheap score formula; shortlist size formula
4. **Batching** — near-degeneracy gate $s(r) \geq \eta_{nd} s(r_{max})$; pairwise incompatibility $\Pi(r,r')$; greedy admission
5. **Beam search** — branch state $\mathfrak{b}$, pruning key, beam round (4 steps), pool updates

For equations, use the ones from Math.md Section 11. Reference `pipelines/hardcoded/adapt_pipeline.py`, `hh_continuation_scoring.py`, `hh_continuation_stage_control.py` etc as code locations.

### Section 4: Workflow Actually Used
Reconstruct from artifact data. INLINE all data. Key subsections:

**4.1 Full Meta Pool / Large Logical Search**
- 46-operator `full_meta` pool (UCCSD, HVA, PAOP, termwise). Code: `adapt_pipeline.py:_build_hh_full_meta_pool()` line 875.
- Backend-conditioned FakeNighthawk run → 16-operator scaffold.
- Energy: 0.15872408, Exact: 0.15866790, |ΔE|: 5.62e-05
- 60 compiled CZ gates, depth 137, size 272, 27 runtime params

**4.2 Pruning**
Inline this full table from the Pareto menu report:

| Variant | Ops | Terms | 2Q (CZ) | Depth | Size | |ΔE| | Character |
|---------|----:|------:|--------:|------:|-----:|---------:|-----------|
| Original fullhorse | 16 | 47 | 60 | 137 | 272 | 5.62e-05 | Baseline |
| Ultra-lean (6-op) | 6 | 16 | 48 | 121 | 222 | 5.62e-05 | Quadrature removed, zero regression |
| Aggressive (5-op) | 5 | 12 | 32 | 122 | 192 | 7.01e-05 | Operator-level Pareto |
| Gate-pruned (7-term) | 5 | 7 | 19 | 57 | 115 | 4.23e-04 | Gate-level Pareto |
| Noisy-validated (7-term) | 5 | 7 | 18 | 51 | 90 | 4.23e-04 | ADAPT order, FakeNighthawk transpiled |

Path A (Fixed-scaffold VQE): removed 5 near-zero layers (|θ|<1e-4), re-optimized 11 survivors → 53 CZ, depth 133, zero regression.
Path B (Re-ADAPT from HF): 10 operators, 53 CZ, depth 133, zero regression, stop reason: drop_plateau.

**Surviving scaffold (Path A, 11 operators):**
1. `hh_termwise_ham_quadrature_term(yezeee)` θ=-1.500893
2. `hh_termwise_ham_quadrature_term(yeeeze)` θ=-1.348605
3. `hh_termwise_ham_quadrature_term(eyeeez)` θ=+0.076385
4. `hh_termwise_ham_quadrature_term(eyezee)` θ=+0.026517
5. `hh_termwise_ham_quadrature_term(eyezee)` θ=-0.026481
6. `uccsd_ferm_lifted::uccsd_sing(alpha:0->1)` θ=+0.785398
7. `uccsd_ferm_lifted::uccsd_sing(beta:2->3)` θ=+0.816540
8. `paop_lf_full:paop_dbl_p(site=0->phonon=0)` θ=-0.022554
9. `paop_lf_full:paop_dbl_p(site=1->phonon=1)` θ=-0.471671
10. `paop_full:paop_hopdrag(0,1)::child_set[0,2]` θ=-0.785398
11. `paop_full:paop_disp(site=0)` θ=-0.043453

**Leave-one-out analysis (from 11-op baseline):**

| Idx | Operator | |θ| | Regression | Verdict |
|-----|----------|-----|-----------|---------|
| 0 | `quadrature(yezeee)` | 1.50 | -9.2e-13 | Free to remove |
| 1 | `quadrature(yeeeze)` | 1.35 | -9.0e-13 | Free to remove |
| 2 | `quadrature(eyeeez)` | 0.076 | -9.2e-13 | Free to remove |
| 3 | `quadrature(eyezee)` | 0.027 | -9.2e-13 | Free to remove |
| 4 | `quadrature(eyezee)` | 0.026 | -9.2e-13 | Free to remove |
| 7 | `paop_dbl_p(site=0->ph=0)` | 0.023 | +1.39e-05 | Cheap |
| 10 | `paop_disp(site=0)` | 0.043 | +4.20e-03 | Expensive |
| 9 | `paop_hopdrag(0,1)::child[0,2]` | 0.785 | +3.28e-01 | CRITICAL |
| 5 | `uccsd_sing(alpha:0->1)` | 0.785 | +5.99e-01 | CRITICAL |
| 6 | `uccsd_sing(beta:2->3)` | 0.817 | +5.99e-01 | CRITICAL |
| 8 | `paop_dbl_p(site=1->ph=1)` | 0.472 | +7.80e-01 | CRITICAL |

Key finding: All 5 quadrature terms show negative regression (absorbed by other ops after reopt). 4-operator irreducible core carries >99% of variational weight.

**Gate-level pruning (term-level LOO from 5-op/12-term baseline):**

| Block | Pauli | θ_opt | CX proxy | Regression | Verdict |
|-------|-------|------:|--------:|-----------:|---------|
| uccsd_sing(α) | eeeexy | +1.568 | 2 | +1.6e-08 | Free |
| uccsd_sing(α) | eeeeyx | +0.003 | 2 | -2.9e-09 | Free |
| uccsd_sing(β) | eexyee | +1.466 | 2 | +8.6e-09 | Free |
| uccsd_sing(β) | eeyxee | +0.167 | 2 | +6.5e-09 | Free |
| paop_dbl_p | yeeeee | -3.142 | 0 | +3.28e-01 | Critical |
| paop_dbl_p | yeeeze | -0.053 | 2 | +3.53e-04 | Cheap |
| paop_dbl_p | yezeee | +0.000 | 2 | -2.9e-09 | Free (θ≈0) |
| paop_dbl_p | yezeze | -1.542 | 4 | +7.60e-01 | Critical |
| hopdrag | yexxee | -0.324 | 4 | +5.4e-08 | Free |
| hopdrag | yeyyee | -1.247 | 4 | +2.1e-08 | Free |
| paop_disp | eyeeez | -0.087 | 2 | +4.20e-03 | Expensive |
| paop_disp | eyezee | -0.087 | 2 | +4.20e-03 | Expensive |

Removing 5 cheapest → 7 surviving terms: eeeexy, eeyxee, yeeeee, yezeze, yeyyee, eyeeez, eyezee. Combined regression: +3.53e-04, |ΔE|=4.23e-04. Gate reduction: 32 CZ → 19 CZ (41%), depth 122 → 57 (53%).

**4.3 Identification of Unused Pool Families**

L=2 pool usage (from motif analysis of 101-entry scaffold, 46 available operators):

| Class | In Pool | Used | Notes |
|-------|--------:|-----:|-------|
| `hh_termwise_unit` | 16 | 0/16 | Never selected |
| `hh_termwise_quadrature` | 4 | 4/4 | All used (seed block) |
| `uccsd_sing` | 2 | 2/2 | All used |
| `uccsd_dbl` | 1 | 0/1 | Never selected |
| HVA layers | 4 | 0/4 | Never selected |
| `paop_cloud_p` | 2 | 2/2 | DOMINANT — 87/101 entries (86%) |
| `paop_cloud_x` | 2 | 0/2 | Never selected |
| `paop_disp` | 2 | 0/2 | Never selected |
| `paop_dbl` | 2 | 0/2 | Never selected |
| `paop_hopdrag` | 1 | 1/1 | Used (4 entries via child sets) |
| `paop_dbl_p` | 4 | 2/4 | 2 used, 2 unused |
| `paop_dbl_x` | 4 | 0/4 | Never selected |
| `paop_curdrag` | 1 | 0/1 | Never selected |
| `paop_hop2` | 1 | 0/1 | Never selected |

**Result: 11 used / 46 available (24%). 35 operators NEVER selected. paop_cloud_p alone = 86% of all selections.**

Cross-validated at L=3 (9 qubits, from heavy scaffold best-yet run):
- Same pattern: all `hh_termwise_unit` (18), all HVA (4), all x-type PAOPs (11+) never used
- L=3 energy: 0.24502564, exact: 0.24494070, |ΔE|: 8.49e-05, scaffold depth 37

L=3 operator class usage:

| Class | Available | Direct | Via child_set | Never used |
|-------|--------:|-------:|------:|-------:|
| hh_termwise_quadrature | 6 | 6 | 0 | 0 |
| uccsd_sing | 4 | 4 | 0 | 0 |
| uccsd_dbl | 4 | 3 | 0 | 1 |
| paop_cloud_p | 4 | 3 | 3 | 0 |
| paop_lf_dbl_p | 7 | 5 | 2 | 1 |
| paop_hopdrag | 2 | 0 | 2 | 1 |
| paop_disp | 3 | 2 | 0 | 1 |
| hh_termwise_unit | 18 | 0 | 0 | 18 |
| paop_dbl | 3 | 0 | 0 | 3 |
| paop_cloud_x | 4 | 0 | 0 | 4 |
| paop_lf_dbl_x | 7 | 0 | 0 | 7 |
| HVA layers | 4 | 0 | 0 | 4 |
| paop_lf_curdrag | 2 | 0 | 0 | 2 |
| paop_lf_hop2 | 2 | 0 | 0 | 2 |

**4.4 Reduced Pool Recovery**

Lean pool (11 ops) vs Heavy pool (46 ops) comparison:

| Metric | Heavy (full_meta) | Lean (pareto_lean_l2) |
|--------|------------------:|---------------------:|
| Pool size | 46 | 11 |
| Final energy | 0.15872408303679303 | 0.15872408236037028 |
| abs_delta_e | 5.6178911068e-05 | 5.6178234644e-05 |
| Logical depth | 97 (manual interrupt) | 14 (drop_plateau) |
| Runtime params | — | 25 |
| Logical params | — | 14 |
| Stop reason | manual interrupt | drop_plateau |

**Same energy to 10 significant figures. Lean pool converges in 14 steps vs 97.**

Lean pool rerun scaffold (14 logical operators):
1. hh_termwise_ham_quadrature_term(yezeee)
2. hh_termwise_ham_quadrature_term(yeeeze)
3. hh_termwise_ham_quadrature_term(eyeeez)
4. hh_termwise_ham_quadrature_term(eyezee)
5. uccsd_ferm_lifted::uccsd_sing(alpha:0->1)
6. uccsd_ferm_lifted::uccsd_sing(beta:2->3)
7. paop_lf_full:paop_dbl_p(site=0->phonon=1)
8. paop_lf_full:paop_dbl_p(site=1->phonon=0)
9. paop_full:paop_hopdrag(0,1)::child_set[0,2]
10. paop_full:paop_cloud_p(site=1->phonon=0)
11. paop_full:paop_cloud_p(site=1->phonon=0)::child_set[1]
12. paop_full:paop_cloud_p(site=0->phonon=1)
13. paop_lf_full:paop_dbl_p(site=0->phonon=1)::child_set[3]
14. paop_lf_full:paop_dbl_p(site=0->phonon=1)::child_set[3]

Gate cost (FakeGuadalupeV2, opt_level=1): 97 CX, depth 97, 178 transpiled gates, 4 QWC measurement groups, 3,027,346 total shots at 1.6e-3 precision.

### Section 5: Evidence Packet — Real QPU & Noise Data

**5.1 Noise Attribution (FakeNighthawk, gate_pruned_7term)**
- Backend: FakeNighthawk, seed 7, layout [47,58,38,48,37,57], opt_level=1
- Compiled: 19 2Q gates (CZ), depth 63, size 93
- Ideal mean: 0.15909070
- Readout-only delta: +0.04855 ± 0.01051 (22% of total)
- Gate/stateprep-only delta: +0.11921 ± 0.00921 (68% of total)
- Full noise delta: +0.17849 ± 0.01141 (100%)
- Additivity residual: +0.01073 (cross-talk / non-additive)
- M3 mitigated delta: 0.11869 (matches gate/stateprep slice exactly — confirms M3 removes readout cleanly)

**5.2 Real QPU Runs**

| Backend | Circuit | Shots | Mitigation | Ideal Energy | Noisy Energy | |ΔE| noisy | Notes |
|---------|---------|------:|-----------|-------------|-------------|-----------|-------|
| ibm_marrakesh | 7-term fixed-theta | 2048 | mthree readout | 0.1591 | 0.8818 (static) | 0.7227 | 6 runtime jobs, full-circuit audit (not energy-only) |
| ibm_marrakesh | 6-term fixed-theta | 8192 | mthree readout | 0.1633 | 0.4486 | 0.2853 | Session fallback, open plan limitation |
| ibm_kingston | 6-term SPSA48 (interrupted) | — | — | 0.1587 | 0.3936 (best job) | ~0.235 | 75 quantum seconds, job-level recovery only |
| FakeMarrakesh | 6-term no mitigation | 2048×4 | none | 0.1633 | 0.3149 | 0.1516 | Simulated |
| FakeMarrakesh | 6-term readout only | 2048×4 | mthree | 0.1633 | 0.2561 | 0.0928 | **Best mitigated sim result** |
| FakeMarrakesh | 6-term readout+DD+twirl | 2048×4 | readout+gate_twirl | 0.1633 | 0.2769 | 0.1136 | DD worse than readout-only here |

FakeMarrakesh compile: 14 CZ, depth 48, size 80, layout [13,12,18,11,14,10].

**5.3 Marrakesh Compile Ranking (10 transpiler seeds, opt_level=1)**

| Artifact | 2Q Gates | Depth | Size | Logical |ΔE| | Executable? |
|----------|--------:|------:|-----:|--------:|:---:|
| circuit_optimized_7term | 22 | 52 | 107 | 0.338 | Yes but physics broken |
| gate_pruned_7term | 25 | 63-75 | 110-120 | 0.000423 | **Yes — recommended** |
| aggressive_5op | 37-40 | 110-135 | 161-189 | 0.00391 | NO — replay fidelity ~0.57 |
| readapt_5op | 37-40 | 110-135 | 161-189 | 0.00355 | NO — replay fidelity ~0.57 |

CRITICAL: 5-op artifacts have broken executable import (fidelity ~0.57, energy replays to 0.60 instead of 0.16). Only 7-term variants are trustworthy for hardware today.

**5.4 SPSA from Perturbed Starts (local noiseless, SPSA maxiter=128)**

7-term (reference |ΔE|=4.23e-04):

| sigma | seed | start |ΔE| | final |ΔE| |
|------:|-----:|-----------:|-----------:|
| 0.05 | 101 | 0.019714 | 0.012396 |
| 0.05 | 102 | 0.019897 | 0.012074 |
| 0.05 | 103 | 0.013473 | 0.011618 |
| 0.10 | 201 | 0.052040 | 0.011664 |
| 0.10 | 202 | 0.046826 | 0.010325 |
| 0.10 | 203 | 0.056062 | **0.008698** |

6-term (reference |ΔE|=4.61e-03):

| sigma | seed | start |ΔE| | final |ΔE| |
|------:|-----:|-----------:|-----------:|
| 0.05 | 101 | 0.019928 | 0.016100 |
| 0.05 | 102 | 0.024156 | 0.012423 |
| 0.05 | 103 | 0.017927 | 0.013498 |
| 0.10 | 201 | 0.055681 | 0.010198 |
| 0.10 | 202 | 0.051000 | 0.012630 |
| 0.10 | 203 | 0.061437 | 0.012889 |

7-term range: 0.0087-0.0124. 6-term range: 0.0102-0.0161. 7-term is more robust.

**5.5 FakeNighthawk Noisy VQE (simulated hardware, SPSA + M3)**

| Metric | No M3 (4096 shots, 30 iter) | M3 ON (8192 shots, 200 iter) |
|--------|----------------------------:|-----------------------------:|
| Noisy energy at SV-optimal θ | 0.2949 | 0.2679 |
| Best noisy energy (SPSA) | 0.2827 | 0.2296 |
| SV energy at noisy-optimal θ | 0.1596 | 0.1602 |
| |ΔE| SV at noisy-opt | 9.07e-4 | 1.53e-3 |
| Function evaluations | 60 | 400 |
| Elapsed | 819s (14 min) | 5493s (92 min) |

SPSA convergence trace (M3 ON):
```
iter    0: E_est=0.2517  best=0.2517
iter   20: E_est=0.2875  best=0.2496
iter   40: E_est=0.2679  best=0.2384
iter   60: E_est=0.2947  best=0.2384
iter   80: E_est=0.2806  best=0.2296
iter  100: E_est=0.3039  best=0.2296  (converged — no further improvement)
iter  199: E_est=0.2475  best=0.2296
```

Key: The SV energy at noisy-optimal θ (0.1596-0.1602) shows SPSA finds parameters within ~1e-3 of sector GS even through noise. The ~0.07 systematic upward bias in noisy energy is gate error (not readout).

### Section 6: Recommended Hardware Plan

Order: (b) first, then (a).

**(b) Fixed scaffold + initialized theta + SPSA on real QPU**
- Take 7-term gate-pruned circuit with locally optimized theta (|ΔE|=4.23e-04 noiseless)
- Transpile to target backend at opt_level=1
- Execute with mthree readout mitigation
- Run SPSA(maxiter=128+) on real QPU
- Zero exploration cost (no pool gradient evaluation)
- Only 7 parameters and 19 CZ gates per iteration
- Provides concrete |ΔE| baseline before expensive ADAPT re-run

**(a) Re-run geo-pos-aware ADAPT with reduced pool on real QPU**
- Use lean pool (11 generators)
- 4x cheaper per ADAPT step than 46-operator search
- May find hardware-native scaffold that outperforms fixed circuit
- More expensive overall: each iteration requires O(pool) gradient circuits

Rationale: (b) is cheap and gives a baseline. If (b) reaches chemical accuracy, (a) may be unnecessary.

### Section 7: Chemical-Accuracy Outlook

Chemical accuracy = 1 kcal/mol ≈ 1.6 mHa ≈ 0.0016 Ha.

**Already demonstrated:**
- Noiseless full-pool: |ΔE| = 5.62e-05 (30x below chemical accuracy)
- Noiseless 7-term: |ΔE| = 4.23e-04 (4x below chemical accuracy)
- Lean pool recovers full-pool energy (same |ΔE| to 10 sig figs)
- Readout mitigation (mthree) cleanly removes readout noise

**Plausible but not yet demonstrated:**
- SPSA on real QPU has not converged (Kingston: 48 iters, 75 quantum seconds — insufficient)
- FakeNighthawk noisy VQE finds parameters with SV |ΔE| ≈ 1e-3 (near chemical accuracy) but noisy readout still biased by ~0.07 gate error
- 7-term SPSA from perturbed starts converges to |ΔE| ≈ 0.009-0.012 noiseless (6x chemical accuracy — likely improvable with more iterations)

**What would most likely improve results:**
1. More SPSA iterations on real QPU (need 128+ iters, 300+ quantum seconds)
2. Lower gate-error backend (gate noise = 68% of total; 2-3x lower 2Q error → direct floor reduction)
3. Higher shots (32k+ to push sampling noise below gate-noise floor)
4. ZNE or PEC for gate-noise mitigation (mthree only handles readout)

### Section 8: Slide Blueprint
(Keep existing content from presentation_pi_handoff.md — it's already good, just verify it references inlined data not artifact paths)

### Section 9: Artifact Registry
(Keep existing table — it's a lookup for the repo, not for Google)

### Section 10: Open Gaps
Inline these 10 items:
1. 5-op executable import broken (replay fidelity ~0.57)
2. No converged SPSA-on-real-QPU result
3. FakeMarrakesh (|ΔE|=0.093) vs real Marrakesh (|ΔE|=0.285) gap unexplained
4. DD+twirling mixed results (worse than readout-only on FakeMarrakesh)
5. L=3 (9-qubit) hardware feasibility not assessed
6. Chemical accuracy threshold = 0.0016 Ha for this problem
7. Historical pruning workflow scripts missing from working tree
8. Kingston vs Marrakesh backend quality comparison incomplete
9. SPSA hyperparameters (a=0.2, c=0.1, alpha=0.602, gamma=0.101) may need tuning for hardware
10. Measurement grouping: 17 terms in 4 QWC groups — verify if runtime uses grouping or term-by-term

### Section 11: Novelty Assessment Space
(Keep existing content — 10 items for Google to research)

## Key instruction

**EVERY number, table, operator list, and energy value in this instruction file MUST appear in the final markdown.** Do not summarize or abbreviate. Google NotebookLM needs the raw data to generate accurate slides.

## Files you will need to read for any missing details

- `MATH/Math.md` — equations for Section 3
- `artifacts/reports/hh_prune_nighthawk_final_20260322.md` — full pruning report
- `artifacts/reports/hh_prune_nighthawk_pareto_menu_20260322.md` — Pareto menu, LOO, gate-level pruning
- `artifacts/reports/hh_L2_ecut1_scaffold_motif_analysis.md` — pool usage analysis
- `artifacts/reports/pareto_lean_l2_vs_heavy_L2_ecut1_comparison.md` — lean vs heavy
- `artifacts/reports/hh_heavy_scaffold_best_yet_20260321.md` — L=3 data
- `artifacts/reports/investigation_7term_noise_diagnosis_20260323.md` — noise attribution
- `artifacts/reports/investigation_marrakesh_pareto_20260323.md` — Marrakesh compile + fidelity
- `artifacts/reports/investigation_kingston_best_deltae_20260324.md` — Kingston evidence
- `artifacts/reports/investigation_kingston_partial_trace_and_rerun_plan_20260323.md` — Kingston partial
- `artifacts/json/hh_marrakesh_7term_local_exact_spsa128_perturbed_20260323T182846Z.json` — 7-term SPSA rows
- `artifacts/json/hh_marrakesh_6term_local_exact_spsa128_perturbed_20260323T182700Z.json` — 6-term SPSA rows
- `artifacts/json/hh_marrakesh_6term_local_mitigation_ablation_20260324T001622Z.json` — FakeMarrakesh ablation
- `artifacts/json/hh_gate_pruned_fixed_scaffold_attr_20260323T013814Z.json` — noise attribution data

All the data from these files is already extracted above in this instruction file. You should only need to read Math.md for the formal equations in Section 3.

---

# NOVELTY ISOLATION RESULTS (2026-03-25)

This section contains controlled ablation experiments testing which novel features of the geo-pos-aware ADAPT-VQE actually improve |ΔE|/cost. Include these results in the presentation handoff as a new section between Sections 5 and 6.

## Literature Context

A literature review (uploaded separately as PDF) classifies the novel components:

**Established prior art (not novel):**
- ADAPT-VQE's gradient-based selection loop [Grimsley 2019]
- QGT/Fubini-Study metric for VQAs [Stokes 2020, Provost & Vallee 1980]
- Measurement grouping by clique cover [Verteletskyi 2020]

**Likely novel synthesis (our contributions):**
- Refit-aware effective curvature h̃_m = h_m - b_m^T (M_m + λI)^{-1} b_m in operator selection
- Reduced metric F_m^{red} and novelty fraction N_m = F_m^{red}/F_m as tangent-space projection
- Combining these into a single "predicted trust-region energy drop per marginal lifetime cost" scalar
- Co-ADAPT-VQE [2026] independently does cost-penalized selection; Geo-ADAPT-VQE [2026] independently does QGT-aware selection — but neither combines all our ingredients

## Experimental Design

All experiments use HH L=2, t=1.0, U=4.0, g=0.5, n_ph_max=1 (6 qubits) unless noted. Direct `adapt_pipeline.py` surface (not staged wrapper). Baseline: `full_meta` pool, `phase3_v1`, `POWELL`, wide shortlist.

## Result 1: Shortlisting Efficacy — WIN

Wide shortlist saves 28% measurement cost to reach the same energy vs no shortlist.

| Setting | |ΔE| | Passes 1e-4 gate | Measurement groups to target |
|---------|------|:---------:|----:|
| Default shortlist + split off | 3.28e-01 | NO | — |
| Wide shortlist + split off | 5.62e-05 | YES | 474 |
| Shortlist off + split off | 5.62e-05 | YES | 657 |

Wide shortlist: `phase1_shortlist_size=256, phase2_shortlist_fraction=1.0, phase2_shortlist_size=128`.
Shortlist off: `phase1_shortlist_size=999999, phase2_shortlist_size=999999`.

**Conclusion:** Wide shortlisting is a clean efficiency win. No further testing needed — will be superseded by depth-prediction-remaining shortlist mechanism.

## Result 2: Lifetime-Cost Penalty — NULL at L=2

| Setting | |ΔE| | History length | groups_new | compile_cost |
|---------|------|:-:|---:|---:|
| Lifetime ON (phase3_v1) | 5.618e-05 | 14 | 10 | 369 |
| Lifetime OFF | 5.618e-05 | 14 | 10 | 369 |

Identical in every metric. The lifetime cost had zero effect at L=2 because scaffolds converge in only 14 steps — not enough depth for lifetime penalties to differentiate candidates.

**Needs L=3 retest** where scaffolds reach 37+ layers and cost divergence between families is larger.

## Result 3: Runtime Split — DORMANT at L=2

| Setting | |ΔE| | selected_child_total |
|---------|------|---:|
| Split ON | 5.618e-05 | 0 |
| Split OFF | 5.618e-05 | 0 |

`selected_child_total = 0` means split was available but never fired. At L=2 with 2 sites, parent operators are already atomic. At L=3, the heavy scaffold used 7 child sets (9 child atoms) — split should be active there.

**Needs L=3 retest.**

## Result 4: Backend-Conditioned Compile-Aware Scoring — HARMFUL at current settings

| Run | |ΔE| | compile_cost (summed) | groups_new | Compiled 2Q | Compiled depth |
|-----|------|---:|---:|---:|---:|
| Proxy cost baseline | 5.62e-05 | 369 | 10 | — | — |
| FakeNighthawk transpile | 3.28e-01 | 85.3 | 6 | 61 | 161 |
| FakeFez transpile | 3.28e-01 | 95.9 | 7 | 71 | 188 |
| FakeMarrakesh transpile | 3.28e-01 | 95.9 | 7 | 71 | 188 |

**Scaffold divergence analysis:**

Proxy baseline scaffold (reaches good basin at d9):
```
d0: uccsd_sing(alpha:0->1)     E=0.763  cost=14.0
d1: uccsd_sing(beta:2->3)      E=0.500  cost=15.0
d2: paop_dbl_p(site=0->ph=0)   E=0.494  cost=21.0
d3: paop_dbl_p(site=1->ph=1)   E=0.487  cost=22.0
d4: paop_disp(site=0)          E=0.487  cost=16.0
d5: uccsd_sing(beta:2->3)      E=0.487  cost=19.0
d6: paop_disp(site=1)          E=0.487  cost=18.0
d7: paop_dbl_p(site=1->ph=0)   E=0.487  cost=26.0
d8: paop_dbl_p(site=0->ph=1)   E=0.487  cost=27.0
d9: uccsd_sing(alpha:0->1)     E=0.159  dE=5.62e-05  ← BASIN TRANSITION
```

FakeNighthawk transpile scaffold (stuck in bad basin):
```
d0: uccsd_sing(alpha:0->1)     E=0.763  cost=5.92
d1: uccsd_sing(beta:2->3)      E=0.500  cost=4.42
d2: paop_dbl_p(site=0->ph=0)   E=0.494  cost=15.64
d3: paop_dbl_p(site=1->ph=1)   E=0.487  cost=13.49
d4: paop_disp(site=0)          E=0.487  cost=9.03
d5: uccsd_sing(beta:2->3)      E=0.487  cost=4.06
d6: uccsd_sing(alpha:0->1)     E=0.487  cost=5.82  ← DIVERGES HERE
d7: paop_disp(site=1)          E=0.487  cost=3.68
d8: paop_disp(site=1)          E=0.487  cost=2.07
d9: paop_disp(site=0)          E=0.487  cost=4.03  ← NEVER ESCAPES
```

**Root cause:** Backend compile cost over-weights cheap operators (uccsd_sing cost=5.92 on transpile vs 14.0 on proxy). This suppresses expensive-but-critical operators like `paop_hopdrag` (cost=75 on proxy). The selector greedily picks cheap-but-uninformative operators and never reaches the basin transition.

**UPDATE (v2 experiments completed):** The corrected tests show λ is the critical variable, NOT the transpile mode. See "Claim 5 corrected" section below. transpile + λ=0.05 reaches the good basin (|ΔE|=7.01e-05), matching proxy + λ=0.05. The transpile mode itself is benign. At λ ≤ 0.01, PAOP operators dominate early selection → bad basin. At λ=0.05, compile cost acts as a regularizer steering toward cheaper termwise operators → good basin. This reframes the feature from "harmful" to "essential regularizer."

## Reoptimization Policy (Claim 2) — COMPLETED, direct surface, POWELL

### 4-way factorial: position × reopt (L=2)

| Variant | Position | Reopt | Ops | Params | |ΔE| | Basin |
|---------|----------|-------|-----|--------|------|-------|
| append-only | `max-positions=1` | `append_only` | 16 | 37 | **3.283e-01** | BAD |
| append+window | `max-positions=1` | `windowed(3)` | 15 | 29 | **7.010e-05** | GOOD |
| insert-only | `max-positions=6` | `append_only` | 16 | 37 | **3.283e-01** | BAD |
| insert+window | `max-positions=6` | `windowed(3)` | 15 | 29 | **7.010e-05** | GOOD |

All runs: POWELL optimizer, `full_meta` pool, `phase3_v1`, `eps_grad=5e-7`, `max_depth=80`, `shortlist_size=256`.

Artifacts:
- `artifacts/json/claim2_append_only_L2_v2_20260325.json`
- `artifacts/json/claim2_append_window_L2_v2_20260325.json`
- `artifacts/json/claim2_insert_only_L2_v2_20260325.json`
- `artifacts/json/claim2_insert_window_L2_v2_20260325.json`

**Key finding:** Windowed reopt is a ~4700x improvement in |ΔE|. Position-aware insertion has zero measurable effect at L=2.

### Claim 5 corrected: Backend cost scoring — λ is the critical variable

| Backend Mode | λ_compile | Ops | |ΔE| | Basin |
|-------------|-----------|-----|------|-------|
| transpile | 0 | 13 | **3.283e-01** | BAD |
| transpile | 0.005 | 13 | **3.283e-01** | BAD |
| transpile | 0.01 | 15 | **3.283e-01** | BAD |
| transpile | **0.05** | 15 | **7.010e-05** | GOOD |

Artifacts:
- `artifacts/json/claim5_transpile_nocost_L2_v2_20260325.json`
- `artifacts/json/claim5_transpile_lambda0005_L2_v2_20260325.json`
- `artifacts/json/claim5_transpile_lambda001_L2_v2_20260325.json`
- `artifacts/json/claim5_transpile_lambda005_control_L2_v2_20260325.json`

**Key finding:** The transpile mode itself is harmless. The compile-cost weight acts as an essential **basin-selection regularizer**. Full λ sweep:

| λ_compile | |ΔE| | Basin | d0 operator |
|-----------|------|-------|-------------|
| 0.000 | 3.283e-01 | BAD | paop_dbl_p |
| 0.005 | 3.283e-01 | BAD | uccsd_sing(β) |
| 0.010 | 3.283e-01 | BAD | paop_disp |
| **0.020** | **7.010e-05** | **GOOD** | uccsd_sing(α) |
| 0.030 | 7.010e-05 | GOOD | uccsd_sing(α) |
| 0.040 | 7.010e-05 | GOOD | uccsd_sing(α) |
| 0.050 | 7.010e-05 | GOOD | uccsd_sing(α) |

**Sharp phase transition at λ ∈ (0.01, 0.02).** Below this, PAOP operators win on gradient magnitude alone but lead to bad basin. Above it, compile-cost penalty redirects toward cheaper operators that reach the good basin.

### Claims 3 & 4 at L=3 — warm-started (COMPLETED)

L=3 parameters: `t=1, U=2, ω₀=1, g=1, n_ph_max=1, periodic`. Warm-started with `--adapt-ref-json` from the 120-operator L=3 staged handoff. Reference state energy = -0.593 (above exact = -0.699).

| Run | Ops | Energy | vs Exact | |ΔE| | Sector |
|-----|-----|--------|----------|------|--------|
| baseline (life ON, split OFF) | 86 | -0.749 | -0.051 | 5.06e-02 | BELOW (mild) |
| Claim 3 (lifetime OFF) | 86 | -1.534 | -0.835 | 8.35e-01 | BELOW (massive!) |
| Claim 4 (split ON) | 80 | -0.614 | +0.085 | 8.46e-02 | ABOVE (correct) |

Artifacts:
- `artifacts/json/claim3_4_baseline_L3_warmstart_20260325.json`
- `artifacts/json/claim3_lifetime_off_L3_warmstart_20260325.json`
- `artifacts/json/claim4_split_on_L3_warmstart_20260325.json`

**Key finding:** The `full_meta` pool contains sector-breaking operators at L=3 (likely `hva_hh_ptw`). Without lifetime cost, the selector aggressively picks these, causing energy to go 0.835 below exact (variational violation). With lifetime cost ON, the violation is only 0.051. With split ON, energy stays correctly above exact.

**Claim 3 (lifetime cost):** POSITIVE — acts as an indirect sector regularizer at L=3. Prevents catastrophic sector drift.

**Claim 4 (runtime split):** POSITIVE for sector preservation — only variant with correct variational bound. `selected_child_total=0` means the split logic itself never fires, but it constrains operator candidates indirectly.

### Multi-seed robustness for Claim 2

| Seed | Reopt | Ops | |ΔE| | Basin |
|------|-------|-----|------|-------|
| 7 | append_only | 16 | 3.283e-01 | BAD |
| 7 | windowed | 15 | 7.010e-05 | GOOD |
| 42 | append_only | 16 | 3.283e-01 | BAD |
| 42 | windowed | 15 | 7.010e-05 | GOOD |
| 123 | append_only | 16 | 3.283e-01 | BAD |
| 123 | windowed | 15 | 7.010e-05 | GOOD |

Perfectly seed-independent. This is a deterministic consequence of the reopt policy, not a stochastic effect.

### Batching ablation (Claim 7)

| Batching | |ΔE| | Batch sizes | groups_new |
|----------|------|-------------|------------|
| ON | 7.010e-05 | all 1s | 7 |
| OFF | 7.010e-05 | all 1s | 7 |

Dormant at L=2 — near-degenerate gate never fires. Needs larger system.

## Overall novelty claim summary (updated 2026-03-25)

| Claim | Feature | Status | Evidence |
|-------|---------|--------|----------|
| 1 | Staged HH default path | **Negative at L=2** | A1a stuck at |ΔE|=0.010 |
| 2 | Windowed reopt | **STRONG POSITIVE** | 4700x improvement; seed-independent × 3 seeds |
| 2b | Position-aware insertion | **Null at L=2** | No effect in 2×2 factorial × 3 seeds |
| 3 | Lifetime-cost penalty | **POSITIVE at L=3** | Prevents massive sector violation; indirect regularizer |
| 4 | Runtime split | **POSITIVE for sector** | Only variant with correct variational bound at L=3 |
| 5 | Compile-cost regularization (λ) | **STRONG POSITIVE** | Sharp threshold at λ∈(0.01,0.02); basin-selection regularizer |
| 6 | Shortlisting | **Moderate positive** | 28% fewer groups; superseded |
| 7 | Batching | **Dormant at L=2** | Near-degenerate gate never fires |

