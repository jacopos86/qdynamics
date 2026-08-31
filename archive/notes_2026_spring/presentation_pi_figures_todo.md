# PI Presentation: Figures & Tables TODO

> **Priority:** P0 = must have for presentation, P1 = strongly recommended, P2 = nice to have

---

## P0 — Essential Figures

### Fig 1: Pool Family Usage Heatmap (L=2)
- **Title:** "Operator Pool Usage: 11 of 46 Families Selected"
- **Source artifact(s):**
  - `artifacts/reports/hh_L2_ecut1_scaffold_motif_analysis.md` (pool usage table)
- **Already exists?** Table exists in report; heatmap/bar chart needs generation.
- **Suggested format:** Horizontal bar chart: each pool family on y-axis, number of selections on x-axis, colored by family class (UCCSD, PAOP, HVA, termwise). Grey out never-used families.
- **Script to generate:** New script; extract counts from the motif analysis report or from the adapt handoff JSON's scaffold label list.

### Fig 2: Pruning Pareto Table / Chart
- **Title:** "Circuit Pruning Pareto Menu: Gate Count vs |ΔE|"
- **Source artifact(s):**
  - `artifacts/reports/hh_prune_nighthawk_pareto_menu_20260322.md`
- **Already exists?** Table exists in report; scatter plot needs generation.
- **Suggested format:** Scatter plot: x-axis = compiled CZ gates, y-axis = |ΔE| (log scale). Label each point (fullhorse, ultra-lean, aggressive, gate-pruned-7term). Draw chemical accuracy line at 1.6 mHa.
- **Script to generate:** Hardcode the 4 Pareto points from the report table; plot with matplotlib.

### Fig 3: Noise Attribution Breakdown
- **Title:** "Noise Source Attribution: Gate Noise Dominates"
- **Source artifact(s):**
  - `artifacts/json/hh_gate_pruned_fixed_scaffold_attr_20260323T013814Z.json`
  - Fields: `readout_only_delta`, `gate_stateprep_only_delta`, `full_noise_delta`
- **Already exists?** No.
- **Suggested format:** Stacked bar chart or pie chart: readout (22%), gate/stateprep (68%), cross-talk/residual (10%).
- **Script to generate:** Parse the attribution JSON; extract delta means; plot.

### Fig 4: ADAPT Energy Convergence Curve
- **Title:** "ADAPT-VQE Convergence: Full Pool vs Lean Pool"
- **Source artifact(s):**
  - Full pool: `artifacts/json/hh_backend_adapt_fullhorse_powell_20260322T194155Z_fakenighthawk.json` (trajectory fields)
  - Lean pool: referenced in `artifacts/reports/pareto_lean_l2_vs_heavy_L2_ecut1_comparison.md`
- **Already exists?** No.
- **Suggested format:** Line plot: x-axis = ADAPT depth (iteration), y-axis = energy. Two lines: full pool (46 ops, 97 steps) and lean pool (11 ops, 14 steps). Horizontal dashed line at exact energy.
- **Script to generate:** Parse adapt handoff JSONs for trajectory arrays; need to locate the lean-pool adapt handoff artifact.

### Fig 5: Lean Pool vs Heavy Pool Comparison Table
- **Title:** "Reduced Pool Recovery: Same Energy, 7x Fewer Iterations"
- **Source artifact(s):**
  - `artifacts/reports/pareto_lean_l2_vs_heavy_L2_ecut1_comparison.md`
- **Already exists?** Table in report; formatted slide-ready table needs extraction.
- **Suggested format:** Simple 2-row comparison table: Pool Size | |ΔE| | ADAPT Depth | CX Count | Converged?
- **Script to generate:** Direct extraction from report.

---

## P1 — Strongly Recommended

### Fig 6: Leave-One-Out Operator Importance
- **Title:** "Operator Importance: Leave-One-Out Regression Analysis"
- **Source artifact(s):**
  - `artifacts/reports/hh_prune_marginal_20260322T202009Z.md`
  - `artifacts/reports/hh_prune_nighthawk_pareto_menu_20260322.md`
- **Already exists?** Table in report.
- **Suggested format:** Horizontal bar chart: each operator on y-axis, regression cost (|ΔE| increase if removed) on x-axis (log scale). Color code: green = free to remove, yellow = marginal, red = critical.
- **Script to generate:** Extract LOO values from prune report; plot.

### Fig 7: Real QPU |ΔE| Summary Table
- **Title:** "Real QPU Execution: Current |ΔE| Status"
- **Source artifact(s):**
  - `artifacts/json/hh_gatepruned7_fixedtheta_runtime_ibm_marrakesh_20260323T145219Z_direct.json`
  - `artifacts/json/hh_gatepruned6_fixedtheta_runtime_ibm_marrakesh_20260323T171528Z.json`
  - `artifacts/json/hh_marrakesh_6term_local_mitigation_ablation_20260324T001622Z.json`
  - `artifacts/json/hh_kingston_6term_spsa48_runtime_partial_recovery_20260323.json`
- **Already exists?** No unified table.
- **Suggested format:** Table: Backend | Circuit | Shots | Mitigation | Noisy |ΔE| | Notes
- **Script to generate:** Parse each JSON; extract relevant fields.

### Fig 8: SPSA Convergence from Perturbed Starts
- **Title:** "SPSA Robustness: 7-term vs 6-term from Random Starts"
- **Source artifact(s):**
  - `artifacts/json/hh_marrakesh_7term_local_exact_spsa128_perturbed_20260323T182846Z.json`
  - `artifacts/json/hh_marrakesh_6term_local_exact_spsa128_perturbed_20260323T182700Z.json`
- **Already exists?** No.
- **Suggested format:** Box plot or violin plot: two distributions of final |ΔE|, one for 7-term, one for 6-term. Or overlay convergence traces.
- **Script to generate:** Parse perturbed SPSA JSONs for per-seed final energies; plot distributions.

### Fig 9: Method Overview Diagram
- **Title:** "Geo-Position-Aware ADAPT-VQE: Algorithm Overview"
- **Source artifact(s):** `MATH/Math.md` Section 11
- **Already exists?** No.
- **Suggested format:** Flow diagram showing: Pool → Cheap Screen → Shortlist → Position Probe → Batch Select → Beam Branch → Scaffold → Prune. Annotate each stage with key parameters.
- **Script to generate:** Manual diagram (draw.io, tikz, or presentation tool). Cannot auto-generate.

---

## P2 — Nice to Have

### Fig 10: Marrakesh Compile Ranking
- **Title:** "Transpilation Quality: Circuit Variants on Marrakesh"
- **Source artifact(s):**
  - `artifacts/reports/investigation_marrakesh_pareto_20260323.md`
- **Already exists?** Table in report.
- **Suggested format:** Bar chart: each variant on x-axis, compiled 2Q gates on y-axis, with depth as secondary axis.
- **Script to generate:** Extract compile ranking table; plot.

### Fig 11: L=2 vs L=3 Pool Family Comparison
- **Title:** "Pool Family Patterns Across System Sizes"
- **Source artifact(s):**
  - `artifacts/reports/hh_L2_ecut1_scaffold_motif_analysis.md` (L=2 motifs)
  - `artifacts/reports/hh_heavy_scaffold_best_yet_20260321.md` (L=3 motifs)
- **Already exists?** Separate tables in each report.
- **Suggested format:** Side-by-side heatmaps or grouped bar chart comparing family usage at L=2 and L=3.
- **Script to generate:** Extract both tables; align by family name; plot.

### Fig 12: Mitigation Ablation Bar Chart
- **Title:** "Error Mitigation Comparison: FakeMarrakesh 6-term"
- **Source artifact(s):**
  - `artifacts/json/hh_marrakesh_6term_local_mitigation_ablation_20260324T001622Z.json`
- **Already exists?** No.
- **Suggested format:** Grouped bar chart: mitigation strategy (none, readout, DD+twirling) vs |ΔE|.
- **Script to generate:** Parse ablation JSON; plot 3 bars.

### Fig 13: Accuracy Ladder Diagram
- **Title:** "Path to Chemical Accuracy: Current Status & Projections"
- **Source artifact(s):** Synthesized from multiple artifacts.
- **Already exists?** No.
- **Suggested format:** Vertical ladder/waterfall: exact → noiseless full (5.6e-05) → noiseless 7-term (4.2e-04) → SPSA converged (~0.01) → FakeMarrakesh mitigated (0.093) → real Marrakesh (0.285) → chemical accuracy target line (0.0016). Show gap to close.
- **Script to generate:** Manual diagram or matplotlib waterfall; hardcode values.

### Fig 14: HH Hamiltonian Schematic
- **Title:** "Holstein-Hubbard Model: Lattice + Phonon Coupling"
- **Source artifact(s):** `MATH/Math.md` Section 6
- **Already exists?** No.
- **Suggested format:** Cartoon of 2-site lattice with electron hopping, on-site repulsion, and phonon displacement modes. Label terms: $-t \sum c^\dagger c$, $U \sum n_\uparrow n_\downarrow$, $g \sum x_i n_i$.
- **Script to generate:** Manual diagram (tikz or draw.io).

---

## Generation Priority Order

1. Fig 2 (Pareto scatter) — simplest, highest impact
2. Fig 3 (Noise attribution) — critical for hardware story
3. Fig 1 (Pool usage) — key novelty evidence
4. Fig 5 (Lean vs heavy table) — simple extraction
5. Fig 7 (QPU summary table) — simple extraction
6. Fig 4 (Convergence curves) — requires locating trajectory data
7. Fig 6 (LOO bars) — moderate effort
8. Fig 8 (SPSA box plots) — moderate effort
9. Fig 9 (Method diagram) — manual, high effort
10. Fig 13 (Accuracy ladder) — manual, moderate effort
11. Remaining P2 figures as time permits
