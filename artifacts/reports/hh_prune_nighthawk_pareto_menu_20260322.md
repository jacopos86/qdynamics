# Nighthawk Pruning — Pareto Menu

**Source:** `hh_backend_adapt_fullhorse_powell_20260322T194155Z_fakenighthawk.json`
**Generated:** 2026-03-22
**Problem:** HH L=2, t=1.0, u=4.0, g=0.5, n_ph_max=1 (6 qubits)
**Exact energy:** 0.15866790413

---

## Executive Summary

Starting from the 16-layer fullhorse ADAPT circuit (60 2Q gates), we built a **Pareto menu of circuit variants** spanning the accuracy-vs-cost frontier through three pruning stages: operator-level, gate-level (individual Pauli term removal), and circuit-level analysis.

The **noisy-validated circuit** uses ADAPT ordering and compiles to **18 CZ gates, depth 51** — a **70% reduction in 2Q gates** and **63% reduction in depth** from the original, at a cost of 4.2e-04 in |ΔE|. A `disp_first` reordering achieves 16 CZ / depth 40 in compile metrics but is **variationally broken** (|ΔE|=0.34) due to non-commutativity of per-Pauli-term rotations — it is not recommended for execution.

The 7-term ADAPT-order circuit was validated on a **simulated noisy backend** (FakeNighthawk + M3 readout mitigation + SPSA). This is a **simulation-backed preflight**, not a live QPU run. The SPSA-found parameters produce a noiseless energy within **1.5e-3 of the sector ground state**, confirming the pruned circuit is a viable candidate for near-term hardware. The residual noise bias (~0.07) is dominated by gate errors, not readout.

## Pareto Frontier — FakeNighthawk

| Variant | Ops | Terms | 2Q (CZ) | Depth | Size | |ΔE| | Character |
|---------|----:|------:|--------:|------:|-----:|---------:|-----------|
| Original fullhorse | 16 | 47 | 60 | 137 | 272 | 5.62e-05 | Baseline |
| Ultra-lean (6-op) | 6 | 16 | 48 | 121 | 222 | 5.62e-05 | Quadrature removed, zero regression |
| Aggressive (5-op) | 5 | 12 | 32 | 122 | 192 | 7.01e-05 | Operator-level Pareto |
| Gate-pruned (7-term) | 5 | 7 | 19 | 57 | 115 | 4.23e-04 | Gate-level Pareto (ADAPT order, pre-transpile) |
| **Noisy-validated (7-term)** | **5** | **7** | **18** | **51** | **90** | **4.23e-04** | **ADAPT order, FakeNighthawk transpiled, noisy-validated** |
| ~~Circuit-optimized~~ | 5 | 7 | 16 | 40 | 86 | **0.34** | ~~disp_first reorder~~ — **variationally broken** (compile metrics only) |

### Reduction Summary

| From → To | 2Q Saved | Depth Saved | Size Saved | |ΔE| Cost |
|-----------|--------:|----------:|----------:|-----------|
| Original → Ultra-lean | -12 (20%) | -16 (12%) | -50 (18%) | None |
| Original → Aggressive | -29 (48%) | -39 (28%) | -126 (46%) | +1.39e-05 |
| Original → Noisy-validated 7-term | **-42 (70%)** | **-86 (63%)** | **-182 (67%)** | +3.67e-04 |
| Aggressive → Noisy-validated 7-term | -14 (44%) | -71 (58%) | -102 (53%) | +3.53e-04 |

---

## Leave-One-Out Analysis (from 11-op baseline)

Each row shows the energy regression when that single operator is removed and remaining params are re-optimized with POWELL.

| Idx | Operator | |θ| | Regression | |ΔE| after | Verdict |
|-----|----------|-----|-----------|----------|---------|
| 0 | `hh_termwise_ham_quadrature_term(yezeee)` | 1.50 | -9.2e-13 | 5.62e-05 | **Free to remove** |
| 1 | `hh_termwise_ham_quadrature_term(yeeeze)` | 1.35 | -9.0e-13 | 5.62e-05 | **Free to remove** |
| 2 | `hh_termwise_ham_quadrature_term(eyeeez)` | 0.076 | -9.2e-13 | 5.62e-05 | **Free to remove** |
| 3 | `hh_termwise_ham_quadrature_term(eyezee)` | 0.027 | -9.2e-13 | 5.62e-05 | **Free to remove** |
| 4 | `hh_termwise_ham_quadrature_term(eyezee)` | 0.026 | -9.2e-13 | 5.62e-05 | **Free to remove** |
| 7 | `paop_lf_full:paop_dbl_p(site=0->phonon=0)` | 0.023 | +1.39e-05 | 7.01e-05 | Cheap (~25% |ΔE| increase) |
| 10 | `paop_full:paop_disp(site=0)` | 0.043 | +4.20e-03 | 4.25e-03 | Expensive |
| 9 | `paop_full:paop_hopdrag(0,1)::child_set[0,2]` | 0.785 | +3.28e-01 | 3.28e-01 | Critical — do not remove |
| 5 | `uccsd_ferm_lifted::uccsd_sing(alpha:0->1)` | 0.785 | +5.99e-01 | 5.99e-01 | Critical — do not remove |
| 6 | `uccsd_ferm_lifted::uccsd_sing(beta:2->3)` | 0.817 | +5.99e-01 | 5.99e-01 | Critical — do not remove |
| 8 | `paop_lf_full:paop_dbl_p(site=1->phonon=1)` | 0.472 | +7.80e-01 | 7.80e-01 | Critical — do not remove |

### Key Finding

All 5 quadrature seed terms (`hh_termwise_ham_quadrature_term`) show **negative regression** — removing them actually makes the optimizer find a very slightly better minimum. This means the quadrature seed block, while useful during ADAPT search as scaffolding, is **fully absorbed into the remaining polaron/UCCSD operators** after re-optimization.

---

## Operator Tiers

### Tier 1 — Irreducible core (cannot remove)

These 4 operators carry >99% of the variational information:

| Operator | θ | Regression if removed | Physics |
|----------|---|----------------------|---------|
| `uccsd_sing(alpha:0->1)` | +0.785 | +0.599 | Spin-up single excitation |
| `uccsd_sing(beta:2->3)` | +0.817 | +0.599 | Spin-down single excitation |
| `paop_dbl_p(site=1->phonon=1)` | -0.472 | +0.780 | Polaron double displacement (site 1) |
| `paop_hopdrag(0,1)::child_set[0,2]` | -0.785 | +0.328 | Drag-assisted electron-phonon hopping |

### Tier 2 — Valuable refinement

| Operator | θ | Regression if removed | Physics |
|----------|---|----------------------|---------|
| `paop_disp(site=0)` | -0.043 | +4.20e-03 | Phonon displacement correction |
| `paop_dbl_p(site=0->phonon=0)` | -0.023 | +1.39e-05 | Polaron double displacement (site 0) |

### Tier 3 — Redundant (free to remove)

| Operator | θ | Regression if removed | Physics |
|----------|---|----------------------|---------|
| `hh_termwise_ham_quadrature` (×5) | varies | ~0 | ADAPT search scaffolding, fully absorbed |

---

## Recommended Variants

### For maximum accuracy at minimum cost: **Ultra-lean (6-op)**

```
Operators:
  1. uccsd_ferm_lifted::uccsd_sing(alpha:0->1)
  2. uccsd_ferm_lifted::uccsd_sing(beta:2->3)
  3. paop_lf_full:paop_dbl_p(site=0->phonon=0)
  4. paop_lf_full:paop_dbl_p(site=1->phonon=1)
  5. paop_full:paop_hopdrag(0,1)::child_set[0,2]
  6. paop_full:paop_disp(site=0)

2Q gates: 48    Depth: 121    |ΔE|: 5.62e-05    Regression: 0
```

This is the **Pareto-optimal choice** — same accuracy as the fullhorse at 20% fewer 2Q gates. The quadrature seed terms contribute nothing after re-optimization.

### For minimum gate count with controlled accuracy loss: **Aggressive (5-op)**

```
Operators:
  1. uccsd_ferm_lifted::uccsd_sing(alpha:0->1)
  2. uccsd_ferm_lifted::uccsd_sing(beta:2->3)
  3. paop_lf_full:paop_dbl_p(site=1->phonon=1)
  4. paop_full:paop_hopdrag(0,1)::child_set[0,2]
  5. paop_full:paop_disp(site=0)

2Q gates: 31    Depth: 98     |ΔE|: 7.01e-05    Regression: +1.39e-05
```

**48% fewer 2Q gates** than the original. The accuracy trade is small: |ΔE| goes from 5.62e-05 to 7.01e-05 (25% relative increase), which may be well within noise floors on real hardware.

### For noise-limited hardware: consider 4-op core

If the noise floor is >1e-3, even `paop_disp(site=0)` could be dropped, leaving just the 4-operator irreducible core. This would likely compile to ~20–25 2Q gates.

---

## Greedy Multi-Prune Results

| ΔE Budget | Operators removed | Remaining depth | |ΔE| after | Notes |
|-----------|-----------------:|---------:|----------:|-------|
| 1e-06 | [0,1,2,3,4] (5 quadrature) | 6 | 5.62e-05 | All quadrature free |
| 1e-05 | [0,1,2,3,4] (5 quadrature) | 6 | 5.62e-05 | Same — budget not yet spent |
| 1e-04 | [0,1,2,3,4,7] (+paop_dbl_p site 0) | 5 | 7.01e-05 | Marginal accuracy trade |
| 5e-04 | [0,1,2,3,4,7] | 5 | 7.01e-05 | No more cheap ops available |
| 1e-03 | [0,1,2,3,4,7] | 5 | 7.01e-05 | Gap to next removal is 4.2e-03 |
| 5e-03 | [0,1,2,3,4,7] | 5 | 7.01e-05 | Would need >4e-03 budget for paop_disp |

The pruning landscape has a **clear step structure**: the 5 quadrature terms are free, `paop_dbl_p(site=0)` costs 1.39e-05, and the next removal (`paop_disp`) costs 4.20e-03 — a 300x jump. This makes the 5-op and 6-op circuits natural Pareto points.

---

## Hardware Cost Comparison: Scaffold-VQE (Replay) vs Re-ADAPT

Both pruning paths (Path A: fix scaffold + re-optimize as conventional VQE; Path B: re-run ADAPT from HF with pruned pool) were transpiled on FakeNighthawk. They converge to **identical operator sets and compiled gate counts** at each pruning level.

### 5-op Aggressive — Detailed Hardware Profile

| Metric | Scaffold-VQE (Path A) | Re-ADAPT (Path B) |
|--------|----------------------:|-------------------:|
| **2Q gates (CZ)** | **31** | **31** |
| **Circuit depth** | **98** | **98** |
| **Total gate count** | **146** | **146** |
| **1Q gates (RZ)** | 70 | 70 |
| **1Q gates (SX)** | 45 | 45 |
| **Physical qubits** | 6 / 120 | 6 / 120 |
| **Qubit layout** | [37, 36, 25, 26, 35, 27] | [37, 36, 25, 26, 35, 27] |
| **QWC measurement groups** | 4 | 4 |
| **Total shots (mHa precision)** | ~3.03M | ~3.03M |
| **Runtime parameters** | 12 | 12 |
| **|ΔE|** | 7.01e-05 | 5.62e-05 |
| **SWAP insertions** | 0 | 0 |

All 6 logical qubits map onto a connected subgraph of the heavy-hex topology with **zero SWAP overhead**. The native basis is CZ — no basis translation required.

### Shot Budget Breakdown (4 QWC groups)

| Group | Shots | % of Budget |
|------:|------:|------------:|
| 1 | 2,539,063 | 83.9% |
| 2 | 195,313 | 6.4% |
| 3 | 195,313 | 6.4% |
| 4 | 97,657 | 3.2% |

### Full Compilation Matrix — All Variants × Both Paths

| Variant | Path | Operators | 2Q (CZ) | Depth | Size | |ΔE| |
|---------|------|----------:|---------:|------:|-----:|---------:|
| Original fullhorse | — | 16 | 60 | 137 | 272 | 5.62e-05 |
| Conservative | Scaffold-VQE | 11 | 53 | 133 | 233 | 5.62e-05 |
| As-is | Re-ADAPT | 10 | 53 | 133 | 233 | 5.62e-05 |
| Ultra-lean | Scaffold-VQE | 6 | 48 | 121 | 222 | 5.62e-05 |
| Ultra-lean | Re-ADAPT | 6 | 48 | 121 | 222 | 5.62e-05 |
| **Aggressive** | **Scaffold-VQE** | **5** | **31** | **98** | **146** | **7.01e-05** |
| **Aggressive** | **Re-ADAPT** | **5** | **31** | **98** | **146** | **5.62e-05** |

### Path Comparison Notes

- **Compiled circuits are identical** — same operators, same gate counts, same layout. The transpiler sees the same abstract circuit regardless of how it was derived.
- **Energy differs only for the 5-op aggressive variant**: Re-ADAPT (Path B) retains |ΔE|=5.62e-05 because it re-discovered the scaffold from scratch with jointly optimized parameters. Scaffold-VQE (Path A) regresses to 7.01e-05 because it only refits remaining parameters after dropping a layer.
- **For the 6-op ultra-lean and above**, both paths achieve identical energy — the quadrature removal is lossless either way.
- **Recommendation**: Use the **Re-ADAPT 5-op** artifact (`hh_prune_nighthawk_readapt_5op.json`) for hardware execution — same gate cost, better energy.

### Reduction from Original

| Metric | Original (16-op) | 5-op Aggressive | Noisy-validated 7-term |
|--------|------------------:|----------------:|---------------------------:|
| 2Q gates | 60 | 31 (-48%) | **18 (-70%)** |
| Circuit depth | 137 | 98 (-28%) | **51 (-63%)** |
| Total gates | 272 | 146 (-46%) | **90 (-67%)** |
| Runtime params | 47 | 12 (-74%) | **7 (-85%)** |
| |ΔE| | 5.62e-05 | 7.01e-05 | 4.23e-04 |

---

## Gate-Level Pruning (Term-Level Analysis)

Beyond operator-level pruning, we can prune individual Pauli rotation terms within operators. Each operator in the 5-op scaffold decomposes into 2–4 Pauli rotation gates (12 total). By zeroing individual terms and re-optimizing, we identify which sub-rotations are expendable.

### Per-Term Leave-One-Out (from re-optimized 5-op baseline)

| Block | Pauli | θ_opt | CX proxy | Regression | |ΔE| after | Verdict |
|-------|-------|------:|---------:|-----------:|----------:|---------|
| [0:0] `uccsd_sing(α)` | `eeeexy` | +1.568 | 2 | +1.6e-08 | 7.01e-05 | **Free** |
| [0:1] `uccsd_sing(α)` | `eeeeyx` | +0.003 | 2 | -2.9e-09 | 7.01e-05 | **Free** |
| [1:0] `uccsd_sing(β)` | `eexyee` | +1.466 | 2 | +8.6e-09 | 7.01e-05 | **Free** |
| [1:1] `uccsd_sing(β)` | `eeyxee` | +0.167 | 2 | +6.5e-09 | 7.01e-05 | **Free** |
| [2:0] `paop_dbl_p` | `yeeeee` | -3.142 | 0 | +3.28e-01 | 3.28e-01 | Critical |
| [2:1] `paop_dbl_p` | `yeeeze` | -0.053 | 2 | +3.53e-04 | 4.23e-04 | Cheap |
| [2:2] `paop_dbl_p` | `yezeee` | +0.000 | 2 | -2.9e-09 | 7.01e-05 | **Free** (θ≈0) |
| [2:3] `paop_dbl_p` | `yezeze` | -1.542 | 4 | +7.60e-01 | 7.60e-01 | Critical |
| [3:0] `hopdrag` | `yexxee` | -0.324 | 4 | +5.4e-08 | 7.02e-05 | **Free** |
| [3:1] `hopdrag` | `yeyyee` | -1.247 | 4 | +2.1e-08 | 7.01e-05 | **Free** |
| [4:0] `paop_disp` | `eyeeez` | -0.087 | 2 | +4.20e-03 | 4.27e-03 | Expensive |
| [4:1] `paop_disp` | `eyezee` | -0.087 | 2 | +4.20e-03 | 4.27e-03 | Expensive |

### Best Within-Block Prune Choices

For each multi-term operator, the best viable term removal (keeping ≥1 term):

| Operator | Remove | Keep | CX Proxy Saved | Regression |
|----------|--------|------|---------------:|-----------:|
| `uccsd_sing(α)` | `eeeeyx` | `eeeexy` | 2 | ~0 |
| `uccsd_sing(β)` | `eexyee` | `eeyxee` | 2 | ~0 |
| `paop_dbl_p(site=1)` | `yeeeze`, `yezeee` | `yeeeee`, `yezeze` | 4 | +3.53e-04 |
| `paop_hopdrag` | `yexxee` | `yeyyee` | 4 | ~0 |
| `paop_disp` | *(no viable pruning)* | both terms | 0 | — |

### Combined Gate-Pruned Circuit: 7 terms (from 12)

Removing all 5 cheapest terms simultaneously and re-optimizing:

```
Remaining terms (7):
  1. eeeexy  (uccsd_sing α, coeff -0.5)
  2. eeyxee  (uccsd_sing β, coeff +0.5)
  3. yeeeee  (paop_dbl_p, coeff +0.25)   — single-qubit, 0 CX!
  4. yezeze  (paop_dbl_p, coeff +0.25)
  5. yeyyee  (paop_hopdrag, coeff -0.5)
  6. eyeeez  (paop_disp, coeff -0.5)
  7. eyezee  (paop_disp, coeff -0.5)

Combined regression: +3.53e-04    |ΔE|: 4.23e-04
```

### Verified Compiled Gate Counts (FakeNighthawk)

| Metric | 5-op (12 terms) | Gate-pruned (7 terms) | Savings |
|--------|----------------:|---------------------:|--------:|
| **2Q gates (CZ)** | **32** | **19** | **-13 (41%)** |
| **Circuit depth** | **122** | **57** | **-65 (53%)** |
| **Total gates** | **192** | **115** | **-77 (40%)** |
| **Abstract CX** | 28 | 16 | -12 (43%) |

### Full Pareto Frontier (updated with gate-level pruning)

| Variant                        | Ops | Terms | 2Q (CZ) | Depth | Size | \|ΔE\|     | Character             |
|--------------------------------|----:|------:|--------:|------:|-----:|-----------:|-----------------------|
| Original fullhorse             |  16 |    47 |      60 |   137 |  272 |   5.62e-05 | Baseline              |
| Ultra-lean (6-op)              |   6 |    16 |      48 |   121 |  222 |   5.62e-05 | Quadrature removed    |
| Aggressive (5-op)              |   5 |    12 |      32 |   122 |  192 |   7.01e-05 | Operator-level Pareto |
| **Gate-pruned (5-op, 7-term)** | **5** |  **7** |      **19** |    **57** |  **115** |   **4.23e-04** | Gate-level Pareto |
| **Circuit-optimized (7-term)** | **5** |  **7** |      **16** |    **40** |   **86** |   **4.23e-04** | **Term reordering + transpiler opt** |

---

## Circuit-Level Optimization (Approaches 1, 3, 4)

After gate-level pruning to 7 terms / 19 CZ, three further circuit-level optimizations were explored:

### Approach 1+3: Pauli Gadget Merging & Term Reordering

The 7 Pauli rotation terms can be reordered to maximize transpiler cancellation opportunities. Terms that share basis changes on the same qubit (e.g., Y on qubit 0) are grouped together so the transpiler can cancel adjacent single-qubit gates between them.

**Ordering sweep results** (12 orderings × 100 seeds, FakeNighthawk opt_level=2):

| Ordering | Best CZ | Best Depth | Best Size | Key Insight |
|----------|--------:|----------:|---------:|-------------|
| `disp_first` [5,6,2,3,4,0,1] | **16** | **40** | **86** | Y₁ terms first, then Y₀ cluster |
| `q0_first` [2,3,4,0,1,5,6] | 16 | 48 | 88 | Y₀ cluster first |
| `reverse` [6,5,4,3,2,1,0] | 16 | 45 | 88 | — |
| `q0_first_v2` [2,4,3,0,1,5,6] | 16 | 49 | 85 | — |
| `original` [0,1,2,3,4,5,6] | 18 | 44 | 92 | No grouping, 2 extra CZ |
| `interleaved` [2,0,3,1,4,5,6] | 17 | 53 | 86 | Mixing prevents cancellations |

**Compile-metric winner: `disp_first` ordering — 16 CZ, depth 40, size 86.**

The grouping enables the transpiler to merge adjacent basis-change gates between consecutive Pauli rotations that act on the same qubit. Three terms share Y on qubit 0 (`yeeeee`, `yezeze`, `yeyyee`) and two share Y on qubit 1 (`eyeeez`, `eyezee`). Placing each group contiguously saves 2 CZ vs the ADAPT ordering.

> **CAVEAT (discovered during noisy VQE validation):** The `disp_first` ordering is **variationally broken** for `per_pauli_term` parameterization. Because the 7 Pauli rotations do not commute, reordering them changes the energy landscape. The `disp_first` ordering can only reach |ΔE|≈0.34 (verified via `differential_evolution` with 420k function evaluations), while ADAPT ordering reaches |ΔE|=4.2e-4. The 16 CZ figure is **compile-metrics only** and should not be used for execution. The **noisy-validated circuit uses ADAPT ordering [0,1,2,3,4,5,6] at 18 CZ, depth 51**.

### Approach 4: Native Gate Decomposition Analysis

Investigated whether hand-crafted CZ-native decompositions could beat the standard Pauli rotation pattern (basis change → CNOT ladder → RZ → reverse).

**Finding: 16 CZ is the theoretical floor** for 7 independent Pauli rotations with this weight distribution.

| Term | Weight | CZ Required | Why |
|------|-------:|------------:|-----|
| `yeeeee` | 1 | 0 | Single-qubit RY, no entanglement |
| `eeeexy`, `eeyxee`, `eyeeez`, `eyezee` | 2 | 2 each | `exp(-iθ ZZ)` requires exactly 2 CZ for general θ |
| `yezeze`, `yeyyee` | 3 | 4 each | Weight-3 CNOT ladder: 2 up + 2 down |
| **Total** | — | **16** | Matches transpiler result |

The key constraint: `exp(-iθ Z⊗Z)` for parametric θ requires exactly 2 CZ gates — verified both analytically (KAK decomposition shows 1 CZ only works at θ=π/2) and via Qiskit's `RZZGate` transpilation. Weight-3 terms inherit this: 2 CZ per rung × 2 rungs = 4 CZ.

**CNOT sharing between terms** (e.g., `yezeze` and `yeyyee` share CX(0,2)) was investigated but cannot reduce CZ count because the terms are non-commuting and the ladder undo/redo pairs are separated by basis-change operations.

### Circuit-Level Optimization Summary

| Stage | 2Q (CZ) | Depth | Size | Improvement |
|-------|--------:|------:|-----:|-------------|
| Gate-pruned (naive order) | 19 | 57 | 115 | Baseline |
| + Term reordering (disp_first) | **16** | **40** | **86** | -3 CZ (16%), -17 depth (30%), -29 size (25%) |
| + Native decomposition | 16 | 40 | 86 | No further improvement (at theoretical floor) |

### Gate-Pruned Pool for Re-ADAPT: Negative Result

We tested feeding the gate-pruning insight back into the operator pool (`pareto_lean_gate_pruned`), replacing multi-term operators with their single surviving Pauli term, and running fresh ADAPT from HF.

**Result: ADAPT fails completely.** Max gradient ~4e-08 (effectively zero). Energy stuck at HF (4.586).

**Root cause:** The two-term structure of UCCSD singles (`-0.5·XY + 0.5·YX`) is the JW image of the anti-Hermitian excitation `a†_p a_q - a†_q a_p`. This anti-Hermitian form is what produces a non-zero commutator `[H, G]|HF⟩` — the gradient ADAPT uses for operator selection. A single Pauli term (e.g., just `XY`) has vanishing gradient from HF because it lacks the antisymmetric structure.

**Conclusion:** The "redundant" Pauli terms are needed for ADAPT's gradient-driven search to work from the reference state. They only become redundant after the circuit has converged and other operators compensate. **Gate-level pruning is strictly a post-ADAPT optimization**, not a pool-level one.

| Pool | Max Gradient (depth 1) | Final |ΔE| | Verdict |
|------|----------------------:|---------:|---------|
| `pareto_lean` (multi-term) | 2.0 | 5.62e-05 | Works |
| `pareto_lean_gate_pruned` (single-term) | 4.4e-08 | 4.43 (HF) | Fails |

### Physics of the Surviving Terms

The 7 surviving terms have a clear physical structure:

1. **`eeeexy`** — Single alpha excitation (0→1). One Pauli term suffices because the optimizer absorbed the `eeeeyx` counterpart.
2. **`eeyxee`** — Single beta excitation (2→3). Same story.
3. **`yeeeee`** — Pure phonon displacement on site 1. **Zero CX cost** — single-qubit Y rotation.
4. **`yezeze`** — Polaron dressing: correlated phonon-electron displacement on site 1.
5. **`yeyyee`** — Drag-assisted hopping: electron-phonon correlated hop between sites.
6. **`eyeeez`** + **`eyezee`** — Phonon displacement on site 0 (both terms needed, ~4.2e-03 each).

---

## Noisy VQE on FakeNighthawk (Simulated Hardware)

The circuit-optimized 7-term ansatz was run as a full shot-based VQE on a simulated noisy backend with readout error mitigation.

### Setup

- **Backend:** `FakeNighthawk` — Qiskit's noise model of IBM's 127-qubit Eagle processor (CZ-native, heavy-hex topology)
- **Simulator:** `AerSimulator.from_backend(FakeNighthawk())` — preserves full 120-qubit layout, gate errors, readout errors, and coupling map
- **Readout mitigation:** M3 (`mthree` v3.0) — probabilistic readout error mitigation via `M3Mitigation.cals_from_system()` + `apply_correction(counts, qubits)`
- **Optimizer:** SPSA (Simultaneous Perturbation Stochastic Approximation) — noise-robust gradient-free optimizer
- **Energy evaluation:** Per-Pauli-term measurement with parity expectation values — each of the 17 Hamiltonian terms measured separately, basis rotations appended post-transpilation

### Circuit Details

The circuit uses **ADAPT ordering** (not `disp_first`) because operator non-commutativity makes the ordering variationally significant. The `disp_first` ordering optimized for gate count (16 CZ) but could only reach E≈0.50 — the ADAPT ordering reaches E≈0.159 at the cost of 2 extra CZ gates.

| Property | Value |
|----------|-------|
| Ordering | `adapt` [0,1,2,3,4,5,6] — matches executor's native sequence |
| 2Q gates (CZ) | 18 |
| Circuit depth | 51 |
| Total gates | 90 |
| Parameters | 7 |
| Physical qubit layout | [48, 38, 36, 47, 37, 27] |
| Transpiler | `optimization_level=2`, `seed_transpiler=17` |

### Reference Energies

Two "exact" ground state energies exist and are easily confused:

| Energy | Value | What it is |
|--------|------:|------------|
| **Sector GS** | **0.15867** | Ground state within the correct (1↑, 1↓) particle number sector. This is the physically meaningful VQE target. Computed via `build_hh_sector_hamiltonian_ed()`. |
| Full H GS | -0.15068 | Ground state of the full 2⁶-dimensional Hamiltonian including all particle number sectors. Lower, but unreachable by particle-number-preserving circuits. |

The per_pauli_term parameterization does not strictly conserve particle number (individual Pauli rotations break the symmetry), so the optimizer can drift into wrong sectors. Starting from the executor's known-good parameters keeps the state approximately within the correct sector.

### Parameter Initialization

The SV starting point is obtained from the `CompiledAnsatzExecutor` (which operates in the correct sector) and converted to circuit parameters:

1. Load the 5-op scaffold from `hh_prune_nighthawk_aggressive_5op.json`
2. Build the executor with `parameterization_mode="per_pauli_term"` (12 runtime params)
3. Optimize the 7-term subspace (zero out the 5 pruned terms) with Powell → executor dt values
4. Convert: `θ_circuit[i] = 2 × dt[i] × coeff[i]` (circuit implements `exp(-iθ/2 · P)`, executor implements `exp(-i·dt·coeff·P)`)
5. Polish in the circuit's own parameter space with Powell

This gives a noiseless (statevector) circuit energy of **0.15909** (|ΔE|=4.23e-4 from sector GS).

### Results

| Metric | No M3 (4096 shots, 30 iter) | **M3 ON (8192 shots, 200 iter)** |
|--------|----------------------------:|----------------------------------:|
| Noisy energy at SV-optimal θ | 0.2949 | 0.2679 |
| **Best noisy energy (SPSA)** | **0.2827** | **0.2296** |
| **SV energy at noisy-optimal θ** | **0.1596** | **0.1602** |
| |ΔE| SV at noisy-opt from sector GS | 9.07e-4 | 1.53e-3 |
| Total function evaluations | 60 | 400 |
| Elapsed time | 819s (14 min) | 5493s (92 min) |

### Interpretation

- **The "true" quality of the SPSA-found parameters is measured by the SV energy at noisy-opt θ** — this removes the systematic noise bias and shows what energy the circuit would produce on ideal hardware at those parameters. Both runs find parameters within ~1e-3 of the sector GS.
- **The noisy energy (0.23 with M3) includes a ~0.07 systematic upward bias** from depolarizing gate errors on FakeNighthawk. M3 corrects readout errors but not gate errors. The bias breaks down as:
  - Readout error contribution: ~0.03 (removed by M3; visible as the 0.295→0.268 drop at initial θ)
  - Gate error contribution: ~0.07 (not mitigable by M3 alone; would require ZNE, PEC, or similar)
- **SPSA converged by iteration ~80** (best energy stopped improving). The remaining 120 iterations confirmed the plateau rather than finding a better minimum.

### SPSA Convergence Trace (M3 ON)

```
iter    0: E_est=0.2517  best=0.2517
iter   20: E_est=0.2875  best=0.2496
iter   40: E_est=0.2679  best=0.2384
iter   60: E_est=0.2947  best=0.2384
iter   80: E_est=0.2806  best=0.2296
iter  100: E_est=0.3039  best=0.2296
iter  120: E_est=0.2833  best=0.2296
iter  140: E_est=0.2851  best=0.2296
iter  160: E_est=0.2823  best=0.2296
iter  180: E_est=0.3122  best=0.2296
iter  199: E_est=0.2475  best=0.2296
```

### Critical Implementation Details (for Replication)

**Qubit mapping bug (exyz ↔ Qiskit convention):** The repo's exyz label convention has position 0 as MSB in the kron product (e.g., `xeeeee` → X on qubit 5 in Qiskit's little-endian convention). The `_pauli_rotation_circuit` function must map exyz position `i` to Qiskit qubit `nq - 1 - i`:

```python
# CORRECT: exyz position i → Qiskit qubit (nq - 1 - i)
for i, p in enumerate(pauli_str):
    q = nq - 1 - i  # exyz MSB → Qiskit highest qubit
    if p == "x":
        qc.h(q)
        active.append(q)
    elif p == "y":
        qc.sdg(q)
        qc.h(q)
        active.append(q)
```

Without this mapping, the circuit applies rotations on the wrong qubits and cannot reach the ground state (optimizer gets stuck at E≈0.0 regardless of ordering — verified with differential_evolution global optimizer over 420k evaluations).

**SparsePauliOp label conversion:** The exyz→IXYZ conversion does NOT reverse the string. `"eeeexy"` → `"IIIIXY"` directly (position-by-position). This was verified: `max(|H_repo - H_SparsePauliOp|) = 0.0`.

**HF state preparation:** `qc.x(0); qc.x(2)` prepares |000101⟩ (index 5). Qiskit qubit 0 = exyz position 5 = LSB, Qiskit qubit 2 = exyz position 3. The HH reference state for L=2, half-filled, blocked indexing has occupancy at these positions.

**AerSimulator creation:** Must use `AerSimulator.from_backend(backend)`, NOT `AerSimulator(noise_model=NoiseModel.from_backend(backend))`. The latter creates a 30-qubit simulator instead of 120, causing M3 calibration to crash with `CircuitError: Index 36 out of range for size 30`.

**Measurement circuit:** After transpilation, basis rotations (H for X, SDG→H for Y) are appended to the physical qubits and retranspiled with `basis_gates=["cz","rz","sx","x","id","measure","reset"], optimization_level=0` to decompose SDG into basis gates without rerouting.

**Parameter binding:** Direct binding `{params[i]: theta[i]}` — no coefficient scaling, no DISP_FIRST_ORDER reindexing. Each `θ_i` is the RZ angle for the i-th Pauli rotation in ADAPT order.

### Reproducing the Run

```bash
# Environment: conda env with qiskit>=2.3, qiskit-aer, qiskit-ibm-runtime, mthree, scipy
conda activate qiskit-sim
pip install mthree  # if not installed

# Full M3 run (≈90 min)
python -u -m pipelines.hardcoded.hh_noisy_vqe_7term \
    --shots 8192 --maxiter 200 --seed 42 \
    --spsa-a 0.03 --spsa-c 0.05

# Quick test without M3 (≈14 min)
python -u -m pipelines.hardcoded.hh_noisy_vqe_7term \
    --shots 4096 --maxiter 30 --no-mthree --seed 42
```

The script outputs progress to stdout and saves a JSON artifact to `artifacts/json/hh_noisy_vqe_7term_<timestamp>.json`.

---

## Artifacts

| File | Description |
|------|-------------|
| `pipelines/hardcoded/hh_prune_nighthawk.py` | Pruning pipeline (scaffold VQE + re-ADAPT) |
| `pipelines/hardcoded/hh_prune_marginal_analysis.py` | Leave-one-out + greedy multi-prune analysis |
| `artifacts/json/hh_prune_nighthawk_pruned_scaffold.json` | 11-op conservative scaffold |
| `artifacts/json/hh_prune_nighthawk_ultra_lean_6op.json` | 6-op ultra-lean scaffold |
| `artifacts/json/hh_prune_nighthawk_aggressive_5op.json` | 5-op aggressive scaffold |
| `artifacts/json/hh_prune_nighthawk_pruned_scaffold_compile_scout.json` | 11-op Nighthawk cost |
| `artifacts/json/hh_prune_nighthawk_ultra_lean_6op_compile_scout.json` | 6-op Nighthawk cost |
| `artifacts/json/hh_prune_nighthawk_aggressive_5op_compile_scout.json` | 5-op Nighthawk cost |
| `artifacts/json/hh_prune_readapt_nighthawk_20260322T201452Z.json` | Re-ADAPT from HF (10-op) |
| `artifacts/json/hh_prune_readapt_nighthawk_20260322T201452Z_compile_scout.json` | Re-ADAPT 10-op Nighthawk cost |
| `artifacts/json/hh_prune_nighthawk_readapt_6op.json` | Re-ADAPT 6-op ultra-lean scaffold |
| `artifacts/json/hh_prune_nighthawk_readapt_5op.json` | Re-ADAPT 5-op aggressive scaffold (recommended for hardware) |
| `artifacts/json/hh_prune_marginal_20260322T202009Z.json` | Full marginal analysis data |
| `artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json` | 7-term gate-pruned scaffold |
| `artifacts/json/hh_prune_nighthawk_gate_pruned_7term_compile_scout.json` | 7-term Nighthawk cost |
| `artifacts/json/hh_prune_nighthawk_term_level_loo.json` | Term-level leave-one-out data |
| `pipelines/hardcoded/hh_noisy_vqe_7term.py` | Noisy VQE pipeline (FakeNighthawk + M3 + SPSA) |
| `artifacts/json/hh_noisy_vqe_7term_20260323T050918Z.json` | Noisy VQE result: M3 ON, 8192 shots, 200 iter |
| `artifacts/json/hh_noisy_vqe_7term_20260323T033354Z.json` | Noisy VQE result: no M3, 4096 shots, 30 iter |
