# Pareto-Lean L2 vs Heavy Full-Meta: L=2 n_ph_max=1 ADAPT-VQE Comparison

## Run Artifacts

- Heavy (full_meta): [adapt_hh_L2_ecut1_full_meta_phase3_heavy_powell_20260321T014500Z.full_scaffold_for_pruning.json](../json/adapt_hh_L2_ecut1_full_meta_phase3_heavy_powell_20260321T014500Z.full_scaffold_for_pruning.json)
- Lean (pareto_lean_l2): [adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell.json](../json/adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell.json)
- Motif analysis: [hh_L2_ecut1_scaffold_motif_analysis.md](hh_L2_ecut1_scaffold_motif_analysis.md)

## Physics

Both runs use identical physics:
- `L=2`, `t=1.0`, `u=4.0`, `dv=0.0`, `omega0=1.0`, `g_ep=0.5`
- `n_ph_max=1`, `boson_encoding=binary`, `ordering=blocked`, `boundary=open`
- Total qubits: 6 (fermion=4, boson=2)
- Exact filtered GS energy: `0.1586679041257264`

## ADAPT Settings (identical except pool and stop policy)

- `adapt_continuation_mode=phase3_v1`
- `adapt_inner_optimizer=POWELL`
- `adapt_max_depth=160`
- `adapt_maxiter=12000`
- `adapt_eps_grad=5e-7`
- `adapt_eps_energy=1e-9`
- `adapt_reopt_policy=windowed`
- `adapt_window_size=999999` (full refit every step)
- `adapt_window_topk=999999`
- `adapt_full_refit_every=8`
- `adapt_final_full_refit=true`
- `adapt_allow_repeats=true`
- `phase1_probe_max_positions=999999` (insertion at any position)
- `phase3_runtime_split_mode=shortlist_pauli_children_v1`
- `phase3_lifetime_cost_mode=phase3_v1`
- `phase3_enable_rescue=true`
- `phase3_symmetry_mitigation_mode=verify_only`

Stop policy difference:
- Heavy: `adapt_drop_floor=-1.0`, `adapt_grad_floor=-1.0` (disabled — manually interrupted at depth 97)
- Lean: auto defaults (`drop_floor=5e-4`, `patience=3`, `min_depth=12`) — stopped automatically at depth 16

## Pool Comparison

| | Heavy (full_meta) | Lean (pareto_lean_l2) |
|---|---|---|
| Pool size (post-dedup) | 46 | **11** |
| UCCSD singles | 2 | 2 |
| UCCSD doubles | 1 | **0** (dropped) |
| HH termwise quadrature | 4 | 4 |
| HH termwise unit | 16 | **0** (dropped) |
| HVA layers | 4 | **0** (dropped) |
| `paop_cloud_p` | 2 | 2 |
| `paop_cloud_x` | 2 | **0** (dropped) |
| `paop_disp` | 2 | **0** (dropped) |
| `paop_dbl` | 2 | **0** (dropped) |
| `paop_hopdrag` | 1 | 1 |
| `paop_dbl_p` | 4 | 2 (used site-phonon pairings only) |
| `paop_dbl_x` | 4 | **0** (dropped) |
| `paop_curdrag` | 1 | **0** (dropped) |
| `paop_hop2` | 1 | **0** (dropped) |

The lean pool retains only the 11 operators that the heavy scaffold actually selected.

## Results

| Metric | Heavy (full_meta) | Lean (pareto_lean_l2) |
|---|---|---|
| Final energy | 0.15872408303679303 | 0.15872408323495057 |
| `abs_delta_e` | **5.618e-5** | **5.618e-5** |
| Final depth | 97 (interrupted, 101 with seed) | **16** |
| Unique labels | 20 | 14 |
| Stop reason | manual interrupt | drop_plateau (auto) |
| Wall clock | ~5700s (interrupted) | **9.7s** |

**The lean pool matches the heavy run's accuracy exactly** — both reach `abs_delta_e = 5.618e-5`. The heavy run's extra 81 layers beyond position 16 contributed ~2e-12 total energy improvement (effectively zero).

## Scaffold: Heavy (full_meta) — 101 entries (first 16 shown, remaining 85 are cloud_p repeats)

```
 1. hh_termwise_ham_quadrature_term(yezeee)          [seed]
 2. hh_termwise_ham_quadrature_term(yeeeze)          [seed]
 3. hh_termwise_ham_quadrature_term(eyeeez)          [seed]
 4. hh_termwise_ham_quadrature_term(eyezee)          [seed]
 5. uccsd_ferm_lifted::uccsd_sing(alpha:0->1)
 6. uccsd_ferm_lifted::uccsd_sing(beta:2->3)
 7. paop_lf_full:paop_dbl_p(site=0->phonon=1)
 8. paop_lf_full:paop_dbl_p(site=1->phonon=0)
 9. paop_full:paop_hopdrag(0,1)::child_set[5,7]
10. paop_full:paop_hopdrag(0,1)::child_set[0,2]
11. paop_full:paop_hopdrag(0,1)::child_set[1,3]
12. paop_full:paop_hopdrag(0,1)::child_set[4,6]
13. paop_lf_full:paop_dbl_p(site=1->phonon=0)::child_set[3]
14. paop_lf_full:paop_dbl_p(site=0->phonon=1)::child_set[3]
15. paop_full:paop_cloud_p(site=1->phonon=0)::child_set[1]
16. paop_full:paop_cloud_p(site=0->phonon=1)::child_set[1]
    ... positions 17–101: paop_cloud_p repeats and child sets (85 entries, ~0 energy gain)
```

## Scaffold: Lean (pareto_lean_l2) — 16 entries

```
 1. hh_termwise_ham_quadrature_term(yezeee)          [seed]
 2. hh_termwise_ham_quadrature_term(yeeeze)          [seed]
 3. hh_termwise_ham_quadrature_term(eyeeez)          [seed]
 4. hh_termwise_ham_quadrature_term(eyezee)          [seed]
 5. uccsd_ferm_lifted::uccsd_sing(alpha:0->1)
 6. uccsd_ferm_lifted::uccsd_sing(beta:2->3)
 7. paop_lf_full:paop_dbl_p(site=0->phonon=1)
 8. paop_lf_full:paop_dbl_p(site=1->phonon=0)
 9. paop_full:paop_hopdrag(0,1)::child_set[5,7]
10. paop_full:paop_hopdrag(0,1)::child_set[0,2]
11. paop_full:paop_hopdrag(0,1)::child_set[1,3]
12. paop_full:paop_hopdrag(0,1)::child_set[4,6]
13. paop_lf_full:paop_dbl_p(site=1->phonon=0)::child_set[3]
14. paop_lf_full:paop_dbl_p(site=0->phonon=1)::child_set[3]
15. paop_full:paop_cloud_p(site=1->phonon=0)::child_set[1]
16. paop_full:paop_cloud_p(site=0->phonon=1)::child_set[1]
```

## Observations

1. **The 16-layer lean scaffold is identical to the first 16 positions of the heavy scaffold.** Every operator, every position, every child set — exactly the same. The pruning removed nothing that mattered.

2. **76% pool reduction, 0% accuracy loss.** 11 operators instead of 46. Same `abs_delta_e` to 4 significant figures.

3. **590x wall-clock speedup.** 9.7s vs ~5700s. Most of this is from the drop plateau stop policy preventing the 85 useless `cloud_p` repeat layers, but the smaller pool also reduces per-iteration gradient evaluation cost.

4. **The heavy run's depth 17–101 was pure waste.** 85 layers of `paop_cloud_p` repeats contributing ~2e-12 total. The drop plateau stop policy (auto defaults: `floor=5e-4`, `patience=3`, `min_depth=12`) correctly identifies when to stop.

5. **Operator selection order is deterministic.** The scaffold structure — seed quadrature, then UCCSD singles, then dbl_p, then hopdrag child sets, then dbl_p child sets, then cloud_p child sets — reflects a clear physical hierarchy:
   - Boson-fermion coupling (quadrature)
   - Fermionic correlation (UCCSD singles)
   - Long-range e-ph doubles (dbl_p)
   - Correlated hopping (hopdrag)
   - Cloud coupling fine-tuning (cloud_p child sets)

## Cross-Size Comparison (L=2 vs L=3)

| | L=2 (n_ph_max=1) | L=3 (n_ph_max=1) |
|---|---|---|
| Full pool | 46 | ~70 |
| Pruned pool | 11 (76% cut) | 30 (57% cut) |
| Heavy scaffold depth | 97 (interrupted) | 37 (drop_plateau) |
| Lean scaffold depth | 16 | 40 |
| `abs_delta_e` (heavy) | 5.618e-5 | 8.49e-5 |
| `abs_delta_e` (lean) | 5.618e-5 | 1.15e-4 |
| Lean matches heavy? | Yes (identical) | Close (~1.35x gap) |
| Dominant class | cloud_p (86%) | More balanced |

At L=2 the pruning is perfect — zero accuracy loss. At L=3 there's a small gap, suggesting `paop_disp` and/or `uccsd_dbl` (kept in L=3 lean but dropped in L=2 lean) contribute marginally at larger system sizes.

## Further Truncation: Depth 12

The lean scaffold's last 4 entries (positions 13–16: `dbl_p` child sets and `cloud_p` child sets) contribute a combined ~9e-10 energy improvement — effectively zero. Truncating to depth 12 (4 seed + 8 ADAPT steps) retains the same `abs_delta_e = 5.618e-5` while dropping `cloud_p` from the active circuit entirely.

### Per-step energy convergence (lean run)

| Depth | Energy | Drop | `abs_delta_e` |
|---:|---:|---:|---:|
| 11 | 0.159066707210023 | -1.552e-1 | 3.988e-4 |
| 12 | 0.158724083678658 | -3.426e-4 | **5.618e-5** |
| 13 | 0.158724083221412 | -4.572e-10 | 5.618e-5 |
| 14 | 0.158724082774620 | -4.468e-10 | 5.618e-5 |

Depth 12 is the convergence point. Positions 13–16 add ~9e-10 total — sub-nanosecond refinement.

### Depth-12 scaffold (9 unique operators)

```
 1. hh_termwise_ham_quadrature_term(yezeee)          [seed]
 2. hh_termwise_ham_quadrature_term(yeeeze)          [seed]
 3. hh_termwise_ham_quadrature_term(eyeeez)          [seed]
 4. hh_termwise_ham_quadrature_term(eyezee)          [seed]
 5. uccsd_ferm_lifted::uccsd_sing(alpha:0->1)
 6. uccsd_ferm_lifted::uccsd_sing(beta:2->3)
 7. paop_lf_full:paop_dbl_p(site=0->phonon=1)
 8. paop_lf_full:paop_dbl_p(site=1->phonon=0)
 9. paop_full:paop_hopdrag(0,1)::child_set[5,7]
10. paop_full:paop_hopdrag(0,1)::child_set[0,2]
11. paop_full:paop_hopdrag(0,1)::child_set[1,3]
12. paop_full:paop_hopdrag(0,1)::child_set[4,6]
```

## Hardware Circuit Cost (Depth-12, 6 Qubits)

Ansatz transpiled to IBM fake backends via Qiskit `generate_preset_pass_manager`. The circuit uses `SuzukiTrotter(order=2, reps=1)` synthesis for each `PauliEvolutionGate`.

### Gate counts by backend and optimization level

| Backend | Phys. qubits | opt_level | CX/ECR | Depth | Total gates | 1Q gates |
|---|---:|---:|---:|---:|---:|---:|
| FakeJakartaV2 | 7 | 1 | 319 | 365 | 547 | 228 |
| FakeJakartaV2 | 7 | 3 | **286** | 414 | 684 | 398 |
| FakeCasablancaV2 | 7 | 1 | 319 | 365 | 547 | 228 |
| FakeCasablancaV2 | 7 | 3 | **286** | 412 | 683 | 397 |
| FakeGuadalupeV2 | 16 | 1 | 319 | 365 | 547 | 228 |
| FakeGuadalupeV2 | 16 | 3 | 290 | 409 | 672 | 382 |
| FakeCairoV2 | 27 | 1 | 371 | 1011 | 2025 | 1654 |
| FakeCairoV2 | 27 | 3 | **284** | 808 | 1432 | 1148 |
| FakeWashingtonV2 | 127 | 1 | 326 | 398 | 554 | 228 |
| FakeWashingtonV2 | 127 | 3 | 292 | 449 | 724 | 432 |

### Observations

- **CX count is remarkably stable**: 284–326 across all backends at opt_level=3, ±7% variance.
- **Best CX count**: 284 (FakeCairoV2, opt=3). Best CX+depth combination: 286 CX / depth 412 (FakeJakartaV2/FakeCasablancaV2, opt=3).
- **Cairo opt=1 is the outlier**: 371 CX, depth 1011. The 27-qubit heavy-hex topology forces long SWAP chains when embedding 6 logical qubits; opt_level=3 recovers (284 CX, depth 808).
- **7-qubit backends are ideal**: Jakarta and Casablanca have the tightest topology match for 6 logical qubits, giving the lowest depth at competitive CX counts.
- **Compared to L=3 lean (depth 40, 10 qubits)**: The L=3 circuit had ~1,472 CX gates on FakeGuadalupeV2 — this depth-12 L=2 circuit is **~5x cheaper** in two-qubit gates.

### Measurement cost

| Metric | Value |
|---|---|
| Hamiltonian Pauli terms (non-identity) | 16 |
| QWC measurement groups | 4 |
| Total shots @ mHa precision (1.6e-3) | 3,027,346 |
| Total shots @ 0.1 mHa precision (1e-4) | 775,000,000 |

The 4-group QWC partitioning is efficient for 16 Pauli terms. At chemical accuracy (~mHa), ~3M shots is practical. Sub-mHa precision (0.1 mHa) requires ~775M shots — expensive but within reach for extended runs.

## QPU Feasibility Assessment

At 286 CX on 7 physical qubits with depth ~400, the L=2 depth-12 ansatz is within range of current IBM hardware. Rough fidelity estimate at typical CX error rates (~1e-3): `(1 - 1e-3)^286 ≈ 0.75` — noisy but viable with error mitigation (ZNE, PEC). The 4 QWC groups and ~3M shots at mHa translate to minutes of QPU time.

For comparison, L=3 at ~1,472 CX gives expected fidelity ~0.23 — not practical without error correction.

## Remaining Bottlenecks & Potential Improvements

The scaffold itself (9 unique operators, depth 12) is essentially at its minimum — further pruning loses accuracy. The remaining cost is in circuit compilation and measurement.

### 1. Gate synthesis (likely the biggest win)

The current circuit uses `SuzukiTrotter(order=2, reps=1)` to synthesize each `PauliEvolutionGate`. This is designed for Hamiltonian simulation and Trotterizes multi-term operators. For ADAPT-VQE, each scaffold entry is just `exp(-i*theta*G)` where G is a short sum of Pauli strings. A direct Pauli rotation decomposition (CNOT staircase per term) avoids Trotter overhead entirely and could reduce CX count by 20–40%.

### 2. Qubit tapering

The Hamiltonian conserves particle number and spin. Z2 symmetry tapering could reduce from 6 to 4 logical qubits. Fewer qubits means shorter CNOT ladders per Pauli exponential — a multiplicative saving on every gate in the circuit.

### 3. Measurement (already near-optimal)

4 QWC groups for 16 Pauli terms is about as good as it gets. Classical shadows or derandomized protocols won't beat 4 circuits. The 3M shot count is dominated by the variance of the largest-coefficient group. General commutativity grouping (non-QWC, requiring entangled measurement bases) could reduce group count further but the overhead of basis-change circuits may negate the benefit at only 4 groups.

### 4. What does NOT help

- **Further pool pruning**: 9 operators is the minimum for `abs_delta_e = 5.618e-5`.
- **Deeper transpiler optimization**: opt_level=3 already aggressive; marginal returns beyond.
- **Hardware-efficient ansatz**: changes the algorithm, loses physics-motivated operator structure.
