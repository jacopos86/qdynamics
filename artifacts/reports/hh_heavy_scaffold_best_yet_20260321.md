# HH Heavy Scaffold Best-Yet Run Note (2026-03-21)

Primary artifacts:
- JSON: [adapt_hh_L3_full_meta_phase3_heavy_powell_20260321T000723Z.json](../json/adapt_hh_L3_full_meta_phase3_heavy_powell_20260321T000723Z.json)
- Log: [adapt_hh_L3_full_meta_phase3_heavy_powell_20260321T000723Z.log](../logs/adapt_hh_L3_full_meta_phase3_heavy_powell_20260321T000723Z.log)

## Human Summary

This is the current best-yet broad-pool HH scaffold artifact from the heavy `phase3_v1` ADAPT line after the sector-preserving pool fix.

What changed relative to the earlier broken heavy reruns:
- The HH HVA/full-meta path now uses sector-preserving lifted UCCSD macros instead of non-preserving per-Pauli UCCSD fragments.
- HH base-pool entries are operator-audited for sector preservation before selection.
- Runtime split still probes child atoms aggressively, but only symmetry-safe child sets are allowed into the scaffold.

Why this specific run matters:
- It stayed variational against the filtered exact reference instead of collapsing to an unphysical negative energy.
- It used the full `phase3_v1` continuation machinery with the explicit depth-0 `full_meta` override, large probe window, large shortlist, and `POWELL`.
- It is the cleanest current scaffold for downstream operator-ranking analysis.

### Run Manifest

Physics:
- `problem=hh`
- `L=3`
- `t=1.0`
- `u=4.0`
- `dv=0.0`
- `omega0=1.0`
- `g_ep=0.5`
- `n_ph_max=1`
- `boson_encoding=binary`
- `ordering=blocked`
- `boundary=open`
- `term_order=sorted`

ADAPT / continuation:
- `adapt_pool=full_meta`
- `adapt_continuation_mode=phase3_v1`
- `phase1_depth0_full_meta_override=true`
- `adapt_inner_optimizer=POWELL`
- `adapt_max_depth=160`
- `adapt_maxiter=12000` (recorded in the log start event)
- `adapt_eps_grad=5e-7`
- `adapt_eps_energy=1e-9`
- `adapt_reopt_policy=windowed`
- `adapt_window_size=999999`
- `adapt_window_topk=999999`
- `adapt_full_refit_every=8`
- `adapt_final_full_refit=true`

Probe breadth:
- `phase1_shortlist_size=256`
- `phase1_probe_max_positions=999999`
- `phase2_shortlist_fraction=1.0`
- `phase2_shortlist_size=128`
- `phase2_enable_batching=true`
- `phase2_batch_target_size=8`
- `phase2_batch_size_cap=16`
- `phase2_batch_near_degenerate_ratio=0.98`
- `phase3_runtime_split_mode=shortlist_pauli_children_v1`
- `phase3_lifetime_cost_mode=phase3_v1`
- `phase3_enable_rescue=true`
- `phase3_symmetry_mitigation_mode=verify_only`

Execution:
- `initial_state_source=adapt_vqe`
- `dense_eigh_max_dim=0`
- `trotter_steps=192`
- `skip_pdf=true`

### Outcome

Main result:
- Final energy: `0.24502564220194084`
- Exact filtered reference: `0.2449407001279144`
- Final `abs_delta_e`: `8.49420740264284e-05`
- Stop reason: `drop_plateau`
- Wall-clock time: `116.904638 s`
- Final scaffold depth: `37`
- Final scaffold unique labels: `30`
- Final scaffold contains duplicates: yes

Important comparison to the earlier bad run:
- The old heavy rerun went negative and violated the sector-filtered variational comparison.
- This rerun did not. The negative-energy pathology is gone in this artifact.

Recorded heartbeats:
- Present throughout as `hardcoded_adapt_scipy_heartbeat`

Scaffold structure:
- `stage_events` recorded a `seed_complete` event with `num_seed_ops=6`.
- The first six scaffold entries are the six HH quadrature terms, which appears consistent with that seed stage. This sentence is an inference from the artifact ordering plus the recorded seed event.

### Search Cost

Search / optimization cost:
- Total objective evaluations: `84,518`
- Objective evaluations inside Powell optimizations: `84,368`
- Peak single-depth optimizer cost: `8,910` evaluations at depth `30`
- Total optimizer time: `35.36185553041287 s`
- Total recorded gradient-evaluation time: `0.037293703528121114 s`

Beam / runtime-split probing cost:
- Parent probes: `1,164`
- Child atoms evaluated: `4,920`
- Child atoms rejected by symmetry: `1,864`
- Admissible child sets formed: `13,370`
- Child sets actually probed/scored: `205`
- Child sets admitted into the scaffold: `7`
- Child atoms represented inside those admitted child sets: `9`

Proxy-cost caveat:
- The current compile-cost numbers are heuristic `phase3_v1_proxy` values.
- They are not exact decomposed gate counts.
- In particular, do not read the recorded `cx_proxy_total=2.0` at the last history depth as the literal exact final CX count for the scaffold.
- Exact gate-cost accounting should be done later, after the dedicated cost-data patch.

### What The Scaffold Actually Used

High-level pattern:
- HH quadrature terms dominated. All `6` available `hh_termwise_quadrature` generators appeared.
- No `hh_termwise_unit` generator survived into the scaffold.
- All `4` available lifted UCCSD singles appeared.
- `3` of the `4` available lifted UCCSD doubles appeared.
- PAOP usage concentrated in `paop_cloud_p`, `paop_lf_dbl_p`, `paop_disp`, and symmetry-safe child sets derived from `paop_cloud_p`, `paop_hopdrag`, and `paop_lf_dbl_p`.
- The layerwise HVA macros (`hop_layer`, `onsite_layer`, `phonon_layer`, `eph_layer`) were available but never used.
- The `x`-type PAOP variants (`paop_cloud_x`, `paop_lf_dbl_x`) were available but never used.

Operator-class accounting after the HH symmetry audit:

| Operator class | Available after audit | Direct in final scaffold | Via child sets | Never used |
| --- | ---: | ---: | ---: | ---: |
| `hh_termwise_quadrature` | 6 | 6 | 0 | 0 |
| `uccsd_sing` | 4 | 4 | 0 | 0 |
| `uccsd_dbl` | 4 | 3 | 0 | 1 |
| `paop_cloud_p` | 4 | 3 | 3 | 0 |
| `paop_lf_dbl_p` | 7 | 5 | 2 | 1 |
| `paop_hopdrag` | 2 | 0 | 2 | 1 |
| `paop_disp` | 3 | 2 | 0 | 1 |
| `hh_termwise_unit` | 18 | 0 | 0 | 18 |
| `paop_dbl` | 3 | 0 | 0 | 3 |
| `paop_cloud_x` | 4 | 0 | 0 | 4 |
| `paop_lf_dbl_x` | 7 | 0 | 0 | 7 |
| `hop_layer` | 1 | 0 | 0 | 1 |
| `onsite_layer` | 1 | 0 | 0 | 1 |
| `phonon_layer` | 1 | 0 | 0 | 1 |
| `eph_layer` | 1 | 0 | 0 | 1 |
| `paop_lf_curdrag` | 2 | 0 | 0 | 2 |
| `paop_lf_hop2` | 2 | 0 | 0 | 2 |

Interpretation of the table:
- "Direct in final scaffold" means the macro label itself appears in the final scaffold.
- "Via child sets" means a symmetry-safe `::child_set[...]` derived from that parent appears in the final scaffold.
- "Never used" treats a parent as used if either the macro itself or one of its symmetry-safe child sets appears.

Directly selected classes:
- `hh_termwise_quadrature`
- `uccsd_sing`
- `uccsd_dbl`
- `paop_cloud_p`
- `paop_lf_dbl_p`
- `paop_disp`

Classes used only through symmetry-safe child sets:
- `paop_hopdrag`

Classes never used at all:
- `hop_layer`
- `onsite_layer`
- `phonon_layer`
- `eph_layer`
- `hh_termwise_unit`
- `paop_dbl`
- `paop_cloud_x`
- `paop_lf_curdrag`
- `paop_lf_hop2`
- `paop_lf_dbl_x`

Repeated labels in the final scaffold:
- All six HH quadrature terms appear twice.
- `uccsd_ferm_lifted::uccsd_sing(beta:3->4)` appears twice.

### What Was Rejected

Hard-rejected before selection:
- `8` base generators were removed by the operator-level HH symmetry audit before ADAPT selection started.
- All `8` were `hh_termwise_ham_unit_term(...)` labels with `xx/yy`-style structure:
  - `hh_termwise_ham_unit_term(eeeeeeyye)`
  - `hh_termwise_ham_unit_term(eeeeeexxe)`
  - `hh_termwise_ham_unit_term(eeeeeeeyy)`
  - `hh_termwise_ham_unit_term(eeeeeeexx)`
  - `hh_termwise_ham_unit_term(eeeexxeee)`
  - `hh_termwise_ham_unit_term(eeeeyyeee)`
  - `hh_termwise_ham_unit_term(eeexxeeee)`
  - `hh_termwise_ham_unit_term(eeeyyeeee)`

Runtime-split rejections:
- `1,864` child atoms were rejected by the symmetry gate during runtime split probing.
- The final scaffold contains no raw `::split[...]` labels.
- The admitted split-derived content entered only through symmetry-safe `::child_set[...]` labels.

Post-run prune behavior:
- The pre-prune scaffold had `38` operators.
- The prune phase considered removing `6` operators.
- Only `1` removal was accepted, so the final scaffold length became `37`.
- The accepted removal was one duplicate `hh_termwise_ham_quadrature_term(eeyeezeee)`.
- The other `5` removal attempts were rejected because they regressed energy too much.

## Machine / Agent Data

### Artifact Pointers

- JSON: [adapt_hh_L3_full_meta_phase3_heavy_powell_20260321T000723Z.json](../json/adapt_hh_L3_full_meta_phase3_heavy_powell_20260321T000723Z.json)
- Log: [adapt_hh_L3_full_meta_phase3_heavy_powell_20260321T000723Z.log](../logs/adapt_hh_L3_full_meta_phase3_heavy_powell_20260321T000723Z.log)

### Key Scalars

- `settings.adapt_pool = full_meta`
- `adapt_vqe.pool_type = phase3_v1`
- `adapt_vqe.continuation_mode = phase3_v1`
- `adapt_vqe.phase1_depth0_full_meta_override = true`
- `adapt_vqe.energy = 0.24502564220194084`
- `adapt_vqe.exact_gs_energy = 0.2449407001279144`
- `adapt_vqe.abs_delta_e = 8.49420740264284e-05`
- `adapt_vqe.stop_reason = drop_plateau`
- `adapt_vqe.ansatz_depth = 37`
- `adapt_vqe.elapsed_s = 116.90463811089285`
- `adapt_vqe.nfev_total = 84518`
- `history length = 32`
- `sum(history.nfev_opt) = 84368`
- `sum(history.optimizer_elapsed_s) = 35.36185553041287`
- `sum(history.gradient_eval_elapsed_s) = 0.037293703528121114`
- `max(history.nfev_opt) = 8910 at depth 30`
- `last history compile_cost_proxy.proxy_total = 41.5`
- `last history compile_cost_proxy.cx_proxy_total = 2.0`
- `sum(history.compile_cost_proxy.proxy_total) = 1254.0`

### Pool Accounting

Raw / dedup / audited:
- `raw_total = 99`
- `dedup_total = 78`
- `removed_count = 8`
- `kept_count = 70`

Raw component counts before dedup:
- `raw_uccsd_lifted = 8`
- `raw_hva = 12`
- `raw_hh_termwise_augmented = 32`
- `raw_paop_full = 16`
- `raw_paop_lf_full = 31`

Post-audit available counts by class:
- `uccsd_sing = 4`
- `uccsd_dbl = 4`
- `hop_layer = 1`
- `onsite_layer = 1`
- `phonon_layer = 1`
- `eph_layer = 1`
- `hh_termwise_unit = 18`
- `hh_termwise_quadrature = 6`
- `paop_disp = 3`
- `paop_dbl = 3`
- `paop_hopdrag = 2`
- `paop_cloud_p = 4`
- `paop_cloud_x = 4`
- `paop_lf_curdrag = 2`
- `paop_lf_hop2 = 2`
- `paop_lf_dbl_p = 7`
- `paop_lf_dbl_x = 7`

### Ordered Final Scaffold

<details>
<summary>Final scaffold labels in order (37 positions)</summary>

1. `hh_termwise_ham_quadrature_term(yeezeeeee)`
2. `hh_termwise_ham_quadrature_term(yeeeeezee)`
3. `hh_termwise_ham_quadrature_term(eeyeezeee)`
4. `hh_termwise_ham_quadrature_term(eeyeeeeez)`
5. `hh_termwise_ham_quadrature_term(eyeeeeeze)`
6. `hh_termwise_ham_quadrature_term(eyeezeeee)`
7. `uccsd_ferm_lifted::uccsd_sing(alpha:1->2)`
8. `uccsd_ferm_lifted::uccsd_sing(beta:3->4)`
9. `uccsd_ferm_lifted::uccsd_sing(alpha:0->2)`
10. `uccsd_ferm_lifted::uccsd_sing(beta:3->5)`
11. `paop_full:paop_cloud_p(site=2->phonon=1)`
12. `paop_full:paop_cloud_p(site=1->phonon=2)`
13. `paop_full:paop_cloud_p(site=1->phonon=0)`
14. `paop_lf_full:paop_dbl_p(site=0->phonon=1)`
15. `uccsd_ferm_lifted::uccsd_dbl(ab:0,3->2,4)`
16. `paop_lf_full:paop_dbl_p(site=0->phonon=0)`
17. `paop_lf_full:paop_dbl_p(site=2->phonon=2)`
18. `uccsd_ferm_lifted::uccsd_dbl(ab:0,3->2,5)`
19. `paop_lf_full:paop_dbl_p(site=1->phonon=1)`
20. `paop_lf_full:paop_dbl_p(site=1->phonon=2)::child_set[3]`
21. `paop_full:paop_hopdrag(0,1)::child_set[5,7]`
22. `paop_lf_full:paop_dbl_p(site=2->phonon=1)`
23. `paop_full:paop_cloud_p(site=0->phonon=1)::child_set[0]`
24. `paop_full:paop_disp(site=2)`
25. `uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,4)`
26. `paop_full:paop_cloud_p(site=1->phonon=2)::child_set[0]`
27. `paop_full:paop_cloud_p(site=1->phonon=0)::child_set[0]`
28. `paop_full:paop_disp(site=0)`
29. `paop_full:paop_hopdrag(0,1)::child_set[0,2]`
30. `uccsd_ferm_lifted::uccsd_sing(beta:3->4)`
31. `paop_lf_full:paop_dbl_p(site=0->phonon=0)::child_set[3]`
32. `hh_termwise_ham_quadrature_term(eyeeeeeze)`
33. `hh_termwise_ham_quadrature_term(eeyeezeee)`
34. `hh_termwise_ham_quadrature_term(yeezeeeee)`
35. `hh_termwise_ham_quadrature_term(yeeeeezee)`
36. `hh_termwise_ham_quadrature_term(eyeezeeee)`
37. `hh_termwise_ham_quadrature_term(eeyeeeeez)`

</details>

### Direct Base-Generator Labels Used

<details>
<summary>Unique direct macro labels used (23)</summary>

- `hh_termwise_ham_quadrature_term(yeezeeeee)`
- `hh_termwise_ham_quadrature_term(yeeeeezee)`
- `hh_termwise_ham_quadrature_term(eeyeezeee)`
- `hh_termwise_ham_quadrature_term(eeyeeeeez)`
- `hh_termwise_ham_quadrature_term(eyeeeeeze)`
- `hh_termwise_ham_quadrature_term(eyeezeeee)`
- `uccsd_ferm_lifted::uccsd_sing(alpha:1->2)`
- `uccsd_ferm_lifted::uccsd_sing(beta:3->4)`
- `uccsd_ferm_lifted::uccsd_sing(alpha:0->2)`
- `uccsd_ferm_lifted::uccsd_sing(beta:3->5)`
- `paop_full:paop_cloud_p(site=2->phonon=1)`
- `paop_full:paop_cloud_p(site=1->phonon=2)`
- `paop_full:paop_cloud_p(site=1->phonon=0)`
- `paop_lf_full:paop_dbl_p(site=0->phonon=1)`
- `uccsd_ferm_lifted::uccsd_dbl(ab:0,3->2,4)`
- `paop_lf_full:paop_dbl_p(site=0->phonon=0)`
- `paop_lf_full:paop_dbl_p(site=2->phonon=2)`
- `uccsd_ferm_lifted::uccsd_dbl(ab:0,3->2,5)`
- `paop_lf_full:paop_dbl_p(site=1->phonon=1)`
- `paop_lf_full:paop_dbl_p(site=2->phonon=1)`
- `paop_full:paop_disp(site=2)`
- `uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,4)`
- `paop_full:paop_disp(site=0)`

</details>

### Symmetry-Safe Child-Set Labels Used

<details>
<summary>Unique child-set labels used (7)</summary>

- `paop_lf_full:paop_dbl_p(site=1->phonon=2)::child_set[3]`
- `paop_full:paop_hopdrag(0,1)::child_set[5,7]`
- `paop_full:paop_cloud_p(site=0->phonon=1)::child_set[0]`
- `paop_full:paop_cloud_p(site=1->phonon=2)::child_set[0]`
- `paop_full:paop_cloud_p(site=1->phonon=0)::child_set[0]`
- `paop_full:paop_hopdrag(0,1)::child_set[0,2]`
- `paop_lf_full:paop_dbl_p(site=0->phonon=0)::child_set[3]`

</details>

### Hard-Rejected Base Generators

<details>
<summary>Base generators removed before selection by the HH operator-level symmetry audit (8)</summary>

- `hh_termwise_ham_unit_term(eeeeeeyye)`
- `hh_termwise_ham_unit_term(eeeeeexxe)`
- `hh_termwise_ham_unit_term(eeeeeeeyy)`
- `hh_termwise_ham_unit_term(eeeeeeexx)`
- `hh_termwise_ham_unit_term(eeeexxeee)`
- `hh_termwise_ham_unit_term(eeeeyyeee)`
- `hh_termwise_ham_unit_term(eeexxeeee)`
- `hh_termwise_ham_unit_term(eeeyyeeee)`

</details>

### Runtime-Split Summary

- `mode = shortlist_pauli_children_v1`
- `probed_parent_count = 1164`
- `evaluated_child_count = 4920`
- `rejected_child_count_symmetry = 1864`
- `admissible_child_set_count = 13370`
- `probe_parent_win_count = 314`
- `probe_child_set_count = 205`
- `selected_child_set_count = 7`
- `selected_child_count = 9`

<details>
<summary>Child atoms represented inside the admitted child sets (9)</summary>

- `paop_lf_full:paop_dbl_p(site=1->phonon=2)::split[3]::yeeezeeze`
- `paop_full:paop_hopdrag(0,1)::split[5]::eeyeeeeyy`
- `paop_full:paop_hopdrag(0,1)::split[7]::eeyeeeexx`
- `paop_full:paop_cloud_p(site=0->phonon=1)::split[0]::eyeeeeeez`
- `paop_full:paop_cloud_p(site=1->phonon=2)::split[0]::yeeeeeeze`
- `paop_full:paop_cloud_p(site=1->phonon=0)::split[0]::eeyeeeeze`
- `paop_full:paop_hopdrag(0,1)::split[0]::eyeeyyeee`
- `paop_full:paop_hopdrag(0,1)::split[2]::eyeexxeee`
- `paop_lf_full:paop_dbl_p(site=0->phonon=0)::split[3]::eeyeezeez`

</details>

### Post-Prune Decisions

Interpretation:
- `accepted=true` below means the removal was accepted by the prune stage, so that operator was actually pruned.
- `accepted=false` means the attempted removal was rejected and the operator stayed.

<details>
<summary>Prune decisions (6 candidates)</summary>

- `accepted=true`, `label=hh_termwise_ham_quadrature_term(eeyeezeee)`, `reason=accepted`, `regression=1.6559531523796522e-11`
- `accepted=false`, `label=hh_termwise_ham_quadrature_term(eyeezeeee)`, `reason=regression_exceeded`, `regression=0.00337224075966977`
- `accepted=false`, `label=hh_termwise_ham_quadrature_term(eeyeezeee)`, `reason=regression_exceeded`, `regression=0.0002973483766656182`
- `accepted=false`, `label=paop_lf_full:paop_dbl_p(site=1->phonon=2)::child_set[3]`, `reason=regression_exceeded`, `regression=0.0002990876751184368`
- `accepted=false`, `label=hh_termwise_ham_quadrature_term(yeezeeeee)`, `reason=regression_exceeded`, `regression=8.117369151849907e-07`
- `accepted=false`, `label=paop_full:paop_cloud_p(site=1->phonon=2)::child_set[0]`, `reason=regression_exceeded`, `regression=0.00028737746876300596`

</details>

### Base Generators Never Used

<details>
<summary>Available base generators that never appeared directly and never appeared through a child set (44)</summary>

- `uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,5)`
- `hop_layer`
- `onsite_layer`
- `phonon_layer`
- `eph_layer`
- `hh_termwise_ham_unit_term(zeeeeeeee)`
- `hh_termwise_ham_unit_term(ezeeeeeee)`
- `hh_termwise_ham_unit_term(eezeeeeee)`
- `hh_termwise_ham_unit_term(eeeezeeee)`
- `hh_termwise_ham_unit_term(eeeeeeeze)`
- `hh_termwise_ham_unit_term(eeeezeeze)`
- `hh_termwise_ham_unit_term(eeeeezeez)`
- `hh_termwise_ham_unit_term(eeeeeeeez)`
- `hh_termwise_ham_unit_term(eeeeezeee)`
- `hh_termwise_ham_unit_term(eeezeezee)`
- `hh_termwise_ham_unit_term(eeeeeezee)`
- `hh_termwise_ham_unit_term(eeezeeeee)`
- `hh_termwise_ham_unit_term(xeezeeeee)`
- `hh_termwise_ham_unit_term(xeeeeezee)`
- `hh_termwise_ham_unit_term(eexeezeee)`
- `hh_termwise_ham_unit_term(eexeeeeez)`
- `hh_termwise_ham_unit_term(exeeeeeze)`
- `hh_termwise_ham_unit_term(exeezeeee)`
- `paop_full:paop_disp(site=1)`
- `paop_full:paop_dbl(site=0)`
- `paop_full:paop_dbl(site=1)`
- `paop_full:paop_dbl(site=2)`
- `paop_full:paop_hopdrag(1,2)`
- `paop_full:paop_cloud_x(site=0->phonon=1)`
- `paop_full:paop_cloud_x(site=1->phonon=0)`
- `paop_full:paop_cloud_x(site=1->phonon=2)`
- `paop_full:paop_cloud_x(site=2->phonon=1)`
- `paop_lf_full:paop_curdrag(0,1)`
- `paop_lf_full:paop_curdrag(1,2)`
- `paop_lf_full:paop_hop2(0,1)`
- `paop_lf_full:paop_hop2(1,2)`
- `paop_lf_full:paop_dbl_x(site=0->phonon=0)`
- `paop_lf_full:paop_dbl_x(site=0->phonon=1)`
- `paop_lf_full:paop_dbl_p(site=1->phonon=0)`
- `paop_lf_full:paop_dbl_x(site=1->phonon=0)`
- `paop_lf_full:paop_dbl_x(site=1->phonon=1)`
- `paop_lf_full:paop_dbl_x(site=1->phonon=2)`
- `paop_lf_full:paop_dbl_x(site=2->phonon=1)`
- `paop_lf_full:paop_dbl_x(site=2->phonon=2)`

</details>

---

## Motif Analysis & Pareto Pool Design

This section distills the six dominant motifs from the heavy scaffold and derives a pruned Pareto-optimal generator pool.

### Six Motifs

#### 1. HH Quadrature is the backbone — use all 6, allow repeats

All 6 `hh_termwise_quadrature` generators entered the scaffold. Every single one appeared **twice** (once in the seed block, positions 1–6, and again later in positions 32–37). Zero quadrature terms were rejected. This is the highest-value class by both coverage and repeat selection.

#### 2. Lifted UCCSD singles are essential; doubles are nearly so

All 4 `uccsd_sing` generators were selected. 3 of 4 `uccsd_dbl` generators were selected. One double (`ab:1,3->2,5`) was never used. One single (`beta:3->4`) was repeated. These provide the fermionic correlation structure that quadrature alone can't reach.

#### 3. `p`-type PAOPs dominate; `x`-type PAOPs are dead weight

- `paop_cloud_p`: 3/4 direct + 3 via child sets = **full usage**
- `paop_lf_dbl_p`: 5/7 direct + 2 via child sets = **near-full usage**
- `paop_disp`: 2/3 direct
- `paop_hopdrag`: 2 via child sets only (never directly)
- **Zero usage** of: `paop_cloud_x` (0/4), `paop_lf_dbl_x` (0/7), `paop_dbl` (0/3), `paop_lf_curdrag` (0/2), `paop_lf_hop2` (0/2)

The `p`-type (momentum-coupled) operators capture the relevant electron-phonon physics; the `x`-type variants add cost without energy improvement.

#### 4. Layerwise HVA macros contribute nothing

`hop_layer`, `onsite_layer`, `phonon_layer`, `eph_layer` — all 4 available, all 4 unused. The fine-grained operators (quadrature + UCCSD + PAOPs) already capture what the coarse HVA layers would provide, at lower cost.

#### 5. `hh_termwise_unit` generators are uniformly rejected

18 were available post-audit, 0 were used. The `xx/yy`-structured ones were hard-rejected by the symmetry audit (8 removed pre-selection), and the remaining `z/zz`-structured diagonal terms never competed successfully against the quadrature terms. This entire class is prunable.

#### 6. Runtime split adds value through child sets, not raw splits

7 child sets (9 child atoms) made it into the final scaffold. No raw `::split[...]` labels survived — only symmetry-safe `::child_set[...]` composites. The parents that benefited from splitting were `paop_cloud_p`, `paop_hopdrag`, and `paop_lf_dbl_p`. Runtime split should stay on for those classes, but the scoring surface above it is the bottleneck.

### Implied Pareto Pool Design

**Keep (core) — ~26–28 generators:**

| Class | Count | Role |
| --- | ---: | --- |
| `hh_termwise_quadrature` | 6 | Energy backbone, repeat-eligible |
| `uccsd_sing` | 4 | Fermionic singles correlation |
| `uccsd_dbl` | 3–4 | Fermionic doubles correlation |
| `paop_cloud_p` | 4 | e-ph cloud coupling (p-type) |
| `paop_lf_dbl_p` | 7 | Long-range e-ph doubles (p-type) |
| `paop_disp` | 2–3 | Dispersive phonon coupling |

**Keep conditionally (via child sets only):**

| Class | Count | Note |
| --- | ---: | --- |
| `paop_hopdrag` | 1–2 | Only useful through runtime split children |

**Drop entirely — ~44 generators removed:**

| Class | Count | Reason |
| --- | ---: | --- |
| `hh_termwise_unit` | 18 | Never selected, diagonal terms redundant |
| `paop_cloud_x` | 4 | x-type never selected |
| `paop_lf_dbl_x` | 7 | x-type never selected |
| `paop_dbl` | 3 | Never selected |
| `paop_lf_curdrag` | 2 | Never selected |
| `paop_lf_hop2` | 2 | Never selected |
| HVA layers | 4 | Coarse layerwise, redundant with fine-grained ops |

This takes the pool from **70 post-audit generators → ~26–28**, cutting search cost by ~60% while retaining every operator class that actually contributed energy improvement in the best-yet run.
