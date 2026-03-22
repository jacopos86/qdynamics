# Lean logical circuit handoff

This file describes the **logical (pre-backend, pre-routing) Qiskit circuit only** for the lean HH ADAPT result.

## Source artifact
- JSON: `artifacts/json/adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell_rerun_with_ansatz_input_20260321T214822Z.json`

## Purpose
Use this document and the paired PDF to ask which IBM backend is the best fit for this **logical** circuit construction.

## Problem / ansatz
- Model: Hubbard-Holstein
- Circuit type: imported lean `pareto_lean_l2` ADAPT circuit
- L: 2
- n_ph_max: 1
- ordering: `blocked`
- boundary: `open`
- qubits: 6

## Logical-circuit metrics
- logical operator depth: 14
- logical parameters: 14
- runtime parameters: 25
- abstract gate count: 205
- abstract depth: 85
- abstract 2-qubit gates: 58
- abstract 1-qubit / init gates: 147
- count_ops: `{'cx': 58, 'h': 66, 'rz': 25, 's': 27, 'sdg': 27, 'x': 2}`

## Energy context
- saved energy: 0.15872408236037028
- exact sector energy: 0.15866790412572634

## Reference-state provenance
- embedded reference state: True
- source: `hf`
- kind: `reference_state`

## Logical operator sequence
1. `hh_termwise_ham_quadrature_term(yezeee)`
2. `hh_termwise_ham_quadrature_term(yeeeze)`
3. `hh_termwise_ham_quadrature_term(eyeeez)`
4. `hh_termwise_ham_quadrature_term(eyezee)`
5. `uccsd_ferm_lifted::uccsd_sing(alpha:0->1)`
6. `uccsd_ferm_lifted::uccsd_sing(beta:2->3)`
7. `paop_lf_full:paop_dbl_p(site=0->phonon=1)`
8. `paop_lf_full:paop_dbl_p(site=1->phonon=0)`
9. `paop_full:paop_hopdrag(0,1)::child_set[0,2]`
10. `paop_full:paop_cloud_p(site=1->phonon=0)`
11. `paop_full:paop_cloud_p(site=1->phonon=0)::child_set[1]`
12. `paop_full:paop_cloud_p(site=0->phonon=1)`
13. `paop_lf_full:paop_dbl_p(site=0->phonon=1)::child_set[3]`
14. `paop_lf_full:paop_dbl_p(site=0->phonon=1)::child_set[3]`

## Paired PDF
- PDF: `artifacts/pdf/lean_logical_circuit_20260321T214822Z.pdf`

## Notes
- This is the **abstract logical circuit only**.
- It is **not** transpiled, routed, or compiled to any IBM backend in this document.
- Backend choice should therefore focus on qubit count, connectivity headroom, and likely routing pressure for this logical construction.
