# RA-ADAPT appendix L=3 escalation

Source commit: `a50daddf5d5fbd559e2d90fd0e907e1f21c5bef1`.

Route: append-RA, no warm start, Powell inner optimizer, append-only insertion, selection cost K = 0.30 dN2q + 0.30 dD2q + 0.25 dDc.

## Candidate supply

The L=2 guarded single-Pauli exposure is sector-degenerate at L=3: its 9 survivors are diagonal Z words with zero Hartree--Fock gradient and zero tangent self-metric, giving supported response rank 0.

This campaign runs the sector-guarded composite generator pool: 24 generators, supported response rank 8. The controller rank validator was not weakened.

## Energy error by iteration

| cell | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| l3_baseline_h15 | 2.391e+00 | 9.460e-01 | 4.601e-01 | 6.875e-02 | 6.557e-02 | 1.528e-02 | 2.508e-03 | 9.296e-10 | 7.065e-11 | 6.064e-12 | 5.840e-13 | 6.128e-14 | 9.326e-15 | 7.994e-15 | 7.105e-15 |
| l3_metric_ablation_h15 | 2.391e+00 | 9.460e-01 | 4.601e-01 | 6.875e-02 | 6.557e-02 | 1.528e-02 | 2.508e-03 | 9.296e-10 | 7.065e-11 | 6.064e-12 | 5.387e-13 | 3.597e-14 | 0.000e+00 | 3.109e-15 | 3.109e-15 |
| l3_metric_ablation_gated_h15 | 2.391e+00 | 9.460e-01 | 4.601e-01 | 6.875e-02 | 6.557e-02 | 1.528e-02 | 2.508e-03 | 9.296e-10 | 7.065e-11 | 5.625e-12 | 3.362e-13 | 2.442e-14 | 3.109e-15 | 4.441e-16 | 8.882e-16 |
| l3_batching_h15 | 9.293e-01 | 3.221e-02 | 3.434e-10 | 2.502e-12 | 1.377e-13 | 1.510e-14 | 1.155e-14 | 1.021e-14 | 8.438e-15 | 8.438e-15 | 8.438e-15 | - | - | - | - |
| l3_beam_h15 | 2.391e+00 | 9.460e-01 | 4.601e-01 | 6.875e-02 | 6.557e-02 | 1.528e-02 | 2.508e-03 | 9.296e-10 | 7.046e-11 | 5.811e-12 | 5.356e-13 | 6.040e-14 | 1.510e-14 | 6.217e-15 | 4.441e-16 |
| l3_beam_cost_h15 | 2.391e+00 | 9.460e-01 | 4.601e-01 | 6.875e-02 | 6.557e-02 | 1.528e-02 | 2.508e-03 | 1.007e-09 | 7.582e-11 | 6.155e-12 | 6.279e-13 | 7.550e-14 | 1.110e-14 | 3.997e-15 | 2.220e-15 |

## First iteration reaching each accuracy

| cell | 1e-2 | 1e-4 | 1e-6 |
| --- | --- | --- | --- |
| l3_baseline_h15 | round 7 | round 8 | round 8 |
| l3_metric_ablation_h15 | round 7 | round 8 | round 8 |
| l3_metric_ablation_gated_h15 | round 7 | round 8 | round 8 |
| l3_batching_h15 | round 3 | round 3 | round 3 |
| l3_beam_h15 | round 7 | round 8 | round 8 |
| l3_beam_cost_h15 | round 7 | round 8 | round 8 |

## Admitted and pruned generators

| cell | admissions | distinct generators | prune nominations | accepted deletions |
| --- | --- | --- | --- | --- |
| l3_baseline_h15 | 15 | 12 | 0 | 0 |
| l3_metric_ablation_h15 | 15 | 10 | 13 | 5 |
| l3_metric_ablation_gated_h15 | 15 | 9 | 6 | 6 |
| l3_batching_h15 | 16 | 14 | 0 | 0 |
| l3_beam_h15 | 15 | 12 | 0 | 0 |
| l3_beam_cost_h15 | 15 | 12 | 0 | 0 |

### Batch membership

| round | size | admitted generators |
| --- | --- | --- |
| 1 | 3 | `uccsd_sing(alpha:1->2)`, `uccsd_sing(beta:3->4)`, `uccsd_dbl(ab:1,3->2,4)` |
| 2 | 3 | `uccsd_sing(beta:3->5)`, `uccsd_sing(alpha:0->2)`, `uccsd_dbl(ab:0,3->2,5)` |
| 3 | 2 | `uccsd_dbl(ab:1,3->2,4)`, `uccsd_sing(alpha:1->2)` |
| 4 | 1 | `hop(i=0,j=1,spin=0)` |
| 5 | 1 | `hop(i=1,j=2,spin=0)` |
| 6 | 1 | `hop(i=1,j=2,spin=1)` |
| 7 | 1 | `onsite(i=1)` |
| 8 | 1 | `onsite(i=2)` |
| 9 | 1 | `hop(i=0,j=1,spin=1)` |
| 10 | 1 | `ham_term(zeeeee)` |
| 11 | 1 | `ham_term(eeezee)` |

### Beam child and survivor accounting

Explored branch sequences: 58. Survivors are the accepted lineage recorded round by round in `generator_events.json`.

## Does each extension beat minimal RA?

| extension | before round 3 | at any threshold | rounds saved to 1e-6 |
| --- | --- | --- | --- |
| metric_ablation | False | False | 0 |
| metric_ablation_gated | False | False | 0 |
| batching | False | True | 5 |
| beam | False | False | 0 |
| beam_cost | False | False | 0 |

The v1--v5 campaign evidence is preserved unchanged. No CHTC submission and no manuscript edit was made by this campaign.
