# RA-ADAPT appendix L=3 escalation

Source commit: `8501ffd7dfd8c7099d4c5e21dc8b258f2a0648c7`.

Route: append-RA, no warm start, Powell inner optimizer, append-only insertion, selection cost K = 0.30 dN2q + 0.30 dD2q + 0.25 dDc.

## Candidate supply

The L=2 guarded single-Pauli exposure is sector-degenerate at L=3: its 9 survivors are diagonal Z words with zero Hartree--Fock gradient and zero tangent self-metric, giving supported response rank 0.

This campaign runs the sector-guarded composite generator pool: 24 generators, supported response rank 8. The controller rank validator was not weakened.

## Energy error by iteration

| cell | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| l3_baseline_h15 | 2.391e+00 | 9.460e-01 | 4.601e-01 | 6.875e-02 | 6.557e-02 | 1.528e-02 | 2.508e-03 | 1.029e-10 | 2.851e-12 | 3.300e-13 | 4.041e-14 | 1.199e-14 | 4.885e-15 | 3.109e-15 | 3.109e-15 |
| l3_metric_ablation_gated_h15 | 2.391e+00 | 9.460e-01 | 4.601e-01 | 6.875e-02 | 6.557e-02 | 1.528e-02 | 2.508e-03 | 1.029e-10 | 2.851e-12 | 3.997e-14 | 4.885e-15 | 2.220e-15 | 4.441e-16 | 0.000e+00 | 8.882e-16 |
| l3_batching_h15 | 9.293e-01 | 3.221e-02 | 1.038e-07 | 1.813e-11 | 9.575e-13 | 9.148e-14 | 1.332e-14 | 4.441e-15 | 4.441e-15 | 4.441e-15 | 6.217e-15 | - | - | - | - |
| l3_beam_escalated_h15 | 2.391e+00 | 9.460e-01 | 4.601e-01 | 6.875e-02 | 6.611e-02 | 7.264e-03 | 7.379e-04 | 1.251e-10 | 9.840e-12 | 6.892e-13 | 4.308e-14 | 4.885e-15 | 8.882e-16 | 8.882e-16 | 4.441e-16 |

## First iteration reaching each accuracy

| cell | 1e-2 | 1e-4 | 1e-6 |
| --- | --- | --- | --- |
| l3_baseline_h15 | round 7 | round 8 | round 8 |
| l3_metric_ablation_gated_h15 | round 7 | round 8 | round 8 |
| l3_batching_h15 | round 3 | round 3 | round 3 |
| l3_beam_escalated_h15 | round 6 | round 8 | round 8 |

## Admitted and pruned generators

| cell | admissions | distinct generators | prune nominations | accepted deletions |
| --- | --- | --- | --- | --- |
| l3_baseline_h15 | 15 | 10 | 0 | 0 |
| l3_metric_ablation_gated_h15 | 15 | 9 | 6 | 6 |
| l3_batching_h15 | 16 | 14 | 0 | 0 |
| l3_beam_escalated_h15 | 15 | 12 | 0 | 0 |

### Batch membership

| round | size | admitted generators |
| --- | --- | --- |
| 1 | 3 | `uccsd_sing(alpha:1->2)`, `uccsd_sing(beta:3->4)`, `uccsd_dbl(ab:1,3->2,4)` |
| 2 | 3 | `uccsd_sing(alpha:0->2)`, `uccsd_sing(beta:3->5)`, `uccsd_dbl(ab:0,3->2,5)` |
| 3 | 2 | `uccsd_dbl(ab:1,3->2,4)`, `uccsd_sing(beta:3->4)` |
| 4 | 1 | `hop(i=0,j=1,spin=0)` |
| 5 | 1 | `hop(i=1,j=2,spin=0)` |
| 6 | 1 | `hop(i=1,j=2,spin=1)` |
| 7 | 1 | `hop(i=0,j=1,spin=1)` |
| 8 | 1 | `onsite(i=1)` |
| 9 | 1 | `onsite(i=0)` |
| 10 | 1 | `ham_term(zeeeee)` |
| 11 | 1 | `ham_term(eeezee)` |

### Beam child and survivor accounting

Explored branch sequences: 111. Survivors are the accepted lineage recorded round by round in `generator_events.json`.

## Does each extension beat minimal RA?

| extension | before round 3 | at any threshold | rounds saved to 1e-6 |
| --- | --- | --- | --- |
| metric_ablation_gated | False | False | 0 |
| batching | False | True | 5 |
| beam_escalated | False | True | 0 |

The v1--v5 campaign evidence is preserved unchanged. No CHTC submission and no manuscript edit was made by this campaign.
