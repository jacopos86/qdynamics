# RA-ADAPT appendix L=3 escalation

Source commit: `73d652062e3ba589e4d34c7b0fcf35e563012711`.

Route: append-RA, no warm start, Powell inner optimizer, append-only insertion, selection cost K = 0.30 dN2q + 0.30 dD2q + 0.25 dDc.

## Candidate supply

The L=2 guarded single-Pauli exposure is sector-degenerate at L=3: its 9 survivors are diagonal Z words with zero Hartree--Fock gradient and zero tangent self-metric, giving supported response rank 0.

This campaign runs the sector-guarded composite generator pool: 24 generators, supported response rank 8. The controller rank validator was not weakened.

## Energy error by iteration

| cell | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| l3_baseline_h10 | 2.391e+00 | 9.460e-01 | 4.601e-01 | 6.875e-02 | 6.557e-02 | 1.528e-02 | 2.508e-03 | 9.296e-10 | 7.065e-11 | 6.064e-12 |
| l3_metric_ablation_h10 | 2.391e+00 | 9.460e-01 | 4.601e-01 | 6.875e-02 | 6.557e-02 | 1.528e-02 | 2.508e-03 | 9.296e-10 | 7.065e-11 | 6.064e-12 |
| l3_batching_h10 | 9.293e-01 | 3.221e-02 | 3.434e-10 | 2.502e-12 | 1.377e-13 | 1.510e-14 | 1.155e-14 | 1.021e-14 | 8.438e-15 | 8.438e-15 |
| l3_beam_h10 | 2.391e+00 | 9.460e-01 | 4.601e-01 | 6.875e-02 | 6.557e-02 | 1.528e-02 | 2.508e-03 | 9.296e-10 | 7.046e-11 | 5.811e-12 |

## First iteration reaching each accuracy

| cell | 1e-2 | 1e-4 | 1e-6 |
| --- | --- | --- | --- |
| l3_baseline_h10 | round 7 | round 8 | round 8 |
| l3_metric_ablation_h10 | round 7 | round 8 | round 8 |
| l3_batching_h10 | round 3 | round 3 | round 3 |
| l3_beam_h10 | round 7 | round 8 | round 8 |

## Admitted and pruned generators

| cell | admissions | distinct generators | prune nominations | accepted deletions |
| --- | --- | --- | --- | --- |
| l3_baseline_h10 | 10 | 7 | 0 | 0 |
| l3_metric_ablation_h10 | 10 | 7 | 8 | 0 |
| l3_batching_h10 | 15 | 13 | 0 | 0 |
| l3_beam_h10 | 10 | 7 | 0 | 0 |

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

### Beam child and survivor accounting

Explored branch sequences: 38. Survivors are the accepted lineage recorded round by round in `generator_events.json`.

## Does each extension beat minimal RA?

| extension | before round 3 | at any threshold | rounds saved to 1e-6 |
| --- | --- | --- | --- |
| metric_ablation | False | False | 0 |
| batching | False | True | 5 |
| beam | False | False | 0 |

The v1--v5 campaign evidence is preserved unchanged. No CHTC submission and no manuscript edit was made by this campaign.
