# Pure-Hubbard L=4 Qiskit/proxy extension campaign

This candidate campaign runs one shared AVQE comparator and four RA arms
(Append, combinatorial batch, escalated beam, and gated prune) under each of
two Phase-I--III selection-cost policies: exact FakeMarrakesh Qiskit marginals
and the Marrakesh graph-span proxy.  All cells use the same half-filled open
L=4 Hubbard problem, k=50 horizon, tight Powell settings, active Phase 0,
adaptive gamma=0.9 shortlisting, full joint Phase-III gain with stationarity,
and the fixed Fubini--Study-whitened affine accepted-refit chart.

AVQE has no RA selection-cost coordinate, so it is executed once and shared by
both comparisons.  Qiskit compiled resources are reported for every completed
trajectory regardless of the RA selection-cost policy.

The prune cell uses the independently confirmed L=3 early-pruning variant:
small-theta threshold 1e-3, zero admission-protection rounds, regression gate
1e-8, predicted-loss-per-cost gate 1e-8, and measured delete-and-complete-refit
acceptance.

This directory is candidate run support.  It does not authorize manuscript
promotion.
