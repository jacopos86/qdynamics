# SR-SNAKE v4 strong-weak full-response service repair revision v7

This directory is the immutable operational successor for only the failed
strong-weak row of cluster 8878503. It preserves all six v6 manifests as
source-lock comparison records, but `queue.tsv` contains exactly one fresh
strong-weak round-0 to round-30 job. Its scientific route, Hamiltonian,
same-cutoff reference, optimizer budget, and route digest are unchanged from
v6.

- `phase1_energy_model=first_order_fs_trust_v1`;
- `phase2_curvature_policy=measured_required_fail_closed_v1`;
- `phase2_cheap_curvature_proxy_policy=off`;
- no Phase-I/II lambda-F proxy or missing-curvature substitution;
- legacy HH quadrature preseed disabled so depth and controller round both
  start at zero; HVA generators remain available in `full_meta`;
- malformed, absent, unprovenanced, or nonfinite Phase-II curvature aborts the
  run before novelty fallback can execute.

The v6 strong-weak science reached round 30, but rounds 25--30 let the maturity
controller retire Phase III. That violated the already registered v4 contract
requiring a full active-logical-plus-singleton Phase-III response and an
authoritative adaptive-trust receipt on every round. Revision v7 prevents that
retirement only for registered v3/v4 full-response profiles and fails closed if
any round still lacks the response-rank, coordinate, or trust receipts. This is
contract enforcement, not a scientific-setting change. The repair is frozen at
commit `b2400825d3279753e2f1b1a61d4447f0ccbb606c` and tree
`b7bd374befe8e564da1ab4d0c53250f41bfad3c4`.

The bundle must pass its local and archive-only tests, the authenticated remote
image/Qiskit/FakeMarrakesh gate, and the in-image bundle tests before submission.
The generated `submit.sub` must use `requirements = TARGET.HasSIF`.

The bundle validators require exact Phase-I/II policy serialization, forbid
Phase-I/II lambda-F flags in argv, parse every exact scientific argv through
the archived production CLI, require every Phase-II full-candidate occurrence to be
matched by a validated finite curvature receipt, and require zero proxy and
missing-curvature-fallback occurrences.  Existing v4 Phase III, pruning,
adaptive trust, full-response/full-refit, symmetry, padding, and Qiskit gates
remain in force.
