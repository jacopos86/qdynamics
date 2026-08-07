# SR-SNAKE v4 Phase-I/II no-lambda-F six-regime bundle revision v6

This directory is the immutable-successor scaffold for the stale v2 bundle.
It preserves the six v2 Hubbard--Holstein interaction points, resources,
optimizer budgets, and fresh round-0 to round-30 horizon.  It applies the two
user-approved scientific changes: the Phase-I/II energy model below and the
padding-free same-cutoff binary truncations `n_ph=3` for weak Holstein and
`n_ph=7` for strong Holstein.

- `phase1_energy_model=first_order_fs_trust_v1`;
- `phase2_curvature_policy=measured_required_fail_closed_v1`;
- `phase2_cheap_curvature_proxy_policy=off`;
- no Phase-I/II lambda-F proxy or missing-curvature substitution;
- legacy HH quadrature preseed disabled so depth and controller round both
  start at zero; HVA generators remain available in `full_meta`;
- malformed, absent, unprovenanced, or nonfinite Phase-II curvature aborts the
  run before novelty fallback can execute.

Revision v6 is the source-only repair successor to failed/cancelled revision v5.
The scientific argv and route digest are unchanged. Revision v5 passed startup
validation but failed at round 10 when the explicitly preserved all-energy-
models-infeasible novelty fallback selected a rank-gated record and the
post-refit trust updater consumed it as an ordinary energy model. Revision v6
propagates the selected-record geometry-expansion marker into the trust
transaction and uses the existing measured-endpoint geometry-expansion update;
it does not invent a predicted step or change selection. The corrected clean
source passed the former failure point through round 11. It is frozen at commit
`92cf00bb1e7c5c58cc2328c29cdcae9d772adfc0` and tree
`5608d20f6b77d200fa90cfdc0ec5e86feb89a71c`.

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
